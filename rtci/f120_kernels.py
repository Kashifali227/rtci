# SPDX-License-Identifier: GPL-3.0-only
# Derived from the Generic Python Toolbox (GPyT, https://github.com/jabeim/AB-Generic-Python-Toolbox,
# GPL-3.0, by JAB / jabeim), a Python port of the Advanced Bionics Generic MATLAB Toolbox. rtci, 2026 Kashif Ali.
"""Fused numba kernels for the streaming F120 strategy.

Each audio block takes a few kernel calls instead of hundreds of small NumPy calls. Every kernel keeps the
GPyT operation order, so the streaming output matches GPyT to about 1e-14 (validate_gpyt.py checks this).

Frames are stored as rows. Per-frame history lives in ring buffers indexed by the global frame number % RING.
"""
from . import _svml  # noqa: F401,I001  (must be imported before numba, see _svml.py)
import numba
import numpy as np

RING = 1024                     # history ring length in frames; FT frames lag the newest audio frame by < 4


@numba.njit(cache=True)
def frame_kernel(y, g, ring, ring_g, cnt, win, frames, gA):
    """Collect AGC output samples in a 256-sample ring and emit a windowed frame every hop of 20 samples.

    cnt[0] is the number of AGC output samples seen so far, which is also the global index of the next one.
    Frame f is complete when sample 20f+19 arrives and covers samples 20f-236 .. 20f+19, where negative
    indices are the initial zeros. Returns the number of frames written to frames and gA."""
    nf = 0
    for i in range(y.shape[0]):
        o = cnt[0]
        ring[o % 256] = y[i]
        ring_g[o % 256] = g[i]
        if o >= 19 and (o - 19) % 20 == 0:
            for j in range(256):
                frames[nf, j] = ring[(o - 255 + j) % 256] * win[j]
            gA[nf] = g[i]
            nf += 1
        cnt[0] = o + 1
    return nf


@numba.njit(cache=True)
def env_kernel(Xr, Xi, lo, nbin, logCorr, gA, env, engy):
    """Port of GpyT hilbertEnvelopeFunc and channelEnergyFunc. Xr, Xi are (nf, 128); env, engy are (15, nf)."""
    nF, nB = Xr.shape
    eps = np.finfo(np.float64).eps
    for f in range(nF):
        gg = max(gA[f], eps)
        for c in range(lo.size):
            b0 = lo[c]
            e = 0.0
            b = b0
            rem = nbin[c]
            while rem > 0:
                k = 4 if rem >= 4 else rem
                sr = 0.0
                si = 0.0
                for q in range(b, b + k):
                    sg = -1.0 if (q % 2 == 0 and q < nB - 1) else 1.0      # Y[even] = -X[even]
                    sr += sg * Xr[f, q]
                    si += sg * Xi[f, q]
                e = e + sr * sr + si * si
                b += k
                rem -= k
            p = 0.0
            for q in range(b0, b0 + nbin[c]):
                a = np.hypot(Xr[f, q], Xi[f, q])
                p += a * a
            if e > 0:
                v = np.log2(e) + logCorr[c]
                env[c, f] = v if v > 0 else 0.0
            else:
                env[c, f] = 0.0
            engy[c, f] = np.sqrt(p) / gg


@numba.njit(cache=True)
def hist_kernel(Xr, Xi, f0, env_mod, lo, nbin, binToLoc, binWidth, nSteps, rng0, rng1, pk, h_env, h_w, h_pf):
    """Run the spectral peak locator and compute the current-steering weights.

    The peak is located on frames 2+3k and held for 3 frames; frames 0 and 1 use 0. The results are written,
    together with the envelopes, into the per-frame history rings. pk (2, 15) holds the current [freq; loc]."""
    nF = Xr.shape[0]
    floor = 10 ** (-120 / 20)
    nCh = lo.size
    for f in range(nF):
        fr = f0 + f
        if fr % 3 == 2:
            for c in range(nCh):
                mb = lo[c]
                best = -1.0
                for q in range(lo[c], lo[c] + nbin[c]):
                    v = max((Xr[f, q] * Xr[f, q] + Xi[f, q] * Xi[f, q]) / 2, floor)
                    if v > best:
                        best = v
                        mb = q
                mid = np.log2(best)
                left = np.log2(max((Xr[f, mb - 1] ** 2 + Xi[f, mb - 1] ** 2) / 2, floor))
                right = np.log2(max((Xr[f, mb + 1] ** 2 + Xi[f, mb + 1] ** 2) / 2, floor))
                mlr = max(left, right)
                if mid > mlr:
                    bc = 0.5 * (right - left) / (2 * mid - left - right)
                else:
                    bc = 0.5 * (1.0 if right == mlr else 0.0) - 0.5 * (1.0 if left == mlr else 0.0)
                pk[0, c] = binWidth * (mb + bc)
                dli = mb + (1 if bc > 0 else (-1 if bc < 0 else 0))
                pk[1, c] = binToLoc[mb] + bc * abs(binToLoc[mb] - binToLoc[dli])
        r = fr % h_env.shape[0]
        for c in range(nCh):
            h_env[r, c] = env_mod[c, f]
            h_pf[r, c] = pk[0, c]
            raw = max(min(pk[1, c] - c, 1.0), 0.0)
            if nSteps == 1:
                raw = 0.5
            else:
                raw = np.floor(raw * (nSteps - 1) + 0.5) / (nSteps - 1)
            hi = rng0 + raw * (rng1 - rng0)
            h_w[r, c] = 1 - hi
            h_w[r, c + nCh] = hi


@numba.njit(cache=True)
def ft_kernel(k0, last, dsc, df, rateFt, dphi_max, fOn, fOff, maxMod, cm, h_env, h_w, h_pf, phi_cum,
              mapA, mapK, mapClip, el_lo, el_hi, phase_off, sgn, build_el, LN2, LN10, I, Ie, em, eg):
    """Produce every F120 stimulation (FT) frame k >= k0 whose audio frame floor(k*dsc/df) is at most last.

    Each frame goes through carrier synthesis with a running phase, F120 mapping and current steering. The
    outputs are the per-channel current I[:, j] summed over the electrode pair, the per-electrode peak em, and,
    if build_el is set, the biphasic electrodogram in eg[:, 30*j:]. Ie[:, j] is the same current without the
    fine-structure carrier gating (the carrier mode 0 mapping). The acoustic vocoders render that envelope,
    since they cannot represent on/off pulse gating. Returns the number of FT frames."""
    nCh = el_lo.size
    R = h_env.shape[0]
    j = 0
    k = k0
    while j < I.shape[1]:
        idx = int(np.floor((k * dsc) / df))
        if idx > last:
            break
        r = idx % R
        for c in range(nCh):
            f = h_pf[r, c]
            d = min(f / rateFt, dphi_max)
            phi_cum[c] = phi_cum[c] + d
            ph = phi_cum[c] % 1.0
            md = maxMod * (fOff - min(max(f, fOn), fOff)) / (fOff - fOn)
            car = 1 - (md * (1.0 if ph < 0.5 else 0.0))
            e_db = h_env[r, c] * 10 * LN2 / LN10
            elo, ehi = el_lo[c], el_hi[c]
            if cm == 0:
                mlo = mapA[elo] * e_db + mapK[elo]
                mhi = mapA[ehi] * e_db + mapK[ehi]
            elif cm == 1:
                mlo = mapA[elo] * e_db * car + mapK[elo]
                mhi = mapA[ehi] * e_db * car + mapK[ehi]
            else:
                mlo = (mapA[elo] * e_db + mapK[elo]) * car
                mhi = (mapA[ehi] * e_db + mapK[ehi]) * car
            mlo = max(min(mlo, mapClip[elo]), 0.0) * h_w[r, c]
            mhi = max(min(mhi, mapClip[ehi]), 0.0) * h_w[r, c + nCh]
            I[c, j] = mlo + mhi
            Ie[c, j] = (max(min(mapA[elo] * e_db + mapK[elo], mapClip[elo]), 0.0) * h_w[r, c]
                        + max(min(mapA[ehi] * e_db + mapK[ehi], mapClip[ehi]), 0.0) * h_w[r, c + nCh])
            if mlo > em[elo]:
                em[elo] = mlo
            if mhi > em[ehi]:
                em[ehi] = mhi
            if build_el:
                o = 30 * j + phase_off[c]
                eg[elo, o] = sgn * mlo
                eg[elo, o + 1] = -sgn * mlo
                eg[ehi, o] = sgn * mhi
                eg[ehi, o + 1] = -sgn * mhi
        j += 1
        k += 1
    return j


@numba.njit(cache=True)
def preemph_kernel(x, b, a, z, out):
    """Second-order scipy.signal.lfilter (transposed direct form II, a[0] = 1) with state z, same operation order."""
    for i in range(x.shape[0]):
        v = x[i]
        yv = b[0] * v + z[0]
        z[0] = b[1] * v + z[1] - a[1] * yv
        z[1] = b[2] * v - a[2] * yv
        out[i] = yv
