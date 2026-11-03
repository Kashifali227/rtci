# SPDX-License-Identifier: GPL-3.0-only
# Derived from the Generic Python Toolbox (GPyT, https://github.com/jabeim/AB-Generic-Python-Toolbox,
# GPL-3.0, by JAB / jabeim), a Python port of the Advanced Bionics Generic MATLAB Toolbox. rtci, 2026 Kashif Ali.
"""Streaming port of the GPyT HiRes Fidelity 120 pipeline and the GPyT electrodogram vocoder.

The code processes audio block by block and keeps all filter and detector state between calls, so it can run
in real time. Every stage of GPyT's batch pipeline (AB-Generic-Python-Toolbox/pipelineTemplate.py) is causal,
and each one is re-implemented here with that state made explicit:

    17.4 kHz audio --[audio hooks]--> pre-emphasis (IIR, zi) --> dual-loop AGC (numba, per-sample state)
      --> 256-pt windowed buffers @ hop 20 (870 fps) --> FFT --> Hilbert envelopes + channel energies
      --> AB noise reduction (numba, per-frame state) --[envelope hooks]--> spectral peak locator (every 3rd frame)
      --> current-steering weights --> carrier synthesis (running phase) --> F120 mapping
      --> electrodogram (16 electrodes @ 55.56 kHz, biphasic) --> vocoder (5 ms blocks, neural power state)
      --> 20 ERB-spaced tone carriers, magnitudes interpolated between blocks --> audio @ fs_out

Given the same 17.4 kHz signal in any chunking, the output matches the GPyT batch output. validate_gpyt.py
checks this stage by stage. There are two intended differences:
  * The vocoder is causal. GPyT places the magnitude of block b at the start of block b, which is 5 ms of
    lookahead. Here a segment is synthesized once its closing block is known: 5 ms more latency, same samples.
  * GPyT normalizes the whole output file to -25 dBFS RMS, which is not causal. Here a fixed gain is used
    instead (OUT_CAL, calibrated so that 65 dB SPL speech comes out near -25 dBFS, the GPyT level).
"""
from __future__ import annotations
from . import _svml  # noqa: F401,I001  (must be imported before numba, see _svml.py)

import collections
import importlib.util
from fractions import Fraction
from pathlib import Path

import numba
import numpy as np
import scipy.interpolate
import scipy.io
from scipy.signal import lfilter

from . import f120_kernels as K

from .gpyt_locate import GPYT_DIR  # noqa: E402  (GPyT is an external GPL-3.0 dependency)


def _load_vocoder_tools():
    """Import GpyT/Vocoder/vocoderTools.py by path, bypassing GpyT/__init__ (which needs nnresample)."""
    spec = importlib.util.spec_from_file_location("gpyt_vocoderTools", GPYT_DIR / "Vocoder" / "vocoderTools.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VT = _load_vocoder_tools()

FS = 17400                      # implant internal audio rate
PULSE_WIDTH_US = 18             # F120 phase duration; GPyT says not to change it
N_CHAN = 15
N_ELEC = 16

# GPyT normalizes every output file to -25 dBFS RMS, but a stream needs a fixed gain. This value maps the raw
# vocoder output for the AzBio 65 dB SPL reference file to -25 dBFS RMS. It is measured with
# validate_gpyt.py --calibrate and must be re-measured if the vocoder changes.
OUT_CAL = 0.371254


def default_params():
    """Return the stock GPyT F120 parameters, as set in pipelineTemplate.py."""
    win = 0.5 * (np.blackman(256) + np.hanning(256))
    envCoefs = np.array([-19, 55, 153, 277, 426, 596, 784, 983, 1189, 1393, 1587, 1763, 1915, 2035, 2118, 2160,
                         2160, 2118, 2035, 1915, 1763, 1587, 1393, 1189, 983, 784, 596, 426, 277, 153, 55, -19]) / (2 ** 16)
    binToLoc = np.concatenate((np.zeros(6,), np.array([
        256, 640, 896, 1280, 1664, 1920, 2176, 2432, 2688, 2944, 3157, 3328, 3499, 3648, 3776, 3904, 4032,
        4160, 4288, 4416, 4544, 4659, 4762, 4864, 4966, 5069, 5163, 5248, 5333, 5419, 5504, 5589, 5669, 5742,
        5815, 5888, 5961, 6034, 6107, 6176, 6240, 6304, 6368, 6432, 6496, 6560, 6624, 6682, 6733, 6784, 6835,
        6886, 6938, 6989, 7040, 7091, 7142, 7189, 7232, 7275, 7317, 7360, 7403, 7445, 7488, 7531, 7573, 7616,
        7659]), 7679 * np.ones((53,)))) / 512
    return dict(
        fs=FS, nFft=256, nHop=20, nChan=N_CHAN, startBin=6,
        nBinLims=np.array([2, 2, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 10, 56]),
        window=win, pulseWidth=PULSE_WIDTH_US,
        # pre-emphasis
        preNum=np.array([.7688, -1.5376, .7688]), preDen=np.array([1, -1.5299, .5453]),
        # AGC
        kneePt=4.476, compRatio=12,
        tauRelFast=-8 / (17400 * np.log(.9901)) * 1000, tauAttFast=-8 / (17400 * np.log(.25)) * 1000,
        tauRelSlow=-8 / (17400 * np.log(.9988)) * 1000, tauAttSlow=-8 / (17400 * np.log(.9967)) * 1000,
        maxHold=1305, g0=6.908, fastThreshRel=8, cSlowInit=0.5e-3, cFastInit=0.5e-3,
        decFact=8, envBufLen=32, gainBufLen=16, envCoefs=envCoefs,
        # noise reduction
        tau_speech=.0258, tau_noise=.219, threshHold=3, durHold=1.6, maxAtt=-12, snrFloor=-2, snrCeil=45,
        snrSlope=6.5, slopeFact=0.2,
        # peak locator / steering / carrier
        binToLocMap=binToLoc, nDiscreteSteps=9, steeringRange=1.0,
        fModOn=.5, fModOff=1.0, maxModDepth=1.0, deltaPhaseMax=0.5,
        # mapping
        mapM=500 * np.ones(16), mapT=50 * np.ones(16), mapIdr=60 * np.ones(16), mapGain=0 * np.ones(16),
        mapClip=2048 * np.ones(16), chanToElecPair=np.arange(16), carrierMode=1,
        # electrodogram
        channelOrder=np.array([1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12]), cathodicFirst=True,
    )


# ----------------------------------------------------------------------------------------------------------------
# numba kernels. Their state lives in small arrays owned by the caller, so it persists between calls.
# ----------------------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _agc_kernel(x, ctrl, st, ictr, hist_c, hist_x, ring_r, envCoefs, cst, y_out, g_out, use_fixed, fixed):
    """Sample-by-sample port of GpyT dualLoopTdAgcFunc for one channel, with the audio as control signal.

    st = [cSlow, cFast, cFastLowLimit, hold, rsum], ictr = [input count i, raw-gain count m],
    cst = [bAttSlow, bRelSlow, bAttFast, bRelFast, c0, c0_log2, g0, gainSlope, fastHdrm, maxHold, clip].
    Output sample n is x[n-16] * smpGain[n] with smpGain[n] = mean(G[floor(k/8)] for k = n-14 .. n+1).
    Returns the number of output samples written."""
    bAttSlow, bRelSlow, bAttFast, bRelFast = cst[0], cst[1], cst[2], cst[3]
    c0, c0_log2, g0, gainSlope, fastHdrm, maxHold, doclip = cst[4], cst[5], cst[6], cst[7], cst[8], cst[9], cst[10]
    nout = 0
    for q in range(x.shape[0]):
        i = ictr[0]
        hist_c[i % 32] = abs(ctrl[q])
        hist_x[i % 32] = x[q]
        if i >= 6 and (i - 6) % 8 == 0:
            env = 0.0
            for k in range(32):
                s = i - 31 + k
                if s >= 0:
                    env += hist_c[s % 32] * envCoefs[k]
            envFast = min(1.0, max(-1.0, env * fastHdrm))
            cSlow, cFast, cFLL, hold = st[0], st[1], st[2], st[3]
            if env > cSlow:
                fastThr = min(1.0, max(-1.0, cSlow * 10 ** (8 / 20)))
                if env > fastThr:
                    dh = 0.0
                    cFast = envFast * (1 - bAttFast) + cFast * bAttFast
                else:
                    dh = 2.0
                    cFLL = cSlow * 10 ** (-10 / 20)
                    cFast = envFast * (1 - bRelFast) + cFast * bRelFast
                cSlow = min(env, fastThr) * (1 - bAttSlow) + cSlow * bAttSlow
            elif hold == 0:
                dh = 0.0
                cFLL = cSlow * 10 ** (-10 / 20)
                cFast = envFast * (1 - bRelFast) + cFast * bRelFast
                cSlow = env * (1 - bRelSlow) + cSlow * bRelSlow
            else:
                dh = -1.0
                cFLL = cSlow * 10 ** (-10 / 20)
                cFast = envFast * (1 - bRelFast) + cFast * bRelFast
            hold = min(hold + dh, maxHold)
            cSlow = max(cSlow, c0)
            cFast = max(cFast, cFLL)
            c = max(cFast, cSlow)
            g = 2 ** (g0 + gainSlope * max(np.log2(max(c, 1e-16)) - c0_log2, 0.0))
            st[0], st[1], st[2], st[3] = cSlow, cFast, cFLL, hold
            for _ in range(8):                         # raw per-sample gain r[m] = G[floor(m/8)]
                m = ictr[1]
                ring_r[m % 16] = g                     # 16-tap moving average, as lfilter(ones(16)/16)
                acc = 0.0
                for r in range(16):
                    acc += ring_r[r] * (1.0 / 16.0)
                if m >= 1:
                    n = m - 1
                    xs = hist_x[(n - 16) % 32] if n >= 16 else 0.0
                    sg = fixed if use_fixed else acc            # with the AGC off, apply a constant gain
                    yv = xs * sg
                    if doclip > 0:
                        yv = min(1.0, max(-1.0, yv))
                    y_out[nout] = yv
                    g_out[nout] = sg
                    nout += 1
                ictr[1] = m + 1
        ictr[0] = i + 1
    return nout


@numba.njit(cache=True)
def _nr_kernel(A, Vs, Vn, Hold, HoldReady, HoldCount, alpha_s, alpha_n, thr, maxHold, gMin,
               snrFloor, snrCeil, snrSlope, slopeFact, G):
    """Frame-by-frame port of GpyT noiseReductionFunc with noiseEstDecimation = 1. Writes linear gains to G."""
    nCh, nF = A.shape
    for f in range(nF):
        for c in range(nCh):
            la = max(-100.0, 20.0 * np.log10(A[c, f]))
            Vs[c] = alpha_s * Vs[c] + (1 - alpha_s) * la
            steady = (Vs[c] - Vn[c]) < thr
            onset = (not steady) and HoldReady[c]
            holdm = (not steady) and (not HoldReady[c]) and Hold[c]
            upd = steady or ((not steady) and (not HoldReady[c]) and (not Hold[c]))
            if upd:
                Vn[c] = alpha_n * Vn[c] + (1 - alpha_n) * Vs[c]
            if onset:
                Hold[c] = True
                HoldReady[c] = False
                HoldCount[c] = maxHold
            if holdm:
                HoldCount[c] -= 1
                if HoldCount[c] <= 0:
                    Hold[c] = False
            if steady:
                Hold[c] = False
                HoldReady[c] = True
            snr = min(max(Vs[c] - Vn[c], snrFloor), snrCeil)
            G[c, f] = gMin + (1 - gMin) / (1 + np.exp(-slopeFact * (snr - snrSlope)))


@numba.njit(cache=True, fastmath=True)
def _vocoder_kernel(el, normRampT, normOffset, alpha, nl, nlExp, pstate, mAvg, energy):
    """Run the GPyT vocoder neural stage for one block of the electrodogram (16 x B).

    The stage maps electrode currents to the electric field, then to neural activity, through the leaky peak
    integrator of ActivityToPower, and finally to the block energy. pstate holds the last integrator value per
    neural place (GPyT's audioPwr[:, 0] carry-over). normRampT is the field matrix stored electrode-major
    (16 x 300) so that the inner loop over neural places reads contiguous memory.

    The loop over the 300 places has no branches, which lets it vectorize with SIMD exp from SVML (about 3.4x
    faster). The activity (min(exp(nl*x), e^nl) - 1) / (e^nl - 1), with x = max(0, field - offset) / 0.4 * 0.5,
    is exactly 0 for silent places because exp(0) = 1, so no branch is needed. Silent phases only decay.
    Restricting the loop to each electrode's spread support is slower, because variable loop bounds defeat the
    vectorizer. fastmath reassociation changes results at about the 1e-15 level (see validate_gpyt.py)."""
    nE, B = el.shape
    nR = normOffset.shape[0]
    s = np.zeros(nR)
    p = pstate.copy()
    ef = np.empty(nR)
    oma = 1 - alpha
    inv = 1.0 / (nlExp - 1)
    for k in range(B):
        m = 0
        for e in range(nE):
            if el[e, k] > 0:
                m += 1
        if m == 0:
            for r in range(nR):
                p[r] = p[r] * alpha
                s[r] += p[r]
        else:
            for r in range(nR):
                ef[r] = 0.0
            for e in range(nE):
                a = el[e, k]
                if a > 0:
                    for r in range(nR):
                        ef[r] += normRampT[e, r] * a
            for r in range(nR):
                x = max(0.0, ef[r] - normOffset[r]) / 0.4 * 0.5
                act = (min(np.exp(nl * x), nlExp) - 1) * inv
                pv = max(p[r] * alpha + act * oma, act)
                p[r] = pv
                s[r] += pv
    for r in range(nR):
        pstate[r] = p[r]
        energy[r] = (s[r] + p[r]) / mAvg


@numba.njit(cache=False)   # not cached: a cached caller could link a stale copy of _vocoder_kernel
def _gpyt_voc_kernel(el, nblk, blk, normRampT, normOffset, alpha, nl, nlExp, pstate, mAvg, Wtone, prev_mag, st,
                     num, den, fs_float, toneFreqs, phs, tone_ok, gain, out, t_base):
    """Vocode all complete 278-sample blocks of the electrodogram.

    For each block, the neural stage gives the energy, Wtone @ energy gives the tone magnitudes, and the output
    segment between the previous block's anchor and this one is synthesized with the same operation order as
    the per-segment synthesis in GPyT vocoderFunc. st = [blocks done b, has_prev]. Samples per block is the
    exact ratio num/den. Sample times are counted from t_base, and phs holds the carrier phases at t_base, so
    the sine arguments stay small (see StreamingVocoder._rebase). Returns the number of output samples written."""
    nR = normOffset.shape[0]
    nT = toneFreqs.shape[0]
    energy = np.empty(nR)
    e = np.empty((el.shape[0], blk))
    mag = np.empty(nT)
    spb = num / den
    pos = 0
    for i in range(nblk):
        for r in range(el.shape[0]):
            for k in range(blk):
                e[r, k] = el[r, i * blk + k]
        _vocoder_kernel(e, normRampT, normOffset, alpha, nl, nlExp, pstate, mAvg, energy)
        for t in range(nT):
            acc = 0.0
            for r in range(nR):
                acc += Wtone[t, r] * energy[r]
            mag[t] = acc
        b = st[0]
        if st[1] > 0:
            lo = -((-(b - 1) * num) // den)                   # ceil((b-1)*spb) in exact integer arithmetic
            hi = -((-b * num) // den)
            x_lo = float((b - 1) * num) / den
            for n in range(lo, hi):
                tt = (n - t_base) / fs_float
                v = 0.0
                for t in range(nT):
                    if tone_ok[t]:
                        slope = (mag[t] - prev_mag[t]) / spb
                        m = abs(slope * (n - x_lo) + prev_mag[t])
                        v += np.sin(2. * np.pi * toneFreqs[t] * tt + phs[t]) * m
                out[pos] = gain * v
                pos += 1
        for t in range(nT):
            prev_mag[t] = mag[t]
        st[1] = 1
        st[0] = b + 1
    return pos


# ----------------------------------------------------------------------------------------------------------------
def _mono(x, what):
    """Return an audio block as 1-D float64. Shapes (n, 1) and (1, n) are accepted; multichannel data is not."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        if sum(d > 1 for d in x.shape) > 1:
            raise ValueError(f"{what} must be mono (1-D), got shape {x.shape}")
        x = x.ravel()
    return x


def _env_out(e, shape):
    """Check that an EnvHook result is finite and has the (15, n_frames) shape of its input."""
    e = np.asarray(e, dtype=np.float64)
    if e.shape != shape:
        raise ValueError(f"EnvHook returned shape {e.shape}, expected {shape}")
    if not np.isfinite(e).all():
        raise ValueError("EnvHook returned non-finite values")
    return e


# Values the streaming kernels hard-code. Any other value would be silently ignored or would overrun a buffer.
_FIXED_PARAMS = dict(fs=FS, nFft=256, nHop=20, nChan=N_CHAN, decFact=8, envBufLen=32, gainBufLen=16,
                     pulseWidth=PULSE_WIDTH_US)


def validate_params(p):
    """Raise ValueError for any parameter value rtci does not support, rather than processing it wrongly."""
    for k, v in _FIXED_PARAMS.items():
        if k in p and np.any(np.asarray(p[k]) != v):
            raise ValueError(f"parameter {k}={p[k]!r} is not supported (rtci implements {k}={v})")
    for k, v in dict(clipMode="limit", controlMode="naida", noiseEstDecimation=1).items():
        if k in p and (str(p[k]).lower() if isinstance(v, str) else p[k]) != v:
            raise ValueError(f"parameter {k}={p[k]!r} is not supported (rtci implements {k}={v!r})")
    if np.ndim(p["steeringRange"]) != 0:
        raise ValueError("a per-channel steeringRange is not supported; use a scalar")
    pairs = np.asarray(p["chanToElecPair"][:N_CHAN])
    if not np.array_equal(pairs, np.arange(N_CHAN)):
        raise ValueError("chanToElecPair must be the identity mapping 0..14 (other mappings are not implemented)")
    if int(p["carrierMode"]) not in (0, 1, 2):
        raise ValueError(f"carrierMode must be 0, 1 or 2, got {p['carrierMode']!r}")


# ----------------------------------------------------------------------------------------------------------------
class StreamingF120:
    """Turn 17.4 kHz audio into an electrodogram (16 x n at 55.56 kHz), in chunks of any size.

    These attributes may be changed between calls: nr_enabled, agc_enabled, steering ('f120' or 'fixed'),
    carrier_mode (0, 1 or 2), and the hook lists audio_hooks and env_hooks (see hooks.py)."""

    def __init__(self, par=None):
        self.p = p = par or default_params()
        validate_params(p)
        self.nr_enabled = True
        # With the AGC off, a fixed linear gain is applied instead, with no compression. 38.8 dB is the median gain
        # the dual-loop AGC applies to 65 dB SPL speech (measured on the AzBio reference), so normal speech still
        # lands at the same place on the map. Without any gain, speech would fall below T-level.
        self.agc_enabled = True
        self.agc_fixed_db = 38.8
        self.cnt = np.zeros(1, np.int64)                 # AGC output samples seen so far, used for framing
        self.yring, self.gring = np.zeros(256), np.zeros(256)
        self.h_env = np.zeros((K.RING, N_CHAN))
        self.h_w = np.zeros((K.RING, 2 * N_CHAN))
        self.h_pf = np.zeros((K.RING, N_CHAN))
        self.pk = np.zeros((2, N_CHAN))
        self.steering = "f120"
        self.carrier_mode = p["carrierMode"]
        self.audio_hooks = []
        self.env_hooks = []
        self.on_frames = None            # optional telemetry callback, called with a dict per batch of frames
        self.hook_errors = collections.deque(maxlen=5)    # why recent hooks were switched off
        self.n_nonfinite = 0             # count of non-finite input samples replaced by 0
        self.last_I = self.last_I_env = np.zeros((N_CHAN, 0))

        # pre-emphasis
        self.pre_zi = np.zeros(2)
        # AGC
        fs, dec = p["fs"], p["decFact"]
        c0_log2 = p["kneePt"] - 15
        self.agc_cst = np.array([
            np.exp(-dec / fs * 1000 / p["tauAttSlow"]), np.exp(-dec / fs * 1000 / p["tauRelSlow"]),
            np.exp(-dec / fs * 1000 / p["tauAttFast"]), np.exp(-dec / fs * 1000 / p["tauRelFast"]),
            2 ** c0_log2, c0_log2, p["g0"], 1 / p["compRatio"] - 1, 10 ** (-p["fastThreshRel"] / 20),
            p["maxHold"], 1.0])
        self.agc_st = np.array([p["cSlowInit"], p["cFastInit"], p["cFastInit"], 0.0, 0.0])
        self.agc_ictr = np.zeros(2, np.int64)
        self.agc_hc = np.zeros(32)
        self.agc_hx = np.zeros(32)
        self.agc_ring = np.zeros(16)
        self.envCoefs = np.asarray(p["envCoefs"], float)
        # framing
        self.nFft, self.nHop = p["nFft"], p["nHop"]
        self.win = np.asarray(p["window"], float).ravel()
        ov = self.nFft - self.nHop
        self.ybuf = np.zeros(ov)                 # AGC output history; ybuf[0] has global index ybuf_t0
        self.gbuf = np.zeros(ov)
        self.ybuf_t0 = -ov
        self.n_frames = 0                        # next audio-frame index
        # filterbank bookkeeping
        self.startBin = p["startBin"] - 1
        self.nBinLims = p["nBinLims"]
        nb = self.nBinLims
        self.chan_bins = []
        cb = self.startBin
        for c in range(N_CHAN):
            self.chan_bins.append((cb, cb + nb[c]))
            cb += nb[c]
        self.lo_arr = np.array([b[0] for b in self.chan_bins], np.int64)
        self.nb_arr = np.asarray(nb, np.int64)
        logFiltCorrect = np.array([2233, 952, 62, 0]) / (2 ** 10)
        logCorrect = logFiltCorrect + 0 + 16
        self.logCorr = np.array([logCorrect[-1] if nb[c] > logCorrect.size - 1 else logCorrect[nb[c] - 1]
                                 for c in range(N_CHAN)])
        # noise reduction
        dt = p["nHop"] / p["fs"]
        self.nr_as, self.nr_an = np.exp(-dt / p["tau_speech"]), np.exp(-dt / p["tau_noise"])
        self.nr_maxHold = p["durHold"] / dt
        maxAttLin = 10 ** (-abs(p["maxAtt"]) / 20)
        self.nr_gMin = 1 + (maxAttLin - 1) / (1 - 1 / (1 + np.exp(-p["slopeFact"] * (p["snrFloor"] - p["snrSlope"]))))
        # Zeros, not initState: GPyT looks the initState keys up in par, so its initState never takes effect.
        self.nr_Vs, self.nr_Vn = np.zeros(N_CHAN), np.zeros(N_CHAN)
        self.nr_Hold = np.zeros(N_CHAN, np.bool_)
        self.nr_HoldReady = np.ones(N_CHAN, np.bool_)
        self.nr_HoldCount = np.zeros(N_CHAN) + self.nr_maxHold
        # peak locator and steering: frames 0 and 1 use zeros, then the peak of frame 2+3k is held for 3 frames
        self.pk_freq = np.zeros(N_CHAN)
        self.pk_loc = np.zeros(N_CHAN)
        # carrier and FT frames. They read the per-frame history from the ring buffers h_env, h_w and h_pf above.
        self.durFrame = p["nHop"] / p["fs"]
        self.durStimCycle = 2 * p["pulseWidth"] * N_CHAN * 1e-6
        self.rateFt = np.round(1 / self.durStimCycle)
        self.k_ft = 0
        self.phi_cum = np.zeros(N_CHAN)
        # mapping constants
        M, T, IDR, gain = p["mapM"], p["mapT"], p["mapIdr"], p["mapGain"]
        mSat = 30 * 10 * np.log(2) / np.log(10)
        self.mapA = (M - T) / IDR
        self.mapK = M + (M - T) / IDR * (-mSat + 12 + gain)
        self.phase_off = 2 * (p["channelOrder"] - 1)
        self.sign = -1.0 if p["cathodicFirst"] else 1.0
        # constants for the fused kernels
        self.preB = np.asarray(p["preNum"], float)
        self.preA = np.asarray(p["preDen"], float)
        self.binToLoc = np.asarray(p["binToLocMap"], float)
        self.binWidth = p["fs"] / p["nFft"]
        rng = 0.5 + 0.5 * p["steeringRange"] * np.array([-1.0, 1.0])
        self.rng0, self.rng1 = float(rng[0]), float(rng[1])
        self.fOn, self.fOff = self.rateFt * p["fModOn"], self.rateFt * p["fModOff"]
        self.mapClip = np.asarray(p["mapClip"], float)
        self.el_lo = np.asarray(p["chanToElecPair"][:N_CHAN], np.int64)
        self.el_hi = self.el_lo + 1
        self.phase_off_i = np.asarray(self.phase_off, np.int64)

    # --------------------------------------------------------------------------------------------------------
    def process(self, x, build_el=True):
        """Process a 1-D block of 17.4 kHz audio of any length and return the electrodogram chunk (16 x n).

        build_el=False skips building the 55.6 kHz electrodogram, which only the GPyT vocoder needs. The channel
        currents (last_I) and the per-electrode peaks for display (last_elec_max) are always produced."""
        x = _mono(x, "input")
        for h in self.audio_hooks:
            if getattr(h, "enabled", True):
                x = self._run_hook(h, x, lambda h=h, x=x: _mono(h.process(x), "AudioHook output"))
        bad = ~np.isfinite(x)
        if bad.any():                     # a single NaN or inf would corrupt the pre-emphasis and AGC state for good
            self.n_nonfinite += int(bad.sum())
            x = np.where(bad, 0.0, x)
        self._build_el = build_el
        self.last_elec_max = None
        if x.size == 0:                                     # e.g. a buffering hook emitted nothing this call
            self.last_I = self.last_I_env = np.zeros((N_CHAN, 0))
            self._last_agc = (np.zeros(0), np.zeros(0))
            return np.zeros((N_ELEC, 0))
        # The per-frame history ring holds only K.RING frames, so long inputs (offline renders) are split.
        CH = 4096
        if x.size > CH:
            els, Is, Ies, em, ys, gs = [], [], [], None, [], []
            for i in range(0, x.size, CH):
                els.append(self._process_chunk(x[i:i + CH]))
                Is.append(self.last_I)
                Ies.append(self.last_I_env)
                ys.append(self._last_agc[0])
                gs.append(self._last_agc[1])
                if self.last_elec_max is not None:
                    em = self.last_elec_max if em is None else np.maximum(em, self.last_elec_max)
            self.last_I, self.last_elec_max = np.concatenate(Is, axis=1), em
            self.last_I_env = np.concatenate(Ies, axis=1)
            self._last_agc = (np.concatenate(ys), np.concatenate(gs))
            return np.concatenate(els, axis=1)
        return self._process_chunk(x)

    def _run_hook(self, h, fallback, call):
        """Run one plug-in hook, switching it off if it fails.

        A hook that raises or returns invalid output is disabled and reported in hook_errors, and its input is
        passed on unchanged. A faulty block therefore cannot stop or corrupt the chain."""
        try:
            return call()
        except Exception as ex:
            h.enabled = False
            self.hook_errors.append(f"{getattr(h, 'name', type(h).__name__)} switched off: {ex!r}")
            return fallback

    def _process_chunk(self, x):
        self.last_elec_max = None
        p = self.p
        xp = np.empty(x.size)
        K.preemph_kernel(x, self.preB, self.preA, self.pre_zi, xp)
        y = np.empty(xp.size + 16)
        g = np.empty(xp.size + 16)
        fixed = 10 ** (self.agc_fixed_db / 20)
        n = _agc_kernel(xp, xp, self.agc_st, self.agc_ictr, self.agc_hc, self.agc_hx, self.agc_ring,
                        self.envCoefs, self.agc_cst, y, g, not self.agc_enabled, fixed)
        self._last_agc = (y[:n], g[:n])
        self._frames(y[:n], g[:n])
        return self._ft_frames()

    def _frames(self, y, g):
        maxnf = y.size // self.nHop + 2
        frames = np.empty((maxnf, self.nFft))
        gA = np.empty(maxnf)
        f0 = self.n_frames
        nf = K.frame_kernel(y, g, self.yring, self.gring, self.cnt, self.win, frames, gA)
        if nf == 0:
            return
        self.n_frames += nf
        X = np.fft.fft(frames[:nf], self.nFft, axis=1)[:, : self.nFft // 2]      # (nf, 128)
        Xr, Xi = np.ascontiguousarray(X.real), np.ascontiguousarray(X.imag)
        env = np.empty((N_CHAN, nf))
        engy = np.empty((N_CHAN, nf))
        K.env_kernel(Xr, Xi, self.lo_arr, self.nb_arr, self.logCorr, gA[:nf], env, engy)
        if self.nr_enabled:
            G = np.empty_like(engy)
            _nr_kernel(engy, self.nr_Vs, self.nr_Vn, self.nr_Hold, self.nr_HoldReady, self.nr_HoldCount,
                       self.nr_as, self.nr_an, float(self.p["threshHold"]), self.nr_maxHold, self.nr_gMin,
                       float(self.p["snrFloor"]), float(self.p["snrCeil"]), float(self.p["snrSlope"]),
                       float(self.p["slopeFact"]), G)
            env_mod = env + 2 * np.log2(G)
        else:
            G = None
            env_mod = env
        if self.env_hooks:
            active = [h for h in self.env_hooks if getattr(h, "enabled", True)]
            if active:
                ctx = dict(frame0=f0, fft=X.T, energy=engy, nr_gain=G, env_raw=env)
                for h in active:
                    env_mod = self._run_hook(h, env_mod, lambda h=h, e=env_mod: _env_out(h.process(e, ctx), e.shape))
        nSteps = self.p["nDiscreteSteps"] if self.steering == "f120" else 1
        K.hist_kernel(Xr, Xi, f0, np.ascontiguousarray(env_mod), self.lo_arr, self.nb_arr, self.binToLoc,
                      self.binWidth, nSteps, self.rng0, self.rng1, self.pk, self.h_env, self.h_w, self.h_pf)
        if self.on_frames is not None:
            self.on_frames(dict(frame0=f0, env=env_mod, energy=engy))

    def _ft_frames(self):
        """Build electrodogram columns for every FT (stimulation) frame whose source audio frame exists."""
        p = self.p
        last = self.n_frames - 1
        if last < 0:
            self.last_I = self.last_I_env = np.zeros((N_CHAN, 0))
            return np.zeros((N_ELEC, 0))
        cap = max(4, int((last + 1) * self.durFrame / self.durStimCycle) + 4 - self.k_ft)
        I = np.empty((N_CHAN, cap))
        Ie = np.empty((N_CHAN, cap))
        em = np.zeros(N_ELEC)
        eg = np.zeros((N_ELEC, 30 * cap)) if self._build_el else np.zeros((N_ELEC, 1))
        nft = K.ft_kernel(self.k_ft, last, self.durStimCycle, self.durFrame, self.rateFt, p["deltaPhaseMax"],
                          self.fOn, self.fOff, p["maxModDepth"], self.carrier_mode, self.h_env, self.h_w, self.h_pf,
                          self.phi_cum, self.mapA, self.mapK, self.mapClip, self.el_lo, self.el_hi,
                          self.phase_off_i, self.sign, self._build_el, np.log(2), np.log(10), I, Ie, em, eg)
        if nft == 0:
            self.last_I = self.last_I_env = np.zeros((N_CHAN, 0))
            return np.zeros((N_ELEC, 0))
        self.k_ft += nft
        if self.phi_cum.max() > 1e6:                        # bound the running phase; an integer shift is exact
            self.phi_cum -= np.floor(self.phi_cum)
        self.last_I = I[:, :nft]
        self.last_I_env = Ie[:, :nft]
        self.last_elec_max = em
        return eg[:, : 30 * nft] if self._build_el else np.zeros((N_ELEC, 0))


# ----------------------------------------------------------------------------------------------------------------
class StreamingVocoder:
    """Streaming port of GpyT vocoderFunc: electrodogram (16 x n at 55.56 kHz) in, audio at fs_out out.

    timing='gpyt' puts block anchors at b*tAvg with tAvg = 278/55556 s, as GPyT assumes. Use it with
                  fs_out = GPYT_AUDIO_FS to reproduce GPyT sample for sample.
    timing='real' puts block anchors at b*278*18 us, the true pulse timebase. This gives exactly fs_out samples
                  per second of input, so the output does not drift against the sound card."""

    REBASE = 2 ** 21          # carrier time origin moves forward every this many output samples, see _rebase()

    def __init__(self, fs_out=48000, timing="real", phases=None, n_carriers=20, gain=OUT_CAL, seed=0):
        captFs = 55556
        captTs = 1 / captFs
        nElec = N_ELEC
        nNeural = 300
        tAvg = np.ceil(0.005 / captTs) * captTs
        self.blk = int(np.round(tAvg / captTs))               # 278
        mAvg = np.round(tAvg / captTs)
        self.mAvg = mAvg
        audioFs = np.ceil(tAvg * 48000) / tAvg                # GPyT's internal rate (vocoderFunc default audioFs)
        nFFT = int(np.round(2 * tAvg / (1 / audioFs)))        # 482
        self.gpyt_audio_fs = audioFs
        elecFreqs = np.logspace(np.log10(381.5), np.log10(5046.4), nElec)
        spread = scipy.io.loadmat(str(GPYT_DIR / "MatlabSupportFiles" / "spread.mat"))
        neuralLocsOct = np.append(np.log2(np.linspace(150, 850, 40)), np.linspace(np.log2(870), np.log2(8000), 260))
        neuralLocsOct = np.interp(np.linspace(1, neuralLocsOct.size, nNeural), np.arange(1, neuralLocsOct.size + 1),
                                  neuralLocsOct)
        self.alpha = np.exp(-1 / ((10 / 1000) * captFs))
        MCL = np.ones(nElec) * 500 * 1.2
        T = np.ones(nElec) * 50
        charge2EF = np.zeros((nNeural, nElec))
        elecFreqOct = np.log2(elecFreqs)
        for iEl in range(nElec):
            f = scipy.interpolate.interp1d(spread["fOct"][:, 0] + elecFreqOct[iEl], spread["voltage"][:, 0],
                                           fill_value="extrapolate")
            sv = f(neuralLocsOct)
            sv[sv < 0] = 0
            charge2EF[:, iEl] = sv
        mNeurToBin = VT.NeurToBinMatrix(neuralLocsOct, nFFT, audioFs)
        Mn = np.interp(neuralLocsOct, elecFreqOct, MCL)
        Mn[neuralLocsOct < elecFreqOct[0]] = MCL[0]
        Mn[neuralLocsOct > elecFreqOct[-1]] = MCL[-1]
        Tn = np.interp(neuralLocsOct, elecFreqOct, T)
        Tn[neuralLocsOct < elecFreqOct[0]] = T[0]
        Tn[neuralLocsOct > elecFreqOct[-1]] = T[-1]
        self.normRamp = np.ascontiguousarray(np.multiply(charge2EF.T, 1 / (Mn - Tn)).T)
        self.normOffset = (Tn / (Mn - Tn)).reshape((Tn.size, 1))
        self.nl = 5
        self.nlExp = np.exp(self.nl)
        self.toneFreqs = np.array(VT.generate_cfs(20, 20000, n_carriers))
        fftFreqs = np.arange(1, np.floor(nFFT / 2) + 1) * audioFs / nFFT
        # GPyT gets each tone magnitude by linear interpolation of mNeurToBin @ energy from fftFreqs to toneFreqs.
        # Both steps are linear, so they fold into one matrix.
        Wi = scipy.interpolate.interp1d(fftFreqs, np.eye(fftFreqs.size), axis=0, fill_value="extrapolate")(self.toneFreqs)
        self.Wtone = Wi @ mNeurToBin                          # (n_carriers x 300)
        if phases is None:                                    # GPyT draws random carrier phases; we seed them
            phases = 2 * np.pi * np.random.default_rng(seed).random(int(np.floor(nFFT / 2)))
        self.phs = np.asarray(phases, float)[:n_carriers]
        self.t_base = 0                                       # carrier time origin (output sample index)
        self.phs_base = self.phs.copy()                       # carrier phases at t_base
        self.pstate = np.zeros(nNeural)                       # GPyT audioPwr[:, 0] carry-over
        self.normOffset1 = np.ascontiguousarray(self.normOffset.ravel())
        self.normRampT = np.ascontiguousarray(self.normRamp.T)   # electrode-major for the neural kernel
        self.fs_out = fs_out
        self.set_timing(timing)
        self.gain = gain
        self.elbuf = np.zeros((nElec, 0))
        self.prev_mag = None
        self.b = 0                                            # blocks processed
        self.n_out = 0                                        # output samples emitted

    def set_timing(self, timing):
        blk_dur = Fraction(278, 55556) if timing == "gpyt" else Fraction(self.blk * PULSE_WIDTH_US, 10 ** 6)
        fs = Fraction(self.fs_out).limit_denominator(10 ** 9) if not isinstance(self.fs_out, Fraction) else self.fs_out
        self.spb = blk_dur * fs                               # output samples per vocoder block (exact)
        self.fs_exact = fs
        self.fs_float = float(fs)
        self.tone_ok = self.toneFreqs < 0.45 * self.fs_float  # drop carriers that would alias at a low fs_out

    def process(self, el):
        """Vocode a (16 x n) electrodogram chunk. Returns 1-D audio at fs_out; the length varies per call."""
        if el.shape[1]:
            self.elbuf = np.concatenate((self.elbuf, el), axis=1) if self.elbuf.shape[1] else el
        nblk = self.elbuf.shape[1] // self.blk
        if nblk == 0:
            return np.zeros(0)
        num, den = self.spb.numerator, self.spb.denominator
        out = np.empty(nblk * (num // den + 2))
        if self.prev_mag is None:
            self.prev_mag = np.zeros(self.Wtone.shape[0])
            self._st = np.array([self.b, 0], np.int64)
        self._st[0] = self.b
        self._rebase()
        n = _gpyt_voc_kernel(np.ascontiguousarray(self.elbuf), nblk, self.blk, self.normRampT, self.normOffset1,
                             self.alpha, float(self.nl), self.nlExp, self.pstate, float(self.mAvg), self.Wtone,
                             self.prev_mag, self._st, num, den, self.fs_float, self.toneFreqs, self.phs_base,
                             self.tone_ok, self.gain, out, self.t_base)
        self.b = int(self._st[0])
        self.n_out = -((-(self.b - 1) * num) // den) if self._st[1] else 0   # ceil((b-1)*spb), the last anchor
        self.elbuf = self.elbuf[:, nblk * self.blk:]
        return out[:n]

    def _rebase(self):
        """Move the carrier time origin forward every 2**21 samples (~44 s at 48 kHz).

        The carriers are sin(2 pi f t + phase). Measured from the stream start, the argument grows without bound,
        and numba's sin slows down as it grows (33 ns per call at start, 57 ns after 10 minutes, 69 ns after 30),
        which pushed a long live session past its block budget. The phase advance up to the new origin is computed
        in exact rational arithmetic, so the output is unchanged up to float rounding. Before the first rebase it
        is bit-identical to synthesis on absolute time, which keeps the GPyT equivalence check exact."""
        num, den = self.spb.numerator, self.spb.denominator
        n_now = -((-max(self.b - 1, 0) * num) // den)
        if n_now - self.t_base < self.REBASE:
            return
        cyc = [Fraction(float(f)) * n_now / self.fs_exact for f in self.toneFreqs]
        self.phs_base = np.array([(p + 2 * np.pi * float(c - (c.numerator // c.denominator))) % (2 * np.pi)
                                  for p, c in zip(self.phs, cyc)])
        self.t_base = n_now

class CIProcessor:
    """The full chain from 17.4 kHz audio to vocoded audio at fs_out.

    It holds the strategy and the selected vocoder. The vocoder is either 'gpyt' (the GPyT electrodogram
    vocoder) or one of the acoustic vocoders in vocoders_rt ('noise', 'sine' or 'get'), which are driven by the
    per-channel stimulation currents of the same F120 frames."""

    def __init__(self, fs_out=48000, timing="real", phases=None, par=None, vocoder="gpyt", voc_opts=None, seed=0):
        """seed sets the random carriers: GPyT tone phases, noise bands and GET phases. Give each ear of a
        bilateral simulation its own seed, otherwise both ears get identical carriers, unlike two real implants."""
        self.fs_out, self.timing, self.par, self.seed = fs_out, timing, par, seed
        self.strategy = StreamingF120(par)
        self.vocoder = StreamingVocoder(fs_out=fs_out, timing=timing, phases=phases, seed=seed)
        self.on_electrodogram = None
        self.set_vocoder(vocoder, **(voc_opts or {}))

    def build_vocoder(self, kind, **opts):
        """Build a vocoder without touching the running chain. This is slow; install_vocoder() then swaps it in."""
        if kind == "gpyt":
            return None
        from .vocoders_rt import AcousticVocoder
        return AcousticVocoder(kind, self.fs_out, par=self.par, **{"seed": self.seed, **opts})

    def set_vocoder(self, kind, **opts):
        self.install_vocoder(kind, opts, self.build_vocoder(kind, **opts))

    def install_vocoder(self, kind, opts, voc):
        """Align a built vocoder to the running stimulation timebase and make it current. This is fast."""
        self.kind, self.voc_opts = kind, opts
        self.acoustic = voc
        if voc is not None:
            self.acoustic.k = self.strategy.k_ft
            self.acoustic.n = int(-((-self.acoustic.k * self.acoustic.spf) // 1))
            if kind == "get":
                self.acoustic.emitted = self.acoustic.n - self.acoustic.H
                self.acoustic.env_t0 = self.acoustic.n
                a = self.acoustic                           # first pulse of each channel at or after sample n
                a.pulse = np.ceil(a.n * a.rate / a.fs - a.offs).astype(np.int64)

    def process(self, x17):
        el = self.strategy.process(x17, build_el=self.acoustic is None)
        em = self.strategy.last_elec_max
        if self.on_electrodogram is not None and em is not None:
            self.on_electrodogram(em)
        if self.acoustic is None:
            return self.vocoder.process(el)
        return self.acoustic.process(self.strategy.last_I_env)
