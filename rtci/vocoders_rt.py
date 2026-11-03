# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
"""
Streaming acoustic CI vocoders (noise-band, sine and GET) driven by the F120 stimulation.

All three vocoders take what the implant delivers: the per-channel stimulation current of every F120 stimulation
frame (15 channels at 1852 frames/s, in clinical units, after AGC, noise reduction, current steering and mapping).
The AGC, noise reduction, steering and hook switches therefore affect every vocoder in the same way, and only the
acoustic carrier differs.

The fine-structure carrier (carrierMode 1 or 2) is the exception. It gates the pulses on and off at the channel's
peak frequency, which an envelope vocoder cannot represent. Rendering the gated current would only lower the low
channels by about 6 dB and smear energy outside the bands. These vocoders therefore receive the same current without
the gating (StreamingF120.last_I_env), and the fine-structure switch is audible only through the GPyT vocoder.

Current to acoustic envelope
    The clinical map is inverted to a 60 dB acoustic dynamic range, level_dB = IDR * (I - M) / (M - T), so that T
    maps to -60 dB and M to 0 dB. Currents at or below T give zero amplitude.

Envelope to audio
    The amplitude is held over each 540 us stimulation frame, smoothed by a second-order low-pass (env_lp, 300 Hz by
    default, the usual temporal-envelope cutoff) and multiplied by one carrier per analysis band.

Bands
    The F120 analysis channels themselves (FFT-bin groups from 306 Hz to 8.05 kHz), with carriers at the geometric
    band centers.

Carriers
    noise   Noise-band vocoder after Shannon, Zeng, Kamath, Wygonski & Ekelid (1995), Science 270:303-304. Each
            channel uses white noise shaped by a 4th-order Butterworth band-pass magnitude and normalized to the RMS
            of a sine. The carriers are precomputed as seamless 5 s loops, so they cost nothing per block.

    sine    Tone vocoder after Dorman, Loizou & Rainey (1997), JASA 102:2403, with a sinusoid at each band center.

    get     Pulsatile Gaussian-enveloped tones after Meng, Zhou, Lu & Zeng, "Pulsatile Gaussian-Enveloped Tones
            (GET) vocoders for cochlear-implant simulation" (medRxiv 10.1101/2022.02.21.22270929). It follows the
            authors' reference implementation (github.com/BetterCI/GETVocoder, GETvoc.m), re-implemented here with
            no code copied.

            Each channel carries a pulse train at pulse_rate (100 pps by default). The channels are interleaved in
            the F120 stimulation order, so channel c fires slot_c / 15 of a period after slot 0. The reference also
            places each atom at its own electrode's pulse time. Firing all channels together would stack 15 peaks,
            giving a crest factor of about 29 dB that clips at normal speech levels.

            Pulse k adds the Gaussian envelope A_k exp(-pi (t - t_k)^2 / D^2) with D = l / fc (l = 3), truncated at
            +/-1.5 D. A_k is the acoustic envelope at the pulse time. Overlapping envelopes are combined by their
            maximum, which is the reference default. The channel envelope then multiplies a continuous carrier
            sin(2 pi fc t + phi) on absolute time, with a random phase per channel, so the atoms never interfere at
            any pulse rate.

            Like the reference, which rescales each channel to the RMS of its current, every channel is normalized
            so that its output follows its envelope regardless of fc and the pulse rate. A steady envelope A gives a
            channel RMS of A / sqrt(2), as in the sine vocoder. This avoids a spectral tilt and keeps the level
            independent of the rate. Because the envelopes are centered on the pulse time, GET needs 1.5 D of the
            lowest band as look-ahead latency, about 13 ms at l = 3.
"""
from __future__ import annotations
from . import _svml  # noqa: F401,I001  (must precede numba: enables SVML vector math, see _svml.py)

from fractions import Fraction

import numpy as np
from scipy.signal import butter, lfilter

import numba


@numba.njit(cache=True)
def _env_kernel(I, k, n, n1, num, den, held, T, M, IDR, b, a, zi, env):
    """Convert stimulation current to the acoustic envelope for samples n to n1 - 1.

    Inverts the clinical map (T to -60 dB, M to 0 dB, zero at or below T), holds each value over its 540 us
    stimulation frame and applies the second-order low-pass. The filter matches scipy's lfilter (transposed direct
    form II) with state zi per channel, and the result is clamped at 0. Sample m belongs to frame
    floor(m * den / num) - k, where -1 is the frame held over from the previous call.
    """
    nCh, nF = I.shape
    amp = np.empty((nCh, nF))
    for c in range(nCh):
        for f in range(nF):
            x = I[c, f]
            lvl = IDR * (x - M) / (M - T)
            amp[c, f] = 10 ** (lvl / 20) if x > T else 0.0
    for i in range(n1 - n):
        fi = ((n + i) * den) // num - k
        for c in range(nCh):
            v = held[c] if fi < 0 else amp[c, fi]
            yv = b[0] * v + zi[c, 0]
            zi[c, 0] = b[1] * v + zi[c, 1] - a[1] * yv
            zi[c, 1] = b[2] * v - a[2] * yv
            env[c, i] = yv if yv > 0 else 0.0
    if nF:
        for c in range(nCh):
            held[c] = amp[c, nF - 1]


@numba.njit(cache=True)
def _noise_kernel(env, table, n0, gain, out):
    L = table.shape[1]
    for i in range(env.shape[1]):
        j = (n0 + i) % L
        v = 0.0
        for c in range(env.shape[0]):
            v += env[c, i] * table[c, j]
        out[i] = gain * v


@numba.njit(cache=True, fastmath=True)
def _sine_kernel(env, phase, w, ok, gain, out):
    """Mix the sine carriers, cos(phase + w * (t - 1)) for t = 1 to ns, and advance each phase by w * ns mod 2 pi."""
    ns = env.shape[1]
    for i in range(ns):
        out[i] = 0.0
    for c in range(env.shape[0]):
        if ok[c]:
            for i in range(ns):
                out[i] += env[c, i] * np.cos(phase[c] + w[c] * (i + 1) - w[c])
        phase[c] = (phase[c] + w[c] * ns) % (2 * np.pi)
    for i in range(ns):
        out[i] *= gain


@numba.njit(cache=True)
def _get_kernel(env_hist, env_t0, pulse, offs, step, n1, gauss, halves, ok, acc, emitted):
    """Add the Gaussian atoms of every pulse before sample n1 to the per-channel GET envelopes.

    pulse[c] is the index of the next pulse of channel c, which sits at sample p = round((pulse[c] + offs[c]) * step).
    Each pulse raises acc[c, i],
    the envelope at output sample emitted + i, to at least amplitude * Gaussian(i - p), so overlaps combine by
    their maximum as in GETvoc.m (ConvType 1). The amplitude is the acoustic envelope at the pulse time. The offsets
    (stimulation slot / 15) interleave the channels like the implant's electrode pulses. `pulse` is advanced in
    place.
    """
    for c in range(env_hist.shape[0]):
        h = halves[c]
        while True:
            p = int(np.round((pulse[c] + offs[c]) * step))
            if p >= n1:
                break
            if p < env_t0:                                    # rounds to a sample before the stored envelope
                pulse[c] += 1
                continue
            av = env_hist[c, p - env_t0]
            if av > 0 and ok[c]:
                i0 = p - h - emitted
                for t in range(2 * h + 1):
                    v = av * gauss[c, t]
                    if v > acc[c, i0 + t]:
                        acc[c, i0 + t] = v
            pulse[c] += 1


@numba.njit(cache=True)
def _get_emit(acc, m, emitted, w, phi, norm, ok, gain, out):
    """Write m output samples, gain * sum_c norm[c] * acc[c, i] * sin(w[c] * (emitted + i) + phi[c]).

    The carriers run on absolute sample time, so they stay continuous across calls.
    """
    for i in range(m):
        out[i] = 0.0
    for c in range(acc.shape[0]):
        if ok[c]:
            for i in range(m):
                a = acc[c, i]
                if a > 0:
                    out[i] += norm[c] * a * np.sin(w[c] * (emitted + i) + phi[c])
    for i in range(m):
        out[i] *= gain

from .f120_stream import FS, N_CHAN, PULSE_WIDTH_US, default_params

KINDS = {
    "gpyt": "GPyT neural (electric-field spread)",
    "noise": "Noise-band (Shannon 1995)",
    "sine": "Sine / tone (Dorman 1997)",
    "get": "GET pulsatile (Meng et al. 2022)",
}

# Fixed output gain per carrier that brings 65 dB SPL speech (the AzBio reference) to -25 dBFS RMS, matching the
# GPyT vocoder. Measured with `python -m rtci.vocoders_rt --calibrate` at env_lp 300 Hz, GET 100 pps and l = 3.
VOC_CAL = {"noise": 0.1060, "sine": 0.1081, "get": 0.1110}


def f120_bands(par=None):
    """Return the band edges and centers in Hz of the 15 F120 analysis channels (256-point FFT bins at 17.4 kHz)."""
    p = par or default_params()
    bw = p["fs"] / p["nFft"]
    lo = p["startBin"] - 1
    edges = [lo]
    for n in p["nBinLims"]:
        edges.append(edges[-1] + int(n))
    edges = (np.array(edges) - 0.5) * bw
    return edges, np.sqrt(edges[:-1] * edges[1:])


_TABLES = {}
NOISE_LOOP_S = 5.0


def _noise_table(fs, edges, seed):
    """Return the per-band noise carriers as seamless loops.

    White noise is shaped in the frequency domain by a 4th-order Butterworth band-pass magnitude. The filtering is
    circular, so the loop has no seam. Each band is normalized to the RMS of a sine, 1 / sqrt(2). The table is
    computed once; a 5 s loop is not heard as repetition and costs nothing per block.
    """
    key = (fs, edges, seed)
    if key not in _TABLES:
        L = int(NOISE_LOOP_S * fs)
        rng = np.random.default_rng(seed)
        W = np.fft.rfft(rng.standard_normal((len(edges) - 1, L)), axis=1)
        f = np.fft.rfftfreq(L, 1 / fs)
        tab = np.empty((len(edges) - 1, L), np.float32)
        for c, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            f0, bw = np.sqrt(lo * hi), hi - lo
            with np.errstate(divide="ignore"):
                q = (f ** 2 - f0 ** 2) / (np.maximum(f, 1e-9) * bw)          # band-pass -> low-pass transform
            mag = 1 / np.sqrt(1 + q ** 8)                                  # 4th-order Butterworth magnitude
            y = np.fft.irfft(W[c] * mag, L)
            tab[c] = y / np.sqrt(np.mean(y ** 2)) / np.sqrt(2)
        _TABLES[key] = tab
    return _TABLES[key].copy()


class AcousticVocoder:
    """Streaming noise-band, sine or GET vocoder that renders F120 stimulation currents as audio at fs_out."""

    def __init__(self, kind, fs_out, env_lp=300.0, pulse_rate=100.0, get_l=3.0, seed=0, par=None):
        if kind not in ("noise", "sine", "get"):
            raise ValueError(f"unknown acoustic vocoder {kind!r}")
        if not env_lp > 0:
            raise ValueError(f"env_lp must be > 0 Hz, got {env_lp!r}")
        if kind == "get" and not (pulse_rate > 0 and get_l > 0):
            raise ValueError(f"GET needs pulse_rate > 0 and l > 0, got {pulse_rate!r}, {get_l!r}")
        p = par or default_params()
        for k in ("mapT", "mapM", "mapIdr"):              # the map inversion below uses one value for all channels
            if np.ptp(np.asarray(p[k], float)[:N_CHAN + 1]) != 0:
                raise ValueError(f"the acoustic vocoders need a uniform {k} across electrodes")
        self.kind, self.fs = kind, float(fs_out)
        self.T, self.M = float(p["mapT"][0]), float(p["mapM"][0])
        self.IDR = float(p["mapIdr"][0])
        self.edges, self.fc = f120_bands(p)
        nyq = 0.5 * self.fs
        self.ok = self.edges[1:] < 0.98 * nyq                 # bands that fit under the output Nyquist
        # Frame k starts at exactly k * 30 * 18 us. Keeping samples per frame as an exact fraction stops the
        # stimulation timebase from drifting against the output clock.
        self.spf = Fraction(2 * N_CHAN * PULSE_WIDTH_US, 10 ** 6) * Fraction(fs_out).limit_denominator(10 ** 9)
        self.k = 0                                            # stimulation frames consumed
        self.n = 0                                            # envelope samples produced
        self.held = np.zeros(N_CHAN)
        b, a = butter(2, env_lp / nyq)
        self.lp_b, self.lp_a = b, a
        self.lp_zi = np.zeros((N_CHAN, max(len(a), len(b)) - 1))
        self.gain = VOC_CAL[kind]
        self.rng = np.random.default_rng(seed)
        if kind == "noise":
            self.table = _noise_table(self.fs, tuple(self.edges), seed)     # (15, 5 s) band-noise loops
            self.table[~self.ok] = 0
        elif kind == "sine":
            self.phase = self.rng.uniform(0, 2 * np.pi, N_CHAN)   # random start phase per channel and per ear
            self.w = 2 * np.pi * self.fc / self.fs
        else:
            self.rate = float(pulse_rate)
            self.offs = (np.asarray(p["channelOrder"], float)[:N_CHAN] - 1) / N_CHAN   # stimulation slot / 15
            self.phi = self.rng.uniform(0, 2 * np.pi, N_CHAN)
            self.w = 2 * np.pi * self.fc / self.fs
            D = get_l / self.fc * self.fs                     # effective Gaussian duration per channel (samples)
            self.halves_arr = np.ceil(1.5 * D).astype(np.int64)   # truncated at +/-1.5 D, as in GETvoc.m
            self.H = int(self.halves_arr.max())               # output lags the envelope by H samples
            self.gauss = np.zeros((N_CHAN, 2 * self.H + 1))
            self.norm = np.zeros(N_CHAN)
            period = self.fs / self.rate
            for c in range(N_CHAN):
                h = int(self.halves_arr[c])
                tt = np.arange(-h, h + 1)
                self.gauss[c, :tt.size] = np.exp(-np.pi * tt ** 2 / D[c] ** 2)
                # Normalize by the RMS over one period of a steady unit-amplitude, max-combined pulse train.
                t = np.arange(int(np.ceil(period))) + 0.0
                k = np.arange(-int(np.ceil(1.5 * D[c] / period)) - 1, int(np.ceil(1.5 * D[c] / period)) + 2)
                d = t[:, None] - k[None, :] * period
                g = np.where(np.abs(d) <= h, np.exp(-np.pi * d ** 2 / D[c] ** 2), 0.0).max(axis=1)
                self.norm[c] = 1.0 / np.sqrt(np.mean(g ** 2))
            self.acc = np.zeros((N_CHAN, 4 * self.H + 1))     # acc[:, i] = envelope at output sample (emitted + i)
            self.emitted = -self.H                            # the first H output samples are the look-ahead
            self.pulse = np.zeros(N_CHAN, np.int64)           # next pulse index per channel
            self.env_hist = np.zeros((N_CHAN, 0))
            self.env_t0 = 0
        self.latency_s = (self.H / self.fs) if kind == "get" else 0.0

    # ------------------------------------------------------------------------------------------------
    def current_to_amp(self, I):
        """Map stimulation current to linear acoustic amplitude (zero at or below T)."""
        lvl = self.IDR * (I - self.M) / (self.M - self.T)
        return np.where(I > self.T, 10 ** (lvl / 20), 0.0)

    def process(self, I):
        """Render the next (15, n_frames) block of stimulation currents and return the new audio samples."""
        I = np.ascontiguousarray(I, dtype=float) if I.shape[1] else np.zeros((N_CHAN, 0))
        k1 = self.k + I.shape[1]
        n1 = int(-((-k1 * self.spf) // 1))                    # samples with t < k1 * T_frame
        ns = n1 - self.n
        if ns <= 0:
            if I.shape[1]:                                    # frames arrived before the next sample is due
                self.held = self.current_to_amp(I[:, -1])
            self.k = k1
            return np.zeros(0)
        env = np.empty((N_CHAN, ns))
        _env_kernel(I, self.k, self.n, n1, self.spf.numerator, self.spf.denominator, self.held, self.T, self.M,
                    self.IDR, self.lp_b, self.lp_a, self.lp_zi, env)
        n0 = self.n
        self.n, self.k = n1, k1
        out = np.empty(ns)
        if self.kind == "noise":
            _noise_kernel(env, self.table, n0, self.gain, out)
            return out
        if self.kind == "sine":
            _sine_kernel(env, self.phase, self.w, self.ok, self.gain, out)
            return out
        return self.gain * self._get(env, n0, n1)

    def _get(self, env, n0, n1):
        self.env_hist = np.concatenate((self.env_hist, env), axis=1)
        need = n1 + self.H + 1 - self.emitted                 # envelopes of pulses < n1 end before n1 + H
        if need > self.acc.shape[1]:
            self.acc = np.concatenate((self.acc, np.zeros((N_CHAN, need - self.acc.shape[1]))), axis=1)
        _get_kernel(self.env_hist, self.env_t0, self.pulse, self.offs, self.fs / self.rate, n1, self.gauss,
                    self.halves_arr, self.ok, self.acc, self.emitted)
        # No future pulse can reach output samples before n1 - H, so they are final.
        m = n1 - self.H - self.emitted
        out = np.empty(max(m, 0))
        if m > 0:
            _get_emit(self.acc, m, self.emitted, self.w, self.phi, self.norm, self.ok, 1.0, out)
            self.acc = np.concatenate((self.acc[:, m:], np.zeros((N_CHAN, m))), axis=1)
            self.emitted += m
        keep = n1 - self.env_t0                               # every channel is done with pulses before n1
        self.env_hist, self.env_t0 = self.env_hist[:, keep:], self.env_t0 + keep
        return out


# ------------------------------------------------------------------------------------------------ self-test
def _main():
    import argparse
    import sys
    from math import gcd
    from pathlib import Path

    import soundfile as sf
    from scipy.signal import resample_poly, hilbert

    from . import engine as E
    from .f120_stream import GPYT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    x, sr = sf.read(str(GPYT_DIR / "Sounds" / "AzBio_3sent_65dBSPL.wav"), always_2d=True)
    fs = 48000
    g = gcd(int(sr), fs)
    x = resample_poly(x[:, 0], fs // g, int(sr) // g)
    edges, fc = f120_bands()
    print("F120 bands (Hz):", np.round(edges).astype(int).tolist())

    def band_env(y):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b, a_ = butter(3, [lo / (fs / 2), min(hi, 0.98 * fs / 2) / (fs / 2)], btype="band")
            e = np.abs(hilbert(lfilter(b, a_, y)))
            bl, al = butter(2, 30 / (fs / 2))
            out.append(lfilter(bl, al, e)[:: fs // 200])
        return np.array(out)

    ref = band_env(x)
    for kind in ("gpyt", "noise", "sine", "get"):
        eng = E.Engine(fs, 480)
        eng.set_vocoder(kind)
        y = E.process_file(eng, x)
        # The same signal fed in random chunk sizes must give the same samples.
        np.random.seed(0)
        eng2 = E.Engine(fs, 480)
        eng2.set_vocoder(kind)
        rng = np.random.default_rng(3)
        ys, i = [], 0
        while i < x.size:
            nn = int(rng.integers(1, 3000))
            ys.append(eng2.proc.process(eng2.rs.process(x[i:i + nn])))
            i += nn
        y2 = np.concatenate(ys)
        np.random.seed(0)
        eng3 = E.Engine(fs, 480)
        eng3.set_vocoder(kind)
        y3 = eng3.proc.process(eng3.rs.process(x))
        m = min(y2.size, y3.size)
        inv = np.max(np.abs(y2[:m] - y3[:m])) / (np.max(np.abs(y3)) + 1e-12)
        # Lag-compensated per-band correlation with the clean input's temporal envelopes.
        ve = band_env(y)
        best = -1
        for lag in range(0, 30):
            T_ = min(ref.shape[1], ve.shape[1] - lag)
            r = np.nanmean([np.corrcoef(ref[c, :T_], ve[c, lag:lag + T_])[0, 1] for c in range(N_CHAN)])
            best = max(best, r)
        rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)
        cal = 10 ** (-25 / 20) / np.sqrt(np.mean(y ** 2)) if kind != "gpyt" else 1.0
        print(f"{kind:<6} rms {rms:6.1f} dBFS  finite {np.isfinite(y).all()}  chunk-invariance {inv:.1e}  "
              f"band-envelope corr vs clean {best:.3f}" + (f"   -> VOC_CAL x{cal:.4f}" if kind != "gpyt" else ""))
        out_dir = Path.cwd() / "rtci_demo_out"
        out_dir.mkdir(exist_ok=True)
        sf.write(str(out_dir / f"azbio65_voc_{kind}.wav"), y.astype(np.float32), fs)


if __name__ == "__main__":
    _main()
