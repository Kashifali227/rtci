# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
"""
Quiet-room gate that mutes the microphone and preamp noise floor, so a silent room sounds silent.

In a quiet room the UR22 inputs carry about 40 dB SPL of broadband self-noise. The CI's dual-loop AGC gives up to
about 42 dB of gain to soft input, which is what makes soft sounds audible on a real processor. As a result the
floor ends up above T-level in about half of all channel-frames (in channel 15, 4.2 to 8 kHz, all of the time) and
is heard as a constant hiss. Low-frequency rumble is not the issue, because the F120 pre-emphasis removes it
(-27 dB at 100 Hz).

The gate only makes its decision from the microphone signal. The engine applies the resulting gain to the output of
each ear, so the CI chain still processes the unmodified microphone signal and its AGC behaves as it would on a real
processor. Gating the input instead makes the AGC pump the hiss up between openings. Since the CI output lags the
microphone by the chain latency, the engine delays the gain by (latency - 5 ms). The gate therefore opens 5 ms
before a sound reaches the ear, which avoids clipped onsets without adding delay. The SSD normal-hearing ear uses the
same look-ahead when delay matching is on. The undelayed natural paths (the natural and mix listening modes, and the
SSD ear without delay matching) have no look-ahead, so the first 2 to 5 ms of a sound that starts from silence are
attenuated by the detector and the 1 ms attack.

Measured in a quiet room through a Steinberg UR22mkII (0 dB mic gain, 300 Hz high-pass detector band, 10 ms):

    room floor          median 24 dB SPL, 99th percentile 39, clicks up to about 42-49
    speech at 65 dB SPL active frames: 10th percentile about 49, median about 60

The default threshold is therefore the tracked floor plus 18 dB, about 40 dB SPL in that room. It is capped at
50 dB SPL so that the gate stays open in a genuinely noisy place, since it must not act as an extra noise reduction.
Very soft speech (about 45 dB SPL) overlaps the room's own transients and is partly gated. Lower the margin if that
matters.

  level   300 Hz high-passed power, one-pole smoothed over about 10 ms, in dB on the engine's SPL scale.
  floor   Asymmetric tracker on the level. It falls at about 20 dB/s. It rises at about 2 dB/s only while the
          level is within 6 dB of the floor, i.e. while the floor itself is drifting, and at only about 0.1 dB/s
          while a real sound is present. This keeps continuous sound below the 50 dB cap, such as music at 45 to
          55 dB SPL, from being learned as background and muted. A steady background that really does get louder
          (a fan switched on) stays audible until it is learned at 0.1 dB/s, which is acceptable because it is a
          real sound.
  gain    0 dB above the threshold. Below it, a 6 dB soft knee down to -max_att_db (60 dB).
  timing  About 1 ms attack, 120 ms hold after the last sub-block above the threshold, about 50 ms release.
"""
from . import _svml  # noqa: F401,I001  (must precede numba: enables SVML vector math, see _svml.py)
import numba
import numpy as np
from scipy.signal import butter


@numba.njit(cache=True)
def _gate_kernel(x, coef, st, prm, out):
    """Run the gate sample by sample and write the linear gain to `out`.

    Each sample passes a 300 Hz high-pass biquad and accumulates into a sub-block power. Once per sub-block the
    smoothed level, floor tracker, threshold and hold update the target gain in dB. The gain then follows the target
    with per-sample one-pole attack and release smoothing.

    st  = [z1, z2, pacc, cnt, p_s, floor (nan = unset), hold, g_db, level, thr, target, margin]
    prm = [sub, a_smooth, off, down, up, cap, hold_n, max_att, knee, a_att_smp, a_rel_smp, rise_win, up_slow]"""
    b0, b1, b2, a1, a2 = coef[0], coef[1], coef[2], coef[3], coef[4]
    sub, a_s, off, down, up, cap = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5]
    hold_n, max_att, knee, a_att, a_rel = prm[6], prm[7], prm[8], prm[9], prm[10]
    rise_win, up_slow = prm[11], prm[12]
    margin = st[11]
    for i in range(x.shape[0]):
        v = x[i]
        y = b0 * v + st[0]                                    # transposed direct form II
        st[0] = b1 * v - a1 * y + st[1]
        st[1] = b2 * v - a2 * y
        st[2] += y * y
        st[3] += 1.0
        if st[3] >= sub:
            p = st[2] / st[3]
            st[2] = 0.0
            st[3] = 0.0
            st[4] = p if st[4] < 0 else a_s * st[4] + (1 - a_s) * p
            L = 10.0 * np.log10(st[4] + 1e-20) + off
            if np.isnan(st[5]):
                st[5] = L
            elif L < st[5]:
                st[5] -= down
            elif L < st[5] + rise_win:                          # close to the floor, so the floor itself is drifting
                st[5] += up
            else:                                               # real sound such as speech or music
                st[5] += up_slow
            thr = min(st[5] + margin, cap)
            if L >= thr:
                st[6] = hold_n
            elif st[6] > 0:
                st[6] -= 1
            if st[6] > 0:
                st[10] = 0.0
            else:
                st[10] = -max_att * min(max((thr - L) / knee, 0.0), 1.0)
            st[8], st[9] = L, thr
        a = a_att if st[10] > st[7] else a_rel
        st[7] = a * st[7] + (1 - a) * st[10]
        out[i] = 10.0 ** (st[7] / 20.0)


class SilenceGate:
    """Level-tracking gate that computes a per-sample output gain from the calibrated microphone signal."""

    def __init__(self, fs, spl_offset, margin_db=18.0, cap_spl=50.0, max_att_db=60.0, knee_db=6.0,
                 hold_ms=120.0, attack_ms=1.0, release_ms=50.0, smooth_ms=10.0):
        self.fs, self.off = float(fs), spl_offset
        self.margin_db, self.cap_spl, self.max_att_db, self.knee_db = margin_db, cap_spl, max_att_db, knee_db
        self.sub = max(16, int(round(0.0015 * fs)))           # detector sub-block (~1.5 ms)
        dt = self.sub / self.fs
        self.a_att = np.exp(-dt / (attack_ms * 1e-3))
        self.a_rel = np.exp(-dt / (release_ms * 1e-3))
        self.a_smooth = np.exp(-dt / (smooth_ms * 1e-3))
        self.hold_n = int(round(hold_ms * 1e-3 / dt))
        self.down, self.up = 0.03, 0.003                      # floor tracker steps (dB per sub-block)
        b, a = butter(2, 300, "high", fs=fs)
        self.coef = np.array([b[0], b[1], b[2], a[1], a[2]], float)
        smp = 1.0 / self.fs                                   # the kernel smooths attack and release per sample
        self.prm = np.array([self.sub, self.a_smooth, self.off, self.down, self.up, self.cap_spl, self.hold_n,
                             self.max_att_db, self.knee_db, np.exp(-smp / (attack_ms * 1e-3)),
                             np.exp(-smp / (release_ms * 1e-3)),
                             6.0, self.up / 20], float)                # rise window 6 dB; 0.1 dB/s above it
        self.st = np.array([0, 0, 0, 0, -1.0, np.nan, 0, 0, -np.inf, -np.inf, 0, margin_db], float)
        self.floor = None
        self.g = 0.0                                          # gain, dB
        self.enabled = True
        self.level, self.thresh, self.open = -np.inf, -np.inf, True

    def gains(self, x):
        """Return the per-sample linear gain for a calibrated microphone block of any length."""
        n = x.size
        if not self.enabled:
            self.g, self.open = 0.0, True
            self.st[7] = 0.0
            return np.ones(n)
        out = np.empty(n)
        self.st[11] = self.margin_db
        _gate_kernel(np.ascontiguousarray(x, dtype=np.float64), self.coef, self.st, self.prm, out)
        self.level, self.thresh, self.g = self.st[8], self.st[9], self.st[7]
        self.floor = None if np.isnan(self.st[5]) else self.st[5]
        self.open = self.g > -3.0
        return out


class DelayLine:
    """Integer-sample delay for a gain or audio stream.

    The delay can change between calls without dropping or zeroing what is already buffered. When it grows by k
    samples, k samples are inserted. For a gain stream (hold=True) the oldest buffered value is repeated, so a closed
    gate stays closed instead of briefly opening. For audio (hold=False) `fill` is inserted, which is silence by
    default. `fill` is also used before anything has been buffered.
    """

    def __init__(self, fill=1.0, hold=True):
        self.fill, self.hold = fill, hold
        self.buf = np.zeros(0)

    def process(self, x, d):
        d = max(0, int(d))
        if self.buf.size != d:
            pad = self.buf[0] if (self.hold and self.buf.size) else self.fill
            self.buf = (np.concatenate((np.full(d - self.buf.size, pad), self.buf)) if d > self.buf.size
                        else self.buf[self.buf.size - d:])
        if d == 0:
            return x
        z = np.concatenate((self.buf, x))
        self.buf = z[-d:]
        return z[:x.size]
