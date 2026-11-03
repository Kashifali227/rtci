# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
"""Streaming rational polyphase FIR resampler with an exact ratio and an even output per call.

soxr's stream resampler works internally in large blocks and returns 0, 0, 540, 0, 0, 540 ... samples per
10 ms call. That makes the CI chain bursty, with CPU spikes and 20 ms of extra buffering. This resampler emits
each output sample as soon as its last input sample has arrived. 48 kHz to 17.4 kHz is up 29, down 80, and the
group delay is about 2.0 ms at 44.1, 48 and 96 kHz.
"""
from . import _svml  # noqa: F401,I001  (must be imported before numba, see _svml.py)
from fractions import Fraction

import numba
import numpy as np
from scipy.signal import firwin, kaiserord


@numba.njit(cache=True)
def _poly_kernel(H, buf, g0, j0, j1, up, down, L, out):
    """Compute y[j] = sum_m H[ph, m] * x[base - m] with base = floor(j*down/up) and ph = j*down mod up.

    x[n] is read from buf[n - g0]."""
    for j in range(j0, j1):
        base = j * down // up
        ph = j * down % up
        b = base - g0
        acc = 0.0
        for m in range(L):
            acc += H[ph, m] * buf[b - m]
        out[j - j0] = acc


class PolyResampler:
    """Resample a stream from fs_in to fs_out with a Kaiser-windowed polyphase FIR filter."""

    def __init__(self, fs_in, fs_out, atten_db=70.0, pass_hz=None, stop_hz=None):
        r = Fraction(int(fs_out), int(fs_in))
        self.up, self.down = r.numerator, r.denominator
        nyq = min(fs_in, fs_out) / 2
        stop = stop_hz or nyq
        pas = pass_hz or 0.875 * nyq
        fs_up = fs_in * self.up
        n, beta = kaiserord(atten_db, (stop - pas) / (fs_up / 2))
        self.L = L = int(np.ceil(n / self.up))
        h = firwin(L * self.up, (pas + stop) / 2, fs=fs_up, window=("kaiser", beta)) * self.up
        self.H = np.ascontiguousarray(h.reshape(L, self.up).T)       # H[phase, m] = h[phase + up*m]
        self.hist = np.zeros(L - 1)
        self.n_in = 0                                                 # inputs consumed so far
        self.j = 0                                                    # next output index
        self.delay_s = (L * self.up - 1) / 2 / fs_up                  # group delay in seconds

    def process(self, x):
        x = np.asarray(x, float)
        buf = np.concatenate((self.hist, x))
        g0 = self.n_in - (self.L - 1)                                 # global input index of buf[0]
        total = self.n_in + x.size
        j1 = -((-total * self.up) // self.down)                       # outputs j with floor(j*down/up) < total
        if j1 <= self.j:
            self.hist, self.n_in = buf[-(self.L - 1):], total
            return np.zeros(0)
        y = np.empty(j1 - self.j)
        _poly_kernel(self.H, buf, g0, self.j, j1, self.up, self.down, self.L, y)
        self.hist, self.n_in, self.j = buf[-(self.L - 1):], total, j1
        return y
