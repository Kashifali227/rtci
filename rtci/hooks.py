# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
"""
Plug-in points for trying out new processing blocks live in the real-time CI simulator.

The F120 chain (see f120_stream.StreamingF120) has two insertion points.

AudioHook: process(x) -> y
    x and y are 1-D float arrays at 17.4 kHz, and their length can vary from call to call. The hook runs before
    pre-emphasis and the AGC, which is where a front-end denoiser, beamformer or dereverberation stage would sit.
    It may buffer internally, so a single call can return more or fewer samples than it received. Over time it must
    return one output sample per input sample; a fixed latency is fine.

EnvHook: process(env, ctx) -> env
    env holds the (15, n_frames) channel envelopes at 870 frames/s, in GPyT's log2-power domain and after the stock
    noise-reduction gain. ctx is a dict with these keys:

        frame0    stream index of the first frame in this block
        fft       (128, n) complex STFT
        energy    (15, n) AGC-compensated linear amplitudes
        nr_gain   (15, n) linear noise-reduction gain, or None
        env_raw   (15, n) envelopes before noise reduction

    This is where an envelope-domain noise reduction, a dereverberation mask or a contrast enhancement would sit.
    For orientation on the map: envelopes below about 3.9 (log2) produce no current, the M-level is reached at
    about 26 (log2), and one log2 unit is 3.01 dB.

Every hook has a `name`, an `enabled` flag that the GUI toggles live, a `latency_ms` giving the delay it adds, and a
`reset()` method. The latency keeps the quiet-room gate and the SSD delay aligned with the CI output. `reset()` is
called whenever the hook is switched on, so a stateful block never resumes with stale buffers.

A hook that raises, returns the wrong shape or returns non-finite values is switched off automatically, and its
input is passed on unchanged. An EnvHook must return the same shape as its input. The reason is shown in the GUI's
error line (StreamingF120.hook_errors), so a faulty block can neither stop the audio nor corrupt the chain.

To add a block without editing the package, register a factory before the engine is created:

    import rtci.hooks as H

    class Louder(H.EnvHook):
        name = "Louder by 3 dB"
        def process(self, env, ctx):
            return env + 1.0            # one log2 unit is 3 dB

    H.register_hook(Louder)             # any callable that returns a new hook instance
    # then start the GUI from the same script:  import rtci.app; rtci.app.main()

Each ear gets its own instance, because hooks hold state. The GUI lists registered hooks automatically.
"""
import numpy as np


class AudioHook:
    name = "audio hook"
    enabled = False
    latency_ms = 0.0

    def process(self, x):
        return x

    def reset(self):
        """Clear internal state. Called when the hook is switched on."""


class EnvHook:
    name = "envelope hook"
    enabled = False
    latency_ms = 0.0

    def process(self, env, ctx):
        return env

    def reset(self):
        """Clear internal state. Called when the hook is switched on."""


# ------------------------------------------------------------------------------------------- examples
class WienerNR(AudioHook):
    """Single-microphone STFT Wiener filter, as a stand-in for a learned denoiser.

    Uses a decision-directed a priori SNR and a minimum-tracking noise PSD estimate. The 256-point sqrt-Hann STFT
    with hop 128 at 17.4 kHz adds 128 samples (7.4 ms) of latency.
    """
    name = "Wiener NR (audio-domain example)"
    latency_ms = 128 / 17.4                                   # overlap-add delay

    def __init__(self, floor_db=-15.0):
        self.n, self.h = 256, 128
        self.win = np.sqrt(np.hanning(self.n + 1)[:-1])            # periodic, so win^2 sums to 1 at 50% overlap
        self.floor = 10 ** (floor_db / 20)
        self.reset()

    def reset(self):
        self.inbuf = np.zeros(self.n - self.h)
        self.ola = np.zeros(self.n)
        self.noise = None
        self.prev = None

    def process(self, x):
        buf = np.concatenate((self.inbuf, x))
        out = []
        pos = 0
        while pos + self.n <= buf.size:
            X = np.fft.rfft(buf[pos:pos + self.n] * self.win)
            P = X.real ** 2 + X.imag ** 2 + 1e-20
            if self.noise is None:
                self.noise, self.prev = P.copy(), P.copy()
            self.noise = np.where(P < self.noise, 0.8 * self.noise + 0.2 * P, self.noise * 1.005)
            post = P / self.noise
            xi = 0.98 * self.prev / self.noise + 0.02 * np.maximum(post - 1, 0)
            G = np.maximum(xi / (1 + xi), self.floor)
            S = G * X
            self.prev = S.real ** 2 + S.imag ** 2
            self.ola += np.fft.irfft(S, self.n) * self.win
            out.append(self.ola[:self.h].copy())
            self.ola = np.concatenate((self.ola[self.h:], np.zeros(self.h)))
            pos += self.h
        self.inbuf = buf[pos:]
        return np.concatenate(out) if out else np.zeros(0)


class SpectralContrast(EnvHook):
    """Expand the across-channel envelope contrast in each frame as mean + k * (env - mean), in the log domain."""
    name = "Spectral contrast x1.5 (envelope example)"

    def __init__(self, k=1.5):
        self.k = k

    def process(self, env, ctx):
        mu = env.mean(axis=0, keepdims=True)
        return np.maximum(mu + self.k * (env - mu), 0)


class ElectrodeDropout(EnvHook):
    """Silence selected analysis channels, for example to simulate deactivated electrodes."""

    def __init__(self, channels=(6, 7, 8)):
        self.channels = list(channels)
        self.name = f"Drop channels {', '.join(str(c + 1) for c in self.channels)} (envelope example)"

    def process(self, env, ctx):
        env = env.copy()
        env[self.channels] = 0
        return env


_REGISTERED = []


def register_hook(factory):
    """Register a user hook.

    `factory` is a callable, usually the class itself, that returns a new AudioHook or EnvHook instance. It is
    called once per ear. Register hooks before creating the Engine or starting the GUI.
    """
    _REGISTERED.append(factory)
    return factory


def make_hooks():
    """Return new hook instances for one ear.

    Hooks hold state, so instances must never be shared between ears or engines.
    """
    audio, env = [WienerNR()], [SpectralContrast(), ElectrodeDropout()]
    for f in _REGISTERED:
        h = f()
        if isinstance(h, AudioHook):
            audio.append(h)
        elif isinstance(h, EnvHook):
            env.append(h)
        else:
            raise TypeError(f"{f!r} returned {type(h).__name__}, expected an AudioHook or EnvHook instance")
    return audio, env
