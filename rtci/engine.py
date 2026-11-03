# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
"""
Real-time wrapper around the streaming F120 and vocoder chain, with per-ear hearing configurations.

Each ear (EarChain) applies the input gain to the microphone signal at the device rate, resamples it to 17.4 kHz
with a polyphase stream resampler and runs it through a CIProcessor (F120 strategy plus vocoder, output at the
device rate). An output FIFO absorbs the 5 ms block granularity of the vocoders.

Hearing configurations (Engine.config):

    bilateral   Two independent processors, each with its own AGC, noise reduction and hooks, like two unlinked
                implants. The left mic feeds the left CI and the right mic the right CI. With routing "one", a
                single mic feeds one processor that is heard in both ears.
    unilateral  One CI on `ci_side`, fed by that side's mic. The other ear is silent.
    ssd         Single-sided deafness. The CI on `ci_side` is fed by that side's mic, and the other ear hears its
                own mic without vocoding, as the normal-hearing ear. That ear is level-matched to the CI ear
                (NAT_CAL) and can be delayed to match the CI ear's processing latency.

Each ear's output then goes through the listening mode (CI, natural or mix), the volume and a safety limiter.

Levels follow the GPyT convention, in which -43.6 dBFS RMS corresponds to 65 dB SPL (the AzBio_3sent_65dBSPL
reference file), so SPL = dBFS_rms + SPL_OFFSET. The input gain is the microphone calibration, and `calibrate()`
sets it from live speech.
"""
from __future__ import annotations
from . import _svml  # noqa: F401,I001  (must precede numba: enables SVML vector math, see _svml.py)

import collections
import threading
import time

import numba
import numpy as np
from .f120_stream import FS, CIProcessor
from .hooks import make_hooks
from .resampler import PolyResampler
from .gate import SilenceGate, DelayLine

SPL_OFFSET = 108.6           # dB SPL at 0 dBFS RMS (the GPyT reference file has 65 dB SPL at -43.6 dBFS RMS)
REF_ACTIVE_DBFS = None       # active speech level of the reference file, computed on first use by calibrate()
# Gain of the natural (non-vocoded) path. Every vocoder outputs 65 dB SPL speech at -25 dBFS RMS, while the
# calibrated mic signal sits at 65 - SPL_OFFSET = -43.6 dBFS. A gain of +18.6 dB makes the natural path and the CI
# equally loud at the same SPL.
NAT_CAL = 10 ** ((-25.0 - (65.0 - SPL_OFFSET)) / 20)
SIDES = ("left", "right")
IN_GAIN_RANGE_DB = (-30.0, 60.0)   # mic calibration range, shared by the GUI slider and calibrate()


def active_level_dbfs(x, fs):
    """Return the active speech level, the RMS over the loudest 50% of 20 ms frames, so pauses do not count."""
    n = int(0.02 * fs)
    m = x.size // n
    if m < 5:
        return -120.0
    e = np.mean(x[:m * n].reshape(m, n) ** 2, axis=1)
    e = np.sort(e)[m // 2:]
    return 10 * np.log10(np.mean(e) + 1e-20)


def _ref_active_dbfs():
    global REF_ACTIVE_DBFS
    if REF_ACTIVE_DBFS is None:
        import soundfile as sf
        from .f120_stream import GPYT_DIR
        x, sr = sf.read(str(GPYT_DIR / "Sounds" / "AzBio_3sent_65dBSPL.wav"), always_2d=True)
        REF_ACTIVE_DBFS = active_level_dbfs(x[:, 0], sr)
    return REF_ACTIVE_DBFS


@numba.njit(cache=True)
def _limit_kernel(y, g0, rel, ceiling, out):
    """Limit one block and return the new gain.

    Attack is instant, with the gain ramped linearly across the block. The gain recovers by the factor `rel` per
    block, and the result is hard-clipped at the ceiling.
    """
    n = y.shape[0]
    pk = 0.0
    for i in range(n):
        a = abs(y[i])
        if a > pk:
            pk = a
    target = min(1.0, ceiling / pk) if pk > 0 else 1.0
    new = target if target < g0 else target + (g0 - target) * rel
    for i in range(n):
        g = g0 + (new - g0) * i / n
        v = y[i] * g
        out[i] = min(max(v, -ceiling), ceiling)
    return new


class EarChain:
    """One implant: resampler, F120 strategy, vocoder and output FIFO, with its own stateful hook instances."""

    def __init__(self, eng, side):
        self.eng, self.side = eng, side
        self.proc = None
        self.audio_hooks, self.env_hooks = make_hooks()
        self.used_last = False
        self.gate_dl = DelayLine(1.0)                       # quiet-room gate gain, delayed to this ear's latency
        self.reset()

    def reset(self):
        old = self.proc
        kind, opts = (old.kind, old.voc_opts) if old is not None else ("gpyt", {})
        # A separate seed per ear, so two unlinked implants do not share identical random carriers.
        self.proc = CIProcessor(fs_out=self.eng.fs, timing="real", vocoder=kind, voc_opts=opts,
                                seed=SIDES.index(self.side))
        st = self.proc.strategy
        st.audio_hooks, st.env_hooks = self.audio_hooks, self.env_hooks
        if old is not None:
            so = old.strategy
            st.nr_enabled, st.steering, st.carrier_mode = so.nr_enabled, so.steering, so.carrier_mode
            st.agc_enabled = so.agc_enabled
        st.on_frames = lambda d: self.eng.env_cols.append(d["env"].max(axis=1)) if self.eng.display is self else None
        self.proc.on_electrodogram = (lambda em: self.eng.elec_cols.append(em)
                                      if self.eng.display is self else None)
        self.rs = PolyResampler(self.eng.fs, FS)
        self.n_in = self.n_out = 0                          # samples fed in, and samples taken out or dropped
        self.fifo = np.zeros(0)
        self.flush()

    def flush(self):
        self.n_out += self.fifo.size                        # dropped samples count as taken out
        self.fifo = np.zeros(0)
        self.primed = False

    def latency_ms(self):
        """Return the exact latency in ms from the mic to this ear's CI output, excluding the sound card.

        Every sample that enters the ear leaves it after the chain's fixed delay (ALGO_LATENCY_MS, which includes
        the vocoder's own buffering) plus the number of samples by which the input has run ahead of the output
        (n_in - n_out). That difference covers the FIFO cushion and every dropout and re-prime, so the result tracks
        the block size, the prefill self-tuning and vocoder switches. Enabled audio hooks add their latency_ms.
        """
        hooks = sum(getattr(h, "latency_ms", 0.0) for h in self.audio_hooks if h.enabled)
        return ALGO_LATENCY_MS[self.proc.kind] + hooks + 1000 * (self.n_in - self.n_out) / self.eng.fs

    def ci_block(self, x):
        """Return the CI-vocoded block for a calibrated mic block at the device rate, with the same length."""
        e, n = self.eng, x.size
        if not self.used_last:                              # unused since the last block, so the output is stale
            self.flush()
        self.used_last = True
        self.n_in += n
        try:
            y = self.proc.process(self.rs.process(x))
        except Exception as ex:                             # report it, but keep the audio thread running
            e.errors.append(f"{self.side}: {ex!r}")
            y = np.zeros(0)
        he = self.proc.strategy.hook_errors
        while he:
            e.errors.append(f"{self.side}: {he.popleft()}")
        if y.size:
            self.fifo = np.concatenate((self.fifo, y))
        if not self.primed and self.fifo.size >= e.prefill + n:
            self.primed = True
        if self.primed and self.fifo.size >= n:
            ci = self.fifo[:n]
            self.fifo = self.fifo[n:]
            self.n_out += n
        else:
            if self.primed:
                e.underruns += 1
                e._underrun_now = True                      # the engine re-primes all active ears together
                # The output is burstier than the prefill covers, so grow the cushion by 2 ms (up to 30 ms).
                if e._prefill_fixed is None:
                    e.prefill = min(e.prefill + int(0.002 * e.fs), int(0.03 * e.fs))
                    self.primed = False
            ci = np.zeros(n)
        if self.fifo.size > e.prefill + 2 * n:              # bound latency after a hiccup
            keep = e.prefill + n
            self.n_out += self.fifo.size - keep
            self.fifo = self.fifo[-keep:]
        return ci


class Engine:
    """Duplex audio engine that runs one EarChain per ear in the selected hearing configuration."""

    def __init__(self, fs=48000, blocksize=480, prefill=None):
        self.fs = fs
        self.blocksize = blocksize
        self.lock = threading.Lock()
        # Live controls, read once per block.
        self.mode = "ci"                  # "ci", "natural" or "mix", applied to the CI ears
        self.mix = 0.5                    # CI share in "mix" mode
        self.in_gain_db = 0.0
        self.volume_db = 0.0
        self.config = "bilateral"         # "bilateral", "unilateral" or "ssd"
        self.ci_side = "left"             # CI ear for unilateral and ssd
        self.routing = "1L2R"             # "1L2R" (input 1 is the left mic), "1R2L", or "one" mic for both sides
        self.mic_channel = 0              # input index for routing "one", or -1 to average all inputs
        self.nh_gain_db = 0.0             # SSD normal-hearing ear level in dB, relative to level-matched
        self.nh_delay_ms = 0.0            # SSD normal-hearing ear delay; 0 is an undelayed acoustic ear
        self.gate_enabled = False         # quiet-room gate (gate.py); the GUI and CLI enable it by default
        self.gate_margin_db = 18.0
        self.gate_lead_ms = 5.0           # how long before a sound reaches the CI ear the gate opens
        self.ch_dbfs = None               # per-input level meter in dBFS, peak hold decaying 1.5 dB per block
        # Telemetry.
        self.level_spl = -np.inf
        self.cpu = 0.0
        self.cpu_peak = 0.0
        self.underruns = 0
        self.xruns = 0
        self.errors = collections.deque(maxlen=5)
        self._underrun_now = False
        self._cpu_acc = [0.0, 0.0]        # audio-thread CPU seconds and audio seconds since the last cpu update
        self.elec_cols = collections.deque(maxlen=4000)   # per-callback max anodic amplitude per electrode
        self.env_cols = collections.deque(maxlen=4000)
        self.stream = None
        self._cal = None
        self._prefill_fixed = prefill
        self.prefill = prefill if prefill is not None else self.default_prefill(fs, blocksize)
        self._ur_seen, self._clean = 0, 0
        self.display = None
        self.chains = {}
        self.reset()

    @staticmethod
    def default_prefill(fs, blocksize):
        """Return the FIFO cushion in samples: 12.5 ms minus one block, but at least 5 ms.

        The vocoders emit audio in 5 ms segments. A smaller cushion at large block sizes caused a dropout at every
        start at 48 kHz with 512-sample blocks and at 44.1 kHz with 480-sample blocks.
        """
        return int(round(max(0.005 * fs, (600 - blocksize * 48000 / fs) * fs / 48000)))

    def set_rate(self, fs, blocksize=None):
        """Move the chains to a new device rate or block size, rebuilding the resamplers and vocoder timing."""
        self.fs = int(fs)
        self.blocksize = blocksize or self.blocksize
        if self._prefill_fixed is None:
            self.prefill = self.default_prefill(self.fs, self.blocksize)
        self.reset()

    # ------------------------------------------------------------------------------------------- chains
    def reset(self):
        with self.lock:
            if not self.chains:
                self.chains = {s: EarChain(self, s) for s in SIDES}
            else:
                for c in self.chains.values():
                    c.reset()
            self.display = self.chains["left"]
            self.lim_g = {s: 1.0 for s in SIDES}
            self.dly = {s: DelayLine(0.0, hold=False) for s in SIDES}      # SSD normal-hearing ear audio
            self.dly_g = {s: DelayLine(1.0) for s in SIDES}                # and its quiet-room gate gain
            self.gates = {s: SilenceGate(self.fs, SPL_OFFSET, margin_db=self.gate_margin_db) for s in SIDES}

    @property
    def proc(self):                    # the left (primary) chain, for the single-ear API of the offline tools
        return self.chains["left"].proc

    @property
    def rs(self):
        return self.chains["left"].rs

    @property
    def fifo(self):
        return self.display.fifo

    @property
    def primed(self):
        return self.chains["left"].primed

    @property
    def strategy(self):
        return self.proc.strategy

    @property
    def strategies(self):
        return [c.proc.strategy for c in self.chains.values()]

    def set_vocoder(self, kind, **opts):
        """Switch every ear to vocoder `kind` ('gpyt', 'noise', 'sine' or 'get') while running.

        The strategies keep running. The output FIFOs are flushed and re-primed, because the vocoders have
        different latencies.
        """
        built = {s: c.proc.build_vocoder(kind, **opts) for s, c in self.chains.items()}   # built outside the lock
        with self.lock:
            for s, c in self.chains.items():
                c.proc.install_vocoder(kind, opts, built[s])
                c.flush()

    def set_hook(self, i, enabled):
        """Switch hook number i on or off on every ear, counting audio hooks first and then envelope hooks.

        Switching a hook on resets it, so a stateful block never resumes with stale buffers.
        """
        with self.lock:
            for c in self.chains.values():
                h = (c.audio_hooks + c.env_hooks)[i]
                if enabled and not h.enabled:
                    h.reset()
                h.enabled = enabled

    # ------------------------------------------------------------------------------------------- processing
    def _level(self, x):
        r = np.sqrt(np.mean(x ** 2)) + 1e-12
        v = 20 * np.log10(r) + SPL_OFFSET
        self.level_spl = 0.8 * self.level_spl + 0.2 * v if np.isfinite(self.level_spl) else v

    def _ear(self, chain, x, g_now):
        """Return the CI ear's output for the calibrated mic block x in the current listening mode.

        g_now is the per-sample quiet-room gate gain for this mic, aligned with the mic signal. The CI part receives
        it delayed by the ear's latency minus the gate lead.
        """
        mode = self.mode
        if mode == "natural":
            chain.used_last = False
            return x * NAT_CAL * g_now
        ci = chain.ci_block(x)
        if self.gate_enabled:
            d = (chain.latency_ms() - self.gate_lead_ms) * 1e-3 * self.fs
            ci = ci * chain.gate_dl.process(g_now, d)
        return ci if mode == "ci" else self.mix * ci + (1 - self.mix) * x * NAT_CAL * g_now

    def _delay(self, side, x, ms):
        """Delay the SSD normal-hearing ear by `ms`.

        A change of delay inserts or drops only the difference and keeps the buffered audio. Changes under 0.5 ms
        are ignored, so the automatic delay matching does not chatter.
        """
        dl = self.dly[side]
        d = int(round(ms * 1e-3 * self.fs))
        if d > 0 and abs(d - dl.buf.size) < 0.0005 * self.fs:
            d = dl.buf.size
        return dl.process(x, d)

    def _gate_gains(self, mics):
        """Return the quiet-room gate gain per side, using one gate per mic and evaluating a shared mic once."""
        n = next(iter(mics.values())).size
        if not self.gate_enabled:
            return {s: np.ones(n) for s in SIDES}
        for gt in self.gates.values():
            gt.margin_db = self.gate_margin_db
        gl = self.gates["left"].gains(mics["left"])
        gr = gl if mics["right"] is mics["left"] else self.gates["right"].gains(mics["right"])
        return {"left": gl, "right": gr}

    def process_ears(self, mics):
        """Process a dict of calibrated mic blocks per side and return the output block per side, before volume."""
        n = next(iter(mics.values())).size
        gg = self._gate_gains(mics)
        out = {}
        with self.lock:
            cfg = self.config
            if cfg == "bilateral":
                if self.routing == "one":
                    self.display = self.chains["left"]
                    self.chains["right"].used_last = False
                    y = self._ear(self.chains["left"], mics["left"], gg["left"])
                    out = {"left": y, "right": y}
                else:
                    self.display = self.chains["left"]
                    out = {s: self._ear(self.chains[s], mics[s], gg[s]) for s in SIDES}
                    L, R = self.chains["left"], self.chains["right"]
                    if L.primed and R.primed and L.fifo.size != R.fifo.size:
                        # Both ears run the same vocoder timing, so equal FIFO sizes mean equal delay. A difference
                        # means one ear lost output through an error or a dropout, so trim the fuller ear back into
                        # alignment.
                        late = L if L.fifo.size > R.fifo.size else R
                        k = abs(L.fifo.size - R.fifo.size)
                        late.fifo, late.n_out = late.fifo[k:], late.n_out + k
            else:
                ci, other = self.ci_side, ("right" if self.ci_side == "left" else "left")
                self.display = self.chains[ci]
                self.chains[other].used_last = False
                out[ci] = self._ear(self.chains[ci], mics[ci], gg[ci])
                if cfg == "unilateral":
                    out[other] = np.zeros(n)
                else:                                        # ssd: the normal-hearing ear is not vocoded
                    nh = self._delay(other, mics[other] * NAT_CAL * 10 ** (self.nh_gain_db / 20), self.nh_delay_ms)
                    g = gg[other]
                    if self.gate_enabled:                    # a delayed ear lets the gate open gate_lead_ms early
                        d = self.dly[other].buf.size - self.gate_lead_ms * 1e-3 * self.fs
                        g = self.dly_g[other].process(g, d)
                    out[other] = nh * g
            if self._underrun_now:                           # an ear ran dry; re-prime all active ears from equal
                self._underrun_now = False                   # FIFO sizes so they stay time-aligned
                active = [c for c in self.chains.values() if c.used_last]
                if active:
                    m = min(c.fifo.size for c in active)
                    for c in active:
                        c.n_out += c.fifo.size - m
                        c.fifo = c.fifo[c.fifo.size - m:]
                        c.primed = False
        return out

    def _finish(self, side, y):
        return self._limit(side, y * 10 ** (self.volume_db / 20))

    def process(self, mic):
        """Process one mic block through the left (primary) chain. Used by the offline tools and benchmarks."""
        t0 = (time.perf_counter(), time.thread_time())
        x = mic.astype(np.float64) * 10 ** (self.in_gain_db / 20)
        self._level(x)
        cal = self._cal                                       # calibrate() may reset it from another thread
        if cal is not None:
            cal.append(x.copy())
        with self.lock:
            g = self._gate_gains({"left": x, "right": x})["left"]
            y = self._ear(self.chains["left"], x, g)
        out = self._finish("left", y)
        self._load(t0, mic.size)
        return out

    def process_multi(self, indata):
        """Process an (n, n_in) device block and return (n, 2) left and right output for the hearing configuration."""
        t0 = (time.perf_counter(), time.thread_time())
        g = 10 ** (self.in_gain_db / 20)
        x = indata.astype(np.float64) * g
        nin = x.shape[1]
        if self.routing == "one" or nin == 1:
            if self.mic_channel < 0:
                m = x.mean(axis=1)
            else:
                m = x[:, min(self.mic_channel, nin - 1)]
            mics = {"left": m, "right": m}
        else:
            iL, iR = (0, 1) if self.routing == "1L2R" else (1, 0)
            mics = {"left": x[:, iL], "right": x[:, iR]}
        disp = self.ci_side if self.config != "bilateral" else "left"
        self._level(mics[disp])
        cal = self._cal                                       # calibrate() may reset it from another thread
        if cal is not None:
            cal.append(mics[disp].copy())
        ears = self.process_ears(mics)
        out = np.stack([self._finish(s, ears[s]) for s in SIDES], axis=1)
        self._load(t0, indata.shape[0])
        return out

    def _recover_prefill(self, n):
        """Shrink a FIFO cushion grown by dropouts by 1 ms per 5 s without dropouts, down to the default."""
        if self._prefill_fixed is not None:
            return
        if self.underruns != self._ur_seen:
            self._ur_seen, self._clean = self.underruns, 0
            return
        self._clean += n
        base = self.default_prefill(self.fs, self.blocksize)
        if self._clean >= 5 * self.fs and self.prefill > base:
            self.prefill = max(base, self.prefill - int(0.001 * self.fs))
            self._clean = 0

    def _load(self, t0, n):
        self._recover_prefill(n)
        dur = n / self.fs
        # cpu is the audio thread's compute time per second of audio. Thread CPU time has a coarse clock on Windows
        # (15.6 ms ticks, longer than a block), so it is only read as a ratio of sums over windows of at least 1 s.
        # wall is the elapsed time per block from the high-resolution clock, including time spent waiting for the
        # interpreter lock during GUI redraws. Dropouts happen when it exceeds the device buffer. cpu_peak is the
        # peak wall load.
        wall = (time.perf_counter() - t0[0]) / dur
        acc = self._cpu_acc
        acc[0] += time.thread_time() - t0[1]
        acc[1] += dur
        if acc[1] >= 1.0:
            self.cpu = acc[0] / acc[1]
            acc[0] = acc[1] = 0.0
        self.cpu_peak = max(self.cpu_peak * 0.999, wall)
        self.wall = 0.95 * getattr(self, "wall", 0.0) + 0.05 * wall

    def _limit(self, side, y, ceiling=0.89):
        """Apply the per-ear block limiter: instant attack ramped across the block, 100 ms release, hard clip."""
        if y.size == 0:
            return y
        out = np.empty(y.size)
        self.lim_g[side] = _limit_kernel(np.ascontiguousarray(y, dtype=np.float64), self.lim_g[side],
                                         np.exp(-y.size / (0.1 * self.fs)), ceiling, out)
        return out

    # ------------------------------------------------------------------------------------------- audio I/O
    def start(self, in_dev=None, out_dev=None, in_channels=None, fs=None, exclusive=False):
        """Open and start a duplex stream.

        With fs=None the stream runs at the input device's native rate (44.1 kHz on the UR22mkII, for example), so
        Windows does no extra resampling. The CI chains adapt to that rate.
        """
        import sounddevice as sd
        self.stop()
        di = sd.query_devices(in_dev if in_dev is not None else sd.default.device[0], "input")
        do = sd.query_devices(out_dev if out_dev is not None else sd.default.device[1], "output")
        fs = int(fs or di["default_samplerate"])
        if fs != self.fs:
            self.set_rate(fs)
        nin = in_channels or min(2, di["max_input_channels"])
        nout = min(2, do["max_output_channels"])
        extra = None
        api = sd.query_hostapis(di["hostapi"])["name"]
        if "WASAPI" in api:
            extra = sd.WasapiSettings(exclusive=True) if exclusive else sd.WasapiSettings(auto_convert=True)
        self.reset()
        self._nin, self._nout = nin, nout
        self.stream = sd.Stream(device=(di["index"], do["index"]), samplerate=self.fs, blocksize=self.blocksize,
                                dtype="float32", channels=(nin, nout), latency="low", callback=self._cb,
                                extra_settings=(extra, extra) if extra else None)
        self.stream.start()
        self.devices = (f"{di['name']} [{api}]", f"{do['name']}")
        self.n_in = nin
        return self.stream

    def _cb(self, indata, outdata, frames, t, status):
        """Handle the sounddevice callback, catching every exception because one would stop the stream silently."""
        try:
            if status:
                self.xruns += 1
            p = np.mean(indata.astype(np.float64) ** 2, axis=0)
            lv = 10 * np.log10(p + 1e-20)
            self.ch_dbfs = (lv if self.ch_dbfs is None or len(self.ch_dbfs) != lv.size
                            else np.maximum(lv, self.ch_dbfs - 1.5))
            out = self.process_multi(indata)
            if outdata.shape[1] >= 2:
                outdata[:, :2] = out.astype(np.float32)
                outdata[:, 2:] = 0
            else:
                outdata[:, 0] = out[:, 0 if self.ci_side == "left" or self.config == "bilateral" else 1]
        except Exception as ex:
            outdata.fill(0)
            self.errors.append(f"audio callback: {ex!r}")

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None

    def latency_ms(self):
        """Return the device, fixed chain and buffering latencies in ms for the displayed ear.

        The last two add up to EarChain.latency_ms() without hooks.
        """
        dev = sum(self.stream.latency) * 1000 if self.stream is not None else 0.0
        c = self.display
        return dev, ALGO_LATENCY_MS[c.proc.kind], 1000 * (c.n_in - c.n_out) / self.fs

    def ci_delay_ms(self):
        """Return the CI ear's delay relative to an unprocessed path (chain, hooks and FIFO), for SSD alignment."""
        side = self.ci_side if self.config != "bilateral" else "left"
        return self.chains[side].latency_ms()

    def gate_status(self):
        """Return (floor, threshold, open) for the gate on the displayed CI ear's mic."""
        side = self.ci_side if self.config != "bilateral" else "left"
        gt = self.gates[side]
        return gt.floor, gt.thresh, gt.open

    # ------------------------------------------------------------------------------------------- calibration
    def calibrate(self, seconds=5.0, target_spl=65.0):
        """Set the input gain from `seconds` of live speech recorded from the CI ear's mic.

        The talker should speak at a normal level from about 1 m. The gain is chosen so the speech reads
        target_spl, using the same active-level measure as the GPyT 65 dB SPL reference file.
        """
        self._cal = []
        time.sleep(seconds)
        blocks, self._cal = self._cal, None
        if not blocks:
            raise RuntimeError("no audio arrived while calibrating. Is the stream running?")
        rec = np.concatenate(blocks)
        g_cur = self.in_gain_db
        lvl = active_level_dbfs(rec / 10 ** (g_cur / 20), self.fs)      # level at 0 dB input gain
        ref = _ref_active_dbfs() + (target_spl - 65.0)
        self.in_gain_db = float(np.clip(ref - lvl, *IN_GAIN_RANGE_DB))
        return lvl, self.in_gain_db


# Fixed part of the mic-to-output latency per vocoder, excluding the sound card and the input-ahead-of-output term
# (n_in - n_out, see EarChain.latency_ms). Median of 1 kHz tone-burst onsets measured at 44.1 and 48 kHz with
# blocks of 128 to 1024 samples; the values agree to within about 1 ms across block sizes. GET includes its 13 ms
# envelope look-ahead. The Wiener hook adds 7.4 ms of overlap-add delay. Totals at 44.1 kHz with 128-sample blocks
# are about 23 ms for gpyt, 21 ms for noise, 20 ms for sine and 31 ms for GET.
ALGO_LATENCY_MS = {"gpyt": 2.7, "noise": 10.2, "sine": 8.3, "get": 19.3}


def process_file(engine, x, blocksize=None):
    """Run a whole signal at engine.fs through the real-time path of the left ear in callback-sized blocks."""
    bs = blocksize or engine.blocksize
    out = [engine.process(x[i:i + bs]) for i in range(0, x.size - bs + 1, bs)]
    return np.concatenate(out) if out else np.zeros(0)


def process_file_multi(engine, x2, blocksize=None):
    """Run an (N, 2) signal through the per-ear hearing configuration and return the (N', 2) output.

    Used to test the bilateral, unilateral and SSD configurations offline.
    """
    bs = blocksize or engine.blocksize
    out = [engine.process_multi(x2[i:i + bs]) for i in range(0, x2.shape[0] - bs + 1, bs)]
    return np.concatenate(out) if out else np.zeros((0, 2))
