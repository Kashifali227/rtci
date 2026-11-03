"""Robustness of the strategy, the plug-in hooks and the engine against bad input, faulty hooks, unsupported
parameters, dropouts and callback errors. Each failure must be contained without stopping or corrupting the chain."""
import collections

import numpy as np
import pytest

from rtci import engine as E
from rtci import f120_stream as F
from rtci import hooks as H
from rtci.gate import DelayLine

from conftest import FS, FS_DEV, at_spl, db


def _tone(fs, f=1000.0, spl=65.0, dur=1.0):
    t = np.arange(int(dur * fs)) / fs
    return at_spl(np.sin(2 * np.pi * f * t), spl)


# --------------------------------------------------------------------------------------------- strategy input
def test_one_nan_sample_does_not_silence_the_implant(speech):
    x = F.CIProcessor(fs_out=FS_DEV).strategy
    x17 = np.interp(np.arange(0, speech.size, FS_DEV / FS), np.arange(speech.size), speech)
    ref = F.CIProcessor(fs_out=FS_DEV)
    bad = x17.copy()
    bad[5000], bad[5001] = np.nan, np.inf
    el_ref = ref.strategy.process(x17)
    el_bad = x.process(bad)
    half = el_ref.shape[1] // 2
    assert x.n_nonfinite == 2
    assert np.isfinite(el_bad).all()
    assert np.isfinite(x.pre_zi).all() and np.isfinite(x.agc_st).all()
    assert np.sqrt(np.mean(el_bad[:, half:] ** 2)) > 0.9 * np.sqrt(np.mean(el_ref[:, half:] ** 2))


def test_multichannel_input_is_rejected():
    st = F.StreamingF120()
    with pytest.raises(ValueError, match="mono"):
        st.process(np.zeros((1000, 2)))
    st.process(np.zeros((1000, 1)))                         # a single column is fine


@pytest.mark.parametrize("key,val", [("nHop", 40), ("nFft", 512), ("decFact", 4), ("steeringRange", np.ones(15)),
                                     ("clipMode", "none"), ("chanToElecPair", np.arange(1, 17))])
def test_unsupported_parameters_raise(key, val):
    p = F.default_params()
    p[key] = val
    with pytest.raises(ValueError):
        F.StreamingF120(p)


def test_agc_off_applies_exactly_the_fixed_gain():
    st = F.StreamingF120()
    st.agc_enabled, st.agc_fixed_db = False, 20.0
    st.process(_tone(FS, spl=60))
    assert np.allclose(st._last_agc[1], 10.0)


def test_long_inputs_keep_the_whole_agc_output():
    x = _tone(FS, dur=1.0)
    a, b = F.StreamingF120(), F.StreamingF120()
    a.process(x)                                            # more than 4096 samples, so it is split internally
    pieces = []
    for i in range(0, x.size, 1000):
        b.process(x[i:i + 1000])
        pieces.append(b._last_agc[0])
    assert np.array_equal(a._last_agc[0], np.concatenate(pieces))


def test_gpyt_vocoder_n_out_counts_emitted_samples():
    v = F.StreamingVocoder(fs_out=FS_DEV)
    rng = np.random.default_rng(0)
    total = 0
    for _ in range(40):
        total += v.process(rng.uniform(0, 300, (16, int(rng.integers(100, 3000))))).size
    assert v.n_out == total


def test_random_carriers_are_seeded_and_leave_the_global_rng_alone():
    np.random.seed(123)
    a = np.random.rand()
    np.random.seed(123)
    v1, v2 = F.StreamingVocoder(seed=4), F.StreamingVocoder(seed=4)
    assert np.random.rand() == a
    assert np.array_equal(v1.phs, v2.phs) and not np.array_equal(v1.phs, F.StreamingVocoder(seed=5).phs)


# --------------------------------------------------------------------------------------------- hooks
class _Raises(H.EnvHook):
    name = "raises"
    enabled = True

    def process(self, env, ctx):
        raise RuntimeError("boom")


class _WrongShape(H.EnvHook):
    name = "wrong shape"
    enabled = True

    def process(self, env, ctx):
        return env[:, :1]


class _Stereo(H.AudioHook):
    name = "stereo"
    enabled = True

    def process(self, x):
        return np.stack([x, x], 1)


@pytest.mark.parametrize("hook", [_Raises(), _WrongShape(), _Stereo()])
def test_faulty_hook_is_switched_off_and_the_chain_continues(hook):
    st, ref = F.StreamingF120(), F.StreamingF120()
    (st.env_hooks if isinstance(hook, H.EnvHook) else st.audio_hooks).append(hook)
    x = _tone(FS, dur=0.5)
    el, el_ref = st.process(x), ref.process(x)
    assert not hook.enabled and len(st.hook_errors) == 1
    assert np.array_equal(el, el_ref)                       # the faulty block is bypassed, as if it were absent


def test_hook_error_reaches_the_engine_error_line():
    e = E.Engine(FS_DEV, 128)
    e.chains["left"].env_hooks.append(_Raises())
    E.process_file(e, _tone(FS_DEV, dur=0.3))
    assert any("raises" in m for m in e.errors)


def test_switching_a_hook_on_resets_its_state():
    e = E.Engine(FS_DEV, 128)
    w = e.chains["left"].audio_hooks[0]
    w.enabled = True
    w.process(np.ones(1000))
    w.enabled = False
    e.set_hook(0, True)
    assert not w.inbuf.any() and not w.ola.any() and w.noise is None


def test_register_hook_adds_a_fresh_instance_per_ear():
    class Mine(H.EnvHook):
        name = "mine"

    H.register_hook(Mine)
    try:
        e = E.Engine(FS_DEV, 128)
        mine = [h for c in e.chains.values() for h in c.env_hooks if isinstance(h, Mine)]
        assert len(mine) == 2 and mine[0] is not mine[1]
    finally:
        H._REGISTERED.remove(Mine)


# --------------------------------------------------------------------------------------------- engine
def test_a_dropout_on_one_ear_leaves_the_ears_aligned(speech):
    e = E.Engine(FS_DEV, 128)
    x2 = np.stack([speech, speech], 1)[: 3 * FS_DEV]
    proc = e.chains["left"].proc
    orig = proc.process
    calls = collections.Counter()

    def flaky(x17):                                         # the left ear's processing fails for three blocks
        calls["n"] += 1
        if 200 <= calls["n"] < 203:
            orig(x17)
            raise RuntimeError("simulated fault")
        return orig(x17)

    proc.process = flaky
    Y = E.process_file_multi(e, x2)
    Y_ref = E.process_file_multi(E.Engine(FS_DEV, 128), x2)  # the same run without the fault
    assert any("simulated fault" in m for m in e.errors)
    assert e.chains["left"].fifo.size == e.chains["right"].fifo.size
    # Well after the fault, each ear equals the fault-free run shifted by the same number of samples, so the ears
    # are time-aligned again. The shared shift is the output the faulty ear lost.
    i0, n = 2 * FS_DEV, FS_DEV // 2

    def shift(ch):
        return next(k for k in range(0, 2000) if np.allclose(Y[i0:i0 + n, ch], Y_ref[i0 + k:i0 + k + n, ch], atol=1e-9))

    assert shift(0) == shift(1)


@pytest.mark.parametrize("kind", ["gpyt", "noise", "sine", "get"])
@pytest.mark.parametrize("fs,bs", [(44100, 128), (48000, 512)])
def test_reported_latency_matches_the_measured_onset(kind, fs, bs):
    e = E.Engine(fs, bs)
    e.set_vocoder(kind)
    t = np.arange(int(1.5 * fs)) / fs
    x = np.where(t >= 1.0, at_spl(np.sin(2 * np.pi * 1000 * t), 65), 0.0)
    y = E.process_file(e, x)
    n = int(0.001 * fs)
    r = np.sqrt(np.mean(y[: y.size // n * n].reshape(-1, n) ** 2, 1))
    on = (np.argmax(r > 0.5 * np.median(r[int(1.2 * fs / n):])) + 0.5) * n / fs - 1.0
    assert abs(1000 * on - e.chains["left"].latency_ms()) < 3.0


def test_changing_the_ssd_delay_does_not_drop_audio(speech):
    e = E.Engine(FS_DEV, 128)
    e.config, e.ci_side, e.nh_delay_ms = "ssd", "left", 20.0
    x2 = np.stack([speech, speech], 1)[: 2 * FS_DEV]
    out = []
    for i in range(0, x2.shape[0] - 128 + 1, 128):
        if i == FS_DEV:
            e.nh_delay_ms = 22.0                            # as the automatic delay matching does after a prefill step
        out.append(e.process_multi(x2[i:i + 128])[:, 1])
    y = np.concatenate(out)[FS_DEV - 2000:FS_DEV + 4000]
    run, longest = 0, 0
    for v in y:
        run = run + 1 if v == 0 else 0
        longest = max(longest, run)
    assert longest <= int(0.0025 * FS_DEV)                  # at most the 2 ms inserted, not a whole buffer (~26 ms)


def test_delay_line_padding_holds_a_closed_gate_closed():
    d = DelayLine(1.0)
    d.process(np.full(100, 0.001), 50)
    assert np.all(d.process(np.full(10, 0.001), 80) == 0.001)


def test_audio_callback_never_raises():
    e = E.Engine(FS_DEV, 128)

    def boom(_):
        raise RuntimeError("x")

    e.process_multi = boom
    out = np.ones((128, 2), np.float32)
    e._cb(np.zeros((128, 2), np.float32), out, 128, None, None)
    assert not out.any() and "audio callback" in e.errors[-1]


def test_calibrate_without_audio_raises_a_clear_error():
    e = E.Engine(FS_DEV, 128)
    with pytest.raises(RuntimeError, match="no audio"):
        e.calibrate(seconds=0.01)


def test_svml_and_plain_kernels_use_separate_caches():
    import numba.core.config as cfg
    from rtci import _svml
    assert cfg.CACHE_DIR.endswith("numba-svml" if _svml.USING_SVML else "numba-nosvml")
