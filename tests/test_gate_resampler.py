"""Quiet-room gate and streaming resampler.

The gate must silence a microphone noise floor, leave speech intact, and stay open for continuous sound such as
music, which a floor tracker can wrongly learn as background. The resampler must have a flat passband, reject
aliases and be invariant to chunk size."""
import numpy as np
import pytest

from rtci import engine as E
from rtci.resampler import PolyResampler

from conftest import FS_DEV, at_spl, db


def _run(x, gate):
    e = E.Engine(FS_DEV, 128)
    e.gate_enabled = gate
    return E.process_file(e, x)


def test_noise_floor_is_silenced(noise_floor):
    x = noise_floor[: FS_DEV * 6]
    off, on = _run(x, False), _run(x, True)
    tail = slice(FS_DEV * 2, None)                        # after the floor has been learned
    assert db(off[tail]) > -70                            # without the gate the hiss is audible
    assert db(on[tail]) < db(off[tail]) - 25              # with the gate it is gone


@pytest.mark.parametrize("spl", [55, 65])
def test_continuous_music_is_not_muted(music, noise_floor, spl):
    n = FS_DEV * 15
    x = noise_floor[:n].copy()
    x[FS_DEV:] += at_spl(music[: n - FS_DEV], spl)
    off, on = _run(x, False), _run(x, True)
    for t in range(3, 14, 2):                             # every 2 s, 3 s .. 13 s into the file
        w = slice(t * FS_DEV, (t + 1) * FS_DEV)
        assert db(on[w]) > db(off[w]) - 3, f"gate muted continuous music at t={t}s"


def test_floor_estimate_does_not_creep_during_music(music, noise_floor):
    """The learned floor must stay near the true floor while music plays continuously."""
    from rtci.gate import SilenceGate
    g = SilenceGate(FS_DEV, E.SPL_OFFSET)
    for i in range(0, FS_DEV * 2, 128):                   # learn the quiet floor
        g.gains(noise_floor[i:i + 128])
    f0 = g.floor
    x = noise_floor[FS_DEV * 2: FS_DEV * 12] + at_spl(music[: FS_DEV * 10], 55)
    for i in range(0, x.size, 128):
        g.gains(x[i:i + 128])
    assert g.floor - f0 < 3.0, f"floor crept up {g.floor - f0:.1f} dB during 10 s of music"


def test_speech_level_unchanged_by_gate(speech, noise_floor):
    x = np.concatenate((noise_floor[: FS_DEV * 2], speech + noise_floor[FS_DEV * 2: FS_DEV * 2 + speech.size]))
    seg = slice(FS_DEV * 2, None)
    assert abs(db(_run(x, True)[seg]) - db(_run(x, False)[seg])) < 0.5


@pytest.mark.parametrize("f,expect", [(1000, 0.0), (7500, 0.0), (9500, -70.0)])
def test_resampler_response(f, expect):
    t = np.arange(48000) / 48000
    y = PolyResampler(48000, 17400).process(np.sin(2 * np.pi * f * t))
    lvl = 20 * np.log10(np.sqrt(np.mean(y[2000:-100] ** 2)) * np.sqrt(2))
    assert (abs(lvl - expect) < 0.1) if expect == 0.0 else (lvl < expect)


def test_resampler_chunk_invariance():
    x = np.random.default_rng(0).standard_normal(48000)
    whole = PolyResampler(48000, 17400).process(x)
    r = PolyResampler(48000, 17400)
    rng = np.random.default_rng(1)
    parts, i = [], 0
    while i < x.size:
        n = int(rng.integers(1, 1500))
        parts.append(r.process(x[i:i + n]))
        i += n
    y = np.concatenate(parts)
    assert y.size == whole.size == 17400 and np.max(np.abs(y - whole)) == 0.0
