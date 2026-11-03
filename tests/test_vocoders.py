"""Every vocoder has a calibrated level, does not depend on how the audio is chunked, and has per-band envelopes
that follow the input."""
import numpy as np
import pytest
from scipy.signal import butter, hilbert, lfilter

from rtci import engine as E
from rtci.vocoders_rt import f120_bands

from conftest import FS_DEV, db

KINDS = ["gpyt", "noise", "sine", "get"]


def _engine(kind):
    np.random.seed(0)                                     # GPyT vocoder draws random carrier phases
    e = E.Engine(FS_DEV, 128)
    e.set_vocoder(kind, **({} if kind == "gpyt" else {"env_lp": 300.0}))
    return e


def _band_env(y):
    edges, _ = f120_bands()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b, a = butter(3, [lo / (FS_DEV / 2), min(hi, 0.98 * FS_DEV / 2) / (FS_DEV / 2)], btype="band")
        env = np.abs(hilbert(lfilter(b, a, y)))
        bl, al = butter(2, 30 / (FS_DEV / 2))
        out.append(lfilter(bl, al, env)[:: FS_DEV // 200])
    return np.array(out)


@pytest.mark.parametrize("kind", KINDS)
def test_level_65dB_speech_is_minus25dBFS(kind, speech):
    y = E.process_file(_engine(kind), speech)
    assert abs(db(y) + 25.0) < 0.5


@pytest.mark.parametrize("kind", KINDS)
def test_chunk_invariance(kind, speech):
    x = speech[: FS_DEV * 3]
    a, b = _engine(kind), _engine(kind)
    ya = a.proc.process(a.rs.process(x))
    rng = np.random.default_rng(3)
    parts, i = [], 0
    while i < x.size:
        n = int(rng.integers(1, 3000))
        parts.append(b.proc.process(b.rs.process(x[i:i + n])))
        i += n
    yb = np.concatenate(parts)
    m = min(ya.size, yb.size)
    assert np.max(np.abs(ya[:m] - yb[:m])) / np.max(np.abs(ya)) < 1e-9


@pytest.mark.parametrize("kind", KINDS)
def test_band_envelopes_follow_input(kind, speech):
    y = E.process_file(_engine(kind), speech)
    R, V = _band_env(speech), _band_env(y)
    best = max(np.nanmean([np.corrcoef(R[c, :min(R.shape[1], V.shape[1] - lag)],
                                       V[c, lag:lag + min(R.shape[1], V.shape[1] - lag)])[0, 1] for c in range(15)])
               for lag in range(30))
    assert best > 0.8


@pytest.mark.parametrize("kind", KINDS)
def test_digital_silence_gives_silence(kind):
    y = E.process_file(_engine(kind), np.zeros(FS_DEV * 2))
    assert np.max(np.abs(y)) == 0.0
