"""Hearing configurations. Each ear follows its own microphone, unilateral silences the other ear, and in SSD the
normal-hearing ear is the other microphone, unvocoded, level-matched and optionally delayed."""
import numpy as np
import pytest

from rtci import engine as E

from conftest import FS_DEV, at_spl


def _env(v, n=int(0.01 * FS_DEV)):
    m = v.size // n
    return np.sqrt(np.mean(v[:m * n].reshape(m, n) ** 2, 1))


def _corr(a, b):
    m = min(a.size, b.size)
    return max(np.corrcoef(a[:m - k], b[k:m])[0, 1] for k in range(8))


@pytest.fixture(scope="module")
def two_mics(speech):
    rng = np.random.default_rng(5)
    # noise with an amplitude modulation unrelated to the speech, so the two envelopes are distinguishable
    t = np.arange(speech.size) / FS_DEV
    other = at_spl(rng.standard_normal(speech.size) * (1.2 + np.sin(2 * np.pi * 1.3 * t)), 65)
    return np.stack([speech, other], 1)


def _run(cfg, side, routing="1L2R", **kw):
    e = E.Engine(FS_DEV, 128)
    e.config, e.ci_side, e.routing = cfg, side, routing
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def test_bilateral_ears_follow_their_own_mic(two_mics):
    Y = E.process_file_multi(_run("bilateral", "left"), two_mics)
    s, o = _env(two_mics[:, 0]), _env(two_mics[:, 1])
    assert _corr(s, _env(Y[:, 0])) > 0.8 and _corr(o, _env(Y[:, 0])) < 0.3
    assert _corr(o, _env(Y[:, 1])) > 0.5 and _corr(s, _env(Y[:, 1])) < 0.3


def test_swapped_routing_swaps_ears(two_mics):
    Y = E.process_file_multi(_run("bilateral", "left", routing="1R2L"), two_mics)
    assert _corr(_env(two_mics[:, 0]), _env(Y[:, 1])) > 0.8


@pytest.mark.parametrize("side,other", [("left", 1), ("right", 0)])
def test_unilateral_other_ear_silent(two_mics, side, other):
    Y = E.process_file_multi(_run("unilateral", side), two_mics)
    assert np.max(np.abs(Y[:, other])) == 0.0
    assert np.max(np.abs(Y[:, 1 - other])) > 0.01


def test_ssd_normal_ear_is_unvocoded_mic(two_mics):
    Y = E.process_file_multi(_run("ssd", "left"), two_mics)
    ref = np.clip(two_mics[:Y.shape[0], 1] * E.NAT_CAL, -0.89, 0.89)
    assert np.max(np.abs(Y[:, 1] - ref)) < 1e-12


def test_ssd_delay_alignment(two_mics):
    e = _run("ssd", "left")
    e.nh_delay_ms = e.ci_delay_ms()
    Y = E.process_file_multi(e, two_mics)
    d = int(round(e.nh_delay_ms * 1e-3 * FS_DEV))
    assert np.max(np.abs(Y[d:, 1] - np.clip(two_mics[:Y.shape[0] - d, 1] * E.NAT_CAL, -0.89, 0.89))) < 1e-12


def test_natural_and_ci_equally_loud(speech):
    a = E.Engine(FS_DEV, 128)
    a.mode = "natural"
    b = E.Engine(FS_DEV, 128)
    ya, yb = E.process_file(a, speech), E.process_file(b, speech)
    assert abs(20 * np.log10(np.std(ya) / np.std(yb))) < 1.0
