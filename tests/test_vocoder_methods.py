"""Method checks for the acoustic vocoders. GET level is flat across channels and pulse rates and does not clip at
normal speech level, bilateral ears get independent carriers, and the fine-structure switch does not change the
acoustic vocoders' level."""
import numpy as np
import pytest

from rtci import engine as E
from rtci.f120_stream import N_CHAN, default_params
from rtci.vocoders_rt import AcousticVocoder

from conftest import FS_DEV, at_spl, db

NFR = int(1.5 * 1851.85)                                     # 1.5 s of stimulation frames


def _chan_level(kind, c, **kw):
    v = AcousticVocoder(kind, FS_DEV, **kw)
    current = np.zeros((N_CHAN, NFR))
    current[c] = 400.0                                       # steady current between T (50) and M (500)
    y = np.concatenate([v.process(current[:, i:i + 37]) for i in range(0, NFR, 37)])[int(0.3 * FS_DEV):]
    return db(y / v.gain)


@pytest.mark.parametrize("rate", [100.0, 200.0, 300.0, 500.0])
def test_get_level_is_flat_across_channels_and_rates(rate):
    sine = [_chan_level("sine", c) for c in range(N_CHAN)]
    get = [_chan_level("get", c, pulse_rate=rate) - sine[c] for c in range(N_CHAN)]
    assert max(abs(g) for g in get) < 0.5                   # no spectral tilt, no rate-dependent level


def test_get_does_not_reach_the_limiter_at_65_db_spl(speech):
    e = E.Engine(FS_DEV, 128)
    e.set_vocoder("get")
    e._limit = lambda side, y: y                            # look at the vocoder output before the safety limiter
    y = E.process_file(e, speech)
    assert np.max(np.abs(y)) < 0.89 and abs(db(y) + 25) < 1.0


@pytest.mark.parametrize("kind", ["gpyt", "noise", "sine", "get"])
def test_bilateral_ears_have_independent_carriers(kind, speech):
    e = E.Engine(FS_DEV, 128)
    e.set_vocoder(kind)
    x = speech[: 3 * FS_DEV]
    Y = E.process_file_multi(e, np.stack([x, 0.5 * x], 1))  # same sound, 6 dB ILD
    r = np.corrcoef(Y[FS_DEV // 2:, 0], Y[FS_DEV // 2:, 1])[0, 1]
    assert abs(r) < 0.7                                     # identical carriers would give ~0.92


@pytest.mark.parametrize("kind", ["noise", "sine", "get"])
def test_fine_structure_switch_does_not_change_acoustic_level(kind):
    x = at_spl(np.sin(2 * np.pi * 500 * np.arange(2 * FS_DEV) / FS_DEV), 65)
    lv = []
    for cm in (0, 1):
        e = E.Engine(FS_DEV, 128)
        e.set_vocoder(kind)
        for st in e.strategies:
            st.carrier_mode = cm
        lv.append(db(E.process_file(e, x)[FS_DEV // 2:]))
    assert abs(lv[1] - lv[0]) < 1.0                         # rendering the gated current would lower it by ~5.5 dB


def test_non_uniform_map_is_rejected():
    p = default_params()
    p["mapM"] = np.linspace(400, 600, 16)
    with pytest.raises(ValueError, match="uniform"):
        AcousticVocoder("sine", FS_DEV, par=p)


@pytest.mark.parametrize("kw", [dict(env_lp=0.0), dict(pulse_rate=0.0)])
def test_invalid_vocoder_options_raise(kw):
    with pytest.raises(ValueError):
        AcousticVocoder("get", FS_DEV, **kw)
