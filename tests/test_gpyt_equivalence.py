"""The streaming engine must reproduce GPyT's own batch pipeline (pipelineTemplate.processingPipeline), stage by stage,
when fed the same 17.4 kHz signal in arbitrary chunk sizes."""
from fractions import Fraction

import numpy as np
import pytest

from rtci import f120_stream as F
from rtci.f120_stream import GPYT_DIR

from conftest import load


def _rel(a, b):
    n = min(a.shape[-1], b.shape[-1])
    return np.max(np.abs(a[..., :n] - b[..., :n])) / (np.max(np.abs(b[..., :n])) + 1e-30)


def test_matches_gpyt_batch(gpyt_modules):
    GpyT, PT = gpyt_modules
    x17 = load(GPYT_DIR / "Sounds" / "AzBio_3sent_65dBSPL.wav", F.FS)
    GpyT.Frontend.readWavFunc = lambda par: (x17[None, :], "test")
    GpyT.validateOutputFunc = lambda *a, **k: False
    np.random.seed(0)
    ref = PT.processingPipeline("unused.wav")

    np.random.seed(0)
    phases = 2 * np.pi * np.random.rand(241)
    proc = F.CIProcessor(fs_out=Fraction(241 * 55556, 278), timing="gpyt", phases=phases)
    envs, els, agc, out = [], [], [], []
    proc.strategy.on_frames = lambda d: envs.append(d["env"])
    rng = np.random.default_rng(1)
    i = 0
    while i < x17.size:
        n = int(rng.integers(1, 2000))
        el = proc.strategy.process(x17[i:i + n])
        agc.append(proc.strategy._last_agc[0].copy())
        els.append(el)
        out.append(proc.vocoder.process(el))
        i += n
    assert _rel(np.concatenate(agc), ref["agc"]["wavOut"][0]) < 1e-12
    assert _rel(np.concatenate(envs, axis=1), ref["sig_frm_hilbertMod"]) < 1e-10
    assert _rel(np.concatenate(els, axis=1), ref["elGram"]) < 1e-10
    ga, sa = ref["audioOut"], np.concatenate(out)
    n = min(ga.size, sa.size) - 241                       # GPyT's last block is extrapolated
    k = np.dot(ga[:n], sa[:n]) / np.dot(sa[:n], sa[:n])   # GPyT normalizes the whole file; rtci uses a fixed gain
    assert _rel(k * sa[:n], ga[:n]) < 1e-10
    assert np.corrcoef(sa[:n], ga[:n])[0, 1] > 0.999999999


@pytest.mark.parametrize("fs", [44100, 48000])
def test_live_vocoder_timing_matches_an_interpolation_reference(fs):
    """Compare the live path (timing='real': true pulse timebase, fractional samples per block, causal) with an
    independent reference. The reference interpolates the per-block tone magnitudes linearly with np.interp at
    every output sample time and sums over the tone carriers. Input is fed in random chunks."""
    x17 = load(F.GPYT_DIR / "Sounds" / "AzBio_3sent_65dBSPL.wav", F.FS)[: 3 * F.FS]
    st = F.StreamingF120()
    el = np.concatenate([st.process(x17[i:i + 500]) for i in range(0, x17.size, 500)], 1)
    ph = 2 * np.pi * np.random.default_rng(0).random(241)
    v = F.StreamingVocoder(fs_out=fs, timing="real", phases=ph)
    rng = np.random.default_rng(2)
    outs, i = [], 0
    while i < el.shape[1]:
        n = int(rng.integers(1, 3000))
        outs.append(v.process(el[:, i:i + n]))
        i += n
    y = np.concatenate(outs)
    ref_v = F.StreamingVocoder(fs_out=fs, timing="real", phases=ph)
    nb = el.shape[1] // ref_v.blk
    mags, en = np.empty((nb, ref_v.Wtone.shape[0])), np.empty(300)
    for b in range(nb):
        F._vocoder_kernel(np.ascontiguousarray(el[:, b * ref_v.blk:(b + 1) * ref_v.blk]), ref_v.normRampT,
                          ref_v.normOffset1, ref_v.alpha, float(ref_v.nl), ref_v.nlExp, ref_v.pstate,
                          float(ref_v.mAvg), en)
        mags[b] = ref_v.Wtone @ en
    T = ref_v.blk * F.PULSE_WIDTH_US * 1e-6
    t = np.arange(int(np.ceil((nb - 1) * T * fs))) / fs
    ref = np.zeros(t.size)
    for k in np.flatnonzero(ref_v.tone_ok):
        ref += np.sin(2 * np.pi * ref_v.toneFreqs[k] * t + ph[k]) * np.abs(np.interp(t, np.arange(nb) * T, mags[:, k]))
    ref *= ref_v.gain
    assert y.size == ref.size == v.n_out
    assert np.max(np.abs(y - ref)) / np.max(np.abs(ref)) < 1e-10
