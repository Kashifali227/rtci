"""
Check that the streaming engine reproduces GPyT's batch F120 pipeline and vocoder.

The script runs GPyT's own pipelineTemplate.processingPipeline on a file, then feeds the same 17.4 kHz signal
through f120_stream in random chunk sizes, as a sound card would. It compares the AGC output, the NR-modified
Hilbert envelopes, the electrodogram and the vocoder audio stage by stage. It also measures the fixed output
calibration (OUT_CAL) that replaces GPyT's whole-file -25 dBFS normalization.

    py -3.13 validate_gpyt.py [--wav path] [--chunk-max 2000]
"""
import rtci._svml  # noqa: F401,I001  (import before numba to enable SVML vector math)
import argparse
import sys
import time
import types
from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
from rtci.gpyt_locate import GPYT_DIR as _G  # noqa: E402
TOOLBOX = _G.parent
sys.path.insert(0, str(TOOLBOX))
sys.modules.setdefault("nnresample", types.SimpleNamespace(resample=None))   # only readWav needs it, and it is bypassed
try:
    import matplotlib.pyplot  # noqa: F401  (GpyT/__init__ imports its plotting module)
except ImportError:
    _mpl = types.ModuleType("matplotlib"); _mpl.pyplot = types.ModuleType("matplotlib.pyplot")
    sys.modules["matplotlib"], sys.modules["matplotlib.pyplot"] = _mpl, _mpl.pyplot

from rtci import f120_stream as F


def load17(path):
    x, sr = sf.read(str(path), always_2d=True)
    x = x[:, 0]
    g = gcd(int(sr), F.FS)
    return resample_poly(x, F.FS // g, int(sr) // g)


def run_gpyt(x17):
    import GpyT
    import pipelineTemplate as PT
    GpyT.Frontend.readWavFunc = lambda par: (x17[None, :], "stream-test")
    GpyT.validateOutputFunc = lambda *a, **k: False
    np.random.seed(0)
    t = time.perf_counter()
    r = PT.processingPipeline("unused.wav")
    return r, time.perf_counter() - t


def run_stream(x17, chunk_max, seed=1):
    np.random.seed(0)
    phases = 2 * np.pi * np.random.rand(241)
    fs_g = Fraction(241 * 55556, 278)                     # GPyT's internal vocoder rate, exactly 241 samples/block
    proc = F.CIProcessor(fs_out=fs_g, timing="gpyt", phases=phases)
    envs, els, agc, outs = [], [], [], []
    proc.strategy.on_frames = lambda d: envs.append(d["env"])
    rng = np.random.default_rng(seed)
    i = 0
    t = time.perf_counter()
    while i < x17.size:
        n = int(rng.integers(1, chunk_max))
        el = proc.strategy.process(x17[i:i + n])
        agc.append(proc.strategy._last_agc[0].copy())
        els.append(el)
        outs.append(proc.vocoder.process(el))
        i += n
    dt = time.perf_counter() - t
    return dict(agc=np.concatenate(agc), env=np.concatenate(envs, axis=1), el=np.concatenate(els, axis=1),
                audio=np.concatenate(outs)), dt


def cmp(name, a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    err = np.max(np.abs(a - b))
    scale = np.max(np.abs(b)) + 1e-30
    print(f"  {name:<22} n={n:>9}  max|diff| = {err:.3e}   (rel. to peak {err / scale:.2e})")
    return err / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(TOOLBOX / "GpyT" / "Sounds" / "AzBio_3sent_65dBSPL.wav"))
    ap.add_argument("--chunk-max", type=int, default=2000)
    a = ap.parse_args()

    x17 = load17(a.wav)
    dur = x17.size / F.FS
    print(f"input: {Path(a.wav).name}  {dur:.2f} s @ {F.FS} Hz, rms {20 * np.log10(np.sqrt(np.mean(x17 ** 2))):.1f} dBFS")
    r, tg = run_gpyt(x17)
    s, ts = run_stream(x17, a.chunk_max)
    print(f"GPyT batch: {tg:.2f} s    streaming (random chunks 1..{a.chunk_max}): {ts:.2f} s"
          f", real-time factor {ts / dur:.3f} (must be < 1)")
    worst = []
    worst.append(cmp("AGC output", s["agc"], r["agc"]["wavOut"][0]))
    worst.append(cmp("envelopes (+NR)", s["env"], r["sig_frm_hilbertMod"]))
    worst.append(cmp("electrodogram", s["el"], r["elGram"]))
    ga = r["audioOut"]
    sa = s["audio"]
    n = min(ga.size, sa.size) - 241                         # GPyT's last block is extrapolated; skip it
    k = np.dot(ga[:n], sa[:n]) / np.dot(sa[:n], sa[:n])   # GPyT's global normalization factor
    worst.append(cmp("vocoder audio (scaled)", k * sa[:n], ga[:n]))
    corr = np.corrcoef(sa[:n], ga[:n])[0, 1]
    print(f"  vocoder audio correlation = {corr:.9f}")
    raw_rms = np.sqrt(np.mean(sa ** 2))
    print(f"\noutput level at OUT_CAL={F.OUT_CAL}: {20 * np.log10(raw_rms):.2f} dBFS rms "
          f"(GPyT normalizes to -25; multiply OUT_CAL by {10 ** (-25 / 20) / raw_rms:.4f} to match)")
    ok = max(worst) < 1e-12                 # actual error is ~2e-14 of peak; the tight bound makes a real regression fail
    print("\nPASS: streaming engine reproduces GPyT" if ok else "\nFAIL: mismatch above 1e-12 of peak")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
