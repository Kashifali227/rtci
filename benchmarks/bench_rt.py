"""Per-block cost of the live path (process_multi) for every configuration, with a CPU-speed reference.

    py -3.13 bench_rt.py

Numbers are a percentage of the real-time budget; below 100 % the chain keeps up."""
import rtci._svml  # noqa: F401,I001  (import before numba to enable SVML vector math)
import time
import numpy as np
from rtci import engine as E


def ref():
    t = time.perf_counter()
    a, b = np.random.rand(300, 16), np.random.rand(16, 278)
    for _ in range(2000):
        np.exp(a @ b * 0.01)
    return time.perf_counter() - t


def load(cfg, voc, bs, fs=44100, n=None):
    eng = E.Engine(fs, bs)
    eng.config, eng.gate_enabled = cfg, True
    eng.set_vocoder(voc, **({} if voc == "gpyt" else {"env_lp": 300.0}))
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((fs * 4, 2)) * 0.02).astype(np.float32)      # ~70 dB SPL noise keeps every stage active
    for i in range(0, fs, bs):
        eng.process_multi(X[i:i + bs])
    t = time.perf_counter()
    m = 0
    for i in range(fs, X.shape[0] - bs, bs):
        eng.process_multi(X[i:i + bs])
        m += bs
    return (time.perf_counter() - t) / (m / fs)


if __name__ == "__main__":
    print(f"CPU reference {ref():.2f} s  (full speed ~1.3 s; larger = throttled)")
    for bs in (128, 256):
        row = f"block {bs}:"
        for cfg in ("ssd", "bilateral"):
            for voc in ("gpyt", "noise", "sine", "get"):
                row += f"  {cfg[:3]}/{voc} {100 * load(cfg, voc, bs):4.0f}%"
        print(row, flush=True)
