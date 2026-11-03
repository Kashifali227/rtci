"""
Measure whether the chain runs in real time on this machine and how much latency it adds.

  1. per-callback compute time against the block budget, after JIT warm-up, for each block size
  2. the FIFO prefill actually needed so that the output never underruns
  3. algorithmic latency from mic to vocoded output (1 kHz tone-burst onset, -6 dB crossing), sound card excluded

It also writes demo WAVs of reference speech through the real-time path to ./rtci_demo_out/.

    py -3.13 bench.py
"""
import rtci._svml  # noqa: F401,I001  (import before numba to enable SVML vector math)
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
from rtci import engine as E
from rtci.f120_stream import GPYT_DIR

FS = 48000


def load(path, fs=FS):
    x, sr = sf.read(str(path), always_2d=True)
    g = gcd(int(sr), fs)
    return resample_poly(x[:, 0], fs // g, int(sr) // g)


def onset(x, fs, frac=0.5, win_ms=2.0):
    n = int(win_ms * 1e-3 * fs)
    m = x.size // n
    r = np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1))
    steady = np.median(r[int(0.25 * m):]) if m > 8 else r.max()
    i = np.argmax(r > frac * steady)
    return (i + 0.5) * n / fs


def main():
    speech = load(GPYT_DIR / "Sounds" / "AzBio_3sent_65dBSPL.wav")
    out_dir = Path.cwd() / "rtci_demo_out"
    out_dir.mkdir(exist_ok=True)

    # the first call triggers the numba JIT compile
    E.Engine(FS, 480).process(np.zeros(4800))

    print("block  budget   mean    p99     max    load   prefill-needed")
    for bs in (256, 480, 960):
        eng = E.Engine(FS, bs, prefill=4000)
        x = np.tile(speech, 3)
        times, lv = [], []
        for i in range(0, x.size - bs + 1, bs):
            t = time.perf_counter()
            eng.process(x[i:i + bs])
            times.append(time.perf_counter() - t)
            if eng.primed:
                lv.append(eng.fifo.size)
        t = np.array(times[20:]) * 1000
        budget = 1000 * bs / FS
        need = 4000 - min(lv) if lv else -1
        print(f"{bs:5d}  {budget:5.2f}ms {t.mean():6.2f}ms {np.percentile(t, 99):6.2f}ms {t.max():6.2f}ms "
              f"{100 * t.mean() / budget:5.1f}%   {need:4d} samples ({1000 * need / FS:.1f} ms)  underruns={eng.underruns}")

    # latency: silence, then a 1 kHz tone at ~65 dB SPL
    eng = E.Engine(FS, 480)
    tt = np.arange(int(1.5 * FS)) / FS
    amp = 10 ** ((65 - E.SPL_OFFSET) / 20) * np.sqrt(2)
    burst = np.where(tt >= 1.0, amp * np.sin(2 * np.pi * 1000 * tt), 0.0)
    y = E.process_file(eng, burst)
    lat = (onset(y[int(0.5 * FS):], FS) - onset(burst[int(0.5 * FS):], FS)) * 1000
    print(f"\nalgorithmic latency (mic to vocoded output, incl. FIFO prefill {1000 * eng.prefill / FS:.1f} ms): {lat:.1f} ms")

    # demo WAVs through the exact real-time path
    for name, setup in (("ci_stock", lambda e: None),
                        ("ci_no_nr", lambda e: setattr(e.strategy, "nr_enabled", False))):
        e = E.Engine(FS, 480)
        setup(e)
        y = E.process_file(e, speech)
        sf.write(str(out_dir / f"azbio65_{name}.wav"), y.astype(np.float32), FS)
        print(f"wrote rtci_demo_out/azbio65_{name}.wav  rms {20 * np.log10(np.sqrt(np.mean(y ** 2))):.1f} dBFS, "
              f"peak {20 * np.log10(np.max(np.abs(y))):.1f} dBFS")


if __name__ == "__main__":
    main()
