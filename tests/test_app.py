"""Command line and offline rendering (no audio hardware needed)."""
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from rtci import app
from rtci import engine as E

from conftest import FS_DEV, db


def _rtci(*args):
    return subprocess.run([sys.executable, "-m", "rtci", *args], capture_output=True, text=True, timeout=300)


def test_help_does_not_need_gpyt_or_audio(tmp_path):
    r = subprocess.run([sys.executable, "-c", "import sys, rtci.app; sys.exit('rtci.engine' in sys.modules)"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr                      # importing the app must not load the engine or GPyT


@pytest.mark.parametrize("args,msg", [(["--block", "0"], "--block"), (["--get-rate", "0"], "--get-rate"),
                                      (["--env-lp", "0"], "--env-lp"), (["--outfile", "x.wav"], "--outfile")])
def test_cli_rejects_bad_values(args, msg):
    r = _rtci(*args)
    assert r.returncode == 2 and msg in r.stderr


@pytest.fixture
def engine():
    app._load_engine()
    e = E.Engine(FS_DEV, 128)
    e.gate_enabled = False
    return e


def test_render_stereo_file_keeps_both_channels(engine, speech, tmp_path):
    x = speech[: 2 * FS_DEV]
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, np.stack([np.zeros_like(x), x], 1), FS_DEV)
    app.render_file(str(src), str(dst), engine)
    y, sr = sf.read(dst, always_2d=True)
    assert sr == FS_DEV and y.shape == (x.size, 2)          # stereo, same length as the input (delay trimmed)
    assert db(y[:, 1]) > -40 and db(y[:, 0]) < -100         # speech only on the right channel reaches the right ear only


def test_render_uses_the_hearing_configuration(engine, speech, tmp_path):
    x = speech[: 2 * FS_DEV]
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, x, FS_DEV)                                # a mono file feeds both microphones
    engine.config, engine.ci_side = "unilateral", "right"
    app.render_file(str(src), str(dst), engine)
    y, _ = sf.read(dst, always_2d=True)
    assert np.max(np.abs(y[:, 0])) == 0 and db(y[:, 1]) > -40


def test_render_trim_aligns_output_with_input(engine, tmp_path):
    t = np.arange(2 * FS_DEV) / FS_DEV
    x = np.where(t >= 1.0, 0.05 * np.sin(2 * np.pi * 1000 * t), 0.0)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, x, FS_DEV)
    app.render_file(str(src), str(dst), engine)
    y, _ = sf.read(dst, always_2d=True)
    n = int(0.001 * FS_DEV)
    r = np.sqrt(np.mean(y[: y.shape[0] // n * n, 0].reshape(-1, n) ** 2, 1))
    on = (np.argmax(r > 0.5 * np.median(r[int(1.3 * FS_DEV / n):])) + 0.5) * n / FS_DEV
    assert abs(on - 1.0) < 0.004
