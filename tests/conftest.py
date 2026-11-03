import sys
import types
from math import gcd

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

import rtci._svml  # noqa: F401  (before numba)
from rtci import engine as E
from rtci.f120_stream import FS, GPYT_DIR

FS_DEV = 44100


def load(path, fs):
    x, sr = sf.read(str(path), always_2d=True)
    g = gcd(int(sr), fs)
    return resample_poly(x[:, 0], fs // g, int(sr) // g)


def at_spl(x, spl):
    """Scale x to `spl` dB SPL on the engine's level scale (GPyT reference: -43.6 dBFS RMS == 65 dB SPL)."""
    return x * 10 ** ((spl - (20 * np.log10(np.std(x)) + E.SPL_OFFSET)) / 20)


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x) ** 2)) + 1e-12)


@pytest.fixture(scope="session")
def speech():
    """AzBio sentences shipped with GPyT, at 65 dB SPL, device rate."""
    return load(GPYT_DIR / "Sounds" / "AzBio_3sent_65dBSPL.wav", FS_DEV)


@pytest.fixture(scope="session")
def music():
    return load(GPYT_DIR / "Sounds" / "BobbyBareDropKickMeJesus.wav", FS_DEV)


@pytest.fixture(scope="session")
def noise_floor():
    """Synthetic microphone self-noise, 20 s of white noise at 25 dB SPL. This matches the in-band (300 Hz and up)
    level measured for a real quiet room through a Steinberg UR22mkII (median ~24 dB SPL in the gate's detector
    band)."""
    rng = np.random.default_rng(7)
    return at_spl(rng.standard_normal(FS_DEV * 20), 25.0)


@pytest.fixture(scope="session")
def gpyt_modules():
    """Import GPyT's own batch pipeline (pipelineTemplate.processingPipeline) with its optional deps stubbed."""
    sys.modules.setdefault("nnresample", types.SimpleNamespace(resample=None))
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        mpl = types.ModuleType("matplotlib")
        mpl.pyplot = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"], sys.modules["matplotlib.pyplot"] = mpl, mpl.pyplot
    sys.path.insert(0, str(GPYT_DIR.parent))
    import GpyT
    import pipelineTemplate
    return GpyT, pipelineTemplate


__all__ = ["FS", "FS_DEV", "load", "at_spl", "db"]
