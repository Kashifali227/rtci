# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
"""Locate the Generic Python Toolbox (GPyT, GPL-3.0), an external dependency.

rtci re-implements the GPyT F120 pipeline as a streaming engine and reuses GPyT's vocoder helpers and data
files: Vocoder/vocoderTools.py and MatlabSupportFiles/spread.mat, plus Sounds/AzBio_3sent_65dBSPL.wav for level
calibration and the equivalence test. GPyT is searched for in this order:
  1. The environment variable RTCI_GPYT, pointing at the toolbox root or at the GpyT package folder inside it.
     If it is set but wrong, that is an error; another copy is never used in its place.
  2. An importable GpyT package.
  3. A folder named "AB-Generic-Python-Toolbox" next to the rtci source folder, in ./third_party/, or in the
     current working directory.
GPyT is only read, never installed. Its setup.py pins 2020 versions of numpy, scipy and numba that do not
install on Python 3.10 or later, so do not pip install it.
"""
import importlib.util
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_HINT = ("Clone https://github.com/jabeim/AB-Generic-Python-Toolbox and either place it next to the rtci folder as "
         "'AB-Generic-Python-Toolbox' or set the environment variable RTCI_GPYT to its path. See README.md -> "
         "Installation.")


def _as_pkg(p):
    """Return the GpyT package folder at p or p/GpyT, or None if neither holds the files rtci needs."""
    p = Path(p)
    for c in (p, p / "GpyT"):
        if (c / "Vocoder" / "vocoderTools.py").is_file() and (c / "MatlabSupportFiles" / "spread.mat").is_file():
            return c.resolve()
    return None


def find_gpyt():
    """Return the path of the GpyT package folder, or raise FileNotFoundError."""
    env = os.environ.get("RTCI_GPYT")
    if env:
        p = _as_pkg(env)
        if p is None:
            raise FileNotFoundError(f"RTCI_GPYT={env!r} does not contain GPyT (no GpyT/Vocoder/vocoderTools.py). "
                                    + _HINT)
        return p
    cands = []
    try:
        spec = importlib.util.find_spec("GpyT")
        if spec and spec.submodule_search_locations:
            cands += list(spec.submodule_search_locations)
    except (ImportError, ValueError):
        pass
    cands += [_REPO.parent / "AB-Generic-Python-Toolbox", _REPO / "third_party" / "AB-Generic-Python-Toolbox",
              Path.cwd() / "AB-Generic-Python-Toolbox"]
    for c in cands:
        p = _as_pkg(c)
        if p is not None:
            return p
    raise FileNotFoundError("GPyT (Advanced Bionics Generic Python Toolbox) not found. " + _HINT)


GPYT_DIR = find_gpyt()
