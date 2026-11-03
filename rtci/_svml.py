# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
r"""numba setup shared by every rtci module. It must be imported before numba.

1. It lets numba find Intel's SVML vector math library (the pip package icc_rt), so that exp and similar
   functions in fastmath kernels are vectorized. This matters because the neural stage of the GPyT vocoder is
   bound by exp. pip --user installs the DLL under %APPDATA%\Python\Library\bin, which numba does not search,
   so that folder is added to PATH. Without icc_rt, numba simply runs without SVML.

2. It keeps separate on-disk kernel caches for SVML and non-SVML builds. numba's cache key does not record
   whether SVML was used, and a kernel cached with SVML links SVML symbols. Loading such a kernel in a process
   without SVML (numba imported before rtci, NUMBA_DISABLE_INTEL_SVML=1, or icc_rt uninstalled) aborts the
   process with "LLVM ERROR: Symbol not found: __svml_...". The cache lives in a per-user directory, so
   read-only installs work too. A NUMBA_CACHE_DIR set by the user is respected, and the split is made inside it.
"""
import os
import site
import sys

for _d in (os.path.join(site.getuserbase(), "Library", "bin"), os.path.join(sys.prefix, "Library", "bin")):
    if os.path.isfile(os.path.join(_d, "svml_dispmd.dll")) and _d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")


def _cache_root():
    if os.environ.get("NUMBA_CACHE_DIR"):
        return os.environ["NUMBA_CACHE_DIR"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "rtci", "cache")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches", "rtci")
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "rtci")


import numba.core.config as _cfg  # noqa: E402  (after the PATH setup, so numba finds SVML if it is installed)

USING_SVML = bool(_cfg.USING_SVML)
_cfg.CACHE_DIR = os.path.join(_cache_root(), "numba-" + ("svml" if USING_SVML else "nosvml"))
