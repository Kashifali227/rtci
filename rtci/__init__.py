# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
"""Real-time cochlear-implant simulator.

rtci runs a streaming Advanced Bionics HiRes Fidelity 120 strategy that matches GPyT exactly, on live microphone
input. It offers selectable vocoders (GPyT neural, noise-band, sine and GET), bilateral, unilateral and
single-sided deafness configurations, and plug-in points for testing new processing blocks by ear."""
__version__ = "0.1.0"
