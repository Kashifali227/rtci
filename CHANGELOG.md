# Changelog

All notable changes to rtci are recorded here. Versions follow [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-03

First public release.

- Streaming, stateful re-implementation of the GPyT HiRes Fidelity 120 strategy. It matches GPyT's batch output
  to floating-point precision for any input chunking.
- Vocoders, switchable live: GPyT neural vocoder, noise-band, sine, and pulsatile Gaussian-enveloped tones (GET).
- Hearing configurations: bilateral, unilateral, single-sided deafness.
- Live switches for AGC, noise reduction, current steering and the fine-structure carrier; CI / natural A/B.
- Quiet-room gate for microphone self-noise.
- Audio-domain and envelope-domain plug-in hooks, with `register_hook()`.
- Tk GUI with a live electrodogram, a command-line mode, and offline file rendering in any hearing configuration.
- Test suite (90 tests), plus GPyT validation and benchmark scripts.
