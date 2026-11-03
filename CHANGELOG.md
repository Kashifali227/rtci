# Changelog

All notable changes to rtci are recorded here. Versions follow [semantic versioning](https://semver.org/).

## [0.1.0] - unreleased

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

### Fixed after a code audit (2026-11-02), before the first release

- Crash: kernels cached with Intel SVML aborted the process ("LLVM ERROR: Symbol not found: __svml_exp4") when a
  later session ran without SVML. SVML and non-SVML builds now use separate caches in a per-user folder.
- A single NaN or inf input sample silenced an ear permanently. Non-finite samples are now replaced and counted.
- GET: level and channel balance changed with the pulse rate (+8 dB at 500 pps, single channels off by up to
  26 dB), and the spectrum tilted by ~10 dB. GET now follows the authors' reference implementation: Gaussian
  envelope `exp(-pi t^2 / D^2)` with `D = l/fc` (the previous envelope was sqrt(2) wider), truncated at ±1.5 D and
  combined by maximum, on a continuous carrier, with channels interleaved in the F120 stimulation order and the
  level normalized per channel. It is flat to ±0.2 dB at every rate, and its latency is 31 ms instead of ~37 ms.
- Bilateral ears shared identical random carriers, giving an interaural correlation of ~0.92 against ~0.5 for
  GPyT's own vocoder. Each ear now has its own seed.
- Fine-structure mode lowered the acoustic vocoders' low channels by ~5.5 dB. These vocoders now render the
  current without the fine-structure gating, and their calibration was measured again.
- A failing or malformed plug-in hook replayed stale stimulation or read out of bounds. Hooks are now guarded: a
  faulty hook is switched off and reported. Hooks are reset when they are switched on.
- Unsupported parameters (e.g. `nHop`, `nFft`, `decFact`) were silently ignored or corrupted memory. They now
  raise `ValueError`. Multichannel input to the strategy now raises an error instead of being interleaved.
- `--file` rendered only input channel 0, through the left ear, and ignored `--config`. It now renders stereo in
  the chosen configuration, aligned with the input.
- Bilateral ears could drift 0.5–1.6 ms apart after a dropout. Both ears now re-prime together and stay aligned.
- Latency figures: the reported latency left out rtci's output buffer. It is now computed exactly, and the README
  and paper were corrected (20–23 ms, 31 ms with GET, at 44.1 kHz / 128).
- Smaller fixes: audio-callback errors no longer stop the stream silently; changing the SSD delay no longer drops
  ~26 ms of audio; the quiet-room gate no longer opens briefly when the latency changes; the Windows CPU meter is
  meaningful; the default buffer no longer causes a dropout at large blocks; calibration without audio reports an
  error instead of freezing the GUI; command-line values are validated; `--help` and `--list` work without GPyT;
  terminal mode and audio-API defaults work on non-Windows systems; an invalid `RTCI_GPYT` is reported.
- License: GPL-3.0-only, since GPyT is GPL v3 without "or later". The ported modules now carry GPyT attribution.
