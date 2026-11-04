# rtci: a real-time cochlear-implant simulator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23946047.svg)](https://doi.org/10.5281/zenodo.23946047)

rtci turns a laptop and a headset into a live cochlear-implant (CI) simulator. Microphone audio is processed by a
streaming implementation of the Advanced Bionics HiRes Fidelity 120 (F120) sound-coding strategy, rendered through
a CI vocoder, and played to the headphones. With a USB audio interface, the total latency is about 55–60 ms.

The strategy is a block-by-block re-implementation of the [Generic Python Toolbox (GPyT)](https://github.com/jabeim/AB-Generic-Python-Toolbox).
Given the same signal, it reproduces GPyT's batch output to floating-point precision at every stage, from the AGC
and the envelopes through the electrodogram to the vocoder audio. A test checks this on every run. Experiments
that use GPyT offline can therefore be heard live, with the same processing.

rtci is meant for:
- listening demonstrations;
- teaching;
- piloting experiments;
- testing new processing blocks by ear in real time (noise reduction, dereverberation, envelope manipulations)
  before running a formal study.

## Features

- **Strategy.** The GPyT F120 pipeline: pre-emphasis, dual-loop AGC, 256-point FFT filterbank at 870 frames/s,
  Hilbert envelopes, AB noise reduction, spectral peak locator, current steering, carrier synthesis, F120 mapping,
  and a 16-electrode biphasic electrodogram at 55.6 kHz.
- **Vocoders**, switchable live. All of them are driven by the same F120 stimulation, so only the acoustic carrier
  differs. Each is calibrated so that 65 dB SPL speech plays at −25 dBFS:

  | vocoder | reference |
  |---|---|
  | GPyT neural vocoder (electric-field spread → neural activity → tone complex) | GPyT |
  | noise-band | Shannon et al., 1995 |
  | sine / tone | Dorman et al., 1997 |
  | pulsatile Gaussian-enveloped tones (GET) | Meng et al., 2022 |

  GET is a re-implementation of the authors' reference code ([BetterCI/GETVocoder](https://github.com/BetterCI/GETVocoder)).
  It places Gaussian envelopes of duration *l*/*f*c on a continuous carrier and pulses the channels in the F120
  stimulation order. Its level is the same at every pulse rate (100–500 pps) and flat across channels.
  The fine-structure carrier switch affects only the GPyT vocoder. The envelope vocoders render the stimulation
  current without the fine-structure on/off gating, which they cannot represent.

- **Hearing configurations:**
  - bilateral: two independent processors, one per microphone;
  - unilateral: the other ear is silent;
  - single-sided deafness (SSD): CI on one ear, while the other ear hears its own microphone unvocoded and
    level-matched, with optional delay matching.
- **Live switches** for AGC, noise reduction, current steering and the fine-structure carrier. One key switches
  between CI and natural sound for A/B listening.
- **Calibration.** Speech at a normal level is mapped to 65 dB SPL, the level of GPyT's reference file.
- **Quiet-room gate.** It mutes microphone self-noise in silence without affecting speech or music.
- **Display.** A live electrodogram view.
- **Plug-in points** (`rtci/hooks.py`) for your own blocks:
  - an *audio* hook at 17.4 kHz, before the AGC;
  - an *envelope* hook on the 15 channel envelopes, after noise reduction.

  Register a block with `rtci.hooks.register_hook(MyBlock)`; the `hooks.py` docstring has an example. Each block
  gets an on/off switch in the GUI. A block that raises an error or returns invalid output is switched off
  automatically and reported, so it cannot stop the audio.
- **Offline rendering** through the same real-time path, in any hearing configuration, with stereo output.

## Installation

You need Python ≥ 3.10, a full-duplex sound card, and a wired or USB headset or audio interface. Bluetooth
headsets drop to narrowband "hands-free" mode when their microphone is open.

Live operation has been tested on Windows 11 with ASIO and WASAPI. The code also selects Core Audio on macOS and
ALSA on Linux, and the test suite runs on Windows and Linux, but live audio on macOS and Linux has not been tested
yet. Reports are welcome.

```bash
git clone https://github.com/Kashifali227/rtci.git
git clone https://github.com/jabeim/AB-Generic-Python-Toolbox.git    # GPyT, GPL-3.0 (validated at commit 56b980e)
cd rtci
pip install -e .[test]          # add ,fast on x86 for Intel SVML (vectorized exp, ~3x faster GPyT vocoder)
```

rtci looks for GPyT in a sibling `AB-Generic-Python-Toolbox` folder, in `AB-Generic-Python-Toolbox` under the
current directory, or at the path in the `RTCI_GPYT` environment variable. GPyT is only read, never installed.
Do not `pip install` it: its `setup.py` pins 2020 versions of numpy, scipy and numba that do not install on
Python ≥ 3.10.

## Usage

```bash
rtci                                   # GUI (or: python -m rtci); pick devices, Start, Calibrate; space = CI / natural
rtci --list                            # list audio devices (Windows: run_ci_simulator.bat also works)
rtci --nogui --config ssd --ci-side left
rtci --file speech.wav --vocoder get --outfile speech_ci.wav   # offline render through the identical real-time path
rtci --file scene.wav --config ssd --ci-side right --outfile scene_ssd.wav   # stereo file: channel 1 = left mic
```

On Windows, ASIO (installed with most audio interfaces) or WASAPI exclusive mode gives the lowest latency.
rtci selects the interface's own driver automatically.

Offline renders are stereo files [left, right], aligned with the input. Use `--no-trim` to keep the processing
delay. A mono input file feeds both microphones.

## Validation and tests

```bash
pytest                                 # 90 tests, ~1 min
python benchmarks/validate_gpyt.py     # stage-by-stage report against GPyT's pipelineTemplate
python benchmarks/bench_rt.py          # CPU load per configuration
```

| check | result |
|---|---|
| equivalence with GPyT, input fed in random chunk sizes (`validate_gpyt.py`) | AGC 9e-16, envelopes 2e-14, electrodogram 1e-14, vocoder audio 6e-16 (relative to peak); correlation 1.000000000; passes at 1e-12 |
| equivalence in the test suite | same comparison, asserted < 1e-10 (AGC < 1e-12) |
| every vocoder | 65 dB SPL speech → −25 dBFS RMS without limiting; output independent of chunk size; per-band envelopes follow the input (r > 0.8); GET flat across channels and pulse rates (± 0.5 dB) |
| hearing configurations | each ear follows its own microphone; bilateral ears get independent carriers and stay time-aligned after a fault; unilateral far ear exactly silent; SSD normal ear bit-identical to its microphone |
| quiet-room gate | noise floor silenced (> 25 dB lower); continuous music not muted; speech level unchanged |
| robustness | NaN/inf input, faulty hooks, unsupported parameters and audio-callback errors cannot stop or corrupt the chain |
| latency | the reported latency matches the measured onset within 3 ms |

The GPyT comparisons use GPyT's own block timing at GPyT's output rate (`timing="gpyt"`). This is what makes a
sample-for-sample comparison possible. The live configuration, with the device-rate resampler, real pulse timing
and the causal vocoder, is checked separately against an independent interpolation reference and for chunk
invariance.

**Performance.** Measured with `benchmarks/bench_rt.py` on an Intel i9-9980HK laptop at 44.1 kHz, as a percentage
of one core:

| | block 128 | block 256 |
|---|---|---|
| SSD / unilateral | 7–10 % | 4–8 % |
| bilateral | 12–18 % | 7–12 % |

**Latency.** At 44.1 kHz with 128-sample blocks, rtci itself adds about 20–23 ms from microphone input to
headphone output. GET needs look-ahead for its Gaussian envelopes, which raises this to 31 ms. The figure includes
rtci's output buffer, which absorbs the vocoder's 5 ms output segments. The audio interface adds the rest. A
Steinberg UR22mkII with ASIO at the driver's default buffer adds about 35 ms, for ≈ 55–60 ms in total. The GUI
shows the live figure.

## Two deliberate differences from GPyT

1. **The vocoder is causal.** GPyT assigns each 5 ms block's magnitude to the start of that block, which needs
   5 ms of look-ahead. rtci plays each segment once it is complete. The samples are the same, 5 ms later.
2. **The output gain is fixed.** GPyT normalizes every file to −25 dBFS, which cannot be done live. rtci uses a
   fixed gain, calibrated so that 65 dB SPL speech plays at −25 dBFS.

## Contributing and support

Please use the GitHub issue tracker for bug reports, questions and feature requests. Pull requests are welcome.
Run `pytest` first, and for any change to the strategy or the vocoders, keep `benchmarks/validate_gpyt.py`
passing. New processing blocks belong in `rtci/hooks.py`; see the examples there.

## Citing

If you use rtci in research, please cite the software and GPyT, on which it is built. The citation is in
`CITATION.cff`, and the archived release has a Zenodo DOI.

## License and disclaimer

rtci is licensed under GPL-3.0-only, the license of GPyT (GNU GPL version 3), from which the strategy is derived.
The strategy modules (`rtci/f120_stream.py`, `rtci/f120_kernels.py`) are ports of GPyT code and say so in their
headers. GPyT itself is a Python port of the Advanced Bionics Generic MATLAB Toolbox.

- rtci is a research and teaching tool. It is not a medical device and is not intended for clinical use.
- Advanced Bionics, HiRes and HiRes Fidelity 120 are trademarks of their owners. They are used here only to
  describe the emulated strategy. This project is not affiliated with or endorsed by Advanced Bionics.
- Vocoders are acoustic models. What you hear approximates CI listening; it is not what any individual CI user
  perceives.
- Start with the volume low.
