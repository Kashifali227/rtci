# Contributing to rtci

Thank you for your interest in rtci. Bug reports, questions, feature requests and pull requests are all welcome.

## Reporting a problem or asking a question

Open an issue on the GitHub issue tracker. For a bug, please include:
- your operating system, Python version, audio interface and driver (ASIO, WASAPI, Core Audio, ALSA);
- the exact command or GUI settings (vocoder, hearing configuration, AGC/NR/gate switches);
- the full error message or traceback, or a short description of what you heard.

## Contributing code

1. Fork the repository and create a branch for your change.
2. Install the development dependencies with `pip install -e .[test]`. GPyT must be locatable; see the README.
3. Run `pytest` before opening a pull request. All tests must pass.
4. Any change to the strategy (`rtci/f120_stream.py`, `rtci/f120_kernels.py`) or the vocoders
   (`rtci/vocoders_rt.py`) must keep `python benchmarks/validate_gpyt.py` passing, so that rtci's output stays
   equivalent to GPyT's batch output.
5. New processing blocks belong in `rtci/hooks.py`, as an `AudioHook` or `EnvHook`; see the examples there.
6. In the pull request, describe what the change does and how you tested it.

## Support and governance

rtci is maintained by Kashif Ali (Budapest University of Technology and Economics, BME). Issues and pull requests
are answered on a best-effort basis, usually within two weeks. The maintainer decides on scope and releases.
Anyone who contributes substantially is welcome to become a co-maintainer.

## Code of conduct

Be respectful and constructive. Harassment and personal attacks are not tolerated in issues, pull requests or any
other project space.
