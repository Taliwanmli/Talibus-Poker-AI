# Limitations

Talibus is a research prototype with important limitations.

- The game uses abstraction. Action choices are mapped into fixed slots and
  postflop states use card/board clustering.
- Large training buffers, raw logs, PyTorch checkpoints, and ONNX model binaries
  are not tracked directly in Git. The trained ONNX artifacts, model
  specifications, and evaluation/performance metrics are scheduled for public
  release on Tuesday 26 May 2026 as separately managed release artifacts.
- The included compact result pack records simulator measurements and model
  artifact hashes. The separately managed model release is intended to provide
  the public trained artifacts without bloating normal source history.
- Scripted-opponent win rates do not imply performance against strong human
  players or external solver systems.
- This project does not prove Deep CFR convergence in multiplayer poker.
- Some orchestration scripts were developed for a specific local/Windows
  workflow and may need environment-specific adjustment.
- Depth-limited search timings depend heavily on hardware, ONNX Runtime setup,
  thread count, and batch settings.

The intended public framing is a Deep-CFR-style 6-max NLHE research prototype,
not a production poker bot.
