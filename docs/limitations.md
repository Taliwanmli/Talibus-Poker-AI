# Limitations

Talibus is a research prototype with important limitations.

- The game uses abstraction. Action choices are mapped into fixed slots and
  postflop states use card/board clustering.
- Large training buffers, raw logs, PyTorch checkpoints, and ONNX model binaries
  are not committed to Git.
- The included compact result pack records simulator measurements and model
  artifact hashes, but not the full local model files.
- Scripted-opponent win rates do not imply performance against strong human
  players or commercial poker solvers.
- This project does not prove Deep CFR convergence in multiplayer poker.
- Some orchestration scripts were developed for a specific local/Windows
  workflow and may need environment-specific adjustment.
- Depth-limited search timings depend heavily on hardware, ONNX Runtime setup,
  thread count, and batch settings.

The intended public framing is a Deep-CFR-style 6-max NLHE research prototype,
not a production poker bot.
