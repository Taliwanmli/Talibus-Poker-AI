# Remote Windows Training Runbook

This guide describes a general workflow for running larger Talibus training
jobs on a remote Windows machine with an NVIDIA GPU. It is optional; local CPU
or GPU smoke runs are enough for development and testing.

## When To Use This

Use a remote Windows GPU machine when a run needs more compute, memory, or wall
time than a local workstation can provide. The orchestration scripts are plain
Python and Rust binaries, so the same project layout works locally and remotely.

## Prerequisites

- Windows 10/11 or Windows Server.
- Rust stable with Cargo.
- Python 3.10 or newer.
- NVIDIA driver and CUDA-compatible PyTorch build for GPU training.
- Git.
- Enough disk space for `data/` artifacts. Long runs can create many gigabytes
  of buffers, checkpoints, logs, and intermediate samples.

## Prepare The Repository

Clone the repository on the remote machine:

```powershell
git clone https://github.com/Taliwanmli/Talibus-Poker-AI.git Talibus-Poker-AI
Set-Location Talibus-Poker-AI
```

Create the Python environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r training\deep_cfr\requirements.txt
python -m pip install -r eval\requirements.txt
```

Build the Rust binaries:

```powershell
Set-Location solver
cargo build --release -p deep_cfr --bin run_traversals --bin ring_game_eval --bin realtime_play
Set-Location ..
```

Run basic checks:

```powershell
python -m compileall -q .
python -m unittest discover eval
python run_eval_suite.py --help
```

## Start With A Smoke Run

Before launching a long run, execute a small configuration to validate paths,
cluster assets, ONNX Runtime, and PyTorch device selection:

```powershell
python training\deep_cfr\run_deep_cfr.py --help
```

Use the script help output to choose a small number of iterations, traversals,
and training steps. Confirm that `data/` is populated and that the process can
write model artifacts.

## Long Runs

For long runs:

- Keep generated artifacts under `data/`.
- Use a dedicated work directory per experiment.
- Record the exact command, commit hash, Python package versions, Rust version,
  GPU name, and important environment variables.
- Enable optional debug logging only when needed:

```powershell
$env:TALIBUS_DEBUG_LOG = "debug\talibus_debug.jsonl"
```

The most useful public artifacts from a long run are compact summaries:

- final `summary.json` files,
- compact metrics summaries,
- final evaluation reports,
- command and environment notes,
- model artifact names, sizes, and hashes.

Large buffers, raw sample dumps, PyTorch checkpoints, ONNX files, and raw logs
should be stored outside Git. GitHub Releases or Git LFS are better choices for
model binaries that need to be shared.

## Evaluation

After training, build the release binaries and run:

```powershell
python run_eval_suite.py --help
```

The evaluation suite can run model-only ring evaluation, depth-limited runtime
search experiments, mixed-table tests, seat rotation, opponent comparisons,
checkpoint progression, and search-budget sweeps depending on the selected
arguments.

## Result Packaging

Public result packs should be placed under `results/<run-name>/` and should
avoid absolute local paths. Prefer compact, sanitized JSON summaries that keep:

- model and cluster asset identifiers,
- seeds and hand counts,
- opponent configuration,
- budget and threading settings,
- bb/100 values,
- decision-time and iteration metrics,
- known limitations.

Do not include private machine paths, local usernames, raw training buffers, or
large model binaries in Git.
