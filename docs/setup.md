# Setup

This document describes the development setup for Talibus.

## Prerequisites

- Rust stable with Cargo.
- Python 3.10 or newer.
- A C/C++ toolchain suitable for Rust native dependencies.
- Enough disk space for generated `data/` artifacts if running training.
- Optional CUDA-capable GPU for larger PyTorch training runs.

The Rust runtime uses the `ort` crate for ONNX inference. Depending on the
platform, ONNX Runtime may need to be installed or made available through a
dynamic-library path before running binaries that load `.onnx` models.

## Python Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r training/deep_cfr/requirements.txt
pip install -r eval/requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r training\deep_cfr\requirements.txt
python -m pip install -r eval\requirements.txt
```

## Rust Build

From `solver/`:

```bash
cargo check --workspace
cargo test -p cfr
cargo test -p abstraction
cargo build --release -p deep_cfr --bin run_traversals
cargo build --release -p deep_cfr --bin ring_game_eval
cargo build --release -p deep_cfr --bin realtime_play
```

## Smoke Checks

From the repository root:

```bash
python -m py_compile \
  training/deep_cfr/model.py \
  training/deep_cfr/train.py \
  training/deep_cfr/reservoir.py \
  training/deep_cfr/run_deep_cfr.py \
  run_eval_suite.py
```

```bash
python run_eval_suite.py --help
python -m eval.run_league --help
```

## Training Entry Point

The main training orchestrator is:

```bash
python training/deep_cfr/run_deep_cfr.py --help
```

A full 6-max run requires compiled Rust traversal binaries, cluster assets,
and a writable `data/` directory. Use small smoke settings first before
starting a long run.

## Evaluation Entry Point

The main evaluation suite is:

```bash
python run_eval_suite.py --help
```

The suite expects a trained ONNX model and compiled `ring_game_eval` /
`realtime_play` binaries.

## Optional Debug Logs

By default, Talibus does not write debug logs. To enable best-effort JSONL
diagnostics, set:

```bash
export TALIBUS_DEBUG_LOG=debug/talibus_debug.jsonl
```

On Windows PowerShell:

```powershell
$env:TALIBUS_DEBUG_LOG = "debug\talibus_debug.jsonl"
```

