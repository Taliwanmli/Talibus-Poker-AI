# Setup

This document describes the development setup for Talibus.

## Prerequisites

- Rust stable with Cargo.
- Python 3.10 or newer.
- A C/C++ toolchain suitable for Rust native dependencies.
- Enough disk space for generated `data/` artifacts if running training.
- Optional CUDA-capable GPU for larger PyTorch training runs.

The Rust runtime uses `ort` 2.0.0-rc.11 for ONNX inference. Use ONNX Runtime
1.23.x or newer, and make the shared library available to the binaries if it
is not on the platform library path. For example, set `ORT_DYLIB_PATH` to the
ONNX Runtime shared library before running binaries that load `.onnx` models.

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

The repository includes a trained ONNX strategy model under
`artifacts/models/talibus-6max-longrun-opt-v1/`. Full long-run
training/evaluation still requires generated local artifacts and substantial
compute, but lightweight source and model smoke checks are available.

The checks below are the closest lightweight verification path. They verify
that key Python modules parse, command-line entry points are discoverable, and
the fast evaluation tests pass without launching training jobs or creating
large artifacts.

From the repository root, after installing the Python requirements:

```bash
python3 -m py_compile \
  training/deep_cfr/model.py \
  training/deep_cfr/train.py \
  training/deep_cfr/reservoir.py \
  training/deep_cfr/run_deep_cfr.py \
  run_eval_suite.py
```

Verifies that the selected training/evaluation Python files parse correctly.

```bash
python3 run_eval_suite.py --help
```

Verifies that the structured evaluation-suite CLI is importable and exposes its
options without requiring a model artifact.

```bash
python3 -m eval.run_league --help
```

Verifies that the offline evaluation CLI is importable and exposes its options.
Some environments may print a Gym deprecation warning; the smoke check still
passes if the command exits successfully.

```bash
python3 -m unittest discover eval
```

Runs the fast Python evaluation tests. These tests do not require trained ONNX
models or long generated training buffers.

On systems where `python` points to Python 3, use `python` instead of
`python3`.

Optional Rust verification, if Cargo is installed:

```bash
cd solver
cargo check --workspace
```

This checks the Rust workspace without running training or evaluation. It may
take longer on a first build because Cargo has to download and compile
dependencies.

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
`realtime_play` binaries. The released strategy model is:

```text
artifacts/models/talibus-6max-longrun-opt-v1/strategy_shared_best_ring.onnx
```

After building the Rust binaries, run a short scripted-opponent model smoke
test from the repository root:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
solver/target/release/ring_game_eval \
  --model artifacts/models/talibus-6max-longrun-opt-v1/strategy_shared_best_ring.onnx \
  --policy strategy \
  --cluster-dir checkpoints/nlhe_clusters \
  --num-players 6 \
  --hands 100 \
  --opponent tag
```

On Windows PowerShell, set `ORT_DYLIB_PATH` to the compatible
`onnxruntime.dll` if the DLL is not already discoverable:

```powershell
$env:ORT_DYLIB_PATH = "<path-to-onnxruntime.dll>"
.\solver\target\release\ring_game_eval.exe `
  --model .\artifacts\models\talibus-6max-longrun-opt-v1\strategy_shared_best_ring.onnx `
  --policy strategy `
  --cluster-dir .\checkpoints\nlhe_clusters `
  --num-players 6 `
  --hands 100 `
  --opponent tag
```

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
