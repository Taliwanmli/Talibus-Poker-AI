# Talibus

Talibus is an AI-assisted research prototype for six-player no-limit Texas
Hold'em. It combines a Rust poker simulation/runtime stack with a
Deep-CFR-style neural training pipeline, ONNX model deployment, scripted
opponent evaluation, and depth-limited real-time search.

This repository is a cleaned research release extracted from a broader
exploratory development repo. The original repo included abandoned approaches,
generated training artifacts, frontend experiments, and live-client tooling.
This release keeps the core research system.

## What This Contains

- Rust NLHE game simulation and solver crates under `solver/`.
- Deep-CFR-style traversal, sample encoding, ONNX policy inference, and
  real-time search in `solver/deep_cfr`.
- Python/PyTorch model training, reservoir-style sample management, and
  long-run orchestration in `training/deep_cfr`.
- Scripted opponent and evaluation tooling, including TAG/LAG/nit/calling
  station baselines.
- 6-max evaluation harnesses for seat rotation, opponent comparison,
  checkpoint progression, mixed-table testing, and search-budget sweeps.
- Prebuilt abstraction cluster assets under `checkpoints/nlhe_clusters`.

## What This Does Not Claim

Talibus is not presented as a solved poker agent, a superhuman poker system, or
a proof that Deep CFR converges in multiplayer poker. It is a research and
systems-engineering prototype for experimenting with neural CFR-style training
and runtime search in an abstracted 6-max NLHE environment.

Historical long-run model checkpoints and result artifacts are not included in
this cleaned repo. Reproducible compact evaluations should be added under
`results/` as they are recovered or rerun.

## Architecture

The system has four main layers:

1. `solver/game`: Rust NLHE game mechanics, betting, pots, showdown, and legal
   action handling.
2. `solver/cfr`: game-model wrappers, information-set construction, card/board
   abstraction, and fixed action-slot mapping.
3. `solver/deep_cfr`: traversal, feature encoding, ONNX policy inference,
   scripted baselines, ring evaluation, and real-time search.
4. `training/deep_cfr`: PyTorch networks, sample buffers, training loops,
   evaluation orchestration, and model export.

See `docs/architecture.md` for more detail.

## Repository Layout

```text
checkpoints/nlhe_clusters/     Precomputed NLHE abstraction cluster assets
docs/                          Public-facing project documentation
eval/                          Restored evaluation package used by training scripts
results/                       Placeholder for compact reproducible result packs
solver/                        Rust workspace for game, CFR, and Deep CFR runtime
training/deep_cfr/             Python/PyTorch training and long-run orchestration
run_eval_suite.py              Comprehensive local 6-max evaluation runner
deep_cfr_watchdog.ps1          Windows long-run helper used during development
```

## Quick Start

Install Rust and Python 3.10+ first.

```bash
cd solver
cargo check --workspace
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/deep_cfr/requirements.txt
python -m py_compile training/deep_cfr/model.py training/deep_cfr/train.py training/deep_cfr/run_deep_cfr.py
```

The full training and evaluation runs require generated artifacts under `data/`
and trained ONNX checkpoints. Those are intentionally not committed here.

## Evaluation

The main display evaluation entry point is:

```bash
python run_eval_suite.py --help
```

It expects compiled Rust binaries and a trained model checkpoint. See
`docs/evaluation.md` for the intended result structure and how to add recovered
or rerun evaluation outputs.

## Status

This repo is currently suitable as a cleaned implementation and documentation
base. The next required step is to recover or rerun compact 6-max evaluations
and add result summaries under `results/`.

