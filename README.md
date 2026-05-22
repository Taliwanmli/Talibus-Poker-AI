# Talibus

Talibus is a research prototype for six-player no-limit Texas Hold'em. It
combines a Rust poker simulation/runtime stack with a Deep-CFR-style neural
training pipeline, PyTorch model training, ONNX deployment, scripted opponent
evaluation, and depth-limited real-time search.

The project is intended for research, experimentation, and systems engineering
work around imperfect-information games. It is not a production poker bot and
does not claim solved, superhuman, or converged multiplayer poker play.

## Features

- Rust NLHE game mechanics, betting, pots, showdown, and legal-action handling.
- Imperfect-information wrappers and fixed action-slot abstraction.
- Deep-CFR-style traversal and sample generation.
- PyTorch advantage/strategy model training.
- Reservoir and disk-backed training buffers.
- ONNX export and Rust-side ONNX inference.
- Scripted TAG, LAG, nit, and calling-station baseline policies.
- Model-only ring-game evaluation.
- Depth-limited real-time search over a neural policy.
- Evaluation harnesses for seat rotation, checkpoint progression, opponent
  comparison, mixed tables, and search-budget sweeps.
- Compact public result packs under `results/`.

## Repository Layout

```text
checkpoints/nlhe_clusters/     Precomputed abstraction cluster assets
docs/                          Architecture, setup, evaluation, and limitations
eval/                          Python evaluation helpers and regression tests
results/                       Compact public result packs
solver/                        Rust workspace for game, CFR, and runtime code
training/deep_cfr/             PyTorch training and long-run orchestration
run_eval_suite.py              Structured 6-max evaluation runner
deep_cfr_watchdog.ps1          Windows helper for long training runs
```

Large generated buffers, raw logs, PyTorch checkpoints, and ONNX model binaries
are not committed. The included result pack records model metadata and SHA-256
hashes for the local artifacts used to produce the published evaluation.

## Quick Start

Install Rust stable, Python 3.10 or newer, and a native build toolchain. Then:

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

Build and test the Rust workspace:

```bash
cd solver
cargo check --workspace
cargo test -p cfr
cargo test -p abstraction
cargo build --release -p deep_cfr --bin ring_game_eval --bin realtime_play
```

Run Python checks from the repository root:

```bash
python -m compileall -q .
python -m unittest discover eval
python run_eval_suite.py --help
```

See `docs/setup.md` for platform notes, ONNX Runtime details, and the longer
verification checklist.

## Training

The main training entry point is:

```bash
python training/deep_cfr/run_deep_cfr.py --help
```

A full 6-max run requires the Rust traversal binary, abstraction clusters, and
substantial generated data under `data/`. Start with small smoke settings before
launching long runs. Long-run artifacts can be many gigabytes and are excluded
by `.gitignore`.

## Evaluation

The main structured evaluation entry point is:

```bash
python run_eval_suite.py --help
```

The suite expects compiled Rust binaries and a trained ONNX strategy model.
Evaluation outputs are controlled simulator measurements against scripted
baselines. They are useful for regression testing and research comparison, but
they are not evidence of real-money performance or human-level strength.

Included public result pack:

- `results/2026-04-03-laptop-longrun-opt/`

That pack contains a sanitized summary of a 6-max mixed-table simulator
evaluation using the local `strategy_shared_best_ring.onnx` artifact. The six
seat-rotation runs used 1,000 hands per seat, a 2,000 ms search budget, 200 deck
samples, and scripted mixed opponents: TAG, calling station, LAG, nit, TAG.
The reported seat bb/100 values range from 3664.615 to 6222.160, averaging
5008.903 bb/100 across seats. These large values are specific to the scripted
simulator setup and should be interpreted only in that context.

See `docs/evaluation.md` for result-file meanings and interpretation guidance.

## Documentation

- `docs/architecture.md`: component map and runtime/training flow.
- `docs/setup.md`: installation, checks, and common commands.
- `docs/evaluation.md`: evaluation harnesses and published result pack.
- `docs/limitations.md`: scope, caveats, and claims this project does not make.
- `training/deep_cfr/REMOTE_WINDOWS_RUNBOOK.md`: generalized remote Windows
  training workflow for larger experiments.

## License

This project is released under the MIT License. See `LICENSE`.
