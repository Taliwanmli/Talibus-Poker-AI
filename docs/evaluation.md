# Evaluation

Talibus uses controlled simulator evaluation rather than real-money play.

The current evaluation harness supports:

- seat rotation across six positions,
- scripted opponent comparison,
- checkpoint progression,
- model-only policy evaluation,
- real-time search budget sweeps,
- real-time search against different opponent styles,
- mixed-table evaluation.

Scripted baselines include TAG, LAG, nit, and calling-station-like policies.

## Main Entry Point

```bash
python run_eval_suite.py --help
```

The suite expects:

- compiled Rust binaries under `solver/target/release`,
- a trained ONNX model or checkpoint,
- abstraction clusters under `checkpoints/nlhe_clusters`,
- a writable `data/` directory for generated outputs.

## Expected Result Files

Compact result packs should be stored under `results/<run-name>/` with files
such as:

```text
summary.json
seat_rotation.json
opponent_comparison.json
checkpoint_progression.json
search_budget_sweep.json
realtime_opponent_comparison.json
mixed_table_seat_rotation.json
commands.txt
environment.txt
```

Each published result pack should include:

- commit hash,
- model/checkpoint path or hash,
- command line,
- number of hands,
- random seeds,
- opponent setup,
- bb/100 and confidence interval where available,
- known limitations.

## Interpreting Results

Large positive win rates against scripted baselines should be reported as
controlled simulator results, not as evidence of superhuman play. For 6-max
multiplayer poker, exploitability and convergence claims are not directly
equivalent to two-player zero-sum settings.

