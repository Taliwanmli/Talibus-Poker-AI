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

Compact result packs are stored under `results/<run-name>/` with files such as:

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

## Included Result Pack

`results/2026-04-03-laptop-longrun-opt/` contains a compact public summary of a
long 6-max training run and a final mixed-table simulator evaluation.

Important files:

- `summary.json`: top-level sanitized evaluation summary.
- `mixed_table_seat_rotation.json`: six 1,000-hand runs, rotating the model
  through seats 0-5 against scripted mixed opponents.
- `mixed_budget_floor.json`: one 1,000-hand run at a lower search budget.
- `metrics_summary.json`: compact training and final-iteration metrics derived
  from the original local metrics file.
- `training_state.json`: sanitized checkpoint/training state.
- `best_ring_promotion.json`: ring-promotion history from the training run.
- `model_artifacts.json`: local ONNX artifact names, sizes, and SHA-256 hashes.
- `commands.txt` and `environment.txt`: reproducibility notes.

The mixed-table seat-rotation evaluation used:

- 6 players.
- Blinds 10/20.
- Starting stack 2,000 chips.
- 200 deck samples.
- 1,000 hands per seat.
- 2,000 ms search budget.
- Scripted opponents: TAG, calling station, LAG, nit, TAG.
- Model artifact: `strategy_shared_best_ring.onnx`, not committed to Git.

Observed seat bb/100 values:

| Seat | bb/100 | Avg decision ms | Avg iterations/decision |
| ---: | -----: | --------------: | ----------------------: |
| 0 | 4539.885 | 2694.191 | 136825.822 |
| 1 | 5498.015 | 2581.715 | 193268.254 |
| 2 | 3664.615 | 3324.101 | 128664.083 |
| 3 | 5205.470 | 2810.123 | 121001.381 |
| 4 | 4923.275 | 2757.157 | 151768.604 |
| 5 | 6222.160 | 2689.477 | 158766.520 |

Average across seats: 5008.903 bb/100.

The 500 ms budget-floor run from seat 0 reported 4747.525 bb/100 with an
average decision time of 1152.724 ms.

## Interpreting Results

Large positive win rates against scripted baselines should be reported as
controlled simulator results, not as evidence of superhuman play. For 6-max
multiplayer poker, exploitability and convergence claims are not directly
equivalent to two-player zero-sum settings.

The included result pack does not include confidence intervals and uses simple
scripted opponents. Treat it as a showcase and regression reference for this
codebase, not as a general poker-strength benchmark.

