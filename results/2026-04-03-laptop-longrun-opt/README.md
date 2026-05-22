# 2026-04-03 Laptop Longrun Opt

This result pack is a compact public summary of a Talibus 6-max training run
and final controlled simulator evaluation.

## Scope

- Training source run: `deep_cfr_6max_laptop_longrun_opt`.
- Training mode: shared multiseat.
- Completed iterations: 211.
- Players: 6.
- Blinds: 10/20.
- Starting stack: 2,000 chips.
- Evaluation type: simulator evaluation against scripted opponents.
- Model artifact: `strategy_shared_best_ring.onnx`.

The ONNX and PyTorch model files are not committed to Git. See
`model_artifacts.json` for file names, sizes, timestamps, and SHA-256 hashes.

## Files

- `summary.json`: top-level sanitized result summary.
- `mixed_table_seat_rotation.json`: six seat-rotation runs at a 2,000 ms search
  budget.
- `mixed_budget_floor.json`: one seat-0 run at a 500 ms search budget.
- `metrics_summary.json`: compact training metrics derived from the local
  metrics file.
- `training_state.json`: sanitized training state.
- `best_ring_promotion.json`: ring-promotion history.
- `model_artifacts.json`: model artifact metadata and hashes.
- `commands.txt`: command/config notes.
- `environment.txt`: machine and dependency notes.

## Mixed-Table Seat Rotation

The final mixed-table evaluation used 1,000 hands per seat, 200 deck samples,
a 2,000 ms search budget, and scripted opponents in the order TAG, calling
station, LAG, nit, TAG.

| Seat | bb/100 | Avg decision ms | Avg iterations/decision |
| ---: | -----: | --------------: | ----------------------: |
| 0 | 4539.885 | 2694.191 | 136825.822 |
| 1 | 5498.015 | 2581.715 | 193268.254 |
| 2 | 3664.615 | 3324.101 | 128664.083 |
| 3 | 5205.470 | 2810.123 | 121001.381 |
| 4 | 4923.275 | 2757.157 | 151768.604 |
| 5 | 6222.160 | 2689.477 | 158766.520 |

Average across seats: 5008.903 bb/100.

## Budget-Floor Run

The 500 ms budget-floor run from seat 0 reported 4747.525 bb/100, with an
average decision time of 1152.724 ms and 35459.144 average iterations per
decision.

## Interpretation

These are controlled simulator results against simple scripted opponents. They
are useful as a showcase and regression reference for this codebase. They do
not imply superhuman play, real-money performance, or a convergence proof for
multiplayer poker.
