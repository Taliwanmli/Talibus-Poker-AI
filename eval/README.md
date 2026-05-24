# Evaluation Package

This package contains the Python evaluation utilities that are still required
by the Deep CFR training and final-evaluation scripts.

## Included Files

- `run_league.py`: controlled simulator evaluation entry point. The Deep CFR
  training scripts use its `deep_cfr_panel` and `deep_cfr_checkpoint` backends.
- `run_league_parallel.py`: shard/merge helper for larger evaluation runs.
- `deep_cfr_panel.py`: evaluates trained models against saved binary panel
  samples.
- `deep_cfr_checkpoint_worker.py`: loads Deep CFR checkpoints for gameplay
  evaluation.
- `test_*.py`: focused tests for action translation, worker-slot contracts, and
  report merging.

Older Pio/export/front-end helpers from the development repo were intentionally
left out of this cleaned release.

## Typical Use

Most users should call the higher-level scripts instead of invoking this
package directly:

```bash
python run_eval_suite.py --help
python training/deep_cfr/run_final_eval.py --help
```

Direct panel evaluation is available through:

```bash
python -m eval.run_league \
  --eval-backend deep_cfr_panel \
  --match-format 6max \
  --deep-cfr-model-state path/to/model.pt \
  --deep-cfr-samples path/to/panel.bin \
  --report-json results/example/panel_report.json
```

Direct gameplay evaluation from a checkpoint is available through:

```bash
python -m eval.run_league \
  --eval-backend deep_cfr_checkpoint \
  --match-format 6max \
  --deep-cfr-model-state path/to/model.pt \
  --hands 10000 \
  --report-json results/example/gameplay_report.json
```

These commands require the Python dependencies listed in
`training/deep_cfr/requirements.txt` and `eval/requirements.txt`.
