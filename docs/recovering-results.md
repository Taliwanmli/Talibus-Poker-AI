# Recovering Results

Historical long-run artifacts were generated locally in the original
development environment and are not included in this cleaned repository.

When the original machine is available, look for directories like:

```text
data/deep_cfr_6max_laptop_longrun_opt/
data/deep_cfr_6max_truth/
data/deep_cfr_6max_current_reference/
data/deep_cfr_6max_pre_long_run/
```

## Files Worth Copying

Prefer compact, reproducible evidence:

```text
metrics.json
state.json
summary.json
seat_rotation.json
opponent_comparison.json
checkpoint_progression.json
search_budget_sweep.json
realtime_opponent_comparison.json
mixed_table_seat_rotation.json
final_eval_report.json
final_eval_report.md
panel_manifest.json
divergence_gate.json
traversal_benchmark.json
league_panel_test.json
commands.txt
environment.txt
```

Place recovered results under:

```text
results/<date-or-run-name>/
```

## Files To Avoid Committing Directly

Avoid committing bulky generated artifacts:

```text
buffers/
*.onnx
*.pt
*.pth
large raw logs
```

If a checkpoint is small and essential for reproduction, it can be attached to
a GitHub Release. Larger models and buffers should use external artifact
storage.

## Minimum Publishable Result Pack

A result pack should include:

- the commit hash used for the run,
- the exact command line,
- machine/hardware notes,
- model checkpoint identifier or hash,
- random seeds,
- number of hands or panel samples,
- opponent setup,
- bb/100 with confidence interval where available,
- known limitations.

