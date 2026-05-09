# Results

This directory is reserved for compact, reproducible result packs.

The original exploratory development repo generated local artifacts under
`data/deep_cfr_6max_*`, including evaluation JSON files, panel reports,
checkpoint reports, and model outputs. Those generated artifacts are not
included in this cleaned repo.

When results are recovered or rerun, add them here in timestamped directories,
for example:

```text
results/2026-05-compact-6max/
  summary.json
  commands.txt
  environment.txt
  seat_rotation.json
  opponent_comparison.json
  search_budget_sweep.json
```

Do not commit large training buffers or raw model checkpoints directly unless
they are intentionally small. Use GitHub Releases or external artifact storage
for larger checkpoints.

