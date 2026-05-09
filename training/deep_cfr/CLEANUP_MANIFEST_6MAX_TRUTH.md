# 6-Max Truth Cleanup Manifest

Generated during the "6-Max Truthful Evaluation And Utilization" implementation.

## Scope

- Repo-local generated artifacts under `data/`, `solver/target/`, and eval temp outputs.
- External project cache paths:
  - `%LOCALAPPDATA%/WiPoker/phase0_store`
  - `%LOCALAPPDATA%/WiPoker/pio_tree_cache`

## Preserve

- `data/**/metrics.json`
- `data/**/state.json`
- `data/**/holdout_runs/*.json`
- Truth-eval evidence:
  - `data/deep_cfr_6max_truth/divergence_gate.json`
  - `data/deep_cfr_6max_truth/traversal_benchmark.json`
  - `data/deep_cfr_6max_truth/traversal_benchmark_fixed_seed.json`
  - `data/deep_cfr_6max_truth/league_panel_test.json`
- Required abstraction assets:
  - `checkpoints/nlhe_clusters`
- Conservative external keep:
  - `%LOCALAPPDATA%/WiPoker/phase0_store` (large but intentionally not deleted)

## Deleted (Rebuildable)

- `data/deep_cfr_6max_tier_a/buffers` (~400.2 MB)
- `data/deep_cfr_6max_tier_b/buffers` (~600.2 MB)
- `data/deep_cfr_6max_tier_c/buffers` (~1000.4 MB)
- `data/deep_cfr_hu_repair_proof_100_10k/buffers` (~2017.0 MB)
- `data/deep_cfr_hu_holdout_decision_seed43_220/buffers` (~2017.0 MB)
- `data/deep_cfr_6max_truth/bench_runs` (~1086.2 MB)
- `data/deep_cfr_6max_truth/bench_runs_fixed_seed` (~1549.5 MB)
- `solver/target` (~811.4 MB)
- Python `__pycache__` directories under repo
- `%LOCALAPPDATA%/WiPoker/pio_tree_cache` (~11.1 MB)

Approximate reclaimed space: ~9.4 GB.

## Post-Cleanup Snapshot (Largest Remaining)

- `%LOCALAPPDATA%/WiPoker/phase0_store` ~4563.8 MB (preserved)
- `data/deep_cfr_6max_smoke` ~418.3 MB
- `data/deep_cfr_6max_truth` ~352.4 MB
- `data/deep_cfr_6max_truth_smoke` ~142.5 MB
- `data/deep_cfr_6max_tier_c` ~64.4 MB

