# 6-Max Quality vs Machine Knobs

This guide separates training-quality decisions from hardware execution decisions.

## Quality Knobs (Do Not Downscale Just For One Machine)

- `--hidden-dim`
- `--bottleneck-dim`
- `--traversals`
- `--training-steps`
- `--buffer-size`
- `--batch-size`
- `--deck-samples`
- `--iterations`
- `--max-sample-reuse-per-iter`
- `--dropout-p`
- `--lr`
- `--weight-decay`
- `--adv-huber-delta`

These knobs define data quality, optimization behavior, and representational capacity.
Select them using independent holdout metrics and pilot trend evidence.

## Machine Knobs (Execution Profile Only)

- `--traversal-workers`
- `--traversal-progress-batch`
- `--traversal-seat-chunks`
- `--device`
- `--evaluator-panel-device`
- `--evaluator-panel-batch-size`

These knobs should be tuned for throughput/latency on a specific machine after quality config is chosen.

## Cloud Handoff Rule

If the chosen quality config fails local runtime/storage limits, move the same quality config to stronger hardware.

Recommended handoff thresholds (adjustable by project needs):

- projected long-run ETA exceeds local cap (for example, `72h`)
- projected disk growth exceeds local cap (for example, `100GB`)

Do not first lower quality knobs only to fit local hardware; use cloud or stronger compute when needed.

