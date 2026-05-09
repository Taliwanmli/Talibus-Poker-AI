# 6-Max Deep CFR Proof Gates

This file defines the pre-long-run validation protocol for 6-max scale selection.

## Goal

Choose quality scale by independent holdout evidence first, then decide local vs cloud execution without changing the chosen quality config.

## Gate 1: Build + Contract

### Required checks

- `cargo check --workspace` passes.
- `cargo test -p deep_cfr encoding` passes.
- `cargo test -p abstraction equity` passes.
- `py -3 -m py_compile training/deep_cfr/run_deep_cfr.py training/deep_cfr/train.py eval/deep_cfr_panel.py eval/run_league.py`.

### Go/No-Go

- **Go**: all checks pass with no sample/action contract mismatch.
- **No-Go**: any build/test/contract failure.

## Gate 2: Independent Panel Gate

### Command template

`py -3 training/deep_cfr/build_eval_panels.py --model-onnx data/deep_cfr_6max_tier_c/models/advantage_shared.onnx --output-dir data/deep_cfr_6max_pre_long_run/panels --required-seats 0,1,2,3,4,5 --min-seat-samples 1 --max-dominant-seat-fraction 0.70 --report-json data/deep_cfr_6max_pre_long_run/panel_manifest.json`

### Required checks

- Screen and holdout panel are generated from different seeds.
- Panel manifest records per-seat counts for both panels.
- Coverage gate passes for both panels:
  - no required seat is empty,
  - required seats satisfy min samples,
  - dominant seat fraction is below configured threshold.

### Go/No-Go

- **Go**: `panel_manifest.json` status is `ok`.
- **No-Go**: either panel fails coverage gate.

## Gate 3: Divergence Gate On Screen Panel

### Command template

`py -3 training/deep_cfr/run_divergence_gate.py --panel data/deep_cfr_6max_pre_long_run/panels/screen_panel_strategy.bin --sample-type strategy --model random=... --model baseline=... --model candidate=... --report-json data/deep_cfr_6max_pre_long_run/divergence_gate.json`

### Required checks

- Compare at least three checkpoints (for example random/init, current baseline, larger candidate).
- All compared checkpoints produce distinct policy fingerprints on the same screen panel.

### Go/No-Go

- **Go**: `gate_passed=true`.
- **No-Go**: any pair has identical fingerprint hash.

## Gate 4: Scale Search Gate (Quality First)

### Command template

`py -3 training/deep_cfr/run_scale_search.py --work-root data/deep_cfr_6max_pre_long_run/scale_search --screen-panel ... --holdout-panel ... --report-json data/deep_cfr_6max_pre_long_run/scale_search/report.json --candidate ...`

### Required checks

- Run short candidate runs and evaluate each candidate on both screen and independent holdout.
- Capture at least:
  - `top1_match_rate`,
  - `cross_entropy`,
  - `masked_l1`,
  - fingerprint hash,
  - seat breakdown/coverage,
  - iteration/traversal wall time,
  - disk growth per iteration.
- Rank by holdout quality first; runtime/disk only breaks ties.

### Go/No-Go

- **Go**: at least one candidate reports successful holdout evaluation and becomes recommended.
- **No-Go**: no valid candidate/evaluation result.

## Gate 5: Pilot Gate + Runtime Gate

### Command templates

- Pilot trend:
  `py -3 training/deep_cfr/run_pilot_gate.py --scale-search-report ... --screen-panel ... --holdout-panel ... --work-dir ... --report-json ...`
- Machine benchmark:
  `py -3 training/deep_cfr/benchmark_traversal_utilization.py --work-root ... --report-json ...`
- Final handoff:
  `py -3 training/deep_cfr/build_pre_long_run_handoff.py --panel-manifest ... --divergence-report ... --scale-search-report ... --benchmark-report ... --pilot-report ... --output-json ...`

### Required checks

- Pilot report shows improving or stable non-regressing holdout trend across checkpoints.
- Runtime profile is predictable enough to project ETA and disk budget.
- Final handoff manifest explicitly separates:
  - quality knobs,
  - machine knobs,
  - cloud handoff condition for unchanged chosen quality config.

### Go/No-Go

- **Go**: pilot gate and runtime gate both pass.
- **No-Go**: unstable holdout trend or non-actionable runtime projection.

## Ready For Long-Run Decision

Project is run-ready only when all gates pass:

- `Pass Panel Gate`
- `Pass Divergence Gate`
- `Pass Scale Gate`
- `Pass Pilot Gate`
- `Pass Runtime Gate`

At that point the workflow stops at a pre-long-run handoff package and requires a separate explicit approval to start any long run.
