# HU Deep CFR Proof Gates

This file defines the minimum validation protocol before any long HU run or 6-max work.

## Short Validation Run (Architecture Sanity)

### Command template

`py -3 training/deep_cfr/run_deep_cfr.py --iterations 20 --traversals 4000 --strategy-every 5 --diagnostic-every 5 --exploit-samples 512 --exploit-fit-samples 512 --exploit-eval-samples 512 --h2h-hands 2000 --fixed-eval-seed 12345`

### Required checks (all must pass)

- Strategy sample files are produced every iteration and strategy reservoir grows over time.
- Both advantage reservoirs (`p0`, `p1`) grow over time.
- Diagnostics run on fixed seeds (same panel each checkpoint).
- Strategy metrics are present (`exploitability_strategy`, `h2h_strategy_vs_tag`, `h2h_strategy_vs_random`).
- No catastrophic strategy failure (for example, persistent extreme negative bb/100 against random over multiple checkpoints).

### Go/No-Go

- **Go**: all checks pass and metric deltas are numerically stable enough to compare checkpoints.
- **No-Go**: any check fails, or metrics are structurally missing/invalid.

## Medium Validation Run (HU Proof Run)

### Command template

`py -3 training/deep_cfr/run_deep_cfr.py --iterations 100 --traversals 10000 --strategy-every 10 --diagnostic-every 10 --exploit-samples 1000 --exploit-fit-samples 1000 --exploit-eval-samples 1000 --h2h-hands 10000 --fixed-eval-seed 12345`

### Required checks (all must pass)

- At least 3 diagnostic checkpoints exist for trend comparison.
- Strategy exploitability does not regress monotonically across checkpoints.
- Strategy vs TAG and strategy vs random do not show persistent collapse.
- Advantage smoke exploitability is finite and tracks as a secondary signal.

### Go/No-Go

- **Go**: repeatable improvement or at least consistent non-collapse on fixed panels.
- **No-Go**: unstable/choppy panels with no reproducible checkpoint improvement.

## Escalation Rule

Only proceed to long HU training (200+ iterations) or 6-max preparation if both short and medium gates are **Go**.

## 6-Max Note

For the multiplayer structural validation protocol, use `training/deep_cfr/PROOF_GATES_6MAX.md`.

