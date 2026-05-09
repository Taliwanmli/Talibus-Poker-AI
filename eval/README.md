# Offline EV Harness (`eval/`)

This harness runs large-volume **offline** NLHE simulations and evaluates the WiPoker decision engine as Hero.

It does **not** integrate with any live poker platform.

## Contents

- `policy_worker.mjs`: persistent Node JSONL worker that calls `getRecommendation(street, handState)`.
- `run_league.py`: Python league runner using a poker simulator (`pokerenv`) for 2-6 players.
- `requirements.txt`: Python dependencies for the harness.

## Setup

1. Build the engine bundle used by the worker:

```bash
npm run eval:build
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r eval/requirements.txt
```

## Run

### 10k hands (default 6-max, 100bb, argmax hero strategy)

```bash
python3 eval/run_league.py --hands 10000 --seed 42
```

### Fast smoke eval (compact artifacts + retention)

```bash
python3 eval/run_smoke_eval.py --hands 1000 --seed 42
```

Writes:
- `reports/report_smoke.json`
- `reports/metrics.json`
- Per-run archive in `reports/runs/<run_id>/`

### Local autonomous Phase 0 loop (Windows)

Persistent local store path:
- `%LOCALAPPDATA%\WiPoker\phase0_store\`

Canonical launcher command:

```powershell
npm run phase0:local
```

#### Quick start (PowerShell)

Temporary (current terminal only):

```powershell
$env:PIO_EXE = "C:\path\to\PioSOLVER3-edge.exe"
.\scripts\run_phase0_local.ps1
```

Permanent (User env var):

```powershell
[Environment]::SetEnvironmentVariable("PIO_EXE", "C:\path\to\PioSOLVER3-edge.exe", "User")
```

After setting permanently, open a new terminal and run:

```powershell
.\scripts\run_phase0_local.ps1
```

Default launcher profile (`canonical_overnight`) now injects a safe baseline:
- `--mode smoke`
- `--eval_hands 8000`
- `--sleep_seconds 0`
- `--promote_top_boards 10`
- `--max_export_spots 120`
- `--max_export_minutes 25`
- `--stall_threshold 2`
- `--keep_last_cycles 30`
- `--max_gb 80`
- `--hero_mode worker`
- `--preflop_selection_mode sample`
- `--policy_selection_mode hybrid`
- `--policy_exact_only`
- `--policy_rng_seed 42`
- `--preflop_rng_seed 42`

Any explicit CLI args you pass still override these defaults.

#### Quick start (CMD)

Temporary (current terminal only):

```bat
set "PIO_EXE=C:\path\to\PioSOLVER3-edge.exe"
scripts\run_phase0_local.bat
```

Permanent (User env var):

```bat
setx PIO_EXE "C:\path\to\PioSOLVER3-edge.exe"
```

After setting permanently, open a new terminal and run:

```bat
scripts\run_phase0_local.bat
```

Run exactly 3 cycles:

```powershell
.\scripts\run_phase0_local.ps1 --max_cycles 3
```

### Pio command-line probe (Windows, spaced paths)

Use this to verify PioSolver CLI I/O and END-delimited parsing without opening GUI:

```powershell
npm run pio:probe -- --PioExe "C:\my space\Computing\Casual Coding stuff\PioSolver 3.3-EN\PioSOLVER\PioSOLVER3-edge.exe"
```

The probe sends `set_end_string END`, `is_ready`, and terminates with `exit` (not `quit`).

Optional tree check (also probes `show_hand_order` + `show_node root`):

```powershell
npm run pio:probe -- --PioExe "C:\my space\Computing\Casual Coding stuff\PioSolver 3.3-EN\PioSOLVER\PioSOLVER3-edge.exe" --TreePath "C:\path\to\tree.cfr" --TimeoutSec 20
```

Probe logs are written to:
- `eval/reports/pio_probe/pio_probe_<timestamp>.log`

### 100k hands

```bash
python3 eval/run_league.py --hands 100000 --seed 42 --progress-every 5000
```

### Matrix experiments (seeds x opponents)

Run multiple seeds/opponents and aggregate results:

```bash
python3 eval/run_league.py \
  --match_format hu \
  --seeds 42,43,44 \
  --hands_per_seed 100000 \
  --opponents station,nit,lag
```

Per-run JSON reports are saved to `eval/runs/<opponent>_<seed>.json`, and aggregate output is saved to `eval/runs/aggregate_summary.json`.

### Optional safety cap per hand

```bash
python3 eval/run_league.py --hands 10000 --max-actions-per-hand 500
```

### Hero policy modes

`--hero_mode` supports:

- `argmax` (default, evaluation mode)
- `sample`
- `always_fold` (check when possible, else fold)
- `random` (uniform random legal action)

Examples:

```bash
python3 eval/run_league.py --hands 10000 --hero_mode argmax
python3 eval/run_league.py --hands 10000 --hero_mode sample
```

### Rust blueprint policy worker

`run_league.py` accepts any worker command via `--policy-cmd`.

Default (Node worker):

```bash
python3 eval/run_league.py --hands 10000 --policy-cmd "node eval/policy_worker.mjs"
```

Rust blueprint worker:

```bash
python3 eval/run_league.py --hands 10000 --hero_mode worker \
  --policy-cmd "cargo run --manifest-path solver/Cargo.toml -p player --bin blueprint_policy_worker --release"
```

Smoke eval through the same Rust worker path:

```bash
python3 eval/run_smoke_eval.py --hands 200 --mode smoke --hero_mode worker \
  --policy_source blueprint \
  --blueprint_file solver/checkpoints/kuhn_linear_smoke/latest.blueprint \
  --blueprint_mode argmax
```

Blueprint worker env vars:
- `WIPOKER_BLUEPRINT_FILE` (default `checkpoints/latest.blueprint`)
- `WIPOKER_BLUEPRINT_MODE` (`sample` or `argmax`)
- `WIPOKER_BLUEPRINT_SEED` (default `17`)
- `WIPOKER_BLUEPRINT_STRICT` (`1` to fail if blueprint file cannot be loaded)

Convert a CFR checkpoint into a blueprint file:

```bash
cargo run --manifest-path solver/Cargo.toml -p player --bin export_blueprint_from_checkpoint --release -- \
  --checkpoint checkpoints/kuhn_linear/latest.ckpt \
  --out checkpoints/latest.blueprint
```

### Preflop sizing adapter flags (Hero only)

Preflop raise translation uses deterministic sizes in bb (raise-to semantics):

- `--open_size` (default `2.5`)
- `--threebet_size` (default `9`)
- `--fourbet_size` (default `22`)

Examples:

```bash
python3 eval/run_league.py --hands 10000 --hero_mode argmax --open_size 2.5 --threebet_size 9 --fourbet_size 22
```

Optional assertion to catch bad preflop translation:

```bash
python3 eval/run_league.py --hands 1000 --hero_mode argmax --assert_large_preflop_raises_allin
```

### Decision tracing

Trace first `N` hands to JSONL:

```bash
python3 eval/run_league.py --hands 10000 --hero_mode argmax --trace_hands 200
```

Output file defaults to `eval/trace.jsonl`.

### JSON report export

Write all summary/diagnostic metrics to JSON:

```bash
python3 eval/run_league.py --hands 10000 --hero_mode argmax --report_json eval/reports/report.json
```

### Biggest loser hand dump

Dump the `K` biggest losing hands with full action sequence and hero decision records:

```bash
python3 eval/run_league.py --hands 10000 --hero_mode argmax --dump_biggest_losers 20
```

Default output file: `eval/biggest_losers.jsonl`.

### With rake (percent + cap)

```bash
python3 eval/run_league.py \
  --hands 20000 \
  --rake-mode percent \
  --rake-percent 0.05 \
  --rake-cap-bb 3
```

### With fixed rake

```bash
python3 eval/run_league.py --hands 20000 --rake-mode fixed --rake-fixed-bb 0.15
```

`--rake_bb_per_hand` is an alias for fixed rake:

```bash
python3 eval/run_league.py --hands 20000 --rake_bb_per_hand 0.15
```

## Opponent Baselines

Opponent seats are controlled by simple baseline bots:

- `call_station`
- `nit`
- `lag`

Default lineup rotates these bot types. You can override:

```bash
python3 eval/run_league.py --opponent-lineup call_station,nit,lag,lag
```

## Output

- CSV file: `eval/results.csv`
- Columns:
  - `hand_index`
  - `bb_delta` (rake-adjusted)
  - `cumulative_bb` (rake-adjusted cumulative)
  - `bb_delta_raw`
  - `cumulative_bb_raw`
  - `rake_bb`

End-of-run summary includes:

- `bb/100`
- per-hand standard deviation
- standard error
- 95% confidence interval
- raw and rake-adjusted results
- engine status counts (`ok`, `missing_fields`, `unavailable`)
- `fallback_count` (safe fallback uses when engine status is not `ok`)
- per-street hero decision diagnostics (`decisions`, `ok`, `unavailable`, `fallbacks`)
- EV decomposition by forced/preflop/flop/turn/river attribution
- hero style metrics (`VPIP`, `PFR`, `3bet`, AF, fold-to-bet, showdown when VPIP)
- action translation audit (bet fractions, clipped/adjusted counts)
- preflop sizing safety counters (`preflop_raise_clamped`, `preflop_raise_converted_to_allin`, `illegal_action_avoided`)
- preflop leak report by node category (`count`, `total_bb`, `avg_bb`, fold/call/raise%)
- raise outlier report (`>20bb`, `>40bb`, `>80bb`, top 10 largest raises, clipped/adjusted by context)
- bet sizing distribution buckets for `BET33`/`BET75` and forced-to-halfpot counts
- effective small/large bet size fractions and MDF guardrail analysis targets

### Safe fallback behavior

When engine status is not `ok`, hero action is forced to:

- facing `FACING_BET` / `FACING_RAISE` / `FACING_CHECK_RAISE`: `FOLD` if legal else `CALL`
- facing `NO_BET`: `CHECK` if legal else `FOLD`

This avoids noisy baseline contamination from arbitrary fallback bots.

### Baseline sanity helper

Run all three baseline modes side-by-side:

```bash
python3 eval/run_baselines.py --hands 10000 --seed 42
```

## Runtime notes

- A persistent Node worker is used to avoid per-decision startup cost.
- 100k hands can take significant time depending on CPU and policy fallback frequency.
- Use `--progress-every` to tune console output cadence.
- The runner applies a runtime `pokerenv` compatibility patch to avoid known infinite-loop edge cases.

## Strategy Artifact Paths (repo-relative defaults)

Strategy lookup pipeline scripts now default to project-local paths:

- `eval/generated_pio_exports/` for generated per-board Pio exports (`.jsonl`, `.cfr`)
- `eval/strategy_store/` for merged StrategyStore outputs (`*.jsonl`)
- `eval/reports/` for report JSON outputs used by store diagnostics
- `eval/missing/` for missing spot summaries
- `eval/tmp/` for temporary exporter scratch files

Examples:

```bash
python3 eval/summarize_missing_spots.py --report_json eval/reports/report_store.json --out eval/missing/missing_spots_top.txt --top_n 2000
python3 eval/build_pio_store_from_missing.py --missing_file eval/missing/missing_spots_top.txt --out_store eval/strategy_store/pio_store_from_missing.jsonl --max_boards 50 --max_spots_per_board 200
python3 eval/auto_pio_store_loop.py --hands 20000 --seed 42 --workers 8 --iters 1
```

## Determinism

Use `--seed` for deterministic runs:

```bash
python3 eval/run_league.py --hands 10000 --seed 1337
```

## Troubleshooting

- `policy worker exited unexpectedly`:
  - Run `npm run eval:build`.
  - Confirm `node eval/policy_worker.mjs` starts.
- `unknown bot`:
  - Use only `call_station`, `nit`, `lag` in `--opponent-lineup`.
- If simulator package updates break behavior:
  - Recreate venv and reinstall from `eval/requirements.txt`.
