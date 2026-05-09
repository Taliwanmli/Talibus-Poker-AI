# Remote Windows Runbook

This is the operator manual for a rented Windows machine.

Use this after you have copied or unzipped the `remote_session` bundle onto the remote host.

## What Scale Frontier Actually Does

`scale_frontier` does not do an unlimited automatic search.

It tests the candidate configs currently encoded in the remote phase runner, compares them, and writes the winner into the handoff artifacts.

Current default behavior is a bounded ladder frontier:

- stage-1 probe tests incumbent plus one single-axis challenger per major axis (`width`, `traversals`, `training_steps`, `buffer_size`)
- stage-2 ladder auto-builds a small focused pool around the winning axis, with one interaction challenger
- stage-3 confirm reruns only top candidates (plus incumbent) to finalize the winner
- the final report written to `scale_frontier\report.json` is what pilot/handoff use

The default stage-1 file is intentionally small (`training/deep_cfr/frontier_candidates/stage1_challengers.json`):

- incumbent: `tier_c_baseline`
- width probe: `wider_model`
- traversal probe: `more_traversal`
- training-steps probe: `more_training_steps`
- RAM / reservoir probe: `larger_buffer`

This keeps runtime bounded while still giving the ladder enough signal to choose an axis and escalate intelligently.

Then `longrun` reads that chosen config and applies it directly. You do not need to manually edit hyperparameters between `handoff` and `longrun`.

Important boundary:

- if the strongest config you want is already one of the tested candidates, no code edit is needed
- if you want to test a brand new stronger config that is not in the candidate list, you must edit the candidate list before building and uploading the bundle

## Bootstrap Policy (Current)

- Python packages are installed automatically (`wheelhouse` first, then optional online fallback).
- Python itself can now be auto-installed from bundled installers when no suitable Python is found.
- NVIDIA drivers are verify-only: bootstrap checks `nvidia-smi` and CUDA usability, but does not install drivers.
- Disk and runtime prerequisites are checked early and recorded in `machine\hardware_probe.json`.

## From Your Current Machine

Right now, before renting the cloud machine, do this on your own computer.

### Recommended Local Build

Build the upload bundle with the explicit CUDA torch spec and local wheelhouse:

```powershell
Set-Location "C:\my space\Computing\Casual Coding stuff\TALIBUS"
py -3 .\training\deep_cfr\build_remote_windows_bundle.py `
  --session-root ".\data\remote_session_upload" `
  --zip-output ".\data\remote_session_upload.zip" `
  --clean `
  --build-wheelhouse `
  --python-installer "C:\path\to\python-3.11.9-amd64.exe" `
  --torch-wheel-spec "torch==2.6.0+cu124" `
  --torch-index-url "https://download.pytorch.org/whl/cu124"
```

This is the safest upload bundle because:

- the remote machine can install from the bundled wheelhouse first
- the bundle carries the exact CUDA torch build we want
- the bundle can auto-install Python if the machine is missing a suitable interpreter
- the remote machine still has online fallback if needed

If you want a smaller upload and are willing to let the remote machine download Python packages itself, omit `--build-wheelhouse`.

### What To Upload

Upload either:

- `data\remote_session_upload.zip`

or the unzipped folder:

- `data\remote_session_upload`

On the remote machine, the session root should look like one folder containing `repo`, `env`, `machine`, `logs`, `scale_frontier`, `pilot`, `handoff`, `longrun`, and `export`.

## Recommendation

Use one command to run the gated setup through `handoff`, then do one short human check, then launch the real long run in detached mode.

That is the best default for the first rented machine because:

- `bootstrap` can still reveal driver or install problems
- `benchmark` picks machine knobs from measured data
- `scale_frontier` and `pilot` are explicit gates
- the real `longrun` may take many hours, so it is worth spending 30 seconds checking the handoff first

If you really want a single unattended command from start to longrun, that is supported too.

## Before You Start

1. Log into the rented Windows machine.
2. Copy or unzip the session bundle to one folder, for example `D:\remote_session`.
3. Open PowerShell.
4. Change into the session folder:

```powershell
Set-Location "D:\remote_session"
```

All commands below assume your current directory is the session root.

## First 60 Seconds On The Cloud Machine

1. Connect to the Windows host.
2. Copy or unzip the session bundle to a local disk path, for example `D:\remote_session`.
3. Open PowerShell.
4. Run:

```powershell
Set-Location "D:\remote_session"
```

5. Confirm the repo exists:

```powershell
Test-Path .\repo\scripts\run_remote_day.ps1
```

You should see `True`.

## Fastest Safe Option

Run the full gated flow in one command:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_day.ps1 -SessionRoot .
```

What this does:

- runs `bootstrap`
- runs `benchmark`
- runs `smoke`
- runs `scale_frontier`
- runs `pilot`
- runs `handoff`
- stops before `longrun`

Then open and read:

```powershell
notepad .\handoff\pre_long_run_handoff.md
```

If the handoff looks clean, start the real run:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase longrun -UseDetached
```

During longrun, lightweight 6-max checkpoint evaluation is enabled by default:

- periodic screen checks on the fixed screen panel
- holdout confirmation only when a screen challenger improves
- best-confirmed checkpoint promotion into `longrun\work\models\strategy_shared_best_confirmed.pt` (and matching ONNX/state companions)
- checkpoint-eval state and reports under `longrun\work\checkpoint_eval`
- after a clean longrun completion, `final_eval` auto-runs and writes the heavier final evidence pack under `longrun\work\final_eval`

`run_remote_phase.ps1 -Phase longrun` enforces handoff gates by default (`ready_for_long_run_decision=true` and `requires_cloud=false`) before launching the run.

This is the recommended workflow.

## Fully Automatic Option

If you want one command that will try to go all the way through and auto-start the detached long run only when the gates pass:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_day.ps1 -SessionRoot . -StartLongRunIfReady
```

This script will:

- stop with an error if a gate fails
- stop with an error if the handoff still says the machine is not good enough
- start `longrun` in detached mode only if the handoff is clean

Use this only if you are comfortable letting it continue without pausing at the handoff review step.

## Step-By-Step Option

If you want full manual control, run these one by one:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\bootstrap_remote_windows.ps1 -SessionRoot .
```

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase benchmark
```

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase smoke
```

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase scale_frontier
```

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase pilot
```

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase handoff
```

Read the handoff:

```powershell
notepad .\handoff\pre_long_run_handoff.md
```

If it is clean:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase longrun -UseDetached
```

## How To Check Status

After detached long run starts:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase status
```

Important logs are under `.\logs`, especially:

- `longrun.supervisor.status.json` (heartbeat, attempt, restart count, progress, last ETA)
- `longrun.supervisor.summary.json` (final supervisor outcome, including `auto_final_eval_enabled` and final-eval status)
- `longrun_child_###.out.log` / `.err.log` (per-attempt trainer logs)

`status` now reports the supervisor payload directly, including heartbeat age, latest progress from `state.json`/`metrics.json`, and the current `final_eval` state.

If the remote desktop disconnects, detached longrun should keep running.

## How To Resume

If you intentionally want to start the detached long run command again with `--resume`:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase resume
```

Note:

- the longrun supervisor already does bounded auto-resume on crashes
- do not run `resume` repeatedly while a supervisor is already running

## How To Export Everything At The End

When the run is done, export the session pack:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\export_remote_session.ps1 -SessionRoot .
```

This produces a zip under `.\export`.

That zip is the structured handoff pack you should download back to your own machine.

It includes:

- `bundle_manifest.json`
- `logs`
- `panels`
- `scale_frontier`
- `pilot`
- `handoff`
- `longrun`
- `longrun\work\final_eval` inside the exported `longrun` tree
- optional debug logs when `TALIBUS_DEBUG_LOG` was set
- curated `env` files
- curated `machine` JSON reports

It intentionally does not copy the full virtualenv or scratch benchmark work directories, because those are bulky and not needed for decision reconstruction.

## What To Check In The Handoff

The important fields are:

- `ready_for_long_run_decision`
- `requires_cloud`
- the chosen config
- the runtime projection

For this remote workflow, the key go/no-go is:

- `ready_for_long_run_decision` should be `true`
- `requires_cloud` should be `false`

If either of those is wrong, do not start the long run yet.

## What I Recommend You Personally Do

On the first remote machine:

1. Use the one-command gated flow: `run_remote_day.ps1`
2. Read `handoff\pre_long_run_handoff.md`
3. Start `longrun` separately in detached mode
4. Use `status` to verify it is alive
5. Export at the end

That gives you almost all the convenience of one command, but keeps the final expensive step under a human check.

## End-To-End Checklist

1. Build `remote_session_upload.zip` on your local machine.
2. Rent the Windows host.
3. Upload or unzip the bundle onto the host.
4. Open PowerShell in the session root.
5. Run the gated day command:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_day.ps1 -SessionRoot .
```

6. Read `handoff\pre_long_run_handoff.md`.
7. If the handoff is good, start detached longrun:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase longrun -UseDetached
```

8. Check status once:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\run_remote_phase.ps1 -SessionRoot . -Phase status
```

9. Disconnect if you want. The detached run should continue.
10. Reconnect later and check `status` again.
11. When the run is done, export:

```powershell
powershell -ExecutionPolicy Bypass -File .\repo\scripts\export_remote_session.ps1 -SessionRoot .
```

12. Download the newest zip from `.\export`.
