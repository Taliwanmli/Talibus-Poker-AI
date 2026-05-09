param(
    [int]$TargetIterations = 200,
    [int]$RestartDelaySeconds = 10
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$workDir = "data/deep_cfr_6max_laptop_longrun_opt"
$statePath = Join-Path $repoRoot "$workDir/state.json"

function Get-CompletedIterations {
    param([string]$Path)

    if (-not (Test-Path -Path $Path)) {
        return -1
    }

    try {
        $state = Get-Content -Path $Path -Raw | ConvertFrom-Json
        if ($null -eq $state.completed_iterations) {
            return -1
        }
        return [int]$state.completed_iterations
    }
    catch {
        Write-Host "[watchdog] warning: failed to parse state file '$Path': $($_.Exception.Message)" -ForegroundColor Yellow
        return -1
    }
}

function Write-WatchdogLine {
    param(
        [string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

$runArgs = @(
    "-3",
    "training/deep_cfr/run_deep_cfr.py",
    "--iterations", $TargetIterations,
    "--traversals", "4500",
    "--training-steps", "6000",
    "--batch-size", "4096",
    "--lr", "0.0005",
    "--weight-decay", "0.0001",
    "--dropout-p", "0.1",
    "--hidden-dim", "1024",
    "--bottleneck-dim", "512",
    "--buffer-size", "4000000",
    "--num-players", "6",
    "--starting-stack", "2000",
    "--small-blind", "10",
    "--big-blind", "20",
    "--seed", "42",
    "--deck-samples", "200",
    "--strategy-every", "1",
    "--diagnostic-every", "25",
    "--checkpoint-eval-every", "5",
    "--checkpoint-eval-screen-panel-file", "data/deep_cfr_6max_current_reference/panels/screen_panel_strategy.bin",
    "--checkpoint-eval-holdout-panel-file", "data/deep_cfr_6max_current_reference/panels/holdout_panel_strategy.bin",
    "--traversal-workers", "18",
    "--traversal-consolidate-processes",
    "--traversal-gpu-batch",
    "--traversal-gpu-batch-size", "256",
    "--traversal-gpu-batch-timeout-us", "200",
    "--traversal-fp16-onnx",
    "--traversal-cpu-affinity", "0-15",
    "--traversal-progress-batch", "240",
    "--buffer-save-every", "10",
    "--work-dir", $workDir,
    "--ring-eval-every", "1",
    "--ring-eval-hands", "1000",
    "--model-checkpoint-every", "10",
    "--resume"
)

Write-WatchdogLine "watchdog started in '$repoRoot'" Cyan
Write-WatchdogLine "target iterations: $TargetIterations | restart delay: $RestartDelaySeconds s" Cyan
Write-WatchdogLine "state path: $statePath" DarkGray
Write-WatchdogLine "stop manually any time with Ctrl+C" DarkGray

$attempt = 0
while ($true) {
    $completedBefore = Get-CompletedIterations -Path $statePath
    if ($completedBefore -ge $TargetIterations) {
        Write-WatchdogLine "target already reached (completed=$completedBefore)." Green
        break
    }

    $attempt += 1
    Write-WatchdogLine "launching attempt #$attempt (completed=$completedBefore, target=$TargetIterations)..." Cyan

    & py @runArgs
    $exitCode = $LASTEXITCODE

    $completedAfter = Get-CompletedIterations -Path $statePath
    if ($completedAfter -ge $TargetIterations) {
        Write-WatchdogLine "target reached (completed=$completedAfter). watchdog exiting." Green
        break
    }

    Write-WatchdogLine "run exited with code $exitCode at completed=$completedAfter; restarting in $RestartDelaySeconds s..." Yellow
    Start-Sleep -Seconds $RestartDelaySeconds
}
