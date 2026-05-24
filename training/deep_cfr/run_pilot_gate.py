#!/usr/bin/env python3
"""Run medium pilot checkpoints and evaluate holdout trend gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUN_SCRIPT = SCRIPT_DIR / "run_deep_cfr.py"


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    hidden_dim: int
    bottleneck_dim: int
    traversals: int
    training_steps: int
    buffer_size: int
    batch_size: int
    deck_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot gate for selected 6-max candidate config.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--screen-panel", type=Path, required=True)
    parser.add_argument("--holdout-panel", type=Path, required=True)
    parser.add_argument("--scale-search-report", type=Path, default=None)
    parser.add_argument("--candidate-name", type=str, default=None)

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--bottleneck-dim", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--deck-samples", type=int, default=None)

    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--total-iterations", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traversal-workers", type=int, default=0)
    parser.add_argument("--traversal-progress-batch", type=int, default=500)
    parser.add_argument("--traversal-seat-chunks", type=int, default=1)
    parser.add_argument(
        "--checkpoint-timeout-sec",
        type=int,
        default=0,
        help="Optional timeout per checkpoint training call in seconds (0 disables timeout).",
    )
    parser.add_argument("--starting-stack", type=int, default=2000)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--big-blind", type=int, default=20)

    parser.add_argument("--evaluator-panel-max-samples", type=int, default=10_000)
    parser.add_argument("--evaluator-panel-batch-size", type=int, default=2048)
    parser.add_argument("--evaluator-panel-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--evaluator-required-seats", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--evaluator-min-seat-samples", type=int, default=1)
    parser.add_argument("--evaluator-max-dominant-seat-fraction", type=float, default=0.70)
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--holdout-seed", type=int, default=4242)

    parser.add_argument("--min-total-improvement", type=float, default=1.0)
    parser.add_argument("--max-step-regression", type=float, default=3.0)
    parser.add_argument("--max-final-regression", type=float, default=0.5)
    parser.add_argument("--require-coverage-pass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.total_iterations <= 0:
        raise ValueError("--total-iterations must be > 0")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")
    if args.num_players < 2 or args.num_players > 6:
        raise ValueError("--num-players must be in [2, 6]")
    if args.traversal_workers < 0:
        raise ValueError("--traversal-workers must be >= 0")
    if args.traversal_progress_batch <= 0:
        raise ValueError("--traversal-progress-batch must be > 0")
    if args.traversal_seat_chunks <= 0:
        raise ValueError("--traversal-seat-chunks must be > 0")
    if args.checkpoint_timeout_sec < 0:
        raise ValueError("--checkpoint-timeout-sec must be >= 0")
    if args.evaluator_panel_max_samples < 0:
        raise ValueError("--evaluator-panel-max-samples must be >= 0")
    if args.evaluator_panel_batch_size <= 0:
        raise ValueError("--evaluator-panel-batch-size must be > 0")
    if args.evaluator_min_seat_samples < 0:
        raise ValueError("--evaluator-min-seat-samples must be >= 0")
    if not (0.0 <= args.evaluator_max_dominant_seat_fraction <= 1.0):
        raise ValueError("--evaluator-max-dominant-seat-fraction must be in [0, 1]")
    return args


def candidate_from_scale_search(path: Path, candidate_name: str | None) -> CandidateConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    results = payload.get("results", [])
    if isinstance(results, list):
        for row in results:
            if isinstance(row, dict):
                cand = row.get("candidate")
                if isinstance(cand, dict):
                    candidates.append(cand)
    if not candidates:
        recommended = payload.get("recommended_candidate", {})
        if isinstance(recommended, dict):
            cand = recommended.get("candidate", {})
            if isinstance(cand, dict):
                candidates.append(cand)
    if not candidates:
        raise ValueError("scale search report has no candidate configs")

    picked: dict[str, Any] | None = None
    if candidate_name:
        for cand in candidates:
            if str(cand.get("name")) == candidate_name:
                picked = cand
                break
        if picked is None:
            raise ValueError(f"candidate-name not found in scale search report: {candidate_name}")
    else:
        recommended = payload.get("recommended_candidate", {})
        if isinstance(recommended, dict):
            cand = recommended.get("candidate", {})
            if isinstance(cand, dict) and cand:
                picked = cand
        if picked is None:
            picked = candidates[0]

    return CandidateConfig(
        name=str(picked["name"]),
        hidden_dim=int(picked["hidden_dim"]),
        bottleneck_dim=int(picked["bottleneck_dim"]),
        traversals=int(picked["traversals"]),
        training_steps=int(picked["training_steps"]),
        buffer_size=int(picked["buffer_size"]),
        batch_size=int(picked["batch_size"]),
        deck_samples=int(picked["deck_samples"]),
    )


def candidate_from_explicit_args(args: argparse.Namespace) -> CandidateConfig | None:
    required = (
        args.name,
        args.hidden_dim,
        args.bottleneck_dim,
        args.traversals,
        args.training_steps,
        args.buffer_size,
        args.batch_size,
        args.deck_samples,
    )
    if all(value is None for value in required):
        return None
    if any(value is None for value in required):
        raise ValueError(
            "explicit candidate requires all fields: --name, --hidden-dim, --bottleneck-dim, "
            "--traversals, --training-steps, --buffer-size, --batch-size, --deck-samples"
        )
    return CandidateConfig(
        name=str(args.name),
        hidden_dim=int(args.hidden_dim),
        bottleneck_dim=int(args.bottleneck_dim),
        traversals=int(args.traversals),
        training_steps=int(args.training_steps),
        buffer_size=int(args.buffer_size),
        batch_size=int(args.batch_size),
        deck_samples=int(args.deck_samples),
    )


def resolve_candidate(args: argparse.Namespace) -> CandidateConfig:
    explicit = candidate_from_explicit_args(args)
    if explicit is not None:
        return explicit
    if args.scale_search_report is None:
        raise ValueError("provide candidate fields directly or pass --scale-search-report")
    return candidate_from_scale_search(args.scale_search_report.resolve(), args.candidate_name)


def run_training_to_iteration(
    *,
    args: argparse.Namespace,
    cfg: CandidateConfig,
    target_iteration: int,
    resume: bool,
) -> dict[str, Any]:
    if args.dry_run:
        return {"status": "dry_run", "target_iteration": target_iteration, "resume": resume}
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--num-players",
        str(args.num_players),
        "--iterations",
        str(target_iteration),
        "--traversals",
        str(cfg.traversals),
        "--training-steps",
        str(cfg.training_steps),
        "--batch-size",
        str(cfg.batch_size),
        "--buffer-size",
        str(cfg.buffer_size),
        "--hidden-dim",
        str(cfg.hidden_dim),
        "--bottleneck-dim",
        str(cfg.bottleneck_dim),
        "--deck-samples",
        str(cfg.deck_samples),
        "--strategy-every",
        "1",
        "--league-eval-every",
        "0",
        "--diagnostic-every",
        "0",
        "--exploitability-every",
        "0",
        "--h2h-only-diagnostics",
        "--seed",
        str(args.seed),
        "--traversal-workers",
        str(args.traversal_workers),
        "--traversal-progress-batch",
        str(args.traversal_progress_batch),
            "--traversal-seat-chunks",
            str(args.traversal_seat_chunks),
        "--starting-stack",
        str(args.starting_stack),
        "--small-blind",
        str(args.small_blind),
        "--big-blind",
        str(args.big_blind),
        "--work-dir",
        str(args.work_dir.resolve()),
    ]
    if resume:
        command.append("--resume")
    timeout_sec = int(args.checkpoint_timeout_sec)
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout_text, stderr_text = proc.communicate(
            timeout=(timeout_sec if timeout_sec > 0 else None)
        )
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.kill()
        finally:
            try:
                stdout_text, stderr_text = proc.communicate(timeout=15)
            except Exception:
                stdout_text = ""
                stderr_text = ""
        return {
            "status": "error",
            "error": f"checkpoint timed out after {timeout_sec}s",
            "timed_out": True,
            "timeout_sec": int(timeout_sec),
            "target_iteration": target_iteration,
            "resume": resume,
            "stdout_tail": (stdout_text or "").strip()[-2000:],
            "stderr_tail": (stderr_text or "").strip()[-2000:],
        }
    if proc.returncode != 0:
        tail = (stderr_text or "").strip() or (stdout_text or "").strip()
        return {"status": "error", "error": tail, "target_iteration": target_iteration, "resume": resume}
    metrics_path = args.work_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "status": "error",
            "error": f"missing metrics.json: {metrics_path}",
            "target_iteration": target_iteration,
        }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    iterations = metrics.get("iterations", [])
    latest: dict[str, Any] = {}
    if isinstance(iterations, list):
        for item in reversed(iterations):
            if not isinstance(item, dict):
                continue
            if int(item.get("iteration", -1)) == int(target_iteration):
                latest = item
                break
        if not latest and iterations and isinstance(iterations[-1], dict):
            latest = iterations[-1]
    return {
        "status": "ok",
        "target_iteration": target_iteration,
        "latest_metrics": latest,
        "metrics_path": str(metrics_path.resolve()),
    }


def run_panel_eval(
    *,
    args: argparse.Namespace,
    model_state: Path,
    panel_path: Path,
    panel_label: str,
    seed: int,
    report_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    if args.dry_run:
        return {"status": "dry_run", "panel_label": panel_label}
    command = [
        sys.executable,
        "-m",
        "eval.run_league",
        "--eval-backend",
        "deep_cfr_panel",
        "--players",
        str(args.num_players),
        "--seed",
        str(seed),
        "--hero-mode",
        "argmax",
        "--deep-cfr-model-state",
        str(model_state),
        "--deep-cfr-samples",
        str(panel_path),
        "--deep-cfr-sample-type",
        "strategy",
        "--deep-cfr-max-samples",
        str(args.evaluator_panel_max_samples),
        "--deep-cfr-batch-size",
        str(args.evaluator_panel_batch_size),
        "--deep-cfr-device",
        str(args.evaluator_panel_device),
        "--deep-cfr-label",
        panel_label,
        "--deep-cfr-required-seats",
        str(args.evaluator_required_seats),
        "--deep-cfr-min-seat-samples",
        str(args.evaluator_min_seat_samples),
        "--deep-cfr-max-dominant-seat-fraction",
        str(args.evaluator_max_dominant_seat_fraction),
        "--report-json",
        str(report_path),
        "--results",
        str(results_path),
        "--quiet",
    ]
    if args.num_players == 6:
        command.extend(["--match-format", "6max"])
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip() or proc.stdout.strip()
        return {"status": "error", "error": tail}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
    panel = diagnostics.get("deep_cfr_panel", {}) if isinstance(diagnostics, dict) else {}
    metrics = panel.get("metrics", {}) if isinstance(panel, dict) else {}
    coverage = panel.get("coverage", {}) if isinstance(panel, dict) else {}
    return {
        "status": "ok",
        "report_json": str(report_path.resolve()),
        "results_csv": str(results_path.resolve()),
        "score": float(panel.get("score", 0.0)) if isinstance(panel, dict) else 0.0,
        "top1_match_rate": float(metrics.get("top1_match_rate", 0.0)) if isinstance(metrics, dict) else 0.0,
        "cross_entropy": float(metrics.get("cross_entropy", 0.0)) if isinstance(metrics, dict) else 0.0,
        "masked_l1": float(metrics.get("masked_l1", 0.0)) if isinstance(metrics, dict) else 0.0,
        "fingerprint_hash": str(panel.get("fingerprint", {}).get("hash", "")) if isinstance(panel, dict) else "",
        "coverage": coverage,
    }


def checkpoint_schedule(total_iterations: int, checkpoint_every: int) -> list[int]:
    out: list[int] = []
    current = checkpoint_every
    while current < total_iterations:
        out.append(current)
        current += checkpoint_every
    if not out or out[-1] != total_iterations:
        out.append(total_iterations)
    return out


def compute_trend(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0}
    x = np.arange(len(scores), dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0]) if len(scores) > 1 else 0.0
    deltas = [float(scores[idx] - scores[idx - 1]) for idx in range(1, len(scores))]
    return {
        "count": len(scores),
        "first": float(scores[0]),
        "last": float(scores[-1]),
        "max": float(np.max(y)),
        "min": float(np.min(y)),
        "slope_per_checkpoint": slope,
        "deltas": deltas,
    }


def main() -> int:
    args = parse_args()
    if not RUN_SCRIPT.exists():
        raise FileNotFoundError(f"missing run script: {RUN_SCRIPT}")
    if not args.screen_panel.exists():
        raise FileNotFoundError(f"missing screen panel: {args.screen_panel}")
    if not args.holdout_panel.exists():
        raise FileNotFoundError(f"missing holdout panel: {args.holdout_panel}")

    cfg = resolve_candidate(args)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    schedule = checkpoint_schedule(int(args.total_iterations), int(args.checkpoint_every))

    checkpoints: list[dict[str, Any]] = []
    for idx, iter_target in enumerate(schedule):
        training = run_training_to_iteration(
            args=args,
            cfg=cfg,
            target_iteration=int(iter_target),
            resume=(idx > 0),
        )
        entry: dict[str, Any] = {
            "iteration": int(iter_target),
            "training": training,
            "status": str(training.get("status", "error")),
        }
        if str(training.get("status")) == "ok":
            strategy_state = args.work_dir / "models" / "strategy_shared.pt"
            if not strategy_state.exists() and not args.dry_run:
                entry["status"] = "error"
                entry["error"] = f"missing strategy checkpoint: {strategy_state}"
            else:
                report_dir = args.work_dir / "pilot_reports" / f"iter{iter_target:03d}"
                report_dir.mkdir(parents=True, exist_ok=True)
                screen_eval = run_panel_eval(
                    args=args,
                    model_state=strategy_state,
                    panel_path=args.screen_panel.resolve(),
                    panel_label=f"pilot_iter{iter_target:03d}_screen",
                    seed=int(args.screen_seed ^ (iter_target * 0x9E37_79B9)),
                    report_path=report_dir / "screen.json",
                    results_path=report_dir / "screen.csv",
                )
                holdout_eval = run_panel_eval(
                    args=args,
                    model_state=strategy_state,
                    panel_path=args.holdout_panel.resolve(),
                    panel_label=f"pilot_iter{iter_target:03d}_holdout",
                    seed=int(args.holdout_seed ^ (iter_target * 0xA24B_AED4)),
                    report_path=report_dir / "holdout.json",
                    results_path=report_dir / "holdout.csv",
                )
                entry["screen_eval"] = screen_eval
                entry["holdout_eval"] = holdout_eval
                if str(screen_eval.get("status")) != "ok" or str(holdout_eval.get("status")) != "ok":
                    entry["status"] = "error"
        checkpoints.append(entry)

    valid_holdout_scores: list[float] = []
    coverage_ok = True
    for entry in checkpoints:
        if str(entry.get("status")) != "ok":
            continue
        holdout_eval = entry.get("holdout_eval", {})
        if isinstance(holdout_eval, dict):
            valid_holdout_scores.append(float(holdout_eval.get("top1_match_rate", 0.0) * 100.0))
            if args.require_coverage_pass:
                coverage = holdout_eval.get("coverage", {})
                if isinstance(coverage, dict) and str(coverage.get("status", "failed")) != "ok":
                    coverage_ok = False
    trend = compute_trend(valid_holdout_scores)

    improving = False
    non_regressing = False
    step_regression_ok = True
    if int(trend.get("count", 0)) > 0:
        first = float(trend.get("first", 0.0))
        last = float(trend.get("last", 0.0))
        improving = last >= first + float(args.min_total_improvement)
        non_regressing = last >= first - float(args.max_final_regression)
        for delta in trend.get("deltas", []):
            if float(delta) < -float(args.max_step_regression):
                step_regression_ok = False
                break

    all_ok = all(str(entry.get("status")) == "ok" for entry in checkpoints)
    pass_pilot_gate = all_ok and coverage_ok and step_regression_ok and (improving or non_regressing)

    report = {
        "status": "ok" if pass_pilot_gate else "failed",
        "candidate": {
            "name": cfg.name,
            "hidden_dim": cfg.hidden_dim,
            "bottleneck_dim": cfg.bottleneck_dim,
            "traversals": cfg.traversals,
            "training_steps": cfg.training_steps,
            "buffer_size": cfg.buffer_size,
            "batch_size": cfg.batch_size,
            "deck_samples": cfg.deck_samples,
        },
        "config": {
            "work_dir": str(args.work_dir),
            "screen_panel": str(args.screen_panel.resolve()),
            "holdout_panel": str(args.holdout_panel.resolve()),
            "total_iterations": int(args.total_iterations),
            "checkpoint_every": int(args.checkpoint_every),
            "schedule": schedule,
            "num_players": int(args.num_players),
            "seed": int(args.seed),
            "traversal_workers": int(args.traversal_workers),
            "traversal_progress_batch": int(args.traversal_progress_batch),
            "traversal_seat_chunks": int(args.traversal_seat_chunks),
            "checkpoint_timeout_sec": int(args.checkpoint_timeout_sec),
            "evaluator_panel_max_samples": int(args.evaluator_panel_max_samples),
            "evaluator_panel_batch_size": int(args.evaluator_panel_batch_size),
            "evaluator_panel_device": str(args.evaluator_panel_device),
            "evaluator_required_seats": str(args.evaluator_required_seats),
            "evaluator_min_seat_samples": int(args.evaluator_min_seat_samples),
            "evaluator_max_dominant_seat_fraction": float(args.evaluator_max_dominant_seat_fraction),
            "min_total_improvement": float(args.min_total_improvement),
            "max_step_regression": float(args.max_step_regression),
            "max_final_regression": float(args.max_final_regression),
            "require_coverage_pass": bool(args.require_coverage_pass),
            "dry_run": bool(args.dry_run),
        },
        "checkpoints": checkpoints,
        "trend": trend,
        "gates": {
            "all_checkpoints_ok": all_ok,
            "coverage_ok": coverage_ok,
            "step_regression_ok": step_regression_ok,
            "improving": improving,
            "non_regressing": non_regressing,
            "pass_pilot_gate": pass_pilot_gate,
        },
    }

    args.report_json = args.report_json.resolve()
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if pass_pilot_gate else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
