#!/usr/bin/env python3
"""Run heavyweight final evaluation for a completed longrun checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-longrun final evaluation bundle.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Longrun work directory.")
    parser.add_argument("--screen-panel", type=Path, required=True)
    parser.add_argument("--holdout-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-max-samples", type=int, default=50_000)
    parser.add_argument("--panel-batch-size", type=int, default=4096)
    parser.add_argument("--panel-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--panel-required-seats", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--panel-min-seat-samples", type=int, default=1)
    parser.add_argument("--panel-max-dominant-seat-fraction", type=float, default=0.70)
    parser.add_argument("--gameplay-hands-per-seed", type=int, default=5000)
    parser.add_argument("--gameplay-seeds", type=str, default="42,4242,1337")
    parser.add_argument("--gameplay-opponents", type=str, default="station,nit,lag,mixed")
    parser.add_argument("--gameplay-policy-mode", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--checkpoint-state", type=Path, default=None, help="Optional explicit model checkpoint.")
    args = parser.parse_args()
    if args.panel_max_samples <= 0:
        raise ValueError("--panel-max-samples must be > 0")
    if args.panel_batch_size <= 0:
        raise ValueError("--panel-batch-size must be > 0")
    if args.panel_min_seat_samples < 0:
        raise ValueError("--panel-min-seat-samples must be >= 0")
    if not (0.0 <= float(args.panel_max_dominant_seat_fraction) <= 1.0):
        raise ValueError("--panel-max-dominant-seat-fraction must be in [0,1]")
    if args.gameplay_hands_per_seed <= 0:
        raise ValueError("--gameplay-hands-per-seed must be > 0")
    return args


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_checkpoint_state(work_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        explicit_path = explicit.resolve()
        if not explicit_path.exists():
            raise FileNotFoundError(f"checkpoint-state not found: {explicit_path}")
        return explicit_path

    checkpoint_eval_state = work_dir / "checkpoint_eval" / "best_checkpoint.json"
    if checkpoint_eval_state.exists():
        try:
            payload = read_json(checkpoint_eval_state)
            best_confirmed = payload.get("best_holdout")
            if isinstance(best_confirmed, dict):
                state_path = best_confirmed.get("state_path")
                if isinstance(state_path, str) and state_path:
                    candidate = Path(state_path).resolve()
                    if candidate.exists():
                        return candidate
        except Exception:
            pass

    candidates = [
        work_dir / "models" / "strategy_shared_best_confirmed.pt",
        work_dir / "models" / "strategy_best_confirmed.pt",
        work_dir / "models" / "strategy_shared.pt",
        work_dir / "models" / "strategy.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("unable to resolve a strategy checkpoint for final evaluation")


def run_command(command: list[str], *, log_path: Path) -> tuple[int, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    combined = {
        "command": command,
        "exit_code": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    log_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return int(proc.returncode), proc.stdout, proc.stderr


def run_panel_eval(
    *,
    args: argparse.Namespace,
    model_state: Path,
    panel_path: Path,
    panel_label: str,
    seed: int,
    report_json: Path,
    results_csv: Path,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "eval.run_league",
        "--eval-backend",
        "deep_cfr_panel",
        "--match-format",
        "6max",
        "--players",
        "6",
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
        str(int(args.panel_max_samples)),
        "--deep-cfr-batch-size",
        str(int(args.panel_batch_size)),
        "--deep-cfr-device",
        str(args.panel_device),
        "--deep-cfr-label",
        panel_label,
        "--deep-cfr-required-seats",
        str(args.panel_required_seats),
        "--deep-cfr-min-seat-samples",
        str(int(args.panel_min_seat_samples)),
        "--deep-cfr-max-dominant-seat-fraction",
        str(float(args.panel_max_dominant_seat_fraction)),
        "--report-json",
        str(report_json),
        "--results",
        str(results_csv),
        "--quiet",
    ]
    code, stdout_text, stderr_text = run_command(command, log_path=log_path)
    if code != 0:
        tail = (stderr_text or stdout_text).strip()[-2000:]
        return {"status": "error", "error": tail, "report_json": str(report_json), "results_csv": str(results_csv)}
    if not report_json.exists():
        return {
            "status": "error",
            "error": f"missing panel report: {report_json}",
            "report_json": str(report_json),
            "results_csv": str(results_csv),
        }
    report = read_json(report_json)
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
    panel = diagnostics.get("deep_cfr_panel", {}) if isinstance(diagnostics, dict) else {}
    metrics = panel.get("metrics", {}) if isinstance(panel, dict) else {}
    coverage = panel.get("coverage", {}) if isinstance(panel, dict) else {}
    return {
        "status": "ok",
        "report_json": str(report_json.resolve()),
        "results_csv": str(results_csv.resolve()),
        "score": float(panel.get("score", 0.0)) if isinstance(panel, dict) else 0.0,
        "score_ci_low": float(panel.get("score_ci_low", 0.0)) if isinstance(panel, dict) else 0.0,
        "score_ci_high": float(panel.get("score_ci_high", 0.0)) if isinstance(panel, dict) else 0.0,
        "top1_match_rate": float(metrics.get("top1_match_rate", 0.0)) if isinstance(metrics, dict) else 0.0,
        "cross_entropy": float(metrics.get("cross_entropy", 0.0)) if isinstance(metrics, dict) else 0.0,
        "masked_l1": float(metrics.get("masked_l1", 0.0)) if isinstance(metrics, dict) else 0.0,
        "coverage": coverage,
    }


def run_gameplay_eval(
    *,
    args: argparse.Namespace,
    model_state: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "eval.run_league",
        "--eval-backend",
        "deep_cfr_checkpoint",
        "--deep-cfr-model-state",
        str(model_state),
        "--deep-cfr-device",
        str(args.panel_device),
        "--deep-cfr-policy-mode",
        str(args.gameplay_policy_mode),
        "--match-format",
        "6max",
        "--players",
        "6",
        "--hero-mode",
        "worker",
        "--seeds",
        str(args.gameplay_seeds),
        "--hands-per-seed",
        str(int(args.gameplay_hands_per_seed)),
        "--opponents",
        str(args.gameplay_opponents),
        "--runs-dir",
        str(output_dir),
        "--quiet",
    ]
    code, stdout_text, stderr_text = run_command(command, log_path=log_path)
    if code != 0:
        tail = (stderr_text or stdout_text).strip()[-2000:]
        return {"status": "error", "error": tail, "output_dir": str(output_dir)}
    aggregate_json = output_dir / "aggregate_summary.json"
    if not aggregate_json.exists():
        return {
            "status": "error",
            "error": f"missing gameplay aggregate summary: {aggregate_json}",
            "output_dir": str(output_dir),
        }
    report = read_json(aggregate_json)
    overall = report.get("overall", {}) if isinstance(report, dict) else {}
    by_opponent = report.get("by_opponent", {}) if isinstance(report, dict) else {}
    diagnostics = report.get("diagnostics_by_opponent", {}) if isinstance(report, dict) else {}
    return {
        "status": "ok",
        "aggregate_json": str(aggregate_json.resolve()),
        "output_dir": str(output_dir.resolve()),
        "runs": report.get("runs", []) if isinstance(report, dict) else [],
        "overall": overall,
        "by_opponent": by_opponent,
        "diagnostics_by_opponent": diagnostics,
    }


def build_markdown(report: dict[str, Any]) -> str:
    panel = report.get("panel_eval", {})
    gameplay = report.get("gameplay_eval", {})
    lines = [
        "# Final Evaluation Report",
        "",
        f"- status: `{report.get('status')}`",
        f"- model_state: `{report.get('model_state')}`",
        "",
        "## Panel (Heavy)",
        f"- screen top1: `{panel.get('screen', {}).get('top1_match_rate')}`",
        f"- holdout top1: `{panel.get('holdout', {}).get('top1_match_rate')}`",
        f"- screen coverage: `{panel.get('screen', {}).get('coverage', {}).get('status')}`",
        f"- holdout coverage: `{panel.get('holdout', {}).get('coverage', {}).get('status')}`",
        "",
        "## 6-Max Gameplay",
        f"- overall mean_bb100: `{gameplay.get('overall', {}).get('mean_bb100')}`",
        f"- overall ci95_bb100: `{gameplay.get('overall', {}).get('ci95_bb100')}`",
        f"- opponents: `{report.get('config', {}).get('gameplay_opponents')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.screen_panel.exists():
        raise FileNotFoundError(f"screen panel not found: {args.screen_panel}")
    if not args.holdout_panel.exists():
        raise FileNotFoundError(f"holdout panel not found: {args.holdout_panel}")

    model_state = resolve_checkpoint_state(work_dir, args.checkpoint_state)

    panel_dir = output_dir / "panel"
    gameplay_dir = output_dir / "gameplay"
    logs_dir = output_dir / "logs"
    panel_dir.mkdir(parents=True, exist_ok=True)
    gameplay_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    screen_eval = run_panel_eval(
        args=args,
        model_state=model_state,
        panel_path=args.screen_panel.resolve(),
        panel_label="final_screen",
        seed=int(args.seed),
        report_json=panel_dir / "screen_report.json",
        results_csv=panel_dir / "screen_results.csv",
        log_path=logs_dir / "screen_cmd.json",
    )
    holdout_eval = run_panel_eval(
        args=args,
        model_state=model_state,
        panel_path=args.holdout_panel.resolve(),
        panel_label="final_holdout",
        seed=int(args.seed ^ 0xA24B_AED4),
        report_json=panel_dir / "holdout_report.json",
        results_csv=panel_dir / "holdout_results.csv",
        log_path=logs_dir / "holdout_cmd.json",
    )
    gameplay_eval = run_gameplay_eval(
        args=args,
        model_state=model_state,
        output_dir=gameplay_dir,
        log_path=logs_dir / "gameplay_cmd.json",
    )

    status = "ok"
    if any(
        str(entry.get("status", "error")) != "ok"
        for entry in (screen_eval, holdout_eval, gameplay_eval)
    ):
        status = "failed"

    report = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "output_dir": str(output_dir),
        "model_state": str(model_state),
        "config": {
            "seed": int(args.seed),
            "panel_max_samples": int(args.panel_max_samples),
            "panel_batch_size": int(args.panel_batch_size),
            "panel_device": str(args.panel_device),
            "panel_required_seats": str(args.panel_required_seats),
            "panel_min_seat_samples": int(args.panel_min_seat_samples),
            "panel_max_dominant_seat_fraction": float(args.panel_max_dominant_seat_fraction),
            "gameplay_hands_per_seed": int(args.gameplay_hands_per_seed),
            "gameplay_seeds": str(args.gameplay_seeds),
            "gameplay_opponents": str(args.gameplay_opponents),
            "gameplay_policy_mode": str(args.gameplay_policy_mode),
        },
        "panel_eval": {
            "screen": screen_eval,
            "holdout": holdout_eval,
        },
        "gameplay_eval": gameplay_eval,
        "artifacts": {
            "logs_dir": str(logs_dir),
            "panel_dir": str(panel_dir),
            "gameplay_dir": str(gameplay_dir),
        },
    }

    report_json = output_dir / "final_eval_report.json"
    report_md = output_dir / "final_eval_report.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
