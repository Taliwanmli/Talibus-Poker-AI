#!/usr/bin/env python3
"""Run short 6-max scale search with independent screen/holdout panels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


DEFAULT_CANDIDATES: tuple[CandidateConfig, ...] = (
    CandidateConfig(
        name="baseline_large",
        hidden_dim=1024,
        bottleneck_dim=512,
        traversals=3000,
        training_steps=240,
        buffer_size=250_000,
        batch_size=4096,
        deck_samples=200,
    ),
    CandidateConfig(
        name="wider_model",
        hidden_dim=1280,
        bottleneck_dim=640,
        traversals=3000,
        training_steps=240,
        buffer_size=250_000,
        batch_size=4096,
        deck_samples=200,
    ),
    CandidateConfig(
        name="more_traversal",
        hidden_dim=1024,
        bottleneck_dim=512,
        traversals=4500,
        training_steps=240,
        buffer_size=250_000,
        batch_size=4096,
        deck_samples=200,
    ),
)


def parse_candidate_spec(raw: str) -> CandidateConfig:
    data: dict[str, str] = {}
    for token in raw.split(","):
        text = token.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"invalid --candidate token (expected key=value): {text!r}")
        key, value = text.split("=", 1)
        data[key.strip()] = value.strip()
    required = (
        "name",
        "hidden_dim",
        "bottleneck_dim",
        "traversals",
        "training_steps",
        "buffer_size",
        "batch_size",
        "deck_samples",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing candidate field(s): {missing}")
    return CandidateConfig(
        name=str(data["name"]),
        hidden_dim=int(data["hidden_dim"]),
        bottleneck_dim=int(data["bottleneck_dim"]),
        traversals=int(data["traversals"]),
        training_steps=int(data["training_steps"]),
        buffer_size=int(data["buffer_size"]),
        batch_size=int(data["batch_size"]),
        deck_samples=int(data["deck_samples"]),
    )


def parse_candidates_json(path: Path) -> list[CandidateConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--candidates-json must contain a JSON array")
    out: list[CandidateConfig] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"--candidates-json[{idx}] must be an object")
        try:
            out.append(
                CandidateConfig(
                    name=str(item["name"]),
                    hidden_dim=int(item["hidden_dim"]),
                    bottleneck_dim=int(item["bottleneck_dim"]),
                    traversals=int(item["traversals"]),
                    training_steps=int(item["training_steps"]),
                    buffer_size=int(item["buffer_size"]),
                    batch_size=int(item["batch_size"]),
                    deck_samples=int(item["deck_samples"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"--candidates-json[{idx}] missing field: {exc}") from exc
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short 6-max scale search.")
    parser.add_argument("--work-root", type=Path, required=True, help="Parent directory for per-candidate runs.")
    parser.add_argument("--screen-panel", type=Path, required=True, help="Independent screen panel binary.")
    parser.add_argument("--holdout-panel", type=Path, required=True, help="Independent holdout panel binary.")
    parser.add_argument("--report-json", type=Path, required=True, help="Scale search report output.")
    parser.add_argument("--candidates-json", type=Path, default=None, help="Optional JSON list of candidates.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help=(
            "Candidate spec: "
            "name=...,hidden_dim=...,bottleneck_dim=...,traversals=...,training_steps=...,"
            "buffer_size=...,batch_size=...,deck_samples=..."
        ),
    )
    parser.add_argument(
        "--use-default-candidates",
        action="store_true",
        help="Include built-in candidates if no explicit candidate list is supplied.",
    )
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--strategy-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--holdout-seed", type=int, default=4242)
    parser.add_argument("--traversal-workers", type=int, default=0)
    parser.add_argument("--traversal-progress-batch", type=int, default=500)
    parser.add_argument("--traversal-seat-chunks", type=int, default=1)
    parser.add_argument("--evaluator-panel-max-samples", type=int, default=10_000)
    parser.add_argument("--evaluator-panel-batch-size", type=int, default=2048)
    parser.add_argument("--evaluator-panel-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--evaluator-required-seats", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--evaluator-min-seat-samples", type=int, default=1)
    parser.add_argument("--evaluator-max-dominant-seat-fraction", type=float, default=0.70)
    parser.add_argument("--starting-stack", type=int, default=2000)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--big-blind", type=int, default=20)
    parser.add_argument(
        "--candidate-timeout-sec",
        type=int,
        default=0,
        help="Optional per-candidate training timeout in seconds (0 disables timeout).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Emit plan/report skeleton without running commands.")
    args = parser.parse_args()

    if args.num_players < 2 or args.num_players > 6:
        raise ValueError("--num-players must be in [2, 6]")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.strategy_every < 0:
        raise ValueError("--strategy-every must be >= 0")
    if args.traversal_workers < 0:
        raise ValueError("--traversal-workers must be >= 0")
    if args.traversal_progress_batch <= 0:
        raise ValueError("--traversal-progress-batch must be > 0")
    if args.traversal_seat_chunks <= 0:
        raise ValueError("--traversal-seat-chunks must be > 0")
    if args.evaluator_panel_max_samples < 0:
        raise ValueError("--evaluator-panel-max-samples must be >= 0")
    if args.evaluator_panel_batch_size <= 0:
        raise ValueError("--evaluator-panel-batch-size must be > 0")
    if args.evaluator_min_seat_samples < 0:
        raise ValueError("--evaluator-min-seat-samples must be >= 0")
    if not (0.0 <= args.evaluator_max_dominant_seat_fraction <= 1.0):
        raise ValueError("--evaluator-max-dominant-seat-fraction must be in [0, 1]")
    if args.candidate_timeout_sec < 0:
        raise ValueError("--candidate-timeout-sec must be >= 0")
    return args


def gather_candidates(args: argparse.Namespace) -> list[CandidateConfig]:
    out: list[CandidateConfig] = []
    if args.candidates_json is not None:
        out.extend(parse_candidates_json(args.candidates_json.resolve()))
    for raw in args.candidate:
        out.append(parse_candidate_spec(raw))
    if not out:
        out.extend(DEFAULT_CANDIDATES if args.use_default_candidates else [])
    if not out:
        raise ValueError("no candidates provided; pass --candidate/--candidates-json or --use-default-candidates")
    seen: set[str] = set()
    deduped: list[CandidateConfig] = []
    for cfg in out:
        if cfg.name in seen:
            raise ValueError(f"duplicate candidate name: {cfg.name}")
        seen.add(cfg.name)
        deduped.append(cfg)
    return deduped


def run_training_candidate(
    *,
    args: argparse.Namespace,
    cfg: CandidateConfig,
    candidate_work_dir: Path,
) -> dict[str, Any]:
    if args.dry_run:
        return {"status": "dry_run", "work_dir": str(candidate_work_dir.resolve())}
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--num-players",
        str(args.num_players),
        "--iterations",
        str(args.iterations),
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
        str(args.strategy_every),
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
        str(candidate_work_dir),
    ]
    # Allow fast reruns after downstream evaluation failures by reusing
    # completed candidate checkpoints/work dirs when available.
    if (candidate_work_dir / "state.json").exists():
        command.append("--resume")
    started = time.perf_counter()
    timeout_sec = int(args.candidate_timeout_sec)
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout_text, stderr_text = proc.communicate(timeout=(timeout_sec if timeout_sec > 0 else None))
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
        elapsed = time.perf_counter() - started
        return {
            "status": "error",
            "error": f"candidate training timed out after {timeout_sec}s",
            "timed_out": True,
            "timeout_sec": int(timeout_sec),
            "elapsed_sec": elapsed,
            "work_dir": str(candidate_work_dir.resolve()),
            "stdout_tail": (stdout_text or "").strip()[-2000:],
            "stderr_tail": (stderr_text or "").strip()[-2000:],
        }
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        tail = (stderr_text or "").strip() or (stdout_text or "").strip()
        return {
            "status": "error",
            "error": tail,
            "elapsed_sec": elapsed,
            "work_dir": str(candidate_work_dir.resolve()),
        }

    metrics_path = candidate_work_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "status": "error",
            "error": f"missing metrics.json: {metrics_path}",
            "elapsed_sec": elapsed,
            "work_dir": str(candidate_work_dir.resolve()),
        }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    entries = metrics.get("iterations", [])
    if not isinstance(entries, list) or not entries:
        return {
            "status": "error",
            "error": "metrics.json has no iterations",
            "elapsed_sec": elapsed,
            "work_dir": str(candidate_work_dir.resolve()),
        }
    first = entries[0] if isinstance(entries[0], dict) else {}
    last = entries[-1] if isinstance(entries[-1], dict) else {}
    times = last.get("times_sec", {}) if isinstance(last, dict) else {}
    artifact_first = first.get("artifact_bytes", {}) if isinstance(first, dict) else {}
    artifact_last = last.get("artifact_bytes", {}) if isinstance(last, dict) else {}
    first_work = int(artifact_first.get("work_dir", 0)) if isinstance(artifact_first, dict) else 0
    last_work = int(artifact_last.get("work_dir", 0)) if isinstance(artifact_last, dict) else 0
    iter_count = max(1, len(entries))
    growth_per_iter = float(last_work - first_work) / float(max(1, iter_count - 1))
    model_state = candidate_work_dir / "models" / "strategy_shared.pt"
    if not model_state.exists():
        return {
            "status": "error",
            "error": f"missing strategy checkpoint: {model_state}",
            "elapsed_sec": elapsed,
            "work_dir": str(candidate_work_dir.resolve()),
        }
    return {
        "status": "ok",
        "elapsed_sec": elapsed,
        "work_dir": str(candidate_work_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "model_state": str(model_state.resolve()),
        "times_sec": {
            "iteration_total": float(times.get("iteration_total", 0.0)),
            "traverse_wall": float(times.get("traverse_wall", times.get("traverse_total", 0.0))),
            "traverse_sum_seat": float(
                times.get("traverse_sum_seat_time", times.get("traverse_total", 0.0))
            ),
            "train_total": float(times.get("train_total", 0.0)),
        },
        "artifact_bytes": {
            "first_work_dir": int(first_work),
            "last_work_dir": int(last_work),
            "growth_per_iter": float(growth_per_iter),
        },
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
    if not report_path.exists():
        return {"status": "error", "error": f"missing panel report: {report_path}"}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
    panel = diagnostics.get("deep_cfr_panel", {}) if isinstance(diagnostics, dict) else {}
    metrics = panel.get("metrics", {}) if isinstance(panel, dict) else {}
    coverage = panel.get("coverage", {}) if isinstance(panel, dict) else {}
    fingerprint = panel.get("fingerprint", {}) if isinstance(panel, dict) else {}
    return {
        "status": "ok",
        "report_json": str(report_path.resolve()),
        "results_csv": str(results_path.resolve()),
        "panel_score": float(panel.get("score", 0.0)) if isinstance(panel, dict) else 0.0,
        "top1_match_rate": float(metrics.get("top1_match_rate", 0.0)) if isinstance(metrics, dict) else 0.0,
        "cross_entropy": float(metrics.get("cross_entropy", 0.0)) if isinstance(metrics, dict) else 0.0,
        "masked_l1": float(metrics.get("masked_l1", 0.0)) if isinstance(metrics, dict) else 0.0,
        "fingerprint_hash": str(fingerprint.get("hash", "")) if isinstance(fingerprint, dict) else "",
        "seat_breakdown": panel.get("seat_breakdown", {}) if isinstance(panel, dict) else {},
        "coverage": coverage,
    }


def ranking_key(entry: dict[str, Any]) -> tuple[float, float, float, float, float]:
    holdout = entry.get("holdout_eval", {})
    training = entry.get("training", {})
    return (
        -float(holdout.get("top1_match_rate", 0.0)),
        float(holdout.get("cross_entropy", 1e9)),
        float(holdout.get("masked_l1", 1e9)),
        float(training.get("times_sec", {}).get("iteration_total", 1e9)),
        float(training.get("artifact_bytes", {}).get("growth_per_iter", 1e18)),
    )


def main() -> int:
    args = parse_args()
    if not RUN_SCRIPT.exists():
        raise FileNotFoundError(f"missing run script: {RUN_SCRIPT}")
    if not args.screen_panel.exists():
        raise FileNotFoundError(f"missing screen panel: {args.screen_panel}")
    if not args.holdout_panel.exists():
        raise FileNotFoundError(f"missing holdout panel: {args.holdout_panel}")
    candidates = gather_candidates(args)
    args.work_root = args.work_root.resolve()
    args.work_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, cfg in enumerate(candidates):
        candidate_dir = args.work_root / cfg.name
        panel_reports_dir = candidate_dir / "panel_reports"
        panel_reports_dir.mkdir(parents=True, exist_ok=True)
        training = run_training_candidate(args=args, cfg=cfg, candidate_work_dir=candidate_dir)
        entry: dict[str, Any] = {
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
            "status": str(training.get("status", "error")),
            "training": training,
        }
        if str(training.get("status")) == "ok":
            model_state = Path(str(training["model_state"]))
            screen_eval = run_panel_eval(
                args=args,
                model_state=model_state,
                panel_path=args.screen_panel.resolve(),
                panel_label=f"{cfg.name}_screen",
                seed=int(args.screen_seed ^ ((idx + 1) * 0x9E37_79B9)),
                report_path=panel_reports_dir / "screen.json",
                results_path=panel_reports_dir / "screen.csv",
            )
            holdout_eval = run_panel_eval(
                args=args,
                model_state=model_state,
                panel_path=args.holdout_panel.resolve(),
                panel_label=f"{cfg.name}_holdout",
                seed=int(args.holdout_seed ^ ((idx + 1) * 0xA24B_AED4)),
                report_path=panel_reports_dir / "holdout.json",
                results_path=panel_reports_dir / "holdout.csv",
            )
            entry["screen_eval"] = screen_eval
            entry["holdout_eval"] = holdout_eval
            if str(screen_eval.get("status")) != "ok" or str(holdout_eval.get("status")) != "ok":
                entry["status"] = "error"
        results.append(entry)

    valid = [row for row in results if str(row.get("status")) == "ok"]
    ranked = sorted(valid, key=ranking_key)
    best = ranked[0] if ranked else None
    report = {
        "status": "ok" if best is not None else "failed",
        "config": {
            "num_players": int(args.num_players),
            "iterations": int(args.iterations),
            "strategy_every": int(args.strategy_every),
            "seed": int(args.seed),
            "screen_seed": int(args.screen_seed),
            "holdout_seed": int(args.holdout_seed),
            "screen_panel": str(args.screen_panel.resolve()),
            "holdout_panel": str(args.holdout_panel.resolve()),
            "traversal_workers": int(args.traversal_workers),
            "traversal_progress_batch": int(args.traversal_progress_batch),
            "traversal_seat_chunks": int(args.traversal_seat_chunks),
            "candidate_timeout_sec": int(args.candidate_timeout_sec),
            "evaluator_panel_max_samples": int(args.evaluator_panel_max_samples),
            "evaluator_panel_batch_size": int(args.evaluator_panel_batch_size),
            "evaluator_panel_device": str(args.evaluator_panel_device),
            "evaluator_required_seats": str(args.evaluator_required_seats),
            "evaluator_min_seat_samples": int(args.evaluator_min_seat_samples),
            "evaluator_max_dominant_seat_fraction": float(args.evaluator_max_dominant_seat_fraction),
            "work_root": str(args.work_root),
            "dry_run": bool(args.dry_run),
        },
        "results": results,
        "ranking": [row.get("candidate", {}).get("name") for row in ranked],
        "recommended_candidate": best,
    }

    args.report_json = args.report_json.resolve()
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if best is not None else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
