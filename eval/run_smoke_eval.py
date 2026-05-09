#!/usr/bin/env python3
"""
Run a fast eval smoke test, write compact metrics, and prune old run artifacts.

Outputs:
  reports/report_<mode>.json
  reports/metrics_<mode>.json
  reports/runs/<run_id>/{report_<mode>.json,metrics_<mode>.json}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
RUNS_DIR = REPORTS_DIR / "runs"
DEFAULT_HANDS_BY_MODE = {
    "smoke": 200,
    "confirm": 2000,
}


def load_json_with_bom_fallback(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast eval smoke test and write reports/metrics.json."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "confirm"],
        default="smoke",
        help="Run mode; controls default hand count and output filenames.",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=None,
        help="Optional hand override. Defaults: smoke=200, confirm=2000.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip npm run eval:build before running smoke eval.",
    )
    parser.add_argument(
        "--keep_last",
        type=int,
        default=3,
        help="Retention: keep last N runs (+latest + best).",
    )
    parser.add_argument(
        "--retain_heavy",
        action="store_true",
        help="Retain per-run heavy artifacts (results.csv, trace).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel eval workers. Defaults to EVAL_JOBS or max(1, cpu_count-2).",
    )
    parser.add_argument(
        "--policy_use_approx",
        dest="policy_use_approx",
        action="store_true",
        default=None,
        help="Allow approx StrategyStore hits to drive policy actions.",
    )
    parser.add_argument(
        "--policy_exact_only",
        dest="policy_use_approx",
        action="store_false",
        help="Force exact-only policy actions even when approx hits are available.",
    )
    parser.add_argument(
        "--policy_selection_mode",
        choices=["sample", "argmax", "hybrid"],
        default=None,
        help=(
            "Strategy-store action selection mode. "
            "sample=solver-faithful weighted sampling, "
            "hybrid=argmax for dominant spots + sample for close mixes, "
            "argmax=deterministic max-frequency action."
        ),
    )
    parser.add_argument(
        "--policy_rng_seed",
        type=int,
        default=None,
        help=(
            "Optional seeded RNG for sample-mode reproducibility. "
            "Ignored in argmax mode."
        ),
    )
    parser.add_argument(
        "--policy_hybrid_top_prob_threshold",
        type=float,
        default=None,
        help=(
            "Hybrid mode threshold: use deterministic argmax if top probability >= this value. "
            "Default=0.80."
        ),
    )
    parser.add_argument(
        "--policy_hybrid_gap_threshold",
        type=float,
        default=None,
        help=(
            "Hybrid mode threshold: use deterministic argmax if (top - second) >= this value. "
            "Default=0.25."
        ),
    )
    parser.add_argument(
        "--preflop_selection_mode",
        choices=["sample", "argmax"],
        default=None,
        help=(
            "Preflop chart action mode. sample=chart-frequency sampling, "
            "argmax=deterministic highest-frequency action."
        ),
    )
    parser.add_argument(
        "--preflop_rng_seed",
        type=int,
        default=None,
        help="Optional seeded RNG for preflop sample-mode reproducibility.",
    )
    parser.add_argument(
        "--hero-mode",
        "--hero_mode",
        dest="hero_mode",
        choices=["worker", "argmax", "sample", "always_fold", "random"],
        default=None,
        help=(
            "Hero execution mode passed to run_league. "
            "When omitted, auto mode uses worker when either pre/post policy uses sampling."
        ),
    )
    parser.add_argument(
        "--policy-cmd",
        "--policy_cmd",
        dest="policy_cmd",
        default=None,
        help=(
            "Override policy worker command (for example: "
            "\"cargo run --manifest-path solver/Cargo.toml -p player --bin blueprint_policy_worker --release\")."
        ),
    )
    parser.add_argument(
        "--policy-source",
        "--policy_source",
        dest="policy_source",
        choices=["node", "blueprint"],
        default=None,
        help=(
            "Policy source preset. node uses eval/policy_worker.mjs. "
            "blueprint uses Rust blueprint_policy_worker."
        ),
    )
    parser.add_argument(
        "--blueprint-file",
        "--blueprint_file",
        dest="blueprint_file",
        default=None,
        help="Blueprint file path for Rust blueprint worker.",
    )
    parser.add_argument(
        "--blueprint-mode",
        "--blueprint_mode",
        dest="blueprint_mode",
        choices=["sample", "argmax"],
        default=None,
        help="Rust blueprint worker action mode.",
    )
    parser.add_argument(
        "--blueprint-seed",
        "--blueprint_seed",
        dest="blueprint_seed",
        type=int,
        default=None,
        help="Rust blueprint worker RNG seed.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    resolved = cmd
    if sys.platform == "win32":
        exe = cmd[0]
        if exe in {"npm", "npx"}:
            resolved = ["cmd", "/c", exe, *cmd[1:]]
        elif shutil.which(exe) is None and not exe.lower().endswith(".cmd"):
            cmd_exe = f"{exe}.cmd"
            if shutil.which(cmd_exe) is not None:
                resolved = [cmd_exe, *cmd[1:]]

    printable = " ".join(resolved)
    print(f"[smoke-eval] running: {printable}")
    subprocess.run(resolved, check=True, cwd=str(cwd), env=env)


def parse_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    if parsed <= 0:
        return default
    return parsed


def default_jobs() -> int:
    env_jobs = os.environ.get("EVAL_JOBS")
    if env_jobs is not None:
        return parse_positive_int(env_jobs, 1)
    cpu = os.cpu_count() or 1
    return max(1, cpu - 2)


def default_hands(mode: str) -> int:
    return int(DEFAULT_HANDS_BY_MODE.get(mode, DEFAULT_HANDS_BY_MODE["smoke"]))


def resolve_hero_mode(
    *,
    cli_value: str | None,
    env_value: str | None,
    preflop_selection_mode: str,
    policy_selection_mode: str,
) -> tuple[str, str]:
    allowed = {"worker", "argmax", "sample", "always_fold", "random"}
    if cli_value is not None:
        candidate = str(cli_value).strip().lower()
        if candidate in allowed:
            return candidate, "cli"
    if env_value is not None:
        candidate = str(env_value).strip().lower()
        if candidate in allowed:
            return candidate, "env"
    if (
        str(preflop_selection_mode).strip().lower() == "argmax"
        and str(policy_selection_mode).strip().lower() == "argmax"
    ):
        return "argmax", "auto_argmax"
    return "worker", "auto_worker"


def resolve_policy_source(cli_value: str | None, env_value: str | None) -> str:
    allowed = {"node", "blueprint"}
    if cli_value is not None:
        candidate = str(cli_value).strip().lower()
        if candidate in allowed:
            return candidate
    if env_value is not None:
        candidate = str(env_value).strip().lower()
        if candidate in allowed:
            return candidate
    return "node"


def resolve_policy_cmd(
    *,
    cli_value: str | None,
    env_value: str | None,
    policy_source: str,
) -> tuple[str, str]:
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip(), "cli"
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip(), "env"
    if policy_source == "blueprint":
        return (
            "cargo run --manifest-path solver/Cargo.toml -p player --bin blueprint_policy_worker --release",
            "policy_source_blueprint",
        )
    return "node eval/policy_worker.mjs", "default_node"


def resolve_strategy_store_file(repo_root: Path) -> Path | None:
    env_store_raw = os.environ.get("WIPOKER_STRATEGY_STORE_FILE")
    if env_store_raw:
        candidate = Path(env_store_raw.strip())
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        resolved = candidate.resolve(strict=False)
        if resolved.exists() and resolved.is_file():
            return resolved
        return None

    candidates: list[Path] = []

    # Prefer merged stores under eval/strategy_store when available.
    strategy_store_dir = repo_root / "eval" / "strategy_store"
    if strategy_store_dir.exists():
        candidates.extend(
            sorted(
                strategy_store_dir.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )

    # Then check common legacy store paths.
    for rel in (
        "eval/pio_store_big_parallel.jsonl",
        "eval/pio_store_big.jsonl",
        "eval/pio_store_from_missing.jsonl",
    ):
        candidates.append(repo_root / rel)

    # Fall back to generated Pio exports.
    generated_dir = repo_root / "eval" / "generated_pio_exports"
    if generated_dir.exists():
        generated = sorted(
            generated_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        candidates.extend(generated)

    # Last-resort fixture stores.
    for rel in (
        "src/__fixtures__/strategy_store_nodes_concat.json",
        "src/__fixtures__/strategy_store_fuzzy_nodes.jsonl",
    ):
        candidates.append(repo_root / rel)

    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def resolve_strategy_store_env(repo_root: Path) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    explicit_use = os.environ.get("WIPOKER_USE_PIO_STORE")
    explicit_enabled = env_flag("WIPOKER_USE_PIO_STORE", default=False)
    store_path = resolve_strategy_store_file(repo_root)

    if explicit_use is not None:
        if explicit_enabled:
            if store_path is None:
                raise ValueError(
                    "WIPOKER_USE_PIO_STORE is enabled, but no valid "
                    "WIPOKER_STRATEGY_STORE_FILE could be resolved."
                )
            env["WIPOKER_USE_PIO_STORE"] = "1"
            env["WIPOKER_STRATEGY_STORE_FILE"] = str(store_path)
            return env, f"enabled (explicit env): {store_path}"
        env["WIPOKER_USE_PIO_STORE"] = "0"
        return env, "disabled (explicit env)"

    if store_path is not None:
        env["WIPOKER_USE_PIO_STORE"] = "1"
        env["WIPOKER_STRATEGY_STORE_FILE"] = str(store_path)
        return env, f"enabled (auto): {store_path}"

    env["WIPOKER_USE_PIO_STORE"] = "0"
    return env, "disabled (no store file discovered)"


def extract_metrics(
    report_payload: dict[str, Any],
    *,
    run_id: str,
    mode: str,
    hands: int,
    seed: int,
    timestamp_utc: str,
) -> dict[str, Any]:
    summary = report_payload.get("summary", {}) if isinstance(report_payload, dict) else {}
    raw = summary.get("raw", {}) if isinstance(summary, dict) else {}
    net = summary.get("net", {}) if isinstance(summary, dict) else {}
    status_counts = summary.get("status_counts", {}) if isinstance(summary, dict) else {}
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
    config = report_payload.get("config", {}) if isinstance(report_payload, dict) else {}
    strategy_store = (
        diagnostics.get("strategy_store", {}) if isinstance(diagnostics, dict) else {}
    )
    fidelity = (
        diagnostics.get("policy_execution_fidelity", {})
        if isinstance(diagnostics, dict)
        else {}
    )
    translation_audit = (
        diagnostics.get("action_translation_audit", {})
        if isinstance(diagnostics, dict)
        else {}
    )
    preflop_fallback = (
        diagnostics.get("preflop_fallback_unavailable", {})
        if isinstance(diagnostics, dict)
        else {}
    )

    bb_per_100 = net.get("bb_per_100")
    if not isinstance(bb_per_100, (int, float)):
        bb_per_100 = raw.get("bb_per_100", 0.0)

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "eval_mode": mode,
        "hands": int(hands),
        "seed": int(seed),
        "bb_per_100": float(bb_per_100),
        "fallback_count": int(summary.get("fallback_count", 0))
        if isinstance(summary, dict)
        else 0,
        "missing_fields_count": int(status_counts.get("missing_fields", 0))
        if isinstance(status_counts, dict)
        else 0,
        "store_enabled": bool(strategy_store.get("enabled", False))
        if isinstance(strategy_store, dict)
        else False,
        "hero_mode_reported": str(config.get("hero_mode", "unknown"))
        if isinstance(config, dict)
        else "unknown",
        "preflop_selection_mode_reported": str(
            config.get("preflop_selection_mode_effective", "unknown")
        )
        if isinstance(config, dict)
        else "unknown",
        "postflop_selection_mode_reported": str(
            config.get("postflop_selection_mode_effective", "unknown")
        )
        if isinstance(config, dict)
        else "unknown",
    }

    if isinstance(strategy_store, dict):
        metrics["store_hit_count"] = int(strategy_store.get("store_hit_count", 0))
        metrics["store_miss_count"] = int(strategy_store.get("store_miss_count", 0))
        metrics["store_load_ms"] = float(strategy_store.get("store_load_ms", 0.0))
        metrics["store_file_size_bytes"] = int(
            strategy_store.get("store_file_size_bytes", 0)
        )
        metrics["store_quality_exact_hits"] = int(
            strategy_store.get("quality_exact_hits", 0)
        )
        metrics["store_quality_approx_hits"] = int(
            strategy_store.get("quality_approx_hits", 0)
        )
        coverage_hits = int(
            strategy_store.get(
                "coverage_hit_count_total",
                strategy_store.get("store_hit_count", 0),
            )
        )
        coverage_misses = int(
            strategy_store.get(
                "coverage_miss_count",
                strategy_store.get("store_miss_count", 0),
            )
        )
        coverage_attempts = max(
            0,
            int(
                strategy_store.get(
                    "coverage_attempt_count",
                    coverage_hits + coverage_misses,
                )
            ),
        )
        policy_hits = int(
            strategy_store.get(
                "policy_hit_count",
                strategy_store.get("quality_exact_hits", 0),
            )
        )
        policy_attempts = max(
            0,
            int(strategy_store.get("policy_attempt_count", coverage_attempts)),
        )
        if policy_attempts > 0:
            policy_hits = max(0, min(policy_hits, policy_attempts))
        metrics["coverage_hit_count_total"] = coverage_hits
        metrics["coverage_hit_count_exact"] = int(
            strategy_store.get(
                "coverage_hit_count_exact",
                strategy_store.get("quality_exact_hits", 0),
            )
        )
        metrics["coverage_hit_count_approx"] = int(
            strategy_store.get(
                "coverage_hit_count_approx",
                strategy_store.get("quality_approx_hits", 0),
            )
        )
        metrics["coverage_miss_count"] = coverage_misses
        metrics["coverage_attempt_count"] = coverage_attempts
        metrics["coverage_hit_rate_total"] = float(
            strategy_store.get(
                "coverage_hit_rate_total",
                (float(coverage_hits) / float(coverage_attempts))
                if coverage_attempts > 0
                else 0.0,
            )
        )
        metrics["policy_use_approx"] = bool(strategy_store.get("policy_use_approx", False))
        metrics["policy_selection_mode"] = str(
            strategy_store.get("policy_selection_mode", "sample")
        ).strip().lower()
        metrics["policy_hybrid_top_prob_threshold"] = float(
            strategy_store.get("policy_hybrid_top_prob_threshold", 0.8)
        )
        metrics["policy_hybrid_gap_threshold"] = float(
            strategy_store.get("policy_hybrid_gap_threshold", 0.25)
        )
        seed_value = strategy_store.get("policy_selection_seed")
        metrics["policy_selection_seed"] = (
            int(seed_value) if isinstance(seed_value, (int, float)) else None
        )
        metrics["policy_hit_count"] = policy_hits
        metrics["policy_hits_exact"] = int(
            strategy_store.get(
                "policy_hits_exact",
                strategy_store.get("quality_exact_hits", 0),
            )
        )
        metrics["policy_hits_approx"] = int(
            strategy_store.get("policy_hits_approx", 0)
        )
        metrics["policy_miss_count"] = max(0, policy_attempts - policy_hits)
        metrics["policy_attempt_count"] = policy_attempts
        metrics["policy_hit_rate"] = (
            float(policy_hits) / float(policy_attempts) if policy_attempts > 0 else 0.0
        )
        metrics["exact_provenance_match_hits"] = int(
            strategy_store.get("exact_provenance_match_hits", 0)
        )
        metrics["exact_provenance_mismatch_hits"] = int(
            strategy_store.get("exact_provenance_mismatch_hits", 0)
        )
        metrics["exact_provenance_unknown_hits"] = int(
            strategy_store.get("exact_provenance_unknown_hits", 0)
        )

    if isinstance(fidelity, dict):
        metrics["selection_source_counts"] = (
            fidelity.get("selection_source_counts", {})
            if isinstance(fidelity.get("selection_source_counts", {}), dict)
            else {}
        )
        metrics["selection_source_by_street"] = (
            fidelity.get("selection_source_by_street", {})
            if isinstance(fidelity.get("selection_source_by_street", {}), dict)
            else {}
        )
        metrics["exact_hit_and_translated_count"] = int(
            fidelity.get(
                "exact_hit_and_translated_count",
                translation_audit.get("exact_hit_and_translated_count", 0),
            )
        )
        metrics["exact_hit_and_unmodified_count"] = int(
            fidelity.get(
                "exact_hit_and_unmodified_count",
                translation_audit.get("exact_hit_and_unmodified_count", 0),
            )
        )
        metrics["exact_hit_translation_total"] = int(
            fidelity.get(
                "exact_hit_translation_total",
                translation_audit.get("exact_hit_translation_total", 0),
            )
        )
        metrics["exact_hit_unmodified_rate"] = float(
            fidelity.get("exact_hit_unmodified_rate", 0.0)
        )
        metrics["exact_hit_translation_by_street"] = (
            fidelity.get("exact_hit_translation_by_street", {})
            if isinstance(fidelity.get("exact_hit_translation_by_street", {}), dict)
            else {}
        )

    if isinstance(translation_audit, dict):
        metrics["translation_adjusted_count"] = int(
            translation_audit.get("translation_adjusted_count", 0)
        )
        metrics["raise_clamped_count"] = int(
            translation_audit.get("raise_clamped_count", 0)
        )
        metrics["exact_requested_size_used_count"] = int(
            translation_audit.get("exact_requested_size_used_count", 0)
        )

    if isinstance(preflop_fallback, dict):
        metrics["preflop_decision_count"] = int(preflop_fallback.get("decisions", 0))
        metrics["preflop_unavailable_count"] = int(
            preflop_fallback.get("unavailable", 0)
        )
        metrics["preflop_fallback_count"] = int(preflop_fallback.get("fallbacks", 0))

    return metrics


def _read_metric_bb(run_dir: Path) -> float:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return float("-inf")
    try:
        payload = load_json_with_bom_fallback(metrics_path)
    except Exception:
        return float("-inf")
    value = payload.get("bb_per_100")
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


def select_runs_to_keep(run_dirs: list[Path], *, keep_last: int) -> set[Path]:
    if not run_dirs:
        return set()

    ordered = sorted(run_dirs, key=lambda path: path.name)
    complete = [
        path
        for path in ordered
        if any(path.glob("report_*.json"))
    ]

    keep: set[Path] = {ordered[-1]}  # Always keep latest folder, even if incomplete.
    if not complete:
        return keep

    last_n_complete = set(complete[-max(1, keep_last) :])
    best_complete = max(complete, key=_read_metric_bb)
    keep.update(last_n_complete)
    keep.add(best_complete)
    return keep


def prune_runs(*, runs_dir: Path, keep_last: int) -> list[str]:
    run_dirs = [entry for entry in runs_dir.iterdir() if entry.is_dir()]
    keep = select_runs_to_keep(run_dirs, keep_last=keep_last)
    deleted: list[str] = []
    for run_dir in run_dirs:
        if run_dir in keep:
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        deleted.append(run_dir.name)
    return deleted


def remove_heavy_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    hands = args.hands if args.hands is not None else default_hands(args.mode)
    if hands <= 0:
        raise ValueError("--hands must be > 0")
    if args.keep_last <= 0:
        raise ValueError("--keep_last must be > 0")
    jobs = args.jobs if args.jobs and args.jobs > 0 else default_jobs()
    jobs = max(1, min(jobs, hands))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")
    timestamp_utc = now.isoformat()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_name = f"report_{args.mode}.json"
    metrics_name = f"metrics_{args.mode}.json"
    report_path = run_dir / report_name
    store_env, store_mode = resolve_strategy_store_env(REPO_ROOT)
    policy_source = resolve_policy_source(
        cli_value=args.policy_source,
        env_value=os.environ.get("WIPOKER_POLICY_SOURCE"),
    )
    policy_cmd, policy_cmd_resolution = resolve_policy_cmd(
        cli_value=args.policy_cmd,
        env_value=os.environ.get("WIPOKER_POLICY_CMD"),
        policy_source=policy_source,
    )
    if policy_source == "blueprint":
        store_env["WIPOKER_USE_PIO_STORE"] = "0"
        store_env.pop("WIPOKER_STRATEGY_STORE_FILE", None)
        store_mode = "disabled (blueprint policy source)"
    store_env["WIPOKER_POLICY_SOURCE"] = policy_source
    store_env["WIPOKER_POLICY_CMD"] = policy_cmd
    if policy_source == "blueprint":
        if args.blueprint_file is not None:
            store_env["WIPOKER_BLUEPRINT_FILE"] = str(args.blueprint_file)
        elif os.environ.get("WIPOKER_BLUEPRINT_FILE") is not None:
            store_env["WIPOKER_BLUEPRINT_FILE"] = str(
                os.environ.get("WIPOKER_BLUEPRINT_FILE")
            )
        if args.blueprint_mode is not None:
            store_env["WIPOKER_BLUEPRINT_MODE"] = str(args.blueprint_mode)
        elif os.environ.get("WIPOKER_BLUEPRINT_MODE") is not None:
            store_env["WIPOKER_BLUEPRINT_MODE"] = str(
                os.environ.get("WIPOKER_BLUEPRINT_MODE")
            )
        if args.blueprint_seed is not None:
            store_env["WIPOKER_BLUEPRINT_SEED"] = str(int(args.blueprint_seed))
        elif os.environ.get("WIPOKER_BLUEPRINT_SEED") is not None:
            store_env["WIPOKER_BLUEPRINT_SEED"] = str(
                os.environ.get("WIPOKER_BLUEPRINT_SEED")
            )
    env_policy_use_approx = env_flag("WIPOKER_POLICY_USE_APPROX", default=False)
    policy_use_approx = (
        bool(args.policy_use_approx)
        if args.policy_use_approx is not None
        else bool(env_policy_use_approx)
    )
    store_env["WIPOKER_POLICY_USE_APPROX"] = "1" if policy_use_approx else "0"
    env_policy_selection_mode = (
        os.environ.get("WIPOKER_POLICY_SELECTION_MODE")
        or os.environ.get("POLICY_SELECTION_MODE")
    )
    policy_selection_mode = (
        str(args.policy_selection_mode).strip().lower()
        if args.policy_selection_mode is not None
        else str(env_policy_selection_mode or "sample").strip().lower()
    )
    if policy_selection_mode not in {"sample", "argmax", "hybrid"}:
        policy_selection_mode = "sample"
    store_env["WIPOKER_POLICY_SELECTION_MODE"] = policy_selection_mode
    env_hybrid_top_raw = (
        os.environ.get("WIPOKER_POLICY_HYBRID_TOP_PROB_THRESHOLD")
        or os.environ.get("POLICY_HYBRID_TOP_PROB_THRESHOLD")
    )
    env_hybrid_gap_raw = (
        os.environ.get("WIPOKER_POLICY_HYBRID_GAP_THRESHOLD")
        or os.environ.get("POLICY_HYBRID_GAP_THRESHOLD")
    )
    if args.policy_hybrid_top_prob_threshold is not None:
        store_env["WIPOKER_POLICY_HYBRID_TOP_PROB_THRESHOLD"] = str(
            float(args.policy_hybrid_top_prob_threshold)
        )
    elif env_hybrid_top_raw is not None:
        store_env["WIPOKER_POLICY_HYBRID_TOP_PROB_THRESHOLD"] = str(env_hybrid_top_raw)
    if args.policy_hybrid_gap_threshold is not None:
        store_env["WIPOKER_POLICY_HYBRID_GAP_THRESHOLD"] = str(
            float(args.policy_hybrid_gap_threshold)
        )
    elif env_hybrid_gap_raw is not None:
        store_env["WIPOKER_POLICY_HYBRID_GAP_THRESHOLD"] = str(env_hybrid_gap_raw)
    if args.policy_rng_seed is not None:
        store_env["WIPOKER_POLICY_RNG_SEED"] = str(int(args.policy_rng_seed))
    elif os.environ.get("WIPOKER_POLICY_RNG_SEED") is not None:
        store_env["WIPOKER_POLICY_RNG_SEED"] = str(
            os.environ.get("WIPOKER_POLICY_RNG_SEED")
        )
    elif os.environ.get("POLICY_RNG_SEED") is not None:
        store_env["WIPOKER_POLICY_RNG_SEED"] = str(os.environ.get("POLICY_RNG_SEED"))
    env_preflop_mode_raw = (
        os.environ.get("WIPOKER_PREFLOP_SELECTION_MODE")
        or os.environ.get("PREFLOP_SELECTION_MODE")
    )
    preflop_selection_mode = (
        str(args.preflop_selection_mode).strip().lower()
        if args.preflop_selection_mode is not None
        else str(env_preflop_mode_raw or "sample").strip().lower()
    )
    if preflop_selection_mode not in {"sample", "argmax"}:
        preflop_selection_mode = "sample"
    store_env["WIPOKER_PREFLOP_SELECTION_MODE"] = preflop_selection_mode
    if args.preflop_rng_seed is not None:
        store_env["WIPOKER_PREFLOP_RNG_SEED"] = str(int(args.preflop_rng_seed))
    elif os.environ.get("WIPOKER_PREFLOP_RNG_SEED") is not None:
        store_env["WIPOKER_PREFLOP_RNG_SEED"] = str(
            os.environ.get("WIPOKER_PREFLOP_RNG_SEED")
        )
    elif os.environ.get("PREFLOP_RNG_SEED") is not None:
        store_env["WIPOKER_PREFLOP_RNG_SEED"] = str(os.environ.get("PREFLOP_RNG_SEED"))
    env_hero_mode = (
        os.environ.get("WIPOKER_HERO_MODE")
        or os.environ.get("HERO_MODE")
    )
    hero_mode, hero_mode_resolution = resolve_hero_mode(
        cli_value=args.hero_mode,
        env_value=env_hero_mode,
        preflop_selection_mode=preflop_selection_mode,
        policy_selection_mode=policy_selection_mode,
    )
    store_env["WIPOKER_HERO_MODE"] = hero_mode

    if not args.skip_build:
        run_command(["npm", "run", "eval:build"], cwd=REPO_ROOT)

    parallel_started = time.monotonic()
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "eval" / "run_league_parallel.py"),
            "--hands",
            str(hands),
            "--seed",
            str(args.seed),
            "--workers",
            str(jobs),
            "--trace_hands",
            "0",
            "--report_json",
            str(report_path),
            "--hero_mode",
            str(hero_mode),
            "--policy_cmd",
            str(policy_cmd),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        env=store_env,
    )
    parallel_eval_wall_s = max(0.0, time.monotonic() - parallel_started)

    report_payload = load_json_with_bom_fallback(report_path)
    metrics = extract_metrics(
        report_payload,
        run_id=run_id,
        mode=args.mode,
        hands=hands,
        seed=args.seed,
        timestamp_utc=timestamp_utc,
    )
    metrics["preflop_selection_mode"] = preflop_selection_mode
    metrics["preflop_rng_seed"] = (
        int(store_env["WIPOKER_PREFLOP_RNG_SEED"])
        if "WIPOKER_PREFLOP_RNG_SEED" in store_env
        else None
    )
    metrics["hero_mode_effective"] = hero_mode
    metrics["hero_mode_resolution"] = hero_mode_resolution
    metrics["policy_source_effective"] = policy_source
    metrics["policy_cmd_effective"] = policy_cmd
    metrics["policy_cmd_resolution"] = policy_cmd_resolution
    metrics["blueprint_file"] = store_env.get("WIPOKER_BLUEPRINT_FILE")
    metrics["blueprint_mode"] = store_env.get("WIPOKER_BLUEPRINT_MODE")
    metrics["blueprint_seed"] = (
        int(store_env["WIPOKER_BLUEPRINT_SEED"])
        if "WIPOKER_BLUEPRINT_SEED" in store_env
        else None
    )
    metrics["parallel_eval_wall_s"] = float(parallel_eval_wall_s)

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / metrics_name).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Canonical small summaries requested by smoke plumbing.
    (REPORTS_DIR / report_name).write_text(
        json.dumps(report_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORTS_DIR / metrics_name).write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    # Keep latest generic metrics alias for compatibility.
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if not args.retain_heavy:
        parallel_artifacts = []
        for pattern in (
            "*.results.csv",
            "*.trace.jsonl",
            "report_*.part*.json",
        ):
            parallel_artifacts.extend(run_dir.glob(pattern))
        remove_heavy_files(
            [
                REPO_ROOT / "eval" / "results.csv",
                REPO_ROOT / "eval" / "trace.jsonl",
                *parallel_artifacts,
            ]
        )

    deleted_runs = prune_runs(runs_dir=RUNS_DIR, keep_last=args.keep_last)
    if deleted_runs:
        print(f"[smoke-eval] pruned runs: {deleted_runs}")

    print(f"[smoke-eval] run_id={run_id}")
    print(f"[smoke-eval] mode={args.mode} hands={hands}")
    print(f"[smoke-eval] jobs={jobs}")
    print(f"[smoke-eval] strategy_store={store_mode}")
    print(f"[smoke-eval] policy_use_approx={str(policy_use_approx).lower()}")
    print(f"[smoke-eval] policy_selection_mode={policy_selection_mode}")
    print(f"[smoke-eval] policy_source={policy_source}")
    print(
        f"[smoke-eval] policy_cmd={policy_cmd} "
        f"(resolution={policy_cmd_resolution})"
    )
    if "WIPOKER_BLUEPRINT_FILE" in store_env:
        print(f"[smoke-eval] blueprint_file={store_env['WIPOKER_BLUEPRINT_FILE']}")
    if "WIPOKER_BLUEPRINT_MODE" in store_env:
        print(f"[smoke-eval] blueprint_mode={store_env['WIPOKER_BLUEPRINT_MODE']}")
    if "WIPOKER_BLUEPRINT_SEED" in store_env:
        print(f"[smoke-eval] blueprint_seed={store_env['WIPOKER_BLUEPRINT_SEED']}")
    if "WIPOKER_POLICY_HYBRID_TOP_PROB_THRESHOLD" in store_env:
        print(
            "[smoke-eval] policy_hybrid_top_prob_threshold="
            f"{store_env['WIPOKER_POLICY_HYBRID_TOP_PROB_THRESHOLD']}"
        )
    if "WIPOKER_POLICY_HYBRID_GAP_THRESHOLD" in store_env:
        print(
            "[smoke-eval] policy_hybrid_gap_threshold="
            f"{store_env['WIPOKER_POLICY_HYBRID_GAP_THRESHOLD']}"
        )
    if "WIPOKER_POLICY_RNG_SEED" in store_env:
        print(f"[smoke-eval] policy_rng_seed={store_env['WIPOKER_POLICY_RNG_SEED']}")
    print(f"[smoke-eval] preflop_selection_mode={preflop_selection_mode}")
    if "WIPOKER_PREFLOP_RNG_SEED" in store_env:
        print(f"[smoke-eval] preflop_rng_seed={store_env['WIPOKER_PREFLOP_RNG_SEED']}")
    print(
        f"[smoke-eval] hero_mode_effective={hero_mode} "
        f"(resolution={hero_mode_resolution})"
    )
    print(f"[smoke-eval] parallel_eval_wall_s={parallel_eval_wall_s:.2f}")
    print(f"[smoke-eval] report={REPORTS_DIR / report_name}")
    print(f"[smoke-eval] metrics={REPORTS_DIR / metrics_name}")
    print(
        "[smoke-eval] bb_per_100={bb:.2f} fallback_count={fallback} missing_fields_count={missing}".format(
            bb=float(metrics["bb_per_100"]),
            fallback=int(metrics["fallback_count"]),
            missing=int(metrics["missing_fields_count"]),
        )
    )
    if "store_hit_count" in metrics and "store_miss_count" in metrics:
        print(
            "[smoke-eval] store_enabled={enabled} store_hit_count={hit} store_miss_count={miss} "
            "store_quality_exact_hits={quality_exact} store_quality_approx_hits={quality_approx}".format(
                enabled=bool(metrics.get("store_enabled", False)),
                hit=int(metrics["store_hit_count"]),
                miss=int(metrics["store_miss_count"]),
                quality_exact=int(metrics.get("store_quality_exact_hits", 0)),
                quality_approx=int(metrics.get("store_quality_approx_hits", 0)),
            )
        )
    if "coverage_hit_count_total" in metrics and "policy_hit_rate" in metrics:
        print(
            "[smoke-eval] coverage_hits={hit} (exact={exact}, approx={approx}) "
            "coverage_rate={coverage_rate:.4f} policy_rate={policy_rate:.4f} "
            "policy_use_approx={policy_use_approx}".format(
                hit=int(metrics.get("coverage_hit_count_total", 0)),
                exact=int(metrics.get("coverage_hit_count_exact", 0)),
                approx=int(metrics.get("coverage_hit_count_approx", 0)),
                coverage_rate=float(metrics.get("coverage_hit_rate_total", 0.0)),
                policy_rate=float(metrics.get("policy_hit_rate", 0.0)),
                policy_use_approx=bool(metrics.get("policy_use_approx", False)),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
