#!/usr/bin/env python3
"""
Targeted Solve Mode — Cycle Orchestrator

Runs a cycle that:
1. Plays an eval (before_eval) with seed X, collects all missing spots
2. Groups missing spots by board
3. Solves each board in PioSolver, extracts strategy nodes
4. Writes new nodes to approx store, refreshes active store
5. Plays an after_eval with same seed X
6. Reports bb/100 delta and coverage improvement

Usage:
  py -3 eval/run_targeted_solve.py --phase0_root <path> [options]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SPOT_KEY_RE = re.compile(
    r"^street=([^|]+)\|board=(.*?)\|pot=(-?[0-9]+(?:\.[0-9]+)?)\|to_act=([^|]+)\|hist=(.*)$"
)


# ── Utility ───────────────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def safe_rate(num: int, denom: int) -> float:
    return float(num) / float(denom) if denom > 0 else 0.0


# ── ANSI Colors ───────────────────────────────────────────────────────────────

class C:
    """ANSI color codes for console output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"

    @staticmethod
    def delta_color(value: float) -> str:
        if value > 0:
            return C.GREEN
        if value < 0:
            return C.RED
        return C.YELLOW


# ── Console Output ────────────────────────────────────────────────────────────

def print_header(cycle: int, seed: int, max_cycles: int | None) -> None:
    cycle_str = f"{cycle}" if max_cycles is None else f"{cycle}/{max_cycles}"
    print(f"\n{C.BOLD}{C.BG_BLUE}{C.WHITE}"
          f"{'═' * 66}{C.RESET}")
    print(f"{C.BOLD}{C.BG_BLUE}{C.WHITE}"
          f"  TARGETED SOLVE — CYCLE {cycle_str} / seed={seed}"
          f"{' ' * (66 - 35 - len(str(cycle_str)) - len(str(seed)))}{C.RESET}")
    print(f"{C.BOLD}{C.BG_BLUE}{C.WHITE}"
          f"{'═' * 66}{C.RESET}")


def print_phase(phase: str, detail: str = "") -> None:
    phases = {
        "EVAL_BEFORE": C.CYAN,
        "COLLECTING": C.YELLOW,
        "SOLVING": C.GREEN,
        "WRITING": C.YELLOW,
        "EVAL_AFTER": C.CYAN,
        "SUMMARY": C.BOLD,
    }
    color = phases.get(phase, C.WHITE)
    suffix = f"  {detail}" if detail else ""
    try:
        print(f"  {color}\u25b6 {phase}{C.RESET}{suffix}")
    except UnicodeEncodeError:
        print(f"  {color}> {phase}{C.RESET}{suffix}")



def print_progress_bar(current: int, total: int, elapsed_s: float,
                       nodes_so_far: int, board_name: str = "") -> None:
    if total <= 0:
        return
    pct = current / total
    bar_width = 30
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    # ETA
    if current > 0:
        avg_per_board = elapsed_s / current
        remaining = (total - current) * avg_per_board
        eta_str = format_time(remaining)
    else:
        eta_str = "calculating..."

    board_info = f"  {C.DIM}{board_name}{C.RESET}" if board_name else ""
    print(
        f"\r  {C.GREEN}{bar}{C.RESET} {pct * 100:5.1f}%  "
        f"{current}/{total} boards  +{nodes_so_far} nodes  "
        f"ETA: {eta_str}{board_info}    ",
        end="", flush=True,
    )


def print_eval_result(label: str, bb100: float, coverage: float,
                      hands: int, wall_s: float) -> None:
    bb_color = C.GREEN if bb100 >= 0 else C.RED
    print(
        f"  {C.CYAN}{label:14s}{C.RESET} "
        f"{hands} hands  {format_time(wall_s)}  "
        f"bb/100: {bb_color}{bb100:+.2f}{C.RESET}  "
        f"cov: {coverage * 100:.2f}%"
    )


def print_missing_summary(total_spots: int, unique_boards: int,
                          avg_per_board: float) -> None:
    print(
        f"  {C.YELLOW}MISSING{C.RESET}       "
        f"{total_spots} spots → {unique_boards} unique boards "
        f"(avg {avg_per_board:.1f}/board)"
    )


def print_solve_progress(boards_done: int, boards_total: int,
                         nodes: int, elapsed_s: float) -> None:
    rate = elapsed_s / max(1, boards_done)
    print(
        f"  {C.GREEN}SOLVING{C.RESET}       "
        f"{boards_done}/{boards_total} boards  +{nodes} nodes  "
        f"elapsed: {format_time(elapsed_s)}  "
        f"rate: {rate:.1f}s/board"
    )


def print_store_status(exact: int, approx: int, active: int) -> None:
    print(
        f"  {C.DIM}STORE{C.RESET}         "
        f"exact: {exact:,}  approx: {approx:,}  active: {active:,}"
    )


def print_cycle_summary(
    *,
    bb_before: float, bb_after: float,
    cov_before: float, cov_after: float,
    nodes_added: int, spots_solved: int, spots_total: int,
    failures: int,
    exact_count: int, approx_count: int, active_count: int,
    time_eval: float, time_collect: float, time_solve: float,
    time_write: float, time_after_eval: float, total_time: float,
    spots_skipped_by_reason: dict[str, int] | None = None,
) -> None:
    bb_delta = bb_after - bb_before
    cov_delta = cov_after - cov_before
    bb_arrow = "▲" if bb_delta > 0 else ("▼" if bb_delta < 0 else "─")
    cov_arrow = "▲" if cov_delta > 0 else ("▼" if cov_delta < 0 else "─")

    print(f"\n{C.BOLD}{'═' * 66}{C.RESET}")
    print(f"{C.BOLD}  CYCLE COMPLETE — {spots_solved} spots solved "
          f"in {format_time(total_time)}{C.RESET}")
    print(f"{'─' * 66}")
    print(
        f"  {C.BOLD}BB/100{C.RESET}    "
        f"before: {bb_before:+8.2f}   after: {C.delta_color(bb_delta)}{bb_after:+8.2f}{C.RESET}"
        f"    delta: {C.delta_color(bb_delta)}{bb_delta:+.2f}  {bb_arrow}{C.RESET}"
    )
    print(
        f"  {C.BOLD}COVERAGE{C.RESET}  "
        f"before: {cov_before * 100:7.2f}%   after: "
        f"{C.delta_color(cov_delta)}{cov_after * 100:7.2f}%{C.RESET}"
        f"    delta: {C.delta_color(cov_delta)}{cov_delta * 100:+.2f}%  {cov_arrow}{C.RESET}"
    )
    print(
        f"  {C.BOLD}NODES{C.RESET}     "
        f"+{nodes_added} approx  ({spots_solved}/{spots_total} spots solved)"
        + (f"  {C.RED}failures: {failures}{C.RESET}" if failures > 0 else "")
    )
    print(
        f"  {C.BOLD}STORE{C.RESET}     "
        f"exact: {exact_count:,}  approx: {approx_count:,}  "
        f"active: {active_count:,}"
    )
    print(
        f"  {C.BOLD}TIME{C.RESET}      "
        f"eval: {format_time(time_eval)}  collect: {format_time(time_collect)}  "
        f"solve: {format_time(time_solve)}  write: {format_time(time_write)}  "
        f"after: {format_time(time_after_eval)}"
    )
    if spots_solved > 0 and time_solve > 0:
        avg_per_spot = time_solve / spots_solved
        print(f"  {C.BOLD}AVG{C.RESET}       {avg_per_spot:.1f}s per spot")
    if spots_skipped_by_reason:
        skip_parts = [f"{reason}: {count}" for reason, count in spots_skipped_by_reason.items()]
        print(f"  {C.BOLD}SKIPPED{C.RESET}   {', '.join(skip_parts)}")
    print(f"{'=' * 66}\n")


# ── Spot Key Parsing ──────────────────────────────────────────────────────────

def parse_spot_key(spot_key: str) -> dict[str, Any] | None:
    m = SPOT_KEY_RE.match(spot_key.strip())
    if not m:
        return None
    board_raw = m.group(2).strip()
    board_cards = re.findall(r"[2-9TJQKA][shdc]", board_raw, re.IGNORECASE)
    board_str = "".join(board_cards)
    hist_raw = m.group(5).strip()
    history = []
    if hist_raw and hist_raw != "-":
        for entry in hist_raw.split("/"):
            parts = entry.strip().split(":")
            if len(parts) >= 3:
                history.append({
                    "player": parts[0].strip().upper(),
                    "action": parts[1].strip().lower(),
                    "amount": float(parts[2]) if parts[2] else 0.0,
                })
    return {
        "street": m.group(1).strip().upper(),
        "board": board_raw,
        "board_str": board_str,
        "board_cards": board_cards,
        "pot": float(m.group(3)),
        "to_act": m.group(4).strip().upper(),
        "hist_raw": hist_raw,
        "history": history,
    }


def _to_camel_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    """Convert Python snake_case parsed dict to JS camelCase for the worker."""
    return {
        "street": parsed["street"],
        "board": parsed["board"],
        "boardStr": parsed["board_str"],
        "boardCards": parsed["board_cards"],
        "pot": parsed["pot"],
        "toAct": parsed["to_act"],
        "histRaw": parsed["hist_raw"],
        "history": parsed["history"],
    }


def prepare_spot_jobs(
    missing_spots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse each missing spot into a solve job. One job per spot."""
    jobs = []
    for item in missing_spots:
        spot_key = item.get("spot_key", "")
        count = int(item.get("count", 1))
        parsed = parse_spot_key(spot_key)
        if not parsed:
            continue
        jobs.append({
            "spot_key": spot_key,
            "count": count,
            "parsed": parsed,
        })
    # Sort by miss frequency descending
    return sorted(jobs, key=lambda j: -j["count"])


def group_spots_by_board(
    missing_spots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group missing spots by board string, sorted by total miss frequency.

    Kept for backward compatibility with tests. The main pipeline now uses
    group_spots_by_flop_pot_stack() which groups by (flop, initial_pot, effective_stack).
    """
    groups: dict[str, dict[str, Any]] = {}
    for item in missing_spots:
        spot_key = item.get("spot_key", "")
        count = int(item.get("count", 1))
        parsed = parse_spot_key(spot_key)
        if not parsed:
            continue
        board_str = parsed["board_str"]
        if board_str not in groups:
            groups[board_str] = {
                "boardStr": board_str,
                "boardCards": parsed["board_cards"],
                "spots": [],
                "totalCount": 0,
            }
        groups[board_str]["spots"].append({
            "spot_key": spot_key,
            "count": count,
            "parsed": parsed,
        })
        groups[board_str]["totalCount"] += count
    return sorted(groups.values(), key=lambda g: -g["totalCount"])


def group_spots_by_flop_pot_stack(
    missing_spots: list[dict[str, Any]],
    effective_stack: int = 10000,
    cfr_steps: int = 500,
    exclude_spot_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Group spots by (flop_3_cards, initial_pot, effective_stack).

    Returns list of BoardGroup dicts matching the new worker input format.
    Each group contains all spots sharing the same flop+pot+stack key.
    Spots in exclude_spot_keys (e.g. chance boundaries) are filtered out.
    """
    excluded = exclude_spot_keys or set()
    groups: dict[tuple, dict[str, Any]] = {}
    for item in missing_spots:
        spot_key = item.get("spot_key", "")
        if spot_key in excluded:
            continue
        count = int(item.get("count", 1))
        parsed = parse_spot_key(spot_key)
        if not parsed:
            continue

        board_cards = parsed["board_cards"]
        flop = "".join(board_cards[:3]) if len(board_cards) >= 3 else "".join(board_cards)

        # Compute initial pot: pot before any postflop action
        hist_total = sum(h.get("amount", 0) for h in parsed["history"])
        initial_pot = round(parsed["pot"] - hist_total)

        key = (flop, initial_pot, effective_stack)
        if key not in groups:
            groups[key] = {
                "flop": flop,
                "initial_pot": initial_pot,
                "effective_stack": effective_stack,
                "cfr_steps": cfr_steps,
                "spots": [],
                "_total_count": 0,
            }
        groups[key]["spots"].append({
            "spot_key": spot_key,
            "count": count,
            "parsed": _to_camel_parsed(parsed),
        })
        groups[key]["_total_count"] += count

    # Sort by total miss frequency descending, strip internal field
    result = sorted(groups.values(), key=lambda g: -g["_total_count"])
    for g in result:
        del g["_total_count"]
    return result


# ── Eval Step ─────────────────────────────────────────────────────────────────

def run_eval(
    *,
    repo_root: Path,
    cycle_dir: Path,
    label: str,
    mode: str,
    hands: int,
    seed: int,
    active_store_path: Path,
    policy_use_approx: bool,
    policy_selection_mode: str,
    policy_rng_seed: int | None,
    hero_mode: str,
) -> dict[str, Any]:
    """Run an eval and return metrics including bb/100, coverage, and missing spots."""
    env = os.environ.copy()
    env["WIPOKER_USE_PIO_STORE"] = "1"
    env["WIPOKER_STRATEGY_STORE_FILE"] = str(active_store_path)
    env["WIPOKER_STORE_MISSING_TOP_N"] = "5000"
    env["WIPOKER_STORE_INCLUDE_APPROX"] = "1"
    env["WIPOKER_POLICY_USE_APPROX"] = "1" if policy_use_approx else "0"
    env["WIPOKER_POLICY_SELECTION_MODE"] = policy_selection_mode
    env["WIPOKER_POLICY_WORKER_HEAVY_DIAGNOSTICS"] = "1"
    env["WIPOKER_STORE_DIAGNOSTICS_TOPLIST_INTERVAL"] = "1"
    if policy_rng_seed is not None:
        env["WIPOKER_POLICY_RNG_SEED"] = str(policy_rng_seed)
    env["WIPOKER_HERO_MODE"] = hero_mode
    env["WIPOKER_PREFLOP_SELECTION_MODE"] = "sample"

    cmd = [
        sys.executable,
        str(repo_root / "eval" / "run_smoke_eval.py"),
        "--mode", mode,
        "--hands", str(hands),
        "--seed", str(seed),
        "--jobs", "1",
        "--skip_build",
        "--policy_use_approx" if policy_use_approx else "--policy_exact_only",
        "--policy_selection_mode", policy_selection_mode,
        "--hero_mode", hero_mode,
    ]
    if policy_rng_seed is not None:
        cmd.extend(["--policy_rng_seed", str(policy_rng_seed)])

    eval_log = cycle_dir / f"{label}_eval.log"
    started = time.monotonic()
    result = subprocess.run(
        cmd, cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=600,
    )
    wall_s = time.monotonic() - started

    if result.returncode != 0:
        eval_log.write_text(
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Eval {label} failed (exit={result.returncode}): "
            f"{result.stderr[:500]}"
        )

    eval_log.write_text(
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )

    # Load report and metrics
    report_path = repo_root / "reports" / f"report_{mode}.json"
    metrics_path = repo_root / "reports" / f"metrics_{mode}.json"
    report = load_json(report_path, {})
    metrics = load_json(metrics_path, {})

    # Copy to cycle dir
    shutil.copy2(report_path, cycle_dir / f"{label}_report.json")
    shutil.copy2(metrics_path, cycle_dir / f"{label}_metrics.json")

    # Extract diagnostics
    summary = report.get("summary", {})
    diagnostics = summary.get("diagnostics", {})
    store_diag = diagnostics.get("strategy_store", {})

    bb_per_100 = float(metrics.get("bb_per_100", 0.0))
    coverage_hits = int(store_diag.get("coverage_hit_count_total",
                                       store_diag.get("store_hit_count", 0)))
    coverage_attempts = int(store_diag.get("coverage_attempt_count", 0))
    coverage_rate = safe_rate(coverage_hits, coverage_attempts)

    # Extract missing spot keys — prefer all_missing_spot_keys (full list)
    missing_spots = store_diag.get("all_missing_spot_keys",
                                   store_diag.get("top_missing_spot_keys", []))
    if not isinstance(missing_spots, list):
        missing_spots = []

    return {
        "bb_per_100": bb_per_100,
        "coverage_rate": coverage_rate,
        "coverage_hits": coverage_hits,
        "coverage_attempts": coverage_attempts,
        "missing_spots": missing_spots,
        "wall_s": wall_s,
        "hands": hands,
        "node_count": int(store_diag.get("node_count", 0)),
    }


# ── Real-Time Progress ────────────────────────────────────────────────────────

def _parse_worker_progress(line: str) -> dict[str, Any] | None:
    """Parse a worker stderr line for progress info."""
    line = line.strip()
    if not line:
        return None

    # Match: [targeted-solve] board_done=X/Y flop=ABC nodes=N failures=F
    m = re.match(
        r"\[targeted-solve\] board_done=(\d+)/(\d+)\s+flop=(\S+)\s+nodes=(\d+)\s+failures=(\d+)",
        line,
    )
    if m:
        return {
            "type": "board_done",
            "current": int(m.group(1)),
            "total": int(m.group(2)),
            "flop": m.group(3),
            "nodes": int(m.group(4)),
            "failures": int(m.group(5)),
        }

    # Match: [targeted-solve] solveBoard flop=ABC ...
    m2 = re.match(r"\[targeted-solve\] solveBoard flop=(\S+)", line)
    if m2:
        return {"type": "solving", "flop": m2.group(1)}

    # Match: [targeted-solve] complete boards=X spots=Y/Z nodes=N failures=F
    m3 = re.match(
        r"\[targeted-solve\] complete boards=(\d+) spots=(\d+)/(\d+) nodes=(\d+) failures=(\d+)",
        line,
    )
    if m3:
        return {
            "type": "complete",
            "boards": int(m3.group(1)),
            "spots_done": int(m3.group(2)),
            "spots_total": int(m3.group(3)),
            "nodes": int(m3.group(4)),
            "failures": int(m3.group(5)),
        }

    return None


def _render_progress_bar(
    *,
    boards_done: int,
    boards_total: int,
    nodes_so_far: int,
    failures_so_far: int,
    elapsed_s: float,
    current_flop: str = "",
    worker_id: str = "",
) -> str:
    """Render a single-line progress bar string."""
    if boards_total <= 0:
        return ""
    pct = boards_done / boards_total
    bar_width = 25
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    if boards_done > 0:
        avg_per_board = elapsed_s / boards_done
        remaining = (boards_total - boards_done) * avg_per_board
        eta_str = format_time(remaining)
    else:
        eta_str = "..."

    prefix = f"W{worker_id} " if worker_id else ""
    flop_str = f" [{current_flop}]" if current_flop else ""
    fail_str = f"  {C.RED}fail:{failures_so_far}{C.RESET}" if failures_so_far > 0 else ""

    return (
        f"  {prefix}{C.GREEN}{bar}{C.RESET} {pct * 100:5.1f}%  "
        f"{boards_done}/{boards_total} boards  +{nodes_so_far} nodes"
        f"{fail_str}  ETA: {eta_str}{flop_str}"
    )


def _stream_worker_with_progress(
    *,
    proc: subprocess.Popen,
    stderr_pipe,
    log_file,
    boards_total: int,
    worker_id: str = "",
) -> dict[str, Any]:
    """Stream worker stderr, parse progress, render live progress bar."""
    state = {
        "boards_done": 0,
        "boards_total": boards_total,
        "nodes_so_far": 0,
        "failures_so_far": 0,
        "current_flop": "",
        "started": time.monotonic(),
    }

    for raw_line in iter(stderr_pipe.readline, ""):
        log_file.write(raw_line)
        log_file.flush()

        progress = _parse_worker_progress(raw_line)
        if not progress:
            continue

        if progress["type"] == "board_done":
            state["boards_done"] = progress["current"]
            state["nodes_so_far"] += progress["nodes"]
            state["failures_so_far"] += progress["failures"]
            state["current_flop"] = progress["flop"]
        elif progress["type"] == "solving":
            state["current_flop"] = progress["flop"]
        elif progress["type"] == "complete":
            state["boards_done"] = progress["boards"]
            state["nodes_so_far"] = progress["nodes"]
            state["failures_so_far"] = progress["failures"]

        elapsed = time.monotonic() - state["started"]
        bar = _render_progress_bar(
            boards_done=state["boards_done"],
            boards_total=state["boards_total"],
            nodes_so_far=state["nodes_so_far"],
            failures_so_far=state["failures_so_far"],
            elapsed_s=elapsed,
            current_flop=state["current_flop"],
            worker_id=worker_id,
        )
        if bar:
            print(f"\r{bar}    ", end="", flush=True)

    return state


# ── Solve Step ────────────────────────────────────────────────────────────────

def run_solve_step(
    *,
    repo_root: Path,
    cycle_dir: Path,
    spot_jobs: list[dict[str, Any]] | None = None,
    board_groups: list[dict[str, Any]] | None = None,
    pio_exe: str,
    cfr_steps: int,
    effective_stack: int,
    timeout_ms: int,
    parallel: int = 1,
) -> dict[str, Any]:
    """Call targeted_solve_worker.mjs to solve spots and extract nodes.

    Supports both legacy flat spot_jobs and new grouped board_groups input.
    When board_groups is provided and parallel > 1, uses parallel workers.
    """
    # Parallel mode with grouped input
    if board_groups is not None and parallel > 1 and len(board_groups) > 1:
        return run_parallel_workers(
            repo_root=repo_root,
            cycle_dir=cycle_dir,
            board_groups=board_groups,
            n_workers=parallel,
            pio_exe=pio_exe,
            timeout_ms=timeout_ms,
        )

    if board_groups is not None:
        # New grouped format
        worker_input = {
            "pio_exe": pio_exe,
            "boards": board_groups,
            "timeout_ms": timeout_ms,
        }
        total_spots = sum(len(g.get("spots", [])) for g in board_groups)
    else:
        # Legacy flat format
        worker_spots = []
        for job in (spot_jobs or []):
            worker_spots.append({
                "spot_key": job["spot_key"],
                "count": job["count"],
                "parsed": _to_camel_parsed(job["parsed"]),
            })
        worker_input = {
            "pio_exe": pio_exe,
            "spots": worker_spots,
            "cfr_steps": cfr_steps,
            "effective_stack": effective_stack,
            "timeout_ms": timeout_ms,
        }
        total_spots = len(worker_spots)

    input_path = cycle_dir / "targeted_solve_input.json"
    write_json(input_path, worker_input)

    output_path = cycle_dir / "targeted_solve_output.jsonl"
    worker_log = cycle_dir / "targeted_solve_worker.log"

    worker_script = str(repo_root / "eval" / "targeted_solve_worker.mjs")
    cmd = f'node "{worker_script}"'

    boards_count = len(board_groups) if board_groups else 0

    started = time.monotonic()
    with open(input_path, "r", encoding="utf-8") as stdin_file:
        with open(output_path, "w", encoding="utf-8") as stdout_file:
            with open(worker_log, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    stdin=stdin_file,
                    stdout=stdout_file,
                    stderr=subprocess.PIPE,
                    shell=True,
                    text=True,
                    bufsize=1,
                )
                _stream_worker_with_progress(
                    proc=proc,
                    stderr_pipe=proc.stderr,
                    log_file=log_file,
                    boards_total=boards_count,
                )
                proc.wait(timeout=max(300, timeout_ms // 1000 * total_spots + 300))
    print()  # newline after progress bar
    wall_s = time.monotonic() - started

    if proc.returncode != 0:
        log_content = worker_log.read_text(encoding="utf-8") if worker_log.exists() else ""
        raise RuntimeError(
            f"Targeted solve worker failed (exit={proc.returncode}): "
            f"{log_content[:500]}"
        )

    # Parse output JSONL — each line is a node or the final summary
    nodes = []
    summary_line = None
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "summary":
                    summary_line = obj
                else:
                    nodes.append(obj)
            except json.JSONDecodeError:
                continue

    return {
        "nodes": nodes,
        "summary": summary_line or {},
        "wall_s": wall_s,
        "worker_log": str(worker_log),
        "input_path": str(input_path),
        "output_path": str(output_path),
    }


# ── Parallel Workers ──────────────────────────────────────────────────────────

def auto_detect_parallel() -> int:
    """Auto-detect optimal parallelism based on CPU cores and available memory.

    Each PioSolver instance uses ~2-4 GB RAM.
    Use cpu_count // 3 (floor 2, cap 6) to keep sustained CPU usage around 70-80%.
    """
    cpu_count = os.cpu_count() or 2
    cpu_based = max(2, cpu_count // 3)

    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_gb = mem.available / (1024 ** 3)
        mem_based = max(1, int(ram_gb // 3))
    except ImportError:
        # Fallback: assume 16 GB available
        mem_based = 5

    result = min(cpu_based, mem_based, 6)
    return max(2, result)


def partition_board_groups(
    board_groups: list[dict[str, Any]],
    n_workers: int,
) -> list[list[dict[str, Any]]]:
    """Partition board groups across N workers using round-robin distribution."""
    if n_workers <= 0:
        n_workers = 1
    n_workers = min(n_workers, len(board_groups))
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for i, group in enumerate(board_groups):
        partitions[i % n_workers].append(group)
    return [p for p in partitions if p]  # remove empty partitions


def merge_worker_outputs(
    output_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge JSONL output from multiple workers into a single node list + summary."""
    all_nodes = []
    merged_summary = {
        "type": "summary",
        "spots_solved": 0,
        "spots_total": 0,
        "nodes_extracted": 0,
        "failures": 0,
        "spots_skipped_by_reason": {},
    }
    for path in output_paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "summary":
                    merged_summary["spots_solved"] += obj.get("spots_solved", 0)
                    merged_summary["spots_total"] += obj.get("spots_total", 0)
                    merged_summary["nodes_extracted"] += obj.get("nodes_extracted", 0)
                    merged_summary["failures"] += obj.get("failures", 0)
                    for reason, count in obj.get("spots_skipped_by_reason", {}).items():
                        merged_summary["spots_skipped_by_reason"][reason] = (
                            merged_summary["spots_skipped_by_reason"].get(reason, 0) + count
                        )
                else:
                    all_nodes.append(obj)
            except json.JSONDecodeError:
                continue
    return all_nodes, merged_summary


def run_parallel_workers(
    *,
    repo_root: Path,
    cycle_dir: Path,
    board_groups: list[dict[str, Any]],
    n_workers: int,
    pio_exe: str,
    timeout_ms: int,
) -> dict[str, Any]:
    """Spawn N worker processes, each with a partition of board groups.

    Streams stderr from all workers for real-time progress display.
    Collects and merges JSONL output from all workers.
    """
    import threading

    partitions = partition_board_groups(board_groups, n_workers)
    actual_workers = len(partitions)

    processes = []
    output_paths = []
    worker_logs = []
    worker_states = []
    lock = threading.Lock()

    worker_script = str(repo_root / "eval" / "targeted_solve_worker.mjs")
    started = time.monotonic()

    for i, partition in enumerate(partitions):
        input_path = cycle_dir / f"worker_{i}_input.json"
        output_path = cycle_dir / f"worker_{i}_output.jsonl"
        log_path = cycle_dir / f"worker_{i}.log"

        worker_input = {
            "pio_exe": pio_exe,
            "boards": partition,
            "timeout_ms": timeout_ms,
        }
        write_json(input_path, worker_input)

        cmd = f'node "{worker_script}"'
        total_spots = sum(len(g.get("spots", [])) for g in partition)
        proc_timeout = max(300, timeout_ms // 1000 * total_spots + 300)

        stdin_file = open(input_path, "r", encoding="utf-8")
        stdout_file = open(output_path, "w", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdin=stdin_file,
            stdout=stdout_file,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            bufsize=1,
        )

        state = {
            "boards_done": 0,
            "boards_total": len(partition),
            "nodes_so_far": 0,
            "failures_so_far": 0,
            "current_flop": "",
            "finished": False,
        }
        worker_states.append(state)

        processes.append({
            "proc": proc,
            "index": i,
            "stdin_file": stdin_file,
            "stdout_file": stdout_file,
            "timeout": proc_timeout,
            "log_path": log_path,
        })
        output_paths.append(output_path)
        worker_logs.append(log_path)

    # Render combined progress for all workers
    def render_combined():
        elapsed = time.monotonic() - started
        total_boards = sum(s["boards_total"] for s in worker_states)
        done_boards = sum(s["boards_done"] for s in worker_states)
        total_nodes = sum(s["nodes_so_far"] for s in worker_states)
        total_fails = sum(s["failures_so_far"] for s in worker_states)
        active_flops = [s["current_flop"] for s in worker_states if s["current_flop"] and not s["finished"]]

        bar = _render_progress_bar(
            boards_done=done_boards,
            boards_total=total_boards,
            nodes_so_far=total_nodes,
            failures_so_far=total_fails,
            elapsed_s=elapsed,
            current_flop=", ".join(active_flops[:3]),
            worker_id=f"×{actual_workers}",
        )
        if bar:
            print(f"\r{bar}    ", end="", flush=True)

    # Stream stderr from each worker in a thread
    def stream_worker(entry, state):
        log_file = open(entry["log_path"], "w", encoding="utf-8")
        try:
            for raw_line in iter(entry["proc"].stderr.readline, ""):
                log_file.write(raw_line)
                log_file.flush()

                progress = _parse_worker_progress(raw_line)
                if not progress:
                    continue

                with lock:
                    if progress["type"] == "board_done":
                        state["boards_done"] = progress["current"]
                        state["nodes_so_far"] += progress["nodes"]
                        state["failures_so_far"] += progress["failures"]
                        state["current_flop"] = progress["flop"]
                    elif progress["type"] == "solving":
                        state["current_flop"] = progress["flop"]
                    elif progress["type"] == "complete":
                        state["boards_done"] = progress["boards"]
                        state["nodes_so_far"] = progress["nodes"]
                        state["failures_so_far"] = progress["failures"]
                        state["finished"] = True

                    render_combined()
        finally:
            log_file.close()

    threads = []
    for entry, state in zip(processes, worker_states):
        t = threading.Thread(target=stream_worker, args=(entry, state), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all workers to complete
    worker_failures = 0
    for entry in processes:
        proc = entry["proc"]
        try:
            proc.wait(timeout=entry["timeout"])
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"\n  {C.RED}Worker {entry['index']} timed out{C.RESET}")
            worker_failures += 1
        finally:
            entry["stdin_file"].close()
            entry["stdout_file"].close()

        if proc.returncode != 0 and proc.returncode is not None:
            log_content = worker_logs[entry["index"]].read_text(encoding="utf-8")[:500] if worker_logs[entry["index"]].exists() else ""
            print(f"\n  {C.YELLOW}Worker {entry['index']} failed (exit={proc.returncode}): {log_content[:200]}{C.RESET}")
            worker_failures += 1

    # Wait for all streaming threads to finish
    for t in threads:
        t.join(timeout=5)

    print()  # newline after progress bar
    wall_s = time.monotonic() - started

    # Merge outputs
    all_nodes, merged_summary = merge_worker_outputs(output_paths)

    return {
        "nodes": all_nodes,
        "summary": merged_summary,
        "wall_s": wall_s,
        "worker_failures": worker_failures,
        "actual_workers": actual_workers,
    }


# ── Store Write Step ──────────────────────────────────────────────────────────

def _validate_node(node: dict[str, Any]) -> str | None:
    """Validate a node before writing to store. Returns error message or None."""
    if not node.get("spot_key"):
        return "missing spot_key"
    strategy = node.get("strategy")
    if not strategy or not isinstance(strategy, dict):
        return "missing strategy"
    by_child = strategy.get("by_child")
    if not isinstance(by_child, list) or len(by_child) == 0:
        return "empty strategy.by_child"
    combo_labels = strategy.get("combo_labels")
    if not isinstance(combo_labels, list) or len(combo_labels) == 0:
        return "empty combo_labels"
    for i, entry in enumerate(by_child):
        token = str(entry.get("token", "")).strip() if isinstance(entry, dict) else ""
        if not token:
            return f"empty token at by_child[{i}]"
        freqs = entry.get("frequencies") if isinstance(entry, dict) else None
        if not isinstance(freqs, list) or len(freqs) != len(combo_labels):
            return f"frequency length mismatch at by_child[{i}] token={token}"
    return None


def write_nodes_to_store(
    nodes: list[dict[str, Any]],
    approx_store_path: Path,
) -> int:
    """Append new nodes to the approx store JSONL file. Rejects invalid nodes."""
    approx_store_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rejected = 0
    with open(approx_store_path, "a", encoding="utf-8") as f:
        for node in nodes:
            err = _validate_node(node)
            if err:
                spot = node.get("spot_key", "?")
                print(f"  {C.YELLOW}REJECT{C.RESET} {spot}: {err}")
                rejected += 1
                continue
            f.write(json.dumps(node, separators=(",", ":")) + "\n")
            written += 1
    if rejected > 0:
        print(f"  {C.YELLOW}Rejected {rejected} invalid node(s){C.RESET}")
    return written


def refresh_active_store(
    *,
    repo_root: Path,
    cycle_dir: Path,
    active_store: Path,
    cumulative_exact_store: Path,
    cumulative_approx_store: Path,
) -> dict[str, Any]:
    """Refresh the active store by merging exact + approx stores."""
    cmd = [
        sys.executable,
        str(repo_root / "eval" / "refresh_strategy_store.py"),
        "--out_store", str(active_store),
        "--no_default_source_dirs",
        "--include_approx",
    ]
    if cumulative_exact_store.exists():
        cmd.extend(["--source_file", str(cumulative_exact_store)])
    if cumulative_approx_store.exists():
        cmd.extend(["--source_file", str(cumulative_approx_store)])

    log_path = cycle_dir / "refresh_active_store.log"
    result = subprocess.run(
        cmd, cwd=str(repo_root),
        capture_output=True, text=True, timeout=120,
    )
    log_path.write_text(
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Store refresh failed: {result.stderr[:300]}")

    return {
        "log_path": str(log_path),
        "active_store_rows": count_jsonl_rows(active_store),
    }


# ── CLI Args ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Targeted Solve Mode — solve exactly the missing spots"
    )
    parser.add_argument(
        "--phase0_root", type=Path, required=True,
        help="Root directory for phase0 stores (e.g. %%LOCALAPPDATA%%\\WiPoker\\phase0_store)",
    )
    parser.add_argument("--mode", default="smoke", help="Eval mode (default: smoke)")
    parser.add_argument("--eval_hands", type=int, default=8000, help="Hands per eval")
    parser.add_argument("--max_solves", type=int, default=None,
                        help="Max boards to solve per cycle (for testing)")
    parser.add_argument("--cfr_steps", type=int, default=500, help="CFR iterations")
    parser.add_argument("--effective_stack", type=int, default=10000, help="Starting stack in cents (100 BB = 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--max_cycles", type=int, default=1, help="Number of cycles")
    parser.add_argument("--timeout_ms", type=int, default=600_000,
                        help="PioSolver command timeout (ms)")
    parser.add_argument("--policy_selection_mode", default="hybrid")
    parser.add_argument("--policy_rng_seed", type=int, default=42)
    parser.add_argument("--hero_mode", default="worker")
    parser.add_argument("--parallel", type=str, default="auto",
                        help="Number of concurrent PioSolver worker processes (default: auto-detect)")
    parser.add_argument("--auto_converge", action="store_true",
                        help="Run cycles until convergence condition is met")
    parser.add_argument("--validation_test", action="store_true",
                        help="Run end-to-end validation with multiple seeds")
    return parser.parse_args()


# ── Convergence Check ─────────────────────────────────────────────────────────

def check_convergence(
    cycle_history: list[dict[str, Any]],
    max_cycles: int | None = None,
) -> str | None:
    """Evaluate termination conditions in priority order.

    Returns the first satisfied condition name, or None if none are satisfied.
    Priority: (a) coverage > 99%, (b) improvement < 0.5pp, (c) missing < 10, (d) max cycles.

    Note: after_coverage is stored as a 0-1 float (e.g. 0.85 = 85%).
    """
    if not cycle_history:
        return None

    latest = cycle_history[-1]
    coverage = latest.get("after_coverage", 0)  # 0-1 scale
    missing = latest.get("missing_spots", float("inf"))

    # (a) Coverage > 99%
    if coverage > 0.99:
        return "coverage_99pct"

    # (b) Coverage improvement < 0.5 percentage points (0.005 in 0-1 scale)
    if len(cycle_history) >= 2:
        prev_coverage = cycle_history[-2].get("after_coverage", 0)
        improvement = coverage - prev_coverage
        if improvement < 0.005:
            return "improvement_below_threshold"

    # (c) Missing spots < 10
    if missing < 10:
        return "missing_below_10"

    # (d) Max cycles reached
    if max_cycles is not None and len(cycle_history) >= max_cycles:
        return "max_cycles"

    return None


# ── Main Cycle ────────────────────────────────────────────────────────────────

def run_cycle(
    *,
    args: argparse.Namespace,
    cycle_index: int,
    phase0_root: Path,
    pio_exe: str,
    exclude_spot_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Run a single targeted solve cycle."""
    seed = args.seed
    cycle_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cycle_dir = phase0_root / "targeted_solve_reports" / f"cycle_{cycle_stamp}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    active_store = phase0_root / "active_store.jsonl"
    cumulative_exact_store = phase0_root / "pio_store_autopilot_loop.jsonl"
    cumulative_approx_store = phase0_root / "pio_store_autopilot_loop_approx.jsonl"

    cycle_started = time.monotonic()
    timing: dict[str, float] = {}

    print_header(cycle_index, seed, args.max_cycles if args.max_cycles > 1 else None)

    # ── Phase 1: Refresh store and run before_eval ────────────────────────
    print_phase("EVAL_BEFORE", f"seed={seed} hands={args.eval_hands}")
    t0 = time.monotonic()

    refresh_active_store(
        repo_root=REPO_ROOT,
        cycle_dir=cycle_dir,
        active_store=active_store,
        cumulative_exact_store=cumulative_exact_store,
        cumulative_approx_store=cumulative_approx_store,
    )

    before_eval = run_eval(
        repo_root=REPO_ROOT,
        cycle_dir=cycle_dir,
        label="before",
        mode=args.mode,
        hands=args.eval_hands,
        seed=seed,
        active_store_path=active_store,
        policy_use_approx=True,
        policy_selection_mode=args.policy_selection_mode,
        policy_rng_seed=args.policy_rng_seed,
        hero_mode=args.hero_mode,
    )
    timing["eval_before"] = time.monotonic() - t0

    print_eval_result(
        "EVAL BEFORE",
        before_eval["bb_per_100"],
        before_eval["coverage_rate"],
        before_eval["hands"],
        before_eval["wall_s"],
    )

    # ── Phase 2: Collect and group missing spots ──────────────────────────
    print_phase("COLLECTING")
    t0 = time.monotonic()

    missing_spots = before_eval["missing_spots"]
    if not missing_spots:
        print(f"  {C.GREEN}No missing spots — store has full coverage.{C.RESET}")
        timing["collect"] = time.monotonic() - t0
        return {"status": "no_missing", "cycle_dir": str(cycle_dir)}

    # Group spots by (flop, initial_pot, effective_stack) for tree reuse
    board_groups = group_spots_by_flop_pot_stack(
        missing_spots,
        effective_stack=args.effective_stack,
        cfr_steps=args.cfr_steps,
        exclude_spot_keys=exclude_spot_keys,
    )

    # Apply --max_solves limit (for testing only) — limit board groups
    if args.max_solves is not None and args.max_solves > 0:
        limited_groups = []
        spots_so_far = 0
        for g in board_groups:
            if spots_so_far >= args.max_solves:
                break
            limited_groups.append(g)
            spots_so_far += len(g["spots"])
        board_groups = limited_groups

    total_spots = sum(len(g["spots"]) for g in board_groups)
    unique_flops = len({g["flop"] for g in board_groups})
    timing["collect"] = time.monotonic() - t0

    print(
        f"  {C.YELLOW}MISSING{C.RESET}       "
        f"{total_spots} spots across {len(board_groups)} board groups ({unique_flops} unique flops)"
    )

    # ── Phase 3: Solve ────────────────────────────────────────────────────
    parallel = getattr(args, "parallel", 1)
    print_phase("SOLVING", f"{total_spots} spots, {len(board_groups)} groups, cfr={args.cfr_steps}, parallel={parallel}")
    t0 = time.monotonic()

    solve_result = run_solve_step(
        repo_root=REPO_ROOT,
        cycle_dir=cycle_dir,
        board_groups=board_groups,
        pio_exe=pio_exe,
        cfr_steps=args.cfr_steps,
        effective_stack=args.effective_stack,
        timeout_ms=args.timeout_ms,
        parallel=parallel,
    )
    timing["solve"] = time.monotonic() - t0

    nodes = solve_result["nodes"]
    solve_summary = solve_result.get("summary", {})
    spots_solved = int(solve_summary.get("spots_solved", 0))
    failures = int(solve_summary.get("failures", 0))
    chance_boundary_keys = solve_summary.get("chance_boundary_spot_keys", [])

    rate = timing["solve"] / max(1, spots_solved)
    print(
        f"  {C.GREEN}SOLVING{C.RESET}       "
        f"{spots_solved}/{total_spots} spots  +{len(nodes)} nodes  "
        f"elapsed: {format_time(timing['solve'])}  "
        f"rate: {rate:.1f}s/spot"
    )

    # Show worker log if there were failures
    if failures > 0:
        worker_log_path = solve_result.get("worker_log", "")
        input_json_path = solve_result.get("input_path", "")
        if input_json_path:
            print(f"  {C.DIM}Worker input: {input_json_path}{C.RESET}")
        if worker_log_path:
            try:
                log_text = Path(worker_log_path).read_text(encoding="utf-8").strip()
                if log_text:
                    print(f"\n  {C.YELLOW}── Worker Log ──{C.RESET}")
                    for log_line in log_text.splitlines()[-30:]:
                        print(f"  {C.DIM}{log_line}{C.RESET}")
                    print()
                else:
                    print(f"  {C.YELLOW}Worker log is empty — worker may have crashed before writing.{C.RESET}")
            except Exception:
                pass

    # ── Phase 4: Write to store ───────────────────────────────────────────
    print_phase("WRITING", f"{len(nodes)} nodes → approx store")
    t0 = time.monotonic()

    written = write_nodes_to_store(nodes, cumulative_approx_store)

    refresh_result = refresh_active_store(
        repo_root=REPO_ROOT,
        cycle_dir=cycle_dir,
        active_store=active_store,
        cumulative_exact_store=cumulative_exact_store,
        cumulative_approx_store=cumulative_approx_store,
    )
    timing["write"] = time.monotonic() - t0

    exact_count = count_jsonl_rows(cumulative_exact_store)
    approx_count = count_jsonl_rows(cumulative_approx_store)
    active_count = int(refresh_result.get("active_store_rows", 0))

    print_store_status(exact_count, approx_count, active_count)

    # ── Phase 5: After eval ───────────────────────────────────────────────
    print_phase("EVAL_AFTER", f"seed={seed} (same seed)")
    t0 = time.monotonic()

    after_eval = run_eval(
        repo_root=REPO_ROOT,
        cycle_dir=cycle_dir,
        label="after",
        mode=args.mode,
        hands=args.eval_hands,
        seed=seed,
        active_store_path=active_store,
        policy_use_approx=True,
        policy_selection_mode=args.policy_selection_mode,
        policy_rng_seed=args.policy_rng_seed,
        hero_mode=args.hero_mode,
    )
    timing["eval_after"] = time.monotonic() - t0

    print_eval_result(
        "EVAL AFTER",
        after_eval["bb_per_100"],
        after_eval["coverage_rate"],
        after_eval["hands"],
        after_eval["wall_s"],
    )

    # ── Phase 6: Summary ──────────────────────────────────────────────────
    total_time = time.monotonic() - cycle_started

    print_cycle_summary(
        bb_before=before_eval["bb_per_100"],
        bb_after=after_eval["bb_per_100"],
        cov_before=before_eval["coverage_rate"],
        cov_after=after_eval["coverage_rate"],
        nodes_added=written,
        spots_solved=spots_solved,
        spots_total=total_spots,
        failures=failures,
        exact_count=exact_count,
        approx_count=approx_count,
        active_count=active_count,
        time_eval=timing.get("eval_before", 0),
        time_collect=timing.get("collect", 0),
        time_solve=timing.get("solve", 0),
        time_write=timing.get("write", 0),
        time_after_eval=timing.get("eval_after", 0),
        total_time=total_time,
        spots_skipped_by_reason=solve_result.get("summary", {}).get("spots_skipped_by_reason") if solve_result else None,
    )

    # Save cycle summary JSON
    cycle_summary = {
        "cycle_index": cycle_index,
        "seed": seed,
        "timestamp": utc_now_iso(),
        "before": {
            "bb_per_100": before_eval["bb_per_100"],
            "coverage_rate": before_eval["coverage_rate"],
            "missing_spots": len(missing_spots),
        },
        "after": {
            "bb_per_100": after_eval["bb_per_100"],
            "coverage_rate": after_eval["coverage_rate"],
        },
        "delta": {
            "bb_per_100": after_eval["bb_per_100"] - before_eval["bb_per_100"],
            "coverage_rate": after_eval["coverage_rate"] - before_eval["coverage_rate"],
        },
        "solve": {
            "spots_solved": spots_solved,
            "spots_total": total_spots,
            "nodes_added": written,
            "failures": failures,
            "spots_skipped_by_reason": solve_result.get("summary", {}).get("spots_skipped_by_reason", {}),
            "solve_wall_time": timing.get("solve", 0),
            "average_time_per_spot": timing.get("solve", 0) / max(1, spots_solved),
        },
        "store": {
            "exact": exact_count,
            "approx": approx_count,
            "active": active_count,
        },
        "timing": timing,
        "total_time_s": total_time,
        "chance_boundary_spot_keys": chance_boundary_keys,
    }
    write_json(cycle_dir / "cycle_summary.json", cycle_summary)

    return cycle_summary


# ── Validation Test ────────────────────────────────────────────────────────────

def validate_seed_result(cycle_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a single seed's cycle results. Returns pass/fail with reasons."""
    failure_reasons = []
    total_nodes = 0
    any_crash = False

    for i, result in enumerate(cycle_results):
        nodes_added = result.get("solve", {}).get("nodes_added", 0)
        total_nodes += nodes_added
        failures = result.get("solve", {}).get("failures", 0)

        # (a) At least one node extracted per cycle
        if nodes_added == 0:
            failure_reasons.append(f"Cycle {i+1}: zero nodes extracted")

        # (b) No worker crashes (failures > spots_solved means crash-level failure)
        if failures > 0 and nodes_added == 0:
            any_crash = True
            failure_reasons.append(f"Cycle {i+1}: worker crash (0 nodes, {failures} failures)")

    # (c) Coverage non-decreasing after solving
    for i in range(1, len(cycle_results)):
        prev_cov = cycle_results[i-1].get("after", {}).get("coverage_rate", 0)
        curr_cov = cycle_results[i].get("after", {}).get("coverage_rate", 0)
        if curr_cov < prev_cov - 0.001:  # small tolerance for float comparison
            failure_reasons.append(
                f"Coverage decreased: cycle {i} ({prev_cov:.4f}) -> cycle {i+1} ({curr_cov:.4f})"
            )

    passed = len(failure_reasons) == 0
    return {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "total_nodes_extracted": total_nodes,
        "any_worker_crashed": any_crash,
    }


def run_validation_test(args, phase0_root: Path, pio_exe: str) -> int:
    """Run end-to-end validation with multiple seeds."""
    seeds = [42, 123, 777]
    validation_hands = 100  # Small hand count for quick validation
    results = []

    print(f"\n{C.BOLD}Validation Test Mode{C.RESET}")
    print(f"  Seeds: {seeds}")
    print(f"  Hands per eval: {validation_hands}")

    for seed in seeds:
        print(f"\n{C.BOLD}--- Seed {seed} ---{C.RESET}")

        # Override args for validation
        val_args = argparse.Namespace(**vars(args))
        val_args.seed = seed
        val_args.eval_hands = validation_hands
        val_args.max_cycles = 3

        cycle_results = []
        for cycle_index in range(1, 4):
            try:
                result = run_cycle(
                    args=val_args,
                    cycle_index=cycle_index,
                    phase0_root=phase0_root,
                    pio_exe=pio_exe,
                )
                cycle_results.append(result)
                if result.get("status") == "no_missing":
                    break
            except Exception as exc:
                print(f"  {C.RED}Cycle {cycle_index} failed: {exc}{C.RESET}")
                cycle_results.append({"solve": {"nodes_added": 0, "failures": 1}})
                break

        validation = validate_seed_result(cycle_results)
        validation["seed"] = seed
        validation["cycles_run"] = len(cycle_results)
        final_cov = cycle_results[-1].get("after", {}).get("coverage_rate", 0) if cycle_results else 0
        validation["final_coverage"] = final_cov
        results.append(validation)

        status = f"{C.GREEN}PASS{C.RESET}" if validation["passed"] else f"{C.RED}FAIL{C.RESET}"
        print(f"  Seed {seed}: {status} (nodes={validation['total_nodes_extracted']}, coverage={final_cov:.4f})")
        if not validation["passed"]:
            for reason in validation["failure_reasons"]:
                print(f"    {C.RED}- {reason}{C.RESET}")

    # Summary
    all_passed = all(r["passed"] for r in results)
    print(f"\n{C.BOLD}Validation Summary{C.RESET}")
    for r in results:
        status = f"{C.GREEN}PASS{C.RESET}" if r["passed"] else f"{C.RED}FAIL{C.RESET}"
        print(f"  Seed {r['seed']}: {status}")

    if not all_passed:
        print(f"\n{C.RED}VALIDATION FAILED{C.RESET}")
        return 1

    print(f"\n{C.GREEN}VALIDATION PASSED{C.RESET}")
    return 0


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    phase0_root = args.phase0_root.resolve()
    phase0_root.mkdir(parents=True, exist_ok=True)

    # Resolve PioSolver executable
    pio_exe = (
        os.environ.get("PIO_EXE")
        or os.environ.get("WIPOKER_PIO_PATH")
        or os.environ.get("PIO_PATH")
        or ""
    ).strip()
    if not pio_exe or not Path(pio_exe).exists():
        print(f"{C.RED}ERROR: PIO executable not found. "
              f"Set PIO_EXE or WIPOKER_PIO_PATH.{C.RESET}")
        return 1

    print(f"\n{C.BOLD}Targeted Solve Mode{C.RESET}")
    print(f"  phase0_root: {phase0_root}")
    print(f"  pio_exe: {pio_exe}")
    print(f"  eval_hands: {args.eval_hands}")
    print(f"  max_solves: {args.max_solves or 'unlimited'}")
    print(f"  cfr_steps: {args.cfr_steps}")
    print(f"  max_cycles: {args.max_cycles}")
    print(f"  seed: {args.seed}")

    # Resolve parallel count
    if args.parallel == "auto" or args.parallel == "0":
        args.parallel = auto_detect_parallel()
        print(f"  parallel: {args.parallel} (auto-detected)")
    else:
        args.parallel = max(1, int(args.parallel))
        print(f"  parallel: {args.parallel}")

    print(f"  auto_converge: {args.auto_converge}")

    # Build TypeScript before first eval
    print(f"  {C.DIM}Building eval TypeScript...{C.RESET}")
    build_result = subprocess.run(
        "npm run eval:build",
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=120,
        shell=True,
    )
    if build_result.returncode != 0:
        print(f"{C.RED}eval:build failed: {build_result.stderr[:300]}{C.RESET}")
        return 1

    # Validation test mode
    if args.validation_test:
        return run_validation_test(args, phase0_root, pio_exe)

    # Auto-converge mode
    if args.auto_converge:
        return _run_auto_converge(args, phase0_root, pio_exe)

    # Standard fixed-cycle mode
    for cycle_index in range(1, args.max_cycles + 1):
        try:
            result = run_cycle(
                args=args,
                cycle_index=cycle_index,
                phase0_root=phase0_root,
                pio_exe=pio_exe,
            )
            if result.get("status") == "no_missing":
                print(f"\n{C.GREEN}All spots covered. Stopping.{C.RESET}")
                break

            missing_count = result.get("before", {}).get("missing_spots", 0)
            if missing_count < 50 and cycle_index > 1:
                print(
                    f"\n{C.YELLOW}Diminishing returns: only {missing_count} "
                    f"missing spots. Consider stopping.{C.RESET}"
                )
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Interrupted by user.{C.RESET}")
            return 130
        except Exception as exc:
            print(f"\n{C.RED}Cycle {cycle_index} failed: {exc}{C.RESET}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


def _run_auto_converge(args, phase0_root: Path, pio_exe: str) -> int:
    """Run cycles until a convergence condition is met."""
    started = time.monotonic()
    cycle_history = []
    max_auto_cycles = args.max_cycles if args.max_cycles > 1 else 20
    chance_boundary_keys: set[str] = set()

    print(f"\n{C.BOLD}Auto-converge mode (max {max_auto_cycles} cycles){C.RESET}")

    for cycle_index in range(1, max_auto_cycles + 1):
        try:
            result = run_cycle(
                args=args,
                cycle_index=cycle_index,
                phase0_root=phase0_root,
                pio_exe=pio_exe,
                exclude_spot_keys=chance_boundary_keys,
            )

            if result.get("status") == "no_missing":
                print(f"\n{C.GREEN}All spots covered. Stopping.{C.RESET}")
                break

            # Accumulate chance boundary spot keys for future cycles
            new_cb_keys = result.get("chance_boundary_spot_keys", [])
            if new_cb_keys:
                chance_boundary_keys.update(new_cb_keys)
                print(f"  {C.DIM}Chance boundaries excluded: {len(chance_boundary_keys)} total{C.RESET}")

            cycle_entry = {
                "cycle_index": cycle_index,
                "before_coverage": result.get("before", {}).get("coverage_rate", 0),
                "after_coverage": result.get("after", {}).get("coverage_rate", 0),
                "before_bb100": result.get("before", {}).get("bb_per_100", 0),
                "after_bb100": result.get("after", {}).get("bb_per_100", 0),
                "nodes_added": result.get("solve", {}).get("nodes_added", 0),
                "spots_solved": result.get("solve", {}).get("spots_solved", 0),
                "failures": result.get("solve", {}).get("failures", 0),
                "missing_spots": result.get("before", {}).get("missing_spots", 0),
                "wall_time_s": result.get("total_time_s", 0),
            }
            cycle_history.append(cycle_entry)

            # Check convergence
            condition = check_convergence(cycle_history, max_cycles=max_auto_cycles)
            if condition:
                print(f"\n{C.GREEN}Convergence: {condition}{C.RESET}")
                print(f"  Final coverage: {cycle_entry['after_coverage'] * 100:.1f}%")
                print(f"  Final bb/100: {cycle_entry['after_bb100']:.2f}")
                break

        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Interrupted by user.{C.RESET}")
            return 130
        except Exception as exc:
            print(f"\n{C.RED}Cycle {cycle_index} failed: {exc}{C.RESET}")
            import traceback
            traceback.print_exc()
            return 1

    # Save convergence summary
    total_wall = time.monotonic() - started
    convergence_summary = {
        "feature": "targeted_solve_auto_converge",
        "started_at": utc_now_iso(),
        "total_wall_time_s": total_wall,
        "termination_reason": check_convergence(cycle_history, max_cycles=max_auto_cycles) or "completed",
        "seed": args.seed,
        "cycles": cycle_history,
        "final_coverage_rate": cycle_history[-1]["after_coverage"] if cycle_history else 0,
        "final_bb_per_100": cycle_history[-1]["after_bb100"] if cycle_history else 0,
    }
    summary_path = phase0_root / "convergence_summary.json"
    write_json(summary_path, convergence_summary)
    print(f"  Convergence summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
