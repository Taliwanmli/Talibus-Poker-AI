#!/usr/bin/env python3
"""Single-command local evaluation suite for 6-max Deep CFR models.

Runs test groups sequentially and writes machine-readable outputs:
  - seat_rotation.json
  - opponent_variety.json
  - checkpoint_progression.json
  - search_budget_sweep.json
  - search_opponent_variety.json
  - mixed_table_seat_rotation.json
  - mixed_budget_floor.json
  - summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = REPO_ROOT / "data" / "deep_cfr_6max_longrun_opt"
DEFAULT_MODELS_DIR = DEFAULT_WORK_DIR / "models"
DEFAULT_CHECKPOINT_DIR = DEFAULT_WORK_DIR / "checkpoints"
DEFAULT_CLUSTER_DIR = REPO_ROOT / "checkpoints" / "nlhe_clusters"
DEFAULT_OUT_DIR = DEFAULT_WORK_DIR / "eval_suite"
RAW_LOG_DIR_NAME = "raw_logs"

RING_JSON_PREFIX = "RING_EVAL_JSON "
RT_RING_JSON_PREFIX = "REALTIME_RING_EVAL_JSON "

DEFAULT_CHECKPOINT_ITERS = [130, 150, 170, 190, 210]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}h{minutes:02}m{secs:02}s"
    if minutes > 0:
        return f"{minutes}m{secs:02}s"
    return f"{secs}s"


def parse_budget_list(raw: str) -> List[int]:
    budgets: List[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"search budget must be > 0, got {value}")
        budgets.append(value)
    if not budgets:
        raise ValueError("budget list cannot be empty")
    return budgets


def normalize_opponent_name(raw: str) -> str:
    item = raw.strip().lower().replace("_", "-")
    if item in {"tag"}:
        return "tag"
    if item in {"calling-station", "callingstation", "station"}:
        return "calling-station"
    if item in {"lag", "loose-aggressive", "looseaggressive"}:
        return "lag"
    if item in {"nit"}:
        return "nit"
    raise ValueError(
        f"unsupported opponent '{raw}'. expected one of: tag, calling-station, lag, nit"
    )


def parse_opponent_list(raw: str) -> List[str]:
    opponents: List[str] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        normalized = normalize_opponent_name(item)
        if normalized not in opponents:
            opponents.append(normalized)
    if not opponents:
        raise ValueError("opponent list cannot be empty")
    return opponents


def parse_mixed_table_opponents(raw: str) -> List[str]:
    opponents: List[str] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        opponents.append(normalize_opponent_name(item))
    if not opponents:
        raise ValueError("mixed table opponent list cannot be empty")
    return opponents


def load_eval_env() -> Dict[str, str]:
    training_dir = REPO_ROOT / "training" / "deep_cfr"
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    try:
        from subprocess_env import build_subprocess_env  # type: ignore
    except Exception as exc:  # pragma: no cover - setup failure path
        raise RuntimeError(
            f"failed to import build_subprocess_env from {training_dir}: {exc}"
        ) from exc
    return build_subprocess_env()


def locate_binary(name: str) -> Path:
    candidates = [
        REPO_ROOT / "solver" / "deep_cfr" / "target_opt2" / "release" / f"{name}.exe",
        REPO_ROOT / "solver" / "deep_cfr" / "target" / "release" / f"{name}.exe",
        REPO_ROOT / "solver" / "deep_cfr" / "target_opt" / "release" / f"{name}.exe",
        REPO_ROOT / "solver" / "target_opt2" / "release" / f"{name}.exe",
        REPO_ROOT / "solver" / "target" / "release" / f"{name}.exe",
        REPO_ROOT / "solver" / "target_opt" / "release" / f"{name}.exe",
        REPO_ROOT / "solver" / "deep_cfr" / "target_opt2" / "release" / name,
        REPO_ROOT / "solver" / "deep_cfr" / "target" / "release" / name,
        REPO_ROOT / "solver" / "deep_cfr" / "target_opt" / "release" / name,
        REPO_ROOT / "solver" / "target_opt2" / "release" / name,
        REPO_ROOT / "solver" / "target" / "release" / name,
        REPO_ROOT / "solver" / "target_opt" / "release" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    candidate_str = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"unable to locate binary '{name}'. looked in:\n{candidate_str}\n"
        "build first with: cargo build --release --bin ring_game_eval --bin realtime_play"
    )


def parse_json_line(line: str, prefix: str) -> Dict[str, Any]:
    payload_text = line[len(prefix) :].strip()
    return json.loads(payload_text)


def run_streaming(
    cmd: Sequence[str],
    env: Dict[str, str],
    cwd: Path,
    log_path: Path,
    json_prefix: str,
    *,
    stop_after_json: bool = False,
    shutdown_timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[suite] cmd: {' '.join(cmd)}")
    payload: Dict[str, Any] | None = None
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                print(line, flush=True)
                log_file.write(raw)
                if line.startswith(json_prefix):
                    payload = parse_json_line(line, json_prefix)
                    if stop_after_json:
                        break
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        if stop_after_json and payload is not None:
            # Some realtime binaries can emit final JSON but keep the process alive.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=shutdown_timeout_sec)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=shutdown_timeout_sec)
            exit_code = proc.returncode if proc.returncode is not None else 0
        else:
            exit_code = proc.wait()
    elapsed = time.perf_counter() - start
    if exit_code != 0 and not (stop_after_json and payload is not None):
        raise RuntimeError(
            f"command failed with exit code {exit_code}: {' '.join(cmd)}\nlog: {log_path}"
        )
    if payload is None:
        raise RuntimeError(
            f"did not find '{json_prefix.strip()}' payload in output\nlog: {log_path}"
        )
    payload["_suite_wall_elapsed_sec"] = elapsed
    return payload


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def estimate_search_hand_seconds(budget_ms: int, decisions_per_hand: float, overhead_sec: float) -> float:
    return max(0.2, decisions_per_hand * (overhead_sec + budget_ms / 1000.0))


def estimate_suite_seconds(args: argparse.Namespace) -> float:
    blueprint_runs = 0
    blueprint_total_hands = 0
    if args.seat_hands > 0:
        blueprint_runs += 6
        blueprint_total_hands += 6 * args.seat_hands
    if args.opponent_hands > 0:
        blueprint_runs += 4
        blueprint_total_hands += 4 * args.opponent_hands
    if args.checkpoint_hands > 0:
        checkpoint_count = len(DEFAULT_CHECKPOINT_ITERS) + 1
        blueprint_runs += checkpoint_count
        blueprint_total_hands += checkpoint_count * args.checkpoint_hands
    blueprint_sec = blueprint_total_hands / max(1.0, args.blueprint_hands_per_sec)

    search_sec = 0.0
    search_runs = 0
    if args.search_hands > 0:
        search_runs += len(args.search_budgets)
        for budget in args.search_budgets:
            hand_sec = estimate_search_hand_seconds(
                budget,
                decisions_per_hand=args.search_decisions_per_hand,
                overhead_sec=args.search_decision_overhead_sec,
            )
            search_sec += args.search_hands * hand_sec

    if args.search_opponent_hands > 0:
        search_runs += len(args.search_opponents)
        opp_hand_sec = estimate_search_hand_seconds(
            args.search_opponent_budget_ms,
            decisions_per_hand=args.search_decisions_per_hand,
            overhead_sec=args.search_decision_overhead_sec,
        )
        search_sec += len(args.search_opponents) * args.search_opponent_hands * opp_hand_sec

    if args.mixed_table_hands > 0:
        search_runs += args.num_players
        mixed_hand_sec = estimate_search_hand_seconds(
            args.mixed_table_budget_ms,
            decisions_per_hand=args.search_decisions_per_hand,
            overhead_sec=args.search_decision_overhead_sec,
        )
        search_sec += args.num_players * args.mixed_table_hands * mixed_hand_sec

    if args.mixed_budget_floor_hands > 0:
        search_runs += 1
        floor_hand_sec = estimate_search_hand_seconds(
            args.mixed_budget_floor_ms,
            decisions_per_hand=args.search_decisions_per_hand,
            overhead_sec=args.search_decision_overhead_sec,
        )
        search_sec += args.mixed_budget_floor_hands * floor_hand_sec

    # modest fixed process/setup cost
    return blueprint_sec + search_sec + (blueprint_runs + search_runs) * 1.0


def build_ring_eval_cmd(
    binary: Path,
    *,
    mode: str,
    opponent: str,
    model: Path,
    policy: str,
    model_seat: int,
    num_players: int,
    hands: int,
    deck_samples: int,
    workers: int,
    seed: int,
    progress_every: int,
    cluster_dir: Path,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
) -> List[str]:
    return [
        str(binary),
        "--mode",
        mode,
        "--opponent",
        opponent,
        "--model",
        str(model),
        "--policy",
        policy,
        "--model-seat",
        str(model_seat),
        "--num-players",
        str(num_players),
        "--hands",
        str(hands),
        "--deck-samples",
        str(deck_samples),
        "--workers",
        str(workers),
        "--seed",
        str(seed),
        "--progress-every",
        str(progress_every),
        "--cluster-dir",
        str(cluster_dir),
        "--starting-stack",
        str(starting_stack),
        "--small-blind",
        str(small_blind),
        "--big-blind",
        str(big_blind),
    ]


def build_rt_ring_eval_cmd(
    binary: Path,
    *,
    opponent: str,
    model: Path,
    policy: str,
    model_seat: int,
    num_players: int,
    hands: int,
    deck_samples: int,
    seed: int,
    progress_every: int,
    cluster_dir: Path,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    budget_ms: int,
    threads: int,
    batch_size: int,
    batch_wait_us: int,
    batch_queue_capacity: int,
) -> List[str]:
    return [
        str(binary),
        "--mode",
        "ring-eval",
        "--opponent",
        opponent,
        "--model",
        str(model),
        "--policy",
        policy,
        "--model-seat",
        str(model_seat),
        "--num-players",
        str(num_players),
        "--hands",
        str(hands),
        "--deck-samples",
        str(deck_samples),
        "--seed",
        str(seed),
        "--progress-every",
        str(progress_every),
        "--cluster-dir",
        str(cluster_dir),
        "--starting-stack",
        str(starting_stack),
        "--small-blind",
        str(small_blind),
        "--big-blind",
        str(big_blind),
        "--time-budget-ms",
        str(budget_ms),
        "--threads",
        str(threads),
        "--batch-size",
        str(batch_size),
        "--batch-wait-us",
        str(batch_wait_us),
        "--batch-queue-capacity",
        str(batch_queue_capacity),
        "--disable-adaptive-budget",
    ]


def build_rt_mixed_ring_eval_cmd(
    binary: Path,
    *,
    opponents: Sequence[str],
    model: Path,
    policy: str,
    model_seat: int,
    num_players: int,
    hands: int,
    deck_samples: int,
    seed: int,
    progress_every: int,
    cluster_dir: Path,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    budget_ms: int,
    threads: int,
    batch_size: int,
    batch_wait_us: int,
    batch_queue_capacity: int,
) -> List[str]:
    return [
        str(binary),
        "--mode",
        "ring-eval",
        "--opponents",
        ",".join(opponents),
        "--model",
        str(model),
        "--policy",
        policy,
        "--model-seat",
        str(model_seat),
        "--num-players",
        str(num_players),
        "--hands",
        str(hands),
        "--deck-samples",
        str(deck_samples),
        "--seed",
        str(seed),
        "--progress-every",
        str(progress_every),
        "--cluster-dir",
        str(cluster_dir),
        "--starting-stack",
        str(starting_stack),
        "--small-blind",
        str(small_blind),
        "--big-blind",
        str(big_blind),
        "--time-budget-ms",
        str(budget_ms),
        "--threads",
        str(threads),
        "--batch-size",
        str(batch_size),
        "--batch-wait-us",
        str(batch_wait_us),
        "--batch-queue-capacity",
        str(batch_queue_capacity),
        "--disable-adaptive-budget",
    ]


def discover_checkpoint_models(
    checkpoint_dir: Path,
    models_dir: Path,
    preferred_iters: Sequence[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for iter_idx in preferred_iters:
        path = checkpoint_dir / f"strategy_iter{iter_idx}.onnx"
        if path.exists():
            out.append(
                {
                    "label": f"iter_{iter_idx}",
                    "iteration": iter_idx,
                    "model_path": str(path),
                }
            )
    best_ring = models_dir / "strategy_shared_best_ring.onnx"
    if best_ring.exists():
        out.append(
            {
                "label": "best_ring",
                "iteration": 154,
                "model_path": str(best_ring),
            }
        )
    if not out:
        raise FileNotFoundError(
            f"no checkpoint models found in {checkpoint_dir} and missing best_ring model in {models_dir}"
        )
    return out


def run_suite(args: argparse.Namespace) -> None:
    if args.seat_hands < 0 or args.opponent_hands < 0 or args.checkpoint_hands < 0:
        raise ValueError("seat/opponent/checkpoint hands must be >= 0")
    if (
        args.search_hands < 0
        or args.search_opponent_hands < 0
        or args.mixed_table_hands < 0
        or args.mixed_budget_floor_hands < 0
    ):
        raise ValueError("search/mixed hands values must be >= 0")
    if args.search_opponent_budget_ms <= 0:
        raise ValueError("--search-opponent-budget-ms must be > 0")
    if args.mixed_table_budget_ms <= 0:
        raise ValueError("--mixed-table-budget-ms must be > 0")
    if args.mixed_budget_floor_ms <= 0:
        raise ValueError("--mixed-budget-floor-ms must be > 0")
    expected_mixed_opponents = args.num_players - 1
    if len(args.mixed_table_opponents) != expected_mixed_opponents:
        raise ValueError(
            f"--mixed-table-opponents must have exactly {expected_mixed_opponents} entries "
            f"for {args.num_players} players (excluding model seat); got {len(args.mixed_table_opponents)}"
        )

    ring_eval_bin = locate_binary("ring_game_eval")
    realtime_play_bin = locate_binary("realtime_play")

    base_model = Path(args.model).resolve()
    cluster_dir = Path(args.cluster_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / RAW_LOG_DIR_NAME

    if not base_model.exists():
        raise FileNotFoundError(f"base model not found: {base_model}")
    if not cluster_dir.exists():
        raise FileNotFoundError(f"cluster dir not found: {cluster_dir}")

    env = load_eval_env()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    suite_est = estimate_suite_seconds(args)
    print("=" * 72)
    print("Deep CFR Comprehensive Evaluation Suite")
    print("=" * 72)
    print(f"[suite] ring_game_eval: {ring_eval_bin}")
    print(f"[suite] realtime_play : {realtime_play_bin}")
    print(f"[suite] base model    : {base_model}")
    print(f"[suite] cluster dir   : {cluster_dir}")
    print(f"[suite] output dir    : {out_dir}")
    print(f"[suite] estimated total runtime: {format_duration(suite_est)}")
    print("=" * 72)

    seat_runs: List[Dict[str, Any]] = []
    if args.seat_hands > 0:
        # -------- Test 1: Seat Rotation --------
        print("\n" + "=" * 72)
        print("TEST 1/7: Seat Rotation (Blueprint vs TAG)")
        print(
            f"  hands/seat={args.seat_hands} seats=0..{args.num_players - 1} seed={args.seat_seed}"
        )
        print("=" * 72)
        for seat in range(args.num_players):
            log_path = raw_dir / f"seat_rotation_seat{seat}.log"
            progress_every = max(1, args.seat_hands // 10)
            cmd = build_ring_eval_cmd(
                ring_eval_bin,
                mode="vs-opponent",
                opponent="tag",
                model=base_model,
                policy=args.policy,
                model_seat=seat,
                num_players=args.num_players,
                hands=args.seat_hands,
                deck_samples=args.deck_samples,
                workers=args.ring_workers,
                seed=args.seat_seed,
                progress_every=progress_every,
                cluster_dir=cluster_dir,
                starting_stack=args.starting_stack,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
            )
            print(f"[suite][seat-{seat}] start")
            payload = run_streaming(
                cmd, env=env, cwd=REPO_ROOT, log_path=log_path, json_prefix=RING_JSON_PREFIX
            )
            payload["seat"] = seat
            payload["log_path"] = str(log_path)
            seat_runs.append(payload)
            print(
                f"[suite][seat-{seat}] done: bb/100={payload['model_bb_per_100']:+.3f} "
                f"(wins={payload['model_wins']} losses={payload['model_losses']})"
            )
        seat_output = {
            "generated_at": utc_now_iso(),
            "config": {
                "hands": args.seat_hands,
                "seed": args.seat_seed,
                "num_players": args.num_players,
                "deck_samples": args.deck_samples,
                "workers": args.ring_workers,
            },
            "runs": seat_runs,
        }
        write_json(out_dir / "seat_rotation.json", seat_output)
    else:
        print("[suite] skipping seat rotation (--seat-hands <= 0)")

    opponent_runs: List[Dict[str, Any]] = []
    if args.opponent_hands > 0:
        # -------- Test 2: Opponent Variety --------
        print("\n" + "=" * 72)
        print("TEST 2/7: Opponent Variety (Blueprint vs scripted profiles)")
        print(
            f"  opponents=tag,calling-station,lag,nit hands/opponent={args.opponent_hands} seed={args.opponent_seed}"
        )
        print("=" * 72)
        opponents = ["tag", "calling-station", "lag", "nit"]
        for opponent in opponents:
            safe_name = opponent.replace("-", "_")
            log_path = raw_dir / f"opponent_{safe_name}.log"
            progress_every = max(1, args.opponent_hands // 10)
            cmd = build_ring_eval_cmd(
                ring_eval_bin,
                mode="vs-opponent",
                opponent=opponent,
                model=base_model,
                policy=args.policy,
                model_seat=args.model_seat,
                num_players=args.num_players,
                hands=args.opponent_hands,
                deck_samples=args.deck_samples,
                workers=args.ring_workers,
                seed=args.opponent_seed,
                progress_every=progress_every,
                cluster_dir=cluster_dir,
                starting_stack=args.starting_stack,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
            )
            print(f"[suite][opponent={opponent}] start")
            payload = run_streaming(
                cmd, env=env, cwd=REPO_ROOT, log_path=log_path, json_prefix=RING_JSON_PREFIX
            )
            payload["configured_opponent"] = opponent
            payload["log_path"] = str(log_path)
            opponent_runs.append(payload)
            print(
                f"[suite][opponent={opponent}] done: bb/100={payload['model_bb_per_100']:+.3f} "
                f"(wins={payload['model_wins']} losses={payload['model_losses']})"
            )
        opponent_output = {
            "generated_at": utc_now_iso(),
            "config": {
                "hands": args.opponent_hands,
                "seed": args.opponent_seed,
                "model_seat": args.model_seat,
                "num_players": args.num_players,
                "deck_samples": args.deck_samples,
                "workers": args.ring_workers,
                "opponents": opponents,
            },
            "runs": opponent_runs,
        }
        write_json(out_dir / "opponent_variety.json", opponent_output)
    else:
        print("[suite] skipping blueprint opponent variety (--opponent-hands <= 0)")

    checkpoint_runs: List[Dict[str, Any]] = []
    if args.checkpoint_hands > 0:
        # -------- Test 3: Checkpoint Progression --------
        print("\n" + "=" * 72)
        print("TEST 3/7: Checkpoint Progression (Blueprint vs TAG)")
        print(
            f"  checkpoints={DEFAULT_CHECKPOINT_ITERS}+best_ring hands/checkpoint={args.checkpoint_hands} seed={args.checkpoint_seed}"
        )
        print("=" * 72)
        checkpoint_models = discover_checkpoint_models(
            checkpoint_dir=checkpoint_dir,
            models_dir=models_dir,
            preferred_iters=DEFAULT_CHECKPOINT_ITERS,
        )
        for checkpoint in checkpoint_models:
            model_path = Path(checkpoint["model_path"])
            label = checkpoint["label"]
            log_path = raw_dir / f"checkpoint_{label}.log"
            progress_every = max(1, args.checkpoint_hands // 10)
            cmd = build_ring_eval_cmd(
                ring_eval_bin,
                mode="vs-opponent",
                opponent="tag",
                model=model_path,
                policy=args.policy,
                model_seat=args.model_seat,
                num_players=args.num_players,
                hands=args.checkpoint_hands,
                deck_samples=args.deck_samples,
                workers=args.ring_workers,
                seed=args.checkpoint_seed,
                progress_every=progress_every,
                cluster_dir=cluster_dir,
                starting_stack=args.starting_stack,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
            )
            print(f"[suite][checkpoint={label}] start")
            payload = run_streaming(
                cmd, env=env, cwd=REPO_ROOT, log_path=log_path, json_prefix=RING_JSON_PREFIX
            )
            payload["checkpoint_label"] = label
            payload["checkpoint_iteration"] = checkpoint.get("iteration")
            payload["model_path"] = str(model_path)
            payload["log_path"] = str(log_path)
            checkpoint_runs.append(payload)
            print(
                f"[suite][checkpoint={label}] done: bb/100={payload['model_bb_per_100']:+.3f} "
                f"(wins={payload['model_wins']} losses={payload['model_losses']})"
            )
        checkpoint_output = {
            "generated_at": utc_now_iso(),
            "config": {
                "hands": args.checkpoint_hands,
                "seed": args.checkpoint_seed,
                "model_seat": args.model_seat,
                "num_players": args.num_players,
                "deck_samples": args.deck_samples,
                "workers": args.ring_workers,
                "checkpoint_labels": [item["label"] for item in checkpoint_models],
            },
            "runs": checkpoint_runs,
        }
        write_json(out_dir / "checkpoint_progression.json", checkpoint_output)
    else:
        print("[suite] skipping checkpoint progression (--checkpoint-hands <= 0)")

    search_runs: List[Dict[str, Any]] = []
    if args.search_hands > 0:
        # -------- Test 4: Search Budget Sweep --------
        print("\n" + "=" * 72)
        print(
            f"TEST 4/7: Search Budget Sweep (Realtime search vs {args.search_budget_opponent.upper()})"
        )
        print(
            f"  budgets_ms={args.search_budgets} hands/budget={args.search_hands} seed={args.search_seed}"
        )
        print("=" * 72)
        for budget in args.search_budgets:
            log_path = raw_dir / f"search_budget_{budget}ms.log"
            progress_every = max(1, args.search_hands // 20)
            cmd = build_rt_ring_eval_cmd(
                realtime_play_bin,
                opponent=args.search_budget_opponent,
                model=base_model,
                policy=args.policy,
                model_seat=args.model_seat,
                num_players=args.num_players,
                hands=args.search_hands,
                deck_samples=args.deck_samples,
                seed=args.search_seed,
                progress_every=progress_every,
                cluster_dir=cluster_dir,
                starting_stack=args.starting_stack,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
                budget_ms=budget,
                threads=args.search_threads,
                batch_size=args.search_batch_size,
                batch_wait_us=args.search_batch_wait_us,
                batch_queue_capacity=args.search_batch_queue_capacity,
            )
            print(f"[suite][search-budget={budget}ms] start")
            payload = run_streaming(
                cmd,
                env=env,
                cwd=REPO_ROOT,
                log_path=log_path,
                json_prefix=RT_RING_JSON_PREFIX,
                stop_after_json=True,
            )
            payload["budget_ms"] = budget
            payload["configured_opponent"] = args.search_budget_opponent
            payload["log_path"] = str(log_path)
            search_runs.append(payload)
            print(
                f"[suite][search-budget={budget}ms] done: bb/100={payload['model_bb_per_100']:+.3f} "
                f"avg_decision_ms={payload['average_decision_ms']:.1f}"
            )
        search_output = {
            "generated_at": utc_now_iso(),
            "config": {
                "hands": args.search_hands,
                "seed": args.search_seed,
                "model_seat": args.model_seat,
                "opponent": args.search_budget_opponent,
                "num_players": args.num_players,
                "deck_samples": args.deck_samples,
                "threads": args.search_threads,
                "batch_size": args.search_batch_size,
                "batch_wait_us": args.search_batch_wait_us,
                "batch_queue_capacity": args.search_batch_queue_capacity,
                "budgets_ms": args.search_budgets,
                "adaptive_budget": False,
            },
            "runs": search_runs,
        }
        write_json(out_dir / "search_budget_sweep.json", search_output)
    else:
        print("[suite] skipping search budget sweep (--search-hands <= 0)")

    search_opponent_runs: List[Dict[str, Any]] = []
    if args.search_opponent_hands > 0:
        # -------- Test 5: Search Opponent Variety --------
        print("\n" + "=" * 72)
        print("TEST 5/7: Search Opponent Variety")
        print(
            f"  opponents={args.search_opponents} budget_ms={args.search_opponent_budget_ms} hands/opponent={args.search_opponent_hands} seed={args.search_opponent_seed}"
        )
        print("=" * 72)
        for opponent in args.search_opponents:
            safe_name = opponent.replace("-", "_")
            log_path = raw_dir / f"search_opponent_{safe_name}.log"
            progress_every = max(1, args.search_opponent_hands // 20)
            cmd = build_rt_ring_eval_cmd(
                realtime_play_bin,
                opponent=opponent,
                model=base_model,
                policy=args.policy,
                model_seat=args.model_seat,
                num_players=args.num_players,
                hands=args.search_opponent_hands,
                deck_samples=args.deck_samples,
                seed=args.search_opponent_seed,
                progress_every=progress_every,
                cluster_dir=cluster_dir,
                starting_stack=args.starting_stack,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
                budget_ms=args.search_opponent_budget_ms,
                threads=args.search_threads,
                batch_size=args.search_batch_size,
                batch_wait_us=args.search_batch_wait_us,
                batch_queue_capacity=args.search_batch_queue_capacity,
            )
            print(f"[suite][search-opponent={opponent}] start")
            payload = run_streaming(
                cmd,
                env=env,
                cwd=REPO_ROOT,
                log_path=log_path,
                json_prefix=RT_RING_JSON_PREFIX,
                stop_after_json=True,
            )
            payload["configured_opponent"] = opponent
            payload["budget_ms"] = args.search_opponent_budget_ms
            payload["log_path"] = str(log_path)
            search_opponent_runs.append(payload)
            print(
                f"[suite][search-opponent={opponent}] done: bb/100={payload['model_bb_per_100']:+.3f} "
                f"avg_decision_ms={payload['average_decision_ms']:.1f}"
            )
        search_opponent_output = {
            "generated_at": utc_now_iso(),
            "config": {
                "hands": args.search_opponent_hands,
                "seed": args.search_opponent_seed,
                "model_seat": args.model_seat,
                "opponents": args.search_opponents,
                "budget_ms": args.search_opponent_budget_ms,
                "num_players": args.num_players,
                "deck_samples": args.deck_samples,
                "threads": args.search_threads,
                "batch_size": args.search_batch_size,
                "batch_wait_us": args.search_batch_wait_us,
                "batch_queue_capacity": args.search_batch_queue_capacity,
                "adaptive_budget": False,
            },
            "runs": search_opponent_runs,
        }
        write_json(out_dir / "search_opponent_variety.json", search_opponent_output)
    else:
        print("[suite] skipping search opponent variety (--search-opponent-hands <= 0)")

    mixed_table_runs: List[Dict[str, Any]] = []
    if args.mixed_table_hands > 0:
        # -------- Test 6: Mixed Table Seat Rotation --------
        print("\n" + "=" * 72)
        print("TEST 6/7: Mixed Table Seat Rotation (Realtime search)")
        print(
            f"  opponents={args.mixed_table_opponents} budget_ms={args.mixed_table_budget_ms} "
            f"hands/seat={args.mixed_table_hands} seed={args.mixed_table_seed}"
        )
        print("=" * 72)
        for seat in range(args.num_players):
            log_path = raw_dir / f"mixed_table_seat{seat}.log"
            progress_every = max(1, args.mixed_table_hands // 20)
            cmd = build_rt_mixed_ring_eval_cmd(
                realtime_play_bin,
                opponents=args.mixed_table_opponents,
                model=base_model,
                policy=args.policy,
                model_seat=seat,
                num_players=args.num_players,
                hands=args.mixed_table_hands,
                deck_samples=args.deck_samples,
                seed=args.mixed_table_seed,
                progress_every=progress_every,
                cluster_dir=cluster_dir,
                starting_stack=args.starting_stack,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
                budget_ms=args.mixed_table_budget_ms,
                threads=args.search_threads,
                batch_size=args.search_batch_size,
                batch_wait_us=args.search_batch_wait_us,
                batch_queue_capacity=args.search_batch_queue_capacity,
            )
            print(f"[suite][mixed-table seat={seat}] start")
            payload = run_streaming(
                cmd,
                env=env,
                cwd=REPO_ROOT,
                log_path=log_path,
                json_prefix=RT_RING_JSON_PREFIX,
                stop_after_json=True,
            )
            payload["seat"] = seat
            payload["budget_ms"] = args.mixed_table_budget_ms
            payload["configured_opponents"] = list(args.mixed_table_opponents)
            payload["log_path"] = str(log_path)
            mixed_table_runs.append(payload)
            print(
                f"[suite][mixed-table seat={seat}] done: bb/100={payload['model_bb_per_100']:+.3f} "
                f"avg_decision_ms={payload['average_decision_ms']:.1f}"
            )
        mixed_table_output = {
            "generated_at": utc_now_iso(),
            "config": {
                "hands": args.mixed_table_hands,
                "seed": args.mixed_table_seed,
                "budget_ms": args.mixed_table_budget_ms,
                "num_players": args.num_players,
                "deck_samples": args.deck_samples,
                "threads": args.search_threads,
                "batch_size": args.search_batch_size,
                "batch_wait_us": args.search_batch_wait_us,
                "batch_queue_capacity": args.search_batch_queue_capacity,
                "opponents": list(args.mixed_table_opponents),
                "adaptive_budget": False,
            },
            "runs": mixed_table_runs,
        }
        write_json(out_dir / "mixed_table_seat_rotation.json", mixed_table_output)
    else:
        print("[suite] skipping mixed table seat rotation (--mixed-table-hands <= 0)")

    mixed_budget_floor_runs: List[Dict[str, Any]] = []
    if args.mixed_budget_floor_hands > 0:
        # -------- Test 7: Mixed Table Budget Floor --------
        print("\n" + "=" * 72)
        print("TEST 7/7: Mixed Table Budget Floor (Realtime search)")
        print(
            f"  opponents={args.mixed_table_opponents} budget_ms={args.mixed_budget_floor_ms} "
            f"hands={args.mixed_budget_floor_hands} seat=0 seed={args.mixed_table_seed}"
        )
        print("=" * 72)
        log_path = raw_dir / "mixed_budget_floor.log"
        progress_every = max(1, args.mixed_budget_floor_hands // 20)
        cmd = build_rt_mixed_ring_eval_cmd(
            realtime_play_bin,
            opponents=args.mixed_table_opponents,
            model=base_model,
            policy=args.policy,
            model_seat=0,
            num_players=args.num_players,
            hands=args.mixed_budget_floor_hands,
            deck_samples=args.deck_samples,
            seed=args.mixed_table_seed,
            progress_every=progress_every,
            cluster_dir=cluster_dir,
            starting_stack=args.starting_stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            budget_ms=args.mixed_budget_floor_ms,
            threads=args.search_threads,
            batch_size=args.search_batch_size,
            batch_wait_us=args.search_batch_wait_us,
            batch_queue_capacity=args.search_batch_queue_capacity,
        )
        print("[suite][mixed-budget-floor] start")
        payload = run_streaming(
            cmd,
            env=env,
            cwd=REPO_ROOT,
            log_path=log_path,
            json_prefix=RT_RING_JSON_PREFIX,
            stop_after_json=True,
        )
        payload["seat"] = 0
        payload["budget_ms"] = args.mixed_budget_floor_ms
        payload["configured_opponents"] = list(args.mixed_table_opponents)
        payload["log_path"] = str(log_path)
        mixed_budget_floor_runs.append(payload)
        print(
            f"[suite][mixed-budget-floor] done: bb/100={payload['model_bb_per_100']:+.3f} "
            f"avg_decision_ms={payload['average_decision_ms']:.1f}"
        )
        mixed_budget_floor_output = {
            "generated_at": utc_now_iso(),
            "config": {
                "hands": args.mixed_budget_floor_hands,
                "seed": args.mixed_table_seed,
                "budget_ms": args.mixed_budget_floor_ms,
                "model_seat": 0,
                "num_players": args.num_players,
                "deck_samples": args.deck_samples,
                "threads": args.search_threads,
                "batch_size": args.search_batch_size,
                "batch_wait_us": args.search_batch_wait_us,
                "batch_queue_capacity": args.search_batch_queue_capacity,
                "opponents": list(args.mixed_table_opponents),
                "adaptive_budget": False,
            },
            "runs": mixed_budget_floor_runs,
        }
        write_json(out_dir / "mixed_budget_floor.json", mixed_budget_floor_output)
    else:
        print("[suite] skipping mixed budget floor (--mixed-budget-floor-hands <= 0)")

    summary = {
        "generated_at": utc_now_iso(),
        "suite_config": {
            "model": str(base_model),
            "cluster_dir": str(cluster_dir),
            "num_players": args.num_players,
            "model_seat": args.model_seat,
            "deck_samples": args.deck_samples,
            "blinds": {"small_blind": args.small_blind, "big_blind": args.big_blind},
            "starting_stack": args.starting_stack,
            "mixed_table_opponents": list(args.mixed_table_opponents),
        },
        "seat_rotation": [
            {"seat": item["seat"], "bb_per_100": item["model_bb_per_100"]}
            for item in seat_runs
        ],
        "opponent_variety": [
            {
                "opponent": item["configured_opponent"],
                "bb_per_100": item["model_bb_per_100"],
            }
            for item in opponent_runs
        ],
        "checkpoint_progression": [
            {
                "checkpoint": item["checkpoint_label"],
                "iteration": item.get("checkpoint_iteration"),
                "bb_per_100": item["model_bb_per_100"],
            }
            for item in checkpoint_runs
        ],
        "search_budget_sweep": [
            {
                "budget_ms": item["budget_ms"],
                "opponent": item.get("configured_opponent", item.get("opponent", "tag")),
                "bb_per_100": item["model_bb_per_100"],
                "average_decision_ms": item["average_decision_ms"],
                "average_iterations_per_decision": item["average_iterations_per_decision"],
            }
            for item in search_runs
        ],
        "search_opponent_variety": [
            {
                "opponent": item["configured_opponent"],
                "budget_ms": item["budget_ms"],
                "bb_per_100": item["model_bb_per_100"],
                "average_decision_ms": item["average_decision_ms"],
                "average_iterations_per_decision": item["average_iterations_per_decision"],
            }
            for item in search_opponent_runs
        ],
        "mixed_table_seat_rotation": [
            {
                "seat": item["seat"],
                "budget_ms": item["budget_ms"],
                "bb_per_100": item["model_bb_per_100"],
                "average_decision_ms": item["average_decision_ms"],
                "average_iterations_per_decision": item["average_iterations_per_decision"],
            }
            for item in mixed_table_runs
        ],
        "mixed_budget_floor": [
            {
                "seat": item["seat"],
                "budget_ms": item["budget_ms"],
                "bb_per_100": item["model_bb_per_100"],
                "average_decision_ms": item["average_decision_ms"],
                "average_iterations_per_decision": item["average_iterations_per_decision"],
            }
            for item in mixed_budget_floor_runs
        ],
        "output_files": {
            "seat_rotation": str(out_dir / "seat_rotation.json") if args.seat_hands > 0 else None,
            "opponent_variety": str(out_dir / "opponent_variety.json")
            if args.opponent_hands > 0
            else None,
            "checkpoint_progression": str(out_dir / "checkpoint_progression.json")
            if args.checkpoint_hands > 0
            else None,
            "search_budget_sweep": str(out_dir / "search_budget_sweep.json")
            if args.search_hands > 0
            else None,
            "search_opponent_variety": str(out_dir / "search_opponent_variety.json")
            if args.search_opponent_hands > 0
            else None,
            "mixed_table_seat_rotation": str(out_dir / "mixed_table_seat_rotation.json")
            if args.mixed_table_hands > 0
            else None,
            "mixed_budget_floor": str(out_dir / "mixed_budget_floor.json")
            if args.mixed_budget_floor_hands > 0
            else None,
            "summary": str(out_dir / "summary.json"),
            "raw_logs_dir": str(raw_dir),
        },
    }
    write_json(out_dir / "summary.json", summary)

    print("\n" + "=" * 72)
    print("EVALUATION SUITE COMPLETE")
    print("=" * 72)
    print(f"[summary] seat_rotation file       : {out_dir / 'seat_rotation.json'}")
    print(f"[summary] opponent_variety file    : {out_dir / 'opponent_variety.json'}")
    print(f"[summary] checkpoint_progress file : {out_dir / 'checkpoint_progression.json'}")
    print(f"[summary] search_budget file       : {out_dir / 'search_budget_sweep.json'}")
    print(f"[summary] search_opponent file     : {out_dir / 'search_opponent_variety.json'}")
    print(f"[summary] mixed_table file         : {out_dir / 'mixed_table_seat_rotation.json'}")
    print(f"[summary] mixed_budget_floor file  : {out_dir / 'mixed_budget_floor.json'}")
    print(f"[summary] consolidated summary     : {out_dir / 'summary.json'}")
    print(f"[summary] raw logs directory       : {raw_dir}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comprehensive local poker AI evaluation suite."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODELS_DIR / "strategy_shared_best_ring.onnx"),
        help="Base strategy ONNX model path.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing model artifacts (for best_ring fallback).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory containing strategy_iter*.onnx checkpoints.",
    )
    parser.add_argument(
        "--cluster-dir",
        default=str(DEFAULT_CLUSTER_DIR),
        help="NLHE cluster directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where suite JSON/log outputs are saved.",
    )
    parser.add_argument("--policy", default="strategy", choices=["strategy", "advantage"])
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--model-seat", type=int, default=0)
    parser.add_argument("--deck-samples", type=int, default=200)
    parser.add_argument("--starting-stack", type=int, default=2000)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--big-blind", type=int, default=20)

    parser.add_argument("--seat-hands", type=int, default=10_000)
    parser.add_argument("--opponent-hands", type=int, default=10_000)
    parser.add_argument("--checkpoint-hands", type=int, default=10_000)
    parser.add_argument("--search-hands", type=int, default=100)
    parser.add_argument("--search-opponent-hands", type=int, default=0)
    parser.add_argument("--mixed-table-hands", type=int, default=0)
    parser.add_argument("--mixed-budget-floor-hands", type=int, default=0)

    parser.add_argument("--seat-seed", type=int, default=777)
    parser.add_argument("--opponent-seed", type=int, default=777)
    parser.add_argument("--checkpoint-seed", type=int, default=777)
    parser.add_argument("--search-seed", type=int, default=888)
    parser.add_argument("--search-opponent-seed", type=int, default=889)
    parser.add_argument("--mixed-table-seed", type=int, default=999)

    parser.add_argument("--ring-workers", type=int, default=0)
    parser.add_argument("--search-threads", type=int, default=24)
    parser.add_argument("--search-batch-size", type=int, default=128)
    parser.add_argument("--search-batch-wait-us", type=int, default=50)
    parser.add_argument("--search-batch-queue-capacity", type=int, default=8192)
    parser.add_argument(
        "--search-budgets",
        type=parse_budget_list,
        default=parse_budget_list("500,1000,2000,5000"),
        help="Comma-separated budget list in ms, e.g. 500,1000,2000,5000",
    )
    parser.add_argument(
        "--search-budget-opponent",
        type=normalize_opponent_name,
        default="tag",
        help="Opponent for search budget sweep (default: tag).",
    )
    parser.add_argument(
        "--search-opponent-budget-ms",
        type=int,
        default=2000,
        help="Single search budget used for search opponent variety test.",
    )
    parser.add_argument(
        "--search-opponents",
        type=parse_opponent_list,
        default=parse_opponent_list("tag,calling-station,lag,nit"),
        help="Comma-separated opponents for search opponent variety test.",
    )
    parser.add_argument(
        "--mixed-table-budget-ms",
        type=int,
        default=2000,
        help="Search budget used for mixed-table seat-rotation test.",
    )
    parser.add_argument(
        "--mixed-table-opponents",
        type=parse_mixed_table_opponents,
        default=parse_mixed_table_opponents("tag,calling-station,lag,nit,tag"),
        help="Comma-separated opponent seats for mixed-table tests (duplicates allowed).",
    )
    parser.add_argument(
        "--mixed-budget-floor-ms",
        type=int,
        default=500,
        help="Lower search budget used for mixed-table budget-floor test.",
    )

    parser.add_argument(
        "--blueprint-hands-per-sec",
        type=float,
        default=1100.0,
        help="ETA hint for ring_game_eval throughput.",
    )
    parser.add_argument(
        "--search-decisions-per-hand",
        type=float,
        default=1.87,
        help="ETA hint for realtime search: average decisions per hand.",
    )
    parser.add_argument(
        "--search-decision-overhead-sec",
        type=float,
        default=0.80,
        help="ETA hint for realtime search: non-budget per-decision overhead.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
