#!/usr/bin/env python3
"""Benchmark traversal utilization across worker/progress settings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUN_SCRIPT = SCRIPT_DIR / "run_deep_cfr.py"


def parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        text = token.strip()
        if not text:
            continue
        out.append(int(text))
    if not out:
        raise ValueError("expected at least one integer value")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark traversal worker/progress settings.")
    parser.add_argument("--work-root", type=Path, required=True, help="Parent folder for temporary benchmark runs.")
    parser.add_argument("--workers", type=str, default="0,4,8")
    parser.add_argument("--progress-batches", type=str, default="250,500")
    parser.add_argument("--seat-chunks", type=int, default=1)
    parser.add_argument("--consolidate-processes", action="store_true")
    parser.add_argument("--gpu-batch", action="store_true")
    parser.add_argument("--gpu-batch-sizes", type=str, default="256")
    parser.add_argument("--gpu-batch-timeouts-us", type=str, default="500")
    parser.add_argument("--gpu-batch-queue-capacity", type=int, default=8192)
    parser.add_argument("--cpu-affinity", type=str, default="")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--fp16-onnx", action="store_true")
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--traversals", type=int, default=300)
    parser.add_argument("--training-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--deck-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def run_case(
    *,
    args: argparse.Namespace,
    workers: int,
    progress_batch: int,
    gpu_batch_size: int,
    gpu_batch_timeout_us: int,
    case_seed: int,
    case_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--num-players",
        str(args.num_players),
        "--iterations",
        str(args.iterations),
        "--traversals",
        str(args.traversals),
        "--training-steps",
        str(args.training_steps),
        "--batch-size",
        str(args.batch_size),
        "--buffer-size",
        str(args.buffer_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--bottleneck-dim",
        str(args.bottleneck_dim),
        "--deck-samples",
        str(args.deck_samples),
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
        str(case_seed),
        "--device",
        str(args.device),
        "--traversal-workers",
        str(workers),
        "--traversal-progress-batch",
        str(progress_batch),
        "--traversal-seat-chunks",
        str(args.seat_chunks),
        "--work-dir",
        str(case_dir),
    ]
    if bool(args.consolidate_processes):
        command.append("--traversal-consolidate-processes")
    if bool(args.gpu_batch):
        command.extend(
            [
                "--traversal-gpu-batch",
                "--traversal-gpu-batch-size",
                str(gpu_batch_size),
                "--traversal-gpu-batch-timeout-us",
                str(gpu_batch_timeout_us),
                "--traversal-gpu-batch-queue-capacity",
                str(args.gpu_batch_queue_capacity),
            ]
        )
    if bool(args.fp16_onnx):
        command.append("--traversal-fp16-onnx")
    cpu_affinity = str(args.cpu_affinity).strip()
    if cpu_affinity:
        command.extend(["--traversal-cpu-affinity", cpu_affinity])
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - started
    metrics_path = case_dir / "metrics.json"
    if proc.returncode != 0:
        tail = proc.stderr.strip() or proc.stdout.strip()
        return {
            "status": "error",
            "workers": workers,
            "progress_batch": progress_batch,
            "gpu_batch_size": gpu_batch_size,
            "gpu_batch_timeout_us": gpu_batch_timeout_us,
            "seed": case_seed,
            "elapsed_sec": elapsed,
            "error": tail,
            "work_dir": str(case_dir),
        }

    if not metrics_path.exists():
        return {
            "status": "error",
            "workers": workers,
            "progress_batch": progress_batch,
            "gpu_batch_size": gpu_batch_size,
            "gpu_batch_timeout_us": gpu_batch_timeout_us,
            "seed": case_seed,
            "elapsed_sec": elapsed,
            "error": f"missing metrics.json: {metrics_path}",
            "work_dir": str(case_dir),
        }
    with metrics_path.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)
    iterations = metrics.get("iterations", [])
    if not isinstance(iterations, list) or not iterations:
        return {
            "status": "error",
            "workers": workers,
            "progress_batch": progress_batch,
            "gpu_batch_size": gpu_batch_size,
            "gpu_batch_timeout_us": gpu_batch_timeout_us,
            "seed": case_seed,
            "elapsed_sec": elapsed,
            "error": "no iteration data in metrics",
            "work_dir": str(case_dir),
        }
    last = iterations[-1] if isinstance(iterations[-1], dict) else {}
    times = last.get("times_sec", {}) if isinstance(last, dict) else {}
    throughput = last.get("throughput", {}) if isinstance(last, dict) else {}

    return {
        "status": "ok",
        "workers": workers,
        "progress_batch": progress_batch,
        "gpu_batch_size": gpu_batch_size,
        "gpu_batch_timeout_us": gpu_batch_timeout_us,
        "seed": case_seed,
        "elapsed_sec": elapsed,
        "work_dir": str(case_dir),
        "traverse_wall_sec": float(times.get("traverse_wall", 0.0)),
        "traverse_sum_seat_sec": float(times.get("traverse_sum_seat_time", times.get("traverse_total", 0.0))),
        "train_total_sec": float(times.get("train_total", 0.0)),
        "iteration_total_sec": float(times.get("iteration_total", 0.0)),
        "samples_per_sec_wall": float(throughput.get("samples_per_sec_traverse_wall", 0.0)),
        "samples_per_sec_seat_sum": float(throughput.get("samples_per_sec_traverse_seat_sum", 0.0)),
        "merged_samples_total": int(throughput.get("merged_samples_total", 0)),
    }


def main() -> int:
    args = parse_args()
    if not RUN_SCRIPT.exists():
        raise FileNotFoundError(f"missing run script: {RUN_SCRIPT}")
    if args.seat_chunks <= 0:
        raise ValueError("--seat-chunks must be > 0")
    workers_list = parse_csv_ints(args.workers)
    progress_list = parse_csv_ints(args.progress_batches)
    gpu_batch_sizes = parse_csv_ints(args.gpu_batch_sizes)
    gpu_batch_timeouts = parse_csv_ints(args.gpu_batch_timeouts_us)
    args.work_root.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    idx = 0
    for workers in workers_list:
        for progress_batch in progress_list:
            for gpu_batch_size in gpu_batch_sizes:
                for gpu_batch_timeout in gpu_batch_timeouts:
                    idx += 1
                    case_seed = int(args.seed)
                    case_dir = (
                        args.work_root
                        / f"w{workers}_p{progress_batch}_b{gpu_batch_size}_t{gpu_batch_timeout}"
                    ).resolve()
                    case = run_case(
                        args=args,
                        workers=int(workers),
                        progress_batch=int(progress_batch),
                        gpu_batch_size=int(gpu_batch_size),
                        gpu_batch_timeout_us=int(gpu_batch_timeout),
                        case_seed=case_seed,
                        case_dir=case_dir,
                    )
                    cases.append(case)

    ok_cases = [case for case in cases if case.get("status") == "ok"]
    ok_sorted = sorted(
        ok_cases,
        key=lambda case: float(case.get("samples_per_sec_wall", 0.0)),
        reverse=True,
    )
    best = ok_sorted[0] if ok_sorted else None
    payload = {
        "status": "ok" if ok_cases else "failed",
        "config": {
            "workers": workers_list,
            "progress_batches": progress_list,
            "gpu_batch_sizes": gpu_batch_sizes,
            "gpu_batch_timeouts_us": gpu_batch_timeouts,
            "gpu_batch": bool(args.gpu_batch),
            "consolidate_processes": bool(args.consolidate_processes),
            "gpu_batch_queue_capacity": int(args.gpu_batch_queue_capacity),
            "cpu_affinity": str(args.cpu_affinity),
            "device": str(args.device),
            "fp16_onnx": bool(args.fp16_onnx),
            "num_players": int(args.num_players),
            "iterations": int(args.iterations),
            "traversals": int(args.traversals),
            "training_steps": int(args.training_steps),
            "batch_size": int(args.batch_size),
            "buffer_size": int(args.buffer_size),
            "hidden_dim": int(args.hidden_dim),
            "bottleneck_dim": int(args.bottleneck_dim),
            "deck_samples": int(args.deck_samples),
            "seat_chunks": int(args.seat_chunks),
            "seed": int(args.seed),
            "work_root": str(args.work_root.resolve()),
        },
        "cases": cases,
        "best_by_wall_throughput": best,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(json.dumps(payload, ensure_ascii=True))
    return 0 if ok_cases else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
