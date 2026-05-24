"""Orchestrate repeated Kuhn Linear CFR runs and emit a metrics log."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunMetrics:
    iterations: int
    workers: int
    seed: int
    player0_ev: float
    target_ev: float
    abs_error: float
    exploitability: float
    infosets: int


def parse_metrics(stdout: str) -> RunMetrics:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        match = re.match(r"^([a-zA-Z0-9_]+)=([-+0-9.eE]+)$", line.strip())
        if not match:
            continue
        values[match.group(1)] = match.group(2)

    required = [
        "iterations",
        "workers",
        "seed",
        "player0_ev",
        "target_ev",
        "abs_error",
        "exploitability",
        "infosets",
    ]
    missing = [k for k in required if k not in values]
    if missing:
        raise ValueError(f"missing metrics keys from solver output: {missing}")

    return RunMetrics(
        iterations=int(values["iterations"]),
        workers=int(values["workers"]),
        seed=int(values["seed"]),
        player0_ev=float(values["player0_ev"]),
        target_ev=float(values["target_ev"]),
        abs_error=float(values["abs_error"]),
        exploitability=float(values["exploitability"]),
        infosets=int(values["infosets"]),
    )


def run_once(
    solver_dir: Path,
    iterations: int,
    workers: int,
    seed: int,
) -> RunMetrics:
    cmd = [
        "cargo",
        "run",
        "-p",
        "cfr",
        "--bin",
        "train_kuhn_linear",
        "--release",
        "--",
        str(iterations),
        str(workers),
        str(seed),
    ]
    proc = subprocess.run(
        cmd,
        cwd=solver_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    return parse_metrics(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated Kuhn Linear CFR training samples.")
    parser.add_argument("--solver-dir", default="solver")
    parser.add_argument("--iterations", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out", default="solver/data/kuhn_linear_metrics.json")
    args = parser.parse_args()

    solver_dir = Path(args.solver_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for idx in range(args.runs):
        run_seed = args.seed + idx
        metrics = run_once(
            solver_dir=solver_dir,
            iterations=args.iterations,
            workers=args.workers,
            seed=run_seed,
        )
        all_runs.append(metrics.__dict__)
        print(
            f"run={idx} seed={run_seed} ev={metrics.player0_ev:.6f} abs_error={metrics.abs_error:.6f} "
            f"exploitability={metrics.exploitability:.6f}"
        )

    payload = {
        "solver_dir": str(solver_dir),
        "iterations_per_run": args.iterations,
        "workers": args.workers,
        "base_seed": args.seed,
        "runs": all_runs,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote metrics to {out_path}")


if __name__ == "__main__":
    main()

