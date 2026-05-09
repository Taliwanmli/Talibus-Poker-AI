#!/usr/bin/env python3
"""Run baseline hero modes and print side-by-side EV summary."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run always_fold/random/argmax baseline evaluations")
    parser.add_argument("--hands", type=int, default=10_000, help="Hands per mode")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--players", type=int, default=6, choices=[2, 3, 4, 5, 6], help="Table player count")
    parser.add_argument("--stack-bb", type=int, default=100, help="Starting stack in bb")
    parser.add_argument("--progress-every", type=int, default=2_000, help="Progress cadence per run")
    parser.add_argument("--rake-mode", choices=["none", "fixed", "percent"], default="none", help="Rake model")
    parser.add_argument("--rake-fixed-bb", type=float, default=0.0)
    parser.add_argument("--rake-percent", type=float, default=0.05)
    parser.add_argument("--rake-cap-bb", type=float, default=3.0)
    parser.add_argument("--build-policy", action="store_true", help="Run npm run eval:build before baseline runs")
    return parser.parse_args()


def parse_summary(stdout_text: str) -> Dict[str, str]:
    bb100_values = re.findall(r"bb/100:\s*([-+]?\d+(?:\.\d+)?)", stdout_text)
    fallback_match = re.search(r"fallback_count:\s*(\d+)", stdout_text)
    ok_match = re.search(r"ok:\s*(\d+)", stdout_text)
    unavailable_match = re.search(r"unavailable:\s*(\d+)", stdout_text)
    missing_match = re.search(r"missing_fields:\s*(\d+)", stdout_text)

    raw_bb100 = bb100_values[0] if len(bb100_values) >= 1 else "n/a"
    net_bb100 = bb100_values[1] if len(bb100_values) >= 2 else "n/a"

    return {
        "raw_bb100": raw_bb100,
        "net_bb100": net_bb100,
        "fallback_count": fallback_match.group(1) if fallback_match else "0",
        "ok_count": ok_match.group(1) if ok_match else "0",
        "unavailable_count": unavailable_match.group(1) if unavailable_match else "0",
        "missing_fields_count": missing_match.group(1) if missing_match else "0",
    }


def run_mode(mode: str, args: argparse.Namespace) -> Dict[str, str]:
    result_path = EVAL_DIR / f"results_{mode}.csv"
    cmd: List[str] = [
        sys.executable,
        "eval/run_league.py",
        "--hands",
        str(args.hands),
        "--seed",
        str(args.seed),
        "--players",
        str(args.players),
        "--stack-bb",
        str(args.stack_bb),
        "--hero_mode",
        mode,
        "--progress-every",
        str(args.progress_every),
        "--rake-mode",
        args.rake_mode,
        "--rake-fixed-bb",
        str(args.rake_fixed_bb),
        "--rake-percent",
        str(args.rake_percent),
        "--rake-cap-bb",
        str(args.rake_cap_bb),
        "--results",
        str(result_path),
    ]

    print(f"\nRunning mode={mode} ...")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    print(proc.stdout)
    return parse_summary(proc.stdout)


def print_table(rows: List[Dict[str, str]]) -> None:
    print("\n=== Baseline Comparison ===")
    header = [
        "mode",
        "raw_bb/100",
        "net_bb/100",
        "fallbacks",
        "ok",
        "unavailable",
        "missing_fields",
    ]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for row in rows:
        print(
            " | ".join(
                [
                    row["mode"],
                    row["raw_bb100"],
                    row["net_bb100"],
                    row["fallback_count"],
                    row["ok_count"],
                    row["unavailable_count"],
                    row["missing_fields_count"],
                ]
            )
        )


def main() -> int:
    args = parse_args()

    if args.build_policy:
        subprocess.run(["npm", "run", "eval:build"], cwd=str(REPO_ROOT), check=True)

    rows: List[Dict[str, str]] = []
    for mode in ["always_fold", "random", "argmax"]:
        metrics = run_mode(mode, args)
        metrics["mode"] = mode
        rows.append(metrics)

    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
