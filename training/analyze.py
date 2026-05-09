"""Offline analysis for solver outputs and training metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def analyze_metrics(path: Path) -> None:
    if not path.exists():
        print(f"metrics not found: {path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points", [])
    if not points:
        print("metrics file has no points")
        return

    best_abs = min(points, key=lambda p: p.get("abs_error", float("inf")))
    best_expl = min(points, key=lambda p: p.get("exploitability", float("inf")))
    avg_abs = mean(p.get("abs_error", 0.0) for p in points)
    avg_expl = mean(p.get("exploitability", 0.0) for p in points)

    print("metrics analysis")
    print(f"points: {len(points)}")
    print(f"target_iterations: {payload.get('total_iterations')}")
    print(
        f"best abs_error: {best_abs.get('abs_error'):.6f} at iter={best_abs.get('iterations')}"
    )
    print(
        f"best exploitability: {best_expl.get('exploitability'):.6f} at iter={best_expl.get('iterations')}"
    )
    print(f"mean abs_error: {avg_abs:.6f}")
    print(f"mean exploitability: {avg_expl:.6f}")


def analyze_blueprint(path: Path) -> None:
    if not path.exists():
        print(f"blueprint not found: {path}")
        return
    size = path.stat().st_size
    print("blueprint analysis")
    print(f"path: {path}")
    print(f"size_bytes: {size}")
    print("detailed blueprint decoding not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exported MCCFR artifacts.")
    parser.add_argument(
        "target",
        nargs="?",
        default="solver/data/kuhn_linear_train_metrics.json",
        help="path to metrics JSON or blueprint/checkpoint binary",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if target.suffix.lower() == ".json":
        analyze_metrics(target)
    else:
        analyze_blueprint(target)


if __name__ == "__main__":
    main()
