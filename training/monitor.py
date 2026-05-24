"""Training progress monitor for MCCFR runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def print_kuhn_summary(payload: dict) -> None:
    points = payload.get("points", [])
    if not points:
        print("no points available")
        return

    latest = points[-1]
    first = points[0]
    avg_abs_error = mean(p.get("abs_error", 0.0) for p in points)
    avg_exploitability = mean(p.get("exploitability", 0.0) for p in points)

    print("run summary")
    print(f"iterations target: {payload.get('total_iterations')}")
    print(f"batches logged: {len(points)}")
    print(
        f"latest: iter={latest.get('iterations')} "
        f"ev={latest.get('player0_ev'):.6f} "
        f"abs_error={latest.get('abs_error'):.6f} "
        f"exploitability={latest.get('exploitability'):.6f}"
    )
    print(
        f"delta from first: abs_error={latest.get('abs_error') - first.get('abs_error'):+.6f} "
        f"exploitability={latest.get('exploitability') - first.get('exploitability'):+.6f}"
    )
    print(f"mean abs_error={avg_abs_error:.6f}")
    print(f"mean exploitability={avg_exploitability:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and summarize MCCFR run metrics.")
    parser.add_argument("metrics", nargs="?", default="solver/data/kuhn_linear_train_metrics.json")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="print raw JSON instead of summarized output",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    payload = load_metrics(metrics_path)
    if not payload:
        print(f"no metrics found at {metrics_path}")
        return

    if args.raw:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print_kuhn_summary(payload)


if __name__ == "__main__":
    main()
