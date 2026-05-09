"""Quick benchmark harness for solver experiments."""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a placeholder benchmark loop.")
    parser.add_argument("--iterations", type=int, default=1_000_000)
    args = parser.parse_args()

    start = time.perf_counter()
    acc = 0
    for i in range(args.iterations):
        acc ^= i
    elapsed = time.perf_counter() - start
    print(f"iterations={args.iterations} elapsed_sec={elapsed:.6f} checksum={acc}")


if __name__ == "__main__":
    main()

