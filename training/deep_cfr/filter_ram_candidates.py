#!/usr/bin/env python3
"""Filter frontier candidates using a RAM-aware reservoir budget gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "name",
    "hidden_dim",
    "bottleneck_dim",
    "traversals",
    "training_steps",
    "buffer_size",
    "batch_size",
    "deck_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter frontier candidates by RAM budget.")
    parser.add_argument("--input-json", type=Path, required=True, help="Input candidate JSON list.")
    parser.add_argument("--output-json", type=Path, required=True, help="Filtered candidate JSON list.")
    parser.add_argument("--report-json", type=Path, required=True, help="Detailed RAM gate report JSON.")
    parser.add_argument("--total-ram-bytes", type=int, required=True, help="Host total RAM in bytes.")
    parser.add_argument(
        "--max-reservoir-ram-fraction",
        type=float,
        default=0.45,
        help="Maximum RAM fraction allowed for estimated in-memory reservoirs.",
    )
    parser.add_argument(
        "--reservoir-safety-factor",
        type=float,
        default=1.20,
        help="Safety multiplier applied to estimated reservoir bytes.",
    )
    parser.add_argument("--input-dim", type=int, default=510)
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument(
        "--reservoir-count",
        type=int,
        default=2,
        help="Number of simultaneously allocated reservoirs for training.",
    )
    args = parser.parse_args()
    if args.total_ram_bytes <= 0:
        raise ValueError("--total-ram-bytes must be > 0")
    if not (0.05 <= float(args.max_reservoir_ram_fraction) <= 0.95):
        raise ValueError("--max-reservoir-ram-fraction must be within [0.05, 0.95]")
    if float(args.reservoir_safety_factor) < 1.0:
        raise ValueError("--reservoir-safety-factor must be >= 1.0")
    if args.input_dim <= 0 or args.max_actions <= 0:
        raise ValueError("--input-dim and --max-actions must be > 0")
    if args.reservoir_count <= 0:
        raise ValueError("--reservoir-count must be > 0")
    return args


def read_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"candidate file must be a JSON list: {path}")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"candidate[{idx}] is not an object")
        for field in REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(f"candidate[{idx}] missing required field: {field}")
        normalized = dict(item)
        for field in REQUIRED_FIELDS:
            if field == "name":
                normalized[field] = str(item[field])
            else:
                normalized[field] = int(item[field])
        out.append(normalized)
    if not out:
        raise ValueError("candidate list is empty")
    return out


def per_entry_bytes(*, input_dim: int, max_actions: int) -> int:
    # Matches ReservoirBuffer storage layout:
    # features float32[input_dim], targets float32[max_actions],
    # action_masks uint8[max_actions], iteration int64.
    return int((input_dim * 4) + (max_actions * 4) + max_actions + 8)


def to_gb(raw_bytes: float) -> float:
    return float(raw_bytes / float(1024**3))


def main() -> int:
    args = parse_args()
    candidates = read_candidates(args.input_json.resolve())
    bytes_per_entry = per_entry_bytes(input_dim=int(args.input_dim), max_actions=int(args.max_actions))
    allowed_bytes = int(float(args.total_ram_bytes) * float(args.max_reservoir_ram_fraction))

    rows: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for cand in candidates:
        estimated_bytes = int(
            math.ceil(
                float(cand["buffer_size"])
                * float(bytes_per_entry)
                * float(args.reservoir_count)
                * float(args.reservoir_safety_factor)
            )
        )
        passes = estimated_bytes <= allowed_bytes
        row = {
            "name": str(cand["name"]),
            "buffer_size": int(cand["buffer_size"]),
            "estimated_reservoir_bytes": int(estimated_bytes),
            "estimated_reservoir_gb": to_gb(estimated_bytes),
            "allowed_reservoir_bytes": int(allowed_bytes),
            "allowed_reservoir_gb": to_gb(allowed_bytes),
            "keep": bool(passes),
        }
        rows.append(row)
        if passes:
            kept.append(cand)
        else:
            dropped.append(cand)

    status = "ok" if kept else "failed"
    report = {
        "status": status,
        "input_json": str(args.input_json.resolve()),
        "output_json": str(args.output_json.resolve()),
        "total_ram_bytes": int(args.total_ram_bytes),
        "total_ram_gb": to_gb(float(args.total_ram_bytes)),
        "max_reservoir_ram_fraction": float(args.max_reservoir_ram_fraction),
        "reservoir_safety_factor": float(args.reservoir_safety_factor),
        "reservoir_count": int(args.reservoir_count),
        "input_dim": int(args.input_dim),
        "max_actions": int(args.max_actions),
        "bytes_per_entry_per_reservoir": int(bytes_per_entry),
        "allowed_reservoir_bytes": int(allowed_bytes),
        "allowed_reservoir_gb": to_gb(allowed_bytes),
        "total_candidates": len(candidates),
        "kept_candidates": len(kept),
        "dropped_candidates": len(dropped),
        "kept_names": [str(c["name"]) for c in kept],
        "dropped_names": [str(c["name"]) for c in dropped],
        "rows": rows,
    }

    args.report_json = args.report_json.resolve()
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if kept:
        args.output_json = args.output_json.resolve()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(kept, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True))
        return 0

    print(json.dumps(report, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
