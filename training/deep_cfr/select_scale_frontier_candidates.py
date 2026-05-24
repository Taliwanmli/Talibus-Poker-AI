#!/usr/bin/env python3
"""Select stage-2 challenger candidates from a scale frontier report."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Select bounded stage-2 candidates from scale search report.")
    parser.add_argument("--scale-report", type=Path, required=True, help="Input report from run_scale_search.py")
    parser.add_argument("--output-json", type=Path, required=True, help="Output candidates JSON for stage-2 search")
    parser.add_argument("--top-k", type=int, default=2, help="How many ranked candidates to carry forward")
    parser.add_argument(
        "--include-name",
        action="append",
        default=[],
        help="Candidate name to force-include when present in report (repeatable).",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    return args


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report is not an object: {path}")
    return payload


def normalize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        if field not in raw:
            raise ValueError(f"candidate missing field: {field}")
        value = raw[field]
        out[field] = str(value) if field == "name" else int(value)
    return out


def main() -> int:
    args = parse_args()
    report = read_json(args.scale_report.resolve())
    results = report.get("results", [])
    if not isinstance(results, list):
        raise ValueError("report.results must be a list")

    by_name_all: dict[str, dict[str, Any]] = {}
    by_name_valid: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        candidate_raw = row.get("candidate", {})
        if not isinstance(candidate_raw, dict):
            continue
        candidate = normalize_candidate(candidate_raw)
        name = str(candidate["name"])
        by_name_all[name] = candidate
        if str(row.get("status", "error")) == "ok":
            by_name_valid[name] = candidate

    ranking = report.get("ranking", [])
    ranked_names: list[str] = []
    if isinstance(ranking, list):
        for item in ranking:
            name = str(item)
            if name in by_name_valid:
                ranked_names.append(name)

    # If ranking is missing or malformed, fallback to deterministic order from valid rows.
    if not ranked_names:
        for name in by_name_valid:
            ranked_names.append(name)

    selected_names: list[str] = []
    for name in ranked_names:
        if len(selected_names) >= int(args.top_k):
            break
        if name not in selected_names:
            selected_names.append(name)

    for forced_name in args.include_name:
        name = str(forced_name)
        if name in by_name_all and name not in selected_names:
            selected_names.append(name)

    selected: list[dict[str, Any]] = [by_name_all[name] for name in selected_names if name in by_name_all]
    if not selected:
        raise ValueError("no candidates selected; check report validity and selection flags")

    args.output_json = args.output_json.resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    payload = {
        "status": "ok",
        "scale_report": str(args.scale_report.resolve()),
        "output_json": str(args.output_json),
        "selected_names": selected_names,
        "selected_count": len(selected_names),
        "top_k": int(args.top_k),
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
