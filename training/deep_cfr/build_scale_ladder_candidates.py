#!/usr/bin/env python3
"""Build bounded stage-2 ladder candidates from stage-1 probe results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FIELDS = (
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
    parser = argparse.ArgumentParser(description="Build stage-2 ladder candidates from stage-1 scale report.")
    parser.add_argument("--probe-report", type=Path, required=True, help="Stage-1 probe report JSON.")
    parser.add_argument("--output-json", type=Path, required=True, help="Stage-2 ladder candidates JSON output.")
    parser.add_argument(
        "--incumbent-name",
        type=str,
        default="tier_c_baseline",
        help="Incumbent candidate name (fallback to recommended candidate if missing).",
    )
    parser.add_argument("--max-hidden-dim", type=int, default=1408)
    parser.add_argument("--max-bottleneck-dim", type=int, default=704)
    parser.add_argument("--max-traversals", type=int, default=5400)
    parser.add_argument("--max-training-steps", type=int, default=420)
    parser.add_argument("--max-buffer-size", type=int, default=650000)
    args = parser.parse_args()
    if args.max_hidden_dim <= 0 or args.max_bottleneck_dim <= 0:
        raise ValueError("max width caps must be > 0")
    if args.max_traversals <= 0 or args.max_training_steps <= 0 or args.max_buffer_size <= 0:
        raise ValueError("max traversal/training/buffer caps must be > 0")
    return args


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing probe report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"probe report is not an object: {path}")
    return payload


def normalize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in FIELDS:
        if field not in raw:
            raise ValueError(f"candidate missing required field: {field}")
        out[field] = str(raw[field]) if field == "name" else int(raw[field])
    return out


def changed_axes(inc: dict[str, Any], cand: dict[str, Any]) -> list[str]:
    axes: list[str] = []
    if int(cand["hidden_dim"]) != int(inc["hidden_dim"]) or int(cand["bottleneck_dim"]) != int(inc["bottleneck_dim"]):
        axes.append("width")
    if int(cand["traversals"]) != int(inc["traversals"]):
        axes.append("traversals")
    if int(cand["training_steps"]) != int(inc["training_steps"]):
        axes.append("training_steps")
    if int(cand["buffer_size"]) != int(inc["buffer_size"]):
        axes.append("buffer_size")
    return axes


def candidate_key(cand: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(cand["hidden_dim"]),
        int(cand["bottleneck_dim"]),
        int(cand["traversals"]),
        int(cand["training_steps"]),
        int(cand["buffer_size"]),
        int(cand["batch_size"]),
        int(cand["deck_samples"]),
    )


def with_name(cand: dict[str, Any], name: str) -> dict[str, Any]:
    out = dict(cand)
    out["name"] = str(name)
    return out


def apply_axis_delta(
    *,
    base: dict[str, Any],
    axis: str,
    delta_hidden: int = 0,
    delta_bottleneck: int = 0,
    delta_traversals: int = 0,
    delta_training_steps: int = 0,
    delta_buffer_size: int = 0,
    caps: dict[str, int],
    name: str,
) -> dict[str, Any]:
    out = dict(base)
    if axis == "width":
        out["hidden_dim"] = int(min(caps["hidden_dim"], int(base["hidden_dim"]) + int(delta_hidden)))
        out["bottleneck_dim"] = int(min(caps["bottleneck_dim"], int(base["bottleneck_dim"]) + int(delta_bottleneck)))
    elif axis == "traversals":
        out["traversals"] = int(min(caps["traversals"], int(base["traversals"]) + int(delta_traversals)))
    elif axis == "training_steps":
        out["training_steps"] = int(min(caps["training_steps"], int(base["training_steps"]) + int(delta_training_steps)))
    elif axis == "buffer_size":
        out["buffer_size"] = int(min(caps["buffer_size"], int(base["buffer_size"]) + int(delta_buffer_size)))
    else:
        raise ValueError(f"unknown axis: {axis}")
    out["name"] = str(name)
    return out


def main() -> int:
    args = parse_args()
    report = read_json(args.probe_report.resolve())
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError("probe report has no results list")

    ranking = report.get("ranking", [])
    ranked_names: list[str] = [str(item) for item in ranking] if isinstance(ranking, list) else []
    rank_pos = {name: idx for idx, name in enumerate(ranked_names)}

    by_name_all: dict[str, dict[str, Any]] = {}
    by_name_valid: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        raw_cand = row.get("candidate", {})
        if not isinstance(raw_cand, dict):
            continue
        cand = normalize_candidate(raw_cand)
        name = str(cand["name"])
        by_name_all[name] = cand
        if str(row.get("status", "error")) == "ok":
            by_name_valid[name] = cand

    if not by_name_all:
        raise ValueError("probe report has no candidates")

    incumbent = by_name_all.get(str(args.incumbent_name))
    if incumbent is None:
        recommended = report.get("recommended_candidate", {})
        rec_cand = recommended.get("candidate", {}) if isinstance(recommended, dict) else {}
        if isinstance(rec_cand, dict) and rec_cand:
            incumbent = normalize_candidate(rec_cand)
        else:
            # deterministic fallback
            first_name = sorted(by_name_all.keys())[0]
            incumbent = by_name_all[first_name]
    assert incumbent is not None

    axis_rows: dict[str, list[dict[str, Any]]] = {
        "width": [],
        "traversals": [],
        "training_steps": [],
        "buffer_size": [],
    }
    for name, cand in by_name_valid.items():
        if candidate_key(cand) == candidate_key(incumbent):
            continue
        axes = changed_axes(incumbent, cand)
        if len(axes) == 1:
            axis_rows[axes[0]].append(cand)

    def axis_rank(axis: str) -> int:
        candidates = axis_rows.get(axis, [])
        if not candidates:
            return 10**9
        best = 10**9
        for cand in candidates:
            pos = rank_pos.get(str(cand["name"]), 10**9)
            best = min(best, int(pos))
        return best

    axis_order = sorted(["width", "traversals", "training_steps", "buffer_size"], key=axis_rank)
    winner_axis = axis_order[0]
    runner_axis = axis_order[1] if len(axis_order) > 1 else None

    # fallback when stage-1 had no valid single-axis rows
    if axis_rank(winner_axis) >= 10**9:
        winner_axis = "width"
        runner_axis = "traversals"

    def best_axis_l1(axis: str) -> dict[str, Any] | None:
        rows = axis_rows.get(axis, [])
        if not rows:
            return None
        rows_sorted = sorted(rows, key=lambda c: rank_pos.get(str(c["name"]), 10**9))
        return rows_sorted[0]

    winner_l1 = best_axis_l1(winner_axis)
    runner_l1 = best_axis_l1(runner_axis) if runner_axis is not None else None

    caps = {
        "hidden_dim": int(args.max_hidden_dim),
        "bottleneck_dim": int(args.max_bottleneck_dim),
        "traversals": int(args.max_traversals),
        "training_steps": int(args.max_training_steps),
        "buffer_size": int(args.max_buffer_size),
    }

    # default deltas if axis l1 is missing
    default_deltas = {
        "width": (256, 128, 0, 0),
        "traversals": (0, 0, 1500, 0),
        "training_steps": (0, 0, 0, 120),
        "buffer_size": (0, 0, 0, 0, 150000),
    }

    if winner_l1 is not None:
        d_hidden = int(winner_l1["hidden_dim"]) - int(incumbent["hidden_dim"])
        d_bottle = int(winner_l1["bottleneck_dim"]) - int(incumbent["bottleneck_dim"])
        d_trav = int(winner_l1["traversals"]) - int(incumbent["traversals"])
        d_steps = int(winner_l1["training_steps"]) - int(incumbent["training_steps"])
        d_buffer = int(winner_l1["buffer_size"]) - int(incumbent["buffer_size"])
        if d_hidden == 0 and d_bottle == 0 and d_trav == 0 and d_steps == 0 and d_buffer == 0:
            if winner_axis == "buffer_size":
                d_hidden, d_bottle, d_trav, d_steps, d_buffer = default_deltas["buffer_size"]
            else:
                d_hidden, d_bottle, d_trav, d_steps = default_deltas[winner_axis]
                d_buffer = 0
    else:
        if winner_axis == "buffer_size":
            d_hidden, d_bottle, d_trav, d_steps, d_buffer = default_deltas["buffer_size"]
        else:
            d_hidden, d_bottle, d_trav, d_steps = default_deltas[winner_axis]
            d_buffer = 0
        winner_l1 = apply_axis_delta(
            base=incumbent,
            axis=winner_axis,
            delta_hidden=d_hidden,
            delta_bottleneck=d_bottle,
            delta_traversals=d_trav,
            delta_training_steps=d_steps,
            delta_buffer_size=d_buffer,
            caps=caps,
            name=f"ladder_{winner_axis}_l1",
        )

    winner_l2 = apply_axis_delta(
        base=incumbent,
        axis=winner_axis,
        delta_hidden=(2 * d_hidden),
        delta_bottleneck=(2 * d_bottle),
        delta_traversals=(2 * d_trav),
        delta_training_steps=(2 * d_steps),
        delta_buffer_size=(2 * d_buffer),
        caps=caps,
        name=f"ladder_{winner_axis}_l2",
    )

    interaction: dict[str, Any] | None = None
    if runner_axis is not None and runner_l1 is not None:
        rd_hidden = int(runner_l1["hidden_dim"]) - int(incumbent["hidden_dim"])
        rd_bottle = int(runner_l1["bottleneck_dim"]) - int(incumbent["bottleneck_dim"])
        rd_trav = int(runner_l1["traversals"]) - int(incumbent["traversals"])
        rd_steps = int(runner_l1["training_steps"]) - int(incumbent["training_steps"])
        rd_buffer = int(runner_l1["buffer_size"]) - int(incumbent["buffer_size"])
        interaction = dict(winner_l2)
        if runner_axis == "width":
            interaction["hidden_dim"] = int(min(caps["hidden_dim"], int(winner_l2["hidden_dim"]) + rd_hidden))
            interaction["bottleneck_dim"] = int(
                min(caps["bottleneck_dim"], int(winner_l2["bottleneck_dim"]) + rd_bottle)
            )
        elif runner_axis == "traversals":
            interaction["traversals"] = int(min(caps["traversals"], int(winner_l2["traversals"]) + rd_trav))
        elif runner_axis == "training_steps":
            interaction["training_steps"] = int(
                min(caps["training_steps"], int(winner_l2["training_steps"]) + rd_steps)
            )
        elif runner_axis == "buffer_size":
            interaction["buffer_size"] = int(
                min(caps["buffer_size"], int(winner_l2["buffer_size"]) + rd_buffer)
            )
        interaction["name"] = f"ladder_{winner_axis}_plus_{runner_axis}"

    out: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, int, int, int, int, int]] = set()

    def push(cand: dict[str, Any] | None) -> None:
        if cand is None:
            return
        key = candidate_key(cand)
        if key in seen_keys:
            return
        seen_keys.add(key)
        out.append(normalize_candidate(cand))

    push(with_name(incumbent, str(incumbent["name"])))
    push(winner_l1)
    push(winner_l2)
    push(runner_l1)
    push(interaction)

    if len(out) < 2:
        raise ValueError("ladder selection produced too few candidates")

    args.output_json = args.output_json.resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    payload = {
        "status": "ok",
        "probe_report": str(args.probe_report.resolve()),
        "output_json": str(args.output_json),
        "winner_axis": winner_axis,
        "runner_axis": runner_axis,
        "candidate_names": [str(c["name"]) for c in out],
        "count": len(out),
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
