#!/usr/bin/env python3
"""Build independent 6-max evaluation panels with seat-coverage gates."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import INPUT_DIM, MAX_ACTIONS
from subprocess_env import build_subprocess_env
from train import STRATEGY_SAMPLE_MAGIC, SAMPLE_VERSION, load_binary_samples


CARD_DIMS = 52
STREET_DIMS = 4
POSITION_DIMS = 6
HERO_POSITION_OFFSET = CARD_DIMS + CARD_DIMS + STREET_DIMS
ACTOR_POSITION_OFFSET = HERO_POSITION_OFFSET + POSITION_DIMS
HEADER_STRUCT = struct.Struct("<4sIII")


@dataclass(frozen=True)
class CoverageGate:
    required_seats: tuple[int, ...]
    min_seat_samples: int
    max_dominant_seat_fraction: float


def parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        text = token.strip()
        if not text:
            continue
        out.append(int(text))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent Deep CFR strategy panels.")
    parser.add_argument("--model-onnx", type=Path, required=True, help="ONNX checkpoint used for traversal.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for panel artifacts.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional report path.")
    parser.add_argument("--rust-binary", type=Path, default=None, help="Path to run_traversals binary.")
    parser.add_argument("--cluster-dir", type=Path, default=REPO_ROOT / "checkpoints" / "nlhe_clusters")
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--traversers", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--traversals-per-traverser", type=int, default=400)
    parser.add_argument("--deck-samples", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-batch", type=int, default=500)
    parser.add_argument("--starting-stack", type=int, default=2000)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--big-blind", type=int, default=20)
    parser.add_argument("--screen-seed", type=int, default=101)
    parser.add_argument("--holdout-seed", type=int, default=202)
    parser.add_argument(
        "--target-samples-per-seat",
        type=int,
        default=3000,
        help="Selected strategy samples per actor seat per panel (0 keeps all).",
    )
    parser.add_argument("--required-seats", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--min-seat-samples", type=int, default=1)
    parser.add_argument("--max-dominant-seat-fraction", type=float, default=0.70)
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw traversal output files.")
    args = parser.parse_args()

    if args.num_players < 2 or args.num_players > POSITION_DIMS:
        raise ValueError(f"--num-players must be in [2, {POSITION_DIMS}]")
    if args.traversals_per_traverser <= 0:
        raise ValueError("--traversals-per-traverser must be > 0")
    if args.deck_samples <= 0:
        raise ValueError("--deck-samples must be > 0")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.progress_batch <= 0:
        raise ValueError("--progress-batch must be > 0")
    if args.target_samples_per_seat < 0:
        raise ValueError("--target-samples-per-seat must be >= 0")
    if args.min_seat_samples < 0:
        raise ValueError("--min-seat-samples must be >= 0")
    if not (0.0 <= args.max_dominant_seat_fraction <= 1.0):
        raise ValueError("--max-dominant-seat-fraction must be in [0, 1]")
    return args


def resolve_rust_binary(args: argparse.Namespace) -> Path:
    if args.rust_binary is not None:
        return args.rust_binary.resolve()
    binary_name = "run_traversals.exe" if sys.platform.startswith("win") else "run_traversals"
    return (REPO_ROOT / "solver" / "target" / "release" / binary_name).resolve()


def actor_indices(features: np.ndarray) -> np.ndarray:
    actor_slice = features[:, ACTOR_POSITION_OFFSET : ACTOR_POSITION_OFFSET + POSITION_DIMS]
    actor_argmax = np.argmax(actor_slice, axis=1)
    actor_peak = np.max(actor_slice, axis=1)
    return np.where(actor_peak > 0.5, actor_argmax, -1).astype(np.int64)


def run_traversal(
    *,
    rust_binary: Path,
    onnx_path: Path,
    cluster_dir: Path,
    output_adv_path: Path,
    output_strategy_path: Path,
    player: int,
    traversals: int,
    progress_batch: int,
    workers: int,
    seed: int,
    num_players: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    deck_samples: int,
) -> dict[str, Any]:
    command = [
        str(rust_binary),
        "--onnx",
        str(onnx_path),
        "--adv-samples-out",
        str(output_adv_path),
        "--strategy-samples-out",
        str(output_strategy_path),
        "--player",
        str(player),
        "--traversals",
        str(traversals),
        "--progress-batch",
        str(progress_batch),
        "--workers",
        str(workers),
        "--iteration",
        "1",
        "--seed",
        str(seed & 0xFFFF_FFFF_FFFF_FFFF),
        "--cluster-dir",
        str(cluster_dir),
        "--deck-samples",
        str(deck_samples),
        "--num-players",
        str(num_players),
        "--starting-stack",
        str(starting_stack),
        "--small-blind",
        str(small_blind),
        "--big-blind",
        str(big_blind),
    ]
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=build_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"run_traversals failed for traverser={player}: {tail}")
    return {
        "player": player,
        "seed": int(seed),
        "advantage_path": str(output_adv_path),
        "strategy_path": str(output_strategy_path),
    }


def load_strategy_parts(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features_parts: list[np.ndarray] = []
    targets_parts: list[np.ndarray] = []
    masks_parts: list[np.ndarray] = []
    iteration_parts: list[np.ndarray] = []
    for path in paths:
        features, targets, action_masks, iterations = load_binary_samples(
            path,
            expected_magic=STRATEGY_SAMPLE_MAGIC,
        )
        features_parts.append(features)
        targets_parts.append(targets)
        masks_parts.append(action_masks)
        iteration_parts.append(iterations)
    if not features_parts:
        raise ValueError("no strategy sample files to load")
    return (
        np.concatenate(features_parts, axis=0),
        np.concatenate(targets_parts, axis=0),
        np.concatenate(masks_parts, axis=0),
        np.concatenate(iteration_parts, axis=0),
    )


def select_stratified_indices(
    *,
    actor_idx: np.ndarray,
    rng: np.random.Generator,
    num_players: int,
    target_samples_per_seat: int,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    raw_counts = {seat: int(np.sum(actor_idx == seat)) for seat in range(num_players)}
    selected: list[np.ndarray] = []
    selected_counts: dict[int, int] = {}
    for seat in range(num_players):
        seat_indices = np.flatnonzero(actor_idx == seat)
        if seat_indices.size == 0:
            selected_counts[seat] = 0
            continue
        if target_samples_per_seat == 0:
            take = int(seat_indices.size)
        else:
            take = int(min(seat_indices.size, target_samples_per_seat))
        if take <= 0:
            selected_counts[seat] = 0
            continue
        if take == seat_indices.size:
            chosen = seat_indices
        else:
            chosen = np.sort(rng.choice(seat_indices, size=take, replace=False))
        selected.append(chosen)
        selected_counts[seat] = int(take)
    if not selected:
        return np.empty((0,), dtype=np.int64), raw_counts, selected_counts
    merged = np.sort(np.concatenate(selected, axis=0))
    return merged.astype(np.int64, copy=False), raw_counts, selected_counts


def write_strategy_panel(
    *,
    path: Path,
    features: np.ndarray,
    targets: np.ndarray,
    action_masks: np.ndarray,
    iterations: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        fp.write(
            HEADER_STRUCT.pack(
                STRATEGY_SAMPLE_MAGIC,
                SAMPLE_VERSION,
                INPUT_DIM,
                MAX_ACTIONS,
            )
        )
        count = int(features.shape[0])
        for idx in range(count):
            fp.write(np.asarray(features[idx], dtype=np.float32).tobytes(order="C"))
            fp.write(np.asarray(targets[idx], dtype=np.float32).tobytes(order="C"))
            mask = np.asarray(action_masks[idx], dtype=np.uint8)
            mask = np.where(mask > 0, 1, 0).astype(np.uint8, copy=False)
            fp.write(mask.tobytes(order="C"))
            fp.write(struct.pack("<I", int(iterations[idx]) & 0xFFFF_FFFF))


def evaluate_coverage(
    *,
    selected_counts: dict[int, int],
    total_selected: int,
    gate: CoverageGate,
) -> dict[str, Any]:
    required_missing = [seat for seat in gate.required_seats if selected_counts.get(seat, 0) == 0]
    min_samples_fail = [
        seat for seat in gate.required_seats if selected_counts.get(seat, 0) < gate.min_seat_samples
    ]
    dominant_seat = -1
    dominant_count = 0
    if selected_counts:
        dominant_seat, dominant_count = max(selected_counts.items(), key=lambda kv: int(kv[1]))
    dominant_fraction = float(dominant_count / total_selected) if total_selected > 0 else 1.0
    reasons: list[str] = []
    if required_missing:
        reasons.append(f"required seat buckets empty: {required_missing}")
    if min_samples_fail:
        reasons.append(
            f"required seats below min samples ({gate.min_seat_samples}): {min_samples_fail}"
        )
    if dominant_fraction > gate.max_dominant_seat_fraction:
        reasons.append(
            "dominant seat fraction exceeds limit: "
            f"seat={dominant_seat} fraction={dominant_fraction:.4f} "
            f"limit={gate.max_dominant_seat_fraction:.4f}"
        )
    passed = len(reasons) == 0
    return {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "required_seats": list(gate.required_seats),
        "min_seat_samples": int(gate.min_seat_samples),
        "max_dominant_seat_fraction": float(gate.max_dominant_seat_fraction),
        "seat_counts_selected": {str(k): int(v) for k, v in selected_counts.items()},
        "total_selected": int(total_selected),
        "dominant_seat": int(dominant_seat),
        "dominant_count": int(dominant_count),
        "dominant_fraction": float(dominant_fraction),
        "reasons": reasons,
    }


def build_panel(
    *,
    panel_name: str,
    base_seed: int,
    args: argparse.Namespace,
    rust_binary: Path,
    traversers: list[int],
    gate: CoverageGate,
) -> dict[str, Any]:
    panel_dir = args.output_dir / panel_name
    raw_dir = panel_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    strategy_files: list[Path] = []
    runs: list[dict[str, Any]] = []
    for traverser in traversers:
        adv_path = raw_dir / f"panel_{panel_name}_adv_p{traverser}.bin"
        strategy_path = raw_dir / f"panel_{panel_name}_strategy_p{traverser}.bin"
        seed = int(base_seed ^ ((traverser + 1) * 0x9E37_79B9) ^ (args.num_players * 0xA24B_AED4))
        run_info = run_traversal(
            rust_binary=rust_binary,
            onnx_path=args.model_onnx.resolve(),
            cluster_dir=args.cluster_dir.resolve(),
            output_adv_path=adv_path,
            output_strategy_path=strategy_path,
            player=int(traverser),
            traversals=int(args.traversals_per_traverser),
            progress_batch=int(args.progress_batch),
            workers=int(args.workers),
            seed=seed,
            num_players=int(args.num_players),
            starting_stack=int(args.starting_stack),
            small_blind=int(args.small_blind),
            big_blind=int(args.big_blind),
            deck_samples=int(args.deck_samples),
        )
        runs.append(run_info)
        strategy_files.append(strategy_path)

    features, targets, action_masks, iterations = load_strategy_parts(strategy_files)
    actor_idx = actor_indices(features)
    rng = np.random.default_rng(int(base_seed))
    selected_idx, raw_counts, selected_counts = select_stratified_indices(
        actor_idx=actor_idx,
        rng=rng,
        num_players=int(args.num_players),
        target_samples_per_seat=int(args.target_samples_per_seat),
    )
    selected_features = features[selected_idx]
    selected_targets = targets[selected_idx]
    selected_masks = action_masks[selected_idx]
    selected_iterations = iterations[selected_idx]
    panel_path = args.output_dir / f"{panel_name}_panel_strategy.bin"
    write_strategy_panel(
        path=panel_path,
        features=selected_features,
        targets=selected_targets,
        action_masks=selected_masks,
        iterations=selected_iterations,
    )
    coverage = evaluate_coverage(
        selected_counts=selected_counts,
        total_selected=int(selected_idx.shape[0]),
        gate=gate,
    )
    if not args.keep_raw:
        for run in runs:
            for key in ("advantage_path", "strategy_path"):
                raw_path = Path(str(run.get(key, "")))
                if raw_path.exists():
                    raw_path.unlink()
        if raw_dir.exists():
            raw_dir.rmdir()

    return {
        "status": str(coverage.get("status", "failed")),
        "panel_name": panel_name,
        "panel_path": str(panel_path.resolve()),
        "seed": int(base_seed),
        "raw_runs": runs,
        "raw_counts_by_seat": {str(k): int(v) for k, v in raw_counts.items()},
        "selected_counts_by_seat": {str(k): int(v) for k, v in selected_counts.items()},
        "total_raw_samples": int(features.shape[0]),
        "total_selected_samples": int(selected_features.shape[0]),
        "coverage": coverage,
    }


def main() -> int:
    args = parse_args()
    rust_binary = resolve_rust_binary(args)
    if not rust_binary.exists():
        raise FileNotFoundError(f"missing run_traversals binary: {rust_binary}")
    if not args.model_onnx.exists():
        raise FileNotFoundError(f"missing model onnx: {args.model_onnx}")
    if not args.cluster_dir.exists():
        raise FileNotFoundError(f"missing cluster dir: {args.cluster_dir}")

    traversers = parse_csv_ints(args.traversers)
    if not traversers:
        raise ValueError("--traversers must include at least one seat index")
    bad = [seat for seat in traversers if seat < 0 or seat >= args.num_players]
    if bad:
        raise ValueError(f"--traversers includes out-of-range seat(s): {bad}")

    required_seats = tuple(parse_csv_ints(args.required_seats))
    bad_required = [seat for seat in required_seats if seat < 0 or seat >= args.num_players]
    if bad_required:
        raise ValueError(f"--required-seats includes out-of-range seat(s): {bad_required}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate = CoverageGate(
        required_seats=required_seats,
        min_seat_samples=int(args.min_seat_samples),
        max_dominant_seat_fraction=float(args.max_dominant_seat_fraction),
    )
    screen = build_panel(
        panel_name="screen",
        base_seed=int(args.screen_seed),
        args=args,
        rust_binary=rust_binary,
        traversers=traversers,
        gate=gate,
    )
    holdout = build_panel(
        panel_name="holdout",
        base_seed=int(args.holdout_seed),
        args=args,
        rust_binary=rust_binary,
        traversers=traversers,
        gate=gate,
    )
    status = "ok" if screen.get("status") == "ok" and holdout.get("status") == "ok" else "failed"
    report = {
        "status": status,
        "config": {
            "model_onnx": str(args.model_onnx.resolve()),
            "rust_binary": str(rust_binary),
            "cluster_dir": str(args.cluster_dir.resolve()),
            "num_players": int(args.num_players),
            "traversers": [int(v) for v in traversers],
            "traversals_per_traverser": int(args.traversals_per_traverser),
            "deck_samples": int(args.deck_samples),
            "workers": int(args.workers),
            "progress_batch": int(args.progress_batch),
            "starting_stack": int(args.starting_stack),
            "small_blind": int(args.small_blind),
            "big_blind": int(args.big_blind),
            "screen_seed": int(args.screen_seed),
            "holdout_seed": int(args.holdout_seed),
            "target_samples_per_seat": int(args.target_samples_per_seat),
            "required_seats": [int(v) for v in required_seats],
            "min_seat_samples": int(args.min_seat_samples),
            "max_dominant_seat_fraction": float(args.max_dominant_seat_fraction),
        },
        "panels": {
            "screen": screen,
            "holdout": holdout,
        },
    }
    report_path = (
        args.report_json.resolve()
        if args.report_json is not None
        else (args.output_dir / "panel_manifest.json").resolve()
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
