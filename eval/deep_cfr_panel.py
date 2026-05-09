#!/usr/bin/env python3
"""Evaluate Deep CFR checkpoints on a fixed binary sample panel.

This script is intentionally deterministic and lightweight:
- reads a fixed sample panel (strategy or advantage binary samples)
- runs one checkpoint
- returns both quality metrics and a stable policy fingerprint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training" / "deep_cfr"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from model import INPUT_DIM, MAX_ACTIONS, DeepCfrNet, ModelConfig
from train import (
    ADVANTAGE_SAMPLE_MAGIC,
    STRATEGY_SAMPLE_MAGIC,
    load_binary_samples,
    load_model_weights,
    resolve_device,
)

CARD_DIMS = 52
STREET_DIMS = 4
POSITION_DIMS = 6
HERO_POSITION_OFFSET = CARD_DIMS + CARD_DIMS + STREET_DIMS
ACTOR_POSITION_OFFSET = HERO_POSITION_OFFSET + POSITION_DIMS


def parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = token.strip()
        if not text:
            continue
        out.append(int(text))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Deep CFR checkpoint on a fixed panel.")
    parser.add_argument("--model-state", type=Path, required=True, help="Path to .pt checkpoint (or .onnx sibling).")
    parser.add_argument("--samples", type=Path, required=True, help="Binary sample panel file.")
    parser.add_argument("--sample-type", choices=["strategy", "advantage"], default="strategy")
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--label", type=str, default="checkpoint")
    parser.add_argument(
        "--required-seats",
        type=str,
        default="",
        help="Comma-separated actor seat IDs that must be represented in strategy panel metrics.",
    )
    parser.add_argument(
        "--min-seat-samples",
        type=int,
        default=0,
        help="Minimum required sample count for each required actor seat.",
    )
    parser.add_argument(
        "--max-dominant-seat-fraction",
        type=float,
        default=1.0,
        help="Fail if one actor seat exceeds this fraction of selected strategy samples.",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def normalize_checkpoint_path(path: Path) -> Path:
    candidate = path
    if candidate.suffix.lower() == ".onnx":
        sibling = candidate.with_suffix(".pt")
        if sibling.exists():
            return sibling
    if not candidate.exists():
        raise FileNotFoundError(f"checkpoint not found: {candidate}")
    return candidate


def infer_model_dims(checkpoint_path: Path) -> tuple[int, int]:
    hidden_dim = 512
    bottleneck_dim = 256
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return hidden_dim, bottleneck_dim
    if isinstance(checkpoint, dict):
        hidden_dim = int(checkpoint.get("hidden_dim", hidden_dim))
        bottleneck_dim = int(checkpoint.get("bottleneck_dim", bottleneck_dim))
    return hidden_dim, bottleneck_dim


def subsample(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    action_masks: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = int(features.shape[0])
    if max_samples <= 0 or total <= max_samples:
        return features, targets, action_masks
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=max_samples, replace=False)
    indices.sort()
    return features[indices], targets[indices], action_masks[indices]


def evaluate_strategy(
    *,
    model: DeepCfrNet,
    features: np.ndarray,
    targets: np.ndarray,
    action_masks: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    model.eval()
    n = int(features.shape[0])
    if n == 0:
        raise ValueError("empty panel")

    probs_out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            x = torch.from_numpy(features[start:end]).to(device=device, dtype=torch.float32)
            mask = torch.from_numpy(action_masks[start:end]).to(device=device, dtype=torch.float32)
            probs = model(x, action_mask=mask, strategy_mode=True)
            probs_out.append(probs.detach().cpu().numpy())
    probs_np = np.concatenate(probs_out, axis=0)

    mask_np = (action_masks > 0).astype(np.float32)
    target_np = np.clip(targets, a_min=0.0, a_max=None).astype(np.float32) * mask_np
    target_mass = target_np.sum(axis=1, keepdims=True)
    uniform_target = mask_np / np.clip(mask_np.sum(axis=1, keepdims=True), a_min=1.0, a_max=None)
    normalized_target = target_np / np.clip(target_mass, a_min=1e-8, a_max=None)
    target_prob = np.where(target_mass > 1e-8, normalized_target, uniform_target).astype(np.float32)

    probs_clipped = np.clip(probs_np, a_min=1e-8, a_max=1.0)
    ce = -np.sum(target_prob * np.log(probs_clipped) * mask_np, axis=1)
    l1 = np.sum(np.abs(probs_np - target_prob) * mask_np, axis=1) / np.clip(
        mask_np.sum(axis=1), a_min=1.0, a_max=None
    )
    entropy = -np.sum(probs_clipped * np.log(probs_clipped) * mask_np, axis=1)

    pred_argmax = np.argmax(probs_np * mask_np, axis=1)
    target_argmax = np.argmax(target_prob * mask_np, axis=1)
    top1_match_rate = float(np.mean(pred_argmax == target_argmax))
    n_float = float(max(1, n))
    p = max(0.0, min(1.0, top1_match_rate))
    se = math.sqrt(max(0.0, p * (1.0 - p) / n_float))
    ci_delta = 1.96 * se
    ci_low = max(0.0, p - ci_delta)
    ci_high = min(1.0, p + ci_delta)

    actor_slice = features[:, ACTOR_POSITION_OFFSET : ACTOR_POSITION_OFFSET + POSITION_DIMS]
    actor_argmax = np.argmax(actor_slice, axis=1)
    actor_peak = np.max(actor_slice, axis=1)
    actor_idx = np.where(actor_peak > 0.5, actor_argmax, -1)
    seat_breakdown: dict[str, Any] = {}
    for seat in range(POSITION_DIMS):
        seat_mask = actor_idx == seat
        seat_count = int(np.sum(seat_mask))
        if seat_count <= 0:
            seat_breakdown[f"seat_{seat}"] = {
                "samples": 0,
                "top1_match_rate": None,
                "cross_entropy": None,
            }
            continue
        seat_top1 = float(np.mean((pred_argmax == target_argmax)[seat_mask]))
        seat_ce = float(np.mean(ce[seat_mask]))
        seat_breakdown[f"seat_{seat}"] = {
            "samples": seat_count,
            "top1_match_rate": seat_top1,
            "cross_entropy": seat_ce,
        }

    hist = np.bincount(pred_argmax, minlength=MAX_ACTIONS).astype(np.int64)
    mean_probs = np.mean(probs_np, axis=0)
    fingerprint_payload = {
        "sample_type": "strategy",
        "argmax_hist": hist.tolist(),
        "mean_probs": [round(float(v), 8) for v in mean_probs],
    }
    fingerprint_json = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    fingerprint_hash = hashlib.sha256(fingerprint_json.encode("utf-8")).hexdigest()

    metrics = {
        "cross_entropy": float(np.mean(ce)),
        "masked_l1": float(np.mean(l1)),
        "entropy": float(np.mean(entropy)),
        "top1_match_rate": top1_match_rate,
        "top1_match_rate_ci_low": float(ci_low),
        "top1_match_rate_ci_high": float(ci_high),
    }
    fingerprint = {
        "hash": fingerprint_hash,
        "argmax_hist": hist.tolist(),
        "mean_probs": [float(v) for v in mean_probs],
    }
    return metrics, fingerprint, seat_breakdown


def evaluate_advantage(
    *,
    model: DeepCfrNet,
    features: np.ndarray,
    targets: np.ndarray,
    action_masks: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    model.eval()
    n = int(features.shape[0])
    if n == 0:
        raise ValueError("empty panel")

    pred_out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            x = torch.from_numpy(features[start:end]).to(device=device, dtype=torch.float32)
            mask = torch.from_numpy(action_masks[start:end]).to(device=device, dtype=torch.float32)
            pred = model(x, action_mask=mask, strategy_mode=False)
            pred_out.append(pred.detach().cpu().numpy())
    pred_np = np.concatenate(pred_out, axis=0)

    mask_np = (action_masks > 0).astype(np.float32)
    error = (pred_np - targets) * mask_np
    mse = np.sum(error * error, axis=1) / np.clip(mask_np.sum(axis=1), a_min=1.0, a_max=None)
    abs_err = np.sum(np.abs(error), axis=1) / np.clip(mask_np.sum(axis=1), a_min=1.0, a_max=None)

    pred_sign = np.sign(pred_np) * mask_np
    target_sign = np.sign(targets) * mask_np
    sign_match = np.sum((pred_sign == target_sign) * mask_np, axis=1) / np.clip(
        mask_np.sum(axis=1), a_min=1.0, a_max=None
    )

    pred_argmax = np.argmax(pred_np * mask_np, axis=1)
    hist = np.bincount(pred_argmax, minlength=MAX_ACTIONS).astype(np.int64)
    mean_pred = np.mean(pred_np, axis=0)
    fingerprint_payload = {
        "sample_type": "advantage",
        "argmax_hist": hist.tolist(),
        "mean_pred": [round(float(v), 8) for v in mean_pred],
    }
    fingerprint_json = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    fingerprint_hash = hashlib.sha256(fingerprint_json.encode("utf-8")).hexdigest()

    metrics = {
        "mse": float(np.mean(mse)),
        "masked_abs_error": float(np.mean(abs_err)),
        "sign_match_rate": float(np.mean(sign_match)),
    }
    fingerprint = {
        "hash": fingerprint_hash,
        "argmax_hist": hist.tolist(),
        "mean_pred": [float(v) for v in mean_pred],
    }
    return metrics, fingerprint, {}


def evaluate_coverage_gate(
    *,
    seat_breakdown: dict[str, Any],
    required_seats: list[int],
    min_seat_samples: int,
    max_dominant_seat_fraction: float,
) -> dict[str, Any]:
    counts: dict[int, int] = {}
    total = 0
    for seat in range(POSITION_DIMS):
        info = seat_breakdown.get(f"seat_{seat}", {})
        samples = int(info.get("samples", 0)) if isinstance(info, dict) else 0
        counts[seat] = samples
        total += samples
    dominant_seat = -1
    dominant_count = 0
    if counts:
        dominant_seat, dominant_count = max(counts.items(), key=lambda kv: int(kv[1]))
    dominant_fraction = float(dominant_count / total) if total > 0 else 1.0
    required_missing = [seat for seat in required_seats if counts.get(seat, 0) == 0]
    below_min = [seat for seat in required_seats if counts.get(seat, 0) < min_seat_samples]
    reasons: list[str] = []
    if required_missing:
        reasons.append(f"required seat buckets empty: {required_missing}")
    if below_min:
        reasons.append(
            f"required seats below min samples ({min_seat_samples}): {below_min}"
        )
    if dominant_fraction > max_dominant_seat_fraction:
        reasons.append(
            "dominant seat fraction exceeds limit: "
            f"seat={dominant_seat} fraction={dominant_fraction:.4f} "
            f"limit={max_dominant_seat_fraction:.4f}"
        )
    passed = len(reasons) == 0
    return {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "required_seats": [int(v) for v in required_seats],
        "min_seat_samples": int(min_seat_samples),
        "max_dominant_seat_fraction": float(max_dominant_seat_fraction),
        "seat_counts": {str(k): int(v) for k, v in counts.items()},
        "total_samples": int(total),
        "dominant_seat": int(dominant_seat),
        "dominant_count": int(dominant_count),
        "dominant_fraction": float(dominant_fraction),
        "reasons": reasons,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_path = normalize_checkpoint_path(args.model_state.resolve())
    samples_path = args.samples.resolve()
    if not samples_path.exists():
        raise FileNotFoundError(f"samples panel not found: {samples_path}")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.min_seat_samples < 0:
        raise ValueError("--min-seat-samples must be >= 0")
    if not (0.0 <= float(args.max_dominant_seat_fraction) <= 1.0):
        raise ValueError("--max-dominant-seat-fraction must be in [0, 1]")
    required_seats = parse_csv_ints(args.required_seats)
    invalid_required = [seat for seat in required_seats if seat < 0 or seat >= POSITION_DIMS]
    if invalid_required:
        raise ValueError(f"--required-seats includes out-of-range seat(s): {invalid_required}")

    expected_magic = STRATEGY_SAMPLE_MAGIC if args.sample_type == "strategy" else ADVANTAGE_SAMPLE_MAGIC
    features, targets, action_masks, _ = load_binary_samples(samples_path, expected_magic=expected_magic)
    total_samples = int(features.shape[0])
    features, targets, action_masks = subsample(
        features=features,
        targets=targets,
        action_masks=action_masks,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    features = np.ascontiguousarray(features, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.float32)
    action_masks = np.ascontiguousarray(np.where(action_masks > 0, 1, 0).astype(np.uint8, copy=False))
    used_samples = int(features.shape[0])

    device = resolve_device(args.device)
    hidden_dim, bottleneck_dim = infer_model_dims(state_path)
    cfg = ModelConfig(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        max_actions=MAX_ACTIONS,
        dropout_p=0.10,
    )
    model = DeepCfrNet(cfg).to(device)
    loaded = load_model_weights(model, state_path, device)
    if not loaded:
        raise RuntimeError(f"failed to load checkpoint weights: {state_path}")

    if args.sample_type == "strategy":
        metrics, fingerprint, seat_breakdown = evaluate_strategy(
            model=model,
            features=features,
            targets=targets,
            action_masks=action_masks,
            batch_size=int(args.batch_size),
            device=device,
        )
        score = float(metrics["top1_match_rate"] * 100.0)
        score_ci_low = float(metrics.get("top1_match_rate_ci_low", metrics["top1_match_rate"]) * 100.0)
        score_ci_high = float(metrics.get("top1_match_rate_ci_high", metrics["top1_match_rate"]) * 100.0)
    else:
        metrics, fingerprint, seat_breakdown = evaluate_advantage(
            model=model,
            features=features,
            targets=targets,
            action_masks=action_masks,
            batch_size=int(args.batch_size),
            device=device,
        )
        score = float(metrics["sign_match_rate"] * 100.0)
        score_ci_low = score
        score_ci_high = score
    coverage = (
        evaluate_coverage_gate(
            seat_breakdown=seat_breakdown,
            required_seats=required_seats,
            min_seat_samples=int(args.min_seat_samples),
            max_dominant_seat_fraction=float(args.max_dominant_seat_fraction),
        )
        if args.sample_type == "strategy"
        else None
    )
    status = "ok"
    if coverage is not None and str(coverage.get("status")) != "ok":
        status = "failed"

    mtime = state_path.stat().st_mtime if state_path.exists() else 0.0
    result = {
        "status": status,
        "backend": "deep_cfr_panel",
        "label": str(args.label),
        "sample_type": str(args.sample_type),
        "sample_count_total": total_samples,
        "sample_count_used": used_samples,
        "seed": int(args.seed),
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "model": {
            "state_path": str(state_path),
            "hidden_dim": hidden_dim,
            "bottleneck_dim": bottleneck_dim,
            "mtime_unix_sec": float(mtime),
        },
        "panel": {
            "samples_path": str(samples_path),
        },
        "metrics": metrics,
        "score": score,
        "score_ci_low": score_ci_low,
        "score_ci_high": score_ci_high,
        "seat_breakdown": seat_breakdown,
        "coverage": coverage,
        "fingerprint": fingerprint,
    }
    return result


def main() -> int:
    args = parse_args()
    result = run(args)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=True))
    return 0 if str(result.get("status", "failed")) == "ok" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
