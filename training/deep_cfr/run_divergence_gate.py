#!/usr/bin/env python3
"""Checkpoint divergence gate for Deep CFR panel evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_EVAL_SCRIPT = REPO_ROOT / "eval" / "deep_cfr_panel.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if any compared checkpoints produce identical panel fingerprints."
    )
    parser.add_argument("--panel", type=Path, required=True, help="Binary fixed panel sample file.")
    parser.add_argument(
        "--sample-type",
        choices=["strategy", "advantage"],
        default="strategy",
        help="Sample type expected by panel file.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec label=path. Repeat for each checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in specs:
        token = str(raw).strip()
        if not token:
            continue
        if "=" in token:
            label, path_raw = token.split("=", 1)
            model_label = label.strip()
            model_path = Path(path_raw.strip())
        else:
            model_path = Path(token)
            model_label = model_path.stem
        if not model_label:
            raise ValueError(f"invalid --model spec (missing label): {raw!r}")
        out.append((model_label, model_path))
    return out


def parse_last_json_line(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("panel evaluator did not return JSON")


def run_panel_eval(
    *,
    label: str,
    model_path: Path,
    panel_path: Path,
    sample_type: str,
    max_samples: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PANEL_EVAL_SCRIPT),
        "--model-state",
        str(model_path),
        "--samples",
        str(panel_path),
        "--sample-type",
        sample_type,
        "--max-samples",
        str(max_samples),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--device",
        device,
        "--label",
        label,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"panel eval failed for {label}: {tail}")
    payload = parse_last_json_line(proc.stdout)
    if str(payload.get("status", "error")) != "ok":
        raise RuntimeError(f"panel eval returned non-ok status for {label}: {payload}")
    return payload


def main() -> int:
    args = parse_args()
    panel_path = args.panel.resolve()
    if not panel_path.exists():
        raise FileNotFoundError(f"panel file not found: {panel_path}")
    if not PANEL_EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"missing panel evaluator script: {PANEL_EVAL_SCRIPT}")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    model_specs = parse_model_specs(list(args.model))
    if len(model_specs) < 3:
        raise ValueError("provide at least 3 --model entries (random/init, short-trained, larger-trained)")

    model_reports: list[dict[str, Any]] = []
    for label, model_path in model_specs:
        report = run_panel_eval(
            label=label,
            model_path=model_path.resolve(),
            panel_path=panel_path,
            sample_type=str(args.sample_type),
            max_samples=int(args.max_samples),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            device=str(args.device),
        )
        model_reports.append(
            {
                "label": label,
                "model_state": str(model_path.resolve()),
                "score": float(report.get("score", 0.0)),
                "metrics": report.get("metrics", {}),
                "fingerprint_hash": str(report.get("fingerprint", {}).get("hash", "")),
                "fingerprint": report.get("fingerprint", {}),
                "panel_eval": report,
            }
        )

    pairwise: list[dict[str, Any]] = []
    failures: list[str] = []
    for i in range(len(model_reports)):
        for j in range(i + 1, len(model_reports)):
            left = model_reports[i]
            right = model_reports[j]
            same_hash = str(left.get("fingerprint_hash", "")) == str(right.get("fingerprint_hash", ""))
            pairwise.append(
                {
                    "left": str(left["label"]),
                    "right": str(right["label"]),
                    "same_fingerprint": bool(same_hash),
                    "left_hash": str(left.get("fingerprint_hash", "")),
                    "right_hash": str(right.get("fingerprint_hash", "")),
                    "left_score": float(left.get("score", 0.0)),
                    "right_score": float(right.get("score", 0.0)),
                }
            )
            if same_hash:
                failures.append(
                    f"identical fingerprint: {left['label']} == {right['label']} ({left.get('fingerprint_hash', '')})"
                )

    gate_passed = len(failures) == 0
    payload = {
        "status": "ok" if gate_passed else "failed",
        "gate_passed": gate_passed,
        "panel": str(panel_path),
        "sample_type": str(args.sample_type),
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "models": model_reports,
        "pairwise": pairwise,
        "failures": failures,
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    print(json.dumps(payload, ensure_ascii=True))
    if not gate_passed:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
