#!/usr/bin/env python3
"""Build final pre-long-run handoff manifest from gate artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from action_slots import MAX_ACTIONS


QUALITY_KNOBS = [
    "--hidden-dim",
    "--bottleneck-dim",
    "--traversals",
    "--training-steps",
    "--buffer-size",
    "--batch-size",
    "--deck-samples",
    "--iterations",
    "--max-sample-reuse-per-iter",
    "--dropout-p",
    "--lr",
    "--weight-decay",
    "--adv-huber-delta",
]

MACHINE_KNOBS = [
    "--traversal-workers",
    "--traversal-progress-batch",
    "--traversal-seat-chunks",
    "--device",
    "--evaluator-panel-device",
    "--evaluator-panel-batch-size",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pre-long-run handoff manifest.")
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--divergence-report", type=Path, required=True)
    parser.add_argument("--scale-search-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--machine-profile", type=Path, default=None)
    parser.add_argument("--hardware-probe", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--target-long-iterations", type=int, default=200)
    parser.add_argument("--max-local-hours", type=float, default=72.0)
    parser.add_argument("--max-local-disk-growth-gb", type=float, default=100.0)
    parser.add_argument("--ram-reservoir-max-fraction", type=float, default=0.45)
    parser.add_argument("--ram-reservoir-safety-factor", type=float, default=1.20)
    parser.add_argument("--ram-input-dim", type=int, default=510)
    parser.add_argument("--ram-max-actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--ram-reservoir-count", type=int, default=2)
    args = parser.parse_args()
    if args.target_long_iterations <= 0:
        raise ValueError("--target-long-iterations must be > 0")
    if args.max_local_hours <= 0:
        raise ValueError("--max-local-hours must be > 0")
    if args.max_local_disk_growth_gb <= 0:
        raise ValueError("--max-local-disk-growth-gb must be > 0")
    if not (0.05 <= float(args.ram_reservoir_max_fraction) <= 0.95):
        raise ValueError("--ram-reservoir-max-fraction must be in [0.05, 0.95]")
    if float(args.ram_reservoir_safety_factor) < 1.0:
        raise ValueError("--ram-reservoir-safety-factor must be >= 1.0")
    if args.ram_input_dim <= 0 or args.ram_max_actions <= 0:
        raise ValueError("--ram-input-dim and --ram-max-actions must be > 0")
    if args.ram_reservoir_count <= 0:
        raise ValueError("--ram-reservoir-count must be > 0")
    return args


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return payload


def estimate_runtime_and_disk(
    *,
    pilot_report: dict[str, Any],
    target_long_iterations: int,
) -> dict[str, Any]:
    checkpoints = pilot_report.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        return {"status": "missing"}
    first_metrics: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
    for item in checkpoints:
        if not isinstance(item, dict):
            continue
        training = item.get("training", {})
        if not isinstance(training, dict):
            continue
        latest = training.get("latest_metrics", {})
        if not isinstance(latest, dict) or not latest:
            continue
        if first_metrics is None:
            first_metrics = latest
        last_metrics = latest
    if first_metrics is None or last_metrics is None:
        return {"status": "missing"}

    times = last_metrics.get("times_sec", {}) if isinstance(last_metrics, dict) else {}
    artifact_first = first_metrics.get("artifact_bytes", {}) if isinstance(first_metrics, dict) else {}
    artifact_last = last_metrics.get("artifact_bytes", {}) if isinstance(last_metrics, dict) else {}
    iteration_total = float(times.get("iteration_total", 0.0)) if isinstance(times, dict) else 0.0
    first_work = int(artifact_first.get("work_dir", 0)) if isinstance(artifact_first, dict) else 0
    last_work = int(artifact_last.get("work_dir", 0)) if isinstance(artifact_last, dict) else 0
    first_iter = int(first_metrics.get("iteration", 1))
    last_iter = int(last_metrics.get("iteration", max(first_iter, 1)))
    growth_per_iter = float(last_work - first_work) / float(max(1, last_iter - first_iter))
    growth_per_iter = max(0.0, growth_per_iter)
    if iteration_total <= 0.0:
        return {"status": "missing"}
    eta_hours = float(iteration_total * target_long_iterations / 3600.0)
    projected_growth_gb = float(growth_per_iter * target_long_iterations / (1024.0**3))
    return {
        "status": "ok",
        "iteration_total_sec": float(iteration_total),
        "work_dir_growth_per_iter_bytes": float(growth_per_iter),
        "eta_hours_for_target_iterations": float(eta_hours),
        "projected_growth_gb_for_target_iterations": float(projected_growth_gb),
        "target_long_iterations": int(target_long_iterations),
    }


def pilot_contains_timeout(pilot_report: dict[str, Any]) -> bool:
    checkpoints = pilot_report.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return False
    for item in checkpoints:
        if not isinstance(item, dict):
            continue
        training = item.get("training", {})
        if not isinstance(training, dict):
            continue
        if bool(training.get("timed_out", False)):
            return True
    return False


def extract_session_disk_free_gb(hardware_probe: dict[str, Any]) -> float | None:
    session_disk = hardware_probe.get("session_disk", {})
    if isinstance(session_disk, dict):
        value = session_disk.get("free_gb")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def extract_gpu_count(hardware_probe: dict[str, Any]) -> int | None:
    nvidia = hardware_probe.get("nvidia", {})
    if isinstance(nvidia, dict):
        value = nvidia.get("gpu_count")
        if isinstance(value, (int, float)):
            return int(value)
    return None


def extract_total_ram_bytes(
    *,
    hardware_probe: dict[str, Any],
    machine_profile: dict[str, Any],
) -> int | None:
    for source in (
        machine_profile.get("hardware", {}) if isinstance(machine_profile, dict) else {},
        hardware_probe,
        hardware_probe.get("memory", {}) if isinstance(hardware_probe, dict) else {},
    ):
        if not isinstance(source, dict):
            continue
        value = source.get("total_ram_bytes")
        if isinstance(value, (int, float)) and float(value) > 0:
            return int(value)

    system_info_path = hardware_probe.get("system_info_json") if isinstance(hardware_probe, dict) else None
    if isinstance(system_info_path, str) and system_info_path:
        try:
            payload = read_json(Path(system_info_path))
            memory = payload.get("memory", {}) if isinstance(payload, dict) else {}
            total = memory.get("TotalPhysicalMemory") if isinstance(memory, dict) else None
            if isinstance(total, (int, float)) and float(total) > 0:
                return int(total)
        except Exception:
            return None
    return None


def estimate_reservoir_peak_bytes(
    *,
    chosen_config: dict[str, Any],
    input_dim: int,
    max_actions: int,
    reservoir_count: int,
    safety_factor: float,
) -> int | None:
    if not isinstance(chosen_config, dict):
        return None
    buffer_size = chosen_config.get("buffer_size")
    if not isinstance(buffer_size, (int, float)):
        return None
    if float(buffer_size) <= 0:
        return None
    bytes_per_entry = (int(input_dim) * 4) + (int(max_actions) * 4) + int(max_actions) + 8
    estimated = (
        float(buffer_size)
        * float(bytes_per_entry)
        * float(reservoir_count)
        * float(safety_factor)
    )
    return int(math.ceil(estimated))


def build_markdown(payload: dict[str, Any]) -> str:
    gates = payload.get("gates", {})
    chosen = payload.get("chosen_config", {})
    runtime = payload.get("runtime_projection", {})
    cloud = payload.get("cloud_handoff", {})
    lines = [
        "# Pre-Long-Run Handoff Manifest",
        "",
        "## Gate Summary",
        f"- panel_gate: `{gates.get('pass_panel_gate')}`",
        f"- divergence_gate: `{gates.get('pass_divergence_gate')}`",
        f"- scale_gate: `{gates.get('pass_scale_gate')}`",
        f"- pilot_gate: `{gates.get('pass_pilot_gate')}`",
        f"- runtime_gate: `{gates.get('pass_runtime_gate')}`",
        f"- ram_gate: `{payload.get('runtime_capacity', {}).get('pass_ram_gate')}`",
        f"- ready_for_long_run_decision: `{gates.get('ready_for_long_run_decision')}`",
        "",
        "## Chosen Quality Config",
        f"- name: `{chosen.get('name')}`",
        f"- hidden_dim: `{chosen.get('hidden_dim')}`",
        f"- bottleneck_dim: `{chosen.get('bottleneck_dim')}`",
        f"- traversals: `{chosen.get('traversals')}`",
        f"- training_steps: `{chosen.get('training_steps')}`",
        f"- buffer_size: `{chosen.get('buffer_size')}`",
        f"- batch_size: `{chosen.get('batch_size')}`",
        f"- deck_samples: `{chosen.get('deck_samples')}`",
        "",
        "## Runtime Projection",
        f"- iteration_total_sec: `{runtime.get('iteration_total_sec')}`",
        f"- eta_hours_for_target_iterations: `{runtime.get('eta_hours_for_target_iterations')}`",
        f"- projected_growth_gb_for_target_iterations: `{runtime.get('projected_growth_gb_for_target_iterations')}`",
        "",
        "## Cloud Handoff",
        f"- requires_cloud: `{cloud.get('requires_cloud')}`",
        f"- reason: `{cloud.get('reason')}`",
        "",
        "## Knob Split",
        "- quality_knobs: " + ", ".join(f"`{k}`" for k in payload.get("knob_split", {}).get("quality_knobs", [])),
        "- machine_knobs: " + ", ".join(f"`{k}`" for k in payload.get("knob_split", {}).get("machine_knobs", [])),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    panel_manifest = read_json(args.panel_manifest.resolve())
    divergence = read_json(args.divergence_report.resolve())
    scale_search = read_json(args.scale_search_report.resolve())
    benchmark = read_json(args.benchmark_report.resolve())
    pilot = read_json(args.pilot_report.resolve())
    machine_profile = (
        read_json(args.machine_profile.resolve())
        if args.machine_profile is not None and args.machine_profile.exists()
        else {}
    )
    hardware_probe = (
        read_json(args.hardware_probe.resolve())
        if args.hardware_probe is not None and args.hardware_probe.exists()
        else {}
    )

    screen = panel_manifest.get("panels", {}).get("screen", {}) if isinstance(panel_manifest.get("panels"), dict) else {}
    holdout = panel_manifest.get("panels", {}).get("holdout", {}) if isinstance(panel_manifest.get("panels"), dict) else {}
    pass_panel_gate = (
        str(panel_manifest.get("status", "failed")) == "ok"
        and str(screen.get("status", "failed")) == "ok"
        and str(holdout.get("status", "failed")) == "ok"
    )
    pass_divergence_gate = bool(divergence.get("gate_passed", False))
    recommended = scale_search.get("recommended_candidate", {})
    chosen_config = recommended.get("candidate", {}) if isinstance(recommended, dict) else {}
    pass_scale_gate = str(scale_search.get("status", "failed")) == "ok" and isinstance(chosen_config, dict) and bool(
        chosen_config
    )
    pass_pilot_gate = bool(pilot.get("gates", {}).get("pass_pilot_gate", False))
    pilot_timed_out = pilot_contains_timeout(pilot)
    runtime_projection = estimate_runtime_and_disk(
        pilot_report=pilot,
        target_long_iterations=int(args.target_long_iterations),
    )
    session_disk_free_gb = extract_session_disk_free_gb(hardware_probe)
    projected_growth_gb = (
        float(runtime_projection.get("projected_growth_gb_for_target_iterations", 0.0))
        if str(runtime_projection.get("status")) == "ok"
        else 0.0
    )
    disk_headroom_gb = None
    pass_disk_gate = True
    if session_disk_free_gb is not None and str(runtime_projection.get("status")) == "ok":
        disk_headroom_gb = float(session_disk_free_gb - projected_growth_gb)
        pass_disk_gate = disk_headroom_gb >= 10.0

    machine_knobs = machine_profile.get("machine_knobs", {}) if isinstance(machine_profile, dict) else {}
    requested_device = str(machine_knobs.get("device", "auto")) if isinstance(machine_knobs, dict) else "auto"
    gpu_count = extract_gpu_count(hardware_probe)
    pass_cuda_gate = True
    if requested_device == "cuda" and gpu_count is not None and gpu_count <= 0:
        pass_cuda_gate = False

    total_ram_bytes = extract_total_ram_bytes(hardware_probe=hardware_probe, machine_profile=machine_profile)
    estimated_reservoir_peak_bytes = estimate_reservoir_peak_bytes(
        chosen_config=chosen_config,
        input_dim=int(args.ram_input_dim),
        max_actions=int(args.ram_max_actions),
        reservoir_count=int(args.ram_reservoir_count),
        safety_factor=float(args.ram_reservoir_safety_factor),
    )
    allowed_reservoir_bytes = (
        int(float(total_ram_bytes) * float(args.ram_reservoir_max_fraction))
        if total_ram_bytes is not None
        else None
    )
    pass_ram_gate = True
    reservoir_headroom_bytes = None
    if total_ram_bytes is not None and estimated_reservoir_peak_bytes is not None and allowed_reservoir_bytes is not None:
        pass_ram_gate = estimated_reservoir_peak_bytes <= allowed_reservoir_bytes
        reservoir_headroom_bytes = int(allowed_reservoir_bytes - estimated_reservoir_peak_bytes)

    pass_runtime_gate = (
        str(benchmark.get("status", "failed")) == "ok"
        and str(runtime_projection.get("status", "missing")) == "ok"
        and pass_disk_gate
        and pass_cuda_gate
        and pass_ram_gate
    )

    requires_cloud = False
    cloud_reason = "local runtime projection within thresholds"
    if pilot_timed_out:
        requires_cloud = True
        cloud_reason = "pilot checkpoint timed out on current workstation for chosen quality config"
    elif not pass_cuda_gate:
        requires_cloud = True
        cloud_reason = "machine profile requests cuda but hardware probe found zero NVIDIA GPUs"
    elif not pass_ram_gate and total_ram_bytes is not None and estimated_reservoir_peak_bytes is not None:
        requires_cloud = True
        cloud_reason = (
            "chosen config reservoir footprint exceeds RAM gate: "
            f"estimated={estimated_reservoir_peak_bytes / (1024.0**3):.2f}GB "
            f"allowed={float(args.ram_reservoir_max_fraction) * (float(total_ram_bytes) / (1024.0**3)):.2f}GB"
        )
    elif not pass_disk_gate and session_disk_free_gb is not None and disk_headroom_gb is not None:
        requires_cloud = True
        cloud_reason = (
            "insufficient disk headroom on target machine: "
            f"free={session_disk_free_gb:.1f}GB projected_growth={projected_growth_gb:.1f}GB "
            f"headroom={disk_headroom_gb:.1f}GB (minimum required 10.0GB)"
        )
    elif pass_runtime_gate:
        eta_hours = float(runtime_projection.get("eta_hours_for_target_iterations", 0.0))
        if eta_hours > float(args.max_local_hours):
            requires_cloud = True
            cloud_reason = (
                f"projected local ETA {eta_hours:.1f}h exceeds threshold {float(args.max_local_hours):.1f}h"
            )
        elif projected_growth_gb > float(args.max_local_disk_growth_gb):
            requires_cloud = True
            cloud_reason = (
                "projected local disk growth "
                f"{projected_growth_gb:.1f}GB exceeds threshold {float(args.max_local_disk_growth_gb):.1f}GB"
            )
    ready = all(
        [
            pass_panel_gate,
            pass_divergence_gate,
            pass_scale_gate,
            pass_pilot_gate,
            pass_runtime_gate,
        ]
    )
    payload = {
        "status": "ok" if ready else "failed",
        "gates": {
            "pass_panel_gate": pass_panel_gate,
            "pass_divergence_gate": pass_divergence_gate,
            "pass_scale_gate": pass_scale_gate,
            "pass_pilot_gate": pass_pilot_gate,
            "pilot_timed_out": pilot_timed_out,
            "pass_runtime_gate": pass_runtime_gate,
            "ready_for_long_run_decision": ready,
        },
        "artifacts": {
            "panel_manifest": str(args.panel_manifest.resolve()),
            "divergence_report": str(args.divergence_report.resolve()),
            "scale_search_report": str(args.scale_search_report.resolve()),
            "benchmark_report": str(args.benchmark_report.resolve()),
            "pilot_report": str(args.pilot_report.resolve()),
        },
        "chosen_config": chosen_config,
        "runtime_projection": runtime_projection,
        "runtime_capacity": {
            "session_disk_free_gb": session_disk_free_gb,
            "projected_growth_gb": projected_growth_gb,
            "disk_headroom_gb": disk_headroom_gb,
            "pass_disk_gate": pass_disk_gate,
            "requested_device": requested_device,
            "gpu_count": gpu_count,
            "pass_cuda_gate": pass_cuda_gate,
            "total_ram_bytes": total_ram_bytes,
            "total_ram_gb": (float(total_ram_bytes) / (1024.0**3)) if total_ram_bytes is not None else None,
            "estimated_reservoir_peak_bytes": estimated_reservoir_peak_bytes,
            "estimated_reservoir_peak_gb": (
                float(estimated_reservoir_peak_bytes) / (1024.0**3)
                if estimated_reservoir_peak_bytes is not None
                else None
            ),
            "allowed_reservoir_bytes": allowed_reservoir_bytes,
            "allowed_reservoir_gb": (
                float(allowed_reservoir_bytes) / (1024.0**3)
                if allowed_reservoir_bytes is not None
                else None
            ),
            "reservoir_ram_headroom_bytes": reservoir_headroom_bytes,
            "reservoir_ram_headroom_gb": (
                float(reservoir_headroom_bytes) / (1024.0**3)
                if reservoir_headroom_bytes is not None
                else None
            ),
            "pass_ram_gate": pass_ram_gate,
            "reservoir_gate_policy": {
                "max_fraction": float(args.ram_reservoir_max_fraction),
                "safety_factor": float(args.ram_reservoir_safety_factor),
                "input_dim": int(args.ram_input_dim),
                "max_actions": int(args.ram_max_actions),
                "reservoir_count": int(args.ram_reservoir_count),
            },
        },
        "cloud_handoff": {
            "max_local_hours": float(args.max_local_hours),
            "max_local_disk_growth_gb": float(args.max_local_disk_growth_gb),
            "requires_cloud": bool(requires_cloud),
            "reason": cloud_reason,
        },
        "machine_profile": machine_profile if machine_profile else benchmark.get("best_by_wall_throughput"),
        "hardware_probe": hardware_probe,
        "checkpoint_cadence": pilot.get("config", {}).get("checkpoint_every"),
        "evaluation_cadence": pilot.get("config", {}).get("checkpoint_every"),
        "knob_split": {
            "quality_knobs": QUALITY_KNOBS,
            "machine_knobs": MACHINE_KNOBS,
        },
    }
    args.output_json = args.output_json.resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md is not None:
        args.output_md = args.output_md.resolve()
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if ready else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
