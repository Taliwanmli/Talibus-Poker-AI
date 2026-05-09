from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

try:
    from colorama import Fore, Style, init as colorama_init
except Exception:  # pragma: no cover - optional styling dependency
    Fore = None  # type: ignore[assignment]
    Style = None  # type: ignore[assignment]
    colorama_init = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import DeepCfrNet, INPUT_DIM, MAX_ACTIONS, ModelConfig
from reservoir import DiskBackedReservoirBuffer, ReservoirBuffer, SampleBatch
from subprocess_env import build_subprocess_env
from train import (
    ADVANTAGE_SAMPLE_MAGIC,
    STRATEGY_SAMPLE_MAGIC,
    configure_warning_filters,
    export_onnx,
    load_binary_samples,
    load_model_weights,
    resolve_device,
    train_advantage_step,
    train_strategy_step,
)

SAMPLE_MAGIC = ADVANTAGE_SAMPLE_MAGIC
SAMPLE_VERSION = 2
HEADER_STRUCT = struct.Struct("<4sIII")
FLOAT_RE = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
LOSS_RE = re.compile(
    rf"step=(\d+)/(\d+)\s+loss=({FLOAT_RE})"
)
EXPLOITABILITY_RE = re.compile(
    rf"result=({FLOAT_RE}) bb/hand "
    rf"\(({FLOAT_RE}) mBB/hand\)"
)
H2H_MODEL_A_RE = re.compile(
    rf"\[h2h\]\s+model_a:\s+({FLOAT_RE}) bb/hand \(({FLOAT_RE}) bb/100\)"
)
RING_EVAL_JSON_PREFIX = "RING_EVAL_JSON "
TRAVERSE_WORKER_RE = re.compile(
    rf"^\[deep-cfr-traverse\] worker=(\d+)\s+jobs=(\d+)\s+traversals=(\d+)\s+"
    rf"load=({FLOAT_RE})s\s+run=({FLOAT_RE})s\s+adv_samples=(\d+)\s+strategy_samples=(\d+)\s*$"
)
TRAVERSE_PAR_SUMMARY_RE = re.compile(
    rf"^\[deep-cfr-traverse\] parallel summary:\s+job_chunk=(\d+)\s+"
    rf"worker_load_total=({FLOAT_RE})s\s+worker_run_total=({FLOAT_RE})s\s+"
    rf"traversals_min=(\d+)\s+traversals_max=(\d+)\s+imbalance=({FLOAT_RE})\s*$"
)
DEFAULT_BUFFER_SAVE_EVERY = 10
DEFAULT_SELECTION_HOLDOUT_SEEDS = 3
DEBUG_LOG_PATH = (SCRIPT_DIR.parent.parent / "debug-1270ea.log").resolve()


# #region agent log
def debug_log(hypothesis_id: str, message: str, data: dict[str, Any], run_id: str = "debug-run") -> None:
    payload = {
        "sessionId": "1270ea",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "training/deep_cfr/run_deep_cfr.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        # Keep debug logging best-effort and never block training.
        pass
# #endregion

@dataclass
class NetworkTrainingContext:
    label: str
    network_type: str
    model: DeepCfrNet
    optimizer: torch.optim.Optimizer
    scaler: torch.cuda.amp.GradScaler
    reservoir: ReservoirBuffer | DiskBackedReservoirBuffer
    state_path: Path
    buffer_path: Path
    onnx_path: Path
    use_amp: bool


@dataclass
class TraversalTask:
    player: int
    process: subprocess.Popen[str]
    advantage_sample_path: Path
    strategy_sample_path: Path
    log_path: Path
    started_at: float
    log_handle: TextIO


@dataclass
class IterationTraversalBundle:
    iteration: int
    model_copy_paths: list[Path]
    tasks: list[TraversalTask]
    consolidated: bool = False
    consolidated_players: list[int] = field(default_factory=list)
    consolidated_adv_paths: dict[int, Path] = field(default_factory=dict)
    consolidated_strategy_paths: dict[int, Path] = field(default_factory=dict)


@dataclass
class TrainingRunStats:
    final_loss: float | None
    steps_ran: int
    steps_target: int
    early_stopped: bool


@dataclass
class HeadToHeadReport:
    bb_per_hand: float
    bb_per_100: float


@dataclass(frozen=True)
class EvalPanelConfig:
    exploit_fit_samples: int
    exploit_eval_samples: int
    exploit_strategy_fit_seed: int
    exploit_strategy_eval_seed: int
    exploit_adv_fit_seed: int
    exploit_adv_eval_seed: int
    h2h_strategy_tag_seed: int
    h2h_strategy_random_seed: int
    h2h_strategy_tag_holdout_seeds: tuple[int, ...]


def _is_color_enabled() -> bool:
    return Fore is not None and Style is not None


def color_bold(text: str) -> str:
    if not _is_color_enabled():
        return text
    return f"{Style.BRIGHT}{text}{Style.RESET_ALL}"


def color_green(text: str) -> str:
    if not _is_color_enabled():
        return text
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"


def color_red(text: str) -> str:
    if not _is_color_enabled():
        return text
    return f"{Fore.RED}{text}{Style.RESET_ALL}"


def color_yellow(text: str) -> str:
    if not _is_color_enabled():
        return text
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"


def color_cyan(text: str) -> str:
    if not _is_color_enabled():
        return text
    return f"{Fore.CYAN}{text}{Style.RESET_ALL}"


def log(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, flush=True)


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    total = int(max(0.0, round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}h{minutes:02}m{secs:02}s"
    if minutes > 0:
        return f"{minutes}m{secs:02}s"
    return f"{secs}s"


def format_bytes(byte_count: int) -> str:
    size = float(max(0, byte_count))
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(size)}{units[unit_idx]}"
    return f"{size:.1f}{units[unit_idx]}"


def format_signed(value: float, decimals: int = 3) -> str:
    return f"{value:+.{decimals}f}"


def format_change_pct(current: float, previous: float | None) -> str:
    if previous is None:
        return "n/a"
    denom = abs(previous)
    if denom < 1e-9:
        return "n/a"
    return f"{((current - previous) / denom) * 100.0:+.1f}%"


def maybe_color_delta(delta: float | None, lower_is_better: bool = True, decimals: int = 6) -> str:
    if delta is None:
        return "n/a"
    text = f"{delta:+.{decimals}f}"
    if abs(delta) < 1e-12:
        return color_yellow(text)
    improved = delta < 0.0 if lower_is_better else delta > 0.0
    return color_green(text) if improved else color_red(text)


def print_iter_header(iteration: int, total: int, eta_seconds: float) -> None:
    iter_label = f"{iteration:03d}/{total}"
    headline = f"=== Iter {iter_label} | ETA ~{format_duration(eta_seconds)} ==="
    log(color_bold(headline))


def print_iter_summary(
    *,
    iter_total: float,
    traverse_time: float,
    train_time: float,
    train_steps_ran: int,
    training_steps_target: int,
    early_stopped: bool,
    advantage_loss: float | None,
    loss_delta: float | None,
    reservoir_estimate: int,
    buffer_size: int,
    work_disk: int,
    samples_disk: int,
    elapsed: float,
    eta: float,
) -> None:
    early_text = f" (early stop @ {train_steps_ran})" if early_stopped else ""
    loss_text = "n/a" if advantage_loss is None else f"{advantage_loss:.6f}"
    loss_delta_text = maybe_color_delta(loss_delta, lower_is_better=True, decimals=6)
    reservoir_fill = (100.0 * reservoir_estimate / buffer_size) if buffer_size > 0 else 0.0
    log(
        color_cyan(
            f"  total={iter_total:.1f}s | traverse={traverse_time:.1f}s | "
            f"train={train_time:.1f}s ({train_steps_ran}/{training_steps_target}){early_text}"
        )
    )
    log(f"  loss={loss_text} (delta={loss_delta_text})")
    log(
        f"  reservoir~{reservoir_estimate:,}/{buffer_size:,} "
        f"({reservoir_fill:.1f}%) | disk work={format_bytes(work_disk)} samples={format_bytes(samples_disk)}"
    )
    log(f"  elapsed={format_duration(elapsed)} | ETA~{format_duration(eta)}")


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def command_to_string(command: list[str]) -> str:
    return " ".join(command)


def safe_delete(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        log(f"[warn] failed to delete {path}: {exc}")


def safe_delete_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        log(f"[warn] failed to delete directory {path}: {exc}")


def safe_copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        log(f"[warn] missing file for copy: {src}")
        return False
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except OSError as exc:
        log(f"[warn] failed to copy {src} -> {dst}: {exc}")
        return False


def get_metric_value(report: dict[str, Any] | None, key: str, field: str) -> float | None:
    if not isinstance(report, dict):
        return None
    metric_obj = report.get(key)
    if not isinstance(metric_obj, dict):
        return None
    value = metric_obj.get(field)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def reset_work_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            safe_delete_tree(child)
        else:
            safe_delete(child)


def bootstrap_split_advantage_artifacts(
    *,
    legacy_onnx: Path,
    legacy_state: Path,
    legacy_buffer: Path,
    advantage_p0_onnx: Path,
    advantage_p1_onnx: Path,
    advantage_p0_state: Path,
    advantage_p1_state: Path,
    advantage_p0_buffer: Path,
    advantage_p1_buffer: Path,
) -> None:
    if legacy_onnx.exists():
        if not advantage_p0_onnx.exists():
            shutil.copy2(legacy_onnx, advantage_p0_onnx)
            log(f"[init] migrated legacy advantage model -> {advantage_p0_onnx.name}")
        if not advantage_p1_onnx.exists():
            shutil.copy2(legacy_onnx, advantage_p1_onnx)
            log(f"[init] migrated legacy advantage model -> {advantage_p1_onnx.name}")

    if legacy_state.exists():
        if not advantage_p0_state.exists():
            shutil.copy2(legacy_state, advantage_p0_state)
            log(f"[init] migrated legacy advantage checkpoint -> {advantage_p0_state.name}")
        if not advantage_p1_state.exists():
            shutil.copy2(legacy_state, advantage_p1_state)
            log(f"[init] migrated legacy advantage checkpoint -> {advantage_p1_state.name}")

    if legacy_buffer.exists():
        if not advantage_p0_buffer.exists():
            shutil.copy2(legacy_buffer, advantage_p0_buffer)
            log(f"[init] migrated legacy advantage buffer -> {advantage_p0_buffer.name}")
        if not advantage_p1_buffer.exists():
            shutil.copy2(legacy_buffer, advantage_p1_buffer)
            log(f"[init] migrated legacy advantage buffer -> {advantage_p1_buffer.name}")


def estimate_reservoir_size(iterations: list[Any], capacity: int) -> int:
    estimate = 0
    for item in iterations:
        if not isinstance(item, dict):
            continue
        samples = item.get("samples")
        if not isinstance(samples, dict):
            continue
        merged = samples.get("merged", 0)
        try:
            merged_count = int(merged)
        except (TypeError, ValueError):
            merged_count = 0
        estimate = min(capacity, estimate + max(0, merged_count))
    return estimate


def estimate_player_reservoir_sizes(iterations: list[Any], capacity: int) -> tuple[int, int]:
    estimate_p0 = 0
    estimate_p1 = 0
    for item in iterations:
        if not isinstance(item, dict):
            continue
        samples = item.get("samples")
        if not isinstance(samples, dict):
            continue
        raw_p0 = samples.get("player0_adv", samples.get("player0", 0))
        raw_p1 = samples.get("player1_adv", samples.get("player1", 0))
        try:
            p0_count = int(raw_p0)
        except (TypeError, ValueError):
            p0_count = 0
        try:
            p1_count = int(raw_p1)
        except (TypeError, ValueError):
            p1_count = 0
        estimate_p0 = min(capacity, estimate_p0 + max(0, p0_count))
        estimate_p1 = min(capacity, estimate_p1 + max(0, p1_count))
    return estimate_p0, estimate_p1


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        return default
    return loaded


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")

def run_streaming_command(
    command: list[str],
    cwd: Path,
    capture_loss: bool = False,
) -> float | None:
    log(f"[run] {command_to_string(command)}")
    # Force UTF-8 subprocess I/O so ONNX exporter logs never crash on codepage-limited consoles.
    env = build_subprocess_env()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        encoding="utf-8",
        errors="replace",
    )

    last_loss: float | None = None
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if line:
                log(line)
                if capture_loss:
                    match = LOSS_RE.search(line)
                    if match is not None:
                        last_loss = float(match.group(3))
            else:
                print("", flush=True)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {command_to_string(command)}"
        )
    return last_loss


def run_exploitability_measurement(
    command: list[str],
    cwd: Path,
) -> tuple[float, float]:
    log(f"[run] {command_to_string(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_subprocess_env(),
        encoding="utf-8",
        errors="replace",
    )

    bb_per_hand: float | None = None
    mbb_per_hand: float | None = None
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if line:
                log(line)
                match = EXPLOITABILITY_RE.search(line)
                if match is not None:
                    bb_per_hand = float(match.group(1))
                    mbb_per_hand = float(match.group(2))
            else:
                print("", flush=True)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {command_to_string(command)}"
        )
    if bb_per_hand is None or mbb_per_hand is None:
        raise RuntimeError("measure_exploitability output did not contain a parsed result line")
    return bb_per_hand, mbb_per_hand


def read_sample_metadata(path: Path, expected_magic: bytes = SAMPLE_MAGIC) -> tuple[int, bytes]:
    file_size = path.stat().st_size
    if file_size < HEADER_STRUCT.size:
        raise ValueError(f"sample file too small: {path}")
    with path.open("rb") as source:
        header = source.read(HEADER_STRUCT.size)
    if len(header) < HEADER_STRUCT.size:
        raise ValueError(f"sample file too small: {path}")
    magic, version, input_dim, max_actions = HEADER_STRUCT.unpack(header)
    if magic != expected_magic:
        raise ValueError(
            f"invalid sample magic in {path}: {magic!r}, expected {expected_magic!r}"
        )
    if version == 0 or version > SAMPLE_VERSION:
        raise ValueError(f"unsupported sample version in {path}: {version}")
    if version == 1:
        record_size = input_dim * 4 + max_actions * 4 + 1 + 4
    else:
        record_size = input_dim * 4 + max_actions * 4 + max_actions + 4
    payload_size = file_size - HEADER_STRUCT.size
    if payload_size % record_size != 0:
        raise ValueError(
            f"invalid payload size in {path}: {payload_size} is not divisible by {record_size}"
        )
    sample_count = payload_size // record_size
    return sample_count, header


def merge_sample_files(
    sample_paths: list[Path],
    output_path: Path,
    *,
    expected_magic: bytes = SAMPLE_MAGIC,
) -> int:
    if not sample_paths:
        raise ValueError("no sample files provided for merge")

    expected_header: bytes | None = None
    total_samples = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as destination:
        for path in sample_paths:
            file_size = path.stat().st_size
            if file_size < HEADER_STRUCT.size:
                raise ValueError(f"sample file too small: {path}")

            with path.open("rb") as source:
                header = source.read(HEADER_STRUCT.size)
                if len(header) < HEADER_STRUCT.size:
                    raise ValueError(f"sample file too small: {path}")
                magic, version, input_dim, max_actions = HEADER_STRUCT.unpack(header)
                if version == 0 or version > SAMPLE_VERSION:
                    raise ValueError(f"unsupported sample version in {path}: {version}")
                if version == 1:
                    record_size = input_dim * 4 + max_actions * 4 + 1 + 4
                else:
                    record_size = input_dim * 4 + max_actions * 4 + max_actions + 4
                payload_size = file_size - HEADER_STRUCT.size
                if payload_size % record_size != 0:
                    raise ValueError(
                        f"invalid payload size in {path}: {payload_size} is not divisible by {record_size}"
                    )

                if expected_header is None:
                    expected_header = header
                    if magic != expected_magic:
                        raise ValueError(f"invalid sample header in {path}")
                    destination.write(expected_header)
                elif header != expected_header:
                    raise ValueError(f"sample header mismatch while merging: {path}")

                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

                total_samples += payload_size // record_size

    assert expected_header is not None
    return total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end Deep CFR (traversal + training) with real-time logs."
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--traversals", type=int, default=20000)
    parser.add_argument("--training-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.10,
        help="Dropout probability used in DeepCfrNet hidden layers.",
    )
    parser.add_argument(
        "--max-sample-reuse-per-iter",
        type=float,
        default=6.0,
        help=(
            "Upper bound on expected per-iteration sample reuse. "
            "Effective steps are capped to keep overfitting in check."
        ),
    )
    parser.add_argument(
        "--adv-huber-delta",
        type=float,
        default=20.0,
        help="Huber-delta for advantage regression loss (robust to noisy targets).",
    )
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument(
        "--buffer-disk-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for disk-backed reservoir buffers (np.memmap). "
            "When provided, sample storage is moved from RAM to disk."
        ),
    )
    parser.add_argument("--cluster-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--rust-binary", type=Path, default=None)
    parser.add_argument("--deck-samples", type=int, default=800)
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden width for Deep CFR networks (shared across advantage/strategy).",
    )
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=256,
        help="Bottleneck width for Deep CFR networks (shared across advantage/strategy).",
    )
    parser.add_argument("--starting-stack", type=int, default=2000)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--big-blind", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=600)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument(
        "--enable-early-stop",
        action="store_true",
        help="Enable loss-plateau early stopping during network training.",
    )
    parser.add_argument(
        "--early-stop-min-steps",
        type=int,
        default=150,
        help="Minimum training steps before early stopping can trigger.",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=0.001,
        help="Minimum relative loss improvement required over the rolling window.",
    )
    parser.add_argument(
        "--strategy-every",
        type=int,
        default=0,
        help="Train/export strategy model every N iterations (0 disables).",
    )
    parser.add_argument(
        "--exploitability-every",
        type=int,
        default=0,
        help="Run exploitability measurement every N iterations (0 disables).",
    )
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=25,
        help="Run diagnostic suite every N iterations (0 disables).",
    )
    parser.add_argument(
        "--h2h-only-diagnostics",
        action="store_true",
        help="Skip exploitability during diagnostics and run only TAG/random head-to-head.",
    )
    parser.add_argument(
        "--h2h-hands",
        type=int,
        default=10_000,
        help="Hands per head-to-head diagnostic matchup.",
    )
    parser.add_argument(
        "--exploit-samples",
        type=int,
        default=1_000,
        help="Sample count used by exploitability diagnostics (used for both root/deck samples).",
    )
    parser.add_argument(
        "--exploit-fit-samples",
        type=int,
        default=0,
        help="Optional BR-fitting panel size override (0 means use --exploit-samples).",
    )
    parser.add_argument(
        "--exploit-eval-samples",
        type=int,
        default=0,
        help="Optional BR-evaluation panel size override (0 means use --exploit-samples).",
    )
    parser.add_argument(
        "--fixed-eval-seed",
        type=int,
        default=0,
        help="Optional fixed seed for proof-oriented evaluation panels (0 derives from --seed).",
    )
    parser.add_argument(
        "--traversal-progress-batch",
        type=int,
        default=500,
        help="Traversal chunk size passed to run_traversals for progress logging.",
    )
    parser.add_argument(
        "--traversal-workers",
        type=int,
        default=0,
        help="Traversal worker threads per run_traversals process (0=auto).",
    )
    parser.add_argument(
        "--traversal-seat-chunks",
        type=int,
        default=1,
        help=(
            "Number of independent traversal chunks launched per seat each iteration. "
            "Values >1 reduce seat-level straggler idle time at the cost of extra process overhead."
        ),
    )
    parser.add_argument(
        "--traversal-consolidate-processes",
        action="store_true",
        help=(
            "Launch one run_traversals process for all seats using --player-list "
            "(only applied when --traversal-seat-chunks=1)."
        ),
    )
    parser.add_argument(
        "--traversal-gpu-batch",
        action="store_true",
        help="Enable batched inference coordinator in run_traversals.",
    )
    parser.add_argument(
        "--traversal-gpu-batch-size",
        type=int,
        default=256,
        help="Maximum batched inference requests per GPU run_traversals call.",
    )
    parser.add_argument(
        "--traversal-gpu-batch-timeout-us",
        type=int,
        default=500,
        help="Maximum microseconds to wait for batch coalescing in run_traversals.",
    )
    parser.add_argument(
        "--traversal-gpu-batch-queue-capacity",
        type=int,
        default=8192,
        help="Queue capacity for batched inference request channel.",
    )
    parser.add_argument(
        "--traversal-gpu-batch-cpu-only",
        action="store_true",
        help="Enable batched mode but force CPU execution provider (debug fallback).",
    )
    parser.add_argument(
        "--traversal-gpu-batch-no-tf32",
        action="store_true",
        help="Disable CUDA TF32 acceleration in batched inference mode.",
    )
    parser.add_argument(
        "--traversal-gpu-device-id",
        type=int,
        default=0,
        help="CUDA device id passed to run_traversals in batched mode.",
    )
    parser.add_argument(
        "--traversal-fp16-onnx",
        action="store_true",
        help="Export and use fp16-optimized ONNX models for traversal inference.",
    )
    parser.add_argument(
        "--traversal-cpu-affinity",
        type=str,
        default="",
        help="Optional CPU affinity string for run_traversals (example: 0-15).",
    )
    parser.add_argument(
        "--buffer-save-every",
        type=int,
        default=DEFAULT_BUFFER_SAVE_EVERY,
        help="Persist reservoir checkpoints every N completed iterations.",
    )
    parser.add_argument(
        "--selection-holdout-seeds",
        type=int,
        default=DEFAULT_SELECTION_HOLDOUT_SEEDS,
        help="Holdout TAG seed count for best-model confirmation.",
    )
    parser.add_argument(
        "--selection-holdout-hands",
        type=int,
        default=0,
        help="Hands per holdout TAG check (0 uses --h2h-hands).",
    )
    parser.add_argument(
        "--selection-random-guardrail",
        type=float,
        default=0.0,
        help="Minimum bb/100 vs random required before candidate promotion.",
    )
    parser.add_argument(
        "--league-eval-every",
        type=int,
        default=0,
        help="Run eval/run_league.py diagnostics every N iterations for multiplayer runs (0 disables).",
    )
    parser.add_argument(
        "--league-hands",
        type=int,
        default=2_000,
        help="Hands per multiplayer league diagnostic run.",
    )
    parser.add_argument(
        "--league-opponents",
        type=str,
        default="mixed",
        help="Opponent profile list passed to eval/run_league.py --opponents.",
    )
    parser.add_argument(
        "--evaluator-backend",
        choices=["league", "deep_cfr_panel"],
        default="league",
        help="Multiplayer diagnostic backend: league (legacy) or deep_cfr_panel (checkpoint-aware).",
    )
    parser.add_argument(
        "--evaluator-model",
        choices=["auto", "strategy", "advantage"],
        default="auto",
        help="Checkpoint family used by deep_cfr_panel backend.",
    )
    parser.add_argument(
        "--evaluator-panel-file",
        type=Path,
        default=None,
        help="Optional fixed binary panel file for deep_cfr_panel backend.",
    )
    parser.add_argument(
        "--evaluator-panel-type",
        choices=["strategy", "advantage"],
        default="strategy",
        help="Sample type expected by evaluator panel file.",
    )
    parser.add_argument(
        "--evaluator-panel-max-samples",
        type=int,
        default=25_000,
        help="Maximum panel samples evaluated per diagnostic run.",
    )
    parser.add_argument(
        "--evaluator-panel-batch-size",
        type=int,
        default=4096,
        help="Panel evaluator batch size.",
    )
    parser.add_argument(
        "--evaluator-panel-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Panel evaluator device request.",
    )
    parser.add_argument(
        "--evaluator-required-seats",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated actor seat IDs required by deep_cfr_panel coverage gate.",
    )
    parser.add_argument(
        "--evaluator-min-seat-samples",
        type=int,
        default=1,
        help="Minimum required samples for each required actor seat in deep_cfr_panel coverage gate.",
    )
    parser.add_argument(
        "--evaluator-max-dominant-seat-fraction",
        type=float,
        default=0.70,
        help="Maximum allowed actor-seat dominance fraction in deep_cfr_panel coverage gate.",
    )
    parser.add_argument(
        "--checkpoint-eval-every",
        type=int,
        default=0,
        help="Run fixed-panel checkpoint evaluation every N iterations in shared-multiseat mode (0 disables).",
    )
    parser.add_argument(
        "--checkpoint-eval-screen-panel-file",
        type=Path,
        default=None,
        help="Binary panel file used for low-cost periodic screen checks.",
    )
    parser.add_argument(
        "--checkpoint-eval-holdout-panel-file",
        type=Path,
        default=None,
        help="Binary panel file used for holdout confirmation when a screen challenger improves.",
    )
    parser.add_argument(
        "--checkpoint-eval-top1-improvement",
        type=float,
        default=0.0,
        help="Minimum top1 improvement (absolute, 0..1 scale) required before holdout confirmation is triggered.",
    )
    parser.add_argument(
        "--checkpoint-eval-require-coverage-pass",
        action="store_true",
        help="Require panel coverage gate pass for screen/holdout checkpoint promotion.",
    )
    parser.add_argument(
        "--ring-eval-every",
        type=int,
        default=0,
        help="Run 6-max ring-game strength evaluation every N iterations (0 disables).",
    )
    parser.add_argument(
        "--ring-eval-hands",
        type=int,
        default=1_000,
        help="Hands per ring-game strength evaluation run.",
    )
    parser.add_argument(
        "--model-checkpoint-every",
        type=int,
        default=10,
        help="Save full model snapshots every N iterations (0 disables).",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    if args.work_dir is None:
        work_dir = repo_root / "data" / "deep_cfr"
    else:
        work_dir = args.work_dir
    work_dir = work_dir.resolve()

    if args.cluster_dir is None:
        cluster_dir = repo_root / "checkpoints" / "nlhe_clusters"
    else:
        cluster_dir = args.cluster_dir
    cluster_dir = cluster_dir.resolve()

    if args.rust_binary is None:
        binary_name = "run_traversals.exe" if sys.platform.startswith("win") else "run_traversals"
        rust_binary = repo_root / "solver" / "target" / "release" / binary_name
    else:
        rust_binary = args.rust_binary
    rust_binary = rust_binary.resolve()

    measure_binary_name = (
        "measure_exploitability.exe" if sys.platform.startswith("win") else "measure_exploitability"
    )
    measure_binary = repo_root / "solver" / "target" / "release" / measure_binary_name
    measure_binary = measure_binary.resolve()

    h2h_binary_name = "head_to_head.exe" if sys.platform.startswith("win") else "head_to_head"
    h2h_binary = repo_root / "solver" / "target" / "release" / h2h_binary_name
    h2h_binary = h2h_binary.resolve()
    ring_eval_binary_name = (
        "ring_game_eval.exe" if sys.platform.startswith("win") else "ring_game_eval"
    )
    ring_eval_binary = repo_root / "solver" / "target" / "release" / ring_eval_binary_name
    ring_eval_binary = ring_eval_binary.resolve()

    return {
        "script_dir": script_dir,
        "repo_root": repo_root,
        "work_dir": work_dir,
        "cluster_dir": cluster_dir,
        "rust_binary": rust_binary,
        "measure_binary": measure_binary,
        "h2h_binary": h2h_binary,
        "ring_eval_binary": ring_eval_binary,
    }


def parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = token.strip()
        if not text:
            continue
        out.append(int(text))
    return out


def validate_args(args: argparse.Namespace) -> None:
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.traversals <= 0:
        raise ValueError("--traversals must be > 0")
    if args.training_steps <= 0:
        raise ValueError("--training-steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not (0.0 <= args.dropout_p < 1.0):
        raise ValueError("--dropout-p must be in [0, 1)")
    if args.max_sample_reuse_per_iter <= 0.0:
        raise ValueError("--max-sample-reuse-per-iter must be > 0")
    if args.adv_huber_delta <= 0.0:
        raise ValueError("--adv-huber-delta must be > 0")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be > 0")
    if args.buffer_disk_dir is not None:
        disk_dir = Path(args.buffer_disk_dir).expanduser()
        if not disk_dir.is_absolute():
            disk_dir = (Path.cwd() / disk_dir).resolve()
        args.buffer_disk_dir = disk_dir
    if args.deck_samples <= 0:
        raise ValueError("--deck-samples must be > 0")
    if args.num_players < 2 or args.num_players > 6:
        raise ValueError("--num-players must be in [2, 6]")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be > 0")
    if args.bottleneck_dim <= 0:
        raise ValueError("--bottleneck-dim must be > 0")
    if args.small_blind <= 0 or args.big_blind <= 0:
        raise ValueError("--small-blind and --big-blind must be > 0")
    if args.starting_stack <= 0:
        raise ValueError("--starting-stack must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if args.traversal_progress_batch <= 0:
        raise ValueError("--traversal-progress-batch must be > 0")
    if args.traversal_workers < 0:
        raise ValueError("--traversal-workers must be >= 0")
    if args.traversal_seat_chunks <= 0:
        raise ValueError("--traversal-seat-chunks must be > 0")
    if args.traversal_gpu_batch_size <= 0:
        raise ValueError("--traversal-gpu-batch-size must be > 0")
    if args.traversal_gpu_batch_timeout_us <= 0:
        raise ValueError("--traversal-gpu-batch-timeout-us must be > 0")
    if args.traversal_gpu_batch_queue_capacity <= 0:
        raise ValueError("--traversal-gpu-batch-queue-capacity must be > 0")
    if args.traversal_gpu_device_id < 0:
        raise ValueError("--traversal-gpu-device-id must be >= 0")
    if args.strategy_every < 0:
        raise ValueError("--strategy-every must be >= 0")
    if args.exploitability_every < 0:
        raise ValueError("--exploitability-every must be >= 0")
    if args.diagnostic_every < 0:
        raise ValueError("--diagnostic-every must be >= 0")
    if args.h2h_hands <= 0:
        raise ValueError("--h2h-hands must be > 0")
    if args.exploit_samples <= 0:
        raise ValueError("--exploit-samples must be > 0")
    if args.exploit_fit_samples < 0:
        raise ValueError("--exploit-fit-samples must be >= 0")
    if args.exploit_eval_samples < 0:
        raise ValueError("--exploit-eval-samples must be >= 0")
    if args.fixed_eval_seed < 0:
        raise ValueError("--fixed-eval-seed must be >= 0")
    if args.buffer_save_every <= 0:
        raise ValueError("--buffer-save-every must be > 0")
    if args.selection_holdout_seeds < 0:
        raise ValueError("--selection-holdout-seeds must be >= 0")
    if args.selection_holdout_hands < 0:
        raise ValueError("--selection-holdout-hands must be >= 0")
    if args.league_eval_every < 0:
        raise ValueError("--league-eval-every must be >= 0")
    if args.league_hands <= 0:
        raise ValueError("--league-hands must be > 0")
    if args.ring_eval_every < 0:
        raise ValueError("--ring-eval-every must be >= 0")
    if args.ring_eval_hands <= 0:
        raise ValueError("--ring-eval-hands must be > 0")
    if args.model_checkpoint_every < 0:
        raise ValueError("--model-checkpoint-every must be >= 0")
    if args.evaluator_panel_max_samples < 0:
        raise ValueError("--evaluator-panel-max-samples must be >= 0")
    if args.evaluator_panel_batch_size <= 0:
        raise ValueError("--evaluator-panel-batch-size must be > 0")
    if args.evaluator_min_seat_samples < 0:
        raise ValueError("--evaluator-min-seat-samples must be >= 0")
    if not (0.0 <= args.evaluator_max_dominant_seat_fraction <= 1.0):
        raise ValueError("--evaluator-max-dominant-seat-fraction must be in [0, 1]")
    if args.checkpoint_eval_every < 0:
        raise ValueError("--checkpoint-eval-every must be >= 0")
    if args.checkpoint_eval_top1_improvement < 0.0:
        raise ValueError("--checkpoint-eval-top1-improvement must be >= 0")
    required_seats = parse_csv_ints(args.evaluator_required_seats)
    bad_required = [seat for seat in required_seats if seat < 0 or seat >= args.num_players]
    if bad_required:
        raise ValueError(f"--evaluator-required-seats includes out-of-range seat(s): {bad_required}")
    args.evaluator_required_seat_ids = tuple(int(v) for v in required_seats)
    if args.checkpoint_eval_screen_panel_file is not None:
        args.checkpoint_eval_screen_panel_file = args.checkpoint_eval_screen_panel_file.resolve()
    if args.checkpoint_eval_holdout_panel_file is not None:
        args.checkpoint_eval_holdout_panel_file = args.checkpoint_eval_holdout_panel_file.resolve()
    if args.checkpoint_eval_every > 0 and args.checkpoint_eval_screen_panel_file is None:
        if args.evaluator_panel_file is not None:
            args.checkpoint_eval_screen_panel_file = args.evaluator_panel_file.resolve()
        else:
            raise ValueError(
                "--checkpoint-eval-every requires --checkpoint-eval-screen-panel-file "
                "(or --evaluator-panel-file as fallback)."
            )
    if (
        args.checkpoint_eval_screen_panel_file is not None
        and not args.checkpoint_eval_screen_panel_file.exists()
    ):
        raise FileNotFoundError(
            f"checkpoint screen panel file not found: {args.checkpoint_eval_screen_panel_file}"
        )
    if (
        args.checkpoint_eval_holdout_panel_file is not None
        and not args.checkpoint_eval_holdout_panel_file.exists()
    ):
        raise FileNotFoundError(
            f"checkpoint holdout panel file not found: {args.checkpoint_eval_holdout_panel_file}"
        )
    if args.early_stop_min_steps <= 0:
        raise ValueError("--early-stop-min-steps must be > 0")
    if args.early_stop_threshold < 0:
        raise ValueError("--early-stop-threshold must be >= 0")


def train_init_model(
    train_script: Path,
    script_dir: Path,
    advantage_onnx: Path,
    advantage_state: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        str(train_script),
        "--model-out",
        str(advantage_onnx),
        "--state-path",
        str(advantage_state),
        "--type",
        "advantage",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--onnx-opset",
        str(args.onnx_opset),
        "--dropout-p",
        str(args.dropout_p),
        "--hidden-dim",
        str(args.hidden_dim),
        "--bottleneck-dim",
        str(args.bottleneck_dim),
        "--init-only",
    ]
    if bool(getattr(args, "traversal_fp16_onnx", False)):
        command.append("--export-fp16-onnx")
    run_streaming_command(command, cwd=script_dir)


def build_traversal_command(
    rust_binary: Path,
    onnx_p0_path: Path,
    onnx_p1_path: Path,
    advantage_samples_out: Path,
    strategy_samples_out: Path,
    player: int,
    iteration: int,
    args: argparse.Namespace,
    cluster_dir: Path,
    seed: int,
) -> list[str]:
    command = [
        str(rust_binary),
        "--onnx-p0",
        str(onnx_p0_path),
        "--onnx-p1",
        str(onnx_p1_path),
        "--adv-samples-out",
        str(advantage_samples_out),
        "--strategy-samples-out",
        str(strategy_samples_out),
        "--player",
        str(player),
        "--traversals",
        str(args.traversals),
        "--progress-batch",
        str(args.traversal_progress_batch),
        "--workers",
        str(args.traversal_workers),
        "--iteration",
        str(iteration),
        "--seed",
        str(seed),
        "--cluster-dir",
        str(cluster_dir),
        "--deck-samples",
        str(args.deck_samples),
        "--num-players",
        str(args.num_players),
        "--starting-stack",
        str(args.starting_stack),
        "--small-blind",
        str(args.small_blind),
        "--big-blind",
        str(args.big_blind),
    ]
    if bool(getattr(args, "traversal_gpu_batch", False)):
        command.extend(
            [
                "--gpu-batch",
                "--gpu-batch-size",
                str(getattr(args, "traversal_gpu_batch_size", 256)),
                "--gpu-batch-timeout-us",
                str(getattr(args, "traversal_gpu_batch_timeout_us", 500)),
                "--gpu-batch-queue-capacity",
                str(getattr(args, "traversal_gpu_batch_queue_capacity", 8192)),
                "--gpu-device-id",
                str(getattr(args, "traversal_gpu_device_id", 0)),
            ]
        )
        if bool(getattr(args, "traversal_gpu_batch_cpu_only", False)):
            command.append("--gpu-batch-cpu-only")
        if bool(getattr(args, "traversal_gpu_batch_no_tf32", False)):
            command.append("--gpu-batch-no-tf32")
    cpu_affinity = str(getattr(args, "traversal_cpu_affinity", "")).strip()
    if cpu_affinity:
        command.extend(["--cpu-affinity", cpu_affinity])
    return command


def run_traversal_for_player(
    rust_binary: Path,
    repo_root: Path,
    onnx_p0_path: Path,
    onnx_p1_path: Path,
    advantage_samples_out: Path,
    strategy_samples_out: Path,
    player: int,
    iteration: int,
    args: argparse.Namespace,
    cluster_dir: Path,
    seed: int,
) -> None:
    command = build_traversal_command(
        rust_binary=rust_binary,
        onnx_p0_path=onnx_p0_path,
        onnx_p1_path=onnx_p1_path,
        advantage_samples_out=advantage_samples_out,
        strategy_samples_out=strategy_samples_out,
        player=player,
        iteration=iteration,
        args=args,
        cluster_dir=cluster_dir,
        seed=seed,
    )
    run_streaming_command(command, cwd=repo_root)


def start_traversal_async(
    *,
    rust_binary: Path,
    repo_root: Path,
    onnx_p0_path: Path,
    onnx_p1_path: Path,
    advantage_samples_out: Path,
    strategy_samples_out: Path,
    player: int,
    iteration: int,
    args: argparse.Namespace,
    cluster_dir: Path,
    seed: int,
    log_path: Path,
) -> TraversalTask:
    command = build_traversal_command(
        rust_binary=rust_binary,
        onnx_p0_path=onnx_p0_path,
        onnx_p1_path=onnx_p1_path,
        advantage_samples_out=advantage_samples_out,
        strategy_samples_out=strategy_samples_out,
        player=player,
        iteration=iteration,
        args=args,
        cluster_dir=cluster_dir,
        seed=seed,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=output_handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_subprocess_env(),
        encoding="utf-8",
        errors="replace",
    )
    return TraversalTask(
        player=player,
        process=process,
        advantage_sample_path=advantage_samples_out,
        strategy_sample_path=strategy_samples_out,
        log_path=log_path,
        started_at=time.perf_counter(),
        log_handle=output_handle,
    )


def start_iteration_traversals_async(
    *,
    iteration: int,
    rust_binary: Path,
    repo_root: Path,
    onnx_p0_path: Path,
    onnx_p1_path: Path,
    models_dir: Path,
    samples_dir: Path,
    traversal_logs_dir: Path,
    args: argparse.Namespace,
    cluster_dir: Path,
) -> IterationTraversalBundle:
    model_copy_p0 = models_dir / f"advantage_p0_bg_iter{iteration:03d}.onnx"
    model_copy_p1 = models_dir / f"advantage_p1_bg_iter{iteration:03d}.onnx"
    shutil.copy2(onnx_p0_path, model_copy_p0)
    shutil.copy2(onnx_p1_path, model_copy_p1)
    seed_p0 = args.seed ^ (iteration * 0x9E37_79B9) ^ 0xA24B_AED4
    seed_p1 = args.seed ^ (iteration * 0xD1B5_4A32) ^ 0x94D0_49BB
    sample_p0_adv = samples_dir / f"samples_adv_p0_iter{iteration:03d}.bin"
    sample_p1_adv = samples_dir / f"samples_adv_p1_iter{iteration:03d}.bin"
    sample_p0_strategy = samples_dir / f"samples_strategy_from_p0_iter{iteration:03d}.bin"
    sample_p1_strategy = samples_dir / f"samples_strategy_from_p1_iter{iteration:03d}.bin"
    log_p0 = traversal_logs_dir / f"traverse_p0_iter{iteration:03d}.log"
    log_p1 = traversal_logs_dir / f"traverse_p1_iter{iteration:03d}.log"
    task_p0 = start_traversal_async(
        rust_binary=rust_binary,
        repo_root=repo_root,
        onnx_p0_path=model_copy_p0,
        onnx_p1_path=model_copy_p1,
        advantage_samples_out=sample_p0_adv,
        strategy_samples_out=sample_p0_strategy,
        player=0,
        iteration=iteration,
        args=args,
        cluster_dir=cluster_dir,
        seed=seed_p0,
        log_path=log_p0,
    )
    task_p1 = start_traversal_async(
        rust_binary=rust_binary,
        repo_root=repo_root,
        onnx_p0_path=model_copy_p0,
        onnx_p1_path=model_copy_p1,
        advantage_samples_out=sample_p1_adv,
        strategy_samples_out=sample_p1_strategy,
        player=1,
        iteration=iteration,
        args=args,
        cluster_dir=cluster_dir,
        seed=seed_p1,
        log_path=log_p1,
    )
    return IterationTraversalBundle(
        iteration=iteration,
        model_copy_paths=[model_copy_p0, model_copy_p1],
        tasks=[task_p0, task_p1],
    )


def read_text_tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    return "\n".join(lines[-max_lines:])


def parse_parallel_traversal_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    workers: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for line in content.splitlines():
        worker_match = TRAVERSE_WORKER_RE.match(line.strip())
        if worker_match:
            workers.append(
                {
                    "worker": int(worker_match.group(1)),
                    "jobs": int(worker_match.group(2)),
                    "traversals": int(worker_match.group(3)),
                    "load_sec": float(worker_match.group(4)),
                    "run_sec": float(worker_match.group(5)),
                    "adv_samples": int(worker_match.group(6)),
                    "strategy_samples": int(worker_match.group(7)),
                }
            )
            continue
        summary_match = TRAVERSE_PAR_SUMMARY_RE.match(line.strip())
        if summary_match:
            summary = {
                "job_chunk": int(summary_match.group(1)),
                "worker_load_total_sec": float(summary_match.group(2)),
                "worker_run_total_sec": float(summary_match.group(3)),
                "worker_traversals_min": int(summary_match.group(4)),
                "worker_traversals_max": int(summary_match.group(5)),
                "worker_traversal_imbalance": float(summary_match.group(6)),
            }
    if not workers and not summary:
        return {}
    workers.sort(key=lambda item: int(item.get("worker", 0)))
    if workers and not summary:
        traversals = [int(item.get("traversals", 0)) for item in workers]
        load_total = sum(float(item.get("load_sec", 0.0)) for item in workers)
        run_total = sum(float(item.get("run_sec", 0.0)) for item in workers)
        min_traversals = min(traversals) if traversals else 0
        max_traversals = max(traversals) if traversals else 0
        imbalance = max_traversals / min_traversals if min_traversals > 0 else 0.0
        summary = {
            "job_chunk": None,
            "worker_load_total_sec": load_total,
            "worker_run_total_sec": run_total,
            "worker_traversals_min": min_traversals,
            "worker_traversals_max": max_traversals,
            "worker_traversal_imbalance": imbalance,
        }
    return {"workers": workers, "summary": summary}


def wait_for_iteration_traversals(bundle: IterationTraversalBundle) -> dict[str, float | int | Path]:
    results: dict[str, float | int | Path] = {}
    for task in bundle.tasks:
        delete_log = False
        try:
            code = task.process.wait()
            elapsed = time.perf_counter() - task.started_at
            if code != 0:
                tail = read_text_tail(task.log_path)
                raise RuntimeError(
                    f"traversal failed for player {task.player} in iter {bundle.iteration} with exit code {code}\n"
                    f"log: {task.log_path}\n{tail}"
                )
            adv_sample_count, _ = read_sample_metadata(
                task.advantage_sample_path,
                expected_magic=ADVANTAGE_SAMPLE_MAGIC,
            )
            strategy_sample_count, _ = read_sample_metadata(
                task.strategy_sample_path,
                expected_magic=STRATEGY_SAMPLE_MAGIC,
            )
            if task.player == 0:
                results["sample_adv_p0"] = task.advantage_sample_path
                results["sample_strategy_from_p0"] = task.strategy_sample_path
                results["p0_adv_samples"] = adv_sample_count
                results["p0_strategy_samples"] = strategy_sample_count
                results["p0_time"] = elapsed
            else:
                results["sample_adv_p1"] = task.advantage_sample_path
                results["sample_strategy_from_p1"] = task.strategy_sample_path
                results["p1_adv_samples"] = adv_sample_count
                results["p1_strategy_samples"] = strategy_sample_count
                results["p1_time"] = elapsed
            delete_log = True
        finally:
            task.log_handle.close()
        if delete_log:
            safe_delete(task.log_path)
    for model_copy_path in bundle.model_copy_paths:
        safe_delete(model_copy_path)
    results["iteration"] = bundle.iteration
    p0 = float(results.get("p0_time", 0.0))
    p1 = float(results.get("p1_time", 0.0))
    results["traverse_total"] = p0 + p1
    return results


def terminate_traversal_bundle(bundle: IterationTraversalBundle | None) -> None:
    if bundle is None:
        return
    for task in bundle.tasks:
        try:
            if task.process.poll() is None:
                task.process.terminate()
                task.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            task.process.kill()
            task.process.wait(timeout=5)
        finally:
            try:
                task.log_handle.close()
            except Exception:
                pass
    for model_copy_path in bundle.model_copy_paths:
        safe_delete(model_copy_path)


def migrate_pickle_reservoir_to_disk(
    *,
    source_path: Path,
    destination: DiskBackedReservoirBuffer,
    default_max_size: int,
) -> int:
    legacy = ReservoirBuffer.load(source_path, default_max_size=default_max_size)
    if legacy.is_empty():
        return 0

    total = len(legacy)
    chunk_size = 50_000
    rng = np.random.default_rng(0)
    log(
        f"[deep-cfr] migrating legacy reservoir from {source_path} -> {destination.disk_dir} "
        f"({total:,} samples)"
    )
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        indices = np.arange(start, stop, dtype=np.int64)
        batch = legacy.sample_from_indices(indices)
        destination.add_many(batch.features, batch.targets, batch.action_masks, batch.iterations, rng)
    destination.seen = max(destination.seen, legacy.seen)
    destination.save()
    log(
        f"[deep-cfr] migration complete: copied={total:,}, seen={destination.seen:,}, "
        f"size={len(destination):,}/{destination.max_size:,}"
    )
    return total


def initialize_training_context(
    label: str,
    network_type: str,
    state_path: Path,
    buffer_path: Path,
    onnx_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> NetworkTrainingContext:
    cfg = ModelConfig(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        max_actions=MAX_ACTIONS,
        dropout_p=args.dropout_p,
    )
    model = DeepCfrNet(cfg).to(device)
    loaded = load_model_weights(model, state_path, device)
    log(
        f"[deep-cfr] {label} model init="
        f"{'checkpoint' if loaded else 'random'} ({state_path})"
    )

    checkpoint_buffer_path = buffer_path
    if args.buffer_disk_dir is not None:
        disk_buffer_dir = Path(args.buffer_disk_dir) / label
        reservoir = DiskBackedReservoirBuffer.load(
            disk_buffer_dir,
            default_max_size=args.buffer_size,
        )
        if reservoir.is_empty() and buffer_path.exists():
            migrated = migrate_pickle_reservoir_to_disk(
                source_path=buffer_path,
                destination=reservoir,
                default_max_size=args.buffer_size,
            )
            if migrated > 0:
                log(f"[deep-cfr] {label} migrated legacy buffer into {disk_buffer_dir}")
        checkpoint_buffer_path = disk_buffer_dir / DiskBackedReservoirBuffer.METADATA_FILE
        log(f"[deep-cfr] {label} using disk-backed reservoir dir={disk_buffer_dir}")
    else:
        reservoir = ReservoirBuffer.load(buffer_path, default_max_size=args.buffer_size)
    log(f"[deep-cfr] {label} reservoir size={len(reservoir)} / {args.buffer_size}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_cuda = device.type == "cuda"
    use_amp = use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    return NetworkTrainingContext(
        label=label,
        network_type=network_type,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        reservoir=reservoir,
        state_path=state_path,
        buffer_path=checkpoint_buffer_path,
        onnx_path=onnx_path,
        use_amp=use_amp,
    )


def save_model_checkpoint(context: NetworkTrainingContext) -> None:
    context.state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": context.model.state_dict(),
            "network_type": context.network_type,
            "input_dim": INPUT_DIM,
            "max_actions": MAX_ACTIONS,
            "hidden_dim": int(context.model.cfg.hidden_dim),
            "bottleneck_dim": int(context.model.cfg.bottleneck_dim),
        },
        context.state_path,
    )
    log(f"[deep-cfr] saved state={context.state_path}")


def save_reservoir_checkpoint(context: NetworkTrainingContext) -> None:
    context.reservoir.save(context.buffer_path)
    log(f"[deep-cfr] saved buffer={context.buffer_path}")


def ingest_samples_into_context(
    *,
    context: NetworkTrainingContext,
    samples_path: Path,
    seed: int,
) -> int:
    expected_magic = (
        ADVANTAGE_SAMPLE_MAGIC if context.network_type == "advantage" else STRATEGY_SAMPLE_MAGIC
    )
    features, targets, action_masks, iterations = load_binary_samples(
        samples_path,
        expected_magic=expected_magic,
    )
    rng = np.random.default_rng(seed)
    context.reservoir.add_many(features, targets, action_masks, iterations, rng)
    log(
        f"[deep-cfr] {context.label}: ingested {len(features)} samples from {samples_path} "
        f"(reservoir={len(context.reservoir)}/{context.reservoir.max_size})"
    )
    return len(features)


def train_network_in_process(
    context: NetworkTrainingContext,
    samples_path: Path,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    ingest_samples: bool = True,
) -> TrainingRunStats:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    features: np.ndarray = np.empty((0, INPUT_DIM), dtype=np.float32)
    targets: np.ndarray = np.empty((0, MAX_ACTIONS), dtype=np.float32)
    action_masks: np.ndarray = np.empty((0, MAX_ACTIONS), dtype=np.uint8)
    iterations: np.ndarray = np.empty((0,), dtype=np.int64)
    if ingest_samples:
        expected_magic = (
            ADVANTAGE_SAMPLE_MAGIC if context.network_type == "advantage" else STRATEGY_SAMPLE_MAGIC
        )
        features, targets, action_masks, iterations = load_binary_samples(
            samples_path,
            expected_magic=expected_magic,
        )
        log(
            f"[deep-cfr] loaded {len(features)} samples from {samples_path} "
            f"(input={INPUT_DIM}, actions={MAX_ACTIONS})"
        )
        context.reservoir.add_many(features, targets, action_masks, iterations, rng)
    log(f"[deep-cfr] reservoir size={len(context.reservoir)} / {args.buffer_size}")
    # #region agent log
    valid_counts = (
        np.asarray(action_masks, dtype=np.int64).sum(axis=1)
        if action_masks.size > 0
        else np.empty((0,), dtype=np.int64)
    )
    valid_hist = np.bincount(valid_counts, minlength=MAX_ACTIONS + 1)
    debug_log(
        "H5",
        "batch_input_profile",
        {
            "network_type": context.network_type,
            "samples_path": str(samples_path),
            "sample_count": int(len(features)),
            "target_abs_mean": float(np.abs(targets).mean()) if targets.size > 0 else 0.0,
            "target_std": float(targets.std()) if targets.size > 0 else 0.0,
            "all_non_positive_target_frac": float((targets.max(axis=1) <= 0.0).mean()) if targets.size > 0 else 0.0,
            "valid_actions_hist": {str(i): int(valid_hist[i]) for i in range(len(valid_hist)) if int(valid_hist[i]) > 0},
            "iteration_min": int(np.min(iterations)) if len(iterations) > 0 else -1,
            "iteration_max": int(np.max(iterations)) if len(iterations) > 0 else -1,
            "iteration_mean": float(np.mean(iterations)) if len(iterations) > 0 else -1.0,
        },
    )
    # #endregion

    if len(context.reservoir) == 0:
        raise ValueError("reservoir is empty after loading samples")

    batch_size = min(args.batch_size, len(context.reservoir))
    max_steps_by_reuse = max(
        1,
        int(math.ceil((len(context.reservoir) * args.max_sample_reuse_per_iter) / batch_size)),
    )
    effective_training_steps = min(args.training_steps, max_steps_by_reuse)
    if effective_training_steps < args.training_steps:
        log(
            "[deep-cfr] capped training steps to "
            f"{effective_training_steps}/{args.training_steps} "
            f"(max reuse {args.max_sample_reuse_per_iter:.2f}x)"
        )
    recommended_training_steps = max(
        1,
        int(math.ceil((len(context.reservoir) * 2.0) / batch_size)),
    )
    if effective_training_steps < recommended_training_steps:
        log(
            "[deep-cfr] WARNING: effective training steps may be too low for reservoir size: "
            f"{effective_training_steps}/{recommended_training_steps} "
            f"(reservoir={len(context.reservoir):,}, batch={batch_size:,})"
        )
    # #region agent log
    debug_log(
        "H3",
        "training_step_budget",
        {
            "network_type": context.network_type,
            "reservoir_size": int(len(context.reservoir)),
            "batch_size": int(batch_size),
            "requested_steps": int(args.training_steps),
            "effective_steps": int(effective_training_steps),
            "max_sample_reuse_per_iter": float(args.max_sample_reuse_per_iter),
        },
    )
    # #endregion
    batch_indices = rng.integers(
        0,
        len(context.reservoir),
        size=(effective_training_steps, batch_size),
        dtype=np.int64,
    )
    log(
        "[deep-cfr] pre-sampled "
        f"{effective_training_steps} batches (batch_size={batch_size})"
    )

    use_cuda = device.type == "cuda"
    context.model.train()
    final_loss: float | None = None
    steps_ran = 0
    early_stopped = False
    loss_window: deque[float] = deque(maxlen=50)
    early_stop_min_steps = min(args.early_stop_min_steps, effective_training_steps)
    early_stop_threshold = args.early_stop_threshold
    pin_x: torch.Tensor | None = None
    pin_y: torch.Tensor | None = None
    pin_mask: torch.Tensor | None = None
    pin_itr: torch.Tensor | None = None
    if use_cuda:
        pin_x = torch.empty((batch_size, INPUT_DIM), dtype=torch.float32, pin_memory=True)
        pin_y = torch.empty((batch_size, MAX_ACTIONS), dtype=torch.float32, pin_memory=True)
        pin_mask = torch.empty((batch_size, MAX_ACTIONS), dtype=torch.float32, pin_memory=True)
        pin_itr = torch.empty((batch_size,), dtype=torch.long, pin_memory=True)
    use_chunked_disk_loading = isinstance(context.reservoir, DiskBackedReservoirBuffer)
    chunk_step_count = 500
    if use_chunked_disk_loading:
        chunk_count = int(math.ceil(effective_training_steps / chunk_step_count))
        approx_chunk_bytes = chunk_step_count * batch_size * (INPUT_DIM * 4 + MAX_ACTIONS * 4 + MAX_ACTIONS + 8)
        log(
            "[deep-cfr] using chunked disk loading: "
            f"{chunk_count} chunks of {chunk_step_count} steps "
            f"(~{approx_chunk_bytes / (1024**3):.2f} GiB/chunk before dedupe)"
        )

    def iter_batches():
        if not use_chunked_disk_loading:
            for idx in range(effective_training_steps):
                yield context.reservoir.sample_from_indices(batch_indices[idx])
            return
        assert isinstance(context.reservoir, DiskBackedReservoirBuffer)
        for chunk_start in range(0, effective_training_steps, chunk_step_count):
            chunk_end = min(chunk_start + chunk_step_count, effective_training_steps)
            chunk_indices = batch_indices[chunk_start:chunk_end]
            flat_indices = chunk_indices.reshape(-1)
            unique_sorted, inverse = np.unique(flat_indices, return_inverse=True)
            cache = context.reservoir.bulk_load_unique(unique_sorted)
            mapped_indices = inverse.reshape(chunk_indices.shape)
            for step_in_chunk in range(chunk_end - chunk_start):
                # Preserve the same sorted-per-batch semantics as sample_from_indices().
                mapped = np.sort(mapped_indices[step_in_chunk], kind="stable")
                yield SampleBatch(
                    cache.features[mapped],
                    cache.targets[mapped],
                    cache.action_masks[mapped],
                    cache.iterations[mapped],
                )
            del cache
            del mapped_indices

    for step, batch in enumerate(iter_batches(), start=1):
        batch_x = torch.from_numpy(batch.features)
        batch_y = torch.from_numpy(batch.targets)
        batch_mask = torch.from_numpy(batch.action_masks)
        batch_itr = torch.from_numpy(batch.iterations)

        if use_cuda:
            assert pin_x is not None
            assert pin_y is not None
            assert pin_mask is not None
            assert pin_itr is not None
            pin_x.copy_(batch_x)
            pin_y.copy_(batch_y)
            pin_mask.copy_(batch_mask.to(dtype=torch.float32))
            pin_itr.copy_(batch_itr)
            batch_x = pin_x
            batch_y = pin_y
            batch_mask = pin_mask
            batch_itr = pin_itr

        batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=use_cuda)
        batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=use_cuda)
        batch_mask = batch_mask.to(device=device, dtype=torch.float32, non_blocking=use_cuda)
        batch_itr = batch_itr.to(device=device, dtype=torch.float32, non_blocking=use_cuda)
        weights = (batch_itr / batch_itr.max().clamp(min=1.0)).clamp(min=1e-4)
        weights = weights / weights.mean().clamp(min=1e-8)

        context.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=context.use_amp):
            if context.network_type == "advantage":
                loss = train_advantage_step(
                    context.model,
                    batch_x,
                    batch_y,
                    batch_mask,
                    weights,
                    huber_delta=args.adv_huber_delta,
                )
            elif context.network_type == "strategy":
                loss = train_strategy_step(context.model, batch_x, batch_y, batch_mask, weights)
            else:
                raise ValueError(f"unsupported network_type={context.network_type}")

        context.scaler.scale(loss).backward()
        context.scaler.unscale_(context.optimizer)
        torch.nn.utils.clip_grad_norm_(context.model.parameters(), max_norm=1.0)
        context.scaler.step(context.optimizer)
        context.scaler.update()

        if step == 1 or step == effective_training_steps:
            with torch.no_grad():
                # #region agent log
                if context.network_type == "strategy":
                    mask = batch_mask
                    target_probs = torch.clamp(batch_y, min=0.0) * mask
                    target_mass = target_probs.sum(dim=1)
                    zero_target_frac = float((target_mass <= 1e-8).float().mean().item())
                    positive_action_count = (target_probs > 0.0).sum(dim=1).float()
                    target_norm = target_probs / target_mass.unsqueeze(1).clamp(min=1e-8)
                    entropy = -(target_norm * torch.log(target_norm.clamp(min=1e-8))).sum(dim=1)
                    pred_probs = context.model(batch_x, action_mask=batch_mask, strategy_mode=True)
                    pred_entropy = -(pred_probs * torch.log(pred_probs.clamp(min=1e-8)) * mask).sum(dim=1)
                    debug_log(
                        "H1",
                        "strategy_target_quality",
                        {
                            "step": int(step),
                            "effective_steps": int(effective_training_steps),
                            "zero_target_frac": zero_target_frac,
                            "positive_actions_mean": float(positive_action_count.mean().item()),
                            "target_entropy_mean": float(entropy.mean().item()),
                            "pred_entropy_mean": float(pred_entropy.mean().item()),
                        },
                    )
                    target_mass_col = target_probs.sum(dim=1, keepdim=True)
                    uniform_target = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    effective_target = torch.where(
                        target_mass_col > 1e-8,
                        target_probs / target_mass_col.clamp(min=1e-8),
                        uniform_target,
                    )
                    effective_entropy = -(
                        effective_target * torch.log(effective_target.clamp(min=1e-8)) * mask
                    ).sum(dim=1)
                    debug_log(
                        "H1",
                        "strategy_effective_target_quality",
                        {
                            "step": int(step),
                            "effective_steps": int(effective_training_steps),
                            "uniform_fallback_frac": zero_target_frac,
                            "effective_target_entropy_mean": float(effective_entropy.mean().item()),
                        },
                    )
                # #endregion
                # #region agent log
                if context.network_type == "advantage":
                    pred_adv = context.model(batch_x, action_mask=batch_mask, strategy_mode=False)
                    all_non_pos_pred_frac = float((pred_adv.max(dim=1).values <= 0.0).float().mean().item())
                    all_non_pos_target_frac = float((batch_y.max(dim=1).values <= 0.0).float().mean().item())
                    debug_log(
                        "H2",
                        "advantage_sign_profile",
                        {
                            "step": int(step),
                            "effective_steps": int(effective_training_steps),
                            "all_non_positive_pred_frac": all_non_pos_pred_frac,
                            "all_non_positive_target_frac": all_non_pos_target_frac,
                            "pred_abs_mean": float(pred_adv.abs().mean().item()),
                            "target_abs_mean": float(batch_y.abs().mean().item()),
                        },
                    )
                # #endregion

        final_loss = float(loss.item())
        steps_ran = step
        loss_window.append(final_loss)
        if step == 1 or step % args.log_every == 0 or step == effective_training_steps:
            log(f"[deep-cfr] step={step}/{effective_training_steps} loss={final_loss:.6f}")

        if (
            args.enable_early_stop
            and step >= early_stop_min_steps
            and len(loss_window) == loss_window.maxlen
        ):
            history = list(loss_window)
            midpoint = len(history) // 2
            first_avg = float(np.mean(history[:midpoint]))
            second_avg = float(np.mean(history[midpoint:]))
            denom = max(abs(first_avg), 1e-8)
            relative_improvement = (first_avg - second_avg) / denom
            if relative_improvement < early_stop_threshold:
                early_stopped = True
                log(
                    f"[deep-cfr] early stop @ step {step}/{effective_training_steps} "
                    f"(loss plateau, rel_improve={relative_improvement * 100.0:.3f}%)"
                )
                break

    gc.collect()
    save_model_checkpoint(context)
    fp16_out = context.onnx_path.with_name(f"{context.onnx_path.stem}.fp16{context.onnx_path.suffix}")
    export_onnx(
        context.model,
        context.onnx_path,
        context.network_type,
        args.onnx_opset,
        device,
        export_fp16=bool(getattr(args, "traversal_fp16_onnx", False)),
        fp16_out=fp16_out,
    )
    log(f"[deep-cfr] exported onnx={context.onnx_path}")
    if bool(getattr(args, "traversal_fp16_onnx", False)):
        log(f"[deep-cfr] exported onnx fp16={fp16_out}")
    return TrainingRunStats(
        final_loss=final_loss,
        steps_ran=steps_ran,
        steps_target=effective_training_steps,
        early_stopped=early_stopped,
    )


def measure_exploitability(
    *,
    measure_binary: Path,
    repo_root: Path,
    cluster_dir: Path,
    args: argparse.Namespace,
    fit_samples: int,
    eval_samples: int,
    fit_seed: int,
    eval_seed: int,
    onnx_path: Path | None = None,
    onnx_p0_path: Path | None = None,
    onnx_p1_path: Path | None = None,
    policy_output: str = "advantage",
    policy_output_p0: str | None = None,
    policy_output_p1: str | None = None,
) -> dict[str, float | int]:
    if onnx_path is None and (onnx_p0_path is None or onnx_p1_path is None):
        raise ValueError("either onnx_path or both onnx_p0_path/onnx_p1_path must be provided")

    progress_every = max(1, min(fit_samples, eval_samples) // 10)
    command = [
        str(measure_binary),
        "--cluster-dir",
        str(cluster_dir),
        "--fit-samples",
        str(fit_samples),
        "--eval-samples",
        str(eval_samples),
        "--deck-samples",
        str(args.deck_samples),
        "--fit-seed",
        str(fit_seed),
        "--eval-seed",
        str(eval_seed),
        "--progress-every",
        str(progress_every),
        "--starting-stack",
        str(args.starting_stack),
        "--small-blind",
        str(args.small_blind),
        "--big-blind",
        str(args.big_blind),
    ]
    if onnx_path is not None:
        command.extend(["--onnx", str(onnx_path), "--policy-output", policy_output])
    else:
        assert onnx_p0_path is not None
        assert onnx_p1_path is not None
        command.extend(["--onnx-p0", str(onnx_p0_path), "--onnx-p1", str(onnx_p1_path)])
        command.extend(["--policy-output", policy_output])
        if policy_output_p0 is not None:
            command.extend(["--policy-output-p0", policy_output_p0])
        if policy_output_p1 is not None:
            command.extend(["--policy-output-p1", policy_output_p1])

    bb_per_hand, mbb_per_hand = run_exploitability_measurement(command, cwd=repo_root)
    return {
        "bb_per_hand": bb_per_hand,
        "mbb_per_hand": mbb_per_hand,
        "fit_samples": fit_samples,
        "eval_samples": eval_samples,
        "fit_seed": fit_seed,
        "eval_seed": eval_seed,
    }


def measure_advantage_exploitability(
    measure_binary: Path,
    repo_root: Path,
    onnx_p0_path: Path,
    onnx_p1_path: Path,
    cluster_dir: Path,
    args: argparse.Namespace,
    fit_samples: int,
    eval_samples: int,
    fit_seed: int,
    eval_seed: int,
) -> dict[str, float | int]:
    return measure_exploitability(
        measure_binary=measure_binary,
        repo_root=repo_root,
        cluster_dir=cluster_dir,
        args=args,
        fit_samples=fit_samples,
        eval_samples=eval_samples,
        fit_seed=fit_seed,
        eval_seed=eval_seed,
        onnx_p0_path=onnx_p0_path,
        onnx_p1_path=onnx_p1_path,
        policy_output="advantage",
        policy_output_p0="advantage",
        policy_output_p1="advantage",
    )


def run_head_to_head_measurement(command: list[str], cwd: Path) -> HeadToHeadReport:
    log(f"[run] {command_to_string(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_subprocess_env(),
        encoding="utf-8",
        errors="replace",
    )
    model_a_bb_per_hand: float | None = None
    model_a_bb_per_100: float | None = None
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            log(line)
            match = H2H_MODEL_A_RE.search(line)
            if match is not None:
                model_a_bb_per_hand = float(match.group(1))
                model_a_bb_per_100 = float(match.group(2))
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {command_to_string(command)}"
        )
    if model_a_bb_per_hand is None or model_a_bb_per_100 is None:
        raise RuntimeError("head_to_head output did not contain a parsed model_a result line")
    return HeadToHeadReport(bb_per_hand=model_a_bb_per_hand, bb_per_100=model_a_bb_per_100)


def run_head_to_head(
    *,
    h2h_binary: Path,
    repo_root: Path,
    cluster_dir: Path,
    args: argparse.Namespace,
    seed: int,
    model_a_path: Path | None = None,
    policy_a: str = "advantage",
    tag_a: bool = False,
    random_a: bool = False,
    model_b_path: Path | None = None,
    policy_b: str = "advantage",
    tag_b: bool = False,
    random_b: bool = False,
    hands: int | None = None,
) -> dict[str, float | int]:
    hand_count = int(hands) if hands is not None and hands > 0 else int(args.h2h_hands)
    command = [
        str(h2h_binary),
        "--cluster-dir",
        str(cluster_dir),
        "--hands",
        str(hand_count),
        "--starting-stack",
        str(args.starting_stack),
        "--small-blind",
        str(args.small_blind),
        "--big-blind",
        str(args.big_blind),
        "--seed",
        str(seed),
        "--progress-every",
        str(max(1, hand_count // 10)),
    ]
    if tag_a:
        command.append("--tag-a")
    elif random_a:
        command.append("--random-a")
    else:
        if model_a_path is None:
            raise ValueError("model_a_path is required unless tag_a or random_a is set")
        command.extend(["--model-a", str(model_a_path), "--policy-a", policy_a])

    if tag_b:
        command.append("--tag-b")
    elif random_b:
        command.append("--random-b")
    else:
        if model_b_path is None:
            raise ValueError("model_b_path is required unless tag_b or random_b is set")
        command.extend(["--model-b", str(model_b_path), "--policy-b", policy_b])

    report = run_head_to_head_measurement(command, cwd=repo_root)
    return {
        "bb_per_hand": report.bb_per_hand,
        "bb_per_100": report.bb_per_100,
        "hands": hand_count,
        "seed": seed,
    }


def run_ring_game_eval_measurement(command: list[str], cwd: Path) -> dict[str, Any]:
    log(f"[run] {command_to_string(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_subprocess_env(),
        encoding="utf-8",
        errors="replace",
    )
    parsed_report: dict[str, Any] | None = None
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            log(line)
            if line.startswith(RING_EVAL_JSON_PREFIX):
                payload = line[len(RING_EVAL_JSON_PREFIX) :].strip()
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"failed to parse ring_game_eval JSON payload: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise RuntimeError("ring_game_eval JSON payload was not an object")
                parsed_report = parsed
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {command_to_string(command)}"
        )
    if parsed_report is None:
        raise RuntimeError("ring_game_eval output did not contain a parsed JSON payload")
    return parsed_report


def run_ring_game_eval(
    *,
    ring_eval_binary: Path,
    repo_root: Path,
    cluster_dir: Path,
    args: argparse.Namespace,
    model_path: Path,
    seed: int,
    opponent: str = "tag",
) -> dict[str, Any]:
    hand_count = int(args.ring_eval_hands)
    deck_samples = min(max(1, int(args.deck_samples)), 200)
    worker_count = 4
    command = [
        str(ring_eval_binary),
        "--mode",
        "vs-opponent",
        "--opponent",
        str(opponent),
        "--model",
        str(model_path),
        "--policy",
        "strategy",
        "--model-seat",
        "0",
        "--num-players",
        str(args.num_players),
        "--hands",
        str(hand_count),
        "--deck-samples",
        str(deck_samples),
        "--workers",
        str(worker_count),
        "--seed",
        str(seed),
        "--progress-every",
        str(max(1, hand_count // 10)),
        "--cluster-dir",
        str(cluster_dir),
        "--starting-stack",
        str(args.starting_stack),
        "--small-blind",
        str(args.small_blind),
        "--big-blind",
        str(args.big_blind),
    ]
    report = run_ring_game_eval_measurement(command, cwd=repo_root)
    raw_per_seat_bb_per_hand = report.get("per_seat_bb_per_hand", [])
    raw_per_seat_bb_per_100 = report.get("per_seat_bb_per_100", [])
    per_seat_bb_per_hand = (
        [float(value) for value in raw_per_seat_bb_per_hand if isinstance(value, (int, float))]
        if isinstance(raw_per_seat_bb_per_hand, list)
        else []
    )
    per_seat_bb_per_100 = (
        [float(value) for value in raw_per_seat_bb_per_100 if isinstance(value, (int, float))]
        if isinstance(raw_per_seat_bb_per_100, list)
        else []
    )
    return {
        "status": "ok",
        "mode": str(report.get("mode", "vs_opponent")),
        "opponent": str(report.get("opponent", "tag")),
        "hands": int(report.get("hands", hand_count)),
        "seed": int(seed),
        "num_players": int(report.get("num_players", args.num_players)),
        "model_seat": int(report.get("model_seat", 0)),
        "model_bb_per_hand": float(report.get("model_bb_per_hand", 0.0)),
        "model_bb_per_100": float(report.get("model_bb_per_100", 0.0)),
        "per_seat_bb_per_hand": per_seat_bb_per_hand,
        "per_seat_bb_per_100": per_seat_bb_per_100,
        "model_wins": int(report.get("model_wins", 0)),
        "model_losses": int(report.get("model_losses", 0)),
        "ties": int(report.get("ties", 0)),
        "zero_sum_check_bb_per_hand": float(report.get("zero_sum_check_bb_per_hand", 0.0)),
        "elapsed_sec": float(report.get("elapsed_sec", 0.0)),
        "workers": worker_count,
        "deck_samples": deck_samples,
    }


def last_diagnostics(metrics_iterations: list[Any]) -> dict[str, Any] | None:
    for entry in reversed(metrics_iterations):
        if not isinstance(entry, dict):
            continue
        diagnostics = entry.get("diagnostics")
        if isinstance(diagnostics, dict):
            return diagnostics
    return None


def first_diagnostics(metrics_iterations: list[Any]) -> dict[str, Any] | None:
    for entry in metrics_iterations:
        if not isinstance(entry, dict):
            continue
        diagnostics = entry.get("diagnostics")
        if isinstance(diagnostics, dict):
            return diagnostics
    return None


def build_eval_panel_config(args: argparse.Namespace) -> EvalPanelConfig:
    fit_samples = args.exploit_fit_samples if args.exploit_fit_samples > 0 else args.exploit_samples
    eval_samples = args.exploit_eval_samples if args.exploit_eval_samples > 0 else args.exploit_samples
    base_seed = args.fixed_eval_seed if args.fixed_eval_seed > 0 else (args.seed ^ 0x6A09_E667)
    holdout_seeds: list[int] = []
    seed_cursor = base_seed ^ 0xC2B2_AE35
    for idx in range(args.selection_holdout_seeds):
        seed_cursor = (
            (seed_cursor * 1_664_525)
            + 1_013_904_223
            + ((idx + 1) * 2_654_435_761)
        ) & 0xFFFF_FFFF
        holdout_seeds.append(seed_cursor)
    return EvalPanelConfig(
        exploit_fit_samples=fit_samples,
        exploit_eval_samples=eval_samples,
        exploit_strategy_fit_seed=base_seed ^ 0xBB67_AE85,
        exploit_strategy_eval_seed=base_seed ^ 0x3C6E_F372,
        exploit_adv_fit_seed=base_seed ^ 0xA54F_F53A,
        exploit_adv_eval_seed=base_seed ^ 0x510E_527F,
        h2h_strategy_tag_seed=base_seed ^ 0x9B05_688C,
        h2h_strategy_random_seed=base_seed ^ 0x1F83_D9AB,
        h2h_strategy_tag_holdout_seeds=tuple(holdout_seeds),
    )


def run_strategy_model_selection(
    *,
    iteration: int,
    h2h_binary: Path,
    repo_root: Path,
    cluster_dir: Path,
    args: argparse.Namespace,
    eval_panel: EvalPanelConfig,
    strategy_onnx: Path,
    strategy_state: Path,
    strategy_latest_onnx: Path,
    strategy_latest_state: Path,
    strategy_best_candidate_onnx: Path,
    strategy_best_candidate_state: Path,
    strategy_best_confirmed_onnx: Path,
    strategy_best_confirmed_state: Path,
    diagnostics_report: dict[str, Any],
    model_selection_state: dict[str, Any],
) -> dict[str, Any]:
    if not strategy_onnx.exists() or not strategy_state.exists():
        return {
            "status": "skipped",
            "reason": "missing_strategy_artifacts",
            "iteration": iteration,
        }

    latest_copied = bool(
        safe_copy_file(strategy_onnx, strategy_latest_onnx)
        and safe_copy_file(strategy_state, strategy_latest_state)
    )
    result: dict[str, Any] = {
        "status": "evaluated",
        "iteration": iteration,
        "latest_saved": latest_copied,
    }

    current_tag = get_metric_value(diagnostics_report, "h2h_strategy_vs_tag", "bb_per_100")
    current_random = get_metric_value(diagnostics_report, "h2h_strategy_vs_random", "bb_per_100")
    if current_tag is None:
        result["status"] = "skipped"
        result["reason"] = "missing_tag_metric"
        return result

    result["current_tag_bb_per_100"] = current_tag
    result["current_random_bb_per_100"] = current_random
    random_guardrail = float(args.selection_random_guardrail)
    random_pass = current_random is None or current_random >= random_guardrail
    result["random_guardrail_bb_per_100"] = random_guardrail
    result["random_guardrail_passed"] = random_pass
    if not random_pass:
        result["status"] = "skipped"
        result["reason"] = "random_guardrail_not_met"
        return result

    best_candidate_obj = model_selection_state.get("best_candidate")
    best_candidate_tag = get_metric_value({"x": best_candidate_obj}, "x", "tag_bb_per_100")
    if best_candidate_tag is None:
        best_candidate_tag = float("-inf")
    result["best_candidate_tag_bb_per_100_before"] = best_candidate_tag
    improved_candidate = current_tag > best_candidate_tag
    result["improved_candidate"] = improved_candidate
    if not improved_candidate:
        result["status"] = "no_improvement"
        return result

    candidate_saved = bool(
        safe_copy_file(strategy_onnx, strategy_best_candidate_onnx)
        and safe_copy_file(strategy_state, strategy_best_candidate_state)
    )
    holdout_hands = int(args.selection_holdout_hands) if args.selection_holdout_hands > 0 else int(args.h2h_hands)
    holdout_scores: list[float] = []
    holdout_details: list[dict[str, float | int]] = []
    for seed in eval_panel.h2h_strategy_tag_holdout_seeds:
        holdout_report = run_head_to_head(
            h2h_binary=h2h_binary,
            repo_root=repo_root,
            cluster_dir=cluster_dir,
            args=args,
            seed=seed,
            model_a_path=strategy_onnx,
            policy_a="strategy",
            tag_b=True,
            hands=holdout_hands,
        )
        holdout_details.append(holdout_report)
        metric_value = holdout_report.get("bb_per_100")
        if isinstance(metric_value, (int, float)):
            holdout_scores.append(float(metric_value))

    holdout_avg = float(sum(holdout_scores) / len(holdout_scores)) if holdout_scores else None
    candidate_payload: dict[str, Any] = {
        "iteration": int(iteration),
        "tag_bb_per_100": float(current_tag),
        "random_bb_per_100": current_random,
        "holdout_tag_bb_per_100_avg": holdout_avg,
        "holdout_runs": holdout_details,
        "holdout_hands": holdout_hands,
        "onnx_path": str(strategy_best_candidate_onnx),
        "state_path": str(strategy_best_candidate_state),
    }
    model_selection_state["best_candidate"] = candidate_payload

    result["status"] = "candidate_updated"
    result["candidate_saved"] = candidate_saved
    result["holdout_tag_bb_per_100_avg"] = holdout_avg
    result["holdout_runs"] = holdout_details

    best_confirmed_obj = model_selection_state.get("best_confirmed")
    best_confirmed_holdout = get_metric_value(
        {"x": best_confirmed_obj},
        "x",
        "holdout_tag_bb_per_100_avg",
    )
    if best_confirmed_holdout is None:
        best_confirmed_holdout = float("-inf")
    result["best_confirmed_holdout_bb_per_100_before"] = best_confirmed_holdout

    if holdout_avg is None:
        result["promoted_confirmed"] = False
        result["promotion_reason"] = "missing_holdout_scores"
        return result
    if holdout_avg <= best_confirmed_holdout:
        result["promoted_confirmed"] = False
        result["promotion_reason"] = "holdout_not_better"
        return result

    confirmed_saved = bool(
        safe_copy_file(strategy_onnx, strategy_best_confirmed_onnx)
        and safe_copy_file(strategy_state, strategy_best_confirmed_state)
    )
    confirmed_payload: dict[str, Any] = dict(candidate_payload)
    confirmed_payload["onnx_path"] = str(strategy_best_confirmed_onnx)
    confirmed_payload["state_path"] = str(strategy_best_confirmed_state)
    confirmed_payload["promoted_at_iteration"] = int(iteration)
    model_selection_state["best_confirmed"] = confirmed_payload

    result["promoted_confirmed"] = True
    result["confirmed_saved"] = confirmed_saved
    return result


def print_diagnostic_comparison(
    iteration: int,
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    baseline: dict[str, Any] | None = None,
) -> None:
    log(color_bold(f"  -- Diagnostic @ iter {iteration:03d} --"))
    rows = [
        ("exploitability_strategy", "Exploit(strat)", "mbb_per_hand", True, "mBB/hand"),
        ("exploitability_advantage_smoke", "Exploit(adv-smk)", "mbb_per_hand", True, "mBB/hand"),
        ("h2h_strategy_vs_tag", "Strat vs TAG", "bb_per_100", False, "bb/100"),
        ("h2h_strategy_vs_random", "Strat vs Random", "bb_per_100", False, "bb/100"),
    ]
    for key, label, field, lower_is_better, unit in rows:
        metric = current.get(key)
        if not isinstance(metric, dict):
            continue
        current_value_raw = metric.get(field)
        if not isinstance(current_value_raw, (int, float)):
            continue
        current_value = float(current_value_raw)
        previous_value: float | None = None
        if isinstance(previous, dict):
            prev_metric = previous.get(key)
            if isinstance(prev_metric, dict):
                prev_raw = prev_metric.get(field)
                if isinstance(prev_raw, (int, float)):
                    previous_value = float(prev_raw)
        baseline_value: float | None = None
        if isinstance(baseline, dict):
            base_metric = baseline.get(key)
            if isinstance(base_metric, dict):
                base_raw = base_metric.get(field)
                if isinstance(base_raw, (int, float)):
                    baseline_value = float(base_raw)

        if previous_value is None:
            delta_text = color_yellow("new")
            prev_text = "n/a"
        else:
            delta = current_value - previous_value
            changed = format_signed(delta, decimals=3)
            pct = format_change_pct(current_value, previous_value)
            if abs(delta) < 1e-12:
                delta_text = color_yellow(f"{changed} ({pct})")
            else:
                improved = delta < 0.0 if lower_is_better else delta > 0.0
                direction = "improved" if improved else "worse"
                payload = f"{changed} ({pct}) {direction}"
                delta_text = color_green(payload) if improved else color_red(payload)
            prev_text = f"{previous_value:+.3f}"
        baseline_delta_text = "n/a"
        if baseline_value is not None:
            base_delta = current_value - baseline_value
            baseline_delta_text = format_signed(base_delta, decimals=3)
        log(
            f"    {label:<18} curr={current_value:+.3f} {unit:<8} "
            f"prev={prev_text:<12} delta={delta_text} baseline_delta={baseline_delta_text}"
        )


def print_diagnostic_table(
    iteration: int,
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    baseline: dict[str, Any] | None = None,
) -> None:
    print_diagnostic_comparison(iteration, current, previous, baseline)


def run_diagnostic_suite(
    *,
    iteration: int,
    h2h_binary: Path,
    measure_binary: Path,
    repo_root: Path,
    cluster_dir: Path,
    args: argparse.Namespace,
    advantage_p0_onnx: Path,
    advantage_p1_onnx: Path,
    strategy_onnx: Path,
    strategy_trained: bool,
    eval_panel: EvalPanelConfig,
    h2h_only: bool,
    previous_diagnostics: dict[str, Any] | None,
    baseline_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}

    strategy_available = strategy_onnx.exists()
    if not h2h_only:
        diagnostics["exploitability_advantage_smoke"] = measure_advantage_exploitability(
            measure_binary=measure_binary,
            repo_root=repo_root,
            onnx_p0_path=advantage_p0_onnx,
            onnx_p1_path=advantage_p1_onnx,
            cluster_dir=cluster_dir,
            args=args,
            fit_samples=eval_panel.exploit_fit_samples,
            eval_samples=eval_panel.exploit_eval_samples,
            fit_seed=eval_panel.exploit_adv_fit_seed,
            eval_seed=eval_panel.exploit_adv_eval_seed,
        )

    if strategy_available:
        if not h2h_only:
            diagnostics["exploitability_strategy"] = measure_exploitability(
                measure_binary=measure_binary,
                repo_root=repo_root,
                cluster_dir=cluster_dir,
                args=args,
                fit_samples=eval_panel.exploit_fit_samples,
                eval_samples=eval_panel.exploit_eval_samples,
                fit_seed=eval_panel.exploit_strategy_fit_seed,
                eval_seed=eval_panel.exploit_strategy_eval_seed,
                onnx_path=strategy_onnx,
                policy_output="strategy",
            )
        diagnostics["h2h_strategy_vs_tag"] = run_head_to_head(
            h2h_binary=h2h_binary,
            repo_root=repo_root,
            cluster_dir=cluster_dir,
            args=args,
            seed=eval_panel.h2h_strategy_tag_seed,
            model_a_path=strategy_onnx,
            policy_a="strategy",
            tag_b=True,
        )
        diagnostics["h2h_strategy_vs_random"] = run_head_to_head(
            h2h_binary=h2h_binary,
            repo_root=repo_root,
            cluster_dir=cluster_dir,
            args=args,
            seed=eval_panel.h2h_strategy_random_seed,
            model_a_path=strategy_onnx,
            policy_a="strategy",
            random_b=True,
        )
    else:
        diagnostics["strategy_unavailable"] = {
            "strategy_trained_this_iter": strategy_trained,
            "strategy_path": str(strategy_onnx),
        }

    # #region agent log
    exploit_strategy_obj = diagnostics.get("exploitability_strategy", {})
    exploit_adv_obj = diagnostics.get("exploitability_advantage_smoke", {})
    strat_tag_obj = diagnostics.get("h2h_strategy_vs_tag", {})
    strat_rand_obj = diagnostics.get("h2h_strategy_vs_random", {})
    debug_log(
        "H4",
        "diagnostic_snapshot",
        {
            "iteration": int(iteration),
            "exploit_strategy_bb_per_hand": (
                float(exploit_strategy_obj.get("bb_per_hand"))
                if isinstance(exploit_strategy_obj, dict) and "bb_per_hand" in exploit_strategy_obj
                else None
            ),
            "exploit_adv_smoke_bb_per_hand": (
                float(exploit_adv_obj.get("bb_per_hand"))
                if isinstance(exploit_adv_obj, dict) and "bb_per_hand" in exploit_adv_obj
                else None
            ),
            "strategy_available": bool(strategy_available),
            "strat_vs_tag_bb100": (
                float(strat_tag_obj.get("bb_per_100"))
                if strategy_available and isinstance(strat_tag_obj, dict) and "bb_per_100" in strat_tag_obj
                else None
            ),
            "strat_vs_random_bb100": (
                float(strat_rand_obj.get("bb_per_100"))
                if strategy_available and isinstance(strat_rand_obj, dict) and "bb_per_100" in strat_rand_obj
                else None
            ),
        },
    )
    # #endregion

    print_diagnostic_table(iteration, diagnostics, previous_diagnostics, baseline_diagnostics)
    return diagnostics


def bootstrap_shared_advantage_artifacts(
    *,
    shared_onnx: Path,
    shared_state: Path,
    shared_buffer: Path,
    legacy_shared_onnx: Path,
    legacy_shared_state: Path,
    legacy_shared_buffer: Path,
    split_p0_onnx: Path,
    split_p1_onnx: Path,
    split_p0_state: Path,
    split_p1_state: Path,
    split_p0_buffer: Path,
    split_p1_buffer: Path,
) -> None:
    if not shared_onnx.exists():
        source_onnx = None
        for candidate in (legacy_shared_onnx, split_p0_onnx, split_p1_onnx):
            if candidate.exists():
                source_onnx = candidate
                break
        if source_onnx is not None:
            shutil.copy2(source_onnx, shared_onnx)
            log(f"[init] migrated shared advantage model from {source_onnx.name} -> {shared_onnx.name}")

    if not shared_state.exists():
        source_state = None
        for candidate in (legacy_shared_state, split_p0_state, split_p1_state):
            if candidate.exists():
                source_state = candidate
                break
        if source_state is not None:
            shutil.copy2(source_state, shared_state)
            log(
                f"[init] migrated shared advantage checkpoint from {source_state.name} -> {shared_state.name}"
            )

    if not shared_buffer.exists():
        source_buffer = None
        for candidate in (legacy_shared_buffer, split_p0_buffer, split_p1_buffer):
            if candidate.exists():
                source_buffer = candidate
                break
        if source_buffer is not None:
            shutil.copy2(source_buffer, shared_buffer)
            log(f"[init] migrated shared advantage buffer from {source_buffer.name} -> {shared_buffer.name}")


def build_traversal_command_shared(
    rust_binary: Path,
    shared_onnx_path: Path,
    advantage_samples_out: Path,
    strategy_samples_out: Path,
    player: int,
    traversals: int,
    iteration: int,
    args: argparse.Namespace,
    cluster_dir: Path,
    seed: int,
) -> list[str]:
    command = [
        str(rust_binary),
        "--onnx",
        str(shared_onnx_path),
        "--adv-samples-out",
        str(advantage_samples_out),
        "--strategy-samples-out",
        str(strategy_samples_out),
        "--player",
        str(player),
        "--traversals",
        str(traversals),
        "--progress-batch",
        str(args.traversal_progress_batch),
        "--workers",
        str(args.traversal_workers),
        "--iteration",
        str(iteration),
        "--seed",
        str(seed),
        "--cluster-dir",
        str(cluster_dir),
        "--deck-samples",
        str(args.deck_samples),
        "--num-players",
        str(args.num_players),
        "--starting-stack",
        str(args.starting_stack),
        "--small-blind",
        str(args.small_blind),
        "--big-blind",
        str(args.big_blind),
    ]
    if bool(getattr(args, "traversal_gpu_batch", False)):
        command.extend(
            [
                "--gpu-batch",
                "--gpu-batch-size",
                str(getattr(args, "traversal_gpu_batch_size", 256)),
                "--gpu-batch-timeout-us",
                str(getattr(args, "traversal_gpu_batch_timeout_us", 500)),
                "--gpu-batch-queue-capacity",
                str(getattr(args, "traversal_gpu_batch_queue_capacity", 8192)),
                "--gpu-device-id",
                str(getattr(args, "traversal_gpu_device_id", 0)),
            ]
        )
        if bool(getattr(args, "traversal_gpu_batch_cpu_only", False)):
            command.append("--gpu-batch-cpu-only")
        if bool(getattr(args, "traversal_gpu_batch_no_tf32", False)):
            command.append("--gpu-batch-no-tf32")
    cpu_affinity = str(getattr(args, "traversal_cpu_affinity", "")).strip()
    if cpu_affinity:
        command.extend(["--cpu-affinity", cpu_affinity])
    return command


def build_traversal_command_shared_multi(
    *,
    rust_binary: Path,
    shared_onnx_path: Path,
    advantage_samples_template: Path,
    strategy_samples_template: Path,
    players: list[int],
    traversals: int,
    iteration: int,
    args: argparse.Namespace,
    cluster_dir: Path,
    seed: int,
) -> list[str]:
    command = [
        str(rust_binary),
        "--onnx",
        str(shared_onnx_path),
        "--adv-samples-out",
        str(advantage_samples_template),
        "--strategy-samples-out",
        str(strategy_samples_template),
        "--adv-samples-template",
        str(advantage_samples_template),
        "--strategy-samples-template",
        str(strategy_samples_template),
        "--player",
        "0",
        "--player-list",
        ",".join(str(player) for player in players),
        "--traversals",
        str(traversals),
        "--progress-batch",
        str(args.traversal_progress_batch),
        "--workers",
        str(args.traversal_workers),
        "--iteration",
        str(iteration),
        "--seed",
        str(seed),
        "--cluster-dir",
        str(cluster_dir),
        "--deck-samples",
        str(args.deck_samples),
        "--num-players",
        str(args.num_players),
        "--starting-stack",
        str(args.starting_stack),
        "--small-blind",
        str(args.small_blind),
        "--big-blind",
        str(args.big_blind),
    ]
    if bool(getattr(args, "traversal_gpu_batch", False)):
        command.extend(
            [
                "--gpu-batch",
                "--gpu-batch-size",
                str(getattr(args, "traversal_gpu_batch_size", 256)),
                "--gpu-batch-timeout-us",
                str(getattr(args, "traversal_gpu_batch_timeout_us", 500)),
                "--gpu-batch-queue-capacity",
                str(getattr(args, "traversal_gpu_batch_queue_capacity", 8192)),
                "--gpu-device-id",
                str(getattr(args, "traversal_gpu_device_id", 0)),
            ]
        )
        if bool(getattr(args, "traversal_gpu_batch_cpu_only", False)):
            command.append("--gpu-batch-cpu-only")
        if bool(getattr(args, "traversal_gpu_batch_no_tf32", False)):
            command.append("--gpu-batch-no-tf32")
    cpu_affinity = str(getattr(args, "traversal_cpu_affinity", "")).strip()
    if cpu_affinity:
        command.extend(["--cpu-affinity", cpu_affinity])
    return command


def start_traversal_async_shared(
    *,
    rust_binary: Path,
    repo_root: Path,
    shared_onnx_path: Path,
    advantage_samples_out: Path,
    strategy_samples_out: Path,
    player: int,
    traversals: int,
    iteration: int,
    args: argparse.Namespace,
    cluster_dir: Path,
    seed: int,
    log_path: Path,
) -> TraversalTask:
    command = build_traversal_command_shared(
        rust_binary=rust_binary,
        shared_onnx_path=shared_onnx_path,
        advantage_samples_out=advantage_samples_out,
        strategy_samples_out=strategy_samples_out,
        player=player,
        traversals=traversals,
        iteration=iteration,
        args=args,
        cluster_dir=cluster_dir,
        seed=seed,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=output_handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_subprocess_env(),
        encoding="utf-8",
        errors="replace",
    )
    return TraversalTask(
        player=player,
        process=process,
        advantage_sample_path=advantage_samples_out,
        strategy_sample_path=strategy_samples_out,
        log_path=log_path,
        started_at=time.perf_counter(),
        log_handle=output_handle,
    )


def split_traversal_counts(total_traversals: int, chunk_count: int) -> list[int]:
    chunk_count = max(1, int(chunk_count))
    base = total_traversals // chunk_count
    extra = total_traversals % chunk_count
    out: list[int] = []
    for idx in range(chunk_count):
        count = base + (1 if idx < extra else 0)
        if count > 0:
            out.append(count)
    if not out:
        out.append(int(total_traversals))
    return out


def start_iteration_traversals_async_shared(
    *,
    iteration: int,
    rust_binary: Path,
    repo_root: Path,
    shared_onnx_path: Path,
    models_dir: Path,
    samples_dir: Path,
    traversal_logs_dir: Path,
    args: argparse.Namespace,
    cluster_dir: Path,
) -> IterationTraversalBundle:
    model_copy = models_dir / f"advantage_shared_bg_iter{iteration:03d}.onnx"
    shutil.copy2(shared_onnx_path, model_copy)
    seat_chunks = max(1, int(getattr(args, "traversal_seat_chunks", 1)))
    consolidate = bool(getattr(args, "traversal_consolidate_processes", False)) and seat_chunks == 1
    if consolidate:
        players = list(range(args.num_players))
        sample_adv_template = samples_dir / f"samples_adv_p{{player}}_iter{iteration:03d}.bin"
        sample_strategy_template = (
            samples_dir / f"samples_strategy_from_p{{player}}_iter{iteration:03d}.bin"
        )
        log_path = traversal_logs_dir / f"traverse_all_iter{iteration:03d}.log"
        seed = args.seed ^ (iteration * 0x9E37_79B9) ^ (args.num_players * 0x94D0_49BB)
        command = build_traversal_command_shared_multi(
            rust_binary=rust_binary,
            shared_onnx_path=model_copy,
            advantage_samples_template=sample_adv_template,
            strategy_samples_template=sample_strategy_template,
            players=players,
            traversals=int(args.traversals),
            iteration=iteration,
            args=args,
            cluster_dir=cluster_dir,
            seed=seed,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=build_subprocess_env(),
            encoding="utf-8",
            errors="replace",
        )
        adv_paths = {
            player: Path(str(sample_adv_template).replace("{player}", str(player)))
            for player in players
        }
        strategy_paths = {
            player: Path(str(sample_strategy_template).replace("{player}", str(player)))
            for player in players
        }
        first_player = players[0] if players else 0
        task = TraversalTask(
            player=-1,
            process=process,
            advantage_sample_path=adv_paths.get(first_player, sample_adv_template),
            strategy_sample_path=strategy_paths.get(first_player, sample_strategy_template),
            log_path=log_path,
            started_at=time.perf_counter(),
            log_handle=output_handle,
        )
        return IterationTraversalBundle(
            iteration=iteration,
            model_copy_paths=[model_copy],
            tasks=[task],
            consolidated=True,
            consolidated_players=players,
            consolidated_adv_paths=adv_paths,
            consolidated_strategy_paths=strategy_paths,
        )

    tasks: list[TraversalTask] = []
    for player in range(args.num_players):
        traversal_chunks = split_traversal_counts(int(args.traversals), seat_chunks)
        for chunk_idx, chunk_traversals in enumerate(traversal_chunks):
            if seat_chunks <= 1:
                sample_adv = samples_dir / f"samples_adv_p{player}_iter{iteration:03d}.bin"
                sample_strategy = (
                    samples_dir / f"samples_strategy_from_p{player}_iter{iteration:03d}.bin"
                )
                log_path = traversal_logs_dir / f"traverse_p{player}_iter{iteration:03d}.log"
            else:
                sample_adv = (
                    samples_dir / f"samples_adv_p{player}_iter{iteration:03d}_c{chunk_idx:03d}.bin"
                )
                sample_strategy = (
                    samples_dir
                    / f"samples_strategy_from_p{player}_iter{iteration:03d}_c{chunk_idx:03d}.bin"
                )
                log_path = (
                    traversal_logs_dir
                    / f"traverse_p{player}_iter{iteration:03d}_c{chunk_idx:03d}.log"
                )
            seed = (
                args.seed
                ^ (iteration * 0x9E37_79B9)
                ^ ((player + 1) * 0xA24B_AED4)
                ^ ((chunk_idx + 1) * 0xC2B2_AE35)
                ^ (args.num_players * 0x94D0_49BB)
            )
            tasks.append(
                start_traversal_async_shared(
                    rust_binary=rust_binary,
                    repo_root=repo_root,
                    shared_onnx_path=model_copy,
                    advantage_samples_out=sample_adv,
                    strategy_samples_out=sample_strategy,
                    player=player,
                    traversals=int(chunk_traversals),
                    iteration=iteration,
                    args=args,
                    cluster_dir=cluster_dir,
                    seed=seed,
                    log_path=log_path,
                )
            )
    return IterationTraversalBundle(
        iteration=iteration,
        model_copy_paths=[model_copy],
        tasks=tasks,
    )


def wait_for_iteration_traversals_shared(
    bundle: IterationTraversalBundle,
) -> dict[str, Any]:
    players: list[dict[str, Any]] = []
    per_player: dict[int, dict[str, Any]] = {}
    traverse_total = 0.0
    wall_started = time.perf_counter()
    if bundle.consolidated:
        if not bundle.tasks:
            raise RuntimeError(
                f"consolidated traversal bundle for iter {bundle.iteration} has no process task"
            )
        task = bundle.tasks[0]
        delete_log = False
        try:
            code = task.process.wait()
            finished_at = time.perf_counter()
            elapsed = finished_at - task.started_at
            if code != 0:
                tail = read_text_tail(task.log_path)
                raise RuntimeError(
                    f"consolidated traversal failed in iter {bundle.iteration} with exit code {code}\n"
                    f"log: {task.log_path}\n{tail}"
                )
            runtime_profile = parse_parallel_traversal_log(task.log_path)
            runtime_summary = runtime_profile.get("summary", {}) if isinstance(runtime_profile, dict) else {}
            per_player_elapsed = elapsed / float(max(1, len(bundle.consolidated_players)))
            for player in bundle.consolidated_players:
                adv_path = bundle.consolidated_adv_paths.get(player)
                strategy_path = bundle.consolidated_strategy_paths.get(player)
                if adv_path is None or not adv_path.exists():
                    raise RuntimeError(
                        f"missing consolidated advantage samples for player {player} in iter {bundle.iteration}"
                    )
                if strategy_path is None or not strategy_path.exists():
                    raise RuntimeError(
                        f"missing consolidated strategy samples for player {player} in iter {bundle.iteration}"
                    )
                adv_count, _ = read_sample_metadata(
                    adv_path,
                    expected_magic=ADVANTAGE_SAMPLE_MAGIC,
                )
                strat_count, _ = read_sample_metadata(
                    strategy_path,
                    expected_magic=STRATEGY_SAMPLE_MAGIC,
                )
                players.append(
                    {
                        "player": int(player),
                        "advantage_sample_path": adv_path,
                        "strategy_sample_path": strategy_path,
                        "advantage_samples": int(adv_count),
                        "strategy_samples": int(strat_count),
                        "time_sec": float(per_player_elapsed),
                        "runtime_profile": {"summary": runtime_summary} if runtime_summary else {},
                        "seat_chunk_count": 1,
                    }
                )
                traverse_total += float(per_player_elapsed)
            delete_log = True
        finally:
            task.log_handle.close()
        if delete_log:
            safe_delete(task.log_path)
        for model_copy_path in bundle.model_copy_paths:
            safe_delete(model_copy_path)
        players.sort(key=lambda item: int(item.get("player", 0)))
        seat_time_min_sec = 0.0
        seat_time_max_sec = 0.0
        seat_time_avg_sec = 0.0
        seat_time_ratio = 0.0
        fastest_player: int | None = None
        slowest_player: int | None = None
        if players:
            times = [float(item.get("time_sec", 0.0)) for item in players if isinstance(item, dict)]
            if times:
                seat_time_min_sec = min(times)
                seat_time_max_sec = max(times)
                seat_time_avg_sec = sum(times) / float(len(times))
                seat_time_ratio = (
                    seat_time_max_sec / seat_time_min_sec if seat_time_min_sec > 1e-9 else 0.0
                )
            fastest = min(players, key=lambda item: float(item.get("time_sec", 0.0)))
            slowest = max(players, key=lambda item: float(item.get("time_sec", 0.0)))
            fastest_player = int(fastest.get("player", -1))
            slowest_player = int(slowest.get("player", -1))
        return {
            "iteration": bundle.iteration,
            "players": players,
            "traverse_total": traverse_total,
            "traverse_wall": max(0.0, time.perf_counter() - wall_started),
            "seat_time_min_sec": seat_time_min_sec,
            "seat_time_max_sec": seat_time_max_sec,
            "seat_time_avg_sec": seat_time_avg_sec,
            "seat_time_ratio_max_to_min": seat_time_ratio,
            "fastest_player": fastest_player,
            "slowest_player": slowest_player,
        }

    for task in bundle.tasks:
        delete_log = False
        try:
            code = task.process.wait()
            finished_at = time.perf_counter()
            elapsed = finished_at - task.started_at
            if code != 0:
                tail = read_text_tail(task.log_path)
                raise RuntimeError(
                    f"traversal failed for player {task.player} in iter {bundle.iteration} with exit code {code}\n"
                    f"log: {task.log_path}\n{tail}"
                )
            adv_count, _ = read_sample_metadata(
                task.advantage_sample_path,
                expected_magic=ADVANTAGE_SAMPLE_MAGIC,
            )
            strat_count, _ = read_sample_metadata(
                task.strategy_sample_path,
                expected_magic=STRATEGY_SAMPLE_MAGIC,
            )
            runtime_profile = parse_parallel_traversal_log(task.log_path)
            player_info = per_player.setdefault(
                int(task.player),
                {
                    "adv_paths": [],
                    "strategy_paths": [],
                    "advantage_samples_total": 0,
                    "strategy_samples_total": 0,
                    "started_min": float(task.started_at),
                    "finished_max": float(finished_at),
                    "runtime_summaries": [],
                },
            )
            player_info["adv_paths"].append(task.advantage_sample_path)
            player_info["strategy_paths"].append(task.strategy_sample_path)
            player_info["advantage_samples_total"] = int(
                player_info.get("advantage_samples_total", 0)
            ) + int(adv_count)
            player_info["strategy_samples_total"] = int(
                player_info.get("strategy_samples_total", 0)
            ) + int(strat_count)
            player_info["started_min"] = min(
                float(player_info.get("started_min", task.started_at)),
                float(task.started_at),
            )
            player_info["finished_max"] = max(
                float(player_info.get("finished_max", finished_at)),
                float(finished_at),
            )
            runtime_summary = runtime_profile.get("summary", {}) if isinstance(runtime_profile, dict) else {}
            if isinstance(runtime_summary, dict) and runtime_summary:
                runtime_summaries = player_info.get("runtime_summaries")
                if isinstance(runtime_summaries, list):
                    runtime_summaries.append(runtime_summary)
            delete_log = True
        finally:
            task.log_handle.close()
        if delete_log:
            safe_delete(task.log_path)
    for model_copy_path in bundle.model_copy_paths:
        safe_delete(model_copy_path)
    for player in sorted(per_player):
        info = per_player[player]
        adv_paths = [
            path for path in info.get("adv_paths", []) if isinstance(path, Path) and path.exists()
        ]
        strategy_paths = [
            path for path in info.get("strategy_paths", []) if isinstance(path, Path) and path.exists()
        ]
        if not adv_paths:
            raise RuntimeError(
                f"no advantage sample files collected for player {player} in iter {bundle.iteration}"
            )
        if not strategy_paths:
            raise RuntimeError(
                f"no strategy sample files collected for player {player} in iter {bundle.iteration}"
            )
        if len(adv_paths) == 1:
            merged_adv_path = adv_paths[0]
            merged_adv_samples = int(info.get("advantage_samples_total", 0))
        else:
            merged_adv_path = adv_paths[0].parent / f"samples_adv_p{player}_iter{bundle.iteration:03d}.bin"
            merged_adv_samples = merge_sample_files(
                adv_paths,
                merged_adv_path,
                expected_magic=ADVANTAGE_SAMPLE_MAGIC,
            )
            for path in adv_paths:
                safe_delete(path)
        if len(strategy_paths) == 1:
            merged_strategy_path = strategy_paths[0]
            merged_strategy_samples = int(info.get("strategy_samples_total", 0))
        else:
            merged_strategy_path = (
                strategy_paths[0].parent / f"samples_strategy_from_p{player}_iter{bundle.iteration:03d}.bin"
            )
            merged_strategy_samples = merge_sample_files(
                strategy_paths,
                merged_strategy_path,
                expected_magic=STRATEGY_SAMPLE_MAGIC,
            )
            for path in strategy_paths:
                safe_delete(path)
        started_min = float(info.get("started_min", 0.0))
        finished_max = float(info.get("finished_max", started_min))
        seat_elapsed = max(0.0, finished_max - started_min)
        traverse_total += seat_elapsed
        runtime_summaries = info.get("runtime_summaries", [])
        runtime_summary = {}
        if isinstance(runtime_summaries, list) and runtime_summaries:
            runtime_summary = {
                "chunk_count": len(runtime_summaries),
                "worker_load_total_sec": float(
                    sum(float(item.get("worker_load_total_sec", 0.0)) for item in runtime_summaries)
                ),
                "worker_run_total_sec": float(
                    sum(float(item.get("worker_run_total_sec", 0.0)) for item in runtime_summaries)
                ),
                "worker_traversal_imbalance_max": float(
                    max(float(item.get("worker_traversal_imbalance", 0.0)) for item in runtime_summaries)
                ),
                "worker_traversal_imbalance_avg": float(
                    sum(float(item.get("worker_traversal_imbalance", 0.0)) for item in runtime_summaries)
                    / float(len(runtime_summaries))
                ),
            }
        players.append(
            {
                "player": int(player),
                "advantage_sample_path": merged_adv_path,
                "strategy_sample_path": merged_strategy_path,
                "advantage_samples": int(merged_adv_samples),
                "strategy_samples": int(merged_strategy_samples),
                "time_sec": float(seat_elapsed),
                "runtime_profile": {"summary": runtime_summary} if runtime_summary else {},
                "seat_chunk_count": int(len(adv_paths)),
            }
        )
    players.sort(key=lambda item: int(item.get("player", 0)))
    fastest_player: int | None = None
    slowest_player: int | None = None
    seat_time_min_sec = 0.0
    seat_time_max_sec = 0.0
    seat_time_avg_sec = 0.0
    seat_time_ratio = 0.0
    if players:
        fastest = min(
            players,
            key=lambda item: float(item.get("time_sec", 0.0))
            if isinstance(item, dict)
            else float("inf"),
        )
        slowest = max(
            players,
            key=lambda item: float(item.get("time_sec", 0.0))
            if isinstance(item, dict)
            else 0.0,
        )
        times = [float(item.get("time_sec", 0.0)) for item in players if isinstance(item, dict)]
        if times:
            seat_time_min_sec = min(times)
            seat_time_max_sec = max(times)
            seat_time_avg_sec = sum(times) / float(len(times))
            seat_time_ratio = (
                seat_time_max_sec / seat_time_min_sec if seat_time_min_sec > 1e-9 else 0.0
            )
        if isinstance(fastest, dict):
            fastest_player = int(fastest.get("player", -1))
        if isinstance(slowest, dict):
            slowest_player = int(slowest.get("player", -1))
    return {
        "iteration": bundle.iteration,
        "players": players,
        "traverse_total": traverse_total,
        "traverse_wall": max(0.0, time.perf_counter() - wall_started),
        "seat_time_min_sec": seat_time_min_sec,
        "seat_time_max_sec": seat_time_max_sec,
        "seat_time_avg_sec": seat_time_avg_sec,
        "seat_time_ratio_max_to_min": seat_time_ratio,
        "fastest_player": fastest_player,
        "slowest_player": slowest_player,
    }


def default_evaluator_panel_path(work_dir: Path, args: argparse.Namespace) -> Path:
    panel_dir = work_dir / "eval_panel"
    panel_name = (
        "panel_strategy.bin" if args.evaluator_panel_type == "strategy" else "panel_advantage.bin"
    )
    return panel_dir / panel_name


def maybe_bootstrap_evaluator_panel(
    *,
    work_dir: Path,
    args: argparse.Namespace,
    merged_adv_path: Path,
    merged_strategy_path: Path | None,
) -> None:
    if args.evaluator_backend != "deep_cfr_panel":
        return
    if args.evaluator_panel_file is not None:
        return
    panel_path = default_evaluator_panel_path(work_dir, args)
    if panel_path.exists():
        return
    source = merged_adv_path if args.evaluator_panel_type == "advantage" else merged_strategy_path
    if source is None or not source.exists():
        return
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, panel_path)
    log(f"[deep-cfr] bootstrapped fixed evaluator panel: {panel_path} (from {source.name})")


def resolve_evaluator_model_state(work_dir: Path, args: argparse.Namespace) -> tuple[Path, str]:
    models_dir = work_dir / "models"
    strategy_state = models_dir / "strategy_shared.pt"
    advantage_state = models_dir / "advantage_shared.pt"

    if args.evaluator_model == "strategy":
        if not strategy_state.exists():
            raise FileNotFoundError(
                f"--evaluator-model strategy selected but checkpoint not found: {strategy_state}"
            )
        return strategy_state, "strategy"
    if args.evaluator_model == "advantage":
        if not advantage_state.exists():
            raise FileNotFoundError(
                f"--evaluator-model advantage selected but checkpoint not found: {advantage_state}"
            )
        return advantage_state, "advantage"

    if strategy_state.exists():
        return strategy_state, "strategy"
    if advantage_state.exists():
        return advantage_state, "advantage"
    raise FileNotFoundError(
        "no evaluator checkpoint found for auto mode "
        f"(checked {strategy_state} and {advantage_state})"
    )


def run_multiplayer_league_diagnostic(
    *,
    iteration: int,
    repo_root: Path,
    work_dir: Path,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    eval_script = repo_root / "eval" / "run_league.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"missing multiplayer eval harness: {eval_script}")
    reports_dir = work_dir / "league_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"league_iter{iteration:03d}.json"
    results_path = reports_dir / f"league_iter{iteration:03d}.csv"
    backend = str(args.evaluator_backend)
    if backend == "deep_cfr_panel":
        panel_file = (
            args.evaluator_panel_file.resolve()
            if args.evaluator_panel_file is not None
            else default_evaluator_panel_path(work_dir, args)
        )
        if not panel_file.exists():
            raise FileNotFoundError(
                "deep_cfr_panel backend requires a panel file. "
                f"Expected {panel_file}. You can pass --evaluator-panel-file or let the run bootstrap one."
            )
        model_state, model_family = resolve_evaluator_model_state(work_dir, args)
        command = [
            sys.executable,
            "-m",
            "eval.run_league",
            "--eval-backend",
            "deep_cfr_panel",
            "--players",
            str(args.num_players),
            "--seed",
            str(seed),
            "--hero-mode",
            "argmax",
            "--deep-cfr-model-state",
            str(model_state),
            "--deep-cfr-samples",
            str(panel_file),
            "--deep-cfr-sample-type",
            str(args.evaluator_panel_type),
            "--deep-cfr-max-samples",
            str(args.evaluator_panel_max_samples),
            "--deep-cfr-batch-size",
            str(args.evaluator_panel_batch_size),
            "--deep-cfr-device",
            str(args.evaluator_panel_device),
            "--deep-cfr-label",
            f"iter{iteration:03d}",
            "--deep-cfr-required-seats",
            str(args.evaluator_required_seats),
            "--deep-cfr-min-seat-samples",
            str(args.evaluator_min_seat_samples),
            "--deep-cfr-max-dominant-seat-fraction",
            str(args.evaluator_max_dominant_seat_fraction),
            "--report-json",
            str(report_path),
            "--results",
            str(results_path),
            "--quiet",
        ]
        if args.num_players == 6:
            command.extend(["--match-format", "6max"])
    else:
        model_family = "policy_worker"
        command = [
            sys.executable,
            "-m",
            "eval.run_league",
            "--hands",
            str(args.league_hands),
            "--players",
            str(args.num_players),
            "--seed",
            str(seed),
            "--opponents",
            str(args.league_opponents),
            "--report-json",
            str(report_path),
            "--results",
            str(results_path),
            "--quiet",
        ]
        if args.num_players == 6:
            command.extend(["--match-format", "6max"])
    started = time.perf_counter()
    try:
        run_streaming_command(command, cwd=repo_root)
    except Exception as exc:
        return {
            "status": "error",
            "backend": backend,
            "error": str(exc),
            "elapsed_sec": time.perf_counter() - started,
            "hands": int(args.league_hands),
            "seed": int(seed),
            "players": int(args.num_players),
            "opponents": str(args.league_opponents),
            "report_json": str(report_path),
            "results_csv": str(results_path),
        }
    elapsed = time.perf_counter() - started
    report = read_json(report_path, {})
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
    evaluator_identity = (
        diagnostics.get("evaluator_identity", {}) if isinstance(diagnostics, dict) else {}
    )
    deep_panel = diagnostics.get("deep_cfr_panel", {}) if isinstance(diagnostics, dict) else {}
    panel_fingerprint = (
        deep_panel.get("fingerprint", {}) if isinstance(deep_panel, dict) else {}
    )
    panel_coverage = deep_panel.get("coverage", {}) if isinstance(deep_panel, dict) else {}
    panel_hash = panel_fingerprint.get("hash") if isinstance(panel_fingerprint, dict) else None
    panel_metrics = deep_panel.get("metrics", {}) if isinstance(deep_panel, dict) else {}

    metric_kind = "bb_per_100"
    raw_metric_value = 0.0
    net_metric_value = 0.0
    if backend == "league":
        raw = summary.get("raw", {}) if isinstance(summary, dict) else {}
        net = summary.get("net", {}) if isinstance(summary, dict) else {}
        raw_metric_value = float(raw.get("bb_per_100", 0.0)) if isinstance(raw, dict) else 0.0
        net_metric_value = float(net.get("bb_per_100", 0.0)) if isinstance(net, dict) else 0.0
        if abs(raw_metric_value) < 1e-12 and abs(net_metric_value) < 1e-12:
            aggregate = read_json(repo_root / "eval" / "runs" / "aggregate_summary.json", {})
            overall = aggregate.get("overall", {}) if isinstance(aggregate, dict) else {}
            if isinstance(overall, dict):
                overall_mean = overall.get("mean_bb100")
                if isinstance(overall_mean, (int, float)):
                    raw_metric_value = float(overall_mean)
                    net_metric_value = float(overall_mean)
    else:
        metric_kind = "panel_score"

    panel_score = float(deep_panel.get("score", 0.0)) if isinstance(deep_panel, dict) else 0.0
    panel_score_ci_low = (
        float(deep_panel.get("score_ci_low", panel_score)) if isinstance(deep_panel, dict) else panel_score
    )
    panel_score_ci_high = (
        float(deep_panel.get("score_ci_high", panel_score)) if isinstance(deep_panel, dict) else panel_score
    )
    return {
        "status": "ok",
        "backend": backend,
        "elapsed_sec": elapsed,
        "hands": int(args.league_hands),
        "seed": int(seed),
        "players": int(args.num_players),
        "opponents": str(args.league_opponents),
        "model_family": model_family,
        "metric_kind": metric_kind,
        "raw_metric_value": raw_metric_value,
        "net_metric_value": net_metric_value,
        "panel_score": panel_score,
        "panel_score_ci_low": panel_score_ci_low,
        "panel_score_ci_high": panel_score_ci_high,
        "panel_top1_match_rate": float(panel_metrics.get("top1_match_rate", 0.0))
        if isinstance(panel_metrics, dict)
        else 0.0,
        "panel_cross_entropy": float(panel_metrics.get("cross_entropy", 0.0))
        if isinstance(panel_metrics, dict)
        else 0.0,
        "panel_masked_l1": float(panel_metrics.get("masked_l1", 0.0))
        if isinstance(panel_metrics, dict)
        else 0.0,
        "evaluator_identity": evaluator_identity,
        "panel_fingerprint_hash": str(panel_hash) if panel_hash is not None else None,
        "panel_coverage": panel_coverage,
        "report_json": str(report_path),
        "results_csv": str(results_path),
    }


def panel_ranking_key(entry: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -float(entry.get("top1_match_rate", 0.0)),
        float(entry.get("cross_entropy", 1e9)),
        float(entry.get("masked_l1", 1e9)),
    )


def panel_coverage_passed(entry: dict[str, Any], *, require: bool) -> bool:
    if not require:
        return True
    coverage = entry.get("coverage", {})
    if not isinstance(coverage, dict):
        return False
    return str(coverage.get("status", "failed")) == "ok"


def is_panel_challenger_better(
    challenger: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    top1_min_improvement: float,
) -> bool:
    if incumbent is None:
        return True
    ch_top1 = float(challenger.get("top1_match_rate", 0.0))
    in_top1 = float(incumbent.get("top1_match_rate", 0.0))
    if ch_top1 > in_top1 + float(top1_min_improvement):
        return True
    if ch_top1 + 1e-12 < in_top1:
        return False
    ch_ce = float(challenger.get("cross_entropy", 1e9))
    in_ce = float(incumbent.get("cross_entropy", 1e9))
    if ch_ce + 1e-12 < in_ce:
        return True
    if ch_ce > in_ce + 1e-12:
        return False
    ch_l1 = float(challenger.get("masked_l1", 1e9))
    in_l1 = float(incumbent.get("masked_l1", 1e9))
    return ch_l1 + 1e-12 < in_l1


def run_checkpoint_panel_eval(
    *,
    repo_root: Path,
    model_state: Path,
    panel_file: Path,
    panel_label: str,
    seed: int,
    args: argparse.Namespace,
    report_path: Path,
) -> dict[str, Any]:
    panel_script = repo_root / "eval" / "deep_cfr_panel.py"
    command = [
        sys.executable,
        str(panel_script),
        "--model-state",
        str(model_state),
        "--samples",
        str(panel_file),
        "--sample-type",
        "strategy",
        "--max-samples",
        str(args.evaluator_panel_max_samples),
        "--batch-size",
        str(args.evaluator_panel_batch_size),
        "--seed",
        str(seed),
        "--device",
        str(args.evaluator_panel_device),
        "--label",
        panel_label,
        "--required-seats",
        str(args.evaluator_required_seats),
        "--min-seat-samples",
        str(args.evaluator_min_seat_samples),
        "--max-dominant-seat-fraction",
        str(args.evaluator_max_dominant_seat_fraction),
        "--report-json",
        str(report_path),
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        tail = proc.stderr.strip() or proc.stdout.strip()
        return {
            "status": "error",
            "error": tail,
            "elapsed_sec": elapsed,
            "report_json": str(report_path),
            "panel_file": str(panel_file),
        }
    if not report_path.exists():
        return {
            "status": "error",
            "error": f"missing checkpoint panel report: {report_path}",
            "elapsed_sec": elapsed,
            "report_json": str(report_path),
            "panel_file": str(panel_file),
        }
    panel_result = read_json(report_path, {})
    metrics = panel_result.get("metrics", {}) if isinstance(panel_result, dict) else {}
    coverage = panel_result.get("coverage", {}) if isinstance(panel_result, dict) else {}
    fingerprint = panel_result.get("fingerprint", {}) if isinstance(panel_result, dict) else {}
    return {
        "status": str(panel_result.get("status", "failed")),
        "elapsed_sec": elapsed,
        "report_json": str(report_path),
        "panel_file": str(panel_file),
        "top1_match_rate": float(metrics.get("top1_match_rate", 0.0)) if isinstance(metrics, dict) else 0.0,
        "cross_entropy": float(metrics.get("cross_entropy", 0.0)) if isinstance(metrics, dict) else 0.0,
        "masked_l1": float(metrics.get("masked_l1", 0.0)) if isinstance(metrics, dict) else 0.0,
        "score": float(panel_result.get("score", 0.0)) if isinstance(panel_result, dict) else 0.0,
        "score_ci_low": float(panel_result.get("score_ci_low", 0.0)) if isinstance(panel_result, dict) else 0.0,
        "score_ci_high": float(panel_result.get("score_ci_high", 0.0)) if isinstance(panel_result, dict) else 0.0,
        "coverage": coverage,
        "fingerprint_hash": str(fingerprint.get("hash", "")) if isinstance(fingerprint, dict) else "",
        "seat_breakdown": panel_result.get("seat_breakdown", {}) if isinstance(panel_result, dict) else {},
    }


def load_checkpoint_eval_state(path: Path) -> dict[str, Any]:
    default = {"version": 1, "best_screen": None, "best_holdout": None, "promotions": []}
    payload = read_json(path, default)
    if not isinstance(payload, dict):
        return default
    payload.setdefault("version", 1)
    payload.setdefault("best_screen", None)
    payload.setdefault("best_holdout", None)
    payload.setdefault("promotions", [])
    return payload


def save_checkpoint_eval_state(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _read_ring_eval_bb_per_100(report: Any) -> float | None:
    if not isinstance(report, dict):
        return None
    status = str(report.get("status", "ok")).lower()
    if status not in {"ok", "success"}:
        return None
    value = report.get("model_bb_per_100")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _build_ring_eval_history_from_metrics(metrics_iterations: list[Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for entry in metrics_iterations:
        if not isinstance(entry, dict):
            continue
        iter_raw = entry.get("iteration")
        try:
            iteration = int(iter_raw)
        except (TypeError, ValueError):
            continue
        ring_tag = entry.get("ring_eval_tag")
        if not isinstance(ring_tag, dict):
            fallback = entry.get("ring_eval")
            ring_tag = fallback if isinstance(fallback, dict) else None
        policy_tracking = entry.get("ring_policy_tracking")
        if not isinstance(policy_tracking, dict):
            policy_tracking = {}
        composite = policy_tracking.get("composite_bb_per_100")
        history.append(
            {
                "iteration": iteration,
                "timestamp": entry.get("timestamp"),
                "tag_status": (ring_tag.get("status") if isinstance(ring_tag, dict) else None),
                "tag_bb_per_100": _read_ring_eval_bb_per_100(ring_tag),
                "composite_bb_per_100": (
                    float(composite) if isinstance(composite, (int, float)) else None
                ),
            }
        )
    return history


def load_ring_promotion_state(
    path: Path,
    metrics_iterations: list[Any],
    *,
    window_size: int,
    promotion_margin_bb_per_100: float,
    resume: bool,
) -> dict[str, Any]:
    derived_history = _build_ring_eval_history_from_metrics(metrics_iterations)
    payload: dict[str, Any]
    if resume:
        payload = read_json(path, {})
    else:
        payload = {}

    file_history = payload.get("history")
    history = file_history if isinstance(file_history, list) else []
    if len(derived_history) > len(history):
        history = derived_history

    promotions = payload.get("promotions")
    promotions = promotions if isinstance(promotions, list) else []

    best_raw = payload.get("best_composite_bb_per_100")
    best_composite = float(best_raw) if isinstance(best_raw, (int, float)) else None
    best_iter_raw = payload.get("best_iteration")
    best_iteration = int(best_iter_raw) if isinstance(best_iter_raw, int) else None

    if best_composite is None:
        for item in history:
            if not isinstance(item, dict):
                continue
            value = item.get("composite_bb_per_100")
            if not isinstance(value, (int, float)):
                continue
            candidate = float(value)
            if best_composite is None or candidate > best_composite:
                best_composite = candidate
                iter_value = item.get("iteration")
                best_iteration = int(iter_value) if isinstance(iter_value, int) else best_iteration

    return {
        "version": 1,
        "window_size": int(window_size),
        "promotion_margin_bb_per_100": float(promotion_margin_bb_per_100),
        "best_composite_bb_per_100": best_composite,
        "best_iteration": best_iteration,
        "history": history,
        "promotions": promotions,
        "updated_at": now_iso(),
    }


def save_ring_promotion_state(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    write_json(path, payload)


def compute_ring_smoothed_value(
    history: list[dict[str, Any]],
    field: str,
    window_size: int,
) -> float | None:
    if window_size <= 0:
        return None
    values: list[float] = []
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        value = item.get(field)
        if not isinstance(value, (int, float)):
            continue
        values.append(float(value))
        if len(values) >= window_size:
            break
    if len(values) < window_size:
        return None
    return float(sum(values) / len(values))


def run_shared_multiseat_pipeline(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    script_dir = paths["script_dir"]
    repo_root = paths["repo_root"]
    work_dir = paths["work_dir"]
    cluster_dir = paths["cluster_dir"]
    rust_binary = paths["rust_binary"]
    ring_eval_binary = paths["ring_eval_binary"]
    train_script = script_dir / "train.py"
    if args.evaluator_panel_file is not None:
        args.evaluator_panel_file = args.evaluator_panel_file.resolve()

    if not train_script.exists():
        raise FileNotFoundError(f"missing train script: {train_script}")
    if not cluster_dir.exists():
        raise FileNotFoundError(f"cluster directory not found: {cluster_dir}")
    if not rust_binary.exists():
        raise FileNotFoundError(
            f"Rust binary not found: {rust_binary}\n"
            f"Build it with: cargo build --release -p deep_cfr (from solver/)"
        )
    if args.ring_eval_every > 0 and not ring_eval_binary.exists():
        raise FileNotFoundError(
            f"Ring eval binary not found: {ring_eval_binary}\n"
            f"Build it with: cargo build --release -p deep_cfr --bin ring_game_eval (from solver/)"
        )

    if not args.resume:
        log(color_bold("[init] --resume not set: clearing existing work_dir for fresh start..."))
        reset_work_dir(work_dir)

    models_dir = work_dir / "models"
    samples_dir = work_dir / "samples"
    buffers_dir = work_dir / "buffers"
    traversal_logs_dir = work_dir / "traversal_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    buffers_dir.mkdir(parents=True, exist_ok=True)
    traversal_logs_dir.mkdir(parents=True, exist_ok=True)

    advantage_onnx = models_dir / "advantage_shared.onnx"
    advantage_onnx_fp16 = models_dir / "advantage_shared.fp16.onnx"
    advantage_state = models_dir / "advantage_shared.pt"
    advantage_buffer = buffers_dir / "advantage_shared.pkl"
    strategy_onnx = models_dir / "strategy_shared.onnx"
    strategy_state = models_dir / "strategy_shared.pt"
    strategy_buffer = buffers_dir / "strategy_shared.pkl"
    strategy_best_ring_onnx = models_dir / "strategy_shared_best_ring.onnx"
    strategy_best_ring_state = models_dir / "strategy_shared_best_ring.pt"
    advantage_best_ring_onnx = models_dir / "advantage_shared_best_ring.onnx"
    advantage_best_ring_state = models_dir / "advantage_shared_best_ring.pt"
    model_checkpoints_dir = work_dir / "checkpoints"
    ring_promotion_window_size = 5
    ring_promotion_margin_bb_per_100 = 10.0
    ring_promotion_state_path = work_dir / "best_ring_promotion.json"
    checkpoint_eval_dir = work_dir / "checkpoint_eval"
    checkpoint_eval_reports_dir = checkpoint_eval_dir / "reports"
    checkpoint_eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_eval_reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_eval_state_path = checkpoint_eval_dir / "best_checkpoint.json"
    checkpoint_eval_state = (
        load_checkpoint_eval_state(checkpoint_eval_state_path)
        if args.resume
        else {"version": 1, "best_screen": None, "best_holdout": None, "promotions": []}
    )
    best_screen_entry = (
        checkpoint_eval_state.get("best_screen")
        if isinstance(checkpoint_eval_state.get("best_screen"), dict)
        else None
    )
    best_holdout_entry = (
        checkpoint_eval_state.get("best_holdout")
        if isinstance(checkpoint_eval_state.get("best_holdout"), dict)
        else None
    )
    if args.checkpoint_eval_every > 0:
        save_checkpoint_eval_state(checkpoint_eval_state_path, checkpoint_eval_state)

    state_path = work_dir / "state.json"
    metrics_path = work_dir / "metrics.json"
    state = read_json(
        state_path,
        {
            "version": 2,
            "completed_iterations": 0,
            "updated_at": now_iso(),
        },
    )
    completed_before = int(state.get("completed_iterations", 0))
    if not args.resume:
        completed_before = 0
        state = {
            "version": 2,
            "completed_iterations": 0,
            "updated_at": now_iso(),
        }

    metrics = read_json(
        metrics_path,
        {
            "version": 2,
            "created_at": now_iso(),
            "config": {},
            "iterations": [],
        },
    )
    if not args.resume:
        metrics = {
            "version": 2,
            "created_at": now_iso(),
            "config": {},
            "iterations": [],
        }
    metrics["config"] = {
        "mode": "shared_multiseat",
        "iterations": args.iterations,
        "traversals": args.traversals,
        "training_steps": args.training_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout_p": args.dropout_p,
        "hidden_dim": args.hidden_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "max_sample_reuse_per_iter": args.max_sample_reuse_per_iter,
        "adv_huber_delta": args.adv_huber_delta,
        "buffer_size": args.buffer_size,
        "buffer_disk_dir": str(args.buffer_disk_dir) if args.buffer_disk_dir is not None else None,
        "cluster_dir": str(cluster_dir),
        "work_dir": str(work_dir),
        "rust_binary": str(rust_binary),
        "ring_eval_binary": str(ring_eval_binary),
        "deck_samples": args.deck_samples,
        "num_players": args.num_players,
        "starting_stack": args.starting_stack,
        "small_blind": args.small_blind,
        "big_blind": args.big_blind,
        "seed": args.seed,
        "log_every": args.log_every,
        "device": args.device,
        "onnx_opset": args.onnx_opset,
        "strategy_every": args.strategy_every,
        "diagnostic_every": args.diagnostic_every,
        "league_eval_every": args.league_eval_every,
        "league_hands": args.league_hands,
        "league_opponents": args.league_opponents,
        "ring_eval_every": args.ring_eval_every,
        "ring_eval_hands": args.ring_eval_hands,
        "ring_promotion_window_size": ring_promotion_window_size,
        "ring_promotion_margin_bb_per_100": ring_promotion_margin_bb_per_100,
        "ring_promotion_state_path": str(ring_promotion_state_path),
        "model_checkpoint_every": args.model_checkpoint_every,
        "evaluator_backend": args.evaluator_backend,
        "evaluator_model": args.evaluator_model,
        "evaluator_panel_file": str(args.evaluator_panel_file) if args.evaluator_panel_file else None,
        "evaluator_panel_type": args.evaluator_panel_type,
        "evaluator_panel_max_samples": args.evaluator_panel_max_samples,
        "evaluator_panel_batch_size": args.evaluator_panel_batch_size,
        "evaluator_panel_device": args.evaluator_panel_device,
        "evaluator_required_seats": args.evaluator_required_seats,
        "evaluator_min_seat_samples": args.evaluator_min_seat_samples,
        "evaluator_max_dominant_seat_fraction": args.evaluator_max_dominant_seat_fraction,
        "checkpoint_eval_every": args.checkpoint_eval_every,
        "checkpoint_eval_screen_panel_file": (
            str(args.checkpoint_eval_screen_panel_file) if args.checkpoint_eval_screen_panel_file else None
        ),
        "checkpoint_eval_holdout_panel_file": (
            str(args.checkpoint_eval_holdout_panel_file) if args.checkpoint_eval_holdout_panel_file else None
        ),
        "checkpoint_eval_top1_improvement": args.checkpoint_eval_top1_improvement,
        "checkpoint_eval_require_coverage_pass": bool(args.checkpoint_eval_require_coverage_pass),
        "traversal_progress_batch": args.traversal_progress_batch,
        "traversal_workers_requested": args.traversal_workers,
        "traversal_seat_chunks": args.traversal_seat_chunks,
        "traversal_consolidate_processes": bool(args.traversal_consolidate_processes),
        "traversal_gpu_batch": bool(args.traversal_gpu_batch),
        "traversal_gpu_batch_size": int(args.traversal_gpu_batch_size),
        "traversal_gpu_batch_timeout_us": int(args.traversal_gpu_batch_timeout_us),
        "traversal_gpu_batch_queue_capacity": int(args.traversal_gpu_batch_queue_capacity),
        "traversal_gpu_batch_cpu_only": bool(args.traversal_gpu_batch_cpu_only),
        "traversal_gpu_batch_no_tf32": bool(args.traversal_gpu_batch_no_tf32),
        "traversal_gpu_device_id": int(args.traversal_gpu_device_id),
        "traversal_fp16_onnx": bool(args.traversal_fp16_onnx),
        "traversal_cpu_affinity": str(args.traversal_cpu_affinity),
        "buffer_save_every": args.buffer_save_every,
    }
    metrics_iterations = metrics.get("iterations", [])
    if not isinstance(metrics_iterations, list):
        metrics_iterations = []
    metrics["iterations"] = metrics_iterations
    ring_promotion_state = load_ring_promotion_state(
        ring_promotion_state_path,
        metrics_iterations,
        window_size=ring_promotion_window_size,
        promotion_margin_bb_per_100=ring_promotion_margin_bb_per_100,
        resume=bool(args.resume),
    )
    ring_promotion_history = ring_promotion_state.get("history")
    if not isinstance(ring_promotion_history, list):
        ring_promotion_history = []
    ring_promotion_state["history"] = ring_promotion_history
    ring_promotion_promotions = ring_promotion_state.get("promotions")
    if not isinstance(ring_promotion_promotions, list):
        ring_promotion_promotions = []
    ring_promotion_state["promotions"] = ring_promotion_promotions
    best_ring_composite_raw = ring_promotion_state.get("best_composite_bb_per_100")
    best_ring_composite = (
        float(best_ring_composite_raw) if isinstance(best_ring_composite_raw, (int, float)) else None
    )
    best_ring_iteration_raw = ring_promotion_state.get("best_iteration")
    best_ring_iteration = int(best_ring_iteration_raw) if isinstance(best_ring_iteration_raw, int) else None
    if args.ring_eval_every > 0:
        save_ring_promotion_state(ring_promotion_state_path, ring_promotion_state)

    reservoir_estimate = estimate_reservoir_size(metrics_iterations, args.buffer_size)
    prev_advantage_loss: float | None = None
    for entry in reversed(metrics_iterations):
        if not isinstance(entry, dict):
            continue
        loss_obj = entry.get("loss")
        if not isinstance(loss_obj, dict):
            continue
        value = loss_obj.get("advantage")
        if isinstance(value, (int, float)):
            prev_advantage_loss = float(value)
            break

    if args.resume:
        bootstrap_shared_advantage_artifacts(
            shared_onnx=advantage_onnx,
            shared_state=advantage_state,
            shared_buffer=advantage_buffer,
            legacy_shared_onnx=models_dir / "advantage.onnx",
            legacy_shared_state=models_dir / "advantage.pt",
            legacy_shared_buffer=buffers_dir / "advantage.pkl",
            split_p0_onnx=models_dir / "advantage_p0.onnx",
            split_p1_onnx=models_dir / "advantage_p1.onnx",
            split_p0_state=models_dir / "advantage_p0.pt",
            split_p1_state=models_dir / "advantage_p1.pt",
            split_p0_buffer=buffers_dir / "advantage_p0.pkl",
            split_p1_buffer=buffers_dir / "advantage_p1.pkl",
        )

    if not advantage_onnx.exists():
        if args.resume and completed_before > 0:
            raise FileNotFoundError(f"--resume requested but missing model: {advantage_onnx}")
        log("[init] generating initial random shared advantage ONNX model...")
        train_init_model(train_script, script_dir, advantage_onnx, advantage_state, args)
    else:
        log(f"[init] using existing shared advantage model: {advantage_onnx}")

    device = resolve_device(args.device)
    log(f"[deep-cfr] training device: {device}")
    metrics["config"]["device_resolved"] = str(device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "unknown"
        log(f"[deep-cfr] cuda device: {gpu_name}")
        metrics["config"]["cuda_device_name"] = str(gpu_name)

    advantage_context = initialize_training_context(
        label="advantage_shared",
        network_type="advantage",
        state_path=advantage_state,
        buffer_path=advantage_buffer,
        onnx_path=advantage_onnx,
        args=args,
        device=device,
    )
    strategy_context: NetworkTrainingContext | None = None
    if args.strategy_every > 0:
        strategy_context = initialize_training_context(
            label="strategy_shared",
            network_type="strategy",
            state_path=strategy_state,
            buffer_path=strategy_buffer,
            onnx_path=strategy_onnx,
            args=args,
            device=device,
        )

    write_json(metrics_path, metrics)
    bb_stack = args.starting_stack / max(1, args.big_blind)
    log("========================================")
    log("  Deep CFR Training - Shared Multi-Seat")
    log(f"  iterations: {args.iterations} | traversals/iter/player: {args.traversals}")
    log(f"  players: {args.num_players} | game: {bb_stack:.1f}bb (sb={args.small_blind} bb={args.big_blind})")
    log(
        "  training: "
        f"steps<= {args.training_steps}, batch={args.batch_size}, hidden={args.hidden_dim}, "
        f"bottleneck={args.bottleneck_dim}, dropout={args.dropout_p:.2f}, "
        f"wd={args.weight_decay:g}, huber={args.adv_huber_delta:g}, reuse_cap={args.max_sample_reuse_per_iter:.2f}x"
    )
    log(
        "  diagnostics: "
        f"league_every={args.league_eval_every}, league_hands={args.league_hands}, "
        f"ring_eval_every={args.ring_eval_every}, ring_eval_hands={args.ring_eval_hands}, "
        f"model_checkpoint_every={args.model_checkpoint_every}, "
        f"opponents={args.league_opponents}, backend={args.evaluator_backend}, "
        f"panel_type={args.evaluator_panel_type}, required_seats={args.evaluator_required_seats}, "
        f"min_seat_samples={args.evaluator_min_seat_samples}, "
        f"max_dom_frac={args.evaluator_max_dominant_seat_fraction:.2f}"
    )
    log(
        "  ring promotion: "
        f"window={ring_promotion_window_size}, margin={ring_promotion_margin_bb_per_100:.1f} bb/100, "
        f"state={ring_promotion_state_path}"
    )
    log(
        "  checkpoint eval: "
        f"every={args.checkpoint_eval_every}, "
        f"screen={args.checkpoint_eval_screen_panel_file}, "
        f"holdout={args.checkpoint_eval_holdout_panel_file}, "
        f"require_coverage={bool(args.checkpoint_eval_require_coverage_pass)}"
    )
    log(
        "  traversal runtime: "
        f"workers={args.traversal_workers} progress_batch={args.traversal_progress_batch} "
        f"seat_chunks={args.traversal_seat_chunks} consolidate={bool(args.traversal_consolidate_processes)} "
        f"gpu_batch={bool(args.traversal_gpu_batch)} batch_size={args.traversal_gpu_batch_size} "
        f"timeout_us={args.traversal_gpu_batch_timeout_us} fp16={bool(args.traversal_fp16_onnx)} "
        f"affinity='{args.traversal_cpu_affinity}'"
    )
    log("  proof gates: training/deep_cfr/PROOF_GATES_6MAX.md")
    log(f"  buffer_size: {args.buffer_size:,}")
    if args.buffer_disk_dir is not None:
        log(f"  buffer_disk_dir: {args.buffer_disk_dir}")
    log(f"  work_dir: {work_dir}")
    log("========================================")

    if completed_before >= args.iterations:
        log(
            f"[resume] state already at iteration {completed_before}, target {args.iterations}; nothing to do."
        )
        return

    run_started = time.perf_counter()
    first_iteration = completed_before + 1
    active_bundle: IterationTraversalBundle | None = None
    try:
        for iteration in range(first_iteration, args.iterations + 1):
            iter_started = time.perf_counter()
            iter_label = f"{iteration:03d}/{args.iterations}"
            elapsed_before = time.perf_counter() - run_started
            finished_before = max(0, iteration - first_iteration)
            avg_before = (elapsed_before / finished_before) if finished_before > 0 else 0.0
            eta_hint = avg_before * max(0, args.iterations - iteration + 1)
            print_iter_header(iteration, args.iterations, eta_hint)
            traversal_onnx_path = (
                advantage_onnx_fp16
                if bool(args.traversal_fp16_onnx) and advantage_onnx_fp16.exists()
                else advantage_onnx
            )

            active_bundle = start_iteration_traversals_async_shared(
                iteration=iteration,
                rust_binary=rust_binary,
                repo_root=repo_root,
                shared_onnx_path=traversal_onnx_path,
                models_dir=models_dir,
                samples_dir=samples_dir,
                traversal_logs_dir=traversal_logs_dir,
                args=args,
                cluster_dir=cluster_dir,
            )
            if active_bundle.consolidated:
                log(
                    f"[iter {iter_label}] launched consolidated traversal process "
                    f"(players={active_bundle.consolidated_players})"
                )
            else:
                log(
                    f"[iter {iter_label}] launched {len(active_bundle.tasks)} traversal tasks "
                    f"({args.num_players} seats x {args.traversal_seat_chunks} chunk(s))"
                )
            traversal_report = wait_for_iteration_traversals_shared(active_bundle)
            active_bundle = None

            player_reports = traversal_report.get("players", [])
            if not isinstance(player_reports, list) or len(player_reports) != args.num_players:
                raise RuntimeError(
                    f"expected {args.num_players} traversal reports, got {len(player_reports) if isinstance(player_reports, list) else 'invalid'}"
                )
            total_adv_samples = 0
            total_strategy_samples = 0
            adv_paths: list[Path] = []
            strategy_paths: list[Path] = []
            per_player_times: dict[str, float] = {}
            for player_report in player_reports:
                if not isinstance(player_report, dict):
                    continue
                player_idx = int(player_report.get("player", -1))
                adv_count = int(player_report.get("advantage_samples", 0))
                strat_count = int(player_report.get("strategy_samples", 0))
                elapsed = float(player_report.get("time_sec", 0.0))
                adv_path = player_report.get("advantage_sample_path")
                strategy_path = player_report.get("strategy_sample_path")
                if not isinstance(adv_path, Path) or not isinstance(strategy_path, Path):
                    raise RuntimeError(f"invalid sample path payload for player {player_idx}")
                adv_paths.append(adv_path)
                strategy_paths.append(strategy_path)
                total_adv_samples += adv_count
                total_strategy_samples += strat_count
                per_player_times[f"p{player_idx}"] = elapsed
                log(
                    f"  traversals: p{player_idx} adv={adv_count} strat={strat_count} in {elapsed:.1f}s"
                )
                runtime_profile = player_report.get("runtime_profile", {})
                if isinstance(runtime_profile, dict):
                    runtime_summary = runtime_profile.get("summary", {})
                    if isinstance(runtime_summary, dict) and runtime_summary:
                        chunk_count = int(player_report.get("seat_chunk_count", 1))
                        chunk_label = runtime_summary.get("job_chunk")
                        if chunk_label is None:
                            chunk_label = f"seat_chunks={chunk_count}"
                        imbalance = runtime_summary.get(
                            "worker_traversal_imbalance",
                            runtime_summary.get("worker_traversal_imbalance_max", 0.0),
                        )
                        log(
                            "    runtime: "
                            f"chunk={chunk_label} "
                            f"load_total={float(runtime_summary.get('worker_load_total_sec', 0.0)):.3f}s "
                            f"run_total={float(runtime_summary.get('worker_run_total_sec', 0.0)):.3f}s "
                            f"worker_imbalance={float(imbalance):.3f}"
                        )

            merged_adv_path = samples_dir / f"samples_adv_iter{iteration:03d}.bin"
            merged_adv_samples = merge_sample_files(
                adv_paths,
                merged_adv_path,
                expected_magic=ADVANTAGE_SAMPLE_MAGIC,
            )
            for path in adv_paths:
                safe_delete(path)
            log(
                f"[iter {iter_label}] merged advantage samples: {merged_adv_samples} -> {merged_adv_path}"
            )

            merged_strategy_samples = 0
            merged_strategy_path: Path | None = None
            if strategy_context is not None:
                merged_strategy_path = samples_dir / f"samples_strategy_iter{iteration:03d}.bin"
                merged_strategy_samples = merge_sample_files(
                    strategy_paths,
                    merged_strategy_path,
                    expected_magic=STRATEGY_SAMPLE_MAGIC,
                )
                log(
                    f"[iter {iter_label}] merged strategy samples: {merged_strategy_samples} -> {merged_strategy_path}"
                )
            maybe_bootstrap_evaluator_panel(
                work_dir=work_dir,
                args=args,
                merged_adv_path=merged_adv_path,
                merged_strategy_path=merged_strategy_path,
            )
            for path in strategy_paths:
                safe_delete(path)

            train_started = time.perf_counter()
            advantage_seed = args.seed ^ (iteration * 0xF135_7AEA)
            advantage_train = train_network_in_process(
                context=advantage_context,
                samples_path=merged_adv_path,
                args=args,
                seed=advantage_seed,
                device=device,
            )
            safe_delete(merged_adv_path)
            train_time = time.perf_counter() - train_started
            advantage_loss = advantage_train.final_loss

            strategy_loss: float | None = None
            strategy_trained = False
            strategy_train: TrainingRunStats | None = None
            if strategy_context is not None and merged_strategy_path is not None:
                strategy_seed_ingest = args.seed ^ (iteration * 0x0B7F_4D95)
                ingest_samples_into_context(
                    context=strategy_context,
                    samples_path=merged_strategy_path,
                    seed=strategy_seed_ingest,
                )
                if args.strategy_every > 0 and iteration % args.strategy_every == 0:
                    strategy_seed = args.seed ^ (iteration * 0x5F35_9ACD)
                    strategy_train = train_network_in_process(
                        context=strategy_context,
                        samples_path=merged_strategy_path,
                        args=args,
                        seed=strategy_seed,
                        device=device,
                        ingest_samples=False,
                    )
                    strategy_loss = strategy_train.final_loss
                    strategy_trained = True
            if merged_strategy_path is not None:
                safe_delete(merged_strategy_path)

            diagnostics_report: dict[str, Any] | None = None
            if args.league_eval_every > 0 and iteration % args.league_eval_every == 0:
                log(f"[iter {iter_label}] running multiplayer league diagnostic...")
                league_seed = args.seed ^ (iteration * 0xC2B2_AE35)
                diagnostics_report = {
                    "league_6max": run_multiplayer_league_diagnostic(
                        iteration=iteration,
                        repo_root=repo_root,
                        work_dir=work_dir,
                        args=args,
                        seed=league_seed,
                    )
                }

            checkpoint_eval_report: dict[str, Any] | None = None
            if args.checkpoint_eval_every > 0 and iteration % args.checkpoint_eval_every == 0:
                checkpoint_eval_report = {
                    "status": "started",
                    "iteration": iteration,
                }
                if strategy_context is None:
                    checkpoint_eval_report = {
                        "status": "skipped",
                        "iteration": iteration,
                        "reason": "strategy model is disabled (strategy_every=0)",
                    }
                elif not strategy_state.exists():
                    checkpoint_eval_report = {
                        "status": "error",
                        "iteration": iteration,
                        "reason": f"strategy checkpoint missing: {strategy_state}",
                    }
                elif args.checkpoint_eval_screen_panel_file is None:
                    checkpoint_eval_report = {
                        "status": "error",
                        "iteration": iteration,
                        "reason": "checkpoint screen panel file is not configured",
                    }
                else:
                    log(f"[iter {iter_label}] running checkpoint screen panel evaluation...")
                    screen_report_path = checkpoint_eval_reports_dir / f"screen_iter{iteration:03d}.json"
                    screen_seed = int(args.seed ^ (iteration * 0x4A39_B70D))
                    screen_eval = run_checkpoint_panel_eval(
                        repo_root=repo_root,
                        model_state=strategy_state,
                        panel_file=args.checkpoint_eval_screen_panel_file,
                        panel_label=f"screen_iter{iteration:03d}",
                        seed=screen_seed,
                        args=args,
                        report_path=screen_report_path,
                    )
                    checkpoint_eval_report = {
                        "status": "ok",
                        "iteration": iteration,
                        "screen": screen_eval,
                    }
                    if str(screen_eval.get("status")) != "ok":
                        checkpoint_eval_report["status"] = "error"
                    elif not panel_coverage_passed(
                        screen_eval,
                        require=bool(args.checkpoint_eval_require_coverage_pass),
                    ):
                        checkpoint_eval_report["status"] = "failed_coverage"
                    else:
                        incumbent_screen_result = (
                            best_screen_entry.get("result") if isinstance(best_screen_entry, dict) else None
                        )
                        if is_panel_challenger_better(
                            screen_eval,
                            incumbent_screen_result if isinstance(incumbent_screen_result, dict) else None,
                            top1_min_improvement=float(args.checkpoint_eval_top1_improvement),
                        ):
                            checkpoint_eval_report["screen_improved"] = True
                            best_screen_entry = {
                                "iteration": iteration,
                                "ranking": list(panel_ranking_key(screen_eval)),
                                "result": screen_eval,
                                "model_state": str(strategy_state),
                                "model_onnx": str(strategy_onnx),
                            }
                            checkpoint_eval_state["best_screen"] = best_screen_entry
                            save_checkpoint_eval_state(checkpoint_eval_state_path, checkpoint_eval_state)
                            log(
                                "[checkpoint-eval] new screen best: "
                                f"iter={iteration} top1={float(screen_eval.get('top1_match_rate', 0.0)):.4f} "
                                f"ce={float(screen_eval.get('cross_entropy', 0.0)):.6f} "
                                f"l1={float(screen_eval.get('masked_l1', 0.0)):.6f}"
                            )

                            if args.checkpoint_eval_holdout_panel_file is not None:
                                log(f"[iter {iter_label}] running checkpoint holdout confirmation...")
                                holdout_report_path = checkpoint_eval_reports_dir / f"holdout_iter{iteration:03d}.json"
                                holdout_seed = int(args.seed ^ (iteration * 0x2F11_9D93))
                                holdout_eval = run_checkpoint_panel_eval(
                                    repo_root=repo_root,
                                    model_state=strategy_state,
                                    panel_file=args.checkpoint_eval_holdout_panel_file,
                                    panel_label=f"holdout_iter{iteration:03d}",
                                    seed=holdout_seed,
                                    args=args,
                                    report_path=holdout_report_path,
                                )
                                checkpoint_eval_report["holdout"] = holdout_eval
                                holdout_ok = str(holdout_eval.get("status")) == "ok" and panel_coverage_passed(
                                    holdout_eval,
                                    require=bool(args.checkpoint_eval_require_coverage_pass),
                                )
                                incumbent_holdout_result = (
                                    best_holdout_entry.get("result")
                                    if isinstance(best_holdout_entry, dict)
                                    else None
                                )
                                if holdout_ok and is_panel_challenger_better(
                                    holdout_eval,
                                    incumbent_holdout_result
                                    if isinstance(incumbent_holdout_result, dict)
                                    else None,
                                    top1_min_improvement=float(args.checkpoint_eval_top1_improvement),
                                ):
                                    strategy_best_state = models_dir / "strategy_shared_best_confirmed.pt"
                                    strategy_best_onnx = models_dir / "strategy_shared_best_confirmed.onnx"
                                    advantage_best_state = models_dir / "advantage_shared_best_confirmed.pt"
                                    advantage_best_onnx = models_dir / "advantage_shared_best_confirmed.onnx"
                                    copy_status = {
                                        "strategy_state": safe_copy_file(strategy_state, strategy_best_state),
                                        "strategy_onnx": safe_copy_file(strategy_onnx, strategy_best_onnx),
                                        "advantage_state": safe_copy_file(advantage_state, advantage_best_state),
                                        "advantage_onnx": safe_copy_file(advantage_onnx, advantage_best_onnx),
                                    }
                                    promotion_entry = {
                                        "iteration": iteration,
                                        "timestamp": now_iso(),
                                        "screen": screen_eval,
                                        "holdout": holdout_eval,
                                        "artifacts": {
                                            "strategy_state": str(strategy_best_state),
                                            "strategy_onnx": str(strategy_best_onnx),
                                            "advantage_state": str(advantage_best_state),
                                            "advantage_onnx": str(advantage_best_onnx),
                                        },
                                        "copy_status": copy_status,
                                    }
                                    promotions = checkpoint_eval_state.get("promotions", [])
                                    if not isinstance(promotions, list):
                                        promotions = []
                                    promotions.append(promotion_entry)
                                    checkpoint_eval_state["promotions"] = promotions
                                    best_holdout_entry = {
                                        "iteration": iteration,
                                        "ranking": list(panel_ranking_key(holdout_eval)),
                                        "result": holdout_eval,
                                        "artifacts": promotion_entry["artifacts"],
                                    }
                                    checkpoint_eval_state["best_holdout"] = best_holdout_entry
                                    save_checkpoint_eval_state(checkpoint_eval_state_path, checkpoint_eval_state)
                                    checkpoint_eval_report["promotion"] = promotion_entry
                                    log(
                                        "[checkpoint-eval] promoted new best-confirmed checkpoint: "
                                        f"iter={iteration} top1={float(holdout_eval.get('top1_match_rate', 0.0)):.4f} "
                                        f"ce={float(holdout_eval.get('cross_entropy', 0.0)):.6f} "
                                        f"l1={float(holdout_eval.get('masked_l1', 0.0)):.6f}"
                                    )
                                else:
                                    checkpoint_eval_report["promotion"] = {
                                        "status": "skipped",
                                        "reason": "holdout did not beat incumbent or failed coverage",
                                    }
                            else:
                                checkpoint_eval_report["holdout"] = {
                                    "status": "skipped",
                                    "reason": "holdout panel file is not configured",
                                }
                        else:
                            checkpoint_eval_report["screen_improved"] = False

            ring_eval_report: dict[str, Any] | None = None
            ring_eval_tag_report: dict[str, Any] | None = None
            ring_policy_tracking_report: dict[str, Any] | None = None
            if args.ring_eval_every > 0 and iteration % args.ring_eval_every == 0:
                if args.num_players != 6:
                    ring_eval_tag_report = {
                        "status": "skipped",
                        "iteration": iteration,
                        "reason": "ring eval currently supports 6-player runs only",
                    }
                elif strategy_context is None:
                    ring_eval_tag_report = {
                        "status": "skipped",
                        "iteration": iteration,
                        "reason": "strategy model is disabled (strategy_every=0)",
                    }
                elif not strategy_onnx.exists():
                    ring_eval_tag_report = {
                        "status": "error",
                        "iteration": iteration,
                        "reason": f"strategy ONNX missing for ring eval: {strategy_onnx}",
                    }
                else:
                    ring_eval_seed_tag = int(args.seed ^ (iteration * 0x8B2F_A4C9))
                    try:
                        log(f"[iter {iter_label}] running 6-max ring game eval (model vs 5 TAG)...")
                        ring_eval_tag_report = run_ring_game_eval(
                            ring_eval_binary=ring_eval_binary,
                            repo_root=repo_root,
                            cluster_dir=cluster_dir,
                            args=args,
                            model_path=strategy_onnx,
                            seed=ring_eval_seed_tag,
                            opponent="tag",
                        )
                        ring_eval_tag_report["iteration"] = iteration
                        log(
                            "[ring-eval TAG] "
                            f"model={float(ring_eval_tag_report.get('model_bb_per_100', 0.0)):+.2f} bb/100 "
                            f"({int(ring_eval_tag_report.get('hands', args.ring_eval_hands))} hands, "
                            f"{float(ring_eval_tag_report.get('elapsed_sec', 0.0)):.1f}s)"
                        )
                    except Exception as exc:
                        ring_eval_tag_report = {
                            "status": "error",
                            "iteration": iteration,
                            "seed": ring_eval_seed_tag,
                            "opponent": "tag",
                            "error": str(exc),
                        }
                        log(f"[ring-eval TAG] error: {exc}")

                    tag_bb = _read_ring_eval_bb_per_100(ring_eval_tag_report)
                    history_entry = {
                        "iteration": iteration,
                        "timestamp": now_iso(),
                        "tag_status": (
                            ring_eval_tag_report.get("status")
                            if isinstance(ring_eval_tag_report, dict)
                            else "missing"
                        ),
                        "tag_bb_per_100": tag_bb,
                        "composite_bb_per_100": None,
                    }
                    ring_promotion_history.append(history_entry)

                    smoothed_tag = compute_ring_smoothed_value(
                        ring_promotion_history,
                        "tag_bb_per_100",
                        ring_promotion_window_size,
                    )
                    composite_value = smoothed_tag
                    history_entry["composite_bb_per_100"] = composite_value
                    tag_count = sum(
                        1
                        for item in ring_promotion_history
                        if isinstance(item, dict) and isinstance(item.get("tag_bb_per_100"), (int, float))
                    )
                    ring_policy_tracking_report = {
                        "status": "ok",
                        "window_size": ring_promotion_window_size,
                        "margin_bb_per_100": ring_promotion_margin_bb_per_100,
                        "sample_count_tag": tag_count,
                        "smoothed_tag_bb_per_100": smoothed_tag,
                        "composite_bb_per_100": composite_value,
                        "best_composite_bb_per_100_before": best_ring_composite,
                        "best_iteration_before": best_ring_iteration,
                        "promoted": False,
                    }

                    if composite_value is None:
                        ring_policy_tracking_report["status"] = "insufficient_history"
                        log(
                            "[policy-track] waiting for smoothing window: "
                            f"tag={tag_count}/{ring_promotion_window_size}"
                        )
                    else:
                        ring_policy_tracking_report["best_composite_bb_per_100_before"] = best_ring_composite
                        ring_policy_tracking_report["best_iteration_before"] = best_ring_iteration
                        log(
                            "[policy-track] "
                            f"smoothed TAG={smoothed_tag:+.2f} bb/100 "
                            f"composite={composite_value:+.2f} bb/100 "
                            f"(window={ring_promotion_window_size})"
                        )
                        should_promote = (
                            best_ring_composite is None
                            or composite_value > (best_ring_composite + ring_promotion_margin_bb_per_100)
                        )
                        if should_promote:
                            copy_status = {
                                "strategy_state": safe_copy_file(strategy_state, strategy_best_ring_state),
                                "strategy_onnx": safe_copy_file(strategy_onnx, strategy_best_ring_onnx),
                                "advantage_state": safe_copy_file(advantage_state, advantage_best_ring_state),
                                "advantage_onnx": safe_copy_file(advantage_onnx, advantage_best_ring_onnx),
                            }
                            copied_ok = all(copy_status.values())
                            promotion_entry = {
                                "iteration": iteration,
                                "timestamp": now_iso(),
                                "composite_bb_per_100": composite_value,
                                "smoothed_tag_bb_per_100": smoothed_tag,
                                "tag_eval": ring_eval_tag_report,
                                "best_composite_before": best_ring_composite,
                                "best_iteration_before": best_ring_iteration,
                                "artifacts": {
                                    "strategy_state": str(strategy_best_ring_state),
                                    "strategy_onnx": str(strategy_best_ring_onnx),
                                    "advantage_state": str(advantage_best_ring_state),
                                    "advantage_onnx": str(advantage_best_ring_onnx),
                                },
                                "copy_status": copy_status,
                            }
                            ring_policy_tracking_report["promotion"] = promotion_entry
                            if copied_ok:
                                best_ring_composite = composite_value
                                best_ring_iteration = iteration
                                ring_promotion_promotions.append(promotion_entry)
                                ring_policy_tracking_report["promoted"] = True
                                ring_policy_tracking_report["best_composite_bb_per_100_after"] = best_ring_composite
                                ring_policy_tracking_report["best_iteration_after"] = best_ring_iteration
                                log(
                                    "[policy-track] PROMOTED new best-ring checkpoint: "
                                    f"iter={iteration} composite={composite_value:+.2f} bb/100"
                                )
                            else:
                                ring_policy_tracking_report["status"] = "copy_failed"
                                ring_policy_tracking_report["promoted"] = False
                                log(
                                    "[policy-track] promotion candidate found but artifact copy failed; "
                                    "best-ring checkpoint not updated"
                                )
                        else:
                            ring_policy_tracking_report["best_composite_bb_per_100_after"] = best_ring_composite
                            ring_policy_tracking_report["best_iteration_after"] = best_ring_iteration
                            log(
                                "[policy-track] no promotion: "
                                f"best={best_ring_composite:+.2f} bb/100 @ iter "
                                f"{best_ring_iteration if best_ring_iteration is not None else 'n/a'}"
                            )

                    ring_promotion_state["best_composite_bb_per_100"] = best_ring_composite
                    ring_promotion_state["best_iteration"] = best_ring_iteration
                    ring_promotion_state["history"] = ring_promotion_history
                    ring_promotion_state["promotions"] = ring_promotion_promotions
                    save_ring_promotion_state(ring_promotion_state_path, ring_promotion_state)

                ring_eval_report = ring_eval_tag_report

            model_checkpoint_report: dict[str, Any] | None = None
            if args.model_checkpoint_every > 0 and iteration % args.model_checkpoint_every == 0:
                model_checkpoints_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_suffix = f"iter{iteration:03d}"
                model_checkpoint_paths = {
                    "strategy_state": model_checkpoints_dir / f"strategy_{checkpoint_suffix}.pt",
                    "strategy_onnx": model_checkpoints_dir / f"strategy_{checkpoint_suffix}.onnx",
                    "advantage_state": model_checkpoints_dir / f"advantage_{checkpoint_suffix}.pt",
                    "advantage_onnx": model_checkpoints_dir / f"advantage_{checkpoint_suffix}.onnx",
                }
                copy_status = {
                    key: safe_copy_file(src, dst)
                    for key, src, dst in [
                        ("strategy_state", strategy_state, model_checkpoint_paths["strategy_state"]),
                        ("strategy_onnx", strategy_onnx, model_checkpoint_paths["strategy_onnx"]),
                        ("advantage_state", advantage_state, model_checkpoint_paths["advantage_state"]),
                        ("advantage_onnx", advantage_onnx, model_checkpoint_paths["advantage_onnx"]),
                    ]
                }
                copied_count = sum(1 for value in copy_status.values() if value)
                total_count = len(copy_status)
                model_checkpoint_report = {
                    "status": "ok" if copied_count == total_count else "partial",
                    "iteration": iteration,
                    "every": int(args.model_checkpoint_every),
                    "paths": {key: str(path) for key, path in model_checkpoint_paths.items()},
                    "copy_status": copy_status,
                }
                checkpoint_size_total = sum(
                    path.stat().st_size
                    for path in model_checkpoint_paths.values()
                    if path.exists()
                )
                log(
                    "[checkpoint] saved model snapshot: "
                    f"iter={iteration} files={copied_count}/{total_count} "
                    f"size={format_bytes(checkpoint_size_total)}"
                )

            if iteration % args.buffer_save_every == 0:
                log(f"[iter {iter_label}] checkpointing in-memory reservoirs...")
                save_reservoir_checkpoint(advantage_context)
                if strategy_context is not None:
                    save_reservoir_checkpoint(strategy_context)

            iter_total = time.perf_counter() - iter_started
            traverse_sum_seat = float(traversal_report.get("traverse_total", 0.0))
            traverse_wall = float(traversal_report.get("traverse_wall", traverse_sum_seat))
            seat_time_min = float(traversal_report.get("seat_time_min_sec", 0.0))
            seat_time_max = float(traversal_report.get("seat_time_max_sec", 0.0))
            seat_time_avg = float(traversal_report.get("seat_time_avg_sec", 0.0))
            seat_time_ratio = float(traversal_report.get("seat_time_ratio_max_to_min", 0.0))
            fastest_player = traversal_report.get("fastest_player")
            slowest_player = traversal_report.get("slowest_player")
            traverse_time = traverse_wall
            elapsed = time.perf_counter() - run_started
            done_this_run = iteration - completed_before
            avg_iter = elapsed / max(1, done_this_run)
            remaining = args.iterations - iteration
            eta = avg_iter * remaining
            merged_total_samples = int(merged_adv_samples + merged_strategy_samples)
            throughput_wall = (
                float(merged_total_samples) / traverse_wall if traverse_wall > 1e-9 else 0.0
            )
            throughput_seat = (
                float(merged_total_samples) / traverse_sum_seat if traverse_sum_seat > 1e-9 else 0.0
            )

            reservoir_estimate = min(args.buffer_size, reservoir_estimate + merged_adv_samples)
            samples_disk = directory_size_bytes(samples_dir)
            models_disk = directory_size_bytes(models_dir)
            buffers_disk = directory_size_bytes(buffers_dir)
            work_disk = directory_size_bytes(work_dir)
            loss_delta = (
                (advantage_loss - prev_advantage_loss)
                if isinstance(advantage_loss, float) and isinstance(prev_advantage_loss, float)
                else None
            )
            if isinstance(advantage_loss, float):
                prev_advantage_loss = advantage_loss

            print_iter_summary(
                iter_total=iter_total,
                traverse_time=traverse_time,
                train_time=train_time,
                train_steps_ran=advantage_train.steps_ran,
                training_steps_target=advantage_train.steps_target,
                early_stopped=advantage_train.early_stopped,
                advantage_loss=advantage_loss,
                loss_delta=loss_delta,
                reservoir_estimate=reservoir_estimate,
                buffer_size=args.buffer_size,
                work_disk=work_disk,
                samples_disk=samples_disk,
                elapsed=elapsed,
                eta=eta,
            )
            log(
                "  artifacts: "
                f"models={format_bytes(models_disk)} buffers={format_bytes(buffers_disk)} samples={format_bytes(samples_disk)}"
            )
            log(
                "  traversal profile: "
                f"wall={traverse_wall:.1f}s seat_sum={traverse_sum_seat:.1f}s "
                f"samples={merged_total_samples:,} throughput={throughput_wall:,.1f}/s wall "
                f"({throughput_seat:,.1f}/s seat-sum)"
            )
            if seat_time_max > 1e-9:
                fastest_label = (
                    f"p{int(fastest_player)}" if isinstance(fastest_player, int) and fastest_player >= 0 else "n/a"
                )
                slowest_label = (
                    f"p{int(slowest_player)}" if isinstance(slowest_player, int) and slowest_player >= 0 else "n/a"
                )
                log(
                    "  traversal balance: "
                    f"fastest={fastest_label} {seat_time_min:.1f}s avg={seat_time_avg:.1f}s "
                    f"slowest={slowest_label} {seat_time_max:.1f}s ratio={seat_time_ratio:.3f}"
                )
            log("----------------------------------------")

            metrics_iterations.append(
                {
                    "iteration": iteration,
                    "timestamp": now_iso(),
                    "samples": {
                        "advantage_total_raw": total_adv_samples,
                        "advantage_merged": merged_adv_samples,
                        "strategy_total_raw": total_strategy_samples,
                        "strategy_merged": merged_strategy_samples,
                        "per_player": [
                            {
                                "player": int(item.get("player", -1)),
                                "advantage": int(item.get("advantage_samples", 0)),
                                "strategy": int(item.get("strategy_samples", 0)),
                                "runtime_profile": item.get("runtime_profile", {}),
                            }
                            for item in player_reports
                            if isinstance(item, dict)
                        ],
                    },
                    "loss": {
                        "advantage": advantage_loss,
                        "advantage_delta": loss_delta,
                        "strategy": strategy_loss,
                    },
                    "training": {
                        "advantage_steps_ran": advantage_train.steps_ran,
                        "advantage_steps_target": advantage_train.steps_target,
                        "advantage_early_stopped": advantage_train.early_stopped,
                        "strategy_steps_ran": strategy_train.steps_ran if strategy_train else None,
                        "strategy_steps_target": strategy_train.steps_target if strategy_train else None,
                        "strategy_early_stopped": strategy_train.early_stopped if strategy_train else None,
                    },
                    "times_sec": {
                        "traverse_total": traverse_sum_seat,
                        "traverse_wall": traverse_wall,
                        "traverse_sum_seat_time": traverse_sum_seat,
                        "train_total": train_time,
                        "iteration_total": iter_total,
                        "elapsed_in_run": elapsed,
                        "eta": eta,
                        "per_player_traverse": per_player_times,
                        "traverse_seat_min": seat_time_min,
                        "traverse_seat_avg": seat_time_avg,
                        "traverse_seat_max": seat_time_max,
                        "traverse_seat_ratio_max_to_min": seat_time_ratio,
                    },
                    "throughput": {
                        "merged_samples_total": merged_total_samples,
                        "samples_per_sec_traverse_wall": throughput_wall,
                        "samples_per_sec_traverse_seat_sum": throughput_seat,
                    },
                    "artifact_bytes": {
                        "work_dir": work_disk,
                        "models_dir": models_disk,
                        "buffers_dir": buffers_disk,
                        "samples_dir": samples_disk,
                    },
                    "reservoir_estimate": reservoir_estimate,
                    "strategy_snapshot": strategy_trained,
                    "diagnostics": diagnostics_report,
                    "checkpoint_eval": checkpoint_eval_report,
                    "ring_eval": ring_eval_report,
                    "ring_eval_tag": ring_eval_tag_report,
                    "ring_policy_tracking": ring_policy_tracking_report,
                    "model_checkpoint": model_checkpoint_report,
                    "paths": {
                        "advantage_onnx": str(advantage_onnx),
                        "strategy_onnx": str(strategy_onnx),
                    },
                }
            )
            metrics["updated_at"] = now_iso()
            write_json(metrics_path, metrics)

            state = {
                "version": 2,
                "mode": "shared_multiseat",
                "completed_iterations": iteration,
                "updated_at": now_iso(),
                "advantage_onnx": str(advantage_onnx),
                "advantage_state": str(advantage_state),
                "strategy_onnx": str(strategy_onnx),
                "strategy_state": str(strategy_state),
                "work_dir": str(work_dir),
                "num_players": int(args.num_players),
            }
            write_json(state_path, state)
    except KeyboardInterrupt:
        terminate_traversal_bundle(active_bundle)
        log("[deep-cfr] interrupted by user (Ctrl+C).")
        log("[deep-cfr] saving in-memory reservoirs before exit...")
        save_reservoir_checkpoint(advantage_context)
        if strategy_context is not None:
            save_reservoir_checkpoint(strategy_context)
        log("[deep-cfr] partial progress is saved in state.json and metrics.json.")
        raise
    except Exception as exc:
        terminate_traversal_bundle(active_bundle)
        log(f"[deep-cfr] fatal error: {exc}")
        log("[deep-cfr] attempting best-effort reservoir checkpoint before exit...")
        try:
            save_reservoir_checkpoint(advantage_context)
            if strategy_context is not None:
                save_reservoir_checkpoint(strategy_context)
            log("[deep-cfr] reservoir checkpoint saved during fatal-exit handling.")
        except Exception as save_exc:
            log(f"[warn] fatal-exit reservoir checkpoint failed: {save_exc}")
        raise

    log("[deep-cfr] final reservoir checkpoint...")
    save_reservoir_checkpoint(advantage_context)
    if strategy_context is not None:
        save_reservoir_checkpoint(strategy_context)

    total_runtime = time.perf_counter() - run_started
    final_work_disk = directory_size_bytes(work_dir)
    log(
        f"[deep-cfr] completed iterations {first_iteration}..{args.iterations} "
        f"in {format_duration(total_runtime)}"
    )
    log(
        f"[deep-cfr] reservoir estimate: {reservoir_estimate:,}/{args.buffer_size:,} "
        f"({(100.0 * reservoir_estimate / max(1, args.buffer_size)):.1f}%)"
    )
    if isinstance(best_holdout_entry, dict):
        artifacts = best_holdout_entry.get("artifacts", {})
        best_strategy_state = artifacts.get("strategy_state") if isinstance(artifacts, dict) else None
        log(
            "[checkpoint-eval] best confirmed checkpoint: "
            f"iter={best_holdout_entry.get('iteration')} "
            f"strategy_state={best_strategy_state}"
        )
    elif isinstance(best_screen_entry, dict):
        log(
            "[checkpoint-eval] screen-best tracked (holdout confirm not yet promoted): "
            f"iter={best_screen_entry.get('iteration')}"
        )
    log(f"[checkpoint-eval] state: {checkpoint_eval_state_path}")
    if best_ring_composite is not None:
        log(
            "[policy-track] best ring-promoted checkpoint: "
            f"iter={best_ring_iteration} composite={best_ring_composite:+.2f} bb/100"
        )
    else:
        log("[policy-track] best ring-promoted checkpoint: none")
    log(f"[policy-track] state: {ring_promotion_state_path}")
    log(f"[deep-cfr] work_dir usage: {format_bytes(final_work_disk)}")
    log(f"[deep-cfr] state: {state_path}")
    log(f"[deep-cfr] metrics: {metrics_path}")


def main() -> None:
    configure_warning_filters()
    if colorama_init is not None:
        colorama_init(autoreset=True)
    args = parse_args()
    validate_args(args)
    paths = resolve_paths(args)

    if args.num_players > 2:
        run_shared_multiseat_pipeline(args, paths)
        return

    script_dir = paths["script_dir"]
    repo_root = paths["repo_root"]
    work_dir = paths["work_dir"]
    cluster_dir = paths["cluster_dir"]
    rust_binary = paths["rust_binary"]
    measure_binary = paths["measure_binary"]
    h2h_binary = paths["h2h_binary"]
    train_script = script_dir / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"missing train script: {train_script}")
    if not cluster_dir.exists():
        raise FileNotFoundError(f"cluster directory not found: {cluster_dir}")
    if not rust_binary.exists():
        raise FileNotFoundError(
            f"Rust binary not found: {rust_binary}\n"
            f"Build it with: cargo build --release -p deep_cfr (from solver/)"
        )
    if (args.exploitability_every > 0 or args.diagnostic_every > 0) and not measure_binary.exists():
        raise FileNotFoundError(
            f"Exploitability binary not found: {measure_binary}\n"
            f"Build it with: cargo build --release -p deep_cfr --bin measure_exploitability (from solver/)"
        )
    if args.diagnostic_every > 0 and not h2h_binary.exists():
        raise FileNotFoundError(
            f"Head-to-head binary not found: {h2h_binary}\n"
            f"Build it with: cargo build --release -p deep_cfr --bin head_to_head (from solver/)"
        )

    if not args.resume:
        log(color_bold("[init] --resume not set: clearing existing work_dir for fresh start..."))
        reset_work_dir(work_dir)

    models_dir = work_dir / "models"
    samples_dir = work_dir / "samples"
    buffers_dir = work_dir / "buffers"
    traversal_logs_dir = work_dir / "traversal_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    buffers_dir.mkdir(parents=True, exist_ok=True)
    traversal_logs_dir.mkdir(parents=True, exist_ok=True)

    advantage_p0_onnx = models_dir / "advantage_p0.onnx"
    advantage_p1_onnx = models_dir / "advantage_p1.onnx"
    strategy_onnx = models_dir / "strategy.onnx"
    strategy_latest_onnx = models_dir / "strategy_latest.onnx"
    strategy_best_candidate_onnx = models_dir / "strategy_best_candidate.onnx"
    strategy_best_confirmed_onnx = models_dir / "strategy_best_confirmed.onnx"
    advantage_p0_state = models_dir / "advantage_p0.pt"
    advantage_p1_state = models_dir / "advantage_p1.pt"
    strategy_state = models_dir / "strategy.pt"
    strategy_latest_state = models_dir / "strategy_latest.pt"
    strategy_best_candidate_state = models_dir / "strategy_best_candidate.pt"
    strategy_best_confirmed_state = models_dir / "strategy_best_confirmed.pt"
    advantage_p0_buffer = buffers_dir / "advantage_p0.pkl"
    advantage_p1_buffer = buffers_dir / "advantage_p1.pkl"
    strategy_buffer = buffers_dir / "strategy.pkl"
    legacy_advantage_onnx = models_dir / "advantage.onnx"
    legacy_advantage_state = models_dir / "advantage.pt"
    legacy_advantage_buffer = buffers_dir / "advantage.pkl"
    state_path = work_dir / "state.json"
    metrics_path = work_dir / "metrics.json"

    state = read_json(
        state_path,
        {
            "version": 1,
            "completed_iterations": 0,
            "updated_at": now_iso(),
        },
    )
    completed_before = int(state.get("completed_iterations", 0))
    if not args.resume:
        completed_before = 0
        state = {
            "version": 1,
            "completed_iterations": 0,
            "updated_at": now_iso(),
        }

    metrics = read_json(
        metrics_path,
        {
            "version": 1,
            "created_at": now_iso(),
            "config": {},
            "iterations": [],
        },
    )
    if not args.resume:
        metrics = {
            "version": 1,
            "created_at": now_iso(),
            "config": {},
            "iterations": [],
        }

    metrics["config"] = {
        "iterations": args.iterations,
        "traversals": args.traversals,
        "training_steps": args.training_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout_p": args.dropout_p,
        "hidden_dim": args.hidden_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "max_sample_reuse_per_iter": args.max_sample_reuse_per_iter,
        "adv_huber_delta": args.adv_huber_delta,
        "buffer_size": args.buffer_size,
        "buffer_disk_dir": str(args.buffer_disk_dir) if args.buffer_disk_dir is not None else None,
        "cluster_dir": str(cluster_dir),
        "work_dir": str(work_dir),
        "rust_binary": str(rust_binary),
        "measure_binary": str(measure_binary),
        "h2h_binary": str(h2h_binary),
        "deck_samples": args.deck_samples,
        "num_players": args.num_players,
        "starting_stack": args.starting_stack,
        "small_blind": args.small_blind,
        "big_blind": args.big_blind,
        "seed": args.seed,
        "log_every": args.log_every,
        "device": args.device,
        "onnx_opset": args.onnx_opset,
        "enable_early_stop": args.enable_early_stop,
        "early_stop_min_steps": args.early_stop_min_steps,
        "early_stop_threshold": args.early_stop_threshold,
        "strategy_every": args.strategy_every,
        "exploitability_every": args.exploitability_every,
        "diagnostic_every": args.diagnostic_every,
        "h2h_only_diagnostics": args.h2h_only_diagnostics,
        "h2h_hands": args.h2h_hands,
        "exploit_samples": args.exploit_samples,
        "exploit_fit_samples": args.exploit_fit_samples,
        "exploit_eval_samples": args.exploit_eval_samples,
        "fixed_eval_seed": args.fixed_eval_seed,
        "traversal_progress_batch": args.traversal_progress_batch,
        "traversal_seat_chunks": args.traversal_seat_chunks,
        "traversal_consolidate_processes": bool(args.traversal_consolidate_processes),
        "traversal_gpu_batch": bool(args.traversal_gpu_batch),
        "traversal_gpu_batch_size": int(args.traversal_gpu_batch_size),
        "traversal_gpu_batch_timeout_us": int(args.traversal_gpu_batch_timeout_us),
        "traversal_gpu_batch_queue_capacity": int(args.traversal_gpu_batch_queue_capacity),
        "traversal_gpu_batch_cpu_only": bool(args.traversal_gpu_batch_cpu_only),
        "traversal_gpu_batch_no_tf32": bool(args.traversal_gpu_batch_no_tf32),
        "traversal_gpu_device_id": int(args.traversal_gpu_device_id),
        "traversal_fp16_onnx": bool(args.traversal_fp16_onnx),
        "traversal_cpu_affinity": str(args.traversal_cpu_affinity),
        "buffer_save_every": args.buffer_save_every,
        "selection_holdout_seeds": args.selection_holdout_seeds,
        "selection_holdout_hands": args.selection_holdout_hands,
        "selection_random_guardrail": args.selection_random_guardrail,
        "league_eval_every": args.league_eval_every,
        "league_hands": args.league_hands,
        "league_opponents": args.league_opponents,
    }
    metrics_iterations = metrics.get("iterations", [])
    if not isinstance(metrics_iterations, list):
        metrics_iterations = []
    metrics["iterations"] = metrics_iterations
    model_selection_state = metrics.get("model_selection")
    if not isinstance(model_selection_state, dict):
        model_selection_state = {
            "best_candidate": None,
            "best_confirmed": None,
        }
    metrics["model_selection"] = model_selection_state
    reservoir_estimate_p0, reservoir_estimate_p1 = estimate_player_reservoir_sizes(
        metrics_iterations,
        args.buffer_size,
    )

    prev_advantage_loss_p0: float | None = None
    prev_advantage_loss_p1: float | None = None
    for entry in reversed(metrics_iterations):
        if not isinstance(entry, dict):
            continue
        loss_obj = entry.get("loss")
        if not isinstance(loss_obj, dict):
            continue
        raw_loss_p0 = loss_obj.get("advantage_p0")
        raw_loss_p1 = loss_obj.get("advantage_p1")
        if prev_advantage_loss_p0 is None and isinstance(raw_loss_p0, (int, float)):
            prev_advantage_loss_p0 = float(raw_loss_p0)
        if prev_advantage_loss_p1 is None and isinstance(raw_loss_p1, (int, float)):
            prev_advantage_loss_p1 = float(raw_loss_p1)
        if prev_advantage_loss_p0 is not None and prev_advantage_loss_p1 is not None:
            break
    prev_diagnostics = last_diagnostics(metrics_iterations)
    baseline_diagnostics = first_diagnostics(metrics_iterations)
    eval_panel = build_eval_panel_config(args)
    metrics["evaluation_panel"] = {
        "exploit_fit_samples": eval_panel.exploit_fit_samples,
        "exploit_eval_samples": eval_panel.exploit_eval_samples,
        "exploit_strategy_fit_seed": eval_panel.exploit_strategy_fit_seed,
        "exploit_strategy_eval_seed": eval_panel.exploit_strategy_eval_seed,
        "exploit_adv_fit_seed": eval_panel.exploit_adv_fit_seed,
        "exploit_adv_eval_seed": eval_panel.exploit_adv_eval_seed,
        "h2h_strategy_tag_seed": eval_panel.h2h_strategy_tag_seed,
        "h2h_strategy_random_seed": eval_panel.h2h_strategy_random_seed,
        "h2h_strategy_tag_holdout_seeds": list(eval_panel.h2h_strategy_tag_holdout_seeds),
        "selection_holdout_hands": (
            args.selection_holdout_hands if args.selection_holdout_hands > 0 else args.h2h_hands
        ),
    }

    write_json(metrics_path, metrics)

    bb_stack = args.starting_stack / max(1, args.big_blind)
    log("========================================")
    log("  Deep CFR Training - HU NLHE")
    log(f"  iterations: {args.iterations} | traversals/iter: {args.traversals}")
    log(f"  game: {args.num_players}-player {bb_stack:.1f}bb (sb={args.small_blind} bb={args.big_blind})")
    log(
        "  training: "
        f"steps<= {args.training_steps}, batch={args.batch_size}, "
        f"hidden={args.hidden_dim}, bottleneck={args.bottleneck_dim}, "
        f"dropout={args.dropout_p:.2f}, wd={args.weight_decay:g}, "
        f"huber={args.adv_huber_delta:g}, reuse_cap={args.max_sample_reuse_per_iter:.2f}x"
    )
    log(
        "  eval panel: "
        f"fit={eval_panel.exploit_fit_samples} eval={eval_panel.exploit_eval_samples} "
        f"(fixed seeds from base {args.fixed_eval_seed if args.fixed_eval_seed > 0 else args.seed})"
    )
    log(
        "  diagnostics: "
        f"every {args.diagnostic_every} iters | mode={'h2h-only' if args.h2h_only_diagnostics else 'full'}"
    )
    holdout_hands = args.selection_holdout_hands if args.selection_holdout_hands > 0 else args.h2h_hands
    log(
        "  model selection: "
        f"holdout_seeds={args.selection_holdout_seeds}, holdout_hands={holdout_hands}, "
        f"random_guardrail={args.selection_random_guardrail:+.3f} bb/100"
    )
    log("  proof gates: training/deep_cfr/PROOF_GATES.md")
    log(f"  buffer_size: {args.buffer_size:,}")
    if args.buffer_disk_dir is not None:
        log(f"  buffer_disk_dir: {args.buffer_disk_dir}")
    log(f"  work_dir: {work_dir}")
    log("========================================")

    if completed_before >= args.iterations:
        log(
            f"[resume] state already at iteration {completed_before}, target {args.iterations}; nothing to do."
        )
        return

    if args.resume:
        bootstrap_split_advantage_artifacts(
            legacy_onnx=legacy_advantage_onnx,
            legacy_state=legacy_advantage_state,
            legacy_buffer=legacy_advantage_buffer,
            advantage_p0_onnx=advantage_p0_onnx,
            advantage_p1_onnx=advantage_p1_onnx,
            advantage_p0_state=advantage_p0_state,
            advantage_p1_state=advantage_p1_state,
            advantage_p0_buffer=advantage_p0_buffer,
            advantage_p1_buffer=advantage_p1_buffer,
        )

    if not advantage_p0_onnx.exists():
        if args.resume and completed_before > 0:
            raise FileNotFoundError(
                f"--resume requested but missing model: {advantage_p0_onnx}"
            )
        log("[init] generating initial random advantage_p0 ONNX model...")
        train_init_model(train_script, script_dir, advantage_p0_onnx, advantage_p0_state, args)
    else:
        log(f"[init] using existing advantage model: {advantage_p0_onnx}")

    if not advantage_p1_onnx.exists():
        if args.resume and completed_before > 0:
            raise FileNotFoundError(
                f"--resume requested but missing model: {advantage_p1_onnx}"
            )
        log("[init] generating initial random advantage_p1 ONNX model...")
        train_init_model(train_script, script_dir, advantage_p1_onnx, advantage_p1_state, args)
    else:
        log(f"[init] using existing advantage model: {advantage_p1_onnx}")

    device = resolve_device(args.device)
    log(f"[deep-cfr] training device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "unknown"
        log(f"[deep-cfr] cuda device: {gpu_name}")
    if args.enable_early_stop:
        log(
            "[deep-cfr] early stop: enabled "
            f"(min_steps={args.early_stop_min_steps}, threshold={args.early_stop_threshold:.6f})"
        )
    else:
        log("[deep-cfr] early stop: disabled")

    advantage_context_p0 = initialize_training_context(
        label="advantage_p0",
        network_type="advantage",
        state_path=advantage_p0_state,
        buffer_path=advantage_p0_buffer,
        onnx_path=advantage_p0_onnx,
        args=args,
        device=device,
    )
    advantage_context_p1 = initialize_training_context(
        label="advantage_p1",
        network_type="advantage",
        state_path=advantage_p1_state,
        buffer_path=advantage_p1_buffer,
        onnx_path=advantage_p1_onnx,
        args=args,
        device=device,
    )
    strategy_context: NetworkTrainingContext | None = None
    if args.strategy_every > 0:
        strategy_context = initialize_training_context(
            label="strategy",
            network_type="strategy",
            state_path=strategy_state,
            buffer_path=strategy_buffer,
            onnx_path=strategy_onnx,
            args=args,
            device=device,
        )

    run_started = time.perf_counter()
    first_iteration = completed_before + 1
    active_bundle: IterationTraversalBundle | None = None

    try:
        for iteration in range(first_iteration, args.iterations + 1):
            iter_started = time.perf_counter()
            iter_label = f"{iteration:03d}/{args.iterations}"
            elapsed_before = time.perf_counter() - run_started
            finished_before = max(0, iteration - first_iteration)
            avg_before = (elapsed_before / finished_before) if finished_before > 0 else 0.0
            eta_hint = avg_before * max(0, args.iterations - iteration + 1)
            print_iter_header(iteration, args.iterations, eta_hint)

            active_bundle = start_iteration_traversals_async(
                iteration=iteration,
                rust_binary=rust_binary,
                repo_root=repo_root,
                onnx_p0_path=advantage_p0_onnx,
                onnx_p1_path=advantage_p1_onnx,
                models_dir=models_dir,
                samples_dir=samples_dir,
                traversal_logs_dir=traversal_logs_dir,
                args=args,
                cluster_dir=cluster_dir,
            )
            model_copy_names = ", ".join(path.name for path in active_bundle.model_copy_paths)
            log(
                f"[iter {iter_label}] launched traversal workers "
                f"(snapshot copies: {model_copy_names})"
            )
            log(f"[iter {iter_label}] waiting for traversal workers...")
            traversal_report = wait_for_iteration_traversals(active_bundle)
            active_bundle = None

            sample_adv_p0 = traversal_report["sample_adv_p0"]
            sample_adv_p1 = traversal_report["sample_adv_p1"]
            sample_strategy_from_p0 = traversal_report["sample_strategy_from_p0"]
            sample_strategy_from_p1 = traversal_report["sample_strategy_from_p1"]
            if (
                not isinstance(sample_adv_p0, Path)
                or not isinstance(sample_adv_p1, Path)
                or not isinstance(sample_strategy_from_p0, Path)
                or not isinstance(sample_strategy_from_p1, Path)
            ):
                raise RuntimeError("traversal report missing expected sample paths")
            p0_adv_samples = int(traversal_report.get("p0_adv_samples", 0))
            p1_adv_samples = int(traversal_report.get("p1_adv_samples", 0))
            p0_strategy_samples = int(traversal_report.get("p0_strategy_samples", 0))
            p1_strategy_samples = int(traversal_report.get("p1_strategy_samples", 0))
            p0_time = float(traversal_report.get("p0_time", 0.0))
            p1_time = float(traversal_report.get("p1_time", 0.0))
            log(
                "  traversals: "
                f"p0 adv={p0_adv_samples} strat={p0_strategy_samples} in {p0_time:.1f}s | "
                f"p1 adv={p1_adv_samples} strat={p1_strategy_samples} in {p1_time:.1f}s"
            )

            strategy_samples_merged = 0
            sample_strategy_merged: Path | None = None
            if strategy_context is not None:
                sample_strategy_merged = samples_dir / f"samples_strategy_iter{iteration:03d}.bin"
                strategy_samples_merged = merge_sample_files(
                    [sample_strategy_from_p0, sample_strategy_from_p1],
                    sample_strategy_merged,
                    expected_magic=STRATEGY_SAMPLE_MAGIC,
                )
                log(
                    f"[iter {iter_label}] merged strategy samples: "
                    f"{strategy_samples_merged} -> {sample_strategy_merged}"
                )
            safe_delete(sample_strategy_from_p0)
            safe_delete(sample_strategy_from_p1)

            log(f"[iter {iter_label}] training advantage_p0 network...")
            train_started = time.perf_counter()
            train_seed_p0 = args.seed ^ (iteration * 0xF135_7AEA)
            advantage_train_p0 = train_network_in_process(
                context=advantage_context_p0,
                samples_path=sample_adv_p0,
                args=args,
                seed=train_seed_p0,
                device=device,
            )
            safe_delete(sample_adv_p0)
            log(f"[iter {iter_label}] exported advantage_p0.onnx")

            log(f"[iter {iter_label}] training advantage_p1 network...")
            train_seed_p1 = args.seed ^ (iteration * 0xA24B_AED4)
            advantage_train_p1 = train_network_in_process(
                context=advantage_context_p1,
                samples_path=sample_adv_p1,
                args=args,
                seed=train_seed_p1,
                device=device,
            )
            safe_delete(sample_adv_p1)
            log(f"[iter {iter_label}] exported advantage_p1.onnx")
            train_time = time.perf_counter() - train_started
            advantage_loss_p0 = advantage_train_p0.final_loss
            advantage_loss_p1 = advantage_train_p1.final_loss

            strategy_loss: float | None = None
            strategy_trained = False
            strategy_train: TrainingRunStats | None = None
            if strategy_context is not None and sample_strategy_merged is not None:
                strategy_seed_ingest = args.seed ^ (iteration * 0x0B7F_4D95)
                ingest_samples_into_context(
                    context=strategy_context,
                    samples_path=sample_strategy_merged,
                    seed=strategy_seed_ingest,
                )
            if (
                strategy_context is not None
                and sample_strategy_merged is not None
                and args.strategy_every > 0
                and iteration % args.strategy_every == 0
            ):
                log(f"[iter {iter_label}] training strategy snapshot...")
                strategy_seed = args.seed ^ (iteration * 0x5F35_9ACD)
                strategy_train = train_network_in_process(
                    context=strategy_context,
                    samples_path=sample_strategy_merged,
                    args=args,
                    seed=strategy_seed,
                    device=device,
                    ingest_samples=False,
                )
                strategy_loss = strategy_train.final_loss
                strategy_trained = True
                log(f"[iter {iter_label}] exported strategy.onnx")
            if sample_strategy_merged is not None:
                safe_delete(sample_strategy_merged)

            diagnostics_report: dict[str, Any] | None = None
            if args.diagnostic_every > 0 and iteration % args.diagnostic_every == 0:
                diagnostic_mode = "TAG/random h2h only" if args.h2h_only_diagnostics else "exploitability + TAG/random h2h"
                log(f"[iter {iter_label}] running diagnostic suite ({diagnostic_mode})...")
                diagnostics_report = run_diagnostic_suite(
                    iteration=iteration,
                    h2h_binary=h2h_binary,
                    measure_binary=measure_binary,
                    repo_root=repo_root,
                    cluster_dir=cluster_dir,
                    args=args,
                    advantage_p0_onnx=advantage_p0_onnx,
                    advantage_p1_onnx=advantage_p1_onnx,
                    strategy_onnx=strategy_onnx,
                    strategy_trained=strategy_trained,
                    eval_panel=eval_panel,
                    h2h_only=args.h2h_only_diagnostics,
                    previous_diagnostics=prev_diagnostics,
                    baseline_diagnostics=baseline_diagnostics,
                )
                prev_diagnostics = diagnostics_report
                if baseline_diagnostics is None:
                    baseline_diagnostics = diagnostics_report
            elif args.exploitability_every > 0 and iteration % args.exploitability_every == 0:
                # Backward-compatible lightweight mode: exploitability only.
                exploit_report = measure_advantage_exploitability(
                    measure_binary=measure_binary,
                    repo_root=repo_root,
                    onnx_p0_path=advantage_p0_onnx,
                    onnx_p1_path=advantage_p1_onnx,
                    cluster_dir=cluster_dir,
                    args=args,
                    fit_samples=eval_panel.exploit_fit_samples,
                    eval_samples=eval_panel.exploit_eval_samples,
                    fit_seed=eval_panel.exploit_adv_fit_seed,
                    eval_seed=eval_panel.exploit_adv_eval_seed,
                )
                diagnostics_report = {"exploitability_advantage_smoke": exploit_report}
                print_diagnostic_comparison(
                    iteration,
                    diagnostics_report,
                    prev_diagnostics,
                    baseline_diagnostics,
                )
                prev_diagnostics = diagnostics_report
                if baseline_diagnostics is None:
                    baseline_diagnostics = diagnostics_report

            model_selection_report: dict[str, Any] | None = None
            if strategy_trained and isinstance(diagnostics_report, dict):
                model_selection_report = run_strategy_model_selection(
                    iteration=iteration,
                    h2h_binary=h2h_binary,
                    repo_root=repo_root,
                    cluster_dir=cluster_dir,
                    args=args,
                    eval_panel=eval_panel,
                    strategy_onnx=strategy_onnx,
                    strategy_state=strategy_state,
                    strategy_latest_onnx=strategy_latest_onnx,
                    strategy_latest_state=strategy_latest_state,
                    strategy_best_candidate_onnx=strategy_best_candidate_onnx,
                    strategy_best_candidate_state=strategy_best_candidate_state,
                    strategy_best_confirmed_onnx=strategy_best_confirmed_onnx,
                    strategy_best_confirmed_state=strategy_best_confirmed_state,
                    diagnostics_report=diagnostics_report,
                    model_selection_state=model_selection_state,
                )
                status_text = str(model_selection_report.get("status", "unknown"))
                promoted_text = model_selection_report.get("promoted_confirmed")
                if promoted_text:
                    log(
                        color_green(
                            f"[iter {iter_label}] model selection: promoted confirmed "
                            f"(status={status_text})"
                        )
                    )
                else:
                    log(f"[iter {iter_label}] model selection: status={status_text}")

            if iteration % args.buffer_save_every == 0:
                log(f"[iter {iter_label}] checkpointing in-memory reservoirs...")
                save_reservoir_checkpoint(advantage_context_p0)
                save_reservoir_checkpoint(advantage_context_p1)
                if strategy_context is not None:
                    save_reservoir_checkpoint(strategy_context)

            iter_total = time.perf_counter() - iter_started
            traverse_time = p0_time + p1_time
            elapsed = time.perf_counter() - run_started
            done_this_run = iteration - completed_before
            avg_iter = elapsed / max(1, done_this_run)
            remaining = args.iterations - iteration
            eta = avg_iter * remaining

            reservoir_estimate_p0 = min(args.buffer_size, reservoir_estimate_p0 + p0_adv_samples)
            reservoir_estimate_p1 = min(args.buffer_size, reservoir_estimate_p1 + p1_adv_samples)
            reservoir_estimate = min(
                args.buffer_size,
                int(round((reservoir_estimate_p0 + reservoir_estimate_p1) / 2.0)),
            )
            samples_disk = directory_size_bytes(samples_dir)
            work_disk = directory_size_bytes(work_dir)

            loss_delta: float | None = None
            advantage_loss_values = [
                value for value in [advantage_loss_p0, advantage_loss_p1] if isinstance(value, float)
            ]
            advantage_loss = (
                float(sum(advantage_loss_values) / len(advantage_loss_values))
                if advantage_loss_values
                else None
            )
            previous_loss_values = [
                value
                for value in [prev_advantage_loss_p0, prev_advantage_loss_p1]
                if isinstance(value, float)
            ]
            previous_avg_loss = (
                float(sum(previous_loss_values) / len(previous_loss_values))
                if previous_loss_values
                else None
            )
            if advantage_loss is not None and previous_avg_loss is not None:
                loss_delta = advantage_loss - previous_avg_loss
            if isinstance(advantage_loss_p0, float):
                prev_advantage_loss_p0 = advantage_loss_p0
            if isinstance(advantage_loss_p1, float):
                prev_advantage_loss_p1 = advantage_loss_p1
            print_iter_summary(
                iter_total=iter_total,
                traverse_time=traverse_time,
                train_time=train_time,
                train_steps_ran=advantage_train_p0.steps_ran + advantage_train_p1.steps_ran,
                training_steps_target=advantage_train_p0.steps_target + advantage_train_p1.steps_target,
                early_stopped=advantage_train_p0.early_stopped or advantage_train_p1.early_stopped,
                advantage_loss=advantage_loss,
                loss_delta=loss_delta,
                reservoir_estimate=reservoir_estimate,
                buffer_size=args.buffer_size,
                work_disk=work_disk,
                samples_disk=samples_disk,
                elapsed=elapsed,
                eta=eta,
            )
            if isinstance(advantage_loss_p0, float) and isinstance(advantage_loss_p1, float):
                log(
                    "  advantage losses: "
                    f"p0={advantage_loss_p0:.6f} | p1={advantage_loss_p1:.6f}"
                )
            log("----------------------------------------")

            exploitability_report: dict[str, float | int] | None = None
            if isinstance(diagnostics_report, dict):
                raw = diagnostics_report.get("exploitability_strategy")
                if not isinstance(raw, dict):
                    raw = diagnostics_report.get("exploitability_advantage_smoke")
                if isinstance(raw, dict):
                    exploitability_report = raw

            metrics_iterations.append(
                {
                    "iteration": iteration,
                    "timestamp": now_iso(),
                    "samples": {
                        "player0_adv": p0_adv_samples,
                        "player1_adv": p1_adv_samples,
                        "strategy_from_p0": p0_strategy_samples,
                        "strategy_from_p1": p1_strategy_samples,
                        "strategy_merged": strategy_samples_merged,
                    },
                    "loss": {
                        "advantage": advantage_loss,
                        "advantage_p0": advantage_loss_p0,
                        "advantage_p1": advantage_loss_p1,
                        "advantage_delta": loss_delta,
                        "strategy": strategy_loss,
                    },
                    "training": {
                        "advantage_p0_steps_ran": advantage_train_p0.steps_ran,
                        "advantage_p0_steps_target": advantage_train_p0.steps_target,
                        "advantage_p0_early_stopped": advantage_train_p0.early_stopped,
                        "advantage_p1_steps_ran": advantage_train_p1.steps_ran,
                        "advantage_p1_steps_target": advantage_train_p1.steps_target,
                        "advantage_p1_early_stopped": advantage_train_p1.early_stopped,
                        "strategy_steps_ran": strategy_train.steps_ran if strategy_train else None,
                        "strategy_steps_target": strategy_train.steps_target if strategy_train else None,
                        "strategy_early_stopped": strategy_train.early_stopped if strategy_train else None,
                    },
                    "reservoir_estimate": {
                        "advantage_p0": reservoir_estimate_p0,
                        "advantage_p1": reservoir_estimate_p1,
                    },
                    "times_sec": {
                        "player0_traverse": p0_time,
                        "player1_traverse": p1_time,
                        "traverse_total": traverse_time,
                        "train_advantage": train_time,
                        "iteration_total": iter_total,
                        "elapsed_in_run": elapsed,
                        "eta": eta,
                    },
                    "disk_bytes": {
                        "work_dir": work_disk,
                        "samples_dir": samples_disk,
                    },
                    "strategy_snapshot": strategy_trained,
                    "exploitability": exploitability_report,
                    "diagnostics": diagnostics_report,
                    "model_selection": model_selection_report,
                    "paths": {
                        "sample_adv_p0": str(sample_adv_p0),
                        "sample_adv_p1": str(sample_adv_p1),
                        "sample_strategy_from_p0": str(sample_strategy_from_p0),
                        "sample_strategy_from_p1": str(sample_strategy_from_p1),
                        "advantage_p0_onnx": str(advantage_p0_onnx),
                        "advantage_p1_onnx": str(advantage_p1_onnx),
                        "strategy_onnx": str(strategy_onnx),
                        "strategy_latest_onnx": str(strategy_latest_onnx),
                        "strategy_best_candidate_onnx": str(strategy_best_candidate_onnx),
                        "strategy_best_confirmed_onnx": str(strategy_best_confirmed_onnx),
                    },
                }
            )
            metrics["model_selection"] = model_selection_state
            metrics["updated_at"] = now_iso()
            write_json(metrics_path, metrics)

            state = {
                "version": 1,
                "completed_iterations": iteration,
                "updated_at": now_iso(),
                "advantage_onnx": str(advantage_p0_onnx),
                "advantage_p0_onnx": str(advantage_p0_onnx),
                "advantage_p1_onnx": str(advantage_p1_onnx),
                "advantage_state": str(advantage_p0_state),
                "advantage_p0_state": str(advantage_p0_state),
                "advantage_p1_state": str(advantage_p1_state),
                "strategy_onnx": str(strategy_onnx),
                "strategy_state": str(strategy_state),
                "strategy_latest_onnx": str(strategy_latest_onnx),
                "strategy_best_candidate_onnx": str(strategy_best_candidate_onnx),
                "strategy_best_confirmed_onnx": str(strategy_best_confirmed_onnx),
                "work_dir": str(work_dir),
            }
            write_json(state_path, state)
    except KeyboardInterrupt:
        terminate_traversal_bundle(active_bundle)
        log("[deep-cfr] interrupted by user (Ctrl+C).")
        log("[deep-cfr] saving in-memory reservoirs before exit...")
        save_reservoir_checkpoint(advantage_context_p0)
        save_reservoir_checkpoint(advantage_context_p1)
        if strategy_context is not None:
            save_reservoir_checkpoint(strategy_context)
        log("[deep-cfr] partial progress is saved in state.json and metrics.json.")
        raise
    except Exception:
        terminate_traversal_bundle(active_bundle)
        raise

    log("[deep-cfr] final reservoir checkpoint...")
    save_reservoir_checkpoint(advantage_context_p0)
    save_reservoir_checkpoint(advantage_context_p1)
    if strategy_context is not None:
        save_reservoir_checkpoint(strategy_context)

    total_runtime = time.perf_counter() - run_started
    recent_losses = [
        float(item["loss"]["advantage"])
        for item in metrics_iterations[-10:]
        if isinstance(item, dict)
        and isinstance(item.get("loss"), dict)
        and isinstance(item["loss"].get("advantage"), (int, float))
    ]
    final_work_disk = directory_size_bytes(work_dir)
    log(
        f"[deep-cfr] completed iterations {first_iteration}..{args.iterations} "
        f"in {format_duration(total_runtime)}"
    )
    if recent_losses:
        avg_recent = sum(recent_losses) / len(recent_losses)
        log(f"[deep-cfr] avg advantage loss (last {len(recent_losses)}): {avg_recent:.6f}")
    fill_p0 = (100.0 * reservoir_estimate_p0 / args.buffer_size) if args.buffer_size > 0 else 0.0
    fill_p1 = (100.0 * reservoir_estimate_p1 / args.buffer_size) if args.buffer_size > 0 else 0.0
    log(
        f"[deep-cfr] reservoir estimate: "
        f"p0={reservoir_estimate_p0:,}/{args.buffer_size:,} ({fill_p0:.1f}%) | "
        f"p1={reservoir_estimate_p1:,}/{args.buffer_size:,} ({fill_p1:.1f}%)"
    )
    log(f"[deep-cfr] work_dir usage: {format_bytes(final_work_disk)}")
    log(f"[deep-cfr] state: {state_path}")
    log(f"[deep-cfr] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
