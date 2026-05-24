#!/usr/bin/env python3
"""Build a remote Windows handoff bundle for Deep CFR workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INCLUDE_PATHS = [
    Path("training"),
    Path("solver"),
    Path("eval"),
    Path("scripts"),
    Path("checkpoints/nlhe_clusters"),
]
DEFAULT_ARTIFACT_PATHS = [
    Path("data/deep_cfr_6max_answer/panels/screen_panel_strategy.bin"),
    Path("data/deep_cfr_6max_answer/panels/holdout_panel_strategy.bin"),
    Path("data/deep_cfr_6max_answer/panel_manifest.json"),
    Path("data/deep_cfr_6max_answer/divergence_gate.json"),
    Path("data/deep_cfr_6max_runtime_first/scale_search_wave1/report.json"),
    Path("data/deep_cfr_6max_runtime_first/pilot_report.json"),
    Path("data/deep_cfr_6max_runtime_first/pre_long_run_handoff.json"),
    Path("data/deep_cfr_6max_runtime_first/pre_long_run_handoff.md"),
    Path("data/deep_cfr_6max_prelongrun/pre_long_run_manifest.json"),
    Path("data/deep_cfr_6max_prelongrun/pre_long_run_manifest.md"),
    Path("data/deep_cfr_6max_prelongrun/divergence_report.json"),
    Path("data/deep_cfr_6max_prelongrun/bench_smoke.json"),
    Path("data/deep_cfr_6max_prelongrun/scale_search_smoke/report.json"),
    Path("data/deep_cfr_6max_prelongrun/pilot_smoke/pilot_report.json"),
    Path("data/deep_cfr_6max_prelongrun/smoke_panels/panel_manifest.json"),
    Path("data/deep_cfr_6max_prelongrun/smoke_panels/screen_panel_strategy.bin"),
    Path("data/deep_cfr_6max_prelongrun/smoke_panels/holdout_panel_strategy.bin"),
]
DEFAULT_REQUIREMENT_FILES = [
    Path("training/deep_cfr/requirements.txt"),
    Path("eval/requirements.txt"),
]
DEFAULT_RUST_BINARIES_WIN = [
    "run_traversals.exe",
    "measure_exploitability.exe",
    "head_to_head.exe",
]
DEFAULT_RUST_BINARIES_POSIX = [
    "run_traversals",
    "measure_exploitability",
    "head_to_head",
]
SESSION_SUBDIRS = [
    "env",
    "machine",
    "logs",
    "panels",
    "scale_frontier",
    "pilot",
    "handoff",
    "longrun",
    "export",
    "input",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree_filtered(src: Path, dst: Path) -> None:
    def _ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            "__pycache__",
            ".ruff_cache",
            ".venv",
            "node_modules",
        }
        # We copy prebuilt release binaries explicitly to keep bundle small/predictable.
        if Path(_dir).as_posix().endswith("/solver"):
            ignored.add("target")
        return {name for name in names if name in ignored}

    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)


def run_command(command: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )


def _is_retryable_windows_lock(exc: BaseException) -> bool:
    if not isinstance(exc, PermissionError):
        return False
    winerror = getattr(exc, "winerror", None)
    return winerror in {5, 32}


def _retry_rmtree(path: Path, *, attempts: int = 6, delay_sec: float = 0.75) -> bool:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError as exc:
            if not _is_retryable_windows_lock(exc):
                raise
            if attempt + 1 >= attempts:
                return False
            time.sleep(delay_sec * float(attempt + 1))
    return False


def _remove_path_best_effort(path: Path, *, attempts: int = 6, delay_sec: float = 0.5) -> list[Path]:
    locked: list[Path] = []
    if not path.exists():
        return locked

    if path.is_dir():
        for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
            locked.extend(_remove_path_best_effort(child, attempts=attempts, delay_sec=delay_sec))
        for attempt in range(attempts):
            try:
                path.rmdir()
                return locked
            except FileNotFoundError:
                return locked
            except OSError:
                if attempt + 1 >= attempts:
                    locked.append(path)
                    return locked
                time.sleep(delay_sec * float(attempt + 1))
        return locked

    for attempt in range(attempts):
        try:
            path.unlink()
            return locked
        except FileNotFoundError:
            return locked
        except PermissionError as exc:
            if attempt + 1 >= attempts or not _is_retryable_windows_lock(exc):
                locked.append(path)
                return locked
            time.sleep(delay_sec * float(attempt + 1))
    return locked


def reset_session_root(session_root: Path) -> None:
    if not session_root.exists():
        return
    if _retry_rmtree(session_root):
        return

    print(
        f"[bundle] warning: full cleanup of {session_root} is blocked by a locked file; "
        "falling back to best-effort cleanup",
        file=sys.stderr,
    )
    locked: list[Path] = []
    for child in sorted(session_root.iterdir(), key=lambda item: item.name.lower()):
        locked.extend(_remove_path_best_effort(child))

    if not locked:
        return

    non_wheelhouse_locked: list[Path] = []
    wheelhouse_locked: list[Path] = []
    for item in locked:
        try:
            rel = item.relative_to(session_root)
        except ValueError:
            non_wheelhouse_locked.append(item)
            continue
        if rel.parts and rel.parts[0] == "wheelhouse":
            wheelhouse_locked.append(rel)
        else:
            non_wheelhouse_locked.append(rel)

    if non_wheelhouse_locked:
        joined = ", ".join(str(item) for item in non_wheelhouse_locked[:10])
        raise PermissionError(
            "failed to clean session root; non-wheelhouse files are still locked: "
            f"{joined}"
        )

    joined = ", ".join(str(item) for item in wheelhouse_locked[:10])
    print(
        "[bundle] warning: locked wheelhouse entries were kept and may be reused: "
        f"{joined}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a remote Windows Deep CFR handoff bundle.")
    parser.add_argument(
        "--session-root",
        type=Path,
        required=True,
        help="Output folder for the remote_session bundle root.",
    )
    parser.add_argument(
        "--zip-output",
        type=Path,
        default=None,
        help="Optional zip output path. Defaults to <session-root>.zip.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to snapshot.",
    )
    parser.add_argument(
        "--include-path",
        action="append",
        default=[],
        help="Extra repo-relative path to include (file or directory).",
    )
    parser.add_argument(
        "--wheelhouse-source",
        type=Path,
        default=None,
        help="Existing wheelhouse directory to copy into bundle.",
    )
    parser.add_argument(
        "--build-wheelhouse",
        action="store_true",
        help="Download dependency wheels into bundle/wheelhouse.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used for wheelhouse downloads.",
    )
    parser.add_argument(
        "--torch-wheel-spec",
        action="append",
        default=[],
        help="Optional extra torch package spec(s) to pin GPU build (repeatable).",
    )
    parser.add_argument(
        "--torch-index-url",
        type=str,
        default=None,
        help="Optional index URL used when downloading --torch-wheel-spec packages.",
    )
    parser.add_argument(
        "--installer",
        action="append",
        default=[],
        help="Optional local installer file to include in bundle/installers.",
    )
    parser.add_argument(
        "--python-installer",
        action="append",
        default=[],
        help="Optional Windows Python installer copied to bundle/installers for auto-install.",
    )
    parser.add_argument(
        "--python-min-version",
        type=str,
        default="3.10",
        help="Minimum Python major.minor required by remote bootstrap.",
    )
    parser.add_argument(
        "--disable-python-auto-install",
        action="store_true",
        help="Disable bootstrap auto-install of Python when not present.",
    )
    parser.add_argument(
        "--allow-cpu-only",
        action="store_true",
        help="Allow bootstrap to continue without CUDA-ready NVIDIA stack.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=120.0,
        help="Minimum free disk space required at bootstrap time.",
    )
    parser.add_argument(
        "--extra-input",
        action="append",
        default=[],
        help="Optional file/folder copied to bundle/input.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip zip archive creation.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing --session-root before building.",
    )
    args = parser.parse_args()
    if args.wheelhouse_source is not None and args.build_wheelhouse:
        raise ValueError("choose either --wheelhouse-source or --build-wheelhouse, not both")
    parts = args.python_min_version.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("--python-min-version must use major.minor format (for example: 3.10)")
    if float(args.min_free_disk_gb) <= 0.0:
        raise ValueError("--min-free-disk-gb must be > 0")
    return args


def ensure_rust_binaries(repo_root: Path, session_repo_root: Path) -> list[dict[str, Any]]:
    release_dir = repo_root / "solver" / "target" / "release"
    target_release_dir = session_repo_root / "solver" / "target" / "release"
    target_release_dir.mkdir(parents=True, exist_ok=True)

    binaries = DEFAULT_RUST_BINARIES_WIN if sys.platform.startswith("win") else DEFAULT_RUST_BINARIES_POSIX
    metadata: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in binaries:
        src = release_dir / name
        if not src.exists():
            if name.startswith("run_traversals"):
                missing.append(name)
            continue
        dst = target_release_dir / name
        shutil.copy2(src, dst)
        metadata.append(
            {
                "name": name,
                "path": str(dst),
                "bytes": int(dst.stat().st_size),
                "sha256": sha256_file(dst),
            }
        )
    if missing:
        raise FileNotFoundError(
            "missing required Rust binary/binaries. Build first with "
            "`cargo build --release -p deep_cfr`.\n"
            f"Missing: {missing}"
        )
    return metadata


def build_wheelhouse(args: argparse.Namespace, repo_root: Path, wheelhouse_dir: Path) -> dict[str, Any]:
    wheelhouse_dir.mkdir(parents=True, exist_ok=True)
    py_exe = str(args.python_exe)
    for req in DEFAULT_REQUIREMENT_FILES:
        req_path = (repo_root / req).resolve()
        run_command(
            [py_exe, "-m", "pip", "download", "--dest", str(wheelhouse_dir), "-r", str(req_path)],
            cwd=repo_root,
        )
    for spec in args.torch_wheel_spec:
        command = [py_exe, "-m", "pip", "download", "--dest", str(wheelhouse_dir), spec]
        if args.torch_index_url:
            command.extend(["--index-url", str(args.torch_index_url)])
        run_command(command, cwd=repo_root)
    wheel_files = sorted(wheelhouse_dir.glob("*"))
    return {
        "mode": "downloaded",
        "python_exe": py_exe,
        "count": len(wheel_files),
        "files": [str(path.name) for path in wheel_files],
    }


def copy_existing_wheelhouse(source_dir: Path, wheelhouse_dir: Path) -> dict[str, Any]:
    if not source_dir.exists():
        raise FileNotFoundError(f"--wheelhouse-source not found: {source_dir}")
    shutil.copytree(source_dir, wheelhouse_dir, dirs_exist_ok=True)
    wheel_files = sorted(wheelhouse_dir.glob("*"))
    return {
        "mode": "copied",
        "source": str(source_dir.resolve()),
        "count": len(wheel_files),
        "files": [str(path.name) for path in wheel_files],
    }


def copy_optional_paths(paths: list[str], destination_root: Path) -> list[str]:
    copied: list[str] = []
    for raw in paths:
        src = Path(raw).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"path not found: {src}")
        dst = destination_root / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def to_session_relative_paths(paths: list[str], session_root: Path) -> list[str]:
    out: list[str] = []
    root = session_root.resolve()
    for raw in paths:
        path = Path(raw).resolve()
        try:
            rel = path.relative_to(root)
            out.append(str(rel).replace("\\", "/"))
        except ValueError:
            out.append(str(path))
    return out


def make_zip(session_root: Path, zip_output: Path | None) -> Path:
    if zip_output is None:
        zip_output = session_root.parent / f"{session_root.name}.zip"
    zip_output = zip_output.resolve()
    zip_output.parent.mkdir(parents=True, exist_ok=True)
    if zip_output.exists():
        zip_output.unlink()
    base_name = str(zip_output.with_suffix(""))
    archive_path = shutil.make_archive(
        base_name=base_name,
        format="zip",
        root_dir=str(session_root.parent),
        base_dir=str(session_root.name),
    )
    return Path(archive_path).resolve()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    session_root = args.session_root.resolve()
    session_repo_root = session_root / "repo"
    wheelhouse_dir = session_root / "wheelhouse"
    installers_dir = session_root / "installers"
    input_dir = session_root / "input"

    if args.clean and session_root.exists():
        reset_session_root(session_root)
    session_root.mkdir(parents=True, exist_ok=True)
    for folder in SESSION_SUBDIRS:
        (session_root / folder).mkdir(parents=True, exist_ok=True)

    include_paths = [path for path in DEFAULT_INCLUDE_PATHS]
    include_paths.extend(path for path in DEFAULT_ARTIFACT_PATHS)
    include_paths.extend(Path(value) for value in args.include_path)
    copied_repo_paths: list[str] = []
    for rel_path in include_paths:
        src = (repo_root / rel_path).resolve()
        if not src.exists():
            continue
        dst = (session_repo_root / rel_path).resolve()
        if src.is_dir():
            copy_tree_filtered(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied_repo_paths.append(str(rel_path))

    rust_binaries = ensure_rust_binaries(repo_root, session_repo_root)

    wheelhouse_meta: dict[str, Any]
    if args.wheelhouse_source is not None:
        wheelhouse_meta = copy_existing_wheelhouse(args.wheelhouse_source.resolve(), wheelhouse_dir)
    elif args.build_wheelhouse:
        wheelhouse_meta = build_wheelhouse(args=args, repo_root=repo_root, wheelhouse_dir=wheelhouse_dir)
    else:
        wheelhouse_dir.mkdir(parents=True, exist_ok=True)
        wheelhouse_meta = {"mode": "empty", "count": 0, "files": []}

    copied_installers = copy_optional_paths(args.installer, installers_dir) if args.installer else []
    copied_python_installers = (
        copy_optional_paths(args.python_installer, installers_dir) if args.python_installer else []
    )
    copied_input = copy_optional_paths(args.extra_input, input_dir) if args.extra_input else []
    copied_installers_manifest = to_session_relative_paths(copied_installers, session_root=session_root)
    copied_python_installers_manifest = to_session_relative_paths(
        copied_python_installers,
        session_root=session_root,
    )
    copied_input_manifest = to_session_relative_paths(copied_input, session_root=session_root)

    requirements_meta: list[dict[str, Any]] = []
    for req in DEFAULT_REQUIREMENT_FILES:
        req_path = (session_repo_root / req).resolve()
        if req_path.exists():
            requirements_meta.append(
                {
                    "path": str(req_path),
                    "sha256": sha256_file(req_path),
                }
            )

    manifest = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "session_root": str(session_root),
        "layout_version": "v3",
        "included_repo_paths": copied_repo_paths,
        "requirements": requirements_meta,
        "rust_binaries": rust_binaries,
        "wheelhouse": wheelhouse_meta,
        "torch_install": {
            "wheel_spec": list(args.torch_wheel_spec),
            "index_url": str(args.torch_index_url or ""),
        },
        "python_install": {
            "auto_install": bool(not args.disable_python_auto_install),
            "minimum_version": str(args.python_min_version),
            "installer_paths": copied_python_installers_manifest,
        },
        "prerequisite_policy": {
            "verify_only_nvidia_driver": True,
            "require_cuda": bool(not args.allow_cpu_only),
            "min_free_disk_gb": float(args.min_free_disk_gb),
        },
        "installers": copied_installers_manifest,
        "extra_input_paths": copied_input_manifest,
        "commands": {
            "bootstrap": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\bootstrap_remote_windows.ps1 -SessionRoot .",
            "day_gated": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_day.ps1 -SessionRoot .",
            "day_auto_longrun": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_day.ps1 -SessionRoot . -StartLongRunIfReady",
            "benchmark": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_phase.ps1 -SessionRoot . -Phase benchmark",
            "scale_frontier": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_phase.ps1 -SessionRoot . -Phase scale_frontier",
            "pilot": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_phase.ps1 -SessionRoot . -Phase pilot",
            "handoff": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_phase.ps1 -SessionRoot . -Phase handoff",
            "longrun": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_phase.ps1 -SessionRoot . -Phase longrun -UseDetached",
            "status": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_phase.ps1 -SessionRoot . -Phase status",
            "resume": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\run_remote_phase.ps1 -SessionRoot . -Phase resume",
            "export": "powershell -ExecutionPolicy Bypass -File .\\repo\\scripts\\export_remote_session.ps1 -SessionRoot .",
        },
    }
    manifest_path = session_root / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path = None
    if not args.no_zip:
        archive_path = make_zip(session_root=session_root, zip_output=args.zip_output)

    payload = {
        "status": "ok",
        "session_root": str(session_root),
        "manifest": str(manifest_path),
        "archive": str(archive_path) if archive_path is not None else "",
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted"}, ensure_ascii=True))
        raise SystemExit(1)
