#!/usr/bin/env python3
"""Pre-Loop Validation Runner — 6-step pre-flight check + go/no-go gate.

Validates the entire targeted solve pipeline end-to-end with small hand batches:
  Step 1: Preflop table data integrity
  Step 2: Range vector spot-check (non-uniform, correct blockers)
  Step 3: Small-batch targeted solve cycle
  Step 4: Extracted node strategy quality + GTO range source
  Step 5: Store write/read round-trip
  Step 6: Active store refresh verification

Go/No-Go Gate (evaluated after all steps):
  (a) Preflop table validation passed for all default scenarios
  (b) At least 1 small-batch cycle completed with zero worker crashes
  (c) All extracted nodes have meta.range_source: 'preflop_gto'
  (d) All extracted nodes pass strategy quality validation
  (e) Policy worker successfully loads and queries at least 1 node

Usage:
    py -3 eval/run_pre_loop_validation.py --phase0_root eval/test_isolated_store
    py -3 eval/run_pre_loop_validation.py --phase0_root eval/test_isolated_store --batch_size 10
    py -3 eval/run_pre_loop_validation.py --phase0_root %LOCALAPPDATA%\\WiPoker\\phase0_store --batch_size 25

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 9.1, 9.2, 9.3, 9.4, 9.5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

APPROX_STORE_NAME = "pio_store_autopilot_loop_approx.jsonl"
EXACT_STORE_NAME = "pio_store_autopilot_loop.jsonl"
ACTIVE_STORE_NAME = "active_store.jsonl"

# ── ANSI colors ───────────────────────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


# ── Utilities ─────────────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def count_jsonl_rows(path: Path) -> int:
    """Count non-blank lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_jsonl_nodes(path: Path) -> list[dict[str, Any]]:
    """Read all nodes from a JSONL file."""
    nodes = []
    if not path.exists():
        return nodes
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(json.loads(line))
    return nodes


def print_step(step: int, title: str) -> None:
    print(f"\n{C.BOLD}{'─' * 60}{C.RESET}")
    print(f"{C.BOLD}  Step {step}: {title}{C.RESET}")
    print(f"{C.BOLD}{'─' * 60}{C.RESET}")


def print_pass(msg: str) -> None:
    print(f"  {C.GREEN}PASS{C.RESET}  {msg}")


def print_fail(msg: str) -> None:
    print(f"  {C.RED}FAIL{C.RESET}  {msg}")


def print_info(msg: str) -> None:
    print(f"  {C.DIM}{msg}{C.RESET}")


# ── Step 1: Preflop Table Validation ─────────────────────────────────────────

def step1_preflop_tables() -> dict[str, Any]:
    """Run preflop table validation via Node subprocess."""
    print_step(1, "Preflop Table Validation")

    script = REPO_ROOT / "eval" / "validate_preflop_tables.mjs"
    if not script.exists():
        return {"passed": False, "errors": [f"Script not found: {script}"]}

    try:
        result = subprocess.run(
            ["node", str(script)],
            cwd=str(REPO_ROOT),
            capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"passed": False, "errors": [f"Subprocess error: {exc}"]}

    # Parse JSON from stdout
    try:
        output = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return {
            "passed": False,
            "errors": [
                f"Failed to parse output (exit={result.returncode})",
                f"stdout: {result.stdout[:500]}",
                f"stderr: {result.stderr[:500]}",
            ],
        }

    passed = output.get("valid", False)
    errors = output.get("errors", [])

    if passed:
        print_pass("All preflop table scenarios validated")
    else:
        print_fail(f"Preflop table validation failed with {len(errors)} error(s)")
        for err in errors[:10]:
            print_info(f"  {err}")

    return {"passed": passed, "errors": errors}


# ── Step 2: Range Vector Spot-Check ──────────────────────────────────────────

# Standard 52-card deck for generating random flops
RANKS = "AKQJT98765432"
SUITS = "shdc"
ALL_CARDS = [f"{r}{s}" for r in RANKS for s in SUITS]


def _generate_random_flops(n: int = 3) -> list[list[str]]:
    """Generate n random 3-card flops from the deck."""
    flops = []
    for _ in range(n):
        flop = random.sample(ALL_CARDS, 3)
        flops.append(flop)
    return flops


def _check_range_vector(vector_str: str, board_cards: list[str]) -> dict[str, Any]:
    """Validate a single range vector string."""
    errors = []
    values = vector_str.strip().split()

    if len(values) != 1326:
        return {"passed": False, "errors": [f"Expected 1326 values, got {len(values)}"]}

    floats = []
    for i, v in enumerate(values):
        try:
            f = float(v)
        except ValueError:
            errors.append(f"Value at index {i} is not a float: {v}")
            continue
        if f < 0 or f > 1:
            errors.append(f"Value at index {i} = {f} is outside [0, 1]")
        floats.append(f)

    if errors:
        return {"passed": False, "errors": errors}

    # Check blockers: count zeroed combos
    zero_count = sum(1 for f in floats if f == 0.0)
    if zero_count < 150:
        errors.append(
            f"Only {zero_count} zeroed combos for board {board_cards}, expected >= 150"
        )

    # Check non-uniform: mean of non-zero weights should be < 1.0
    non_zero = [f for f in floats if f > 0]
    if non_zero:
        mean_nz = sum(non_zero) / len(non_zero)
        if mean_nz >= 1.0:
            errors.append(
                f"Mean of non-zero weights = {mean_nz:.4f}, expected < 1.0 (vector is uniform)"
            )
    else:
        errors.append("All weights are zero — range is empty")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "zero_count": zero_count,
        "non_zero_count": len(non_zero),
        "mean_non_zero": sum(non_zero) / len(non_zero) if non_zero else 0,
    }


def step2_range_vector_spot_check() -> dict[str, Any]:
    """Build range vectors for 3 random flops and verify non-uniform + correct blockers."""
    print_step(2, "Range Vector Spot-Check")

    flops = _generate_random_flops(3)
    all_errors: list[str] = []
    flop_results: list[dict[str, Any]] = []

    for flop in flops:
        board_str = json.dumps(flop)
        # Build a small Node script to call buildRangeVector for OOP scenario
        # We need combo labels from PioSolver's hand order, but for validation
        # we can generate the canonical 1326 combos ourselves.
        node_script = f"""
import {{ buildRangeVector, DEFAULT_OOP_SCENARIO }} from './eval/preflop_range_builder.mjs';

// Generate canonical 1326 combo labels (same order as PioSolver's show_hand_order)
const RANKS = 'AKQJT98765432';
const SUITS = ['s', 'h', 'd', 'c'];
const cards = [];
for (const r of RANKS) {{
    for (const s of SUITS) {{
        cards.push(r + s);
    }}
}}
const combos = [];
for (let i = 0; i < cards.length; i++) {{
    for (let j = i + 1; j < cards.length; j++) {{
        combos.push(cards[i] + cards[j]);
    }}
}}

const board = {board_str};
const vector = buildRangeVector(combos, DEFAULT_OOP_SCENARIO, board);
process.stdout.write(vector);
"""
        try:
            result = subprocess.run(
                ["node", "--input-type=module", "-e", node_script],
                cwd=str(REPO_ROOT),
                capture_output=True, encoding="utf-8", errors="replace", timeout=15,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            all_errors.append(f"Flop {flop}: subprocess error: {exc}")
            flop_results.append({"flop": flop, "passed": False, "errors": [str(exc)]})
            continue

        if result.returncode != 0:
            all_errors.append(f"Flop {flop}: node exited {result.returncode}: {result.stderr[:300]}")
            flop_results.append({
                "flop": flop, "passed": False,
                "errors": [f"exit {result.returncode}: {result.stderr[:300]}"],
            })
            continue

        check = _check_range_vector(result.stdout, flop)
        flop_results.append({"flop": flop, **check})

        if check["passed"]:
            print_pass(
                f"Flop {flop}: {check.get('zero_count', '?')} blocked, "
                f"mean_nz={check.get('mean_non_zero', 0):.4f}"
            )
        else:
            for err in check["errors"]:
                all_errors.append(f"Flop {flop}: {err}")
                print_fail(f"Flop {flop}: {err}")

    passed = len(all_errors) == 0
    if passed:
        print_pass("All 3 random flops produced valid non-uniform range vectors")
    else:
        print_fail(f"Range vector spot-check failed with {len(all_errors)} error(s)")

    return {"passed": passed, "errors": all_errors, "flop_results": flop_results}


# ── Step 3: Small-Batch Targeted Solve Cycle ─────────────────────────────────

def step3_targeted_solve_cycle(
    phase0_root: Path, pio_exe: str, batch_size: int, cfr_steps: int,
) -> dict[str, Any]:
    """Run one targeted solve cycle with a small batch against the test store."""
    print_step(3, f"Small-Batch Targeted Solve Cycle (batch_size={batch_size})")

    # Build TypeScript before running eval
    print_info("Building eval TypeScript...")
    build_result = subprocess.run(
        "npm run eval:build",
        cwd=str(REPO_ROOT),
        capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        shell=True,
    )
    if build_result.returncode != 0:
        return {
            "passed": False,
            "errors": [f"eval:build failed: {build_result.stderr[:500]}"],
            "nodes_extracted": 0,
        }

    # Run the targeted solve orchestrator with small batch
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "run_targeted_solve.py"),
        "--phase0_root", str(phase0_root),
        "--eval_hands", str(batch_size),
        "--cfr_steps", str(cfr_steps),
        "--max_cycles", "1",
        "--max_solves", "3",
        "--seed", "42",
        "--hero_mode", "worker",
        "--parallel", "1",
    ]

    env = os.environ.copy()
    env["PIO_EXE"] = pio_exe
    env["PYTHONIOENCODING"] = "utf-8"

    print_info(f"Running: {' '.join(cmd[:6])} ...")
    print_info(f"PIO_EXE={pio_exe}")

    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT), env=env,
            capture_output=True, encoding="utf-8", errors="replace", timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "errors": ["Targeted solve timed out after 600s"],
            "nodes_extracted": 0,
        }
    except OSError as exc:
        return {
            "passed": False,
            "errors": [f"Subprocess error: {exc}"],
            "nodes_extracted": 0,
        }

    # Check for worker crashes in output
    stdout_lower = result.stdout.lower()
    stderr_lower = result.stderr.lower()
    worker_crashed = "worker crash" in stdout_lower or "worker crash" in stderr_lower

    if result.returncode != 0:
        # Non-zero exit is acceptable if "no missing spots" — means store already covers everything
        if "no missing spots" in stdout_lower or "all spots covered" in stdout_lower:
            print_pass("No missing spots — store already has full coverage")
            return {
                "passed": True,
                "errors": [],
                "nodes_extracted": 0,
                "note": "No missing spots — full coverage",
            }

        return {
            "passed": False,
            "errors": [
                f"Targeted solve exited {result.returncode}",
                f"stdout (last 500): {result.stdout[-500:]}",
                f"stderr (last 500): {result.stderr[-500:]}",
            ],
            "nodes_extracted": 0,
            "worker_crashed": worker_crashed,
        }

    # Count nodes in the approx store after the cycle
    approx_store = phase0_root / APPROX_STORE_NAME
    nodes_in_store = count_jsonl_rows(approx_store)

    passed = not worker_crashed and nodes_in_store > 0
    errors = []

    if worker_crashed:
        errors.append("Worker crash detected in output")
    if nodes_in_store == 0:
        # This might be OK if there were no missing spots
        if "no missing spots" in stdout_lower or "all spots covered" in stdout_lower:
            passed = True
            print_pass("No missing spots — store already has full coverage")
        else:
            errors.append("No nodes written to approx store after solve cycle")

    if passed:
        print_pass(f"Solve cycle completed: {nodes_in_store} nodes in approx store")
    else:
        for err in errors:
            print_fail(err)

    return {
        "passed": passed,
        "errors": errors,
        "nodes_extracted": nodes_in_store,
        "worker_crashed": worker_crashed,
        "stdout_tail": result.stdout[-1000:],
    }


# ── Step 4: Node Strategy Quality + Range Source Validation ──────────────────

def step4_validate_nodes(phase0_root: Path) -> dict[str, Any]:
    """For each extracted node, validate strategy quality and range source tag."""
    print_step(4, "Node Strategy Quality & Range Source Validation")

    approx_store = phase0_root / APPROX_STORE_NAME
    nodes = read_jsonl_nodes(approx_store)

    if not nodes:
        print_info("No nodes in approx store — skipping (may be OK if step 3 had no missing spots)")
        return {"passed": True, "errors": [], "nodes_checked": 0, "note": "no nodes to validate"}

    all_errors: list[str] = []
    nodes_checked = 0
    gto_count = 0
    uniform_count = 0

    script = REPO_ROOT / "eval" / "validate_node_strategy.mjs"
    if not script.exists():
        return {"passed": False, "errors": [f"Script not found: {script}"], "nodes_checked": 0}

    for node in nodes:
        spot_key = node.get("spot_key", "<unknown>")
        nodes_checked += 1

        # Check range source tag
        range_source = node.get("meta", {}).get("range_source", "<missing>")
        if range_source == "preflop_gto":
            gto_count += 1
        else:
            uniform_count += 1
            all_errors.append(
                f"{spot_key}: meta.range_source = '{range_source}', expected 'preflop_gto'"
            )

        # Validate strategy via Node subprocess
        node_json = json.dumps(node)
        try:
            result = subprocess.run(
                ["node", str(script)],
                cwd=str(REPO_ROOT),
                input=node_json,
                capture_output=True, encoding="utf-8", errors="replace", timeout=15,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            all_errors.append(f"{spot_key}: validation subprocess error: {exc}")
            continue

        try:
            output = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            all_errors.append(
                f"{spot_key}: failed to parse validation output (exit={result.returncode})"
            )
            continue

        if not output.get("valid", False):
            for err in output.get("errors", []):
                all_errors.append(err)

    passed = len(all_errors) == 0
    if passed:
        print_pass(
            f"All {nodes_checked} nodes validated: "
            f"{gto_count} GTO-sourced, {uniform_count} uniform"
        )
    else:
        print_fail(f"Node validation failed with {len(all_errors)} error(s)")
        for err in all_errors[:10]:
            print_info(f"  {err}")

    return {
        "passed": passed,
        "errors": all_errors,
        "nodes_checked": nodes_checked,
        "gto_count": gto_count,
        "uniform_count": uniform_count,
    }


# ── Step 5: Store Write/Read Round-Trip ──────────────────────────────────────

def step5_store_round_trip(phase0_root: Path) -> dict[str, Any]:
    """Verify that nodes survive write→read round-trip without data loss."""
    print_step(5, "Store Write/Read Round-Trip")

    approx_store = phase0_root / APPROX_STORE_NAME
    nodes = read_jsonl_nodes(approx_store)

    if not nodes:
        print_info("No nodes in approx store — skipping round-trip check")
        return {"passed": True, "errors": [], "nodes_checked": 0, "note": "no nodes to check"}

    all_errors: list[str] = []
    nodes_checked = 0

    for node in nodes:
        spot_key = node.get("spot_key", "<unknown>")
        nodes_checked += 1

        # Serialize to JSON and parse back (simulates write→read)
        serialized = json.dumps(node, separators=(",", ":"))
        try:
            parsed = json.loads(serialized)
        except json.JSONDecodeError as exc:
            all_errors.append(f"{spot_key}: JSON round-trip parse failed: {exc}")
            continue

        # Check spot_key preserved
        if parsed.get("spot_key") != node.get("spot_key"):
            all_errors.append(
                f"{spot_key}: spot_key mismatch after round-trip: "
                f"{parsed.get('spot_key')} != {node.get('spot_key')}"
            )

        # Check meta fields preserved exactly
        orig_meta = node.get("meta", {})
        parsed_meta = parsed.get("meta", {})
        for field in ("range_source", "quality", "cfr_steps", "source"):
            orig_val = orig_meta.get(field)
            parsed_val = parsed_meta.get(field)
            if orig_val != parsed_val:
                all_errors.append(
                    f"{spot_key}: meta.{field} mismatch: {parsed_val} != {orig_val}"
                )

        # Check combo_labels preserved
        orig_labels = node.get("strategy", {}).get("combo_labels", [])
        parsed_labels = parsed.get("strategy", {}).get("combo_labels", [])
        if orig_labels != parsed_labels:
            all_errors.append(
                f"{spot_key}: combo_labels mismatch "
                f"(orig len={len(orig_labels)}, parsed len={len(parsed_labels)})"
            )

        # Check by_child tokens and frequencies preserved within 1e-9
        orig_by_child = node.get("strategy", {}).get("by_child", [])
        parsed_by_child = parsed.get("strategy", {}).get("by_child", [])

        if len(orig_by_child) != len(parsed_by_child):
            all_errors.append(
                f"{spot_key}: by_child length mismatch: "
                f"{len(parsed_by_child)} != {len(orig_by_child)}"
            )
        else:
            for i in range(len(orig_by_child)):
                orig_freqs = orig_by_child[i].get("frequencies", [])
                parsed_freqs = parsed_by_child[i].get("frequencies", [])
                if len(orig_freqs) != len(parsed_freqs):
                    all_errors.append(
                        f"{spot_key}: by_child[{i}].frequencies length mismatch"
                    )
                    continue
                for j in range(len(orig_freqs)):
                    delta = abs(orig_freqs[j] - parsed_freqs[j])
                    if delta > 1e-9:
                        all_errors.append(
                            f"{spot_key}: by_child[{i}].frequencies[{j}] "
                            f"delta={delta:.2e} exceeds 1e-9"
                        )
                        break  # report first violation per child only

    passed = len(all_errors) == 0
    if passed:
        print_pass(f"All {nodes_checked} nodes survive write/read round-trip")
    else:
        print_fail(f"Round-trip check failed with {len(all_errors)} error(s)")
        for err in all_errors[:10]:
            print_info(f"  {err}")

    return {"passed": passed, "errors": all_errors, "nodes_checked": nodes_checked}


# ── Step 6: Active Store Refresh Verification ────────────────────────────────

def step6_active_store_refresh(phase0_root: Path) -> dict[str, Any]:
    """Verify that active store refresh succeeds and contains the new nodes."""
    print_step(6, "Active Store Refresh Verification")

    active_store = phase0_root / ACTIVE_STORE_NAME
    exact_store = phase0_root / EXACT_STORE_NAME
    approx_store = phase0_root / APPROX_STORE_NAME

    # Count nodes before refresh
    approx_count_before = count_jsonl_rows(approx_store)

    # Run refresh
    refresh_script = REPO_ROOT / "eval" / "refresh_strategy_store.py"
    cmd = [
        sys.executable,
        str(refresh_script),
        "--out_store", str(active_store),
        "--no_default_source_dirs",
        "--include_approx",
    ]
    if exact_store.exists():
        cmd.extend(["--source_file", str(exact_store)])
    if approx_store.exists():
        cmd.extend(["--source_file", str(approx_store)])

    print_info(f"Running refresh: {' '.join(cmd[:4])} ...")

    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"passed": False, "errors": [f"Refresh subprocess error: {exc}"]}

    if result.returncode != 0:
        return {
            "passed": False,
            "errors": [
                f"Refresh exited {result.returncode}",
                f"stderr: {result.stderr[:500]}",
            ],
        }

    # Count nodes after refresh
    active_count = count_jsonl_rows(active_store)
    exact_count = count_jsonl_rows(exact_store)
    approx_count = count_jsonl_rows(approx_store)

    errors: list[str] = []

    # Active store should contain at least the approx nodes
    if approx_count_before > 0 and active_count < approx_count_before:
        errors.append(
            f"Active store has {active_count} nodes but approx store has "
            f"{approx_count_before} — new nodes may not be included"
        )

    # If we have approx nodes, verify they appear in the active store
    if approx_count > 0:
        approx_keys = set()
        for node in read_jsonl_nodes(approx_store):
            key = node.get("spot_key")
            if key:
                approx_keys.add(key)

        active_keys = set()
        for node in read_jsonl_nodes(active_store):
            key = node.get("spot_key")
            if key:
                active_keys.add(key)

        missing_keys = approx_keys - active_keys
        if missing_keys:
            errors.append(
                f"{len(missing_keys)} approx node(s) missing from active store: "
                f"{list(missing_keys)[:5]}"
            )

    passed = len(errors) == 0
    if passed:
        print_pass(
            f"Active store refreshed: {active_count} total nodes "
            f"({exact_count} exact + {approx_count} approx)"
        )
    else:
        print_fail(f"Active store refresh verification failed")
        for err in errors:
            print_info(f"  {err}")

    return {
        "passed": passed,
        "errors": errors,
        "active_count": active_count,
        "exact_count": exact_count,
        "approx_count": approx_count,
    }


# ── Policy Worker Probe (for go/no-go criterion e) ───────────────────────────

def _probe_policy_worker(active_store_path: Path) -> dict[str, Any]:
    """Spawn a small Node.js script that loads the active store and queries 1 node.

    Returns {"passed": bool, "errors": [...], "nodes_loaded": int, "query_ok": bool}.
    """
    if not active_store_path.exists():
        return {
            "passed": False,
            "errors": [f"Active store not found: {active_store_path}"],
            "nodes_loaded": 0,
            "query_ok": False,
        }

    # Read the first node from the active store to get a spot_key for querying
    nodes = read_jsonl_nodes(active_store_path)
    if not nodes:
        return {
            "passed": False,
            "errors": ["Active store is empty — no nodes to query"],
            "nodes_loaded": 0,
            "query_ok": False,
        }

    # Pick the first node with a valid spot_key
    test_node = None
    for n in nodes:
        if n.get("spot_key"):
            test_node = n
            break

    if not test_node:
        return {
            "passed": False,
            "errors": ["No node with a valid spot_key found in active store"],
            "nodes_loaded": len(nodes),
            "query_ok": False,
        }

    spot_key = test_node["spot_key"]
    # Pick a combo label from the node's strategy if available
    combo_labels = test_node.get("strategy", {}).get("combo_labels", [])
    test_combo = combo_labels[0] if combo_labels else "AhKd"

    # Build a small inline Node.js script that loads the store and queries it
    probe_script = f"""
import {{ loadStrategyStoreFromFileStreaming, getNodeBySpotKey, queryBySpotKey }} from './src/strategyStore.js';
const storePath = {json.dumps(str(active_store_path))};
const spotKey = {json.dumps(spot_key)};
const combo = {json.dumps(test_combo)};
try {{
    const store = await loadStrategyStoreFromFileStreaming(storePath, {{ format: 'jsonl' }});
    const nodeCount = Array.isArray(store?.nodes) ? store.nodes.length : 0;
    const node = getNodeBySpotKey(store, spotKey);
    const rows = queryBySpotKey(store, spotKey, combo);
    const rangeSource = node?.meta?.range_source ?? null;
    const hasStrategy = rows && rows.length > 0;
    process.stdout.write(JSON.stringify({{
        ok: true,
        nodes_loaded: nodeCount,
        spot_key: spotKey,
        combo: combo,
        range_source: rangeSource,
        has_strategy: hasStrategy,
        rows_count: rows ? rows.length : 0,
    }}) + '\\n');
}} catch (err) {{
    process.stdout.write(JSON.stringify({{
        ok: false,
        error: err.message || String(err),
    }}) + '\\n');
    process.exit(1);
}}
"""

    try:
        result = subprocess.run(
            ["node", "--input-type=module", "-e", probe_script],
            cwd=str(REPO_ROOT),
            capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "passed": False,
            "errors": [f"Policy worker probe subprocess error: {exc}"],
            "nodes_loaded": 0,
            "query_ok": False,
        }

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        return {
            "passed": False,
            "errors": [
                f"Policy worker probe exited {result.returncode}",
                f"stderr: {stderr_tail}",
            ],
            "nodes_loaded": 0,
            "query_ok": False,
        }

    try:
        output = json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, ValueError, IndexError):
        return {
            "passed": False,
            "errors": [f"Failed to parse probe output: {result.stdout[:300]}"],
            "nodes_loaded": 0,
            "query_ok": False,
        }

    if not output.get("ok"):
        return {
            "passed": False,
            "errors": [f"Probe failed: {output.get('error', 'unknown')}"],
            "nodes_loaded": output.get("nodes_loaded", 0),
            "query_ok": False,
        }

    return {
        "passed": True,
        "errors": [],
        "nodes_loaded": output.get("nodes_loaded", 0),
        "query_ok": output.get("has_strategy", False),
        "spot_key": output.get("spot_key"),
        "range_source": output.get("range_source"),
        "rows_count": output.get("rows_count", 0),
    }


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-Loop Validation Runner — 6-step pre-flight check before production loop."
    )
    parser.add_argument(
        "--phase0_root", type=Path, required=True,
        help="Root directory for the store (e.g. eval/test_isolated_store or %%LOCALAPPDATA%%\\WiPoker\\phase0_store)",
    )
    parser.add_argument(
        "--pio_exe", type=str, default=None,
        help="Path to PioSolver executable (or set PIO_EXE env var)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=25,
        help="Number of hands for the small-batch eval cycle (default: 25)",
    )
    parser.add_argument(
        "--cfr_steps", type=int, default=500,
        help="CFR iterations for the solve step (default: 500)",
    )
    return parser.parse_args()


def run_pre_loop_validation(args: argparse.Namespace) -> int:
    """Run all 6 validation steps and produce a go/no-go verdict.

    Returns 0 if all steps pass, 1 if any step fails.
    """
    phase0_root = args.phase0_root.resolve()
    phase0_root.mkdir(parents=True, exist_ok=True)

    # Resolve PioSolver executable
    pio_exe = (
        args.pio_exe
        or os.environ.get("PIO_EXE")
        or os.environ.get("WIPOKER_PIO_PATH")
        or os.environ.get("PIO_PATH")
        or ""
    ).strip()

    print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}  Pre-Loop Validation Runner{C.RESET}")
    print(f"{C.BOLD}{'═' * 60}{C.RESET}")
    print(f"  phase0_root:  {phase0_root}")
    print(f"  pio_exe:      {pio_exe or '(not set)'}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  cfr_steps:    {args.cfr_steps}")
    print(f"  timestamp:    {utc_now_iso()}")

    started = time.monotonic()
    results: dict[str, dict[str, Any]] = {}

    # ── Step 1: Preflop Table Validation ──────────────────────────────────
    results["step1_preflop_tables"] = step1_preflop_tables()

    # ── Step 2: Range Vector Spot-Check ───────────────────────────────────
    results["step2_range_vectors"] = step2_range_vector_spot_check()

    # ── Step 3: Small-Batch Targeted Solve Cycle ──────────────────────────
    if pio_exe and Path(pio_exe).exists():
        results["step3_targeted_solve"] = step3_targeted_solve_cycle(
            phase0_root=phase0_root,
            pio_exe=pio_exe,
            batch_size=args.batch_size,
            cfr_steps=args.cfr_steps,
        )
    else:
        print_step(3, "Small-Batch Targeted Solve Cycle")
        print_fail("PIO_EXE not set or not found — skipping solve cycle")
        print_info(
            "Set --pio_exe or PIO_EXE env var to enable. "
            "Steps 4-6 will validate existing store data if available."
        )
        results["step3_targeted_solve"] = {
            "passed": False,
            "errors": ["PIO_EXE not set or executable not found"],
            "nodes_extracted": 0,
            "skipped": True,
        }

    # ── Step 4: Node Strategy Quality + Range Source ──────────────────────
    results["step4_node_validation"] = step4_validate_nodes(phase0_root)

    # ── Step 5: Store Write/Read Round-Trip ───────────────────────────────
    results["step5_round_trip"] = step5_store_round_trip(phase0_root)

    # ── Step 6: Active Store Refresh ──────────────────────────────────────
    results["step6_active_store"] = step6_active_store_refresh(phase0_root)

    # ── Step Summary ─────────────────────────────────────────────────────
    print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}  Step Summary{C.RESET}")
    print(f"{C.BOLD}{'═' * 60}{C.RESET}")

    step_names = {
        "step1_preflop_tables": "Preflop Table Validation",
        "step2_range_vectors": "Range Vector Spot-Check",
        "step3_targeted_solve": "Small-Batch Targeted Solve",
        "step4_node_validation": "Node Strategy Quality",
        "step5_round_trip": "Store Round-Trip",
        "step6_active_store": "Active Store Refresh",
    }

    for key, name in step_names.items():
        r = results.get(key, {})
        status = f"{C.GREEN}PASS{C.RESET}" if r.get("passed") else f"{C.RED}FAIL{C.RESET}"
        skipped = " (skipped)" if r.get("skipped") else ""
        print(f"  {status}  {name}{skipped}")

    # ── Go/No-Go Gate ─────────────────────────────────────────────────────
    print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}  Go/No-Go Gate{C.RESET}")
    print(f"{C.BOLD}{'═' * 60}{C.RESET}")

    s1 = results.get("step1_preflop_tables", {})
    s3 = results.get("step3_targeted_solve", {})
    s4 = results.get("step4_node_validation", {})

    # (a) Preflop table validation passed for all default scenarios
    criterion_a = s1.get("passed", False)
    # (b) At least 1 small-batch cycle completed with zero worker crashes
    criterion_b = (
        s3.get("passed", False)
        and not s3.get("worker_crashed", False)
    )
    # (c) All extracted nodes have meta.range_source: 'preflop_gto'
    # This checks range source independently — not gated on strategy quality (d)
    criterion_c = s4.get("uniform_count", 0) == 0 and s4.get("nodes_checked", 0) > 0
    # (d) All extracted nodes pass strategy quality validation
    criterion_d = s4.get("passed", False)

    # (e) Policy worker successfully loads and queries at least 1 node
    active_store_path = phase0_root / ACTIVE_STORE_NAME
    print_info("Probing policy worker (load + query)...")
    probe_result = _probe_policy_worker(active_store_path)
    criterion_e = probe_result.get("passed", False) and probe_result.get("query_ok", False)

    criteria = {
        "a_preflop_tables_valid": {
            "passed": criterion_a,
            "label": "(a) Preflop table validation passed for all default scenarios",
            "detail": s1.get("errors", [])[:3] if not criterion_a else [],
        },
        "b_no_worker_crashes": {
            "passed": criterion_b,
            "label": "(b) At least 1 small-batch cycle completed with zero worker crashes",
            "detail": s3.get("errors", [])[:3] if not criterion_b else [],
        },
        "c_all_gto_range_source": {
            "passed": criterion_c,
            "label": "(c) All extracted nodes have meta.range_source: 'preflop_gto'",
            "detail": (
                [f"uniform_count={s4.get('uniform_count', '?')}, gto_count={s4.get('gto_count', '?')}"]
                if not criterion_c else []
            ),
        },
        "d_strategy_quality": {
            "passed": criterion_d,
            "label": "(d) All extracted nodes pass strategy quality validation",
            "detail": s4.get("errors", [])[:3] if not criterion_d else [],
        },
        "e_policy_worker_lookup": {
            "passed": criterion_e,
            "label": "(e) Policy worker successfully loads and queries at least 1 node",
            "detail": probe_result.get("errors", [])[:3] if not criterion_e else [],
        },
    }

    for _key, crit in criteria.items():
        icon = f"{C.GREEN}PASS{C.RESET}" if crit["passed"] else f"{C.RED}FAIL{C.RESET}"
        print(f"  {icon}  {crit['label']}")
        for d in crit.get("detail", []):
            print(f"         {C.DIM}{d}{C.RESET}")

    all_criteria_pass = all(c["passed"] for c in criteria.values())
    elapsed = time.monotonic() - started

    print(f"\n  Elapsed: {elapsed:.1f}s")

    if all_criteria_pass:
        print(f"\n  {C.GREEN}{C.BOLD}PIPELINE HEALTHY — ready for production loop{C.RESET}")
        nodes_extracted = s3.get("nodes_extracted", 0)
        gto_count = s4.get("gto_count", 0)
        nodes_loaded = probe_result.get("nodes_loaded", 0)
        print(f"  Nodes extracted: {nodes_extracted}  |  GTO-sourced: {gto_count}  |  Store loaded: {nodes_loaded}")
    else:
        failing_labels = [
            c["label"] for c in criteria.values() if not c["passed"]
        ]
        print(f"\n  {C.RED}{C.BOLD}PIPELINE NOT READY{C.RESET}")
        print(f"  {len(failing_labels)} criterion/criteria failed:")
        for label in failing_labels:
            print(f"    • {label}")

    # ── Save report JSON ──────────────────────────────────────────────────
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"pre_loop_validation_{report_timestamp}.json"

    # Sanitize step results for JSON
    json_results = {}
    for key, val in results.items():
        json_results[key] = {
            "passed": val.get("passed", False),
            "errors": val.get("errors", []),
        }
        for extra in ("nodes_checked", "nodes_extracted", "gto_count",
                       "uniform_count", "active_count", "exact_count",
                       "approx_count", "note", "skipped", "worker_crashed"):
            if extra in val:
                json_results[key][extra] = val[extra]

    # Serialize criteria for JSON
    json_criteria = {}
    for key, crit in criteria.items():
        json_criteria[key] = {
            "passed": crit["passed"],
            "label": crit["label"],
            "detail": crit.get("detail", []),
        }

    report = {
        "timestamp": utc_now_iso(),
        "phase0_root": str(phase0_root),
        "batch_size": args.batch_size,
        "cfr_steps": args.cfr_steps,
        "pio_exe": pio_exe or None,
        "elapsed_s": round(elapsed, 1),
        "all_criteria_pass": all_criteria_pass,
        "verdict": "PIPELINE HEALTHY" if all_criteria_pass else "PIPELINE NOT READY",
        "go_no_go_criteria": json_criteria,
        "policy_worker_probe": {
            "passed": probe_result.get("passed", False),
            "nodes_loaded": probe_result.get("nodes_loaded", 0),
            "query_ok": probe_result.get("query_ok", False),
            "spot_key": probe_result.get("spot_key"),
            "range_source": probe_result.get("range_source"),
            "errors": probe_result.get("errors", []),
        },
        "steps": json_results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    return 0 if all_criteria_pass else 1


def main() -> int:
    args = parse_args()
    return run_pre_loop_validation(args)


if __name__ == "__main__":
    raise SystemExit(main())
