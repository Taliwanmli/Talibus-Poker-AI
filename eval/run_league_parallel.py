#!/usr/bin/env python3
"""Run eval/run_league.py in parallel shards and merge part reports."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPORT_TOP_MISSING_DEFAULT = 10
SEED_STRIDE = 1_000_003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel wrapper for eval/run_league.py with merged report output."
    )
    parser.add_argument("--hands", type=int, required=True, help="Total hands to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults from env/cpu heuristic.",
    )
    parser.add_argument(
        "--trace-hands",
        "--trace_hands",
        dest="trace_hands",
        type=int,
        default=0,
        help="Trace first N hands per worker (usually keep 0 for speed).",
    )
    parser.add_argument(
        "--report_json",
        dest="report_json",
        type=Path,
        required=True,
        help="Merged report output path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass --quiet to each run_league worker.",
    )
    parser.add_argument(
        "--hero-mode",
        "--hero_mode",
        dest="hero_mode",
        choices=["worker", "argmax", "sample", "always_fold", "random"],
        default="argmax",
        help="Hero execution mode forwarded to each run_league worker.",
    )
    parser.add_argument(
        "--policy-cmd",
        "--policy_cmd",
        dest="policy_cmd",
        default=os.environ.get("WIPOKER_POLICY_CMD", "node eval/policy_worker.mjs"),
        help="Policy worker command forwarded to each run_league worker.",
    )
    return parser.parse_args()


def parse_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    if parsed <= 0:
        return default
    return parsed


def default_workers() -> int:
    env_workers = os.environ.get("WIPOKER_PARALLEL_WORKERS")
    if env_workers is not None:
        return parse_positive_int(env_workers, 1)
    cpu = os.cpu_count() or 1
    return max(1, min(cpu - 1, 8))


def split_hands(total_hands: int, workers: int) -> list[int]:
    workers = max(1, min(workers, total_hands))
    base = total_hands // workers
    rem = total_hands % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def with_part_suffix(path: Path, index: int) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.part{index}{path.suffix}")
    return Path(str(path) + f".part{index}.json")


def sum_numeric_dicts(items: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)):
                out[key] = int(out.get(key, 0) + value)
    return out


def sum_count_maps(items: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)):
                out[str(key)] = int(out.get(str(key), 0) + value)
    return out


def sum_nested_count_maps(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        for outer_key, inner in row.items():
            if not isinstance(inner, dict):
                continue
            bucket = out.setdefault(str(outer_key), {})
            for inner_key, value in inner.items():
                if isinstance(value, (int, float)):
                    bucket[str(inner_key)] = int(bucket.get(str(inner_key), 0) + value)
    return out


def extract_hands(part: dict[str, Any]) -> int:
    summary = part.get("summary", {}) if isinstance(part, dict) else {}
    raw = summary.get("raw", {}) if isinstance(summary, dict) else {}
    if isinstance(raw, dict) and isinstance(raw.get("hands"), (int, float)):
        return int(raw["hands"])
    config = part.get("config", {}) if isinstance(part, dict) else {}
    if isinstance(config, dict) and isinstance(config.get("hands"), (int, float)):
        return int(config["hands"])
    return 0


def extract_total_bb(block: dict[str, Any]) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get("total_bb")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def weighted_bb_per_100(parts: list[dict[str, Any]], key: str, total_hands: int) -> tuple[float, float]:
    weighted_sum = 0.0
    total_bb = 0.0
    have_total_bb = True
    for part in parts:
        summary = part.get("summary", {}) if isinstance(part, dict) else {}
        block = summary.get(key, {}) if isinstance(summary, dict) else {}
        hands = extract_hands(part)
        if hands <= 0:
            continue
        bb100 = float(block.get("bb_per_100", 0.0)) if isinstance(block, dict) else 0.0
        weighted_sum += bb100 * hands
        block_total_bb = extract_total_bb(block if isinstance(block, dict) else {})
        if block_total_bb is None:
            have_total_bb = False
        else:
            total_bb += block_total_bb

    if total_hands <= 0:
        return 0.0, 0.0
    if have_total_bb:
        merged_bb100 = (total_bb / float(total_hands)) * 100.0
        return merged_bb100, total_bb
    merged_bb100 = weighted_sum / float(total_hands)
    merged_total_bb = (merged_bb100 / 100.0) * float(total_hands)
    return merged_bb100, merged_total_bb


def merge_top_missing(parts: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for part in parts:
        summary = part.get("summary", {}) if isinstance(part, dict) else {}
        diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
        strategy_store = (
            diagnostics.get("strategy_store", {}) if isinstance(diagnostics, dict) else {}
        )
        rows = (
            strategy_store.get("top_missing_spot_keys", [])
            if isinstance(strategy_store, dict)
            else []
        )
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            spot_key = str(row.get("spot_key", "")).strip()
            if not spot_key:
                continue
            count_value = row.get("count", 0)
            try:
                count = int(count_value)
            except Exception:
                count = 0
            if count > 0:
                counts[spot_key] += count

    merged = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_n = max(1, limit)
    return [{"spot_key": spot_key, "count": count} for spot_key, count in merged[:top_n]]


def merge_all_missing(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge all_missing_spot_keys from parallel workers without a limit."""
    counts: Counter[str] = Counter()
    for part in parts:
        summary = part.get("summary", {}) if isinstance(part, dict) else {}
        diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
        strategy_store = (
            diagnostics.get("strategy_store", {}) if isinstance(diagnostics, dict) else {}
        )
        # Try all_missing_spot_keys first, fall back to top_missing_spot_keys
        rows = (
            strategy_store.get("all_missing_spot_keys")
            or strategy_store.get("top_missing_spot_keys", [])
        )
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            spot_key = str(row.get("spot_key", "")).strip()
            if not spot_key:
                continue
            count_value = row.get("count", 0)
            try:
                count = int(count_value)
            except Exception:
                count = 0
            if count > 0:
                counts[spot_key] += count
    merged = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"spot_key": spot_key, "count": count} for spot_key, count in merged]


def merge_top_fuzzy_miss_groups(parts: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for part in parts:
        summary = part.get("summary", {}) if isinstance(part, dict) else {}
        diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
        strategy_store = (
            diagnostics.get("strategy_store", {}) if isinstance(diagnostics, dict) else {}
        )
        rows = (
            strategy_store.get("top_fuzzy_miss_groups", [])
            if isinstance(strategy_store, dict)
            else []
        )
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            group = str(row.get("group", "")).strip()
            if not group:
                continue
            count_value = row.get("count", 0)
            try:
                count = int(count_value)
            except Exception:
                count = 0
            if count > 0:
                counts[group] += count

    merged = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_n = max(1, limit)
    return [{"group": group, "count": count} for group, count in merged[:top_n]]


def merge_report_payloads(parts: list[dict[str, Any]], top_missing_n: int) -> dict[str, Any]:
    if not parts:
        raise ValueError("No part reports to merge.")

    template = copy.deepcopy(parts[0])
    config = template.get("config", {})
    summary = template.get("summary", {})
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("Unsupported report schema in part report.")

    total_hands = sum(extract_hands(part) for part in parts)

    raw_bb100, raw_total_bb = weighted_bb_per_100(parts, "raw", total_hands)
    net_bb100, net_total_bb = weighted_bb_per_100(parts, "net", total_hands)

    raw_block = summary.get("raw", {}) if isinstance(summary.get("raw"), dict) else {}
    net_block = summary.get("net", {}) if isinstance(summary.get("net"), dict) else {}
    diagnostics = (
        summary.get("diagnostics", {})
        if isinstance(summary.get("diagnostics"), dict)
        else {}
    )
    strategy_store = (
        diagnostics.get("strategy_store", {})
        if isinstance(diagnostics.get("strategy_store"), dict)
        else {}
    )

    raw_block["hands"] = total_hands
    raw_block["total_bb"] = raw_total_bb
    raw_block["mean_bb_per_hand"] = (raw_total_bb / float(total_hands)) if total_hands > 0 else 0.0
    raw_block["bb_per_100"] = raw_bb100

    net_block["hands"] = total_hands
    net_block["total_bb"] = net_total_bb
    net_block["mean_bb_per_hand"] = (net_total_bb / float(total_hands)) if total_hands > 0 else 0.0
    net_block["bb_per_100"] = net_bb100

    summary["fallback_count"] = int(
        sum(
            int((part.get("summary", {}) or {}).get("fallback_count", 0))
            for part in parts
            if isinstance(part, dict)
        )
    )
    summary["status_counts"] = sum_numeric_dicts(
        [
            (part.get("summary", {}) or {}).get("status_counts", {})
            for part in parts
            if isinstance(part, dict)
        ]
    )

    diagnostics["street_decisions"] = sum_numeric_dicts(
        [
            ((part.get("summary", {}) or {}).get("diagnostics", {}) or {}).get(
                "street_decisions", {}
            )
            for part in parts
            if isinstance(part, dict)
        ]
    )
    diagnostics["street_ok"] = sum_numeric_dicts(
        [
            ((part.get("summary", {}) or {}).get("diagnostics", {}) or {}).get(
                "street_ok", {}
            )
            for part in parts
            if isinstance(part, dict)
        ]
    )
    diagnostics["street_unavailable"] = sum_numeric_dicts(
        [
            ((part.get("summary", {}) or {}).get("diagnostics", {}) or {}).get(
                "street_unavailable", {}
            )
            for part in parts
            if isinstance(part, dict)
        ]
    )
    diagnostics["street_fallbacks"] = sum_numeric_dicts(
        [
            ((part.get("summary", {}) or {}).get("diagnostics", {}) or {}).get(
                "street_fallbacks", {}
            )
            for part in parts
            if isinstance(part, dict)
        ]
    )

    mode_config_merged: dict[str, Any] = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        diagnostics_part = ((part.get("summary", {}) or {}).get("diagnostics", {}) or {})
        mode_config = diagnostics_part.get("mode_config", {})
        if isinstance(mode_config, dict) and mode_config:
            mode_config_merged = dict(mode_config)
            break
    if mode_config_merged:
        diagnostics["mode_config"] = mode_config_merged

    fidelity_parts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        diagnostics_part = ((part.get("summary", {}) or {}).get("diagnostics", {}) or {})
        fidelity_part = diagnostics_part.get("policy_execution_fidelity", {})
        if isinstance(fidelity_part, dict):
            fidelity_parts.append(fidelity_part)
    if fidelity_parts:
        selection_source_counts = sum_count_maps(
            [row.get("selection_source_counts", {}) for row in fidelity_parts]
        )
        selection_source_by_street = sum_nested_count_maps(
            [row.get("selection_source_by_street", {}) for row in fidelity_parts]
        )
        selection_source_by_street_exact_hit = sum_nested_count_maps(
            [row.get("selection_source_by_street_exact_hit", {}) for row in fidelity_parts]
        )
        selection_source_by_street_non_exact_hit = sum_nested_count_maps(
            [row.get("selection_source_by_street_non_exact_hit", {}) for row in fidelity_parts]
        )
        chosen_vs_executed_match_count = sum(
            int(row.get("chosen_vs_executed_match_count", 0)) for row in fidelity_parts
        )
        chosen_vs_executed_total = sum(
            int(row.get("chosen_vs_executed_total", 0)) for row in fidelity_parts
        )
        chosen_vs_recommended_match_count = sum(
            int(row.get("chosen_vs_recommended_match_count", 0)) for row in fidelity_parts
        )
        chosen_vs_recommended_total = sum(
            int(row.get("chosen_vs_recommended_total", 0)) for row in fidelity_parts
        )
        chosen_vs_argmax_match_count = sum(
            int(row.get("chosen_vs_argmax_match_count", 0)) for row in fidelity_parts
        )
        chosen_vs_argmax_total = sum(
            int(row.get("chosen_vs_argmax_total", 0)) for row in fidelity_parts
        )
        exact_hit_and_translated_count = sum(
            int(row.get("exact_hit_and_translated_count", 0)) for row in fidelity_parts
        )
        exact_hit_and_unmodified_count = sum(
            int(row.get("exact_hit_and_unmodified_count", 0)) for row in fidelity_parts
        )
        exact_hit_translation_total = sum(
            int(row.get("exact_hit_translation_total", 0)) for row in fidelity_parts
        )
        exact_hit_translation_by_street_merged: dict[str, dict[str, Any]] = {}
        for row in fidelity_parts:
            per_street = row.get("exact_hit_translation_by_street", {})
            if not isinstance(per_street, dict):
                continue
            for street, values in per_street.items():
                if not isinstance(values, dict):
                    continue
                bucket = exact_hit_translation_by_street_merged.setdefault(
                    str(street),
                    {"translated": 0, "unmodified": 0, "total": 0, "unmodified_rate": 0.0},
                )
                bucket["translated"] = int(bucket.get("translated", 0)) + int(
                    values.get("translated", 0)
                )
                bucket["unmodified"] = int(bucket.get("unmodified", 0)) + int(
                    values.get("unmodified", 0)
                )
                bucket["total"] = int(bucket.get("total", 0)) + int(values.get("total", 0))
        for values in exact_hit_translation_by_street_merged.values():
            total = int(values.get("total", 0))
            values["unmodified_rate"] = (
                float(values.get("unmodified", 0)) / float(total) if total > 0 else 0.0
            )
        hero_mode_effective = str(
            fidelity_parts[0].get("hero_mode_effective", mode_config_merged.get("hero_mode_effective", "unknown"))
        )
        diagnostics["policy_execution_fidelity"] = {
            "hero_mode_effective": hero_mode_effective,
            "selection_source_counts": selection_source_counts,
            "selection_source_by_street": selection_source_by_street,
            "selection_source_by_street_exact_hit": selection_source_by_street_exact_hit,
            "selection_source_by_street_non_exact_hit": selection_source_by_street_non_exact_hit,
            "chosen_vs_executed_match_count": chosen_vs_executed_match_count,
            "chosen_vs_executed_total": chosen_vs_executed_total,
            "chosen_vs_executed_match_rate": (
                float(chosen_vs_executed_match_count) / float(chosen_vs_executed_total)
                if chosen_vs_executed_total > 0
                else 0.0
            ),
            "chosen_vs_recommended_match_count": chosen_vs_recommended_match_count,
            "chosen_vs_recommended_total": chosen_vs_recommended_total,
            "chosen_vs_recommended_match_rate": (
                float(chosen_vs_recommended_match_count)
                / float(chosen_vs_recommended_total)
                if chosen_vs_recommended_total > 0
                else 0.0
            ),
            "chosen_vs_argmax_match_count": chosen_vs_argmax_match_count,
            "chosen_vs_argmax_total": chosen_vs_argmax_total,
            "chosen_vs_argmax_match_rate": (
                float(chosen_vs_argmax_match_count) / float(chosen_vs_argmax_total)
                if chosen_vs_argmax_total > 0
                else 0.0
            ),
            "exact_hit_and_translated_count": exact_hit_and_translated_count,
            "exact_hit_and_unmodified_count": exact_hit_and_unmodified_count,
            "exact_hit_translation_total": exact_hit_translation_total,
            "exact_hit_unmodified_rate": (
                float(exact_hit_and_unmodified_count) / float(exact_hit_translation_total)
                if exact_hit_translation_total > 0
                else 0.0
            ),
            "exact_hit_translation_by_street": exact_hit_translation_by_street_merged,
        }

    preflop_decisions = int(diagnostics.get("street_decisions", {}).get("preflop", 0))
    preflop_ok = int(diagnostics.get("street_ok", {}).get("preflop", 0))
    preflop_unavailable = int(diagnostics.get("street_unavailable", {}).get("preflop", 0))
    preflop_fallbacks = int(diagnostics.get("street_fallbacks", {}).get("preflop", 0))
    diagnostics["preflop_fallback_unavailable"] = {
        "decisions": preflop_decisions,
        "ok": preflop_ok,
        "unavailable": preflop_unavailable,
        "fallbacks": preflop_fallbacks,
        "fallback_rate": (
            float(preflop_fallbacks) / float(preflop_decisions)
            if preflop_decisions > 0
            else 0.0
        ),
        "unavailable_rate": (
            float(preflop_unavailable) / float(preflop_decisions)
            if preflop_decisions > 0
            else 0.0
        ),
    }

    translation_audit_base: dict[str, Any] = (
        dict(diagnostics.get("action_translation_audit", {}))
        if isinstance(diagnostics.get("action_translation_audit", {}), dict)
        else {}
    )
    translation_audit_parts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        diagnostics_part = ((part.get("summary", {}) or {}).get("diagnostics", {}) or {})
        audit = diagnostics_part.get("action_translation_audit", {})
        if isinstance(audit, dict):
            translation_audit_parts.append(audit)
    if translation_audit_parts:
        for key in (
            "translation_clipped_count",
            "translation_adjusted_count",
            "env_legalized_count",
            "clipped_or_adjusted_total",
            "preflop_raise_clamped_count",
            "preflop_raise_converted_to_allin_count",
            "postflop_raise_clamped_count",
            "raise_clamped_count",
            "exact_requested_size_used_count",
            "postflop_raise_guardrail_relaxed_count",
            "exact_hit_and_translated_count",
            "exact_hit_and_unmodified_count",
            "exact_hit_translation_total",
            "engine_size_capped_count",
            "illegal_action_avoided_count",
            "large_preflop_non_allin_raises_over_40bb_count",
            "preflop_open_non_allin_raises_over_15bb_count",
        ):
            translation_audit_base[key] = int(
                sum(int(row.get(key, 0)) for row in translation_audit_parts)
            )
        if int(translation_audit_base.get("raise_clamped_count", 0)) <= 0:
            translation_audit_base["raise_clamped_count"] = int(
                translation_audit_base.get("preflop_raise_clamped_count", 0)
            ) + int(translation_audit_base.get("postflop_raise_clamped_count", 0))
        diagnostics["action_translation_audit"] = translation_audit_base

    store_hits = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("store_hit_count", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    store_misses = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("store_miss_count", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    total_store_lookups = store_hits + store_misses
    strategy_store["store_hit_count"] = store_hits
    strategy_store["store_miss_count"] = store_misses
    strategy_store["store_hit_rate"] = (
        (float(store_hits) / float(total_store_lookups)) if total_store_lookups > 0 else 0.0
    )
    exact_hits = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("exact_hits", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    exact_misses = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("exact_misses", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    fuzzy_hits = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("fuzzy_hits", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    fuzzy_misses = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("fuzzy_misses", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    fuzzy_attempts = fuzzy_hits + fuzzy_misses
    quality_exact_hits = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("quality_exact_hits", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    quality_approx_hits = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("quality_approx_hits", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    quality_total_hits = quality_exact_hits + quality_approx_hits
    policy_use_approx = any(
        bool(
            (
                (
                    (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                ).get("strategy_store", {})
            ).get("policy_use_approx", False)
        )
        for part in parts
        if isinstance(part, dict)
    )
    policy_hit_count = 0
    policy_hits_exact = 0
    policy_hits_approx = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        diagnostics_part = ((part.get("summary", {}) or {}).get("diagnostics", {}) or {})
        strategy_part = diagnostics_part.get("strategy_store", {})
        if not isinstance(strategy_part, dict):
            continue
        policy_hit_count += int(
            strategy_part.get(
                "policy_hit_count",
                strategy_part.get("quality_exact_hits", 0),
            )
        )
        policy_hits_exact += int(
            strategy_part.get(
                "policy_hits_exact",
                strategy_part.get("quality_exact_hits", 0),
            )
        )
        policy_hits_approx += int(strategy_part.get("policy_hits_approx", 0))
    policy_attempt_count = total_store_lookups
    policy_hit_count = max(0, min(policy_hit_count, policy_attempt_count))
    policy_miss_count = max(0, policy_attempt_count - policy_hit_count)
    strategy_store["exact_hits"] = exact_hits
    strategy_store["exact_misses"] = exact_misses
    strategy_store["fuzzy_hits"] = fuzzy_hits
    strategy_store["fuzzy_misses"] = fuzzy_misses
    strategy_store["quality_exact_hits"] = quality_exact_hits
    strategy_store["quality_approx_hits"] = quality_approx_hits
    strategy_store["quality_exact_hit_rate"] = (
        (float(quality_exact_hits) / float(quality_total_hits))
        if quality_total_hits > 0
        else 0.0
    )
    strategy_store["fuzzy_hit_rate"] = (
        (float(fuzzy_hits) / float(fuzzy_attempts)) if fuzzy_attempts > 0 else 0.0
    )
    strategy_store["overall_hit_rate"] = strategy_store["store_hit_rate"]
    strategy_store["coverage_hit_count_total"] = store_hits
    strategy_store["coverage_hit_count_exact"] = quality_exact_hits
    strategy_store["coverage_hit_count_approx"] = quality_approx_hits
    strategy_store["coverage_miss_count"] = store_misses
    strategy_store["coverage_attempt_count"] = total_store_lookups
    strategy_store["coverage_hit_rate_total"] = strategy_store["store_hit_rate"]
    strategy_store["policy_use_approx"] = policy_use_approx
    strategy_store["policy_hit_count"] = policy_hit_count
    strategy_store["policy_hits_exact"] = policy_hits_exact
    strategy_store["policy_hits_approx"] = policy_hits_approx
    strategy_store["policy_miss_count"] = policy_miss_count
    strategy_store["policy_attempt_count"] = policy_attempt_count
    strategy_store["policy_hit_rate"] = (
        (float(policy_hit_count) / float(policy_attempt_count))
        if policy_attempt_count > 0
        else 0.0
    )
    exact_provenance_match_hits = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("exact_provenance_match_hits", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    exact_provenance_mismatch_hits = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("exact_provenance_mismatch_hits", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    exact_provenance_unknown_hits = int(
        sum(
            int(
                (
                    (
                        (part.get("summary", {}) or {}).get("diagnostics", {}) or {}
                    ).get("strategy_store", {})
                ).get("exact_provenance_unknown_hits", 0)
            )
            for part in parts
            if isinstance(part, dict)
        )
    )
    exact_provenance_tracked = exact_provenance_match_hits + exact_provenance_mismatch_hits
    strategy_store["exact_provenance_match_hits"] = exact_provenance_match_hits
    strategy_store["exact_provenance_mismatch_hits"] = exact_provenance_mismatch_hits
    strategy_store["exact_provenance_unknown_hits"] = exact_provenance_unknown_hits
    strategy_store["exact_provenance_mismatch_rate"] = (
        float(exact_provenance_mismatch_hits) / float(exact_provenance_tracked)
        if exact_provenance_tracked > 0
        else 0.0
    )
    strategy_store["top_missing_spot_keys"] = merge_top_missing(parts, top_missing_n)
    strategy_store["all_missing_spot_keys"] = merge_all_missing(parts)
    strategy_store["top_fuzzy_miss_groups"] = merge_top_fuzzy_miss_groups(parts, top_missing_n)
    enabled_any = False
    node_count_max = 0
    store_load_ms_values: list[float] = []
    store_file_size_bytes_max = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        diagnostics_part = ((part.get("summary", {}) or {}).get("diagnostics", {}) or {})
        strategy_part = diagnostics_part.get("strategy_store", {})
        if not isinstance(strategy_part, dict):
            continue
        enabled_any = enabled_any or bool(strategy_part.get("enabled", False))
        try:
            node_count_max = max(node_count_max, int(strategy_part.get("node_count", 0)))
        except Exception:
            pass
        load_ms_raw = strategy_part.get("store_load_ms")
        if isinstance(load_ms_raw, (int, float)):
            store_load_ms_values.append(float(load_ms_raw))
        size_raw = strategy_part.get("store_file_size_bytes")
        if isinstance(size_raw, (int, float)):
            store_file_size_bytes_max = max(store_file_size_bytes_max, int(size_raw))

    strategy_store["enabled"] = enabled_any
    strategy_store["node_count"] = int(node_count_max)
    strategy_store["store_load_ms"] = (
        float(sum(store_load_ms_values) / len(store_load_ms_values))
        if store_load_ms_values
        else 0.0
    )
    strategy_store["store_file_size_bytes"] = int(store_file_size_bytes_max)
    if not strategy_store.get("file_path"):
        for part in parts:
            if not isinstance(part, dict):
                continue
            candidate = (
                ((part.get("summary", {}) or {}).get("diagnostics", {}) or {}).get(
                    "strategy_store", {}
                )
            )
            if isinstance(candidate, dict):
                path = candidate.get("file_path")
                if isinstance(path, str) and path.strip():
                    strategy_store["file_path"] = path
                    break

    diagnostics["strategy_store"] = strategy_store
    summary["diagnostics"] = diagnostics
    summary["raw"] = raw_block
    summary["net"] = net_block

    config["hands"] = total_hands
    config["seed"] = int(config.get("seed", 0))
    config["seeds"] = [int(config["seed"])]

    template["config"] = config
    template["summary"] = summary
    return template


def launch_workers(
    *,
    repo_root: Path,
    hands_per_worker: list[int],
    base_seed: int,
    trace_hands: int,
    report_json: Path,
    quiet: bool,
    hero_mode: str,
    policy_cmd: str,
) -> list[Path]:
    py_launcher = "py" if os.name == "nt" else sys.executable
    use_py_launcher = py_launcher == "py"
    report_paths: list[Path] = []
    procs: list[tuple[int, int, Path, subprocess.Popen[str]]] = []

    for idx, shard_hands in enumerate(hands_per_worker):
        seed_i = int(base_seed + idx * SEED_STRIDE)
        report_i = with_part_suffix(report_json, idx)
        results_i = report_i.with_suffix(".results.csv")
        trace_i = report_i.with_suffix(".trace.jsonl")
        report_paths.append(report_i)

        if use_py_launcher:
            cmd = [
                "py",
                "-3",
                str(repo_root / "eval" / "run_league.py"),
            ]
        else:
            cmd = [sys.executable, str(repo_root / "eval" / "run_league.py")]
        cmd.extend(
            [
                "--hands",
                str(shard_hands),
                "--seed",
                str(seed_i),
                "--trace_hands",
                str(trace_hands),
                "--report_json",
                str(report_i),
                "--results",
                str(results_i),
                "--trace_file",
                str(trace_i),
                "--hero_mode",
                str(hero_mode),
                "--policy-cmd",
                str(policy_cmd),
            ]
        )
        if quiet:
            cmd.append("--quiet")

        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=os.environ.copy(),
            text=True,
        )
        procs.append((idx, seed_i, report_i, proc))
        print(
            f"[parallel] launched worker={idx} hands={shard_hands} seed={seed_i} "
            f"hero_mode={hero_mode} report={report_i}"
        )

    failures: list[str] = []
    for idx, seed_i, report_i, proc in procs:
        code = proc.wait()
        if code != 0:
            failures.append(
                f"worker={idx} seed={seed_i} report={report_i} exit_code={code}"
            )
        else:
            print(f"[parallel] worker={idx} completed: report={report_i}")

    if failures:
        raise RuntimeError(
            "One or more workers failed:\n" + "\n".join(failures)
        )
    return report_paths


def main() -> int:
    args = parse_args()
    if args.hands <= 0:
        raise ValueError("--hands must be > 0")

    workers = args.workers if args.workers and args.workers > 0 else default_workers()
    workers = max(1, min(workers, args.hands))
    shards = split_hands(args.hands, workers)

    repo_root = Path(__file__).resolve().parents[1]
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    part_reports = launch_workers(
        repo_root=repo_root,
        hands_per_worker=shards,
        base_seed=args.seed,
        trace_hands=args.trace_hands,
        report_json=args.report_json,
        quiet=args.quiet,
        hero_mode=args.hero_mode,
        policy_cmd=args.policy_cmd,
    )

    payloads: list[dict[str, Any]] = []
    for report_path in part_reports:
        if not report_path.exists():
            raise FileNotFoundError(f"Missing part report: {report_path}")
        raw_report = report_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_report)
        except json.JSONDecodeError:
            payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
        payloads.append(payload)

    top_n = parse_positive_int(
        os.environ.get("WIPOKER_STORE_MISSING_TOP_N"), REPORT_TOP_MISSING_DEFAULT
    )
    merged = merge_report_payloads(payloads, top_missing_n=top_n)
    args.report_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    merged_summary = merged.get("summary", {}) if isinstance(merged, dict) else {}
    raw = merged_summary.get("raw", {}) if isinstance(merged_summary, dict) else {}
    diagnostics = (
        merged_summary.get("diagnostics", {})
        if isinstance(merged_summary, dict)
        else {}
    )
    strategy_store = (
        diagnostics.get("strategy_store", {})
        if isinstance(diagnostics, dict)
        else {}
    )

    print(
        "[parallel] merged hands={hands} workers={workers} hero_mode={hero_mode} bb/100={bb:.2f} "
        "store_hits={hits} store_misses={misses} store_hit_rate={rate:.4f}".format(
            hands=args.hands,
            workers=workers,
            hero_mode=args.hero_mode,
            bb=float(raw.get("bb_per_100", 0.0)),
            hits=int(strategy_store.get("store_hit_count", 0)),
            misses=int(strategy_store.get("store_miss_count", 0)),
            rate=float(strategy_store.get("store_hit_rate", 0.0)),
        )
    )
    if any(
        key in strategy_store
        for key in (
            "exact_hits",
            "exact_misses",
            "fuzzy_hits",
            "fuzzy_misses",
            "quality_exact_hits",
            "quality_approx_hits",
        )
    ):
        print(
            "[parallel] strategy_store exact_hits={exact_hits} exact_misses={exact_misses} "
            "fuzzy_hits={fuzzy_hits} fuzzy_misses={fuzzy_misses} "
            "quality_exact_hits={quality_exact_hits} quality_approx_hits={quality_approx_hits} "
            "overall_hit_rate={overall_hit_rate:.4f}".format(
                exact_hits=int(strategy_store.get("exact_hits", 0)),
                exact_misses=int(strategy_store.get("exact_misses", 0)),
                fuzzy_hits=int(strategy_store.get("fuzzy_hits", 0)),
                fuzzy_misses=int(strategy_store.get("fuzzy_misses", 0)),
                quality_exact_hits=int(strategy_store.get("quality_exact_hits", 0)),
                quality_approx_hits=int(strategy_store.get("quality_approx_hits", 0)),
                overall_hit_rate=float(strategy_store.get("overall_hit_rate", 0.0)),
            )
        )
    print(f"[parallel] report_json saved: {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
