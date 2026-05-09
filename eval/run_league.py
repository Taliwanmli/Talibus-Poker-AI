#!/usr/bin/env python3
"""Offline EV league runner for WiPoker decision engine.

This harness is intentionally offline-only. It runs large batches of simulated
cash-game hands and calls the TypeScript recommendation engine through a
persistent Node worker process.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pokerenv.common import Action, GameState, PlayerAction, PlayerState
from pokerenv.table import Table
from treys import Card

try:
    from eval.solver.range_adapter import apply_blockers, parse_range_string_to_combo_weights
    from eval.solver.river_cfr import (
        RiverActionConfig,
        SpotInputRiver,
        canonical_hand_key,
        parse_hand_key,
        solve_river_cfr,
    )
    from eval.solver.river_cfr_teacher import (
        build_top_leaks_summary,
        evaluate_teacher_samples,
    )
except ModuleNotFoundError:
    try:
        from solver.range_adapter import apply_blockers, parse_range_string_to_combo_weights
        from solver.river_cfr import (
            RiverActionConfig,
            SpotInputRiver,
            canonical_hand_key,
            parse_hand_key,
            solve_river_cfr,
        )
        from solver.river_cfr_teacher import build_top_leaks_summary, evaluate_teacher_samples
    except ModuleNotFoundError:
        # Optional river-CFR modules are not shipped in all environments.
        # Keep league benchmarking runnable by providing no-op compatibility shims.
        @dataclass(frozen=True)
        class RiverActionConfig:  # type: ignore[no-redef]
            allow_check: bool = True
            bet_sizes_bb: List[float] = field(default_factory=list)
            allow_allin: bool = False

        @dataclass(frozen=True)
        class SpotInputRiver:  # type: ignore[no-redef]
            board: List[str]
            pot_bb: float
            eff_stack_bb: float
            to_act: str
            hero_range: Dict[str, float]
            villain_range: Dict[str, float]
            actions: RiverActionConfig

        _CARD_PATTERN = re.compile(r"^[2-9TJQKA][cdhsCDHS]$")

        def parse_hand_key(text: str) -> Tuple[str, str]:  # type: ignore[no-redef]
            compact = text.strip().replace(" ", "")
            if len(compact) != 4:
                raise ValueError(f"invalid hand key: {text!r}")
            c1 = compact[:2]
            c2 = compact[2:]
            if not _CARD_PATTERN.match(c1) or not _CARD_PATTERN.match(c2):
                raise ValueError(f"invalid hand cards: {text!r}")
            if c1.upper() == c2.upper():
                raise ValueError(f"duplicate cards in hand key: {text!r}")
            return c1.upper(), c2.upper()

        def canonical_hand_key(cards: Sequence[str]) -> str:  # type: ignore[no-redef]
            if len(cards) != 2:
                raise ValueError("canonical_hand_key requires exactly two cards")
            normalized = [str(cards[0]).upper(), str(cards[1]).upper()]
            if normalized[0] == normalized[1]:
                raise ValueError("duplicate cards in canonical_hand_key")
            normalized.sort()
            return "".join(normalized)

        def parse_range_string_to_combo_weights(range_text: str) -> Dict[str, float]:  # type: ignore[no-redef]
            raw = str(range_text or "").strip()
            if not raw:
                return {}
            out: Dict[str, float] = {}
            for token in raw.split(","):
                part = token.strip()
                if not part:
                    continue
                combo = part
                weight = 1.0
                if ":" in part:
                    combo, weight_text = part.split(":", 1)
                    try:
                        weight = float(weight_text.strip())
                    except ValueError:
                        continue
                try:
                    cards = parse_hand_key(combo.strip())
                except ValueError:
                    continue
                key = f"{cards[0]}{cards[1]}"
                out[key] = max(0.0, out.get(key, 0.0) + max(0.0, weight))
            return out

        def apply_blockers(  # type: ignore[no-redef]
            combo_weights: Dict[str, float],
            blocked_cards: Sequence[str],
        ) -> Dict[str, float]:
            blocked = {str(card).strip().upper() for card in blocked_cards if str(card).strip()}
            if not blocked:
                return dict(combo_weights)
            filtered: Dict[str, float] = {}
            for combo_key, weight in combo_weights.items():
                if not isinstance(weight, (int, float)) or float(weight) <= 0:
                    continue
                try:
                    c1, c2 = parse_hand_key(str(combo_key))
                except ValueError:
                    continue
                if c1 in blocked or c2 in blocked:
                    continue
                filtered[f"{c1}{c2}"] = float(weight)
            return filtered

        def solve_river_cfr(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
            raise RuntimeError("river CFR backend unavailable in this environment")

        def build_top_leaks_summary(  # type: ignore[no-redef]
            rows: Sequence[Mapping[str, Any]],
            top_n: int = 10,
        ) -> Dict[str, List[Dict[str, Any]]]:
            if top_n <= 0 or not rows:
                return {"overall": []}
            return {"overall": []}

        def evaluate_teacher_samples(  # type: ignore[no-redef]
            *,
            samples: Sequence[Mapping[str, Any]],
            iters: int,
            max_samples: int,
        ) -> Dict[str, Any]:
            _ = (iters, max_samples)
            return {
                "rows": [],
                "assumption": "disabled_missing_solver_modules",
                "processed": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "solver_runtime_ms_sum": 0.0,
                "sample_count": len(samples),
            }


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DEFAULT = REPO_ROOT / "eval" / "results.csv"
TRACE_DEFAULT = REPO_ROOT / "eval" / "trace.jsonl"


POSITION_MAP_BY_TABLE_SIZE: Dict[int, List[str]] = {
    2: ["SB", "BB"],
    3: ["SB", "BB", "BTN"],
    4: ["SB", "BB", "CO", "BTN"],
    5: ["SB", "BB", "HJ", "CO", "BTN"],
    6: ["SB", "BB", "LJ", "HJ", "CO", "BTN"],
}
DEEP_CFR_SEAT_BY_LABEL: Dict[str, int] = {
    "UTG": 0,
    "LJ": 0,
    "HJ": 1,
    "CO": 2,
    "BTN": 3,
    "SB": 4,
    "BB": 5,
}
ALL_STREETS: Tuple[str, ...] = ("preflop", "flop", "turn", "river")
POSTFLOP_STREETS: Tuple[str, ...] = ("flop", "turn", "river")
ENGINE_MIN_BET_BB = 1.0
SIZE_BUCKETS: Tuple[Tuple[float, Optional[float], str], ...] = (
    (0.0, 2.0, "0-2"),
    (2.0, 4.0, "2-4"),
    (4.0, 6.0, "4-6"),
    (6.0, 10.0, "6-10"),
    (10.0, None, "10+"),
)
RANK_TO_VALUE: Dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
BROADWAY_RANKS = {"T", "J", "Q", "K", "A"}
HU_FACING_3BET_PREMIUM_HAND_KEYS = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
FLOP_POTAWARE_MEDIUM_COUNTER_KEYS = [
    "flop_potaware_limped_checked_medium_to_minbet_dry",
    "flop_potaware_limped_checked_medium_to_minbet_toppair",
    "flop_potaware_limped_checked_medium_to_minbet_bdfd_overcards",
    "flop_potaware_limped_checked_medium_to_minbet_bdfd_twoovers_AK_rainbow",
    "flop_potaware_limped_checked_medium_to_minbet_bdfd_oneover_A_rainbow",
    "flop_potaware_limped_checked_medium_to_minbet_bdfd_twoovers_rainbow_high",
    "flop_potaware_limped_checked_medium_to_check_bdfd_rejected",
    "flop_potaware_limped_checked_medium_to_check_bdfd_relaxed_rejected",
    "flop_potaware_limped_checked_medium_to_check",
]
RAISED_MEDIUM_TINY_TRACKED_ACTIONS: Tuple[str, ...] = ("check", "min_bet")
RAISED_MEDIUM_TINY_COUNTER_KEYS = [
    "flop_potaware_raised_medium_tiny_action_check",
    "flop_potaware_raised_medium_tiny_action_minbet",
    "flop_potaware_raised_medium_tiny_minbet_prevented",
]
NO_BET_TRACKED_ACTIONS: Tuple[str, ...] = (
    "check",
    "bet33",
    "bet50",
    "min_bet",
    "bet75",
)


@dataclass
class StreetState:
    aggression_count: int = 0
    hero_aggressed: bool = False
    hero_aggression_origin: Optional[str] = None


@dataclass
class HandContext:
    preflop_raise_count: int = 0
    last_preflop_aggressor_id: Optional[int] = None
    hu_preflop_was_limped: bool = False
    hu_preflop_bb_iso_raised: bool = False
    hu_preflop_bb_checked: bool = False
    street_history: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    streets: Dict[str, StreetState] = field(
        default_factory=lambda: {
            "preflop": StreetState(),
            "flop": StreetState(),
            "turn": StreetState(),
            "river": StreetState(),
        }
    )


@dataclass
class EngineDecision:
    status: str
    intent: str
    size_bucket: Optional[str]
    raw: Dict[str, Any]


class PolicyWorkerClient:
    def __init__(self, cmd: str, cwd: Path):
        self.cmd = cmd
        self.cwd = cwd
        self.proc = subprocess.Popen(
            shlex.split(cmd),
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def recommend(self, street: str, hand_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("policy worker stdio not available")

        payload = {"street": street, "handState": hand_state}
        line = json.dumps(payload, separators=(",", ":"))
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

        response_line = self.proc.stdout.readline()
        if response_line == "":
            stderr_tail = ""
            if self.proc.stderr is not None:
                stderr_tail = self.proc.stderr.read().strip()
            raise RuntimeError(
                "policy worker exited unexpectedly"
                + (f" (stderr: {stderr_tail})" if stderr_tail else "")
            )

        try:
            return json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON from policy worker: {response_line!r}") from exc

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)


class BotBase:
    name = "base"

    def choose_intent(self, rng: random.Random, can: Dict[str, bool], to_call: float) -> str:
        raise NotImplementedError


class CallStationBot(BotBase):
    name = "call_station"

    def choose_intent(self, rng: random.Random, can: Dict[str, bool], to_call: float) -> str:
        if to_call > 0 and can["call"]:
            roll = rng.random()
            if roll < 0.78:
                return "call"
            if roll < 0.88 and can["raise"]:
                return "raise"
            if can["fold"]:
                return "fold"
        if can["check"]:
            return "check"
        if can["bet33"] and rng.random() < 0.28:
            return "bet33"
        if can["call"]:
            return "call"
        return "fold" if can["fold"] else "check"


class NitBot(BotBase):
    name = "nit"

    def choose_intent(self, rng: random.Random, can: Dict[str, bool], to_call: float) -> str:
        if to_call > 0:
            if can["fold"] and rng.random() < 0.68:
                return "fold"
            if can["call"]:
                return "call"
        if can["check"] and rng.random() < 0.86:
            return "check"
        if can["bet33"] and rng.random() < 0.11:
            return "bet33"
        if can["call"]:
            return "call"
        return "fold" if can["fold"] else "check"


class LAGBot(BotBase):
    name = "lag"

    def choose_intent(self, rng: random.Random, can: Dict[str, bool], to_call: float) -> str:
        if to_call > 0 and can["raise"] and rng.random() < 0.34:
            return "raise"
        if to_call > 0 and can["call"] and rng.random() < 0.52:
            return "call"
        if to_call > 0 and can["fold"] and rng.random() < 0.22:
            return "fold"
        if can["bet75"] and rng.random() < 0.44:
            return "bet75"
        if can["bet33"] and rng.random() < 0.62:
            return "bet33"
        if can["check"]:
            return "check"
        if can["call"]:
            return "call"
        return "fold" if can["fold"] else "check"


BOT_REGISTRY: Dict[str, BotBase] = {
    "call_station": CallStationBot(),
    "nit": NitBot(),
    "lag": LAGBot(),
}

OPPONENT_ALIASES: Dict[str, str] = {
    "station": "call_station",
    "call_station": "call_station",
    "nit": "nit",
    "lag": "lag",
    "mixed": "mixed",
}


def normalize_seed_u32(seed: int) -> int:
    return int(seed) & 0xFFFF_FFFF


def parse_csv_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_seed_list(raw: Optional[str], fallback_seed: int) -> List[int]:
    fallback = normalize_seed_u32(fallback_seed)
    if not raw:
        return [fallback]
    seeds: List[int] = []
    for token in parse_csv_list(raw):
        try:
            seeds.append(normalize_seed_u32(int(token)))
        except ValueError as exc:
            raise ValueError(f"invalid seed '{token}' in --seeds") from exc
    if not seeds:
        return [fallback]
    return seeds


def resolve_players(match_format: Optional[str], players: int) -> int:
    if match_format == "hu":
        return 2
    if match_format == "6max":
        return 6
    return players


def should_run_matrix_mode(args: argparse.Namespace) -> bool:
    return bool(args.seeds or args.opponents or args.hands_per_seed)


def resolve_opponent_profiles(args: argparse.Namespace, players: int) -> List[str]:
    configured = parse_csv_list(args.opponents)
    if configured:
        return [name.lower() for name in configured]
    if should_run_matrix_mode(args):
        return ["mixed"] if players >= 3 else ["station"]
    return []


def sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.lower())


def resolve_lineup_for_profile(profile: str, players: int, default_lineup: str) -> Tuple[str, str]:
    token = profile.lower()
    mapped = OPPONENT_ALIASES.get(token, token)
    label = token

    if mapped == "mixed":
        return "mixed", default_lineup

    if mapped not in BOT_REGISTRY:
        valid = ", ".join(sorted(OPPONENT_ALIASES.keys()))
        raise ValueError(f"unknown opponent profile '{profile}'. Available: {valid}")

    if players <= 2:
        return label, mapped

    lineup = ",".join([mapped] * (players - 1))
    return label, lineup


def pooled_stats_from_runs(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "runs": 0.0,
            "total_hands": 0.0,
            "mean_bb100": 0.0,
            "ci_low_bb100": 0.0,
            "ci_high_bb100": 0.0,
            "min_bb100": 0.0,
            "max_bb100": 0.0,
        }

    total_hands = int(sum(int(row["hands"]) for row in rows))
    mean_per_hand = (
        sum(float(row["mean_bb_per_hand"]) * int(row["hands"]) for row in rows) / total_hands
        if total_hands > 0
        else 0.0
    )

    pooled_var = 0.0
    if total_hands > 1:
        for row in rows:
            n_i = int(row["hands"])
            if n_i <= 0:
                continue
            mean_i = float(row["mean_bb_per_hand"])
            std_i = float(row["stddev_bb"])
            if n_i > 1:
                pooled_var += (n_i - 1) * (std_i ** 2)
            pooled_var += n_i * ((mean_i - mean_per_hand) ** 2)
        pooled_var /= max(1, total_hands - 1)

    pooled_std = math.sqrt(max(0.0, pooled_var))
    pooled_se = pooled_std / math.sqrt(total_hands) if total_hands > 1 else 0.0
    ci_delta = 1.96 * pooled_se

    bb100_values = [float(row["bb_per_100"]) for row in rows]
    return {
        "runs": float(len(rows)),
        "total_hands": float(total_hands),
        "mean_bb100": mean_per_hand * 100.0,
        "ci_low_bb100": (mean_per_hand - ci_delta) * 100.0,
        "ci_high_bb100": (mean_per_hand + ci_delta) * 100.0,
        "min_bb100": min(bb100_values),
        "max_bb100": max(bb100_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline EV league for WiPoker engine")
    parser.add_argument("--hands", type=int, default=10_000, help="Number of hands to simulate")
    parser.add_argument(
        "--hands-per-seed",
        "--hands_per_seed",
        dest="hands_per_seed",
        type=int,
        default=None,
        help="Hands per run when using --seeds matrix mode",
    )
    parser.add_argument("--players", type=int, default=6, choices=[2, 3, 4, 5, 6], help="Players at table")
    parser.add_argument(
        "--match-format",
        "--match_format",
        dest="match_format",
        choices=["hu", "6max"],
        default=None,
        help="Optional table format override (hu=2 players, 6max=6 players)",
    )
    parser.add_argument("--stack-bb", type=int, default=100, help="Starting stack in bb")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list for matrix mode (e.g. 42,43,44)",
    )
    parser.add_argument("--hero-id", type=int, default=0, help="Tracked hero identifier (0..players-1)")
    parser.add_argument(
        "--hero-mode",
        "--hero_mode",
        dest="hero_mode",
        choices=["worker", "argmax", "sample", "always_fold", "random"],
        default="argmax",
        help=(
            "Hero action mode: worker (respect worker executed action), "
            "argmax (default), sample, always_fold, random"
        ),
    )
    parser.add_argument(
        "--eval-backend",
        "--eval_backend",
        dest="eval_backend",
        choices=["policy_worker", "deep_cfr_panel", "deep_cfr_checkpoint"],
        default="policy_worker",
        help="Evaluation backend: policy_worker (default), deep_cfr_panel, or deep_cfr_checkpoint.",
    )
    parser.add_argument("--policy-cmd", default="node eval/policy_worker.mjs", help="Command to run policy worker")
    parser.add_argument(
        "--deep-cfr-model-state",
        "--deep_cfr_model_state",
        dest="deep_cfr_model_state",
        type=Path,
        default=None,
        help="Deep CFR checkpoint (.pt or .onnx sibling) for deep_cfr_panel/deep_cfr_checkpoint backends.",
    )
    parser.add_argument(
        "--deep-cfr-samples",
        "--deep_cfr_samples",
        dest="deep_cfr_samples",
        type=Path,
        default=None,
        help="Binary panel samples path for deep_cfr_panel backend.",
    )
    parser.add_argument(
        "--deep-cfr-sample-type",
        "--deep_cfr_sample_type",
        dest="deep_cfr_sample_type",
        choices=["strategy", "advantage"],
        default="strategy",
        help="Panel sample type for deep_cfr_panel backend.",
    )
    parser.add_argument(
        "--deep-cfr-max-samples",
        "--deep_cfr_max_samples",
        dest="deep_cfr_max_samples",
        type=int,
        default=50_000,
        help="Maximum panel samples to evaluate for deep_cfr_panel backend.",
    )
    parser.add_argument(
        "--deep-cfr-batch-size",
        "--deep_cfr_batch_size",
        dest="deep_cfr_batch_size",
        type=int,
        default=4096,
        help="Batch size for deep_cfr_panel backend.",
    )
    parser.add_argument(
        "--deep-cfr-device",
        "--deep_cfr_device",
        dest="deep_cfr_device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device request for deep_cfr_panel/deep_cfr_checkpoint backends.",
    )
    parser.add_argument(
        "--deep-cfr-policy-mode",
        "--deep_cfr_policy_mode",
        dest="deep_cfr_policy_mode",
        choices=["argmax", "sample"],
        default="argmax",
        help="Action selection mode for deep_cfr_checkpoint gameplay backend.",
    )
    parser.add_argument(
        "--deep-cfr-policy-seed",
        "--deep_cfr_policy_seed",
        dest="deep_cfr_policy_seed",
        type=int,
        default=17,
        help="Base RNG seed for deep_cfr_checkpoint gameplay backend.",
    )
    parser.add_argument(
        "--deep-cfr-label",
        "--deep_cfr_label",
        dest="deep_cfr_label",
        default="panel_eval",
        help="Optional label included in deep_cfr_panel reports.",
    )
    parser.add_argument(
        "--deep-cfr-required-seats",
        "--deep_cfr_required_seats",
        dest="deep_cfr_required_seats",
        default="",
        help="Comma-separated actor seat IDs required in deep_cfr_panel coverage gate.",
    )
    parser.add_argument(
        "--deep-cfr-min-seat-samples",
        "--deep_cfr_min_seat_samples",
        dest="deep_cfr_min_seat_samples",
        type=int,
        default=0,
        help="Minimum required samples for each required actor seat in deep_cfr_panel.",
    )
    parser.add_argument(
        "--deep-cfr-max-dominant-seat-fraction",
        "--deep_cfr_max_dominant_seat_fraction",
        dest="deep_cfr_max_dominant_seat_fraction",
        type=float,
        default=1.0,
        help="Maximum allowed actor-seat dominance fraction in deep_cfr_panel.",
    )
    parser.add_argument("--build-policy", action="store_true", help="Run npm run eval:build before simulation")
    parser.add_argument("--rake-mode", choices=["none", "fixed", "percent"], default="none", help="Rake model")
    parser.add_argument(
        "--rake-bb-per-hand",
        "--rake_bb_per_hand",
        dest="rake_bb_per_hand",
        type=float,
        default=None,
        help="Alias for fixed rake in bb per hand (overrides --rake-mode to fixed)",
    )
    parser.add_argument("--rake-fixed-bb", type=float, default=0.0, help="Fixed rake in bb per hand")
    parser.add_argument("--rake-percent", type=float, default=0.05, help="Percent rake on pot (0..1)")
    parser.add_argument("--rake-cap-bb", type=float, default=3.0, help="Cap for percent rake in bb")
    parser.add_argument(
        "--open-size",
        "--open_size",
        dest="open_size",
        type=float,
        default=2.5,
        help="Preflop open raise size in bb for Hero adapter",
    )
    parser.add_argument(
        "--threebet-size",
        "--threebet_size",
        dest="threebet_size",
        type=float,
        default=9.0,
        help="Preflop 3-bet size in bb for Hero adapter",
    )
    parser.add_argument(
        "--fourbet-size",
        "--fourbet_size",
        dest="fourbet_size",
        type=float,
        default=22.0,
        help="Preflop 4-bet size in bb for Hero adapter",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default=None,
        help="Comma list of opponent profiles for matrix mode (station,nit,lag,mixed)",
    )
    parser.add_argument(
        "--opponent-lineup",
        default="call_station,nit,lag",
        help="Comma list of opponent bot names to rotate (call_station,nit,lag)",
    )
    parser.add_argument(
        "--max-actions-per-hand",
        type=int,
        default=500,
        help="Safety cap to prevent runaway loops in a single simulated hand",
    )
    parser.add_argument("--progress-every", type=int, default=1_000, help="Progress print cadence")
    parser.add_argument(
        "--runs-dir",
        "--runs_dir",
        dest="runs_dir",
        type=Path,
        default=REPO_ROOT / "eval" / "runs",
        help="Output directory for matrix-mode per-run reports",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT, help="CSV output path")
    parser.add_argument(
        "--trace-hands",
        "--trace_hands",
        dest="trace_hands",
        type=int,
        default=0,
        help="Trace hero decisions for first N hands to eval/trace.jsonl",
    )
    parser.add_argument(
        "--trace-file",
        "--trace_file",
        dest="trace_file",
        type=Path,
        default=TRACE_DEFAULT,
        help="Path to write JSONL decision trace",
    )
    parser.add_argument(
        "--report-json",
        "--report_json",
        dest="report_json",
        type=Path,
        default=None,
        help="Optional path to write full summary metrics JSON",
    )
    parser.add_argument(
        "--dump-biggest-losers",
        "--dump_biggest_losers",
        dest="dump_biggest_losers",
        type=int,
        default=0,
        help="Write K biggest losing hands to JSONL",
    )
    parser.add_argument(
        "--biggest-losers-file",
        "--biggest_losers_file",
        dest="biggest_losers_file",
        type=Path,
        default=REPO_ROOT / "eval" / "biggest_losers.jsonl",
        help="Output file for biggest losing hands JSONL",
    )
    parser.add_argument(
        "--assert-large-preflop-raises-allin",
        "--assert_large_preflop_raises_allin",
        dest="assert_large_preflop_raises_allin",
        action="store_true",
        help="Fail run if any Hero preflop raise >40bb is not all-in",
    )
    parser.add_argument(
        "--assert-preflop-open-max15",
        "--assert_preflop_open_max15",
        dest="assert_preflop_open_max15",
        action="store_true",
        help="Fail run if any Hero preflop OPEN raise >15bb is not all-in",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose per-run summary output (useful in matrix mode)",
    )
    parser.add_argument(
        "--_single-run",
        "--_single_run",
        dest="_single_run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def ensure_repo_commands(args: argparse.Namespace) -> None:
    if args.build_policy:
        subprocess.run(["npm", "run", "eval:build"], cwd=str(REPO_ROOT), check=True)


def patch_pokerenv_compat() -> None:
    if getattr(Table, "_wipoker_patched", False):
        return

    original_reset = Table.reset
    original_step = Table.step
    original_get_valid_actions = Table._get_valid_actions

    def _first_to_act_index_postflop(table: Table) -> int:
        active_indices = [
            i
            for i, player in enumerate(table.players)
            if player.state == PlayerState.ACTIVE and not player.all_in
        ]
        if not active_indices:
            return 0
        start_index = 1 if table.n_players == 2 else 0
        after_start = [i for i in active_indices if i >= start_index]
        if after_start:
            return min(after_start)
        return min(active_indices)

    def _draw_one_card(deck: Any) -> Any:
        drawn = deck.draw(1)
        if isinstance(drawn, list):
            return drawn[0]
        return drawn

    def _is_hu_preflop_blind_only_state(table: Table) -> bool:
        return (
            table.n_players == 2
            and table.street == GameState.PREFLOP
            and bool(getattr(table, "_wipoker_preflop_blind_only", False))
            and float(table.bet_to_match) > 0.0
        )

    def get_valid_actions_fixed(self: Table, player: Any) -> Dict[str, Any]:
        valid_actions = original_get_valid_actions(self, player)
        if not isinstance(valid_actions, dict):
            return valid_actions

        if (
            _is_hu_preflop_blind_only_state(self)
            and float(player.bet_this_street) + 1e-9 >= float(self.bet_to_match)
        ):
            actions_list = list(valid_actions.get("actions_list", []))
            actions_set = set(actions_list)
            if PlayerAction.CALL in actions_set:
                actions_list = [action for action in actions_list if action != PlayerAction.CALL]
                actions_set.discard(PlayerAction.CALL)
            if PlayerAction.FOLD in actions_set:
                actions_list = [action for action in actions_list if action != PlayerAction.FOLD]
                actions_set.discard(PlayerAction.FOLD)
            if PlayerAction.CHECK not in actions_set:
                actions_list.insert(0, PlayerAction.CHECK)
            return {
                "actions_list": actions_list,
                "bet_range": valid_actions.get("bet_range", [0, 0]),
            }
        return valid_actions

    def street_transition_fixed(self: Table, transition_to_end: bool = False) -> None:
        transitioned = False

        if self.street == GameState.PREFLOP:
            self.cards = self.deck.draw(3)
            self._write_event(
                "*** FLOP *** [%s %s %s]"
                % (
                    Card.int_to_str(self.cards[0]),
                    Card.int_to_str(self.cards[1]),
                    Card.int_to_str(self.cards[2]),
                )
            )
            self.street = GameState.FLOP
            transitioned = True

        if self.street == GameState.FLOP and (not transitioned or transition_to_end):
            new_card = _draw_one_card(self.deck)
            self.cards.append(new_card)
            self._write_event(
                "*** TURN *** [%s %s %s] [%s]"
                % (
                    Card.int_to_str(self.cards[0]),
                    Card.int_to_str(self.cards[1]),
                    Card.int_to_str(self.cards[2]),
                    Card.int_to_str(self.cards[3]),
                )
            )
            self.street = GameState.TURN
            transitioned = True

        if self.street == GameState.TURN and (not transitioned or transition_to_end):
            new_card = _draw_one_card(self.deck)
            self.cards.append(new_card)
            self._write_event(
                "*** RIVER *** [%s %s %s %s] [%s]"
                % (
                    Card.int_to_str(self.cards[0]),
                    Card.int_to_str(self.cards[1]),
                    Card.int_to_str(self.cards[2]),
                    Card.int_to_str(self.cards[3]),
                    Card.int_to_str(self.cards[4]),
                )
            )
            self.street = GameState.RIVER
            transitioned = True

        if self.street == GameState.RIVER and (not transitioned or transition_to_end):
            if not self.hand_is_over and self.hand_history_enabled:
                self._write_show_down()
            self.hand_is_over = True

        self.street_finished = False
        self.last_bet_placed_by = None
        self.first_to_act = None
        self.bet_to_match = 0
        self.minimum_raise = 0
        self._wipoker_preflop_blind_only = False
        for player in self.players:
            player.finish_street()

    def reset_fixed(self: Table, *reset_args: Any, **reset_kwargs: Any) -> Any:
        self._wipoker_finished = False
        obs = original_reset(self, *reset_args, **reset_kwargs)
        self._wipoker_preflop_blind_only = (
            self.n_players == 2
            and self.street == GameState.PREFLOP
            and self.last_bet_placed_by is not None
            and float(self.bet_to_match) > 0.0
        )
        return obs

    def step_fixed(self: Table, action: Action) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        pre_action_street = self.street
        pre_action_actor = self.players[self.next_player_i] if not self.hand_is_over else None

        # In HU, completing the SB from 0.5bb to 1bb is a limp, not an aggressive action.
        # Temporarily clear blind-posted aggressor state so BB receives a legal preflop node.
        is_blind_completion_call = (
            _is_hu_preflop_blind_only_state(self)
            and action.action_type == PlayerAction.CALL
            and pre_action_actor is not None
            and float(pre_action_actor.bet_this_street) + 1e-9 < float(self.bet_to_match)
        )
        if is_blind_completion_call:
            self.last_bet_placed_by = None

        obs, rewards, done, info = original_step(self, action)

        if pre_action_street == GameState.PREFLOP and action.action_type == PlayerAction.BET:
            self._wipoker_preflop_blind_only = False

        if not done and not self.hand_is_over:
            players_with_actions = [
                p for p in self.players if p.state == PlayerState.ACTIVE and not p.all_in
            ]
            players_who_should_act = [
                p
                for p in players_with_actions
                if (not p.acted_this_street or p.bet_this_street != self.bet_to_match)
            ]

            waiting_for_bb_response_after_limp = (
                _is_hu_preflop_blind_only_state(self)
                and self.last_bet_placed_by is None
            )
            if (
                len(players_with_actions) >= 2
                and len(players_who_should_act) == 0
                and not waiting_for_bb_response_after_limp
            ):
                self.street_finished = True
                self.next_player_i = _first_to_act_index_postflop(self)
                self._street_transition()
                done = bool(self.hand_is_over)
                obs = (
                    np.zeros(self.observation_space.shape[0])
                    if done
                    else self._get_observation(self.players[self.next_player_i])
                )
                rewards = np.asarray([p.get_reward() for p in sorted(self.players)])

        if done and not getattr(self, "_wipoker_finished", False):
            self._distribute_pot()
            self._finish_hand()
            rewards = np.asarray([p.get_reward() for p in sorted(self.players)], dtype=float)
            self._wipoker_finished = True
        return obs, rewards, done, info

    Table._street_transition = street_transition_fixed  # type: ignore[assignment]
    Table._get_valid_actions = get_valid_actions_fixed  # type: ignore[assignment]
    Table.reset = reset_fixed  # type: ignore[assignment]
    Table.step = step_fixed  # type: ignore[assignment]
    Table._wipoker_patched = True  # type: ignore[attr-defined]


def street_key_from_state(game_state: GameState) -> str:
    if game_state == GameState.PREFLOP:
        return "preflop"
    if game_state == GameState.FLOP:
        return "flop"
    if game_state == GameState.TURN:
        return "turn"
    return "river"


def card_to_code(card_int: int) -> str:
    code = Card.int_to_str(card_int)
    return f"{code[0].upper()}{code[1].lower()}"


def cards_to_codes(cards: Iterable[int]) -> List[str]:
    return [card_to_code(card) for card in cards]


def bucket_size_bb(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "unknown"
    numeric = float(value)
    if numeric < 0:
        return "unknown"
    for lower, upper, label in SIZE_BUCKETS:
        if numeric < lower:
            continue
        if upper is None or numeric < upper:
            return label
    return SIZE_BUCKETS[-1][2]


def classify_preflop_hand_bucket(hero_cards: List[str]) -> str:
    if len(hero_cards) != 2:
        return "unknown"
    first = hero_cards[0]
    second = hero_cards[1]
    if len(first) < 2 or len(second) < 2:
        return "unknown"

    rank1, suit1 = first[0].upper(), first[1].lower()
    rank2, suit2 = second[0].upper(), second[1].lower()
    value1 = RANK_TO_VALUE.get(rank1)
    value2 = RANK_TO_VALUE.get(rank2)
    if value1 is None or value2 is None:
        return "unknown"

    suited = suit1 == suit2
    if rank1 == rank2:
        if value1 <= 6:
            return "pairs_22_66"
        if value1 <= 10:
            return "pairs_77_tt"
        return "pairs_jj_plus"

    both_broadway = rank1 in BROADWAY_RANKS and rank2 in BROADWAY_RANKS
    if both_broadway and suited:
        return "suited_broadways"
    if both_broadway and not suited:
        return "offsuit_broadways"

    high = max(value1, value2)
    low = min(value1, value2)
    is_connector = high - low == 1
    if suited and is_connector and low >= 4 and high <= 11:
        return "suited_connectors_54s_tjs"
    if suited:
        return "suited_trash"
    return "offsuit_trash"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except ValueError:
            value = int(default)
    if minimum is not None and value < minimum:
        return int(minimum)
    return value


POTAWARE_DEBUG_ENABLED = env_flag("WIPOKER_POTAWARE_DEBUG", False)
RIVER_CFR_ENABLED = env_flag("WIPOKER_USE_RIVER_CFR", False)
RIVER_CFR_OVERRIDE_ENABLED = env_flag("WIPOKER_RIVER_CFR_OVERRIDE", False)
RIVER_CFR_ITERS = env_int("WIPOKER_RIVER_CFR_ITERS", 2000, minimum=1)
RIVER_CFR_MAX_CALLS_PER_MATCH = env_int(
    "WIPOKER_RIVER_CFR_MAX_CALLS_PER_MATCH",
    999999,
    minimum=0,
)
RIVER_CFR_DEBUG_ENABLED = env_flag("WIPOKER_RIVER_CFR_DEBUG", False)
RIVER_CFR_TEACHER_ITERS = env_int("WIPOKER_RIVER_CFR_TEACHER_ITERS", 200, minimum=1)
RIVER_CFR_TEACHER_MAX_SAMPLES = env_int("WIPOKER_RIVER_CFR_TEACHER_MAX_SAMPLES", 200, minimum=0)
RIVER_CFR_TEACHER_EVAL_IN_RUN = env_flag("WIPOKER_RIVER_CFR_TEACHER_EVAL_IN_RUN", False)
TEACHER_INCLUDE_SHOWDOWN_CARDS = env_flag("WIPOKER_TEACHER_INCLUDE_SHOWDOWN_CARDS", False)
# Phase-C flags:
# - WIPOKER_EXACT_ACTION_FIDELITY_POSTFLOP=1 (default)
#   Prefer exact worker-requested postflop size when available.
# - WIPOKER_RELAX_EXACT_POSTFLOP_RAISE_GUARDRAIL=1
#   Skip postflop raise guardrail cap for exact-size requests (still clipped to env bounds).
EXACT_ACTION_FIDELITY_POSTFLOP = env_flag("WIPOKER_EXACT_ACTION_FIDELITY_POSTFLOP", True)
RELAX_EXACT_POSTFLOP_RAISE_GUARDRAIL = env_flag(
    "WIPOKER_RELAX_EXACT_POSTFLOP_RAISE_GUARDRAIL",
    False,
)


def hero_cards_to_hand_key(hero_cards: List[str]) -> Optional[str]:
    if len(hero_cards) != 2:
        return None
    first = hero_cards[0]
    second = hero_cards[1]
    if len(first) < 2 or len(second) < 2:
        return None
    rank1, suit1 = first[0].upper(), first[1].lower()
    rank2, suit2 = second[0].upper(), second[1].lower()
    value1 = RANK_TO_VALUE.get(rank1)
    value2 = RANK_TO_VALUE.get(rank2)
    if value1 is None or value2 is None:
        return None
    if rank1 == rank2:
        return f"{rank1}{rank2}"
    if value1 > value2:
        high_rank, low_rank = rank1, rank2
    else:
        high_rank, low_rank = rank2, rank1
    suited = "s" if suit1 == suit2 else "o"
    return f"{high_rank}{low_rank}{suited}"


def is_hu_facing_3bet_premium(hero_cards: List[str]) -> bool:
    hand_key = hero_cards_to_hand_key(hero_cards)
    return hand_key in HU_FACING_3BET_PREMIUM_HAND_KEYS if hand_key else False


def classify_final_outcome_class(
    terminal_street: str,
    showdown: bool,
    hero_folded_postflop: bool,
    hero_reward: float,
) -> str:
    if terminal_street == "preflop":
        return "won_preflop" if hero_reward > 0 else "lost_preflop"
    if showdown:
        return "showdown"
    if hero_folded_postflop:
        return "folded_later"
    return "went_to_flop"


def first_villain_preflop_action_after_index(
    action_sequence: List[Dict[str, Any]],
    hero_action_index: int,
    hero_id: int,
) -> Optional[str]:
    for event in action_sequence[hero_action_index + 1 :]:
        if str(event.get("street")) != "preflop":
            break
        if int(event.get("actor_id", -1)) == int(hero_id):
            break
        action_name = str(event.get("executed_action_type", "")).upper()
        if action_name:
            return action_name
    return None


def classify_villain_response_to_open(
    action_sequence: List[Dict[str, Any]],
    hero_action_index: int,
    hero_id: int,
    terminal_street: str,
) -> str:
    response = first_villain_preflop_action_after_index(action_sequence, hero_action_index, hero_id)
    if response == "FOLD":
        return "villain_folded"
    if response == "CALL":
        return "villain_called"
    if response == "BET":
        return "villain_3bet"
    if response == "CHECK":
        return "villain_checked"
    if terminal_street != "preflop":
        return "advanced_to_flop"
    return "unknown"


def classify_villain_response_to_faced_3bet_after_open(
    hero_action_bucket: str,
    action_sequence: List[Dict[str, Any]],
    hero_action_index: int,
    hero_id: int,
    terminal_street: str,
) -> str:
    if hero_action_bucket == "fold":
        return "hero_folded"
    response = first_villain_preflop_action_after_index(action_sequence, hero_action_index, hero_id)
    if hero_action_bucket == "call":
        if terminal_street != "preflop":
            return "to_flop_after_call"
        return "preflop_ended_after_call"
    if hero_action_bucket == "raise":
        if response == "FOLD":
            return "villain_folded_to_4bet"
        if response == "CALL":
            return "villain_called_4bet"
        if response == "BET":
            return "villain_5bet"
        if terminal_street != "preflop":
            return "to_flop_after_4bet"
        return "preflop_ended_after_4bet"
    return "unknown"


def classify_flop_no_bet_action(
    action_type: PlayerAction,
    chosen_intent: str,
    sent_amount_bb: Optional[float],
    pot_bb: float,
) -> str:
    if action_type == PlayerAction.CHECK:
        return "check"
    if action_type != PlayerAction.BET:
        return action_type.name.lower()

    if sent_amount_bb is not None and math.isfinite(float(sent_amount_bb)):
        bet_amount = float(sent_amount_bb)
    else:
        bet_amount = 0.0

    if bet_amount <= ENGINE_MIN_BET_BB + 1e-9:
        return "min_bet"
    if chosen_intent == "bet75":
        return "bet75"
    if chosen_intent == "bet50":
        return "bet50"
    if chosen_intent == "bet33":
        return "bet33"

    if pot_bb > 0 and math.isfinite(pot_bb):
        fraction = bet_amount / pot_bb
        if fraction >= 0.65:
            return "bet75"
        if fraction >= 0.45:
            return "bet50"
        if fraction <= 0.45:
            return "bet33"
    return "other_bet"


def table_position_to_engine_pos(n_players: int, position_index: int) -> str:
    mapping = POSITION_MAP_BY_TABLE_SIZE.get(n_players, POSITION_MAP_BY_TABLE_SIZE[6])
    if position_index < 0:
        return "BTN"
    if position_index >= len(mapping):
        return mapping[-1]
    return mapping[position_index]


def table_position_to_deep_cfr_seat(n_players: int, position_index: int) -> int:
    label = table_position_to_engine_pos(n_players, position_index)
    return int(DEEP_CFR_SEAT_BY_LABEL.get(label, 0))


def _deep_cfr_history_token_from_event(event: Mapping[str, Any]) -> Optional[str]:
    action_name = str(event.get("executed_action_type", "")).strip().upper()
    if action_name == "FOLD":
        return "f"
    if action_name == "CHECK":
        return "x"
    if action_name == "CALL":
        return "c"

    amount = event.get("executed_amount_bb")
    amount_bb = float(amount) if isinstance(amount, (int, float)) and math.isfinite(float(amount)) else 0.0
    to_call_raw = event.get("to_call_bb_before")
    to_call_bb = (
        float(to_call_raw)
        if isinstance(to_call_raw, (int, float)) and math.isfinite(float(to_call_raw))
        else 0.0
    )
    pot_raw = event.get("pot_bb_before")
    pot_bb = float(pot_raw) if isinstance(pot_raw, (int, float)) and math.isfinite(float(pot_raw)) else 1.0

    if action_name == "BET":
        if to_call_bb > 1e-9:
            ratio = max(1.0, amount_bb / max(to_call_bb, 1e-6))
            return f"r:raise_{ratio:.2f}"
        fraction = max(0.0, amount_bb / max(pot_bb, 1e-6))
        return f"b:bet_{fraction:.2f}"
    return None


def build_deep_cfr_worker_context(
    *,
    table: Table,
    hero_id: int,
    actor_id: int,
    context: HandContext,
    action_sequence: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    players_payload: List[Dict[str, Any]] = []
    for player in table.players:
        players_payload.append(
            {
                "id": int(player.identifier),
                "seat_label": table_position_to_engine_pos(table.n_players, int(player.position)),
                "seat_index": table_position_to_deep_cfr_seat(table.n_players, int(player.position)),
                "folded": bool(player.state != PlayerState.ACTIVE),
                "stack_bb": float(player.stack),
                "total_contribution_bb": float(player.money_in_pot),
                "street_contribution_bb": float(player.bet_this_street),
            }
        )

    compact_history: List[Dict[str, Any]] = []
    for event in action_sequence[-24:]:
        if not isinstance(event, Mapping):
            continue
        actor_raw = event.get("actor_id")
        if not isinstance(actor_raw, int):
            continue
        token = _deep_cfr_history_token_from_event(event)
        if token is None:
            continue
        compact_history.append({"token": token, "actor_id": int(actor_raw)})

    starting_stack_bb = 0.0
    if table.players:
        starting_stack_bb = max(float(p.stack + p.money_in_pot) for p in table.players)

    return {
        "hero_id": int(hero_id),
        "actor_id": int(actor_id),
        "table_size": int(table.n_players),
        "starting_stack_bb": float(starting_stack_bb),
        "preflop_raise_count": int(context.preflop_raise_count),
        "players": players_payload,
        "action_history": compact_history,
    }


def get_player_by_identifier(table: Table, identifier: int):
    for player in table.players:
        if int(player.identifier) == int(identifier):
            return player
    raise RuntimeError(f"player with identifier {identifier} not found")


def get_active_players(table: Table) -> List[Any]:
    return [p for p in table.players if p.state == PlayerState.ACTIVE]


def choose_primary_villain(table: Table, hero_id: int, context: HandContext) -> Any:
    active = [p for p in get_active_players(table) if int(p.identifier) != hero_id]
    if context.last_preflop_aggressor_id is not None:
        for player in active:
            if int(player.identifier) == context.last_preflop_aggressor_id:
                return player
    if active:
        active.sort(key=lambda p: p.position, reverse=True)
        return active[0]
    for player in table.players:
        if int(player.identifier) != hero_id:
            return player
    raise RuntimeError("no villain found")


def infer_preflop_spot(preflop_raise_count: int) -> str:
    if preflop_raise_count <= 0:
        return "UNOPENED"
    if preflop_raise_count == 1:
        return "FACING_OPEN"
    if preflop_raise_count == 2:
        return "FACING_3BET"
    return "FACING_4BET"


def infer_pot_type(preflop_raise_count: int) -> str:
    return "3BP" if preflop_raise_count >= 2 else "SRP"


def classify_postflop_preflop_context(preflop_state: Dict[str, Any]) -> str:
    was_limped = bool(preflop_state.get("wasLimped"))
    bb_iso_raised = bool(preflop_state.get("bbIsoRaised"))
    bb_checked = bool(preflop_state.get("bbChecked"))
    if was_limped and bb_iso_raised:
        return "limped_bb_iso"
    if was_limped and bb_checked:
        return "limped_bb_checked"
    if was_limped:
        return "limped_other"
    return "raised_or_3bp"


def extract_flop_potaware_counter_tags(engine_response_payload: Optional[Dict[str, Any]]) -> List[str]:
    if not POTAWARE_DEBUG_ENABLED:
        return []
    if not isinstance(engine_response_payload, dict):
        return []
    recommended = engine_response_payload.get("recommendedAction")
    if not isinstance(recommended, dict):
        return []
    note = recommended.get("note")
    if not isinstance(note, str):
        return []
    tags: List[str] = []
    for key in FLOP_POTAWARE_MEDIUM_COUNTER_KEYS:
        if key in note:
            tags.append(key)
    return tags


def _extract_bracket_tags_from_note(note: str) -> List[str]:
    tags: List[str] = []
    for match in re.finditer(r"\[([^\]]+)\]", note):
        raw = str(match.group(1))
        for token in re.split(r"[,\s]+", raw):
            normalized = token.strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
    return tags


def _extract_postflop_potaware_counter_tags(
    engine_response_payload: Optional[Dict[str, Any]],
    *,
    street_prefix: str,
) -> List[str]:
    if not POTAWARE_DEBUG_ENABLED:
        return []
    if not isinstance(engine_response_payload, dict):
        return []
    recommended = engine_response_payload.get("recommendedAction")
    if not isinstance(recommended, dict):
        return []
    note = recommended.get("note")
    if not isinstance(note, str):
        return []
    return [tag for tag in _extract_bracket_tags_from_note(note) if tag.startswith(street_prefix)]


def extract_turn_potaware_counter_tags(engine_response_payload: Optional[Dict[str, Any]]) -> List[str]:
    return _extract_postflop_potaware_counter_tags(
        engine_response_payload,
        street_prefix="turn_",
    )


def extract_river_potaware_counter_tags(engine_response_payload: Optional[Dict[str, Any]]) -> List[str]:
    return _extract_postflop_potaware_counter_tags(
        engine_response_payload,
        street_prefix="river_",
    )


def extract_raised_medium_tiny_counter_tags(
    engine_response_payload: Optional[Dict[str, Any]]
) -> List[str]:
    if not isinstance(engine_response_payload, dict):
        return []
    recommended = engine_response_payload.get("recommendedAction")
    if not isinstance(recommended, dict):
        return []
    note = recommended.get("note")
    if not isinstance(note, str):
        return []
    tags: List[str] = []
    for key in RAISED_MEDIUM_TINY_COUNTER_KEYS:
        if key in note:
            tags.append(key)
    return tags


def infer_hero_ip(table: Table, hero_id: int) -> bool:
    active = get_active_players(table)
    if not active:
        return False
    hero = get_player_by_identifier(table, hero_id)
    max_pos = max(player.position for player in active)
    return hero.position == max_pos


def infer_facing_action(street: str, table: Table, hero: Any, context: HandContext) -> str:
    if street == "preflop":
        return "NO_BET"

    to_call = max(0.0, float(table.bet_to_match - hero.bet_this_street))
    if table.bet_to_match <= 1e-9 and to_call <= 1e-9:
        return "NO_BET"

    state = context.streets[street]
    if state.hero_aggressed:
        if state.hero_aggression_origin == "NO_BET":
            return "FACING_CHECK_RAISE"
        return "FACING_RAISE"

    if state.aggression_count >= 2:
        return "FACING_RAISE"
    return "FACING_BET"


def _hu_placeholder_ranges(context: HandContext) -> Tuple[Optional[str], Optional[str]]:
    # Placeholder descriptors for HU plumbing until chart-derived ranges are wired.
    if context.preflop_raise_count <= 0:
        return "wide 32%", "wide 32%"
    if context.preflop_raise_count == 1:
        return "tight 18%", "wide 28%"
    if context.preflop_raise_count == 2:
        return "tight 14%", "tight 16%"
    return "tight 10%", "tight 12%"


def _teacher_action_sizes_bb(pot_bb: float, eff_stack_bb: float) -> List[float]:
    sizes: List[float] = []
    seen: Dict[float, float] = {}
    for fraction in (0.33, 0.75):
        value = min(float(pot_bb) * fraction, float(eff_stack_bb))
        if not math.isfinite(value) or value <= 0:
            continue
        if value < ENGINE_MIN_BET_BB - 1e-9:
            continue
        rounded = round(float(value), 6)
        seen[rounded] = float(value)
    for key in sorted(seen.keys()):
        sizes.append(seen[key])
    return sizes


def _board_texture_bucket(board_cards: Sequence[str]) -> str:
    if len(board_cards) != 5:
        return "unknown"
    ranks = [str(card)[0].upper() for card in board_cards if isinstance(card, str) and len(card) >= 2]
    suits = [str(card)[1].lower() for card in board_cards if isinstance(card, str) and len(card) >= 2]
    if len(ranks) != 5 or len(suits) != 5:
        return "unknown"
    rank_counts: Dict[str, int] = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    suit_counts: Dict[str, int] = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1

    max_rank_count = max(rank_counts.values()) if rank_counts else 1
    max_suit_count = max(suit_counts.values()) if suit_counts else 1

    pairedness = "unpaired"
    if max_rank_count >= 3:
        pairedness = "trips_plus"
    elif max_rank_count == 2:
        pairedness = "paired"

    if max_suit_count >= 4:
        flushness = "four_flush"
    elif max_suit_count == 3:
        flushness = "three_flush"
    elif max_suit_count == 2:
        flushness = "two_tone"
    else:
        flushness = "rainbow"

    rank_values = sorted({RANK_TO_VALUE.get(rank, 0) for rank in ranks}, reverse=True)
    straighty = "disconnected"
    if len(rank_values) >= 4 and (max(rank_values) - min(rank_values) <= 6):
        straighty = "coordinated"

    return f"{pairedness}__{flushness}__{straighty}"


def _normalize_public_action_label(action_name: str) -> str:
    normalized = str(action_name).strip().upper()
    if normalized == "BET":
        return "bet"
    if normalized == "CALL":
        return "call"
    if normalized == "CHECK":
        return "check"
    if normalized == "FOLD":
        return "fold"
    return "other"


def _compact_public_action_history(
    *,
    action_sequence: Sequence[Mapping[str, Any]],
    hero_id: int,
    primary_villain_id: Optional[int],
    max_events: int = 24,
) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    start = max(0, len(action_sequence) - max_events)
    for event in action_sequence[start:]:
        if not isinstance(event, Mapping):
            continue
        actor_id_raw = event.get("actor_id")
        if not isinstance(actor_id_raw, int):
            continue
        actor_label = "other"
        if int(actor_id_raw) == int(hero_id):
            actor_label = "hero"
        elif primary_villain_id is not None and int(actor_id_raw) == int(primary_villain_id):
            actor_label = "villain"

        street = str(event.get("street", "unknown"))
        action = _normalize_public_action_label(str(event.get("executed_action_type", "other")))
        amount_bb = event.get("executed_amount_bb")
        to_call_bb = event.get("to_call_bb_before")
        compact.append(
            {
                "street": street,
                "actor": actor_label,
                "action": action,
                "amountBb": float(amount_bb)
                if isinstance(amount_bb, (int, float)) and math.isfinite(float(amount_bb))
                else 0.0,
                "toCallBbBefore": float(to_call_bb)
                if isinstance(to_call_bb, (int, float)) and math.isfinite(float(to_call_bb))
                else 0.0,
            }
        )
    return compact


def build_river_teacher_sample(
    *,
    hand_index: int,
    table: Table,
    hero_id: int,
    context: HandContext,
    hand_state: Dict[str, Any],
    action_sequence: Sequence[Mapping[str, Any]],
    chosen_action_baseline: str,
) -> Optional[Dict[str, Any]]:
    preflop_state = hand_state.get("preflop", {})
    river_state = hand_state.get("river", {})
    if not isinstance(preflop_state, dict) or not isinstance(river_state, dict):
        return None

    board_cards_raw = river_state.get("boardCards")
    hero_hand_raw = preflop_state.get("heroHand")
    pot_raw = river_state.get("potSizeBb")
    eff_raw = river_state.get("effectiveStackBb")
    if not isinstance(board_cards_raw, list) or len(board_cards_raw) != 5:
        return None
    if not isinstance(hero_hand_raw, list) or len(hero_hand_raw) != 2:
        return None
    if not isinstance(pot_raw, (int, float)) or not isinstance(eff_raw, (int, float)):
        return None

    pot_bb = float(pot_raw)
    eff_stack_bb = float(eff_raw)
    if not math.isfinite(pot_bb) or pot_bb <= 0:
        return None
    if not math.isfinite(eff_stack_bb) or eff_stack_bb <= 0:
        return None

    hero_range = preflop_state.get("heroRange") or preflop_state.get("ranges")
    villain_range = preflop_state.get("villainRange")
    if (not isinstance(hero_range, str) or not hero_range) and int(table.n_players) == 2:
        hero_range, _ = _hu_placeholder_ranges(context)
    if (not isinstance(villain_range, str) or not villain_range) and int(table.n_players) == 2:
        _, villain_range = _hu_placeholder_ranges(context)

    primary_villain_id: Optional[int] = None
    try:
        primary_villain = choose_primary_villain(table, hero_id, context)
        primary_villain_id = int(primary_villain.identifier)
    except Exception:
        primary_villain_id = None

    preflop_context = classify_postflop_preflop_context(preflop_state)
    board_cards = [str(card) for card in board_cards_raw]
    action_history_public = _compact_public_action_history(
        action_sequence=action_sequence,
        hero_id=hero_id,
        primary_villain_id=primary_villain_id,
    )
    range_model_state = {
        "heroRangeLabel": str(hero_range) if isinstance(hero_range, str) else None,
        "villainRangeLabel": str(villain_range) if isinstance(villain_range, str) else None,
        "preflopContext": preflop_context,
        "stateVersion": "range_model_v0_input",
    }

    return {
        "schemaVersion": "spot_v0",
        "handIndex": int(hand_index),
        "boardCards": board_cards,
        "potBb": pot_bb,
        "effStackBb": eff_stack_bb,
        "heroHand": [str(card) for card in hero_hand_raw],
        "preflop": {
            "heroRange": str(hero_range) if isinstance(hero_range, str) else None,
            "villainRange": str(villain_range) if isinstance(villain_range, str) else None,
        },
        "actionSizesBb": _teacher_action_sizes_bb(pot_bb, eff_stack_bb),
        "chosenActionBaseline": str(chosen_action_baseline),
        "preflopContext": preflop_context,
        "boardTextureBucket": _board_texture_bucket(board_cards),
        "actionHistoryPublic": action_history_public,
        "rangeModelState": range_model_state,
        "assumptionTag": "range_labels_only",
        "spot": {
            "street": "river",
            "boardCards": board_cards,
            "heroHand": [str(card) for card in hero_hand_raw],
            "potBb": pot_bb,
            "effStackBb": eff_stack_bb,
            "actionSizesBb": _teacher_action_sizes_bb(pot_bb, eff_stack_bb),
            "preflopContext": preflop_context,
            "boardTextureBucket": _board_texture_bucket(board_cards),
            "actionHistoryPublic": action_history_public,
            "rangeModelState": range_model_state,
        },
    }


def maybe_attach_showdown_villain_cards_metadata(
    *,
    hand_metrics: Dict[str, Any],
    table: Table,
    hero_id: int,
    context: HandContext,
) -> None:
    if not TEACHER_INCLUDE_SHOWDOWN_CARDS:
        return
    if not bool(hand_metrics.get("showdown")):
        return
    samples = hand_metrics.get("river_cfr_teacher_samples")
    if not isinstance(samples, list) or not samples:
        return
    try:
        villain = choose_primary_villain(table, hero_id, context)
        candidate = cards_to_codes(villain.cards)
    except Exception:
        return
    if len(candidate) != 2:
        return
    villain_cards = [str(candidate[0]), str(candidate[1])]
    for sample in samples:
        if isinstance(sample, dict):
            sample["villainKnownCardsIfAny"] = list(villain_cards)
            sample["villainCardsSource"] = "showdown_revealed"


def build_hand_state(
    table: Table,
    hero_id: int,
    context: HandContext,
    *,
    include_range_context: bool = False,
) -> Dict[str, Any]:
    hero = get_player_by_identifier(table, hero_id)
    villain = choose_primary_villain(table, hero_id, context)
    street = street_key_from_state(table.street)
    hero_stack_remaining = max(0.0, float(hero.stack))
    active_villain_stacks = [
        max(0.0, float(player.stack))
        for player in get_active_players(table)
        if int(player.identifier) != int(hero_id)
    ]
    if not active_villain_stacks:
        active_villain_stacks = [
            max(0.0, float(player.stack))
            for player in table.players
            if int(player.identifier) != int(hero_id)
        ]
    effective_stack_remaining = (
        min([hero_stack_remaining, *active_villain_stacks])
        if active_villain_stacks
        else hero_stack_remaining
    )

    board_cards = cards_to_codes(table.cards)
    flop_cards = board_cards[:3]
    turn_cards = board_cards[:4]
    river_cards = board_cards[:5]

    hand_state: Dict[str, Any] = {
        "setup": {
            "gameType": "cash",
            "tableSize": 6,
            "effectiveStackBb": float(
                min(player.stack + player.money_in_pot for player in table.players)
            ),
            "numPlayersInHand": int(sum(1 for p in table.players if p.state == PlayerState.ACTIVE)),
        },
        "preflop": {
            "heroHand": cards_to_codes(hero.cards),
            "heroPos": table_position_to_engine_pos(table.n_players, hero.position),
            "villainPos": table_position_to_engine_pos(table.n_players, villain.position),
            "spot": infer_preflop_spot(context.preflop_raise_count),
            "toCallBb": max(0.0, float(table.bet_to_match - hero.bet_this_street)),
            "potType": infer_pot_type(context.preflop_raise_count),
            "wasLimped": bool(context.hu_preflop_was_limped),
            "bbIsoRaised": bool(context.hu_preflop_bb_iso_raised),
            "bbChecked": bool(context.hu_preflop_bb_checked),
        },
        "flop": {
            "boardCards": flop_cards,
            "isHeroIP": infer_hero_ip(table, hero_id),
            "facingAction": "NO_BET",
            "potSizeBb": float(table.pot),
            "toCallBb": max(0.0, float(table.bet_to_match - hero.bet_this_street)),
            "heroStackBb": hero_stack_remaining,
            "effectiveStackBb": effective_stack_remaining,
        },
        "turn": {
            "boardCards": turn_cards,
            "facingAction": "NO_BET",
            "potSizeBb": float(table.pot),
            "toCallBb": max(0.0, float(table.bet_to_match - hero.bet_this_street)),
            "heroStackBb": hero_stack_remaining,
            "effectiveStackBb": effective_stack_remaining,
        },
        "river": {
            "boardCards": river_cards,
            "facingAction": "NO_BET",
            "potSizeBb": float(table.pot),
            "toCallBb": max(0.0, float(table.bet_to_match - hero.bet_this_street)),
            "heroStackBb": hero_stack_remaining,
            "effectiveStackBb": effective_stack_remaining,
        },
    }

    if include_range_context and int(table.n_players) == 2:
        hero_range, villain_range = _hu_placeholder_ranges(context)
        if hero_range:
            hand_state["preflop"]["heroRange"] = hero_range
        if villain_range:
            hand_state["preflop"]["villainRange"] = villain_range

    if context.street_history:
        hand_state["streetHistory"] = dict(context.street_history)

    if street == "flop":
        hand_state["flop"]["facingAction"] = infer_facing_action(street, table, hero, context)
    elif street == "turn":
        hand_state["turn"]["facingAction"] = infer_facing_action(street, table, hero, context)
    elif street == "river":
        hand_state["river"]["facingAction"] = infer_facing_action(street, table, hero, context)

    return hand_state


def is_finite_prob(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def sample_from_mix(mix: Dict[str, Any], rng: random.Random) -> Optional[str]:
    cleaned: List[Tuple[str, float]] = []
    for key, value in mix.items():
        if is_finite_prob(value):
            cleaned.append((key, float(value)))

    if not cleaned:
        return None

    total = sum(value for _, value in cleaned)
    if total <= 0:
        return None

    target = rng.random() * total
    running = 0.0
    for key, value in cleaned:
        running += value
        if running >= target:
            return key

    return cleaned[-1][0]


def argmax_mix(mix: Dict[str, Any]) -> Optional[str]:
    best_key: Optional[str] = None
    best_value = -1.0
    for key, value in mix.items():
        if not is_finite_prob(value):
            continue
        numeric = float(value)
        if numeric > best_value:
            best_value = numeric
            best_key = key
    return best_key


def _finite_positive_size(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric > 0:
            return numeric
    return None


def _intent_from_action_payload(
    action_payload: Any,
) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float], Optional[float], bool]:
    action_type = None
    action_note = ""
    requested_size_bb: Optional[float] = None
    engine_proposed_bb: Optional[float] = None
    engine_final_bb: Optional[float] = None
    was_capped_in_engine = False
    if isinstance(action_payload, dict):
        action_type = action_payload.get("type")
        note = action_payload.get("note")
        if isinstance(note, str):
            action_note = note.lower()
        requested_size_bb = _finite_positive_size(action_payload.get("sizeBb"))
        engine_proposed_bb = _finite_positive_size(action_payload.get("engineProposedBb"))
        engine_final_bb = _finite_positive_size(action_payload.get("engineFinalBb"))
        was_capped_in_engine = bool(action_payload.get("wasCappedInEngine"))

    if action_type == "check":
        return "check", None, None, engine_proposed_bb, engine_final_bb, was_capped_in_engine
    if action_type == "fold":
        return "fold", None, None, engine_proposed_bb, engine_final_bb, was_capped_in_engine
    if action_type == "call":
        return "call", None, None, engine_proposed_bb, engine_final_bb, was_capped_in_engine
    if action_type == "raise":
        return "raise", None, requested_size_bb, engine_proposed_bb, engine_final_bb, was_capped_in_engine
    if action_type == "all_in":
        return (
            "all_in",
            None,
            requested_size_bb,
            engine_proposed_bb,
            engine_final_bb,
            was_capped_in_engine,
        )
    if action_type == "bet":
        if "150" in action_note:
            return (
                "bet150",
                "150",
                requested_size_bb,
                engine_proposed_bb,
                engine_final_bb,
                was_capped_in_engine,
            )
        if "75" in action_note:
            return (
                "bet75",
                "75",
                requested_size_bb,
                engine_proposed_bb,
                engine_final_bb,
                was_capped_in_engine,
            )
        if "50" in action_note:
            return (
                "bet50",
                "50",
                requested_size_bb,
                engine_proposed_bb,
                engine_final_bb,
                was_capped_in_engine,
            )
        return (
            "bet33",
            "33",
            requested_size_bb,
            engine_proposed_bb,
            engine_final_bb,
            was_capped_in_engine,
        )
    return None, None, None, engine_proposed_bb, engine_final_bb, was_capped_in_engine


def _intent_from_result_action(
    result: Dict[str, Any],
    key: str,
) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float], Optional[float], bool]:
    action_payload = result.get(key)
    return _intent_from_action_payload(action_payload)


def _intent_from_recommended_action(
    result: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float], Optional[float], bool]:
    return _intent_from_result_action(result, "recommendedAction")


def _normalize_intent_for_match(intent: Optional[str]) -> Optional[str]:
    if intent is None:
        return None
    normalized = str(intent).strip().lower()
    if normalized in {"bet33", "bet50", "bet75", "bet150"}:
        return "bet"
    return normalized


def intent_from_engine_result(
    result: Dict[str, Any],
    rng: random.Random,
    hero_mode: str,
) -> Tuple[str, Optional[str], Optional[float], Optional[float], Optional[float], bool, Dict[str, Any]]:
    mix = result.get("mix")
    size_bucket: Optional[str] = None
    intent: Optional[str] = None
    requested_size_bb: Optional[float] = None
    engine_proposed_bb: Optional[float] = None
    engine_final_bb: Optional[float] = None
    was_capped_in_engine = False
    selection_source = "fallback"
    fallback_reason = ""
    mix_branch = "none"

    (
        executed_intent,
        executed_bucket,
        executed_size,
        executed_proposed,
        executed_final,
        executed_capped,
    ) = _intent_from_result_action(result, "executedAction")
    (
        rec_intent,
        rec_bucket,
        rec_size,
        rec_proposed,
        rec_final,
        rec_capped,
    ) = _intent_from_recommended_action(result)
    (
        argmax_intent_from_payload,
        _argmax_bucket,
        _argmax_size,
        _argmax_proposed,
        _argmax_final,
        _argmax_capped,
    ) = _intent_from_result_action(result, "argmaxAction")

    if hero_mode == "worker":
        if executed_intent is not None:
            intent = executed_intent
            size_bucket = executed_bucket
            requested_size_bb = executed_size
            engine_proposed_bb = executed_proposed
            engine_final_bb = executed_final
            was_capped_in_engine = executed_capped
            selection_source = "worker_executed"
        elif rec_intent is not None:
            intent = rec_intent
            size_bucket = rec_bucket
            requested_size_bb = rec_size
            engine_proposed_bb = rec_proposed
            engine_final_bb = rec_final
            was_capped_in_engine = rec_capped
            selection_source = "worker_recommended"

    if intent is None and isinstance(mix, dict) and mix:
        if hero_mode == "sample":
            intent = sample_from_mix(mix, rng)
            mix_branch = "sample"
        else:
            intent = argmax_mix(mix)
            mix_branch = "argmax"
        if intent is not None:
            selection_source = "evaluator_mix_reselection"
            if intent in ("bet33", "bet50", "bet75"):
                if intent == "bet33":
                    size_bucket = "33"
                elif intent == "bet50":
                    size_bucket = "50"
                else:
                    size_bucket = "75"
                requested_size_bb = rec_size
                engine_proposed_bb = rec_proposed
                engine_final_bb = rec_final
                was_capped_in_engine = rec_capped

    if intent is None and rec_intent is not None:
        intent = rec_intent
        size_bucket = rec_bucket
        requested_size_bb = rec_size
        engine_proposed_bb = rec_proposed
        engine_final_bb = rec_final
        was_capped_in_engine = rec_capped
        selection_source = "worker_recommended"

    if intent is None and executed_intent is not None:
        intent = executed_intent
        size_bucket = executed_bucket
        requested_size_bb = executed_size
        engine_proposed_bb = executed_proposed
        engine_final_bb = executed_final
        was_capped_in_engine = executed_capped
        selection_source = "worker_executed"

    if intent is None:
        intent = "check"
        fallback_reason = "no_valid_worker_or_mix_action"
        selection_source = "fallback"

    normalized_chosen = _normalize_intent_for_match(intent)
    normalized_worker_executed = _normalize_intent_for_match(executed_intent)
    normalized_worker_recommended = _normalize_intent_for_match(rec_intent)
    normalized_worker_argmax = _normalize_intent_for_match(argmax_intent_from_payload)

    selection_meta = {
        "hero_mode_requested": str(hero_mode),
        "selection_source": selection_source,
        "mix_branch": mix_branch,
        "fallback_reason": fallback_reason,
        "worker_executed_intent": executed_intent,
        "worker_recommended_intent": rec_intent,
        "worker_argmax_intent": argmax_intent_from_payload,
        "chosen_intent": intent,
        "chosen_matches_worker_executed": (
            normalized_worker_executed is not None
            and normalized_chosen == normalized_worker_executed
        ),
        "chosen_matches_worker_recommended": (
            normalized_worker_recommended is not None
            and normalized_chosen == normalized_worker_recommended
        ),
        "chosen_matches_worker_argmax": (
            normalized_worker_argmax is not None
            and normalized_chosen == normalized_worker_argmax
        ),
    }

    return (
        intent,
        size_bucket,
        requested_size_bb,
        engine_proposed_bb,
        engine_final_bb,
        was_capped_in_engine,
        selection_meta,
    )


def intent_from_safe_fallback(facing_action: str, can: Dict[str, bool]) -> Tuple[str, Optional[str]]:
    if facing_action in ("FACING_BET", "FACING_RAISE", "FACING_CHECK_RAISE"):
        if can["fold"]:
            return "fold", None
        if can["call"]:
            return "call", None
        if can["check"]:
            return "check", None
        if can["raise"]:
            return "raise", None
        return "check", None

    if can["check"]:
        return "check", None
    if can["fold"]:
        return "fold", None
    if can["call"]:
        return "call", None
    if can["raise"]:
        return "raise", None
    return "check", None


def intent_from_always_fold_mode(can: Dict[str, bool]) -> Tuple[str, Optional[str]]:
    if can["check"]:
        return "check", None
    if can["fold"]:
        return "fold", None
    if can["call"]:
        return "call", None
    if can["raise"]:
        return "raise", None
    return "check", None


def intent_from_random_mode(
    can: Dict[str, bool],
    to_call: float,
    rng: random.Random,
) -> Tuple[str, Optional[str]]:
    options: List[Tuple[str, Optional[str]]] = []
    if can["check"]:
        options.append(("check", None))
    if can["fold"]:
        options.append(("fold", None))
    if can["call"]:
        options.append(("call", None))
    if can["raise"]:
        if to_call > 1e-9:
            options.append(("raise", None))
        else:
            options.append(("bet33", "33"))
            options.append(("bet50", "50"))
            options.append(("bet75", "75"))

    if not options:
        return "check", None
    return options[rng.randrange(len(options))]


def compact_mix(mix: Any) -> Optional[Dict[str, float]]:
    if not isinstance(mix, dict):
        return None

    compact: Dict[str, float] = {}
    for key, value in mix.items():
        if is_finite_prob(value):
            compact[str(key)] = round(float(value), 6)
    return compact or None


def _canonical_combo_key_from_cards(cards: Sequence[str]) -> Optional[str]:
    if len(cards) != 2:
        return None
    raw = f"{cards[0]}{cards[1]}"
    try:
        parsed = parse_hand_key(raw)
        return canonical_hand_key(parsed)
    except ValueError:
        return None


def _canonical_combo_weight_map(raw_weights: Dict[str, float]) -> Dict[str, float]:
    canonical: Dict[str, float] = {}
    for combo_text, raw_weight in raw_weights.items():
        if not is_finite_prob(raw_weight) or float(raw_weight) <= 0:
            continue
        try:
            cards = parse_hand_key(str(combo_text))
            key = canonical_hand_key(cards)
        except ValueError:
            continue
        canonical[key] = canonical.get(key, 0.0) + float(raw_weight)
    return canonical


def _weighted_average_solver_mix(
    root_action_mix: Dict[str, Dict[str, float]],
    hero_weights_canonical: Dict[str, float],
) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    total_weight = 0.0
    for hand_key, mix in root_action_mix.items():
        weight = float(hero_weights_canonical.get(hand_key, 0.0))
        if weight <= 0:
            continue
        total_weight += weight
        for action_key, raw_prob in mix.items():
            if not is_finite_prob(raw_prob):
                continue
            aggregated[action_key] = aggregated.get(action_key, 0.0) + (weight * float(raw_prob))
    if total_weight <= 0:
        return {}
    normalized = {key: value / total_weight for key, value in aggregated.items()}
    total = sum(normalized.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in normalized.items()}


def _parse_solver_bet_size_from_label(action_label: str) -> Optional[float]:
    if not action_label.startswith("bet_") or not action_label.endswith("bb"):
        return None
    number_text = action_label[len("bet_") : -len("bb")]
    try:
        size = float(number_text)
    except ValueError:
        return None
    if not math.isfinite(size) or size <= 0:
        return None
    return size


def maybe_apply_river_cfr_override(
    *,
    street: str,
    facing_action: str,
    hand_state: Dict[str, Any],
    base_response: Dict[str, Any],
    hand_metrics: Dict[str, Any],
    river_cfr_config: Dict[str, Any],
    river_cfr_state: Dict[str, Any],
    hand_index: int,
) -> Dict[str, Any]:
    if not bool(river_cfr_config.get("override_enabled")):
        return base_response
    if street != "river" or facing_action != "NO_BET":
        return base_response
    if int(river_cfr_state.get("calls", 0)) >= int(river_cfr_config.get("max_calls_per_match", 0)):
        return base_response

    preflop_state = hand_state.get("preflop", {})
    river_state = hand_state.get("river", {})
    if not isinstance(preflop_state, dict) or not isinstance(river_state, dict):
        hand_metrics["river_cfr_skipped_no_range"] += 1
        return base_response

    hero_range_str = preflop_state.get("heroRange") or preflop_state.get("ranges")
    villain_range_str = preflop_state.get("villainRange")
    if not isinstance(hero_range_str, str) or not isinstance(villain_range_str, str):
        hand_metrics["river_cfr_skipped_no_range"] += 1
        return base_response

    hero_combo_weights = parse_range_string_to_combo_weights(hero_range_str)
    villain_combo_weights = parse_range_string_to_combo_weights(villain_range_str)
    if not hero_combo_weights or not villain_combo_weights:
        hand_metrics["river_cfr_parse_fail"] += 1
        return base_response

    board_cards_raw = river_state.get("boardCards", [])
    hero_cards_raw = preflop_state.get("heroHand", [])
    if not isinstance(board_cards_raw, list) or len(board_cards_raw) != 5:
        hand_metrics["river_cfr_skipped_no_range"] += 1
        return base_response
    if not isinstance(hero_cards_raw, list) or len(hero_cards_raw) != 2:
        hand_metrics["river_cfr_skipped_no_range"] += 1
        return base_response

    board_cards = [str(card) for card in board_cards_raw]
    hero_cards = [str(card) for card in hero_cards_raw]

    hero_combo_weights = apply_blockers(hero_combo_weights, board_cards)
    villain_combo_weights = apply_blockers(villain_combo_weights, [*board_cards, *hero_cards])
    if not hero_combo_weights or not villain_combo_weights:
        hand_metrics["river_cfr_skipped_no_range"] += 1
        return base_response

    hero_combo_key = _canonical_combo_key_from_cards(hero_cards)
    if hero_combo_key is None:
        hand_metrics["river_cfr_parse_fail"] += 1
        return base_response
    hero_combo_weights[f"{hero_cards[0]}{hero_cards[1]}"] = max(
        float(hero_combo_weights.get(f"{hero_cards[0]}{hero_cards[1]}", 0.0)),
        1.0,
    )

    pot_bb_raw = river_state.get("potSizeBb")
    eff_stack_bb_raw = river_state.get("effectiveStackBb")
    hero_stack_bb_raw = river_state.get("heroStackBb")
    if (
        not isinstance(pot_bb_raw, (int, float))
        or not isinstance(eff_stack_bb_raw, (int, float))
    ):
        hand_metrics["river_cfr_skipped_no_range"] += 1
        return base_response
    pot_bb = float(pot_bb_raw)
    eff_stack_bb = float(eff_stack_bb_raw)
    if not math.isfinite(pot_bb) or pot_bb <= 0 or not math.isfinite(eff_stack_bb) or eff_stack_bb <= 0:
        hand_metrics["river_cfr_skipped_no_range"] += 1
        return base_response
    hero_stack_cap = float(hero_stack_bb_raw) if isinstance(hero_stack_bb_raw, (int, float)) else eff_stack_bb
    if not math.isfinite(hero_stack_cap) or hero_stack_cap <= 0:
        hero_stack_cap = eff_stack_bb
    max_bet_cap = max(0.0, min(eff_stack_bb, hero_stack_cap))
    if max_bet_cap <= 0:
        hand_metrics["river_cfr_skipped_no_range"] += 1
        return base_response

    seen_bet_sizes: Dict[float, float] = {}
    for fraction in (0.33, 0.75):
        size = min(pot_bb * fraction, max_bet_cap)
        if not math.isfinite(size) or size <= 0:
            continue
        if size < ENGINE_MIN_BET_BB - 1e-9:
            continue
        rounded = round(size, 6)
        seen_bet_sizes[rounded] = size
    bet_sizes_bb = [seen_bet_sizes[key] for key in sorted(seen_bet_sizes.keys())]
    if not bet_sizes_bb:
        return base_response

    hero_canonical_weights = _canonical_combo_weight_map(hero_combo_weights)
    if not hero_canonical_weights:
        hand_metrics["river_cfr_parse_fail"] += 1
        return base_response

    spot = SpotInputRiver(
        board=board_cards,
        pot_bb=pot_bb,
        eff_stack_bb=eff_stack_bb,
        to_act="hero",
        hero_range=hero_combo_weights,
        villain_range=villain_combo_weights,
        actions=RiverActionConfig(
            allow_check=True,
            bet_sizes_bb=bet_sizes_bb,
            allow_allin=False,
        ),
    )

    started = time.perf_counter()
    try:
        solve_result = solve_river_cfr(
            spot,
            iters=int(river_cfr_config.get("override_iters", 2000)),
            seed=int(hand_index),
        )
    except Exception:
        hand_metrics["river_cfr_parse_fail"] += 1
        return base_response
    runtime_ms = (time.perf_counter() - started) * 1000.0

    root_mix_by_hand = solve_result.hero_strategy.root_action_mix
    hero_root_mix = root_mix_by_hand.get(hero_combo_key)
    if not isinstance(hero_root_mix, dict):
        hero_root_mix = _weighted_average_solver_mix(root_mix_by_hand, hero_canonical_weights)
    if not hero_root_mix:
        hand_metrics["river_cfr_parse_fail"] += 1
        return base_response

    selected_label = argmax_mix(hero_root_mix)
    if selected_label is None:
        hand_metrics["river_cfr_parse_fail"] += 1
        return base_response

    selected_intent = "check"
    selected_size_bb: Optional[float] = None
    if selected_label != "check":
        parsed_size = _parse_solver_bet_size_from_label(selected_label)
        if parsed_size is None:
            hand_metrics["river_cfr_parse_fail"] += 1
            return base_response
        selected_size_bb = parsed_size
        ratio = selected_size_bb / pot_bb if pot_bb > 1e-9 else 0.0
        if ratio <= 0.41:
            selected_intent = "bet33"
        elif ratio <= 0.625:
            selected_intent = "bet50"
        else:
            selected_intent = "bet75"

    one_hot_mix = {"check": 0.0, "bet33": 0.0, "bet50": 0.0, "bet75": 0.0}
    one_hot_mix[selected_intent] = 1.0

    recommended_action: Dict[str, Any]
    if selected_intent == "check":
        recommended_action = {"type": "check", "note": "[river_cfr_v1] check root=check"}
    else:
        recommended_action = {
            "type": "bet",
            "sizeBb": float(selected_size_bb) if selected_size_bb is not None else None,
            "engineProposedBb": float(selected_size_bb) if selected_size_bb is not None else None,
            "engineFinalBb": float(selected_size_bb) if selected_size_bb is not None else None,
            "wasCappedInEngine": False,
            "note": f"[river_cfr_v1] {selected_intent} root={selected_label}",
        }

    river_cfr_state["calls"] = int(river_cfr_state.get("calls", 0)) + 1
    hand_metrics["river_cfr_calls"] += 1
    hand_metrics["river_cfr_runtime_ms_sum"] += runtime_ms

    debug_payload: Optional[Dict[str, Any]] = None
    if bool(river_cfr_config.get("debug")):
        debug_payload = {
            "applied": True,
            "iters": int(river_cfr_config.get("override_iters", 2000)),
            "runtime_ms": round(runtime_ms, 3),
            "selected_label": selected_label,
            "selected_intent": selected_intent,
            "hero_combo_key": hero_combo_key,
            "hero_range_size": len(hero_combo_weights),
            "villain_range_size": len(villain_combo_weights),
            "hero_ev_bb": float(solve_result.hero_ev_bb),
            "infoset_count": int(solve_result.infoset_count),
        }

    override: Dict[str, Any] = {
        "status": "ok",
        "mix": one_hot_mix,
        "recommendedAction": recommended_action,
        "explanation": "River CFR+ v1 override applied on river NO_BET node.",
        "missingFields": [],
    }
    if debug_payload is not None:
        override["debug"] = {"river_cfr": debug_payload}
    return override


def build_capabilities(valid_actions: Dict[str, Any]) -> Dict[str, bool]:
    action_set = set(valid_actions.get("actions_list", []))
    can_bet = PlayerAction.BET in action_set
    return {
        "check": PlayerAction.CHECK in action_set,
        "fold": PlayerAction.FOLD in action_set,
        "call": PlayerAction.CALL in action_set,
        "bet33": can_bet,
        "bet50": can_bet,
        "bet75": can_bet,
        "raise": can_bet,
        "all_in": can_bet,
    }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_action_from_intent(
    intent: str,
    table: Table,
    player: Any,
    valid_actions: Dict[str, Any],
    pot_before_action: float,
    *,
    street: str,
    preflop_raise_count_before_action: int,
    pre_stack_bb: float,
    apply_hero_preflop_sizing: bool,
    open_size_bb: float,
    threebet_size_bb: float,
    fourbet_size_bb: float,
    requested_size_bb: Optional[float] = None,
    exact_hit_policy: bool = False,
    preserve_exact_postflop_size: bool = False,
    relax_exact_postflop_raise_guardrail: bool = False,
) -> Tuple[Action, Dict[str, Any]]:
    action_set = set(valid_actions["actions_list"])
    bet_low, bet_high = valid_actions["bet_range"]
    to_call = max(0.0, float(table.bet_to_match - player.bet_this_street))
    metadata: Dict[str, Any] = {
        "requested_intent": intent,
        "requested_action_type": expected_action_type_for_intent(intent).name,
        "translated_action_type": None,
        "target_bet_amount": None,
        "sent_bet_amount": None,
        # pokerenv BET expects "raise-to" (total bet this street), not increment.
        "bet_amount_semantics": "raise_to_total_street_contribution_bb",
        "clipped_to_bounds": False,
        "translation_adjusted": False,
        "is_allin": False,
        "preflop_node": None,
        "preflop_raise_clamped": False,
        "preflop_raise_converted_to_allin": False,
        "postflop_raise_clamped": False,
        "exact_hit_policy": bool(exact_hit_policy),
        "exact_requested_size_used": False,
        "postflop_raise_guardrail_relaxed": False,
        "requested_size_bb": float(requested_size_bb) if _finite_positive_size(requested_size_bb) else None,
    }

    def finalize(action: Action) -> Tuple[Action, Dict[str, Any]]:
        metadata["translated_action_type"] = action.action_type.name
        metadata["translation_adjusted"] = bool(metadata["translation_adjusted"]) or (
            metadata["translated_action_type"] != metadata["requested_action_type"]
        )
        if action.action_type == PlayerAction.BET:
            metadata["sent_bet_amount"] = float(action.bet_amount)
        return action, metadata

    def fallback_action() -> Tuple[Action, Dict[str, Any]]:
        if PlayerAction.CHECK in action_set:
            return finalize(Action(PlayerAction.CHECK))
        if PlayerAction.CALL in action_set:
            return finalize(Action(PlayerAction.CALL))
        if PlayerAction.FOLD in action_set:
            return finalize(Action(PlayerAction.FOLD))
        if PlayerAction.BET in action_set:
            metadata["target_bet_amount"] = float(bet_low)
            metadata["sent_bet_amount"] = float(bet_low)
            return finalize(Action(PlayerAction.BET, float(bet_low)))
        return finalize(Action(PlayerAction.CHECK))

    if intent == "check":
        if PlayerAction.CHECK in action_set:
            return finalize(Action(PlayerAction.CHECK))
        if PlayerAction.CALL in action_set:
            return finalize(Action(PlayerAction.CALL))
        return fallback_action()

    if intent == "fold":
        if PlayerAction.FOLD in action_set:
            return finalize(Action(PlayerAction.FOLD))
        if PlayerAction.CHECK in action_set:
            return finalize(Action(PlayerAction.CHECK))
        return fallback_action()

    if intent == "call":
        if PlayerAction.CALL in action_set:
            return finalize(Action(PlayerAction.CALL))
        if PlayerAction.CHECK in action_set:
            return finalize(Action(PlayerAction.CHECK))
        return fallback_action()

    if intent in ("raise", "all_in", "bet33", "bet50", "bet75", "bet150") and PlayerAction.BET in action_set:
        use_requested_postflop_size = (
            street != "preflop" and _finite_positive_size(requested_size_bb) is not None
        )
        if apply_hero_preflop_sizing and street == "preflop":
            preflop_node = "OPEN"
            if preflop_raise_count_before_action <= 0:
                preflop_node = "OPEN"
                target = clamp(float(open_size_bb), 2.0, 3.5)
                if abs(float(target) - float(open_size_bb)) > 1e-9:
                    metadata["preflop_raise_clamped"] = True
                    metadata["translation_adjusted"] = True
            elif preflop_raise_count_before_action == 1:
                preflop_node = "3BET"
                target = threebet_size_bb
            elif preflop_raise_count_before_action == 2:
                preflop_node = "4BET"
                target = fourbet_size_bb
            else:
                preflop_node = "5BET"
                target = float(bet_high)
                metadata["is_allin"] = True

            metadata["preflop_node"] = preflop_node
            if intent == "all_in":
                target = float(bet_high)
                metadata["is_allin"] = True

            if preflop_node == "OPEN" and intent != "all_in":
                # OPEN node hard guardrail: no accidental huge raises/all-ins from adapter translation.
                max_open_non_allin = min(15.0, float(bet_high) - 1e-6)
                min_open_non_allin = max(2.0, float(bet_low))
                if max_open_non_allin < min_open_non_allin:
                    max_open_non_allin = min_open_non_allin
                guarded_target = clamp(float(target), min_open_non_allin, max_open_non_allin)
                if abs(guarded_target - float(target)) > 1e-9:
                    metadata["preflop_raise_clamped"] = True
                    metadata["translation_adjusted"] = True
                    target = guarded_target

            if (
                not metadata["is_allin"]
                and pre_stack_bb > 0
                and float(target) > (pre_stack_bb * 0.8)
            ):
                if preflop_node == "OPEN":
                    max_open_non_allin = min(15.0, float(bet_high) - 1e-6)
                    min_open_non_allin = max(2.0, float(bet_low))
                    if max_open_non_allin < min_open_non_allin:
                        max_open_non_allin = min_open_non_allin
                    target = clamp(float(target), min_open_non_allin, max_open_non_allin)
                    metadata["preflop_raise_clamped"] = True
                    metadata["translation_adjusted"] = True
                else:
                    target = float(bet_high)
                    metadata["is_allin"] = True
                    metadata["preflop_raise_converted_to_allin"] = True
        elif intent == "all_in":
            target = float(bet_high)
            metadata["is_allin"] = True
        elif intent == "raise":
            if use_requested_postflop_size:
                target = float(requested_size_bb)
                metadata["exact_requested_size_used"] = True
            elif to_call > 0:
                target = to_call * 3.0
            else:
                target = max(1.0, pot_before_action * 0.75)
        elif intent == "bet50":
            if use_requested_postflop_size:
                target = float(requested_size_bb)
                metadata["exact_requested_size_used"] = True
            elif to_call > 0:
                target = max(to_call * 2.75, to_call + 1.0)
            else:
                target = max(1.0, pot_before_action * 0.5)
        elif intent == "bet75":
            if use_requested_postflop_size:
                target = float(requested_size_bb)
                metadata["exact_requested_size_used"] = True
            elif to_call > 0:
                target = to_call * 3.0
            elif _finite_positive_size(requested_size_bb):
                target = float(requested_size_bb)
            else:
                target = max(1.0, pot_before_action * 0.75)
        elif intent == "bet150":
            if use_requested_postflop_size:
                target = float(requested_size_bb)
                metadata["exact_requested_size_used"] = True
            elif to_call > 0:
                target = max(to_call * 4.0, to_call + pot_before_action * 1.5)
            elif _finite_positive_size(requested_size_bb):
                target = float(requested_size_bb)
            else:
                target = max(1.0, pot_before_action * 1.5)
        else:
            if use_requested_postflop_size:
                target = float(requested_size_bb)
                metadata["exact_requested_size_used"] = True
            elif to_call > 0:
                target = max(to_call * 2.5, to_call + 1.0)
            elif _finite_positive_size(requested_size_bb):
                target = float(requested_size_bb)
            else:
                target = max(1.0, pot_before_action * 0.33)

        skip_postflop_raise_guardrail = bool(
            street != "preflop"
            and bool(exact_hit_policy)
            and bool(relax_exact_postflop_raise_guardrail)
            and bool(metadata.get("exact_requested_size_used", False))
        )
        if skip_postflop_raise_guardrail:
            metadata["postflop_raise_guardrail_relaxed"] = True

        if (
            street != "preflop"
            and to_call > 1e-9
            and not metadata["is_allin"]
            and intent in ("raise", "bet75", "bet50", "bet33", "bet150")
            and not skip_postflop_raise_guardrail
        ):
            min_postflop_raise = float(bet_low)
            pot_relative_cap = float(to_call + max(1.0, pot_before_action * 1.25))
            stack_cap = float(pre_stack_bb * 0.75) if pre_stack_bb > 0 else float(bet_high)
            max_non_allin = min(float(bet_high) - 1e-6, pot_relative_cap, stack_cap)
            if max_non_allin < min_postflop_raise:
                max_non_allin = min_postflop_raise
            guarded_target = clamp(float(target), min_postflop_raise, max_non_allin)
            if abs(guarded_target - float(target)) > 1e-9:
                target = guarded_target
                metadata["postflop_raise_clamped"] = True
                metadata["translation_adjusted"] = True

        metadata["target_bet_amount"] = float(target)
        amount = float(clamp(float(target), float(bet_low), float(bet_high)))
        metadata["clipped_to_bounds"] = abs(amount - float(target)) > 1e-9
        if metadata["clipped_to_bounds"] and street == "preflop":
            metadata["preflop_raise_clamped"] = True
        if metadata["clipped_to_bounds"] and street != "preflop" and to_call > 1e-9:
            metadata["postflop_raise_clamped"] = True
        if amount >= float(bet_high) - 1e-9:
            metadata["is_allin"] = True
        return finalize(Action(PlayerAction.BET, amount))

    return fallback_action()


def expected_action_type_for_intent(intent: str) -> PlayerAction:
    if intent == "check":
        return PlayerAction.CHECK
    if intent == "fold":
        return PlayerAction.FOLD
    if intent == "call":
        return PlayerAction.CALL
    return PlayerAction.BET


def action_type_to_text(action_type: PlayerAction, pre_to_call: float) -> str:
    if action_type == PlayerAction.CHECK:
        return "check"
    if action_type == PlayerAction.FOLD:
        return "fold"
    if action_type == PlayerAction.CALL:
        return "call"
    if action_type == PlayerAction.BET:
        return "raise" if pre_to_call > 1e-9 else "bet"
    return "check"


def infer_size_bucket(actual_total_bet: float, pot_before_action: float) -> Optional[str]:
    if pot_before_action <= 1e-9:
        return "other"
    ratio = actual_total_bet / pot_before_action
    if abs(ratio - 0.33) <= 0.18:
        return "33"
    if abs(ratio - 0.75) <= 0.25:
        return "75"
    if abs(ratio - 1.50) <= 0.35:
        return "150"
    return "other"


def update_context_after_action(
    context: HandContext,
    street: str,
    actor: Any,
    hero_id: int,
    action_type: PlayerAction,
    pre_bet_to_match: float,
) -> None:
    state = context.streets[street]

    if action_type == PlayerAction.BET:
        state.aggression_count += 1
        if street == "preflop":
            context.preflop_raise_count += 1
            context.last_preflop_aggressor_id = int(actor.identifier)

        if int(actor.identifier) == hero_id:
            state.hero_aggressed = True
            state.hero_aggression_origin = "NO_BET" if pre_bet_to_match <= 1e-9 else "FACING_RAISE"


def update_hu_preflop_pot_context(
    context: HandContext,
    table: Table,
    street: str,
    action_type: PlayerAction,
    pre_to_call: float,
    preflop_raise_count_before_action: int,
) -> None:
    if table.n_players != 2 or street != "preflop":
        return

    if (
        preflop_raise_count_before_action == 0
        and pre_to_call > 1e-9
        and action_type == PlayerAction.CALL
    ):
        context.hu_preflop_was_limped = True
        return

    if not context.hu_preflop_was_limped:
        return

    if preflop_raise_count_before_action != 0 or pre_to_call > 1e-9:
        return

    if action_type == PlayerAction.BET:
        context.hu_preflop_bb_iso_raised = True
        context.hu_preflop_bb_checked = False
    elif action_type == PlayerAction.CHECK and not context.hu_preflop_bb_iso_raised:
        context.hu_preflop_bb_checked = True


def record_street_history_if_hero(
    context: HandContext,
    street: str,
    actor: Any,
    hero_id: int,
    hero_ip: bool,
    facing_action: str,
    action_type: PlayerAction,
    pre_to_call: float,
    pre_pot: float,
    size_bucket_hint: Optional[str],
) -> None:
    if street not in ("flop", "turn", "river"):
        return
    if int(actor.identifier) != hero_id:
        return

    hero_action = action_type_to_text(action_type, pre_to_call)

    size_bucket: Optional[str] = None
    if action_type == PlayerAction.BET:
        if size_bucket_hint in ("33", "75", "other"):
            size_bucket = size_bucket_hint
        else:
            size_bucket = infer_size_bucket(float(actor.bet_this_street), pre_pot)

    context.street_history[street] = {
        "node": facing_action,
        "heroIP": bool(hero_ip),
        "facingAction": facing_action,
        "heroAction": hero_action,
        "sizeBucket": size_bucket,
        "potSizeBb": float(pre_pot),
    }


def compute_rake(
    rake_mode: str,
    pot_size_bb: float,
    rake_fixed_bb: float,
    rake_percent: float,
    rake_cap_bb: float,
) -> float:
    if rake_mode == "none":
        return 0.0
    if rake_mode == "fixed":
        return max(0.0, rake_fixed_bb)
    rake = max(0.0, pot_size_bb * rake_percent)
    return min(rake, max(0.0, rake_cap_bb))


def summarize(samples: List[float]) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr)) if n else 0.0
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = float(std / math.sqrt(n)) if n > 1 else 0.0
    ci_delta = 1.96 * se
    return {
        "hands": float(n),
        "mean_bb_per_hand": mean,
        "bb_per_100": mean * 100.0,
        "stddev_bb": std,
        "stderr_bb": se,
        "ci95_low_bb_per_hand": mean - ci_delta,
        "ci95_high_bb_per_hand": mean + ci_delta,
        "ci95_low_bb_per_100": (mean - ci_delta) * 100.0,
        "ci95_high_bb_per_100": (mean + ci_delta) * 100.0,
        "total_bb": float(np.sum(arr)) if n else 0.0,
    }


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def build_no_bet_tracker_rows(
    action_stats: Dict[str, Dict[str, float]],
    pot_totals: Dict[str, int],
    action_pot_stats: Dict[str, Dict[str, Dict[str, float]]],
    context_totals: Dict[str, int],
    action_context_stats: Dict[str, Dict[str, Dict[str, float]]],
    total_events: int,
    tracked_actions: Tuple[str, ...] = NO_BET_TRACKED_ACTIONS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_actions = set(action_stats.keys())
    all_actions.update(tracked_actions)
    action_rows: List[Dict[str, Any]] = []
    for action_name in sorted(
        all_actions,
        key=lambda value: (value not in tracked_actions, value),
    ):
        stats = action_stats.get(action_name, {"count": 0.0, "total_bb": 0.0})
        count_value = int(stats.get("count", 0.0))
        total_bb_value = float(stats.get("total_bb", 0.0))
        action_rows.append(
            {
                "action": action_name,
                "count": count_value,
                "freq_pct": pct(count_value, total_events),
                "total_bb": total_bb_value,
                "avg_bb": (total_bb_value / count_value) if count_value > 0 else 0.0,
            }
        )

    pot_bucket_rows: List[Dict[str, Any]] = []
    for pot_bucket, total_count in sorted(
        pot_totals.items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        action_breakdown: List[Dict[str, Any]] = []
        per_action = action_pot_stats.get(pot_bucket, {})
        for action_name, stats in sorted(
            per_action.items(),
            key=lambda item: (-int(item[1].get("count", 0.0)), item[0]),
        ):
            count_value = int(stats.get("count", 0.0))
            total_bb_value = float(stats.get("total_bb", 0.0))
            action_breakdown.append(
                {
                    "action": action_name,
                    "count": count_value,
                    "freq_pct_within_bucket": pct(count_value, int(total_count)),
                    "freq_pct_overall": pct(count_value, total_events),
                    "total_bb": total_bb_value,
                    "avg_bb": (total_bb_value / count_value) if count_value > 0 else 0.0,
                }
            )
        pot_bucket_rows.append(
            {
                "pot_bucket": pot_bucket,
                "count": int(total_count),
                "freq_pct": pct(int(total_count), total_events),
                "action_breakdown": action_breakdown,
            }
        )

    preflop_context_rows: List[Dict[str, Any]] = []
    for context_name, total_count in sorted(
        context_totals.items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        action_breakdown: List[Dict[str, Any]] = []
        per_action = action_context_stats.get(context_name, {})
        for action_name, stats in sorted(
            per_action.items(),
            key=lambda item: (-int(item[1].get("count", 0.0)), item[0]),
        ):
            count_value = int(stats.get("count", 0.0))
            total_bb_value = float(stats.get("total_bb", 0.0))
            action_breakdown.append(
                {
                    "action": action_name,
                    "count": count_value,
                    "freq_pct_within_context": pct(count_value, int(total_count)),
                    "freq_pct_overall": pct(count_value, total_events),
                    "total_bb": total_bb_value,
                    "avg_bb": (total_bb_value / count_value) if count_value > 0 else 0.0,
                }
            )
        preflop_context_rows.append(
            {
                "preflop_context": context_name,
                "count": int(total_count),
                "freq_pct": pct(int(total_count), total_events),
                "action_breakdown": action_breakdown,
            }
        )

    return action_rows, pot_bucket_rows, preflop_context_rows


def merge_no_bet_tracker_summary(
    tracker: Dict[str, Any],
    action_stats: Dict[str, Dict[str, float]],
    context_totals: Dict[str, int],
    context_action_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    if not isinstance(tracker, dict):
        return

    for action_row in tracker.get("action_summary", []):
        action_name = str(action_row.get("action", "unknown"))
        if action_name not in action_stats:
            action_stats[action_name] = {"count": 0.0, "total_bb": 0.0}
        action_stats[action_name]["count"] += float(action_row.get("count", 0))
        action_stats[action_name]["total_bb"] += float(action_row.get("total_bb", 0.0))

    for context_row in tracker.get("preflop_context_summary", []):
        context_name = str(context_row.get("preflop_context", "unknown"))
        context_count = int(context_row.get("count", 0))
        context_totals[context_name] = int(context_totals.get(context_name, 0)) + context_count
        if context_name not in context_action_stats:
            context_action_stats[context_name] = {}
        per_action = context_action_stats[context_name]
        for action_row in context_row.get("action_breakdown", []):
            action_name = str(action_row.get("action", "unknown"))
            if action_name not in per_action:
                per_action[action_name] = {"count": 0.0, "total_bb": 0.0}
            per_action[action_name]["count"] += float(action_row.get("count", 0))
            per_action[action_name]["total_bb"] += float(action_row.get("total_bb", 0.0))


def _teacher_dashboard_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def build_teacher_disagreement_dashboard(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = int(len(rows))
    disagreements = 0
    transition_counts: Dict[Tuple[str, str], int] = {}
    by_preflop_context: Dict[str, Dict[str, int]] = {}
    by_board_texture: Dict[str, Dict[str, int]] = {}
    by_range_source: Dict[str, Dict[str, int]] = {}

    for row in rows:
        baseline = str(row.get("chosenActionBaseline") or "check")
        teacher_action = str(
            row.get("teacher_best_action")
            or row.get("suggestedAction")
            or "check"
        )
        preflop_context = str(row.get("preflopContext") or "unknown")
        board_texture = str(row.get("boardTextureBucket") or "unknown")
        range_source = str(row.get("range_source") or "unknown")
        disagreed = baseline != teacher_action
        if disagreed:
            disagreements += 1
            key = (baseline, teacher_action)
            transition_counts[key] = int(transition_counts.get(key, 0)) + 1

        for bucket, bucket_key in (
            (by_preflop_context, preflop_context),
            (by_board_texture, board_texture),
            (by_range_source, range_source),
        ):
            if bucket_key not in bucket:
                bucket[bucket_key] = {"total": 0, "disagreements": 0}
            bucket[bucket_key]["total"] = int(bucket[bucket_key]["total"]) + 1
            if disagreed:
                bucket[bucket_key]["disagreements"] = int(bucket[bucket_key]["disagreements"]) + 1

    def _bucket_rows(source: Mapping[str, Mapping[str, int]]) -> List[Dict[str, Any]]:
        rows_out: List[Dict[str, Any]] = []
        for key, stats in sorted(
            source.items(),
            key=lambda item: (-int(item[1].get("total", 0)), item[0]),
        ):
            key_total = int(stats.get("total", 0))
            key_disagreements = int(stats.get("disagreements", 0))
            rows_out.append(
                {
                    "key": key,
                    "total": key_total,
                    "disagreements": key_disagreements,
                    "disagreement_rate": _teacher_dashboard_rate(key_disagreements, key_total),
                }
            )
        return rows_out

    top_transitions: List[Dict[str, Any]] = []
    for (baseline, teacher_action), count in sorted(
        transition_counts.items(),
        key=lambda item: (-int(item[1]), item[0][0], item[0][1]),
    )[:10]:
        top_transitions.append(
            {
                "baseline_action": baseline,
                "teacher_action": teacher_action,
                "count": int(count),
                "freq_pct": _teacher_dashboard_rate(int(count), total) * 100.0,
            }
        )

    return {
        "total_processed_samples": total,
        "total_disagreements": int(disagreements),
        "overall_disagreement_rate": _teacher_dashboard_rate(disagreements, total),
        "by_preflop_context": _bucket_rows(by_preflop_context),
        "by_board_texture": _bucket_rows(by_board_texture),
        "by_range_source": _bucket_rows(by_range_source),
        "top_transitions": top_transitions,
    }


def summarize_float_list(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def classify_bet_fraction_bucket(value: float) -> str:
    if not math.isfinite(value):
        return "invalid"
    if value < 0.25:
        return "<0.25"
    if value < 0.40:
        return "0.25-0.40"
    if value < 0.60:
        return "0.40-0.60"
    if value < 0.90:
        return "0.60-0.90"
    return ">=0.90"


def update_fraction_buckets(buckets: Dict[str, int], value: float) -> None:
    key = classify_bet_fraction_bucket(value)
    buckets[key] = buckets.get(key, 0) + 1


def count_near_halfpot(values: List[float], tolerance: float = 0.06) -> int:
    return int(sum(1 for value in values if math.isfinite(value) and abs(value - 0.5) <= tolerance))


def effective_bet_fraction(values: List[float], fallback: float) -> float:
    if not values:
        return fallback
    return float(np.median(np.asarray(values, dtype=float)))


def mdf_continue_target_for_fraction(fraction: float) -> float:
    if not math.isfinite(fraction) or fraction <= 0:
        return 0.5
    mdf = 1.0 / (1.0 + fraction)
    return max(0.35, min(0.85, mdf))


def format_af(aggr: int, calls: int) -> str:
    if calls == 0:
        return "inf" if aggr > 0 else "0.00"
    return f"{aggr / calls:.2f}"


def action_bucket_from_action_name(action_name: str) -> str:
    if action_name == "FOLD":
        return "fold"
    if action_name in ("CALL", "CHECK"):
        return "call"
    if action_name == "BET":
        return "raise"
    return "call"


def infer_preflop_node_events_with_context(
    hero_preflop_events: List[Dict[str, Any]],
    action_sequence: List[Dict[str, Any]],
    hero_id: int,
    terminal_street: str,
    showdown: bool,
    hero_reward: float,
    hero_hand_bucket: str,
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    hero_opened = False
    hero_threebet = False
    faced_fourbet = False
    hero_folded_postflop = any(
        int(event.get("actor_id", -1)) == int(hero_id)
        and str(event.get("street")) in POSTFLOP_STREETS
        and str(event.get("executed_action_type")) == "FOLD"
        for event in action_sequence
    )
    outcome_class = classify_final_outcome_class(
        terminal_street=terminal_street,
        showdown=showdown,
        hero_folded_postflop=hero_folded_postflop,
        hero_reward=hero_reward,
    )

    def build_node_event(
        node: str,
        action_bucket: str,
        note: str,
        rc_before: int,
        pre_to_call: float,
        action_index: int,
        raise_size_bb: Optional[float],
        invested_bb: float,
        villain_response: str,
    ) -> Dict[str, Any]:
        return {
            "node": node,
            "action": action_bucket,
            "note": note,
            "preflop_raise_count_before": rc_before,
            "to_call_bb": pre_to_call,
            "to_call_bucket": bucket_size_bb(pre_to_call),
            "raise_size_bb": raise_size_bb,
            "raise_size_bucket": bucket_size_bb(raise_size_bb),
            "immediate_delta_proxy_bb": -float(invested_bb),
            "villain_response": villain_response,
            "final_outcome_class": outcome_class,
            "hand_bucket": hero_hand_bucket,
            "attribution_method": "full_hand_with_outcome_class",
            "action_sequence_index": action_index,
        }

    for event in hero_preflop_events:
        rc_before = int(event["preflop_raise_count_before"])
        pre_to_call = float(event["pre_to_call"])
        action_name = str(event["action_type"])
        action_bucket = str(event["action_bucket"])
        action_index = int(event.get("action_sequence_index", -1))
        raise_size_raw = event.get("raise_size_bb")
        raise_size_bb = (
            float(raise_size_raw)
            if isinstance(raise_size_raw, (int, float)) and math.isfinite(float(raise_size_raw))
            else None
        )
        invested_bb = float(event.get("invested_bb", 0.0))

        if action_name == "BET" and rc_before == 0:
            hero_opened = True
            nodes.append(
                build_node_event(
                    node="HERO_OPENED",
                    action_bucket="raise",
                    note="first-in raise",
                    rc_before=rc_before,
                    pre_to_call=pre_to_call,
                    action_index=action_index,
                    raise_size_bb=raise_size_bb,
                    invested_bb=invested_bb,
                    villain_response=classify_villain_response_to_open(
                        action_sequence=action_sequence,
                        hero_action_index=action_index,
                        hero_id=hero_id,
                        terminal_street=terminal_street,
                    ),
                )
            )
        elif action_name == "CALL" and rc_before == 0 and pre_to_call > 0:
            nodes.append(
                build_node_event(
                    node="HERO_LIMPED",
                    action_bucket="call",
                    note="open limp",
                    rc_before=rc_before,
                    pre_to_call=pre_to_call,
                    action_index=action_index,
                    raise_size_bb=raise_size_bb,
                    invested_bb=invested_bb,
                    villain_response="n/a",
                )
            )
        elif action_name == "CALL" and rc_before == 1 and pre_to_call > 0:
            nodes.append(
                build_node_event(
                    node="HERO_CALLED_OPEN",
                    action_bucket="call",
                    note="",
                    rc_before=rc_before,
                    pre_to_call=pre_to_call,
                    action_index=action_index,
                    raise_size_bb=raise_size_bb,
                    invested_bb=invested_bb,
                    villain_response="n/a",
                )
            )
        elif action_name == "FOLD" and rc_before == 1 and pre_to_call > 0:
            nodes.append(
                build_node_event(
                    node="HERO_FOLDED_TO_OPEN",
                    action_bucket="fold",
                    note="",
                    rc_before=rc_before,
                    pre_to_call=pre_to_call,
                    action_index=action_index,
                    raise_size_bb=raise_size_bb,
                    invested_bb=invested_bb,
                    villain_response="n/a",
                )
            )

        if action_name == "BET" and rc_before == 1 and pre_to_call > 0:
            hero_threebet = True
            nodes.append(
                build_node_event(
                    node="HERO_3BET",
                    action_bucket="raise",
                    note="",
                    rc_before=rc_before,
                    pre_to_call=pre_to_call,
                    action_index=action_index,
                    raise_size_bb=raise_size_bb,
                    invested_bb=invested_bb,
                    villain_response="n/a",
                )
            )

        if hero_opened and rc_before == 2 and pre_to_call > 0:
            nodes.append(
                build_node_event(
                    node="HERO_FACED_3BET_AFTER_OPEN",
                    action_bucket=action_bucket,
                    note=f"response={action_bucket}",
                    rc_before=rc_before,
                    pre_to_call=pre_to_call,
                    action_index=action_index,
                    raise_size_bb=raise_size_bb,
                    invested_bb=invested_bb,
                    villain_response=classify_villain_response_to_faced_3bet_after_open(
                        hero_action_bucket=action_bucket,
                        action_sequence=action_sequence,
                        hero_action_index=action_index,
                        hero_id=hero_id,
                        terminal_street=terminal_street,
                    ),
                )
            )

        if hero_threebet and rc_before >= 3 and pre_to_call > 0:
            faced_fourbet = True
            nodes.append(
                build_node_event(
                    node="HERO_FACED_4BET_AFTER_3BET",
                    action_bucket=action_bucket,
                    note=f"response={action_bucket}",
                    rc_before=rc_before,
                    pre_to_call=pre_to_call,
                    action_index=action_index,
                    raise_size_bb=raise_size_bb,
                    invested_bb=invested_bb,
                    villain_response="n/a",
                )
            )

    if hero_threebet:
        if faced_fourbet:
            outcome = "faced_4bet"
        elif terminal_street != "preflop":
            outcome = "got_called"
        else:
            outcome = "took_down"
        for node in nodes:
            if node["node"] == "HERO_3BET":
                node["note"] = outcome

    return nodes


def infer_preflop_leak_nodes(
    hero_preflop_events: List[Dict[str, Any]],
    terminal_street: str,
) -> List[Dict[str, str]]:
    nodes: List[Dict[str, str]] = []
    hero_opened = False
    hero_threebet = False
    faced_fourbet = False

    for event in hero_preflop_events:
        rc_before = int(event["preflop_raise_count_before"])
        pre_to_call = float(event["pre_to_call"])
        action_name = str(event["action_type"])
        action_bucket = str(event["action_bucket"])

        if action_name == "BET" and rc_before == 0:
            hero_opened = True
            nodes.append({"node": "HERO_OPENED", "action": "raise", "note": "first-in raise"})
        elif action_name == "CALL" and rc_before == 0 and pre_to_call > 0:
            nodes.append({"node": "HERO_LIMPED", "action": "call", "note": "open limp"})
        elif action_name == "CALL" and rc_before == 1 and pre_to_call > 0:
            nodes.append({"node": "HERO_CALLED_OPEN", "action": "call", "note": ""})
        elif action_name == "FOLD" and rc_before == 1 and pre_to_call > 0:
            nodes.append({"node": "HERO_FOLDED_TO_OPEN", "action": "fold", "note": ""})

        if action_name == "BET" and rc_before == 1 and pre_to_call > 0:
            hero_threebet = True
            nodes.append({"node": "HERO_3BET", "action": "raise", "note": ""})

        if hero_opened and rc_before == 2 and pre_to_call > 0:
            nodes.append(
                {
                    "node": "HERO_FACED_3BET_AFTER_OPEN",
                    "action": action_bucket,
                    "note": f"response={action_bucket}",
                }
            )

        if hero_threebet and rc_before >= 3 and pre_to_call > 0:
            faced_fourbet = True
            nodes.append(
                {
                    "node": "HERO_FACED_4BET_AFTER_3BET",
                    "action": action_bucket,
                    "note": f"response={action_bucket}",
                }
            )

    if hero_threebet:
        if faced_fourbet:
            outcome = "faced_4bet"
        elif terminal_street != "preflop":
            outcome = "got_called"
        else:
            outcome = "took_down"

        for node in nodes:
            if node["node"] == "HERO_3BET":
                node["note"] = outcome

    return nodes


def print_summary(
    raw: Dict[str, float],
    net: Dict[str, float],
    args: argparse.Namespace,
    fallback_count: int,
    diagnostics: Dict[str, Any],
) -> None:
    print("\n=== WiPoker Offline EV Summary ===")
    print(f"hands: {int(raw['hands'])}")
    print(f"players: {args.players}")
    print(f"seed: {args.seed}")
    print(f"hero mode: {args.hero_mode}")
    mode_config = diagnostics.get("mode_config", {})
    if isinstance(mode_config, dict):
        print(
            "effective modes: preflop={pre} postflop={post} hero={hero}".format(
                pre=str(mode_config.get("preflop_selection_mode_effective", "unknown")),
                post=str(mode_config.get("postflop_selection_mode_effective", "unknown")),
                hero=str(mode_config.get("hero_mode_effective", args.hero_mode)),
            )
        )
        print(
            "fidelity_flags: exact_postflop_size={exact_flag} relax_postflop_raise_guardrail={relax_flag}".format(
                exact_flag=bool(
                    mode_config.get("exact_action_fidelity_postflop_enabled", False)
                ),
                relax_flag=bool(
                    mode_config.get(
                        "relax_exact_postflop_raise_guardrail_enabled",
                        False,
                    )
                ),
            )
        )
    print(f"policy cmd: {args.policy_cmd}")
    print(f"rake mode: {args.rake_mode}")
    print(f"fallback_count: {fallback_count}")

    print("\n-- Raw (no rake adjustment) --")
    print(f"total_bb: {raw['total_bb']:.2f}")
    print(f"bb/100: {raw['bb_per_100']:.2f}")
    print(f"stddev_bb_per_hand: {raw['stddev_bb']:.4f}")
    print(f"stderr_bb_per_hand: {raw['stderr_bb']:.6f}")
    print(
        "95% CI bb/100: "
        f"[{raw['ci95_low_bb_per_100']:.2f}, {raw['ci95_high_bb_per_100']:.2f}]"
    )

    print("\n-- Rake-adjusted --")
    print(f"total_bb: {net['total_bb']:.2f}")
    print(f"bb/100: {net['bb_per_100']:.2f}")
    print(f"stddev_bb_per_hand: {net['stddev_bb']:.4f}")
    print(f"stderr_bb_per_hand: {net['stderr_bb']:.6f}")
    print(
        "95% CI bb/100: "
        f"[{net['ci95_low_bb_per_100']:.2f}, {net['ci95_high_bb_per_100']:.2f}]"
    )

    street_decisions = diagnostics["street_decisions"]
    street_ok = diagnostics["street_ok"]
    street_unavailable = diagnostics["street_unavailable"]
    street_fallbacks = diagnostics["street_fallbacks"]
    print("\n-- Hero Decisions By Street --")
    for street in ALL_STREETS:
        print(
            f"{street}: decisions={street_decisions[street]} "
            f"ok={street_ok[street]} unavailable={street_unavailable[street]} "
            f"fallbacks={street_fallbacks[street]}"
        )
    fidelity = diagnostics.get("policy_execution_fidelity", {})
    if isinstance(fidelity, dict):
        print("\n-- Policy Execution Fidelity --")
        print(
            "selection_source_counts="
            f"{json.dumps(fidelity.get('selection_source_counts', {}), sort_keys=True)}"
        )
        print(
            "chosen_vs_executed_match="
            f"{int(fidelity.get('chosen_vs_executed_match_count', 0))}/"
            f"{int(fidelity.get('chosen_vs_executed_total', 0))} "
            f"chosen_vs_argmax_match="
            f"{int(fidelity.get('chosen_vs_argmax_match_count', 0))}/"
            f"{int(fidelity.get('chosen_vs_argmax_total', 0))}"
        )
        print(
            "exact_hit_translation="
            f"{int(fidelity.get('exact_hit_and_unmodified_count', 0))} unmodified / "
            f"{int(fidelity.get('exact_hit_translation_total', 0))} exact-hit translated"
        )

    strategy_store = diagnostics.get("strategy_store", {})
    if isinstance(strategy_store, dict) and strategy_store:
        print("\n-- Strategy Store Lookup --")
        has_fuzzy_breakdown = any(
            key in strategy_store
            for key in (
                "exact_hits",
                "exact_misses",
                "fuzzy_hits",
                "fuzzy_misses",
                "quality_exact_hits",
                "quality_approx_hits",
            )
        )
        if has_fuzzy_breakdown:
            print(
                f"enabled={bool(strategy_store.get('enabled', False))} "
                f"exact_hits={int(strategy_store.get('exact_hits', 0))} "
                f"exact_misses={int(strategy_store.get('exact_misses', 0))} "
                f"fuzzy_hits={int(strategy_store.get('fuzzy_hits', 0))} "
                f"fuzzy_misses={int(strategy_store.get('fuzzy_misses', 0))} "
                f"quality_exact_hits={int(strategy_store.get('quality_exact_hits', 0))} "
                f"quality_approx_hits={int(strategy_store.get('quality_approx_hits', 0))} "
                f"overall_hit_rate={float(strategy_store.get('overall_hit_rate', strategy_store.get('store_hit_rate', 0.0))):.4f}"
            )
        else:
            print(
                f"enabled={bool(strategy_store.get('enabled', False))} "
                f"hits={int(strategy_store.get('store_hit_count', 0))} "
                f"misses={int(strategy_store.get('store_miss_count', 0))} "
                f"hit_rate={float(strategy_store.get('store_hit_rate', 0.0)):.4f}"
            )
        top_missing = strategy_store.get("top_missing_spot_keys", [])
        if isinstance(top_missing, list) and top_missing:
            print("top_missing_spot_keys:")
            for row in top_missing[:10]:
                if not isinstance(row, Mapping):
                    continue
                print(
                    f"  {str(row.get('spot_key', 'unknown'))} "
                    f"(count={int(row.get('count', 0))})"
                )
        top_fuzzy_missing = strategy_store.get("top_fuzzy_miss_groups", [])
        if isinstance(top_fuzzy_missing, list) and top_fuzzy_missing:
            print("top_fuzzy_miss_groups:")
            for row in top_fuzzy_missing[:10]:
                if not isinstance(row, Mapping):
                    continue
                print(
                    f"  {str(row.get('group', 'unknown'))} "
                    f"(count={int(row.get('count', 0))})"
                )

    river_cfr = diagnostics.get("river_cfr", {})
    print("\n-- River CFR v1 --")
    print(
        f"enabled={bool(river_cfr.get('enabled', False))} "
        f"override_enabled={bool(river_cfr.get('override_enabled', False))} "
        f"teacher_enabled={bool(river_cfr.get('teacher_enabled', False))} "
        f"teacher_eval_in_run={bool(river_cfr.get('teacher_eval_in_run_enabled', False))} "
        f"override_calls={int(river_cfr.get('override_calls', 0))} "
        f"override_avg_runtime_ms={float(river_cfr.get('override_runtime_ms_avg', 0.0)):.3f}"
    )
    print(
        f"skipped_no_range={int(river_cfr.get('skipped_no_range', 0))} "
        f"parse_fail={int(river_cfr.get('parse_fail', 0))} "
        f"call_limit_reached={bool(river_cfr.get('match_call_limit_reached', False))}"
    )
    teacher = diagnostics.get("river_cfr_teacher", {})
    print(
        f"teacher_samples={int(teacher.get('captured_samples', 0))} "
        f"teacher_processed={int(teacher.get('processed_samples', 0))} "
        f"teacher_cache_hits={int(teacher.get('cache_hits', 0))} "
        f"teacher_runtime_ms_total={float(teacher.get('runtime_ms_total', 0.0)):.3f}"
    )

    ev = diagnostics["ev_decomposition_bb"]
    print("\n-- EV Decomposition (bb) --")
    print(f"forced_blinds_antes: {ev['forced_blinds_antes']:.2f}")
    print(f"preflop_voluntary: {ev['preflop_voluntary']:.2f}")
    print(f"flop: {ev['flop']:.2f}")
    print(f"turn: {ev['turn']:.2f}")
    print(f"river: {ev['river']:.2f}")
    print(f"attribution_total: {ev['attribution_total']:.2f}")
    print(f"attribution_residual: {ev['attribution_residual']:.6f}")
    print("attribution_note: payouts are attributed to terminal street of each hand.")

    style = diagnostics["hero_style"]
    print("\n-- Hero Style Metrics --")
    print(f"VPIP: {style['vpip_pct']:.2f}% ({style['vpip_hands']}/{style['hands_total']})")
    print(f"PFR: {style['pfr_pct']:.2f}% ({style['pfr_hands']}/{style['hands_total']})")
    print(
        f"3bet: {style['threebet_pct']:.2f}% "
        f"({style['threebet_hands']}/{style['threebet_opportunities']})"
    )
    print(
        f"showdown_when_vpip: {style['showdown_when_vpip_pct']:.2f}% "
        f"({style['showdown_when_vpip_hands']}/{style['vpip_hands']})"
    )
    for street in POSTFLOP_STREETS:
        af = style["af_by_street"][street]
        f2b = style["fold_to_bet_pct_by_street"][street]
        print(
            f"{street}: AF={af} "
            f"fold_to_bet={f2b:.2f}% "
            f"({style['fold_to_bet_folds'][street]}/{style['fold_to_bet_opportunities'][street]})"
        )

    audit = diagnostics["action_translation_audit"]
    print("\n-- Action Translation Audit --")
    print(
        f"translation_clipped={audit['translation_clipped_count']} "
        f"translation_adjusted={audit['translation_adjusted_count']} "
        f"env_legalized={audit['env_legalized_count']} "
        f"clipped_or_adjusted_total={audit['clipped_or_adjusted_total']}"
    )
    print(
        f"preflop_raise_clamped={audit['preflop_raise_clamped_count']} "
        f"preflop_raise_converted_to_allin={audit['preflop_raise_converted_to_allin_count']} "
        f"postflop_raise_clamped={audit.get('postflop_raise_clamped_count', 0)} "
        f"exact_requested_size_used={audit.get('exact_requested_size_used_count', 0)} "
        f"postflop_guardrail_relaxed={audit.get('postflop_raise_guardrail_relaxed_count', 0)} "
        f"engine_size_capped={audit.get('engine_size_capped_count', 0)} "
        f"illegal_action_avoided={audit['illegal_action_avoided_count']} "
        f"preflop_open_non_allin_raises_over_15bb={audit.get('preflop_open_non_allin_raises_over_15bb_count', 0)}"
    )
    capped_rows = audit.get("engine_size_capped_by_street_intent", [])
    if capped_rows:
        print("engine_size_capped by street/intent:")
        for row in capped_rows:
            print(f"{row['street']}:{row['intent']} count={row['count']}")
    for key in ("bet33", "bet50", "bet75"):
        stats = audit[f"{key}_fraction_of_pot"]
        print(
            f"{key}_fraction_of_pot: count={stats['count']} mean={stats['mean']} "
            f"median={stats['median']} min={stats['min']} max={stats['max']}"
        )
        bucket_counts = audit.get(f"{key}_fraction_buckets", {})
        if bucket_counts:
            ordered = ", ".join(f"{bucket}:{count}" for bucket, count in bucket_counts.items())
            print(f"{key}_fraction_buckets: {ordered}")
        forced_halfpot = int(audit.get(f"{key}_forced_halfpot_count", 0))
        print(f"{key}_forced_to_halfpot: {forced_halfpot}")
    raise_stats = audit["raise_size_bb"]
    print(
        f"raise_size_bb: count={raise_stats['count']} mean={raise_stats['mean']} "
        f"median={raise_stats['median']} min={raise_stats['min']} max={raise_stats['max']}"
    )
    effective_small = float(audit.get("effective_small_bet_fraction", 0.33))
    effective_large = float(audit.get("effective_large_bet_fraction", 0.75))
    print(
        f"effective_small_bet_size_of_pot={effective_small:.3f} "
        f"(MDF continue target {mdf_continue_target_for_fraction(effective_small) * 100:.1f}%)"
    )
    print(
        f"effective_large_bet_size_of_pot={effective_large:.3f} "
        f"(MDF continue target {mdf_continue_target_for_fraction(effective_large) * 100:.1f}%)"
    )

    preflop_report = diagnostics["preflop_leak_report"]
    print("\n-- Preflop Leak Report --")
    print("node | count | total_bb | avg_bb | fold% | call% | raise% | notes")
    if preflop_report["rows"]:
        for row in preflop_report["rows"]:
            print(
                f"{row['node']} | {row['count']} | {row['total_bb']:.2f} | {row['avg_bb']:.2f} | "
                f"{row['fold_pct']:.1f} | {row['call_pct']:.1f} | {row['raise_pct']:.1f} | {row['notes']}"
            )
    else:
        print("no preflop node events recorded")
    print("top 5 worst nodes by total_bb:")
    if preflop_report["worst_by_total_bb"]:
        for row in preflop_report["worst_by_total_bb"]:
            print(f"{row['node']} total_bb={row['total_bb']:.2f} avg_bb={row['avg_bb']:.2f} count={row['count']}")
    else:
        print("none")
    print("top 5 worst nodes by avg_bb:")
    if preflop_report["worst_by_avg_bb"]:
        for row in preflop_report["worst_by_avg_bb"]:
            print(f"{row['node']} avg_bb={row['avg_bb']:.2f} total_bb={row['total_bb']:.2f} count={row['count']}")
    else:
        print("none")

    raise_report = diagnostics["raise_outlier_report"]
    print("\n-- Raise Outlier Report --")
    print(
        f"raises >20bb: {raise_report['raise_over_20']['count']} "
        f"({raise_report['raise_over_20']['pct']:.2f}%)"
    )
    print(
        f"raises >40bb: {raise_report['raise_over_40']['count']} "
        f"({raise_report['raise_over_40']['pct']:.2f}%)"
    )
    print(
        f"raises >80bb: {raise_report['raise_over_80']['count']} "
        f"({raise_report['raise_over_80']['pct']:.2f}%)"
    )
    print(
        "preflop OPEN non-allin raises >15bb: "
        f"{int(raise_report.get('preflop_open_non_allin_raises_over_15bb_count', 0))}"
    )
    print("10 biggest raises:")
    if raise_report["largest_raises"]:
        for row in raise_report["largest_raises"]:
            print(
                f"hand={row['hand_index']} street={row['street']} node={row['facing_action']} "
                f"raise_bb={row['raise_size_bb']:.2f} pot_bb={row['pot_bb']:.2f} stack_bb={row['stack_bb']:.2f} "
                f"allin={row['is_allin']}"
            )
    else:
        print("none")
    print("clipped/adjusted by street/node:")
    if raise_report["clipped_adjusted_by_context"]:
        for row in raise_report["clipped_adjusted_by_context"]:
            print(
                f"{row['context']}: count={row['count']} clipped={row['clipped']} "
                f"translation_adjusted={row['translation_adjusted']} env_legalized={row['env_legalized']}"
            )
    else:
        print("none")


def build_opponent_list(lineup: str, player_count: int) -> Dict[int, BotBase]:
    names = [name.strip() for name in lineup.split(",") if name.strip()]
    if not names:
        names = ["call_station", "nit", "lag"]

    bots: Dict[int, BotBase] = {}
    for identifier in range(1, player_count):
        bot_name = names[(identifier - 1) % len(names)]
        if bot_name not in BOT_REGISTRY:
            raise ValueError(
                f"unknown bot '{bot_name}'. Available: {', '.join(sorted(BOT_REGISTRY.keys()))}"
            )
        bots[identifier] = BOT_REGISTRY[bot_name]
    return bots


def run_matrix_experiments(args: argparse.Namespace) -> int:
    seeds = parse_seed_list(args.seeds, args.seed)
    players = resolve_players(args.match_format, args.players)
    hands_per_seed = args.hands_per_seed if args.hands_per_seed is not None else args.hands
    opponent_profiles = resolve_opponent_profiles(args, players)

    if hands_per_seed <= 0:
        raise ValueError("--hands_per_seed/--hands must be > 0")

    if not opponent_profiles:
        raise ValueError("matrix mode requires at least one opponent profile")

    ensure_repo_commands(args)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    run_rows: List[Dict[str, Any]] = []
    matrix_translation_clipped_by_opponent: Dict[str, int] = {}
    matrix_flop_context_totals_by_opponent: Dict[str, Dict[str, int]] = {}
    matrix_flop_context_action_stats_by_opponent: Dict[
        str, Dict[str, Dict[str, Dict[str, float]]]
    ] = {}
    matrix_turn_action_stats_by_opponent: Dict[str, Dict[str, Dict[str, float]]] = {}
    matrix_turn_context_totals_by_opponent: Dict[str, Dict[str, int]] = {}
    matrix_turn_context_action_stats_by_opponent: Dict[
        str, Dict[str, Dict[str, Dict[str, float]]]
    ] = {}
    matrix_river_action_stats_by_opponent: Dict[str, Dict[str, Dict[str, float]]] = {}
    matrix_river_context_totals_by_opponent: Dict[str, Dict[str, int]] = {}
    matrix_river_context_action_stats_by_opponent: Dict[
        str, Dict[str, Dict[str, Dict[str, float]]]
    ] = {}
    matrix_raised_medium_tiny_by_opponent: Dict[str, Dict[str, Any]] = {}
    matrix_teacher_rows_by_opponent: Dict[str, List[Dict[str, Any]]] = {}

    print("\n=== WiPoker Matrix Evaluation ===")
    print(
        f"format={'hu' if players == 2 else '6max'} players={players} "
        f"seeds={seeds} opponents={opponent_profiles} hands_per_seed={hands_per_seed}"
    )

    for profile in opponent_profiles:
        opponent_label, lineup = resolve_lineup_for_profile(profile, players, args.opponent_lineup)
        for seed in seeds:
            report_path = args.runs_dir / f"{sanitize_label(opponent_label)}_{seed}.json"
            cmd = [
                sys.executable,
                str(script_path),
                "--_single_run",
                "--quiet",
                "--hands",
                str(hands_per_seed),
                "--players",
                str(players),
                "--stack-bb",
                str(args.stack_bb),
                "--seed",
                str(seed),
                "--hero-id",
                str(args.hero_id if args.hero_id < players else 0),
                "--hero_mode",
                args.hero_mode,
                "--eval-backend",
                args.eval_backend,
                "--policy-cmd",
                args.policy_cmd,
                "--open_size",
                str(args.open_size),
                "--threebet_size",
                str(args.threebet_size),
                "--fourbet_size",
                str(args.fourbet_size),
                "--opponent-lineup",
                lineup,
                "--max-actions-per-hand",
                str(args.max_actions_per_hand),
                "--progress-every",
                "0",
                "--report_json",
                str(report_path),
            ]
            if args.eval_backend == "deep_cfr_checkpoint":
                if args.deep_cfr_model_state is None:
                    raise ValueError(
                        "--deep-cfr-model-state is required for matrix mode when --eval-backend deep_cfr_checkpoint"
                    )
                cmd.extend(
                    [
                        "--deep-cfr-model-state",
                        str(args.deep_cfr_model_state),
                        "--deep-cfr-device",
                        str(args.deep_cfr_device),
                        "--deep-cfr-policy-mode",
                        str(args.deep_cfr_policy_mode),
                        "--deep-cfr-policy-seed",
                        str(int(args.deep_cfr_policy_seed)),
                    ]
                )

            if args.rake_bb_per_hand is not None:
                cmd.extend(["--rake_bb_per_hand", str(args.rake_bb_per_hand)])
            else:
                cmd.extend(
                    [
                        "--rake-mode",
                        args.rake_mode,
                        "--rake-fixed-bb",
                        str(args.rake_fixed_bb),
                        "--rake-percent",
                        str(args.rake_percent),
                        "--rake-cap-bb",
                        str(args.rake_cap_bb),
                    ]
                )

            if args.assert_large_preflop_raises_allin:
                cmd.append("--assert_large_preflop_raises_allin")
            if args.assert_preflop_open_max15:
                cmd.append("--assert_preflop_open_max15")

            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                tail = proc.stderr.strip() or proc.stdout.strip()
                raise RuntimeError(
                    f"matrix run failed for opponent={opponent_label} seed={seed}: {tail}"
                )

            with report_path.open("r", encoding="utf-8") as fp:
                report = json.load(fp)

            net = report["summary"]["net"]
            diagnostics = report.get("summary", {}).get("diagnostics", {})
            translation_clipped = int(
                diagnostics.get("action_translation_audit", {}).get(
                    "translation_clipped_count", 0
                )
            )
            matrix_translation_clipped_by_opponent[opponent_label] = int(
                matrix_translation_clipped_by_opponent.get(opponent_label, 0)
            ) + translation_clipped
            if opponent_label not in matrix_flop_context_totals_by_opponent:
                matrix_flop_context_totals_by_opponent[opponent_label] = {}
            if opponent_label not in matrix_flop_context_action_stats_by_opponent:
                matrix_flop_context_action_stats_by_opponent[opponent_label] = {}
            if opponent_label not in matrix_turn_action_stats_by_opponent:
                matrix_turn_action_stats_by_opponent[opponent_label] = {}
            if opponent_label not in matrix_turn_context_totals_by_opponent:
                matrix_turn_context_totals_by_opponent[opponent_label] = {}
            if opponent_label not in matrix_turn_context_action_stats_by_opponent:
                matrix_turn_context_action_stats_by_opponent[opponent_label] = {}
            if opponent_label not in matrix_river_action_stats_by_opponent:
                matrix_river_action_stats_by_opponent[opponent_label] = {}
            if opponent_label not in matrix_river_context_totals_by_opponent:
                matrix_river_context_totals_by_opponent[opponent_label] = {}
            if opponent_label not in matrix_river_context_action_stats_by_opponent:
                matrix_river_context_action_stats_by_opponent[opponent_label] = {}
            if opponent_label not in matrix_raised_medium_tiny_by_opponent:
                matrix_raised_medium_tiny_by_opponent[opponent_label] = {
                    "total_events": 0,
                    "raised_or_3bp_medium_tiny_minbet_prevented_count": 0,
                    "raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb": 0.0,
                    "action_stats": {},
                }
            if opponent_label not in matrix_teacher_rows_by_opponent:
                matrix_teacher_rows_by_opponent[opponent_label] = []
            context_totals = matrix_flop_context_totals_by_opponent[opponent_label]
            context_action_stats = matrix_flop_context_action_stats_by_opponent[opponent_label]
            for context_row in diagnostics.get("flop_no_bet_tracker", {}).get(
                "preflop_context_summary", []
            ):
                context_name = str(context_row.get("preflop_context", "unknown"))
                context_count = int(context_row.get("count", 0))
                context_totals[context_name] = int(context_totals.get(context_name, 0)) + context_count
                if context_name not in context_action_stats:
                    context_action_stats[context_name] = {}
                per_action = context_action_stats[context_name]
                for action_row in context_row.get("action_breakdown", []):
                    action_name = str(action_row.get("action", "unknown"))
                    if action_name not in per_action:
                        per_action[action_name] = {"count": 0.0, "total_bb": 0.0}
                    per_action[action_name]["count"] += float(action_row.get("count", 0))
                    per_action[action_name]["total_bb"] += float(action_row.get("total_bb", 0.0))
            merge_no_bet_tracker_summary(
                diagnostics.get("turn_no_bet_tracker", {}),
                matrix_turn_action_stats_by_opponent[opponent_label],
                matrix_turn_context_totals_by_opponent[opponent_label],
                matrix_turn_context_action_stats_by_opponent[opponent_label],
            )
            merge_no_bet_tracker_summary(
                diagnostics.get("river_no_bet_tracker", {}),
                matrix_river_action_stats_by_opponent[opponent_label],
                matrix_river_context_totals_by_opponent[opponent_label],
                matrix_river_context_action_stats_by_opponent[opponent_label],
            )
            suppression_diag = diagnostics.get("raised_or_3bp_medium_tiny_minbet_suppression", {})
            pooled_suppression = matrix_raised_medium_tiny_by_opponent[opponent_label]
            pooled_suppression["total_events"] += int(suppression_diag.get("total_events", 0))
            pooled_suppression["raised_or_3bp_medium_tiny_minbet_prevented_count"] += int(
                suppression_diag.get("raised_or_3bp_medium_tiny_minbet_prevented_count", 0)
            )
            pooled_suppression["raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb"] += float(
                suppression_diag.get("raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb", 0.0)
            )
            pooled_action_stats = pooled_suppression["action_stats"]
            for action_row in suppression_diag.get("action_summary", []):
                action_name = str(action_row.get("action", "unknown"))
                if action_name not in pooled_action_stats:
                    pooled_action_stats[action_name] = {"count": 0.0, "total_bb": 0.0}
                pooled_action_stats[action_name]["count"] += float(action_row.get("count", 0))
                pooled_action_stats[action_name]["total_bb"] += float(action_row.get("total_bb", 0.0))
            teacher_outputs = diagnostics.get("river_cfr_teacher_outputs", [])
            if isinstance(teacher_outputs, list):
                for row in teacher_outputs:
                    if isinstance(row, dict):
                        matrix_teacher_rows_by_opponent[opponent_label].append(row)

            row = {
                "opponent": opponent_label,
                "seed": seed,
                "hands": int(net["hands"]),
                "mean_bb_per_hand": float(net["mean_bb_per_hand"]),
                "bb_per_100": float(net["bb_per_100"]),
                "stddev_bb": float(net["stddev_bb"]),
                "ci_low_bb_per_100": float(net["ci95_low_bb_per_100"]),
                "ci_high_bb_per_100": float(net["ci95_high_bb_per_100"]),
                "translation_clipped_count": translation_clipped,
                "report_path": str(report_path),
            }
            run_rows.append(row)
            print(
                f"run opponent={opponent_label:<12} seed={seed:<5} hands={row['hands']:<8} "
                f"bb/100={row['bb_per_100']:+.2f} std={row['stddev_bb']:.4f} "
                f"95%CI=[{row['ci_low_bb_per_100']:+.2f},{row['ci_high_bb_per_100']:+.2f}] "
                f"report={report_path}"
            )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in run_rows:
        grouped.setdefault(str(row["opponent"]), []).append(row)

    print("\n-- Aggregated Summary --")
    print("opponent | runs | total_hands | mean_bb100 | pooled_CI | min_bb100 | max_bb100")
    for opponent in sorted(grouped.keys()):
        stats = pooled_stats_from_runs(grouped[opponent])
        print(
            f"{opponent} | {int(stats['runs'])} | {int(stats['total_hands'])} | "
            f"{stats['mean_bb100']:+.2f} | "
            f"[{stats['ci_low_bb100']:+.2f}, {stats['ci_high_bb100']:+.2f}] | "
            f"{stats['min_bb100']:+.2f} | {stats['max_bb100']:+.2f}"
        )

    overall = pooled_stats_from_runs(run_rows)
    print("\n-- Overall Aggregate --")
    print(
        f"runs={int(overall['runs'])} total_hands={int(overall['total_hands'])} "
        f"mean_bb100={overall['mean_bb100']:+.2f} "
        f"pooled_CI=[{overall['ci_low_bb100']:+.2f}, {overall['ci_high_bb100']:+.2f}] "
        f"range=[{overall['min_bb100']:+.2f}, {overall['max_bb100']:+.2f}]"
    )

    matrix_flop_context_rows_by_opponent: Dict[str, List[Dict[str, Any]]] = {}
    matrix_flop_total_events_by_opponent: Dict[str, int] = {}
    matrix_turn_no_bet_rows_by_opponent: Dict[str, Dict[str, Any]] = {}
    matrix_river_no_bet_rows_by_opponent: Dict[str, Dict[str, Any]] = {}
    matrix_raised_medium_tiny_rows_by_opponent: Dict[str, Dict[str, Any]] = {}
    matrix_teacher_dashboard_by_opponent: Dict[str, Dict[str, Any]] = {}
    matrix_teacher_top_leaks_by_opponent: Dict[str, Dict[str, Any]] = {}
    for opponent in sorted(grouped.keys()):
        context_totals = matrix_flop_context_totals_by_opponent.get(opponent, {})
        context_action_stats = matrix_flop_context_action_stats_by_opponent.get(opponent, {})
        total_events = sum(int(value) for value in context_totals.values())
        matrix_flop_total_events_by_opponent[opponent] = total_events
        context_rows: List[Dict[str, Any]] = []
        for context_name, total_count in sorted(
            context_totals.items(),
            key=lambda item: (-int(item[1]), item[0]),
        ):
            action_breakdown: List[Dict[str, Any]] = []
            for action_name, stats in sorted(
                context_action_stats.get(context_name, {}).items(),
                key=lambda item: (-int(item[1].get("count", 0.0)), item[0]),
            ):
                count_value = int(stats.get("count", 0.0))
                total_bb_value = float(stats.get("total_bb", 0.0))
                action_breakdown.append(
                    {
                        "action": action_name,
                        "count": count_value,
                        "freq_pct_within_context": pct(count_value, int(total_count)),
                        "freq_pct_overall": pct(count_value, total_events),
                        "total_bb": total_bb_value,
                        "avg_bb": (total_bb_value / count_value) if count_value > 0 else 0.0,
                    }
                )
            context_rows.append(
                {
                    "preflop_context": context_name,
                    "count": int(total_count),
                    "freq_pct": pct(int(total_count), total_events),
                    "action_breakdown": action_breakdown,
                }
            )
        matrix_flop_context_rows_by_opponent[opponent] = context_rows
        turn_context_totals = matrix_turn_context_totals_by_opponent.get(opponent, {})
        turn_total_events = sum(int(value) for value in turn_context_totals.values())
        (
            turn_action_rows,
            _turn_pot_bucket_rows,
            turn_context_rows,
        ) = build_no_bet_tracker_rows(
            action_stats=matrix_turn_action_stats_by_opponent.get(opponent, {}),
            pot_totals={},
            action_pot_stats={},
            context_totals=turn_context_totals,
            action_context_stats=matrix_turn_context_action_stats_by_opponent.get(opponent, {}),
            total_events=turn_total_events,
        )
        matrix_turn_no_bet_rows_by_opponent[opponent] = {
            "total_events": int(turn_total_events),
            "action_summary": turn_action_rows,
            "preflop_context_summary": turn_context_rows,
        }
        river_context_totals = matrix_river_context_totals_by_opponent.get(opponent, {})
        river_total_events = sum(int(value) for value in river_context_totals.values())
        (
            river_action_rows,
            _river_pot_bucket_rows,
            river_context_rows,
        ) = build_no_bet_tracker_rows(
            action_stats=matrix_river_action_stats_by_opponent.get(opponent, {}),
            pot_totals={},
            action_pot_stats={},
            context_totals=river_context_totals,
            action_context_stats=matrix_river_context_action_stats_by_opponent.get(opponent, {}),
            total_events=river_total_events,
        )
        matrix_river_no_bet_rows_by_opponent[opponent] = {
            "total_events": int(river_total_events),
            "action_summary": river_action_rows,
            "preflop_context_summary": river_context_rows,
        }

        suppression_totals = matrix_raised_medium_tiny_by_opponent.get(
            opponent,
            {
                "total_events": 0,
                "raised_or_3bp_medium_tiny_minbet_prevented_count": 0,
                "raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb": 0.0,
                "action_stats": {},
            },
        )
        suppression_action_rows: List[Dict[str, Any]] = []
        for action_name, stats in sorted(
            suppression_totals.get("action_stats", {}).items(),
            key=lambda item: (-int(item[1].get("count", 0.0)), item[0]),
        ):
            count_value = int(stats.get("count", 0.0))
            total_bb_value = float(stats.get("total_bb", 0.0))
            suppression_action_rows.append(
                {
                    "action": action_name,
                    "count": count_value,
                    "freq_pct": pct(count_value, int(suppression_totals.get("total_events", 0))),
                    "total_bb": total_bb_value,
                    "avg_bb": (total_bb_value / count_value) if count_value > 0 else 0.0,
                }
            )
        suppressed_count = int(
            suppression_totals.get("raised_or_3bp_medium_tiny_minbet_prevented_count", 0)
        )
        total_suppression_events = int(suppression_totals.get("total_events", 0))
        matrix_raised_medium_tiny_rows_by_opponent[opponent] = {
            "total_events": total_suppression_events,
            "raised_or_3bp_medium_tiny_minbet_prevented_count": suppressed_count,
            "raised_or_3bp_medium_tiny_minbet_prevented_freq_pct": pct(
                suppressed_count, total_suppression_events
            ),
            "raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb": float(
                suppression_totals.get("raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb", 0.0)
            ),
            "action_summary": suppression_action_rows,
        }
        teacher_rows = matrix_teacher_rows_by_opponent.get(opponent, [])
        teacher_dashboard = build_teacher_disagreement_dashboard(teacher_rows)
        matrix_teacher_dashboard_by_opponent[opponent] = teacher_dashboard
        teacher_top_leaks = build_top_leaks_summary(teacher_rows, top_n=10)
        matrix_teacher_top_leaks_by_opponent[opponent] = {
            "total_processed_samples": int(teacher_dashboard.get("total_processed_samples", 0)),
            "overall_disagreement_rate": float(teacher_dashboard.get("overall_disagreement_rate", 0.0)),
            "top_leaks": list(teacher_top_leaks.get("overall", [])),
        }

    translation_clipped_total = sum(int(value) for value in matrix_translation_clipped_by_opponent.values())
    print("\n-- Translation Clipped Totals --")
    for opponent in sorted(grouped.keys()):
        print(
            f"{opponent}: {int(matrix_translation_clipped_by_opponent.get(opponent, 0))}"
        )
    print(f"overall: {int(translation_clipped_total)}")

    aggregate_path = args.runs_dir / "aggregate_summary.json"
    aggregate_payload = {
        "config": {
            "match_format": "hu" if players == 2 else "6max",
            "players": players,
            "seeds": seeds,
            "opponents": opponent_profiles,
            "hands_per_seed": hands_per_seed,
            "hero_mode": args.hero_mode,
        },
        "runs": run_rows,
        "by_opponent": {name: pooled_stats_from_runs(rows) for name, rows in grouped.items()},
        "diagnostics_by_opponent": {
            name: {
                "translation_clipped_total": int(
                    matrix_translation_clipped_by_opponent.get(name, 0)
                ),
                "flop_no_bet_tracker": {
                    "total_events": int(matrix_flop_total_events_by_opponent.get(name, 0)),
                    "preflop_context_summary": matrix_flop_context_rows_by_opponent.get(name, []),
                },
                "turn_no_bet_tracker": matrix_turn_no_bet_rows_by_opponent.get(
                    name,
                    {
                        "total_events": 0,
                        "action_summary": [],
                        "preflop_context_summary": [],
                    },
                ),
                "river_no_bet_tracker": matrix_river_no_bet_rows_by_opponent.get(
                    name,
                    {
                        "total_events": 0,
                        "action_summary": [],
                        "preflop_context_summary": [],
                    },
                ),
                "raised_or_3bp_medium_tiny_minbet_suppression": matrix_raised_medium_tiny_rows_by_opponent.get(
                    name,
                    {
                        "total_events": 0,
                        "raised_or_3bp_medium_tiny_minbet_prevented_count": 0,
                        "raised_or_3bp_medium_tiny_minbet_prevented_freq_pct": 0.0,
                        "raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb": 0.0,
                        "action_summary": [],
                    },
                ),
                "river_cfr_teacher_disagreement_dashboard": matrix_teacher_dashboard_by_opponent.get(
                    name,
                    {
                        "total_processed_samples": 0,
                        "total_disagreements": 0,
                        "overall_disagreement_rate": 0.0,
                        "by_preflop_context": [],
                        "by_board_texture": [],
                        "by_range_source": [],
                        "top_transitions": [],
                    },
                ),
                "river_cfr_teacher_top_leaks": matrix_teacher_top_leaks_by_opponent.get(
                    name,
                    {
                        "total_processed_samples": 0,
                        "overall_disagreement_rate": 0.0,
                        "top_leaks": [],
                    },
                ),
            }
            for name in sorted(grouped.keys())
        },
        "translation_clipped_totals": {
            "overall": int(translation_clipped_total),
            "by_opponent": {
                name: int(matrix_translation_clipped_by_opponent.get(name, 0))
                for name in sorted(grouped.keys())
            },
        },
        "overall": overall,
    }
    with aggregate_path.open("w", encoding="utf-8") as fp:
        json.dump(aggregate_payload, fp, indent=2)
    print(f"aggregate saved: {aggregate_path}")
    return 0


def run_hand(
    hand_index: int,
    table: Table,
    hero_id: int,
    worker: PolicyWorkerClient,
    rng: random.Random,
    bots: Dict[int, BotBase],
    hero_mode: str,
    status_counts: Dict[str, int],
    max_actions_per_hand: int,
    trace_enabled: bool,
    open_size_bb: float,
    threebet_size_bb: float,
    fourbet_size_bb: float,
    river_cfr_config: Dict[str, Any],
    river_cfr_state: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    debug_trace_nonpremium = env_flag("WIPOKER_DEBUG_TRACE_NONPREMIUM_FACING3BET_RAISE", False)
    debug_trace_any = env_flag("WIPOKER_DEBUG_TRACE_ANY_FACING3BET_RAISE", False)
    context = HandContext()
    obs = table.reset()
    done = False
    hand_debug: Dict[str, Any] = {"engine_calls": 0, "fallback_calls": 0}
    hand_trace_events: List[Dict[str, Any]] = []
    action_sequence: List[Dict[str, Any]] = []
    hero_decisions: List[Dict[str, Any]] = []
    hero_preflop_events: List[Dict[str, Any]] = []
    captured_preflop_raise_trace: Optional[Dict[str, Any]] = None
    hero_player_start = get_player_by_identifier(table, hero_id)
    forced_blinds_antes = float(hero_player_start.money_in_pot)

    hand_metrics: Dict[str, Any] = {
        "street_decisions": {street: 0 for street in ALL_STREETS},
        "street_ok": {street: 0 for street in ALL_STREETS},
        "street_unavailable": {street: 0 for street in ALL_STREETS},
        "street_fallbacks": {street: 0 for street in ALL_STREETS},
        "strategy_store_diagnostics": None,
        "forced_blinds_antes": forced_blinds_antes,
        "voluntary_contrib": {"preflop": 0.0, "flop": 0.0, "turn": 0.0, "river": 0.0},
        "hero_vpip": False,
        "hero_pfr": False,
        "hero_threebet_opp": False,
        "hero_threebet": False,
        "postflop_aggr": {street: 0 for street in POSTFLOP_STREETS},
        "postflop_calls": {street: 0 for street in POSTFLOP_STREETS},
        "fold_to_bet_opportunities": {street: 0 for street in POSTFLOP_STREETS},
        "fold_to_bet_folds": {street: 0 for street in POSTFLOP_STREETS},
        "showdown": False,
        "translation_events": [],
        "preflop_node_events": [],
        "flop_no_bet_events": [],
        "turn_no_bet_events": [],
        "river_no_bet_events": [],
        "hu_unopened_first_action": None,
        "open_raise_then_fold_to_3bet": {"count": 0, "loss_proxy_bb": 0.0},
        "limp_iso_outcome": None,
        "limp_iso_response": None,
        "ev_decomposition": {
            "forced_blinds_antes": 0.0,
            "preflop_voluntary": 0.0,
            "flop": 0.0,
            "turn": 0.0,
            "river": 0.0,
            "attribution_total": 0.0,
            "attribution_residual": 0.0,
        },
        "river_cfr_calls": 0,
        "river_cfr_skipped_no_range": 0,
        "river_cfr_parse_fail": 0,
        "river_cfr_runtime_ms_sum": 0.0,
        "river_cfr_teacher_samples": [],
    }
    action_count = 0

    while not done:
        action_count += 1
        if action_count > max_actions_per_hand:
            raise RuntimeError(
                f"hand exceeded max actions ({max_actions_per_hand}); "
                "simulator entered a runaway loop"
            )

        acting_identifier = int(obs[0])
        actor = get_player_by_identifier(table, acting_identifier)
        street = street_key_from_state(table.street)
        valid_actions = table._get_valid_actions(actor)
        can = build_capabilities(valid_actions)

        pre_bet_to_match = float(table.bet_to_match)
        pre_to_call = max(0.0, float(table.bet_to_match - actor.bet_this_street))
        pre_pot = float(table.pot)
        pre_stack = float(actor.stack + actor.money_in_pot)

        chosen_intent = "check"
        chosen_bucket: Optional[str] = None
        requested_size_bb: Optional[float] = None
        engine_proposed_bb: Optional[float] = None
        engine_final_bb: Optional[float] = None
        was_capped_in_engine = False
        engine_response_payload: Optional[Dict[str, Any]] = None
        hand_state_payload: Optional[Dict[str, Any]] = None
        facing_action = infer_facing_action(street, table, actor, context)
        engine_status = "skipped"
        engine_mix: Optional[Dict[str, float]] = None
        selection_meta: Dict[str, Any] = {
            "hero_mode_requested": str(hero_mode),
            "selection_source": "fallback",
            "mix_branch": "none",
            "fallback_reason": "",
            "worker_executed_intent": None,
            "worker_recommended_intent": None,
            "worker_argmax_intent": None,
            "chosen_intent": None,
            "chosen_matches_worker_executed": False,
            "chosen_matches_worker_recommended": False,
            "chosen_matches_worker_argmax": False,
        }
        exact_hit_policy = False

        if acting_identifier == hero_id:
            hand_metrics["street_decisions"][street] += 1
            if hero_mode == "always_fold":
                chosen_intent, chosen_bucket = intent_from_always_fold_mode(can)
                engine_status = "mode_override:always_fold"
                selection_meta = {
                    **selection_meta,
                    "selection_source": "fallback",
                    "fallback_reason": "mode_override:always_fold",
                    "chosen_intent": chosen_intent,
                }
            elif hero_mode == "random":
                chosen_intent, chosen_bucket = intent_from_random_mode(can, pre_to_call, rng)
                engine_status = "mode_override:random"
                selection_meta = {
                    **selection_meta,
                    "selection_source": "fallback",
                    "fallback_reason": "mode_override:random",
                    "chosen_intent": chosen_intent,
                }
            else:
                hand_state = build_hand_state(
                    table,
                    hero_id,
                    context,
                    include_range_context=bool(river_cfr_config.get("override_enabled")),
                )
                primary_villain_id: Optional[int] = None
                with contextlib.suppress(Exception):
                    primary_villain = choose_primary_villain(table, hero_id, context)
                    primary_villain_id = int(primary_villain.identifier)
                hand_state["publicActionHistory"] = _compact_public_action_history(
                    action_sequence=action_sequence,
                    hero_id=hero_id,
                    primary_villain_id=primary_villain_id,
                )
                hand_state["deepCfrContext"] = build_deep_cfr_worker_context(
                    table=table,
                    hero_id=hero_id,
                    actor_id=acting_identifier,
                    context=context,
                    action_sequence=action_sequence,
                )
                hand_state_payload = hand_state
                response = worker.recommend(street, hand_state)
                engine_response_payload = response
                store_diag = response.get("strategyStoreDiagnostics")
                if isinstance(store_diag, dict):
                    hand_metrics["strategy_store_diagnostics"] = dict(store_diag)
                    exact_hit_policy = bool(
                        store_diag.get("used_store_result", False)
                        and str(store_diag.get("last_lookup_quality", "")).strip().lower()
                        == "exact"
                    )
                status = str(response.get("status", "unavailable"))
                status_counts[status] = status_counts.get(status, 0) + 1
                hand_debug["engine_calls"] += 1
                engine_status = status
                if status == "ok":
                    baseline_intent, _baseline_bucket, *_baseline_tail = intent_from_engine_result(
                        response,
                        rng,
                        "argmax",
                    )
                    if (
                        bool(river_cfr_config.get("teacher_enabled"))
                        and street == "river"
                        and facing_action == "NO_BET"
                    ):
                        sample = build_river_teacher_sample(
                            hand_index=hand_index,
                            table=table,
                            hero_id=hero_id,
                            context=context,
                            hand_state=hand_state,
                            action_sequence=action_sequence,
                            chosen_action_baseline=str(baseline_intent),
                        )
                        if sample is not None:
                            hand_metrics["river_cfr_teacher_samples"].append(sample)
                    if bool(river_cfr_config.get("override_enabled")):
                        response = maybe_apply_river_cfr_override(
                            street=street,
                            facing_action=facing_action,
                            hand_state=hand_state,
                            base_response=response,
                            hand_metrics=hand_metrics,
                            river_cfr_config=river_cfr_config,
                            river_cfr_state=river_cfr_state,
                            hand_index=hand_index,
                        )
                        engine_response_payload = response
                    hand_metrics["street_ok"][street] += 1
                elif status == "unavailable":
                    hand_metrics["street_unavailable"][street] += 1
                engine_mix = compact_mix(response.get("mix"))

                if status == "ok":
                    (
                        chosen_intent,
                        chosen_bucket,
                        requested_size_bb,
                        engine_proposed_bb,
                        engine_final_bb,
                        was_capped_in_engine,
                        selection_meta,
                    ) = intent_from_engine_result(
                        response,
                        rng,
                        hero_mode,
                    )
                else:
                    chosen_intent, chosen_bucket = intent_from_safe_fallback(facing_action, can)
                    requested_size_bb = None
                    engine_proposed_bb = None
                    engine_final_bb = None
                    was_capped_in_engine = False
                    selection_meta = {
                        "hero_mode_requested": str(hero_mode),
                        "selection_source": "fallback",
                        "mix_branch": "none",
                        "fallback_reason": f"engine_status={status}",
                        "worker_executed_intent": None,
                        "worker_recommended_intent": None,
                        "worker_argmax_intent": None,
                        "chosen_intent": chosen_intent,
                        "chosen_matches_worker_executed": False,
                        "chosen_matches_worker_recommended": False,
                        "chosen_matches_worker_argmax": False,
                    }
                    hand_debug["fallback_calls"] += 1
                    hand_metrics["street_fallbacks"][street] += 1
        else:
            bot = bots.get(acting_identifier, BOT_REGISTRY["call_station"])
            chosen_intent = bot.choose_intent(rng, can, pre_to_call)
            if chosen_intent == "bet75":
                chosen_bucket = "75"
            elif chosen_intent == "bet50":
                chosen_bucket = "50"
            elif chosen_intent == "bet33":
                chosen_bucket = "33"

        preflop_raise_count_before_action = context.preflop_raise_count
        action, translation_info = build_action_from_intent(
            chosen_intent,
            table,
            actor,
            valid_actions,
            pre_pot,
            street=street,
            preflop_raise_count_before_action=preflop_raise_count_before_action,
            pre_stack_bb=pre_stack,
            apply_hero_preflop_sizing=(acting_identifier == hero_id),
            open_size_bb=open_size_bb,
            threebet_size_bb=threebet_size_bb,
            fourbet_size_bb=fourbet_size_bb,
            requested_size_bb=requested_size_bb if acting_identifier == hero_id else None,
            exact_hit_policy=bool(exact_hit_policy) if acting_identifier == hero_id else False,
            preserve_exact_postflop_size=bool(EXACT_ACTION_FIDELITY_POSTFLOP),
            relax_exact_postflop_raise_guardrail=bool(
                RELAX_EXACT_POSTFLOP_RAISE_GUARDRAIL
            ),
        )
        translated_action: Dict[str, Any] = {
            "type": action.action_type.name,
        }
        if action.action_type == PlayerAction.BET:
            translated_action["bet_amount"] = round(float(action.bet_amount), 6)

        obs, rewards, done, _info = table.step(action)

        actual_action = actor.history[-1]["action"] if actor.history else action.action_type
        env_legalized_action: Optional[Dict[str, Any]] = None
        requested_action_type_name = str(translation_info.get("requested_action_type", action.action_type.name))
        if requested_action_type_name != action.action_type.name:
            env_legalized_action = {
                "requested_type": requested_action_type_name,
                "executed_type": action.action_type.name,
            }
        if actual_action != action.action_type:
            if env_legalized_action is not None:
                env_legalized_action["executed_type"] = actual_action.name
            else:
                env_legalized_action = {
                    "requested_type": action.action_type.name,
                    "executed_type": actual_action.name,
                }

        action_value = 0.0
        if actor.history:
            last_entry = actor.history[-1]
            raw_value = last_entry.get("value", 0.0) if isinstance(last_entry, dict) else 0.0
            action_value = float(raw_value) if raw_value is not None else 0.0

        action_sequence.append(
            {
                "street": street,
                "actor_id": acting_identifier,
                "pot_bb_before": round(pre_pot, 6),
                "stack_bb_before": round(pre_stack, 6),
                "to_call_bb_before": round(pre_to_call, 6),
                "facing_action": facing_action if acting_identifier == hero_id else None,
                "chosen_intent": chosen_intent,
                "requested_action_type": requested_action_type_name,
                "translated_action_type": action.action_type.name,
                "translated_bet_amount_bb": round(float(action.bet_amount), 6)
                if action.action_type == PlayerAction.BET
                else None,
                "executed_action_type": actual_action.name,
                "executed_amount_bb": round(action_value, 6),
            }
        )

        update_context_after_action(context, street, actor, hero_id, actual_action, pre_bet_to_match)
        update_hu_preflop_pot_context(
            context=context,
            table=table,
            street=street,
            action_type=actual_action,
            pre_to_call=pre_to_call,
            preflop_raise_count_before_action=preflop_raise_count_before_action,
        )

        if acting_identifier == hero_id:
            invested = max(0.0, action_value)
            if street == "preflop":
                hand_metrics["voluntary_contrib"]["preflop"] += invested
                if invested > 0:
                    hand_metrics["hero_vpip"] = True
                if actual_action == PlayerAction.BET:
                    hand_metrics["hero_pfr"] = True
                if pre_to_call > 0 and preflop_raise_count_before_action == 1:
                    hand_metrics["hero_threebet_opp"] = True
                    if actual_action == PlayerAction.BET:
                        hand_metrics["hero_threebet"] = True
                hero_preflop_events.append(
                    {
                        "preflop_raise_count_before": preflop_raise_count_before_action,
                        "pre_to_call": pre_to_call,
                        "action_type": actual_action.name,
                        "action_bucket": action_bucket_from_action_name(actual_action.name),
                        "action_sequence_index": len(action_sequence) - 1,
                        "raise_size_bb": action_value if actual_action == PlayerAction.BET else None,
                        "invested_bb": invested,
                    }
                )

                is_facing_3bet_decision = (
                    preflop_raise_count_before_action == 2 and pre_to_call > 1e-9
                )
                if is_facing_3bet_decision and captured_preflop_raise_trace is None:
                    hero_cards = (
                        list(hand_state_payload.get("preflop", {}).get("heroHand", []))
                        if isinstance(hand_state_payload, dict)
                        else []
                    )
                    hand_key = hero_cards_to_hand_key(hero_cards)
                    is_premium = is_hu_facing_3bet_premium(hero_cards)
                    final_action_is_raise = (
                        chosen_intent in ("raise", "all_in")
                        or action.action_type == PlayerAction.BET
                        or actual_action == PlayerAction.BET
                    )
                    should_capture = final_action_is_raise and (
                        debug_trace_any or (debug_trace_nonpremium and not is_premium)
                    )
                    if should_capture:
                        chart_mix_raw = None
                        post_guardrail_mix = None
                        guardrail_action = None
                        if isinstance(engine_response_payload, dict):
                            debug_payload = engine_response_payload.get("debug")
                            if isinstance(debug_payload, dict):
                                chart_mix_raw = debug_payload.get("baselineMix")
                                post_guardrail_mix = debug_payload.get("adjustedMix")
                            post_guardrail_mix = (
                                post_guardrail_mix
                                if post_guardrail_mix is not None
                                else engine_response_payload.get("mix")
                            )
                            rec_action = engine_response_payload.get("recommendedAction")
                            if isinstance(rec_action, dict):
                                rec_type = rec_action.get("type")
                                guardrail_action = str(rec_type) if rec_type is not None else None

                        captured_preflop_raise_trace = {
                            "match_format": "hu",
                            "hand_index": hand_index,
                            "trigger_mode": (
                                "any_facing_3bet_raise"
                                if debug_trace_any and not debug_trace_nonpremium
                                else "nonpremium_facing_3bet_raise"
                                if debug_trace_nonpremium and not debug_trace_any
                                else "nonpremium_or_any_facing_3bet_raise"
                            ),
                            "input_state": {
                                "heroHand": hero_cards,
                                "heroHandKey": hand_key,
                                "isPremiumFacing3Bet": is_premium,
                                "heroPos": hand_state_payload.get("preflop", {}).get("heroPos")
                                if isinstance(hand_state_payload, dict)
                                else None,
                                "villainPos": hand_state_payload.get("preflop", {}).get("villainPos")
                                if isinstance(hand_state_payload, dict)
                                else None,
                                "spot": hand_state_payload.get("preflop", {}).get("spot")
                                if isinstance(hand_state_payload, dict)
                                else None,
                                "potSizeBb": pre_pot,
                                "toCallBb": pre_to_call,
                                "heroStackBb": pre_stack,
                                "effectiveStackRemainingBb": hand_state_payload.get("flop", {}).get(
                                    "effectiveStackBb"
                                )
                                if isinstance(hand_state_payload, dict)
                                else None,
                            },
                            "engine_chart_recommendation_raw": chart_mix_raw,
                            "engine_explanation": engine_response_payload.get("explanation")
                            if isinstance(engine_response_payload, dict)
                            else None,
                            "engine_post_guardrail": {
                                "recommendedActionType": guardrail_action,
                                "mix": post_guardrail_mix,
                            },
                            "engine_final_selection_before_translator": {
                                "chosen_intent": chosen_intent,
                                "size_bucket": chosen_bucket,
                                "engine_proposed_bb": engine_proposed_bb,
                                "engine_final_bb": engine_final_bb,
                                "was_capped_in_engine": bool(was_capped_in_engine),
                            },
                            "translator_output": {
                                "requested_action_type": translation_info.get("requested_action_type"),
                                "translated_action_type": translation_info.get("translated_action_type"),
                                "executed_action_type": actual_action.name,
                                "translated_bet_amount_bb": translation_info.get("sent_bet_amount"),
                                "target_bet_amount_bb": translation_info.get("target_bet_amount"),
                                "clipped_to_bounds": bool(translation_info.get("clipped_to_bounds")),
                                "translation_adjusted": bool(translation_info.get("translation_adjusted")),
                                "preflop_raise_clamped": bool(translation_info.get("preflop_raise_clamped")),
                                "preflop_raise_converted_to_allin": bool(
                                    translation_info.get("preflop_raise_converted_to_allin")
                                ),
                            },
                        }
            elif street in POSTFLOP_STREETS:
                hand_metrics["voluntary_contrib"][street] += invested
                if actual_action == PlayerAction.BET:
                    hand_metrics["postflop_aggr"][street] += 1
                if actual_action == PlayerAction.CALL:
                    hand_metrics["postflop_calls"][street] += 1
                if facing_action in ("FACING_BET", "FACING_RAISE", "FACING_CHECK_RAISE"):
                    hand_metrics["fold_to_bet_opportunities"][street] += 1
                    if actual_action == PlayerAction.FOLD:
                        hand_metrics["fold_to_bet_folds"][street] += 1

                if street == "flop" and facing_action == "NO_BET":
                    sent_amount = translation_info.get("sent_bet_amount")
                    sent_amount_bb: Optional[float] = None
                    if isinstance(sent_amount, (int, float)) and math.isfinite(float(sent_amount)):
                        sent_amount_bb = float(sent_amount)
                    potaware_counter_tags = extract_flop_potaware_counter_tags(engine_response_payload)
                    action_label = classify_flop_no_bet_action(
                        action_type=actual_action,
                        chosen_intent=chosen_intent,
                        sent_amount_bb=sent_amount_bb,
                        pot_bb=pre_pot,
                    )
                    tiny_pot = (pre_pot * 0.33) < (ENGINE_MIN_BET_BB - 1e-9)
                    preflop_context = "unknown"
                    if isinstance(hand_state_payload, dict):
                        preflop_state = hand_state_payload.get("preflop", {})
                        if isinstance(preflop_state, dict):
                            preflop_context = classify_postflop_preflop_context(preflop_state)
                    raised_medium_tiny_counter_tags = extract_raised_medium_tiny_counter_tags(
                        engine_response_payload
                    )
                    raised_or_3bp_medium_tiny_context = (
                        "flop_potaware_raised_medium_tiny_action_check" in raised_medium_tiny_counter_tags
                        or "flop_potaware_raised_medium_tiny_action_minbet" in raised_medium_tiny_counter_tags
                    )
                    raised_or_3bp_medium_tiny_prevented = (
                        "flop_potaware_raised_medium_tiny_minbet_prevented"
                        in raised_medium_tiny_counter_tags
                    )
                    hand_metrics["flop_no_bet_events"].append(
                        {
                            "action": action_label,
                            "chosen_intent": chosen_intent,
                            "pot_bb": pre_pot,
                            "pot_bucket": "tiny_pot_lt_minbet33" if tiny_pot else "normal_pot",
                            "translated_bet_amount_bb": sent_amount_bb,
                            "preflop_context": preflop_context,
                            "potaware_counter_tags": potaware_counter_tags,
                            "raised_or_3bp_medium_tiny_pot": raised_or_3bp_medium_tiny_context,
                            "raised_or_3bp_medium_tiny_minbet_prevented": raised_or_3bp_medium_tiny_prevented,
                            "raised_medium_tiny_counter_tags": raised_medium_tiny_counter_tags,
                        }
                    )

                if street == "turn" and facing_action == "NO_BET":
                    sent_amount = translation_info.get("sent_bet_amount")
                    sent_amount_bb: Optional[float] = None
                    if isinstance(sent_amount, (int, float)) and math.isfinite(float(sent_amount)):
                        sent_amount_bb = float(sent_amount)
                    potaware_counter_tags = extract_turn_potaware_counter_tags(engine_response_payload)
                    action_label = classify_flop_no_bet_action(
                        action_type=actual_action,
                        chosen_intent=chosen_intent,
                        sent_amount_bb=sent_amount_bb,
                        pot_bb=pre_pot,
                    )
                    tiny_pot = (pre_pot * 0.33) < (ENGINE_MIN_BET_BB - 1e-9)
                    preflop_context = "unknown"
                    if isinstance(hand_state_payload, dict):
                        preflop_state = hand_state_payload.get("preflop", {})
                        if isinstance(preflop_state, dict):
                            preflop_context = classify_postflop_preflop_context(preflop_state)
                    hand_metrics["turn_no_bet_events"].append(
                        {
                            "action": action_label,
                            "chosen_intent": chosen_intent,
                            "pot_bb": pre_pot,
                            "pot_bucket": "tiny_pot_lt_minbet33" if tiny_pot else "normal_pot",
                            "translated_bet_amount_bb": sent_amount_bb,
                            "preflop_context": preflop_context,
                            "potaware_counter_tags": potaware_counter_tags,
                        }
                    )

                if street == "river" and facing_action == "NO_BET":
                    sent_amount = translation_info.get("sent_bet_amount")
                    sent_amount_bb: Optional[float] = None
                    if isinstance(sent_amount, (int, float)) and math.isfinite(float(sent_amount)):
                        sent_amount_bb = float(sent_amount)
                    potaware_counter_tags = extract_river_potaware_counter_tags(engine_response_payload)
                    action_label = classify_flop_no_bet_action(
                        action_type=actual_action,
                        chosen_intent=chosen_intent,
                        sent_amount_bb=sent_amount_bb,
                        pot_bb=pre_pot,
                    )
                    tiny_pot = (pre_pot * 0.33) < (ENGINE_MIN_BET_BB - 1e-9)
                    preflop_context = "unknown"
                    if isinstance(hand_state_payload, dict):
                        preflop_state = hand_state_payload.get("preflop", {})
                        if isinstance(preflop_state, dict):
                            preflop_context = classify_postflop_preflop_context(preflop_state)
                    hand_metrics["river_no_bet_events"].append(
                        {
                            "action": action_label,
                            "chosen_intent": chosen_intent,
                            "pot_bb": pre_pot,
                            "pot_bucket": "tiny_pot_lt_minbet33" if tiny_pot else "normal_pot",
                            "translated_bet_amount_bb": sent_amount_bb,
                            "preflop_context": preflop_context,
                            "potaware_counter_tags": potaware_counter_tags,
                        }
                    )

            record_street_history_if_hero(
                context=context,
                street=street,
                actor=actor,
                hero_id=hero_id,
                hero_ip=infer_hero_ip(table, hero_id),
                facing_action=facing_action,
                action_type=actual_action,
                pre_to_call=pre_to_call,
                pre_pot=pre_pot,
                size_bucket_hint=chosen_bucket,
            )

            if actual_action == PlayerAction.BET:
                sent_amount = translation_info.get("sent_bet_amount")
                action_is_raise = street == "preflop" or pre_to_call > 1e-9 or chosen_intent in ("raise", "all_in")
                illegal_action_avoided = bool(translation_info.get("translation_adjusted")) or (
                    env_legalized_action is not None
                )
                hand_metrics["translation_events"].append(
                    {
                        "street": street,
                        "facing_action": facing_action,
                        "stack_bb": pre_stack,
                        "chosen_intent": chosen_intent,
                        "engine_proposed_bb": engine_proposed_bb,
                        "engine_final_bb": engine_final_bb,
                        "was_capped_in_engine": bool(was_capped_in_engine),
                        "pot_bb": pre_pot,
                        "translated_bet_amount_bb": sent_amount,
                        "bet_size_bb": float(sent_amount)
                        if chosen_intent in ("bet33", "bet50", "bet75") and sent_amount is not None
                        else None,
                        "raise_size_bb": float(sent_amount)
                        if action_is_raise and sent_amount is not None
                        else None,
                        "action_is_raise": action_is_raise,
                        "target_bet_amount_bb": translation_info.get("target_bet_amount"),
                        "clipped_to_bounds": bool(translation_info.get("clipped_to_bounds")),
                        "translation_adjusted": bool(translation_info.get("translation_adjusted")),
                        "env_legalized": env_legalized_action is not None,
                        "illegal_action_avoided": illegal_action_avoided,
                        "is_allin": bool(translation_info.get("is_allin")),
                        "preflop_node": translation_info.get("preflop_node"),
                        "preflop_raise_clamped": bool(translation_info.get("preflop_raise_clamped")),
                        "preflop_raise_converted_to_allin": bool(
                            translation_info.get("preflop_raise_converted_to_allin")
                        ),
                        "postflop_raise_clamped": bool(translation_info.get("postflop_raise_clamped")),
                        "exact_requested_size_used": bool(
                            translation_info.get("exact_requested_size_used")
                        ),
                        "postflop_raise_guardrail_relaxed": bool(
                            translation_info.get("postflop_raise_guardrail_relaxed")
                        ),
                        "selection_source": str(selection_meta.get("selection_source", "fallback")),
                        "exact_hit_policy": bool(exact_hit_policy),
                        "chosen_matches_worker_executed": bool(
                            selection_meta.get("chosen_matches_worker_executed", False)
                        ),
                        "chosen_matches_worker_recommended": bool(
                            selection_meta.get("chosen_matches_worker_recommended", False)
                        ),
                        "chosen_matches_worker_argmax": bool(
                            selection_meta.get("chosen_matches_worker_argmax", False)
                        ),
                    }
                )

            hero_decisions.append(
                {
                    "street": street,
                    "facing_action": facing_action,
                    "engine_status": engine_status,
                    "engine_mix": engine_mix,
                    "chosen_action": chosen_intent,
                    "selection_source": str(selection_meta.get("selection_source", "fallback")),
                    "exact_hit_policy": bool(exact_hit_policy),
                    "worker_executed_action": (
                        engine_response_payload.get("executedAction")
                        if isinstance(engine_response_payload, dict)
                        else None
                    ),
                    "worker_recommended_action": (
                        engine_response_payload.get("recommendedAction")
                        if isinstance(engine_response_payload, dict)
                        else None
                    ),
                    "worker_argmax_action": (
                        engine_response_payload.get("argmaxAction")
                        if isinstance(engine_response_payload, dict)
                        else None
                    ),
                    "chosen_matches_worker_executed": bool(
                        selection_meta.get("chosen_matches_worker_executed", False)
                    ),
                    "chosen_matches_worker_recommended": bool(
                        selection_meta.get("chosen_matches_worker_recommended", False)
                    ),
                    "chosen_matches_worker_argmax": bool(
                        selection_meta.get("chosen_matches_worker_argmax", False)
                    ),
                    "translated_action": translated_action,
                    "env_legalized_action": env_legalized_action,
                    "engine_explanation": engine_response_payload.get("explanation")
                    if isinstance(engine_response_payload, dict)
                    else None,
                    "engine_debug": engine_response_payload.get("debug")
                    if isinstance(engine_response_payload, dict)
                    else None,
                    "selection_meta": selection_meta,
                }
            )

            if trace_enabled:
                hand_trace_events.append(
                    {
                        "hand_index": hand_index,
                        "street": street,
                        "pot_bb": round(pre_pot, 6),
                        "stack_bb": round(pre_stack, 6),
                        "facing_action": facing_action,
                        "engine_status": engine_status,
                        "engine_mix": engine_mix,
                        "chosen_action": chosen_intent,
                        "selection_source": str(selection_meta.get("selection_source", "fallback")),
                        "exact_hit_policy": bool(exact_hit_policy),
                        "translated_action": translated_action,
                        "env_legalized_action": env_legalized_action,
                        "bb_delta": None,
                    }
                )

    hero_reward = float(rewards[hero_id]) if rewards[hero_id] is not None else 0.0
    hero_player_end = get_player_by_identifier(table, hero_id)
    payout_bb = float(hero_player_end.winnings_for_hh)
    terminal_street = street_key_from_state(table.street)
    active_players_end = len([p for p in table.players if p.state == PlayerState.ACTIVE])
    hand_metrics["showdown"] = terminal_street == "river" and active_players_end >= 2
    maybe_attach_showdown_villain_cards_metadata(
        hand_metrics=hand_metrics,
        table=table,
        hero_id=hero_id,
        context=context,
    )
    hero_cards = cards_to_codes(hero_player_start.cards)
    hero_hand_bucket = classify_preflop_hand_bucket(hero_cards)
    hero_folded_postflop = any(
        int(event.get("actor_id", -1)) == int(hero_id)
        and str(event.get("street")) in POSTFLOP_STREETS
        and str(event.get("executed_action_type")) == "FOLD"
        for event in action_sequence
    )
    hand_metrics["final_outcome_class"] = classify_final_outcome_class(
        terminal_street=terminal_street,
        showdown=hand_metrics["showdown"],
        hero_folded_postflop=hero_folded_postflop,
        hero_reward=hero_reward,
    )
    hand_metrics["terminal_street"] = terminal_street

    first_hu_unopened_decision = next(
        (
            event
            for event in hero_preflop_events
            if int(event.get("preflop_raise_count_before", -1)) == 0
            and float(event.get("pre_to_call", 0.0)) > 0.0
        ),
        None,
    )
    if isinstance(first_hu_unopened_decision, dict):
        first_action_type = str(first_hu_unopened_decision.get("action_type", "")).upper()
        if first_action_type == "BET":
            hand_metrics["hu_unopened_first_action"] = "raise"
        elif first_action_type == "CALL":
            hand_metrics["hu_unopened_first_action"] = "limp"
        elif first_action_type == "FOLD":
            hand_metrics["hu_unopened_first_action"] = "fold"

        open_raise_event = first_hu_unopened_decision if first_action_type == "BET" else None
        if isinstance(open_raise_event, dict):
            folded_to_3bet = any(
                int(event.get("preflop_raise_count_before", -1)) == 2
                and float(event.get("pre_to_call", 0.0)) > 0.0
                and str(event.get("action_type", "")).upper() == "FOLD"
                for event in hero_preflop_events
            )
            if folded_to_3bet:
                open_raise_size = float(open_raise_event.get("invested_bb", 0.0))
                hand_metrics["open_raise_then_fold_to_3bet"] = {
                    "count": 1,
                    "loss_proxy_bb": max(0.0, open_raise_size),
                }

        if first_action_type == "CALL":
            hero_action_index = int(first_hu_unopened_decision.get("action_sequence_index", -1))
            villain_response = first_villain_preflop_action_after_index(
                action_sequence=action_sequence,
                hero_action_index=hero_action_index,
                hero_id=hero_id,
            )
            if villain_response == "CHECK":
                hand_metrics["limp_iso_outcome"] = "villain_checked"
            elif villain_response == "BET":
                hand_metrics["limp_iso_outcome"] = "villain_raised"
                hero_response_after_iso = next(
                    (
                        event
                        for event in hero_preflop_events
                        if int(event.get("action_sequence_index", -1)) > hero_action_index
                        and int(event.get("preflop_raise_count_before", -1)) == 1
                        and float(event.get("pre_to_call", 0.0)) > 0.0
                    ),
                    None,
                )
                if isinstance(hero_response_after_iso, dict):
                    hero_iso_action_type = str(hero_response_after_iso.get("action_type", "")).upper()
                    hand_metrics["limp_iso_response"] = action_bucket_from_action_name(hero_iso_action_type)
            elif villain_response is None and terminal_street != "preflop":
                hand_metrics["limp_iso_outcome"] = "villain_checked"
            elif isinstance(villain_response, str) and villain_response:
                hand_metrics["limp_iso_outcome"] = "other"

    hand_metrics["preflop_nodes"] = infer_preflop_leak_nodes(hero_preflop_events, terminal_street)
    if captured_preflop_raise_trace is not None:
        hand_metrics["debug_facing3bet_raise_trace"] = {
            **captured_preflop_raise_trace,
            "bb_delta_raw": hero_reward,
            "terminal_street": terminal_street,
            "preflop_nodes_for_hand": hand_metrics["preflop_nodes"],
        }
    hand_metrics["preflop_node_events"] = infer_preflop_node_events_with_context(
        hero_preflop_events=hero_preflop_events,
        action_sequence=action_sequence,
        hero_id=hero_id,
        terminal_street=terminal_street,
        showdown=hand_metrics["showdown"],
        hero_reward=hero_reward,
        hero_hand_bucket=hero_hand_bucket,
    )

    ev = {
        "forced_blinds_antes": -float(hand_metrics["forced_blinds_antes"]),
        "preflop_voluntary": -float(hand_metrics["voluntary_contrib"]["preflop"]),
        "flop": -float(hand_metrics["voluntary_contrib"]["flop"]),
        "turn": -float(hand_metrics["voluntary_contrib"]["turn"]),
        "river": -float(hand_metrics["voluntary_contrib"]["river"]),
    }
    if terminal_street == "preflop":
        ev["preflop_voluntary"] += payout_bb
    elif terminal_street in ("flop", "turn", "river"):
        ev[terminal_street] += payout_bb

    attribution_total = sum(ev.values())
    residual = hero_reward - attribution_total
    hand_metrics["ev_decomposition"] = {
        "forced_blinds_antes": ev["forced_blinds_antes"],
        "preflop_voluntary": ev["preflop_voluntary"],
        "flop": ev["flop"],
        "turn": ev["turn"],
        "river": ev["river"],
        "attribution_total": attribution_total,
        "attribution_residual": residual,
    }

    for event in hand_trace_events:
        event["bb_delta"] = round(hero_reward, 6)
    final_pot = float(table.pot)
    hand_detail = {
        "hand_index": hand_index,
        "bb_delta": round(hero_reward, 6),
        "terminal_street": terminal_street,
        "action_sequence": action_sequence,
        "hero_decisions": hero_decisions,
    }
    return hero_reward, final_pot, hand_debug, hand_trace_events, hand_metrics, hand_detail


def _parse_last_json_line(stdout_text: str) -> Dict[str, Any]:
    for line in reversed(stdout_text.splitlines()):
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("deep_cfr_panel did not emit a JSON payload")


def build_deep_cfr_checkpoint_policy_cmd(args: argparse.Namespace) -> str:
    if args.deep_cfr_model_state is None:
        raise ValueError("--deep-cfr-model-state is required when --eval-backend deep_cfr_checkpoint")
    worker_script = REPO_ROOT / "eval" / "deep_cfr_checkpoint_worker.py"
    if not worker_script.exists():
        raise FileNotFoundError(f"missing deep_cfr checkpoint policy worker: {worker_script}")
    command_parts = [
        sys.executable,
        str(worker_script),
        "--model-state",
        str(args.deep_cfr_model_state.resolve()),
        "--device",
        str(args.deep_cfr_device),
        "--mode",
        str(args.deep_cfr_policy_mode),
        "--seed",
        str(int(args.deep_cfr_policy_seed)),
    ]
    return shlex.join(command_parts)


def run_deep_cfr_panel_backend(
    args: argparse.Namespace,
    *,
    start_time: float,
    preflop_selection_mode_effective: str,
    postflop_selection_mode_effective: str,
) -> int:
    if args.deep_cfr_model_state is None:
        raise ValueError("--deep-cfr-model-state is required when --eval-backend deep_cfr_panel")
    if args.deep_cfr_samples is None:
        raise ValueError("--deep-cfr-samples is required when --eval-backend deep_cfr_panel")
    if int(args.deep_cfr_min_seat_samples) < 0:
        raise ValueError("--deep-cfr-min-seat-samples must be >= 0")
    if not (0.0 <= float(args.deep_cfr_max_dominant_seat_fraction) <= 1.0):
        raise ValueError("--deep-cfr-max-dominant-seat-fraction must be in [0, 1]")

    panel_script = REPO_ROOT / "eval" / "deep_cfr_panel.py"
    if not panel_script.exists():
        raise FileNotFoundError(f"missing deep_cfr panel evaluator: {panel_script}")

    panel_cmd = [
        sys.executable,
        str(panel_script),
        "--model-state",
        str(args.deep_cfr_model_state),
        "--samples",
        str(args.deep_cfr_samples),
        "--sample-type",
        str(args.deep_cfr_sample_type),
        "--max-samples",
        str(int(args.deep_cfr_max_samples)),
        "--batch-size",
        str(int(args.deep_cfr_batch_size)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.deep_cfr_device),
        "--label",
        str(args.deep_cfr_label),
        "--required-seats",
        str(args.deep_cfr_required_seats),
        "--min-seat-samples",
        str(int(args.deep_cfr_min_seat_samples)),
        "--max-dominant-seat-fraction",
        str(float(args.deep_cfr_max_dominant_seat_fraction)),
    ]

    proc = subprocess.run(
        panel_cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"deep_cfr panel backend failed: {tail}")

    panel_result = _parse_last_json_line(proc.stdout)
    if str(panel_result.get("status", "error")) != "ok":
        raise RuntimeError(f"deep_cfr panel backend returned non-ok status: {panel_result}")

    score = float(panel_result.get("score", 0.0))
    score_ci_low = float(panel_result.get("score_ci_low", score))
    score_ci_high = float(panel_result.get("score_ci_high", score))
    raw_stats = {
        "samples": int(panel_result.get("sample_count_used", 0)),
        "metric_kind": "panel_score",
        "panel_score": float(score),
        "panel_score_ci_low": float(score_ci_low),
        "panel_score_ci_high": float(score_ci_high),
    }
    net_stats = dict(raw_stats)
    diagnostics = {
        "evaluator_identity": {
            "backend": "deep_cfr_panel",
            "model_state": str(args.deep_cfr_model_state),
            "samples": str(args.deep_cfr_samples),
            "sample_type": str(args.deep_cfr_sample_type),
            "max_samples": int(args.deep_cfr_max_samples),
            "batch_size": int(args.deep_cfr_batch_size),
            "device": str(args.deep_cfr_device),
            "seed": int(args.seed),
            "label": str(args.deep_cfr_label),
            "required_seats": str(args.deep_cfr_required_seats),
            "min_seat_samples": int(args.deep_cfr_min_seat_samples),
            "max_dominant_seat_fraction": float(args.deep_cfr_max_dominant_seat_fraction),
            "hero_mode": str(args.hero_mode),
        },
        "deep_cfr_panel": panel_result,
        "seat_breakdown": panel_result.get("seat_breakdown", {}),
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "panel_score", "value": f"{score:.6f}"})

    elapsed_total = time.time() - start_time
    if not args.quiet:
        print(
            "[deep_cfr_panel] "
            f"score={score:.3f} sample_type={args.deep_cfr_sample_type} "
            f"samples={panel_result.get('sample_count_used', 0)}"
        )
        print(f"results saved: {args.results}")

    if args.report_json is not None:
        report_payload = {
            "config": {
                "hands": args.hands,
                "hands_per_seed": args.hands_per_seed,
                "players": args.players,
                "match_format": args.match_format,
                "stack_bb": args.stack_bb,
                "seed": args.seed,
                "seeds": args.seeds,
                "hero_mode": args.hero_mode,
                "eval_backend": args.eval_backend,
                "policy_cmd": args.policy_cmd,
                "deep_cfr_model_state": str(args.deep_cfr_model_state),
                "deep_cfr_samples": str(args.deep_cfr_samples),
                "deep_cfr_sample_type": args.deep_cfr_sample_type,
                "deep_cfr_max_samples": int(args.deep_cfr_max_samples),
                "deep_cfr_batch_size": int(args.deep_cfr_batch_size),
                "deep_cfr_device": str(args.deep_cfr_device),
                "deep_cfr_label": str(args.deep_cfr_label),
                "deep_cfr_required_seats": str(args.deep_cfr_required_seats),
                "deep_cfr_min_seat_samples": int(args.deep_cfr_min_seat_samples),
                "deep_cfr_max_dominant_seat_fraction": float(args.deep_cfr_max_dominant_seat_fraction),
                "preflop_selection_mode_effective": preflop_selection_mode_effective,
                "postflop_selection_mode_effective": postflop_selection_mode_effective,
            },
            "summary": {
                "raw": raw_stats,
                "net": net_stats,
                "fallback_count": 0,
                "status_counts": {"deep_cfr_panel": 1},
                "diagnostics": diagnostics,
                "elapsed_sec": float(elapsed_total),
                "results_csv": str(args.results),
                "results_csv_bytes": int(args.results.stat().st_size) if args.results.exists() else 0,
            },
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as fp:
            json.dump(report_payload, fp, indent=2)
        if not args.quiet:
            print(f"report_json saved: {args.report_json}")
            print(f"elapsed: {elapsed_total:.2f}s")

    return 0


def main() -> int:
    start_time = time.time()
    args = parse_args()
    args.seed = normalize_seed_u32(int(args.seed))
    preflop_selection_mode_effective = str(
        os.environ.get("WIPOKER_PREFLOP_SELECTION_MODE")
        or os.environ.get("PREFLOP_SELECTION_MODE")
        or "unknown"
    ).strip().lower()
    postflop_selection_mode_effective = str(
        os.environ.get("WIPOKER_POLICY_SELECTION_MODE")
        or os.environ.get("POLICY_SELECTION_MODE")
        or "unknown"
    ).strip().lower()

    if args.rake_bb_per_hand is not None:
        args.rake_mode = "fixed"
        args.rake_fixed_bb = float(args.rake_bb_per_hand)

    args.players = resolve_players(args.match_format, args.players)

    if args.eval_backend == "deep_cfr_panel":
        return run_deep_cfr_panel_backend(
            args,
            start_time=start_time,
            preflop_selection_mode_effective=preflop_selection_mode_effective,
            postflop_selection_mode_effective=postflop_selection_mode_effective,
        )

    if args.eval_backend == "deep_cfr_checkpoint":
        args.policy_cmd = build_deep_cfr_checkpoint_policy_cmd(args)
        if not args.quiet:
            print(f"[deep_cfr_checkpoint] policy cmd: {args.policy_cmd}")

    if should_run_matrix_mode(args) and not args._single_run:
        return run_matrix_experiments(args)

    if args.hands <= 0:
        raise ValueError("--hands must be > 0")
    if args.hero_id < 0 or args.hero_id >= args.players:
        raise ValueError("--hero-id must be within [0, players-1]")
    if args.open_size <= 0 or args.threebet_size <= 0 or args.fourbet_size <= 0:
        raise ValueError("--open_size/--threebet_size/--fourbet_size must be > 0")

    ensure_repo_commands(args)
    patch_pokerenv_compat()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    bots = build_opponent_list(args.opponent_lineup, args.players)

    table = Table(
        args.players,
        stack_low=int(args.stack_bb),
        stack_high=int(args.stack_bb) + 1,
        hand_history_location=None,
    )
    table.seed(args.seed)

    worker = PolicyWorkerClient(args.policy_cmd, REPO_ROOT)

    raw_deltas: List[float] = []
    net_deltas: List[float] = []
    cumulative_raw = 0.0
    cumulative_net = 0.0
    status_counts: Dict[str, int] = {}
    fallback_count = 0
    street_decisions = {street: 0 for street in ALL_STREETS}
    street_ok = {street: 0 for street in ALL_STREETS}
    street_unavailable = {street: 0 for street in ALL_STREETS}
    street_fallbacks = {street: 0 for street in ALL_STREETS}
    selection_source_counts: Dict[str, int] = {}
    selection_source_by_street: Dict[str, Dict[str, int]] = {
        street: {} for street in ALL_STREETS
    }
    selection_source_by_street_exact_hit: Dict[str, Dict[str, int]] = {
        street: {} for street in ALL_STREETS
    }
    selection_source_by_street_non_exact_hit: Dict[str, Dict[str, int]] = {
        street: {} for street in ALL_STREETS
    }
    chosen_vs_executed_match_count = 0
    chosen_vs_executed_total = 0
    chosen_vs_recommended_match_count = 0
    chosen_vs_recommended_total = 0
    chosen_vs_argmax_match_count = 0
    chosen_vs_argmax_total = 0
    ev_decomposition_totals = {
        "forced_blinds_antes": 0.0,
        "preflop_voluntary": 0.0,
        "flop": 0.0,
        "turn": 0.0,
        "river": 0.0,
        "attribution_total": 0.0,
        "attribution_residual": 0.0,
    }
    vpip_hands = 0
    pfr_hands = 0
    threebet_opportunities = 0
    threebet_hands = 0
    showdown_when_vpip_hands = 0
    postflop_aggr = {street: 0 for street in POSTFLOP_STREETS}
    postflop_calls = {street: 0 for street in POSTFLOP_STREETS}
    fold_to_bet_opportunities = {street: 0 for street in POSTFLOP_STREETS}
    fold_to_bet_folds = {street: 0 for street in POSTFLOP_STREETS}
    bet33_fractions: List[float] = []
    bet50_fractions: List[float] = []
    bet75_fractions: List[float] = []
    bet33_fraction_buckets: Dict[str, int] = {}
    bet50_fraction_buckets: Dict[str, int] = {}
    bet75_fraction_buckets: Dict[str, int] = {}
    bet33_forced_halfpot_count = 0
    bet50_forced_halfpot_count = 0
    bet75_forced_halfpot_count = 0
    raise_sizes_bb: List[float] = []
    raise_fractions: List[float] = []
    translation_clipped_count = 0
    translation_adjusted_count = 0
    env_legalized_count = 0
    clipped_or_adjusted_total = 0
    preflop_raise_clamped_count = 0
    preflop_raise_converted_to_allin_count = 0
    postflop_raise_clamped_count = 0
    exact_requested_size_used_count = 0
    postflop_raise_guardrail_relaxed_count = 0
    exact_hit_and_translated_count = 0
    exact_hit_and_unmodified_count = 0
    exact_hit_translation_by_street: Dict[str, Dict[str, int]] = {
        street: {"translated": 0, "unmodified": 0, "total": 0}
        for street in ALL_STREETS
    }
    engine_size_capped_count = 0
    illegal_action_avoided_count = 0
    clipped_adjusted_by_context: Dict[str, Dict[str, int]] = {}
    engine_size_capped_by_street_intent: Dict[str, int] = {}
    raise_over_20 = 0
    raise_over_40 = 0
    raise_over_80 = 0
    largest_raises: List[Dict[str, Any]] = []
    large_preflop_non_allin_raises: List[Dict[str, Any]] = []
    preflop_open_non_allin_raises_over_15bb: List[Dict[str, Any]] = []
    preflop_node_stats: Dict[str, Dict[str, Any]] = {}
    preflop_node_context_stats: Dict[str, Dict[str, Any]] = {}
    flop_no_bet_action_stats: Dict[str, Dict[str, float]] = {}
    flop_no_bet_action_pot_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    flop_no_bet_pot_totals: Dict[str, int] = {}
    flop_no_bet_context_totals: Dict[str, int] = {}
    flop_no_bet_action_context_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    flop_no_bet_total_events = 0
    turn_no_bet_action_stats: Dict[str, Dict[str, float]] = {}
    turn_no_bet_action_pot_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    turn_no_bet_pot_totals: Dict[str, int] = {}
    turn_no_bet_context_totals: Dict[str, int] = {}
    turn_no_bet_action_context_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    turn_no_bet_total_events = 0
    river_no_bet_action_stats: Dict[str, Dict[str, float]] = {}
    river_no_bet_action_pot_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    river_no_bet_pot_totals: Dict[str, int] = {}
    river_no_bet_context_totals: Dict[str, int] = {}
    river_no_bet_action_context_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    river_no_bet_total_events = 0
    flop_potaware_medium_counters: Dict[str, int] = {
        key: 0 for key in FLOP_POTAWARE_MEDIUM_COUNTER_KEYS
    }
    raised_or_3bp_medium_tiny_action_stats: Dict[str, Dict[str, float]] = {
        action: {"count": 0.0, "total_bb": 0.0} for action in RAISED_MEDIUM_TINY_TRACKED_ACTIONS
    }
    raised_or_3bp_medium_tiny_other_action_stats: Dict[str, Dict[str, float]] = {}
    raised_or_3bp_medium_tiny_total_events = 0
    raised_or_3bp_medium_tiny_minbet_prevented_count = 0
    raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb = 0.0
    river_cfr_calls_total = 0
    river_cfr_skipped_no_range_total = 0
    river_cfr_parse_fail_total = 0
    river_cfr_runtime_ms_sum_total = 0.0
    river_cfr_teacher_samples: List[Dict[str, Any]] = []
    river_cfr_teacher_samples_dropped = 0
    river_cfr_teacher_outputs: List[Dict[str, Any]] = []
    river_cfr_teacher_summary: Dict[str, Any] = {}
    is_hu_run = int(args.players) == 2
    hu_unopened_action_counts: Dict[str, int] = {"raise": 0, "limp": 0, "fold": 0}
    open_raise_fold_to_3bet_count = 0
    open_raise_fold_to_3bet_loss_proxy_total_bb = 0.0
    limp_iso_outcome_counts: Dict[str, int] = {"villain_checked": 0, "villain_raised": 0, "other": 0}
    limp_iso_response_counts: Dict[str, int] = {"fold": 0, "call": 0, "raise": 0}
    debug_trace_trigger_mode = "off"
    if env_flag("WIPOKER_DEBUG_TRACE_NONPREMIUM_FACING3BET_RAISE", False):
        debug_trace_trigger_mode = "nonpremium_facing_3bet_raise"
    if env_flag("WIPOKER_DEBUG_TRACE_ANY_FACING3BET_RAISE", False):
        debug_trace_trigger_mode = (
            "nonpremium_or_any_facing_3bet_raise"
            if debug_trace_trigger_mode != "off"
            else "any_facing_3bet_raise"
        )
    debug_facing3bet_raise_trace: Optional[Dict[str, Any]] = None
    strategy_store_diagnostics_latest: Dict[str, Any] = {}
    worst_hands: List[Dict[str, Any]] = []
    river_cfr_config = {
        "enabled": bool(RIVER_CFR_ENABLED),
        "override_enabled": bool(RIVER_CFR_ENABLED and RIVER_CFR_OVERRIDE_ENABLED),
        "override_iters": int(RIVER_CFR_ITERS),
        "teacher_enabled": bool(RIVER_CFR_ENABLED),
        "teacher_eval_in_run": bool(RIVER_CFR_ENABLED and RIVER_CFR_TEACHER_EVAL_IN_RUN),
        "teacher_iters": int(RIVER_CFR_TEACHER_ITERS),
        "teacher_max_samples": int(RIVER_CFR_TEACHER_MAX_SAMPLES),
        "max_calls_per_match": int(RIVER_CFR_MAX_CALLS_PER_MATCH),
        "debug": bool(RIVER_CFR_DEBUG_ENABLED),
    }
    river_cfr_state = {"calls": 0}

    args.results.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_hands > 0:
        args.trace_file.parent.mkdir(parents=True, exist_ok=True)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
    if args.dump_biggest_losers > 0:
        args.biggest_losers_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        trace_ctx = (
            args.trace_file.open("w", encoding="utf-8")
            if args.trace_hands > 0
            else contextlib.nullcontext(None)
        )
        with args.results.open("w", newline="", encoding="utf-8") as csv_file, trace_ctx as trace_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "hand_index",
                    "bb_delta",
                    "cumulative_bb",
                    "bb_delta_raw",
                    "cumulative_bb_raw",
                    "rake_bb",
                ]
            )

            for hand_index in range(1, args.hands + 1):
                bb_delta_raw, final_pot, hand_debug, trace_events, hand_metrics, hand_detail = run_hand(
                    hand_index=hand_index,
                    table=table,
                    hero_id=args.hero_id,
                    worker=worker,
                    rng=rng,
                    bots=bots,
                    hero_mode=args.hero_mode,
                    status_counts=status_counts,
                    max_actions_per_hand=args.max_actions_per_hand,
                    trace_enabled=hand_index <= args.trace_hands,
                    open_size_bb=float(args.open_size),
                    threebet_size_bb=float(args.threebet_size),
                    fourbet_size_bb=float(args.fourbet_size),
                    river_cfr_config=river_cfr_config,
                    river_cfr_state=river_cfr_state,
                )
                fallback_count += int(hand_debug.get("fallback_calls", 0))
                for street in ALL_STREETS:
                    street_decisions[street] += int(hand_metrics["street_decisions"][street])
                    street_ok[street] += int(hand_metrics["street_ok"][street])
                    street_unavailable[street] += int(hand_metrics["street_unavailable"][street])
                    street_fallbacks[street] += int(hand_metrics["street_fallbacks"][street])
                river_cfr_calls_total += int(hand_metrics.get("river_cfr_calls", 0))
                river_cfr_skipped_no_range_total += int(hand_metrics.get("river_cfr_skipped_no_range", 0))
                river_cfr_parse_fail_total += int(hand_metrics.get("river_cfr_parse_fail", 0))
                river_cfr_runtime_ms_sum_total += float(hand_metrics.get("river_cfr_runtime_ms_sum", 0.0))
                for sample in hand_metrics.get("river_cfr_teacher_samples", []):
                    if len(river_cfr_teacher_samples) < int(river_cfr_config.get("teacher_max_samples", 0)):
                        river_cfr_teacher_samples.append(sample)
                    else:
                        river_cfr_teacher_samples_dropped += 1

                ev = hand_metrics["ev_decomposition"]
                for key in ev_decomposition_totals:
                    ev_decomposition_totals[key] += float(ev[key])

                if hand_metrics["hero_vpip"]:
                    vpip_hands += 1
                if hand_metrics["hero_pfr"]:
                    pfr_hands += 1
                if hand_metrics["hero_threebet_opp"]:
                    threebet_opportunities += 1
                if hand_metrics["hero_threebet"]:
                    threebet_hands += 1
                if hand_metrics["hero_vpip"] and hand_metrics["showdown"]:
                    showdown_when_vpip_hands += 1

                if debug_facing3bet_raise_trace is None:
                    trace_candidate = hand_metrics.get("debug_facing3bet_raise_trace")
                    if isinstance(trace_candidate, dict):
                        debug_facing3bet_raise_trace = trace_candidate

                store_diag = hand_metrics.get("strategy_store_diagnostics")
                if isinstance(store_diag, dict):
                    latest_attempts = int(strategy_store_diagnostics_latest.get("store_hit_count", 0)) + int(
                        strategy_store_diagnostics_latest.get("store_miss_count", 0)
                    )
                    candidate_attempts = int(store_diag.get("store_hit_count", 0)) + int(
                        store_diag.get("store_miss_count", 0)
                    )
                    if candidate_attempts >= latest_attempts:
                        strategy_store_diagnostics_latest = dict(store_diag)

                for hero_decision in hand_detail.get("hero_decisions", []):
                    if not isinstance(hero_decision, dict):
                        continue
                    decision_street = str(hero_decision.get("street", "")).strip().lower()
                    if decision_street not in ALL_STREETS:
                        continue
                    source = str(
                        hero_decision.get("selection_source", "fallback")
                    ).strip() or "fallback"
                    exact_hit_policy = bool(hero_decision.get("exact_hit_policy", False))
                    selection_source_counts[source] = int(selection_source_counts.get(source, 0)) + 1
                    per_street = selection_source_by_street[decision_street]
                    per_street[source] = int(per_street.get(source, 0)) + 1
                    split_target = (
                        selection_source_by_street_exact_hit
                        if exact_hit_policy
                        else selection_source_by_street_non_exact_hit
                    )
                    split_rows = split_target[decision_street]
                    split_rows[source] = int(split_rows.get(source, 0)) + 1

                    if hero_decision.get("worker_executed_action") is not None:
                        chosen_vs_executed_total += 1
                        if bool(hero_decision.get("chosen_matches_worker_executed", False)):
                            chosen_vs_executed_match_count += 1
                    if hero_decision.get("worker_recommended_action") is not None:
                        chosen_vs_recommended_total += 1
                        if bool(hero_decision.get("chosen_matches_worker_recommended", False)):
                            chosen_vs_recommended_match_count += 1
                    if hero_decision.get("worker_argmax_action") is not None:
                        chosen_vs_argmax_total += 1
                        if bool(hero_decision.get("chosen_matches_worker_argmax", False)):
                            chosen_vs_argmax_match_count += 1

                if is_hu_run:
                    hu_unopened_action = str(hand_metrics.get("hu_unopened_first_action") or "")
                    if hu_unopened_action in hu_unopened_action_counts:
                        hu_unopened_action_counts[hu_unopened_action] += 1

                    open_raise_fold_to_3bet = hand_metrics.get("open_raise_then_fold_to_3bet", {})
                    if isinstance(open_raise_fold_to_3bet, dict):
                        open_raise_fold_count = int(open_raise_fold_to_3bet.get("count", 0))
                        if open_raise_fold_count > 0:
                            open_raise_fold_to_3bet_count += open_raise_fold_count
                            open_raise_fold_to_3bet_loss_proxy_total_bb += float(
                                open_raise_fold_to_3bet.get("loss_proxy_bb", 0.0)
                            )

                    limp_iso_outcome = str(hand_metrics.get("limp_iso_outcome") or "")
                    if limp_iso_outcome in limp_iso_outcome_counts:
                        limp_iso_outcome_counts[limp_iso_outcome] += 1
                        if limp_iso_outcome == "villain_raised":
                            limp_iso_response = str(hand_metrics.get("limp_iso_response") or "")
                            if limp_iso_response in limp_iso_response_counts:
                                limp_iso_response_counts[limp_iso_response] += 1

                for node_entry in hand_metrics["preflop_nodes"]:
                    node = str(node_entry["node"])
                    action_bucket = str(node_entry.get("action", "call"))
                    note = str(node_entry.get("note", ""))
                    if node not in preflop_node_stats:
                        preflop_node_stats[node] = {
                            "count": 0,
                            "total_bb": 0.0,
                            "action_counts": {"fold": 0, "call": 0, "raise": 0},
                            "notes": {},
                        }
                    node_stat = preflop_node_stats[node]
                    node_stat["count"] += 1
                    node_stat["total_bb"] += float(bb_delta_raw)
                    if action_bucket not in node_stat["action_counts"]:
                        action_bucket = "call"
                    node_stat["action_counts"][action_bucket] += 1
                    if note:
                        node_notes = node_stat["notes"]
                        node_notes[note] = int(node_notes.get(note, 0)) + 1

                for node_event in hand_metrics.get("preflop_node_events", []):
                    node = str(node_event.get("node", "UNKNOWN"))
                    action_bucket = str(node_event.get("action", "call"))
                    villain_response = str(node_event.get("villain_response", "unknown"))
                    outcome_class = str(node_event.get("final_outcome_class", "unknown"))
                    to_call_bucket = str(node_event.get("to_call_bucket", "unknown"))
                    raise_size_bucket = str(node_event.get("raise_size_bucket", "unknown"))
                    hand_bucket = str(node_event.get("hand_bucket", "unknown"))
                    immediate_proxy = float(node_event.get("immediate_delta_proxy_bb", 0.0))

                    if node not in preflop_node_context_stats:
                        preflop_node_context_stats[node] = {
                            "count": 0,
                            "total_bb": 0.0,
                            "immediate_proxy_total_bb": 0.0,
                            "action_counts": {},
                            "villain_response_counts": {},
                            "outcome_class_counts": {},
                            "to_call_bucket_stats": {},
                            "raise_size_bucket_stats": {},
                            "hand_bucket_stats": {},
                        }
                    node_stat = preflop_node_context_stats[node]
                    node_stat["count"] += 1
                    node_stat["total_bb"] += float(bb_delta_raw)
                    node_stat["immediate_proxy_total_bb"] += immediate_proxy
                    node_stat["action_counts"][action_bucket] = int(
                        node_stat["action_counts"].get(action_bucket, 0)
                    ) + 1
                    node_stat["villain_response_counts"][villain_response] = int(
                        node_stat["villain_response_counts"].get(villain_response, 0)
                    ) + 1
                    node_stat["outcome_class_counts"][outcome_class] = int(
                        node_stat["outcome_class_counts"].get(outcome_class, 0)
                    ) + 1

                    to_call_stats = node_stat["to_call_bucket_stats"]
                    if to_call_bucket not in to_call_stats:
                        to_call_stats[to_call_bucket] = {"count": 0, "total_bb": 0.0}
                    to_call_stats[to_call_bucket]["count"] += 1
                    to_call_stats[to_call_bucket]["total_bb"] += float(bb_delta_raw)

                    if raise_size_bucket != "unknown":
                        raise_stats = node_stat["raise_size_bucket_stats"]
                        if raise_size_bucket not in raise_stats:
                            raise_stats[raise_size_bucket] = {"count": 0, "total_bb": 0.0}
                        raise_stats[raise_size_bucket]["count"] += 1
                        raise_stats[raise_size_bucket]["total_bb"] += float(bb_delta_raw)

                    hand_bucket_stats = node_stat["hand_bucket_stats"]
                    if hand_bucket not in hand_bucket_stats:
                        hand_bucket_stats[hand_bucket] = {"count": 0, "total_bb": 0.0}
                    hand_bucket_stats[hand_bucket]["count"] += 1
                    hand_bucket_stats[hand_bucket]["total_bb"] += float(bb_delta_raw)

                for flop_event in hand_metrics.get("flop_no_bet_events", []):
                    action = str(flop_event.get("action", "unknown"))
                    pot_bucket = str(flop_event.get("pot_bucket", "unknown"))
                    preflop_context = str(flop_event.get("preflop_context", "unknown"))
                    raised_or_3bp_medium_tiny_pot = bool(
                        flop_event.get("raised_or_3bp_medium_tiny_pot", False)
                    )
                    raised_or_3bp_medium_tiny_minbet_prevented = bool(
                        flop_event.get("raised_or_3bp_medium_tiny_minbet_prevented", False)
                    )
                    potaware_counter_tags_raw = flop_event.get("potaware_counter_tags", [])
                    potaware_counter_tags: List[str] = []
                    if isinstance(potaware_counter_tags_raw, list):
                        for value in potaware_counter_tags_raw:
                            if isinstance(value, str) and value:
                                potaware_counter_tags.append(value)
                    flop_no_bet_total_events += 1
                    flop_no_bet_pot_totals[pot_bucket] = int(flop_no_bet_pot_totals.get(pot_bucket, 0)) + 1
                    flop_no_bet_context_totals[preflop_context] = int(
                        flop_no_bet_context_totals.get(preflop_context, 0)
                    ) + 1
                    for counter_tag in set(potaware_counter_tags):
                        if counter_tag in flop_potaware_medium_counters:
                            flop_potaware_medium_counters[counter_tag] += 1

                    if action not in flop_no_bet_action_stats:
                        flop_no_bet_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    flop_no_bet_action_stats[action]["count"] += 1.0
                    flop_no_bet_action_stats[action]["total_bb"] += float(bb_delta_raw)

                    if pot_bucket not in flop_no_bet_action_pot_stats:
                        flop_no_bet_action_pot_stats[pot_bucket] = {}
                    pot_action_stats = flop_no_bet_action_pot_stats[pot_bucket]
                    if action not in pot_action_stats:
                        pot_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    pot_action_stats[action]["count"] += 1.0
                    pot_action_stats[action]["total_bb"] += float(bb_delta_raw)

                    if preflop_context not in flop_no_bet_action_context_stats:
                        flop_no_bet_action_context_stats[preflop_context] = {}
                    context_action_stats = flop_no_bet_action_context_stats[preflop_context]
                    if action not in context_action_stats:
                        context_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    context_action_stats[action]["count"] += 1.0
                    context_action_stats[action]["total_bb"] += float(bb_delta_raw)

                    if raised_or_3bp_medium_tiny_pot:
                        raised_or_3bp_medium_tiny_total_events += 1
                        target_stats = (
                            raised_or_3bp_medium_tiny_action_stats
                            if action in raised_or_3bp_medium_tiny_action_stats
                            else raised_or_3bp_medium_tiny_other_action_stats
                        )
                        if action not in target_stats:
                            target_stats[action] = {"count": 0.0, "total_bb": 0.0}
                        target_stats[action]["count"] += 1.0
                        target_stats[action]["total_bb"] += float(bb_delta_raw)

                        if raised_or_3bp_medium_tiny_minbet_prevented:
                            raised_or_3bp_medium_tiny_minbet_prevented_count += 1
                            raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb += float(bb_delta_raw)

                for turn_event in hand_metrics.get("turn_no_bet_events", []):
                    action = str(turn_event.get("action", "unknown"))
                    pot_bucket = str(turn_event.get("pot_bucket", "unknown"))
                    preflop_context = str(turn_event.get("preflop_context", "unknown"))
                    turn_no_bet_total_events += 1
                    turn_no_bet_pot_totals[pot_bucket] = int(turn_no_bet_pot_totals.get(pot_bucket, 0)) + 1
                    turn_no_bet_context_totals[preflop_context] = int(
                        turn_no_bet_context_totals.get(preflop_context, 0)
                    ) + 1

                    if action not in turn_no_bet_action_stats:
                        turn_no_bet_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    turn_no_bet_action_stats[action]["count"] += 1.0
                    turn_no_bet_action_stats[action]["total_bb"] += float(bb_delta_raw)

                    if pot_bucket not in turn_no_bet_action_pot_stats:
                        turn_no_bet_action_pot_stats[pot_bucket] = {}
                    pot_action_stats = turn_no_bet_action_pot_stats[pot_bucket]
                    if action not in pot_action_stats:
                        pot_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    pot_action_stats[action]["count"] += 1.0
                    pot_action_stats[action]["total_bb"] += float(bb_delta_raw)

                    if preflop_context not in turn_no_bet_action_context_stats:
                        turn_no_bet_action_context_stats[preflop_context] = {}
                    context_action_stats = turn_no_bet_action_context_stats[preflop_context]
                    if action not in context_action_stats:
                        context_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    context_action_stats[action]["count"] += 1.0
                    context_action_stats[action]["total_bb"] += float(bb_delta_raw)

                for river_event in hand_metrics.get("river_no_bet_events", []):
                    action = str(river_event.get("action", "unknown"))
                    pot_bucket = str(river_event.get("pot_bucket", "unknown"))
                    preflop_context = str(river_event.get("preflop_context", "unknown"))
                    river_no_bet_total_events += 1
                    river_no_bet_pot_totals[pot_bucket] = int(river_no_bet_pot_totals.get(pot_bucket, 0)) + 1
                    river_no_bet_context_totals[preflop_context] = int(
                        river_no_bet_context_totals.get(preflop_context, 0)
                    ) + 1

                    if action not in river_no_bet_action_stats:
                        river_no_bet_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    river_no_bet_action_stats[action]["count"] += 1.0
                    river_no_bet_action_stats[action]["total_bb"] += float(bb_delta_raw)

                    if pot_bucket not in river_no_bet_action_pot_stats:
                        river_no_bet_action_pot_stats[pot_bucket] = {}
                    pot_action_stats = river_no_bet_action_pot_stats[pot_bucket]
                    if action not in pot_action_stats:
                        pot_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    pot_action_stats[action]["count"] += 1.0
                    pot_action_stats[action]["total_bb"] += float(bb_delta_raw)

                    if preflop_context not in river_no_bet_action_context_stats:
                        river_no_bet_action_context_stats[preflop_context] = {}
                    context_action_stats = river_no_bet_action_context_stats[preflop_context]
                    if action not in context_action_stats:
                        context_action_stats[action] = {"count": 0.0, "total_bb": 0.0}
                    context_action_stats[action]["count"] += 1.0
                    context_action_stats[action]["total_bb"] += float(bb_delta_raw)

                for street in POSTFLOP_STREETS:
                    postflop_aggr[street] += int(hand_metrics["postflop_aggr"][street])
                    postflop_calls[street] += int(hand_metrics["postflop_calls"][street])
                    fold_to_bet_opportunities[street] += int(
                        hand_metrics["fold_to_bet_opportunities"][street]
                    )
                    fold_to_bet_folds[street] += int(hand_metrics["fold_to_bet_folds"][street])

                for event in hand_metrics["translation_events"]:
                    pot_bb = float(event["pot_bb"])
                    sent_bb = event["translated_bet_amount_bb"]
                    chosen_intent = str(event["chosen_intent"])
                    street = str(event["street"])
                    facing_action = str(event.get("facing_action") or "UNKNOWN")
                    stack_bb = float(event.get("stack_bb", 0.0))
                    clipped = bool(event["clipped_to_bounds"])
                    translation_adjusted = bool(event["translation_adjusted"])
                    env_legalized = bool(event["env_legalized"])
                    preflop_raise_clamped = bool(event.get("preflop_raise_clamped"))
                    preflop_raise_converted_to_allin = bool(
                        event.get("preflop_raise_converted_to_allin")
                    )
                    postflop_raise_clamped = bool(event.get("postflop_raise_clamped"))
                    exact_requested_size_used = bool(event.get("exact_requested_size_used"))
                    postflop_raise_guardrail_relaxed = bool(
                        event.get("postflop_raise_guardrail_relaxed")
                    )
                    exact_hit_policy = bool(event.get("exact_hit_policy", False))
                    illegal_action_avoided = bool(event.get("illegal_action_avoided"))
                    is_allin = bool(event.get("is_allin"))
                    preflop_node = event.get("preflop_node")
                    was_capped_in_engine = bool(event.get("was_capped_in_engine"))

                    if clipped:
                        translation_clipped_count += 1
                    if translation_adjusted:
                        translation_adjusted_count += 1
                    if env_legalized:
                        env_legalized_count += 1
                    if illegal_action_avoided:
                        illegal_action_avoided_count += 1
                    if preflop_raise_clamped:
                        preflop_raise_clamped_count += 1
                    if preflop_raise_converted_to_allin:
                        preflop_raise_converted_to_allin_count += 1
                    if postflop_raise_clamped:
                        postflop_raise_clamped_count += 1
                    if exact_requested_size_used:
                        exact_requested_size_used_count += 1
                    if postflop_raise_guardrail_relaxed:
                        postflop_raise_guardrail_relaxed_count += 1
                    if was_capped_in_engine:
                        engine_size_capped_count += 1
                        capped_key = f"{street}:{chosen_intent}"
                        engine_size_capped_by_street_intent[capped_key] = (
                            int(engine_size_capped_by_street_intent.get(capped_key, 0)) + 1
                        )
                    if clipped or translation_adjusted or env_legalized:
                        clipped_or_adjusted_total += 1
                        context_key = f"{street}:{facing_action}"
                        if context_key not in clipped_adjusted_by_context:
                            clipped_adjusted_by_context[context_key] = {
                                "count": 0,
                                "clipped": 0,
                                "translation_adjusted": 0,
                                "env_legalized": 0,
                            }
                        context_stat = clipped_adjusted_by_context[context_key]
                        context_stat["count"] += 1
                        if clipped:
                            context_stat["clipped"] += 1
                        if translation_adjusted:
                            context_stat["translation_adjusted"] += 1
                        if env_legalized:
                            context_stat["env_legalized"] += 1

                    if exact_hit_policy:
                        modified = (
                            clipped
                            or translation_adjusted
                            or env_legalized
                            or preflop_raise_clamped
                            or postflop_raise_clamped
                        )
                        if modified:
                            exact_hit_and_translated_count += 1
                            exact_hit_translation_by_street[street]["translated"] += 1
                        else:
                            exact_hit_and_unmodified_count += 1
                            exact_hit_translation_by_street[street]["unmodified"] += 1
                        exact_hit_translation_by_street[street]["total"] += 1

                    if sent_bb is not None and pot_bb > 0:
                        frac = float(sent_bb) / pot_bb
                        if chosen_intent == "bet33":
                            bet33_fractions.append(frac)
                            update_fraction_buckets(bet33_fraction_buckets, frac)
                            if abs(frac - 0.5) <= 0.06:
                                bet33_forced_halfpot_count += 1
                        elif chosen_intent == "bet50":
                            bet50_fractions.append(frac)
                            update_fraction_buckets(bet50_fraction_buckets, frac)
                            if abs(frac - 0.5) <= 0.06:
                                bet50_forced_halfpot_count += 1
                        elif chosen_intent == "bet75":
                            bet75_fractions.append(frac)
                            update_fraction_buckets(bet75_fraction_buckets, frac)
                            if abs(frac - 0.5) <= 0.06:
                                bet75_forced_halfpot_count += 1
                        elif chosen_intent == "raise":
                            raise_fractions.append(frac)
                    if sent_bb is not None and event.get("action_is_raise"):
                        raise_amount = float(sent_bb)
                        raise_sizes_bb.append(raise_amount)
                        if (
                            street == "preflop"
                            and preflop_node == "OPEN"
                            and not is_allin
                            and raise_amount > 15
                        ):
                            preflop_open_non_allin_raises_over_15bb.append(
                                {
                                    "hand_index": hand_index,
                                    "street": street,
                                    "facing_action": facing_action,
                                    "raise_size_bb": round(raise_amount, 6),
                                    "stack_bb": round(stack_bb, 6),
                                    "preflop_node": preflop_node,
                                    "chosen_intent": chosen_intent,
                                }
                            )
                        if raise_amount > 20:
                            raise_over_20 += 1
                        if raise_amount > 40:
                            raise_over_40 += 1
                            if street == "preflop" and not is_allin:
                                large_preflop_non_allin_raises.append(
                                    {
                                        "hand_index": hand_index,
                                        "street": street,
                                        "facing_action": facing_action,
                                        "raise_size_bb": round(raise_amount, 6),
                                        "stack_bb": round(stack_bb, 6),
                                        "preflop_node": preflop_node,
                                        "chosen_intent": chosen_intent,
                                    }
                                )
                        if raise_amount > 80:
                            raise_over_80 += 1
                        largest_raises.append(
                            {
                                "hand_index": hand_index,
                                "street": street,
                                "facing_action": facing_action,
                                "pot_bb": round(pot_bb, 6),
                                "stack_bb": round(stack_bb, 6),
                                "raise_size_bb": round(raise_amount, 6),
                                "is_allin": is_allin,
                                "preflop_node": preflop_node,
                                "chosen_intent": chosen_intent,
                            }
                        )

                if args.dump_biggest_losers > 0:
                    worst_hands.append(hand_detail)
                    worst_hands.sort(key=lambda item: (float(item["bb_delta"]), int(item["hand_index"])))
                    if len(worst_hands) > args.dump_biggest_losers:
                        worst_hands = worst_hands[: args.dump_biggest_losers]

                rake_bb = compute_rake(
                    rake_mode=args.rake_mode,
                    pot_size_bb=final_pot,
                    rake_fixed_bb=args.rake_fixed_bb,
                    rake_percent=args.rake_percent,
                    rake_cap_bb=args.rake_cap_bb,
                )

                bb_delta_net = bb_delta_raw - rake_bb
                raw_deltas.append(bb_delta_raw)
                net_deltas.append(bb_delta_net)

                cumulative_raw += bb_delta_raw
                cumulative_net += bb_delta_net

                writer.writerow(
                    [
                        hand_index,
                        f"{bb_delta_net:.6f}",
                        f"{cumulative_net:.6f}",
                        f"{bb_delta_raw:.6f}",
                        f"{cumulative_raw:.6f}",
                        f"{rake_bb:.6f}",
                    ]
                )
                if hand_index <= args.trace_hands and trace_file is not None:
                    for event in trace_events:
                        trace_file.write(json.dumps(event, separators=(",", ":")) + "\n")

                if args.progress_every > 0 and hand_index % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    speed = hand_index / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{hand_index:>7}/{args.hands}] "
                        f"raw bb/100={np.mean(raw_deltas) * 100:.2f} "
                        f"net bb/100={np.mean(net_deltas) * 100:.2f} "
                        f"({speed:.1f} hands/s)"
                    )

    finally:
        worker.close()

    raw_stats = summarize(raw_deltas)
    net_stats = summarize(net_deltas)
    preflop_leak_rows: List[Dict[str, Any]] = []
    for node, stat in preflop_node_stats.items():
        count = int(stat["count"])
        total_bb = float(stat["total_bb"])
        avg_bb = total_bb / count if count > 0 else 0.0
        action_counts = stat["action_counts"]
        notes_obj = stat["notes"]
        notes_sorted = sorted(notes_obj.items(), key=lambda item: (-int(item[1]), item[0]))
        notes_text = ", ".join([f"{name}({value})" for name, value in notes_sorted[:3]])
        preflop_leak_rows.append(
            {
                "node": node,
                "count": count,
                "total_bb": total_bb,
                "avg_bb": avg_bb,
                "fold_pct": pct(action_counts["fold"], count),
                "call_pct": pct(action_counts["call"], count),
                "raise_pct": pct(action_counts["raise"], count),
                "notes": notes_text,
            }
        )

    preflop_leak_rows.sort(key=lambda row: row["node"])
    worst_nodes_by_total_bb = sorted(preflop_leak_rows, key=lambda row: row["total_bb"])[:5]
    worst_nodes_by_avg_bb = sorted(preflop_leak_rows, key=lambda row: row["avg_bb"])[:5]

    def build_count_rows(
        counts: Dict[str, int],
        total: int,
        key_name: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, value in counts.items():
            count_value = int(value)
            rows.append(
                {
                    key_name: key,
                    "count": count_value,
                    "pct": pct(count_value, total),
                }
            )
        rows.sort(key=lambda row: (-int(row["count"]), str(row[key_name])))
        return rows

    def build_bucket_ev_rows(stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for bucket, item in stats.items():
            count_value = int(item.get("count", 0))
            total_bb_value = float(item.get("total_bb", 0.0))
            rows.append(
                {
                    "bucket": bucket,
                    "count": count_value,
                    "total_bb": total_bb_value,
                    "avg_bb": (total_bb_value / count_value) if count_value > 0 else 0.0,
                }
            )
        rows.sort(key=lambda row: (-int(row["count"]), str(row["bucket"])))
        return rows

    preflop_node_context_rows: List[Dict[str, Any]] = []
    for node, stat in preflop_node_context_stats.items():
        count = int(stat.get("count", 0))
        total_bb = float(stat.get("total_bb", 0.0))
        immediate_proxy_total_bb = float(stat.get("immediate_proxy_total_bb", 0.0))
        preflop_node_context_rows.append(
            {
                "node": node,
                "attribution_method": "full_hand_with_outcome_class",
                "count": count,
                "total_bb": total_bb,
                "avg_bb": (total_bb / count) if count > 0 else 0.0,
                "immediate_delta_proxy_total_bb": immediate_proxy_total_bb,
                "immediate_delta_proxy_avg_bb": (immediate_proxy_total_bb / count) if count > 0 else 0.0,
                "action_counts": build_count_rows(
                    stat.get("action_counts", {}),
                    count,
                    "action",
                ),
                "villain_response_counts": build_count_rows(
                    stat.get("villain_response_counts", {}),
                    count,
                    "villain_response",
                ),
                "outcome_class_counts": build_count_rows(
                    stat.get("outcome_class_counts", {}),
                    count,
                    "outcome_class",
                ),
                "to_call_bb_buckets": build_bucket_ev_rows(stat.get("to_call_bucket_stats", {})),
                "raise_size_bb_buckets": build_bucket_ev_rows(stat.get("raise_size_bucket_stats", {})),
                "hand_bucket_stats": build_bucket_ev_rows(stat.get("hand_bucket_stats", {})),
            }
        )
    preflop_node_context_rows.sort(key=lambda row: str(row["node"]))

    focus_node_names = ("HERO_OPENED", "HERO_FACED_3BET_AFTER_OPEN")
    focus_preflop_nodes: Dict[str, Any] = {}
    for node_name in focus_node_names:
        node_row = next(
            (row for row in preflop_node_context_rows if str(row.get("node")) == node_name),
            None,
        )
        if node_row is not None:
            focus_preflop_nodes[node_name] = node_row

    (
        flop_no_bet_action_rows,
        flop_no_bet_pot_bucket_rows,
        flop_no_bet_preflop_context_rows,
    ) = build_no_bet_tracker_rows(
        action_stats=flop_no_bet_action_stats,
        pot_totals=flop_no_bet_pot_totals,
        action_pot_stats=flop_no_bet_action_pot_stats,
        context_totals=flop_no_bet_context_totals,
        action_context_stats=flop_no_bet_action_context_stats,
        total_events=flop_no_bet_total_events,
    )
    (
        turn_no_bet_action_rows,
        turn_no_bet_pot_bucket_rows,
        turn_no_bet_preflop_context_rows,
    ) = build_no_bet_tracker_rows(
        action_stats=turn_no_bet_action_stats,
        pot_totals=turn_no_bet_pot_totals,
        action_pot_stats=turn_no_bet_action_pot_stats,
        context_totals=turn_no_bet_context_totals,
        action_context_stats=turn_no_bet_action_context_stats,
        total_events=turn_no_bet_total_events,
    )
    (
        river_no_bet_action_rows,
        river_no_bet_pot_bucket_rows,
        river_no_bet_preflop_context_rows,
    ) = build_no_bet_tracker_rows(
        action_stats=river_no_bet_action_stats,
        pot_totals=river_no_bet_pot_totals,
        action_pot_stats=river_no_bet_action_pot_stats,
        context_totals=river_no_bet_context_totals,
        action_context_stats=river_no_bet_action_context_stats,
        total_events=river_no_bet_total_events,
    )

    raised_or_3bp_medium_tiny_action_rows: List[Dict[str, Any]] = []
    merged_raised_or_3bp_medium_tiny_action_stats: Dict[str, Dict[str, float]] = {}
    for source in (raised_or_3bp_medium_tiny_action_stats, raised_or_3bp_medium_tiny_other_action_stats):
        for action_name, stats in source.items():
            if action_name not in merged_raised_or_3bp_medium_tiny_action_stats:
                merged_raised_or_3bp_medium_tiny_action_stats[action_name] = {"count": 0.0, "total_bb": 0.0}
            merged_raised_or_3bp_medium_tiny_action_stats[action_name]["count"] += float(
                stats.get("count", 0.0)
            )
            merged_raised_or_3bp_medium_tiny_action_stats[action_name]["total_bb"] += float(
                stats.get("total_bb", 0.0)
            )
    for action_name, stats in sorted(
        merged_raised_or_3bp_medium_tiny_action_stats.items(),
        key=lambda item: (-int(item[1].get("count", 0.0)), item[0]),
    ):
        count_value = int(stats.get("count", 0.0))
        total_bb_value = float(stats.get("total_bb", 0.0))
        raised_or_3bp_medium_tiny_action_rows.append(
            {
                "action": action_name,
                "count": count_value,
                "freq_pct": pct(count_value, raised_or_3bp_medium_tiny_total_events),
                "total_bb": total_bb_value,
                "avg_bb": (total_bb_value / count_value) if count_value > 0 else 0.0,
            }
        )

    largest_raises.sort(key=lambda item: item["raise_size_bb"], reverse=True)
    top_10_biggest_raises = largest_raises[:10]
    clipped_adjusted_context_rows = sorted(
        [
            {
                "context": context_key,
                "count": values["count"],
                "clipped": values["clipped"],
                "translation_adjusted": values["translation_adjusted"],
                "env_legalized": values["env_legalized"],
            }
            for context_key, values in clipped_adjusted_by_context.items()
        ],
        key=lambda row: row["count"],
        reverse=True,
    )
    engine_size_capped_rows = sorted(
        [
            {
                "street": context_key.split(":", 1)[0],
                "intent": context_key.split(":", 1)[1] if ":" in context_key else "unknown",
                "count": int(count),
            }
            for context_key, count in engine_size_capped_by_street_intent.items()
        ],
        key=lambda row: row["count"],
        reverse=True,
    )
    raise_count_total = len(raise_sizes_bb)
    effective_small_bet_fraction = effective_bet_fraction(bet33_fractions, 0.33)
    effective_large_bet_fraction = effective_bet_fraction(bet75_fractions, 0.75)
    hu_unopened_total = sum(int(value) for value in hu_unopened_action_counts.values())
    hu_unopened_action_rows = build_count_rows(
        hu_unopened_action_counts,
        hu_unopened_total,
        "action",
    )
    limp_iso_total = sum(int(value) for value in limp_iso_outcome_counts.values())
    limp_iso_outcome_rows = build_count_rows(
        limp_iso_outcome_counts,
        limp_iso_total,
        "outcome",
    )
    limp_iso_raised_total = int(limp_iso_outcome_counts.get("villain_raised", 0))
    limp_iso_response_rows = build_count_rows(
        limp_iso_response_counts,
        limp_iso_raised_total,
        "action",
    )
    open_raise_fold_to_3bet_avg_loss_proxy = (
        open_raise_fold_to_3bet_loss_proxy_total_bb / open_raise_fold_to_3bet_count
        if open_raise_fold_to_3bet_count > 0
        else 0.0
    )

    ordered_bucket_keys = ["<0.25", "0.25-0.40", "0.40-0.60", "0.60-0.90", ">=0.90", "invalid"]
    bet33_bucket_summary = {
        key: int(bet33_fraction_buckets.get(key, 0))
        for key in ordered_bucket_keys
        if key in bet33_fraction_buckets
    }
    bet50_bucket_summary = {
        key: int(bet50_fraction_buckets.get(key, 0))
        for key in ordered_bucket_keys
        if key in bet50_fraction_buckets
    }
    bet75_bucket_summary = {
        key: int(bet75_fraction_buckets.get(key, 0))
        for key in ordered_bucket_keys
        if key in bet75_fraction_buckets
    }

    if args.assert_large_preflop_raises_allin and large_preflop_non_allin_raises:
        examples = large_preflop_non_allin_raises[:5]
        raise RuntimeError(
            "assertion failed: found preflop raises >40bb that were not all-in; "
            f"count={len(large_preflop_non_allin_raises)} examples={examples}"
        )
    if args.assert_preflop_open_max15 and preflop_open_non_allin_raises_over_15bb:
        examples = preflop_open_non_allin_raises_over_15bb[:5]
        raise RuntimeError(
            "assertion failed: found preflop OPEN raises >15bb that were not all-in; "
            f"count={len(preflop_open_non_allin_raises_over_15bb)} examples={examples}"
        )

    if (
        bool(river_cfr_config.get("teacher_enabled"))
        and bool(river_cfr_config.get("teacher_eval_in_run"))
        and river_cfr_teacher_samples
    ):
        teacher_started = time.perf_counter()
        teacher_result = evaluate_teacher_samples(
            samples=river_cfr_teacher_samples,
            iters=int(river_cfr_config.get("teacher_iters", 200)),
            max_samples=int(river_cfr_config.get("teacher_max_samples", 200)),
        )
        teacher_elapsed_ms = (time.perf_counter() - teacher_started) * 1000.0
        river_cfr_teacher_outputs = list(teacher_result.get("rows", []))
        river_cfr_teacher_summary = {
            "enabled": True,
            "eval_in_run_enabled": True,
            "assumption": str(teacher_result.get("assumption", "neutral_range_assumption")),
            "iters": int(river_cfr_config.get("teacher_iters", 200)),
            "max_samples": int(river_cfr_config.get("teacher_max_samples", 200)),
            "captured_samples": len(river_cfr_teacher_samples),
            "dropped_samples": int(river_cfr_teacher_samples_dropped),
            "processed_samples": int(teacher_result.get("processed", len(river_cfr_teacher_outputs))),
            "cache_hits": int(teacher_result.get("cache_hits", 0)),
            "cache_misses": int(teacher_result.get("cache_misses", 0)),
            "solver_runtime_ms_sum": float(teacher_result.get("solver_runtime_ms_sum", 0.0)),
            "runtime_ms_total": float(teacher_elapsed_ms),
        }
    else:
        river_cfr_teacher_summary = {
            "enabled": bool(river_cfr_config.get("teacher_enabled")),
            "eval_in_run_enabled": bool(river_cfr_config.get("teacher_eval_in_run")),
            "assumption": "neutral_range_assumption",
            "iters": int(river_cfr_config.get("teacher_iters", 200)),
            "max_samples": int(river_cfr_config.get("teacher_max_samples", 200)),
            "captured_samples": len(river_cfr_teacher_samples),
            "dropped_samples": int(river_cfr_teacher_samples_dropped),
            "processed_samples": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "solver_runtime_ms_sum": 0.0,
            "runtime_ms_total": 0.0,
        }

    def sorted_count_map(values: Dict[str, int]) -> Dict[str, int]:
        return {key: int(values[key]) for key in sorted(values.keys())}

    selection_source_counts_sorted = sorted_count_map(selection_source_counts)
    selection_source_by_street_sorted = {
        street: sorted_count_map(selection_source_by_street.get(street, {}))
        for street in ALL_STREETS
    }
    selection_source_by_street_exact_hit_sorted = {
        street: sorted_count_map(selection_source_by_street_exact_hit.get(street, {}))
        for street in ALL_STREETS
    }
    selection_source_by_street_non_exact_hit_sorted = {
        street: sorted_count_map(selection_source_by_street_non_exact_hit.get(street, {}))
        for street in ALL_STREETS
    }
    exact_hit_translation_total = (
        int(exact_hit_and_translated_count) + int(exact_hit_and_unmodified_count)
    )
    exact_hit_translation_by_street_summary: Dict[str, Dict[str, Any]] = {}
    for street in ALL_STREETS:
        row = exact_hit_translation_by_street.get(
            street, {"translated": 0, "unmodified": 0, "total": 0}
        )
        total = int(row.get("total", 0))
        translated = int(row.get("translated", 0))
        unmodified = int(row.get("unmodified", 0))
        exact_hit_translation_by_street_summary[street] = {
            "translated": translated,
            "unmodified": unmodified,
            "total": total,
            "unmodified_rate": (float(unmodified) / float(total)) if total > 0 else 0.0,
        }

    diagnostics: Dict[str, Any] = {
        "street_decisions": street_decisions,
        "street_ok": street_ok,
        "street_unavailable": street_unavailable,
        "street_fallbacks": street_fallbacks,
        "evaluator_identity": {
            "backend": str(args.eval_backend),
            "policy_cmd": str(args.policy_cmd),
            "deep_cfr_model_state": str(args.deep_cfr_model_state)
            if args.deep_cfr_model_state is not None
            else None,
            "deep_cfr_samples": str(args.deep_cfr_samples) if args.deep_cfr_samples is not None else None,
            "deep_cfr_sample_type": str(args.deep_cfr_sample_type),
            "deep_cfr_max_samples": int(args.deep_cfr_max_samples),
            "deep_cfr_batch_size": int(args.deep_cfr_batch_size),
            "deep_cfr_device": str(args.deep_cfr_device),
            "deep_cfr_label": str(args.deep_cfr_label),
            "deep_cfr_required_seats": str(args.deep_cfr_required_seats),
            "deep_cfr_min_seat_samples": int(args.deep_cfr_min_seat_samples),
            "deep_cfr_max_dominant_seat_fraction": float(args.deep_cfr_max_dominant_seat_fraction),
            "hero_mode": str(args.hero_mode),
            "seed": int(args.seed),
        },
        "policy_execution_fidelity": {
            "hero_mode_effective": str(args.hero_mode),
            "selection_source_counts": selection_source_counts_sorted,
            "selection_source_by_street": selection_source_by_street_sorted,
            "selection_source_by_street_exact_hit": selection_source_by_street_exact_hit_sorted,
            "selection_source_by_street_non_exact_hit": selection_source_by_street_non_exact_hit_sorted,
            "chosen_vs_executed_match_count": int(chosen_vs_executed_match_count),
            "chosen_vs_executed_total": int(chosen_vs_executed_total),
            "chosen_vs_executed_match_rate": (
                float(chosen_vs_executed_match_count) / float(chosen_vs_executed_total)
                if chosen_vs_executed_total > 0
                else 0.0
            ),
            "chosen_vs_recommended_match_count": int(chosen_vs_recommended_match_count),
            "chosen_vs_recommended_total": int(chosen_vs_recommended_total),
            "chosen_vs_recommended_match_rate": (
                float(chosen_vs_recommended_match_count)
                / float(chosen_vs_recommended_total)
                if chosen_vs_recommended_total > 0
                else 0.0
            ),
            "chosen_vs_argmax_match_count": int(chosen_vs_argmax_match_count),
            "chosen_vs_argmax_total": int(chosen_vs_argmax_total),
            "chosen_vs_argmax_match_rate": (
                float(chosen_vs_argmax_match_count) / float(chosen_vs_argmax_total)
                if chosen_vs_argmax_total > 0
                else 0.0
            ),
            "exact_hit_and_translated_count": int(exact_hit_and_translated_count),
            "exact_hit_and_unmodified_count": int(exact_hit_and_unmodified_count),
            "exact_hit_translation_total": int(exact_hit_translation_total),
            "exact_hit_unmodified_rate": (
                float(exact_hit_and_unmodified_count) / float(exact_hit_translation_total)
                if exact_hit_translation_total > 0
                else 0.0
            ),
            "exact_hit_translation_by_street": exact_hit_translation_by_street_summary,
        },
        "mode_config": {
            "hero_mode_effective": str(args.hero_mode),
            "preflop_selection_mode_effective": preflop_selection_mode_effective,
            "postflop_selection_mode_effective": postflop_selection_mode_effective,
            "policy_use_approx_effective": bool(
                strategy_store_diagnostics_latest.get("policy_use_approx", False)
            ),
            "exact_action_fidelity_postflop_enabled": bool(
                EXACT_ACTION_FIDELITY_POSTFLOP
            ),
            "relax_exact_postflop_raise_guardrail_enabled": bool(
                RELAX_EXACT_POSTFLOP_RAISE_GUARDRAIL
            ),
        },
        "preflop_fallback_unavailable": {
            "decisions": int(street_decisions.get("preflop", 0)),
            "ok": int(street_ok.get("preflop", 0)),
            "unavailable": int(street_unavailable.get("preflop", 0)),
            "fallbacks": int(street_fallbacks.get("preflop", 0)),
            "fallback_rate": (
                float(street_fallbacks.get("preflop", 0))
                / float(street_decisions.get("preflop", 0))
                if int(street_decisions.get("preflop", 0)) > 0
                else 0.0
            ),
            "unavailable_rate": (
                float(street_unavailable.get("preflop", 0))
                / float(street_decisions.get("preflop", 0))
                if int(street_decisions.get("preflop", 0)) > 0
                else 0.0
            ),
        },
        "strategy_store": strategy_store_diagnostics_latest,
        "river_cfr": {
            "enabled": bool(river_cfr_config.get("enabled")),
            "override_enabled": bool(river_cfr_config.get("override_enabled")),
            "teacher_enabled": bool(river_cfr_config.get("teacher_enabled")),
            "teacher_eval_in_run_enabled": bool(river_cfr_config.get("teacher_eval_in_run")),
            "override_iters": int(river_cfr_config.get("override_iters", 0)),
            "teacher_iters": int(river_cfr_config.get("teacher_iters", 0)),
            "teacher_max_samples": int(river_cfr_config.get("teacher_max_samples", 0)),
            "max_calls_per_match": int(river_cfr_config.get("max_calls_per_match", 0)),
            "debug_enabled": bool(river_cfr_config.get("debug")),
            "calls": int(river_cfr_calls_total),
            "override_calls": int(river_cfr_calls_total),
            "skipped_no_range": int(river_cfr_skipped_no_range_total),
            "parse_fail": int(river_cfr_parse_fail_total),
            "runtime_ms_sum": float(river_cfr_runtime_ms_sum_total),
            "override_runtime_ms_sum": float(river_cfr_runtime_ms_sum_total),
            "runtime_ms_avg": (
                float(river_cfr_runtime_ms_sum_total) / float(river_cfr_calls_total)
                if river_cfr_calls_total > 0
                else 0.0
            ),
            "override_runtime_ms_avg": (
                float(river_cfr_runtime_ms_sum_total) / float(river_cfr_calls_total)
                if river_cfr_calls_total > 0
                else 0.0
            ),
            "match_call_limit_reached": bool(
                bool(river_cfr_config.get("override_enabled"))
                and int(river_cfr_state.get("calls", 0))
                >= int(river_cfr_config.get("max_calls_per_match", 0))
            ),
        },
        "river_cfr_teacher": river_cfr_teacher_summary,
        "river_cfr_teacher_samples": river_cfr_teacher_samples,
        "river_cfr_teacher_outputs": river_cfr_teacher_outputs,
        "ev_decomposition_bb": {
            "forced_blinds_antes": ev_decomposition_totals["forced_blinds_antes"],
            "preflop_voluntary": ev_decomposition_totals["preflop_voluntary"],
            "flop": ev_decomposition_totals["flop"],
            "turn": ev_decomposition_totals["turn"],
            "river": ev_decomposition_totals["river"],
            "attribution_total": ev_decomposition_totals["attribution_total"],
            "attribution_residual": ev_decomposition_totals["attribution_residual"],
        },
        "hero_style": {
            "hands_total": args.hands,
            "vpip_hands": vpip_hands,
            "vpip_pct": pct(vpip_hands, args.hands),
            "pfr_hands": pfr_hands,
            "pfr_pct": pct(pfr_hands, args.hands),
            "threebet_hands": threebet_hands,
            "threebet_opportunities": threebet_opportunities,
            "threebet_pct": pct(threebet_hands, threebet_opportunities),
            "showdown_when_vpip_hands": showdown_when_vpip_hands,
            "showdown_when_vpip_pct": pct(showdown_when_vpip_hands, vpip_hands),
            "af_by_street": {
                street: format_af(postflop_aggr[street], postflop_calls[street])
                for street in POSTFLOP_STREETS
            },
            "postflop_aggr": postflop_aggr,
            "postflop_calls": postflop_calls,
            "fold_to_bet_opportunities": fold_to_bet_opportunities,
            "fold_to_bet_folds": fold_to_bet_folds,
            "fold_to_bet_pct_by_street": {
                street: pct(fold_to_bet_folds[street], fold_to_bet_opportunities[street])
                for street in POSTFLOP_STREETS
            },
        },
        "preflop_leak_report": {
            "rows": preflop_leak_rows,
            "worst_by_total_bb": worst_nodes_by_total_bb,
            "worst_by_avg_bb": worst_nodes_by_avg_bb,
        },
        "preflop_node_attribution": {
            "attribution_method": "full_hand_with_outcome_class",
            "nodes": preflop_node_context_rows,
            "focus_nodes": focus_preflop_nodes,
        },
        "hu_preflop_pressure": {
            "enabled": is_hu_run,
            "unopened_action_distribution": {
                "total_hands_with_unopened_decision": hu_unopened_total,
                "rows": hu_unopened_action_rows,
            },
            "open_raise_fold_to_3bet_tax": {
                "count": open_raise_fold_to_3bet_count,
                "freq_pct_of_hands": pct(open_raise_fold_to_3bet_count, args.hands),
                "total_loss_proxy_bb": open_raise_fold_to_3bet_loss_proxy_total_bb,
                "avg_loss_proxy_bb": open_raise_fold_to_3bet_avg_loss_proxy,
            },
            "limp_iso_outcomes": {
                "total_limp_hands_tracked": limp_iso_total,
                "rows": limp_iso_outcome_rows,
                "hero_response_vs_iso_raise": {
                    "total": limp_iso_raised_total,
                    "rows": limp_iso_response_rows,
                },
            },
        },
        "flop_no_bet_tracker": {
            "potaware_debug_enabled": POTAWARE_DEBUG_ENABLED,
            "min_bet_bb": ENGINE_MIN_BET_BB,
            "tiny_pot_rule": "pot*0.33 < min_bet_bb",
            "total_events": flop_no_bet_total_events,
            "action_summary": flop_no_bet_action_rows,
            "pot_bucket_summary": flop_no_bet_pot_bucket_rows,
            "preflop_context_summary": flop_no_bet_preflop_context_rows,
            "potaware_medium_counters": {
                key: int(flop_potaware_medium_counters.get(key, 0))
                for key in FLOP_POTAWARE_MEDIUM_COUNTER_KEYS
            },
        },
        "turn_no_bet_tracker": {
            "potaware_debug_enabled": POTAWARE_DEBUG_ENABLED,
            "min_bet_bb": ENGINE_MIN_BET_BB,
            "tiny_pot_rule": "pot*0.33 < min_bet_bb",
            "total_events": turn_no_bet_total_events,
            "action_summary": turn_no_bet_action_rows,
            "pot_bucket_summary": turn_no_bet_pot_bucket_rows,
            "preflop_context_summary": turn_no_bet_preflop_context_rows,
        },
        "river_no_bet_tracker": {
            "potaware_debug_enabled": POTAWARE_DEBUG_ENABLED,
            "min_bet_bb": ENGINE_MIN_BET_BB,
            "tiny_pot_rule": "pot*0.33 < min_bet_bb",
            "total_events": river_no_bet_total_events,
            "action_summary": river_no_bet_action_rows,
            "pot_bucket_summary": river_no_bet_pot_bucket_rows,
            "preflop_context_summary": river_no_bet_preflop_context_rows,
        },
        "raised_or_3bp_medium_tiny_minbet_suppression": {
            "total_events": int(raised_or_3bp_medium_tiny_total_events),
            "raised_or_3bp_medium_tiny_minbet_prevented_count": int(
                raised_or_3bp_medium_tiny_minbet_prevented_count
            ),
            "raised_or_3bp_medium_tiny_minbet_prevented_freq_pct": pct(
                int(raised_or_3bp_medium_tiny_minbet_prevented_count),
                int(raised_or_3bp_medium_tiny_total_events),
            ),
            "raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb": float(
                raised_or_3bp_medium_tiny_minbet_prevented_ev_sum_bb
            ),
            "action_summary": raised_or_3bp_medium_tiny_action_rows,
        },
        "debug_trace": {
            "facing3bet_raise_trigger_mode": debug_trace_trigger_mode,
            "first_facing3bet_raise_trace": debug_facing3bet_raise_trace,
        },
        "raise_outlier_report": {
            "raise_count_total": raise_count_total,
            "raise_over_20": {
                "count": raise_over_20,
                "pct": pct(raise_over_20, raise_count_total),
            },
            "raise_over_40": {
                "count": raise_over_40,
                "pct": pct(raise_over_40, raise_count_total),
            },
            "raise_over_80": {
                "count": raise_over_80,
                "pct": pct(raise_over_80, raise_count_total),
            },
            "preflop_open_non_allin_raises_over_15bb_count": len(
                preflop_open_non_allin_raises_over_15bb
            ),
            "largest_raises": top_10_biggest_raises,
            "clipped_adjusted_by_context": clipped_adjusted_context_rows,
        },
        "action_translation_audit": {
            "bet33_fraction_of_pot": summarize_float_list(bet33_fractions),
            "bet50_fraction_of_pot": summarize_float_list(bet50_fractions),
            "bet75_fraction_of_pot": summarize_float_list(bet75_fractions),
            "bet33_fraction_buckets": bet33_bucket_summary,
            "bet50_fraction_buckets": bet50_bucket_summary,
            "bet75_fraction_buckets": bet75_bucket_summary,
            "bet33_forced_halfpot_count": bet33_forced_halfpot_count,
            "bet50_forced_halfpot_count": bet50_forced_halfpot_count,
            "bet75_forced_halfpot_count": bet75_forced_halfpot_count,
            "effective_small_bet_fraction": effective_small_bet_fraction,
            "effective_large_bet_fraction": effective_large_bet_fraction,
            "mdf_analysis": {
                "small_bet_continue_target_pct": mdf_continue_target_for_fraction(
                    effective_small_bet_fraction
                )
                * 100.0,
                "large_bet_continue_target_pct": mdf_continue_target_for_fraction(
                    effective_large_bet_fraction
                )
                * 100.0,
            },
            "raise_size_bb": summarize_float_list(raise_sizes_bb),
            "raise_fraction_of_pot": summarize_float_list(raise_fractions),
            "translation_clipped_count": translation_clipped_count,
            "translation_adjusted_count": translation_adjusted_count,
            "env_legalized_count": env_legalized_count,
            "clipped_or_adjusted_total": clipped_or_adjusted_total,
            "preflop_raise_clamped_count": preflop_raise_clamped_count,
            "preflop_raise_converted_to_allin_count": preflop_raise_converted_to_allin_count,
            "postflop_raise_clamped_count": postflop_raise_clamped_count,
            "raise_clamped_count": preflop_raise_clamped_count + postflop_raise_clamped_count,
            "exact_requested_size_used_count": exact_requested_size_used_count,
            "postflop_raise_guardrail_relaxed_count": postflop_raise_guardrail_relaxed_count,
            "exact_hit_and_translated_count": exact_hit_and_translated_count,
            "exact_hit_and_unmodified_count": exact_hit_and_unmodified_count,
            "exact_hit_translation_total": exact_hit_translation_total,
            "engine_size_capped_count": engine_size_capped_count,
            "engine_size_capped_by_street_intent": engine_size_capped_rows,
            "illegal_action_avoided_count": illegal_action_avoided_count,
            "large_preflop_non_allin_raises_over_40bb_count": len(large_preflop_non_allin_raises),
            "preflop_open_non_allin_raises_over_15bb_count": len(
                preflop_open_non_allin_raises_over_15bb
            ),
        },
    }

    if not args.quiet:
        print_summary(raw_stats, net_stats, args, fallback_count, diagnostics)
        debug_trace = diagnostics.get("debug_trace", {})
        trigger_mode = str(debug_trace.get("facing3bet_raise_trigger_mode", "off"))
        if trigger_mode != "off":
            print("\n-- Debug Trace: HU FACING_3BET Raise --")
            trace = debug_trace.get("first_facing3bet_raise_trace")
            print(f"trigger_mode: {trigger_mode}")
            if isinstance(trace, dict):
                print(json.dumps(trace, indent=2))
            else:
                print("no matching hand found for trigger in this run")

        total_engine_calls = sum(status_counts.values())
        print("\n-- Engine status counts --")
        if total_engine_calls == 0:
            print("no engine calls recorded")
        else:
            for key in sorted(status_counts.keys()):
                value = status_counts[key]
                pct_value = (value / total_engine_calls) * 100.0
                print(f"{key}: {value} ({pct_value:.2f}%)")

    elapsed_total = time.time() - start_time
    if not args.quiet:
        print(f"\nresults saved: {args.results}")
        if args.trace_hands > 0:
            print(f"trace saved: {args.trace_file}")
    if args.report_json is not None:
        report_payload = {
            "config": {
                "hands": args.hands,
                "hands_per_seed": args.hands_per_seed,
                "players": args.players,
                "match_format": args.match_format,
                "stack_bb": args.stack_bb,
                "seed": args.seed,
                "seeds": args.seeds,
                "hero_mode": args.hero_mode,
                "eval_backend": args.eval_backend,
                "policy_cmd": args.policy_cmd,
                "deep_cfr_model_state": str(args.deep_cfr_model_state)
                if args.deep_cfr_model_state is not None
                else None,
                "deep_cfr_samples": str(args.deep_cfr_samples) if args.deep_cfr_samples is not None else None,
                "deep_cfr_sample_type": args.deep_cfr_sample_type,
                "deep_cfr_max_samples": int(args.deep_cfr_max_samples),
                "deep_cfr_batch_size": int(args.deep_cfr_batch_size),
                "deep_cfr_device": str(args.deep_cfr_device),
                "deep_cfr_label": str(args.deep_cfr_label),
                "deep_cfr_required_seats": str(args.deep_cfr_required_seats),
                "deep_cfr_min_seat_samples": int(args.deep_cfr_min_seat_samples),
                "deep_cfr_max_dominant_seat_fraction": float(args.deep_cfr_max_dominant_seat_fraction),
                "preflop_selection_mode_effective": preflop_selection_mode_effective,
                "postflop_selection_mode_effective": postflop_selection_mode_effective,
                "exact_action_fidelity_postflop_enabled": bool(
                    EXACT_ACTION_FIDELITY_POSTFLOP
                ),
                "relax_exact_postflop_raise_guardrail_enabled": bool(
                    RELAX_EXACT_POSTFLOP_RAISE_GUARDRAIL
                ),
                "opponents": args.opponents,
                "open_size": args.open_size,
                "threebet_size": args.threebet_size,
                "fourbet_size": args.fourbet_size,
                "rake_mode": args.rake_mode,
                "rake_bb_per_hand": args.rake_bb_per_hand,
                "rake_fixed_bb": args.rake_fixed_bb,
                "rake_percent": args.rake_percent,
                "rake_cap_bb": args.rake_cap_bb,
                "opponent_lineup": args.opponent_lineup,
                "max_actions_per_hand": args.max_actions_per_hand,
                "trace_hands": args.trace_hands,
                "dump_biggest_losers": args.dump_biggest_losers,
                "assert_large_preflop_raises_allin": args.assert_large_preflop_raises_allin,
                "assert_preflop_open_max15": args.assert_preflop_open_max15,
            },
            "summary": {
                "raw": raw_stats,
                "net": net_stats,
                "fallback_count": fallback_count,
                "status_counts": status_counts,
                "diagnostics": diagnostics,
                "elapsed_sec": float(elapsed_total),
                "results_csv": str(args.results),
                "results_csv_bytes": int(args.results.stat().st_size) if args.results.exists() else 0,
            },
        }
        with args.report_json.open("w", encoding="utf-8") as fp:
            json.dump(report_payload, fp, indent=2)
        if not args.quiet:
            print(f"report_json saved: {args.report_json}")
    if args.dump_biggest_losers > 0:
        with args.biggest_losers_file.open("w", encoding="utf-8") as fp:
            for hand in worst_hands[: args.dump_biggest_losers]:
                fp.write(json.dumps(hand, separators=(",", ":")) + "\n")
        if not args.quiet:
            print(f"biggest_losers saved: {args.biggest_losers_file}")
    if not args.quiet:
        print(f"elapsed: {elapsed_total:.2f}s")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(1)
