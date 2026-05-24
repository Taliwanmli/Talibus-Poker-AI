#!/usr/bin/env python3
"""Policy worker that serves Deep CFR checkpoint actions over stdio JSON lines."""

from __future__ import annotations

import argparse
import json
import math
import random
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
from action_slots import (
    BET_SLOT_COUNT,
    BET_SLOT_START,
    PREFLOP_OPEN_RAISE_GRID,
    PREFLOP_RERAISE_MULTIPLIER_GRID,
    RAISE_SLOT_COUNT,
    RAISE_SLOT_START,
    SLOT_ALL_IN,
    SLOT_CALL,
    SLOT_CHECK,
    SLOT_FOLD,
    MAX_ACTIONS as ACTION_SLOT_MAX_ACTIONS,
    street_bet_grid,
    street_raise_multiplier_grid,
)
from train import load_model_weights, resolve_device

if MAX_ACTIONS != ACTION_SLOT_MAX_ACTIONS:
    raise RuntimeError(
        f"action slot mismatch: model MAX_ACTIONS={MAX_ACTIONS}, action schema MAX_ACTIONS={ACTION_SLOT_MAX_ACTIONS}"
    )


CARD_DIMS = 52
STREET_DIMS = 4
POSITION_DIMS = 6
HISTORY_SLOTS = 24
HISTORY_ACTION_BUCKET_DIMS = 8
HISTORY_ACTION_DIMS = HISTORY_ACTION_BUCKET_DIMS + POSITION_DIMS + 1

HOLE_OFFSET = 0
BOARD_OFFSET = HOLE_OFFSET + CARD_DIMS
STREET_OFFSET = BOARD_OFFSET + CARD_DIMS
HERO_POSITION_OFFSET = STREET_OFFSET + STREET_DIMS
ACTOR_POSITION_OFFSET = HERO_POSITION_OFFSET + POSITION_DIMS
POT_OFFSET = ACTOR_POSITION_OFFSET + POSITION_DIMS
TO_CALL_OFFSET = POT_OFFSET + 1
HERO_STACK_OFFSET = TO_CALL_OFFSET + 1
HERO_STREET_CONTRIB_OFFSET = HERO_STACK_OFFSET + 1
ACTIVE_PLAYERS_RATIO_OFFSET = HERO_STREET_CONTRIB_OFFSET + 1
PLAYER_ACTIVE_MASK_OFFSET = ACTIVE_PLAYERS_RATIO_OFFSET + 1
PLAYER_STACK_OFFSET = PLAYER_ACTIVE_MASK_OFFSET + POSITION_DIMS
PLAYER_TOTAL_CONTRIB_OFFSET = PLAYER_STACK_OFFSET + POSITION_DIMS
PLAYER_STREET_CONTRIB_OFFSET = PLAYER_TOTAL_CONTRIB_OFFSET + POSITION_DIMS
PREFLOP_RAISE_COUNT_OFFSET = PLAYER_STREET_CONTRIB_OFFSET + POSITION_DIMS
HISTORY_OFFSET = PREFLOP_RAISE_COUNT_OFFSET + 1

ACTION_BUCKET_FOLD = 0
ACTION_BUCKET_CHECK = 1
ACTION_BUCKET_CALL = 2
ACTION_BUCKET_BET_SMALL = 3
ACTION_BUCKET_BET_MEDIUM = 4
ACTION_BUCKET_BET_LARGE = 5
ACTION_BUCKET_RAISE = 6
ACTION_BUCKET_ALL_IN = 7

STREET_TO_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
RANK_TO_INDEX = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
}
SUIT_TO_INDEX = {"C": 0, "D": 1, "H": 2, "S": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep CFR checkpoint stdio worker.")
    parser.add_argument("--model-state", type=Path, required=True, help="Checkpoint .pt (or .onnx sibling).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def normalize_checkpoint_path(path: Path) -> Path:
    candidate = path.resolve()
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


def parse_card_index(raw: str) -> int | None:
    text = str(raw).strip().upper()
    if not text:
        return None
    if len(text) == 3 and text.startswith("10"):
        rank = "T"
        suit = text[2:3]
    elif len(text) >= 2:
        rank = text[0:1]
        suit = text[1:2]
    else:
        return None
    rank_idx = RANK_TO_INDEX.get(rank)
    suit_idx = SUIT_TO_INDEX.get(suit)
    if rank_idx is None or suit_idx is None:
        return None
    return (rank_idx * 4) + suit_idx


def extract_last_number(token: str) -> float:
    current = []
    numbers: list[float] = []
    for ch in token:
        if ch.isdigit() or ch == ".":
            current.append(ch)
        elif current:
            text = "".join(current)
            try:
                numbers.append(float(text))
            except Exception:
                pass
            current = []
    if current:
        text = "".join(current)
        try:
            numbers.append(float(text))
        except Exception:
            pass
    return numbers[-1] if numbers else 0.0


def classify_action(token: str) -> int | None:
    text = str(token).strip().lower()
    if text == "f":
        return ACTION_BUCKET_FOLD
    if text == "x":
        return ACTION_BUCKET_CHECK
    if text == "c":
        return ACTION_BUCKET_CALL
    if text == "ai":
        return ACTION_BUCKET_ALL_IN
    if text.startswith("r:"):
        return ACTION_BUCKET_RAISE
    if text.startswith("b:"):
        amount = max(0.0, extract_last_number(text) / 4.0)
        if amount <= 0.33:
            return ACTION_BUCKET_BET_SMALL
        if amount <= 0.66:
            return ACTION_BUCKET_BET_MEDIUM
        return ACTION_BUCKET_BET_LARGE
    return None


def build_features(street: str, hand_state: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    context = hand_state.get("deepCfrContext", {})
    if not isinstance(context, dict):
        raise ValueError("missing deepCfrContext in handState")
    players = context.get("players", [])
    if not isinstance(players, list) or not players:
        raise ValueError("deepCfrContext.players missing or empty")

    hero_id = int(context.get("hero_id"))
    actor_id = int(context.get("actor_id", hero_id))
    players_by_id: dict[int, dict[str, Any]] = {}
    for row in players:
        if isinstance(row, dict) and "id" in row:
            players_by_id[int(row["id"])] = row
    hero_row = players_by_id.get(hero_id)
    actor_row = players_by_id.get(actor_id, hero_row)
    if hero_row is None:
        raise ValueError("deepCfrContext is missing hero player snapshot")
    if actor_row is None:
        actor_row = hero_row

    street_key = str(street).strip().lower()
    if street_key not in STREET_TO_INDEX:
        raise ValueError(f"invalid street: {street}")
    street_state = hand_state.get(street_key, {})
    if not isinstance(street_state, dict):
        street_state = {}

    out = np.zeros((INPUT_DIM,), dtype=np.float32)

    hero_hand = hand_state.get("preflop", {}).get("heroHand", []) if isinstance(hand_state.get("preflop"), dict) else []
    if isinstance(hero_hand, list):
        for card in hero_hand:
            idx = parse_card_index(str(card))
            if idx is not None and 0 <= idx < CARD_DIMS:
                out[HOLE_OFFSET + idx] = 1.0

    board_cards = street_state.get("boardCards", [])
    if isinstance(board_cards, list):
        for card in board_cards:
            idx = parse_card_index(str(card))
            if idx is not None and 0 <= idx < CARD_DIMS:
                out[BOARD_OFFSET + idx] = 1.0

    out[STREET_OFFSET + STREET_TO_INDEX[street_key]] = 1.0

    hero_seat = int(hero_row.get("seat_index", 0))
    actor_seat = int(actor_row.get("seat_index", hero_seat))
    if 0 <= hero_seat < POSITION_DIMS:
        out[HERO_POSITION_OFFSET + hero_seat] = 1.0
    if 0 <= actor_seat < POSITION_DIMS:
        out[ACTOR_POSITION_OFFSET + actor_seat] = 1.0

    starting_stack = float(context.get("starting_stack_bb", 0.0))
    if not math.isfinite(starting_stack) or starting_stack <= 0.0:
        starting_stack = max(
            1.0,
            max(
                float(row.get("stack_bb", 0.0)) + float(row.get("total_contribution_bb", 0.0))
                for row in players_by_id.values()
            ),
        )

    total_pot = 0.0
    for row in players_by_id.values():
        total_pot += max(0.0, float(row.get("total_contribution_bb", 0.0)))
    table_size = max(1, int(context.get("table_size", len(players_by_id))))
    out[POT_OFFSET] = float(total_pot / max(1.0, starting_stack * table_size))
    to_call_bb = float(street_state.get("toCallBb", 0.0))
    out[TO_CALL_OFFSET] = float(max(0.0, to_call_bb) / max(1.0, starting_stack))
    out[HERO_STACK_OFFSET] = float(max(0.0, float(hero_row.get("stack_bb", 0.0))) / max(1.0, starting_stack))
    out[HERO_STREET_CONTRIB_OFFSET] = float(
        max(0.0, float(hero_row.get("street_contribution_bb", 0.0))) / max(1.0, starting_stack)
    )

    active_count = 0
    for row in players_by_id.values():
        seat = int(row.get("seat_index", -1))
        if seat < 0 or seat >= POSITION_DIMS:
            continue
        folded = bool(row.get("folded", False))
        if not folded:
            active_count += 1
        out[PLAYER_ACTIVE_MASK_OFFSET + seat] = 0.0 if folded else 1.0
        out[PLAYER_STACK_OFFSET + seat] = float(max(0.0, float(row.get("stack_bb", 0.0))) / max(1.0, starting_stack))
        out[PLAYER_TOTAL_CONTRIB_OFFSET + seat] = float(
            max(0.0, float(row.get("total_contribution_bb", 0.0))) / max(1.0, starting_stack)
        )
        out[PLAYER_STREET_CONTRIB_OFFSET + seat] = float(
            max(0.0, float(row.get("street_contribution_bb", 0.0))) / max(1.0, starting_stack)
        )
    out[ACTIVE_PLAYERS_RATIO_OFFSET] = float(active_count / max(1, table_size))
    preflop_raise_count = int(context.get("preflop_raise_count", 0))
    out[PREFLOP_RAISE_COUNT_OFFSET] = float(max(0.0, min(1.0, preflop_raise_count / 6.0)))

    action_history = context.get("action_history", [])
    if not isinstance(action_history, list):
        action_history = []
    start_idx = max(0, len(action_history) - HISTORY_SLOTS)
    for slot, event in enumerate(action_history[start_idx:]):
        if not isinstance(event, dict):
            continue
        token = str(event.get("token", ""))
        bucket = classify_action(token)
        if bucket is None:
            continue
        base = HISTORY_OFFSET + (slot * HISTORY_ACTION_DIMS)
        out[base + bucket] = 1.0
        actor_hist_id = int(event.get("actor_id", -1))
        actor_hist = players_by_id.get(actor_hist_id, {})
        actor_hist_seat = int(actor_hist.get("seat_index", -1)) if isinstance(actor_hist, dict) else -1
        if 0 <= actor_hist_seat < POSITION_DIMS:
            out[base + HISTORY_ACTION_BUCKET_DIMS + actor_hist_seat] = 1.0
        out[base + HISTORY_ACTION_DIMS - 1] = float(max(0.0, min(1.0, extract_last_number(token) / 4.0)))

    return out, {
        "hero_seat_index": hero_seat,
        "actor_seat_index": actor_seat,
        "to_call_bb": max(0.0, to_call_bb),
        "pot_bb": max(0.0, total_pot),
        "hero_stack_bb": max(0.0, float(hero_row.get("stack_bb", 0.0))),
        "hero_street_contrib_bb": max(0.0, float(hero_row.get("street_contribution_bb", 0.0))),
        "street": street_key,
    }


def _all_in_is_distinct(
    *,
    to_call: float,
    hero_street: float,
    hero_stack: float,
    min_bet_bb: float = 1.0,
) -> bool:
    if hero_stack <= 1e-9:
        return False
    current_bet = hero_street + to_call
    max_total = hero_street + hero_stack
    call_amount = min(to_call, hero_stack)
    all_in_is_distinct = not (to_call > 1e-9 and abs(call_amount - hero_stack) <= 1e-9)

    if to_call <= 1e-9:
        if max_total >= min_bet_bb:
            all_in_is_distinct = False
    elif max_total > current_bet:
        min_raise_to = current_bet + max(min_bet_bb, to_call)
        if min_raise_to <= max_total + 1e-9:
            all_in_is_distinct = False
    return all_in_is_distinct


def _dedupe_candidates(
    candidates: list[tuple[float, str]],
    *,
    min_size: float = 0.0,
) -> list[tuple[float, str]]:
    deduped: list[tuple[float, str]] = []
    seen: set[float] = set()
    for size, note in candidates:
        finite_size = float(size)
        if not math.isfinite(finite_size):
            continue
        finite_size = max(min_size, finite_size)
        key = round(finite_size, 4)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((finite_size, note))
    return deduped


def build_slot_action_map(street: str, context: dict[str, Any]) -> tuple[list[int], dict[int, dict[str, Any]]]:
    to_call = float(context.get("to_call_bb", 0.0))
    pot = max(1.0, float(context.get("pot_bb", 1.0)))
    hero_street = max(0.0, float(context.get("hero_street_contrib_bb", 0.0)))
    hero_stack = max(0.0, float(context.get("hero_stack_bb", 0.0)))
    street_key = str(street).strip().lower()

    actions: dict[int, dict[str, Any]] = {}
    if to_call > 1e-9:
        actions[SLOT_FOLD] = {"type": "fold", "note": "[deep_cfr_checkpoint] fold"}
        actions[SLOT_CALL] = {"type": "call", "note": "[deep_cfr_checkpoint] call"}

        current_bet = hero_street + to_call
        raise_min = to_call + 1.0
        if street_key == "preflop":
            raise_candidates = _dedupe_candidates(
                [
                    (
                        max(current_bet * float(multiplier), raise_min),
                        f"[deep_cfr_checkpoint] raise_{float(multiplier):.2f}x",
                    )
                    for multiplier in PREFLOP_RERAISE_MULTIPLIER_GRID
                ],
                min_size=raise_min,
            )
        else:
            reference = max(to_call, 1.0)
            raise_candidates = _dedupe_candidates(
                [
                    (
                        max(current_bet + (float(multiplier) * reference), raise_min),
                        f"[deep_cfr_checkpoint] raise_{float(multiplier):.2f}x",
                    )
                    for multiplier in street_raise_multiplier_grid(street_key)
                ],
                min_size=raise_min,
            )
        for offset, (size_bb, note) in enumerate(raise_candidates[:RAISE_SLOT_COUNT]):
            slot = RAISE_SLOT_START + offset
            actions[slot] = {"type": "raise", "sizeBb": float(size_bb), "note": note}

        if _all_in_is_distinct(
            to_call=to_call,
            hero_street=hero_street,
            hero_stack=hero_stack,
            min_bet_bb=1.0,
        ):
            actions[SLOT_ALL_IN] = {"type": "all_in", "note": "[deep_cfr_checkpoint] all_in"}
    else:
        actions[SLOT_CHECK] = {"type": "check", "note": "[deep_cfr_checkpoint] check"}

        if street_key == "preflop":
            bet_candidates = _dedupe_candidates(
                [
                    (
                        float(multiplier),
                        f"[deep_cfr_checkpoint] open_{float(multiplier):.2f}",
                    )
                    for multiplier in PREFLOP_OPEN_RAISE_GRID
                ],
                min_size=1.0,
            )
            for offset, (size_bb, note) in enumerate(bet_candidates[:BET_SLOT_COUNT]):
                slot = BET_SLOT_START + offset
                actions[slot] = {"type": "raise", "sizeBb": float(size_bb), "note": note}
        else:
            bet_candidates = _dedupe_candidates(
                [
                    (
                        max(1.0, float(fraction) * pot),
                        (
                            "[deep_cfr_checkpoint] bet_033"
                            if float(fraction) <= 0.40
                            else "[deep_cfr_checkpoint] bet_050"
                            if float(fraction) <= 0.60
                            else "[deep_cfr_checkpoint] bet_075"
                            if float(fraction) <= 0.90
                            else "[deep_cfr_checkpoint] bet_150"
                        ),
                    )
                    for fraction in street_bet_grid(street_key)
                ],
                min_size=1.0,
            )
            for offset, (size_bb, note) in enumerate(bet_candidates[:BET_SLOT_COUNT]):
                slot = BET_SLOT_START + offset
                actions[slot] = {"type": "bet", "sizeBb": float(size_bb), "note": note}

        if _all_in_is_distinct(
            to_call=to_call,
            hero_street=hero_street,
            hero_stack=hero_stack,
            min_bet_bb=1.0,
        ):
            actions[SLOT_ALL_IN] = {"type": "all_in", "note": "[deep_cfr_checkpoint] all_in"}

    legal_slots = sorted(slot for slot in actions.keys() if 0 <= slot < MAX_ACTIONS)
    return legal_slots, actions


def sample_slot(
    *,
    rng: random.Random,
    legal_slots: list[int],
    probs: np.ndarray,
) -> int:
    weights = [max(0.0, float(probs[slot])) for slot in legal_slots]
    total = sum(weights)
    if total <= 0.0:
        return int(legal_slots[0])
    target = rng.random() * total
    running = 0.0
    for slot, weight in zip(legal_slots, weights):
        running += weight
        if running >= target:
            return int(slot)
    return int(legal_slots[-1])


def make_unavailable(error_text: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "recommendedAction": None,
        "executedAction": None,
        "argmaxAction": None,
        "mix": None,
        "error": error_text,
    }


def main() -> int:
    args = parse_args()
    model_state = normalize_checkpoint_path(args.model_state)
    device = resolve_device(args.device)
    hidden_dim, bottleneck_dim = infer_model_dims(model_state)
    cfg = ModelConfig(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        max_actions=MAX_ACTIONS,
        dropout_p=0.10,
    )
    model = DeepCfrNet(cfg).to(device)
    if not load_model_weights(model, model_state, device):
        raise RuntimeError(f"failed to load checkpoint weights: {model_state}")
    model.eval()

    rng = random.Random(int(args.seed))
    request_counter = 0

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            street = str(payload.get("street", "")).strip().lower()
            hand_state = payload.get("handState", {})
            if not isinstance(hand_state, dict):
                raise ValueError("handState must be an object")
            features, feature_meta = build_features(street, hand_state)
            legal_slots, slot_actions = build_slot_action_map(street, feature_meta)
            if not legal_slots:
                raise ValueError("no legal Deep CFR policy slots resolved for request")

            action_mask = np.zeros((MAX_ACTIONS,), dtype=np.float32)
            for slot in legal_slots:
                action_mask[int(slot)] = 1.0

            x = torch.from_numpy(features.reshape(1, -1)).to(device=device, dtype=torch.float32)
            mask = torch.from_numpy(action_mask.reshape(1, -1)).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                probs = model(x, action_mask=mask, strategy_mode=True).detach().cpu().numpy()[0]

            argmax_slot = int(max(legal_slots, key=lambda slot: float(probs[slot])))
            if args.mode == "sample":
                # Salt each request to avoid identical sampling across equal states.
                rng.seed(int(args.seed) ^ (request_counter * 0x9E37_79B9))
                selected_slot = sample_slot(rng=rng, legal_slots=legal_slots, probs=probs)
            else:
                selected_slot = argmax_slot
            request_counter += 1

            selected_action = dict(slot_actions[selected_slot])
            argmax_action = dict(slot_actions[argmax_slot])
            response = {
                "status": "ok",
                "recommendedAction": selected_action,
                "executedAction": dict(selected_action),
                "argmaxAction": argmax_action,
                "mix": None,
                "debug": {
                    "backend": "deep_cfr_checkpoint_worker",
                    "selected_slot": int(selected_slot),
                    "argmax_slot": int(argmax_slot),
                    "legal_slots": [int(slot) for slot in legal_slots],
                    "mode": str(args.mode),
                    "model_state": str(model_state),
                },
            }
        except Exception as exc:
            response = make_unavailable(f"deep_cfr_checkpoint_worker_error: {exc}")

        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
