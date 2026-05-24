from __future__ import annotations

from typing import Final

MAX_ACTIONS: Final[int] = 13

SLOT_FOLD: Final[int] = 0
SLOT_CHECK: Final[int] = 1
SLOT_CALL: Final[int] = 2

BET_SLOT_START: Final[int] = 3
BET_SLOT_COUNT: Final[int] = 4

RAISE_SLOT_START: Final[int] = 7
RAISE_SLOT_COUNT: Final[int] = 5

SLOT_ALL_IN: Final[int] = 12

PREFLOP_OPEN_RAISE_GRID: Final[tuple[float, ...]] = (2.0, 2.5, 3.0, 3.5)
PREFLOP_RERAISE_MULTIPLIER_GRID: Final[tuple[float, ...]] = (2.0, 2.3, 2.7, 3.2, 4.0)

FLOP_BET_GRID: Final[tuple[float, ...]] = (0.33, 0.67, 1.0, 1.5)
TURN_BET_GRID: Final[tuple[float, ...]] = (0.50, 0.75, 1.0, 1.5)
RIVER_BET_GRID: Final[tuple[float, ...]] = (0.33, 0.75, 1.25, 2.0)

FLOP_RAISE_MULTIPLIER_GRID: Final[tuple[float, ...]] = (2.0, 2.5, 3.0, 3.5, 4.5)
TURN_RAISE_MULTIPLIER_GRID: Final[tuple[float, ...]] = (2.0, 2.5, 3.0, 3.75, 5.0)
RIVER_RAISE_MULTIPLIER_GRID: Final[tuple[float, ...]] = (1.8, 2.3, 3.0, 4.0, 5.5)


def street_bet_grid(street: str) -> tuple[float, ...]:
    key = str(street).strip().lower()
    if key == "flop":
        return FLOP_BET_GRID
    if key == "turn":
        return TURN_BET_GRID
    if key == "river":
        return RIVER_BET_GRID
    if key == "preflop":
        return PREFLOP_OPEN_RAISE_GRID
    return ()


def street_raise_multiplier_grid(street: str) -> tuple[float, ...]:
    key = str(street).strip().lower()
    if key == "flop":
        return FLOP_RAISE_MULTIPLIER_GRID
    if key == "turn":
        return TURN_RAISE_MULTIPLIER_GRID
    if key == "river":
        return RIVER_RAISE_MULTIPLIER_GRID
    return ()
