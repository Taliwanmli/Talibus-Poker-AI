import re
import unittest
from pathlib import Path

try:
    from eval.deep_cfr_checkpoint_worker import (
        BET_SLOT_COUNT,
        BET_SLOT_START,
        MAX_ACTIONS,
        RAISE_SLOT_COUNT,
        RAISE_SLOT_START,
        SLOT_ALL_IN,
        SLOT_CALL,
        SLOT_CHECK,
        SLOT_FOLD,
        build_slot_action_map,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    BET_SLOT_COUNT = BET_SLOT_START = MAX_ACTIONS = None
    RAISE_SLOT_COUNT = RAISE_SLOT_START = None
    SLOT_ALL_IN = SLOT_CALL = SLOT_CHECK = SLOT_FOLD = None
    build_slot_action_map = None


torch_required = unittest.skipIf(
    build_slot_action_map is None,
    "torch is not installed; skipping checkpoint worker slot tests",
)


@torch_required
class DeepCfrCheckpointSlotParityTests(unittest.TestCase):
    def test_solver_and_worker_slot_constants_match(self):
        repo_root = Path(__file__).resolve().parents[1]
        solver_path = repo_root / "solver" / "cfr" / "src" / "nlhe_game.rs"
        source = solver_path.read_text(encoding="utf-8")

        expected_constants = {
            "POLICY_MAX_ACTIONS": MAX_ACTIONS,
            "POLICY_BET_SLOT_START": BET_SLOT_START,
            "POLICY_BET_SLOT_COUNT": BET_SLOT_COUNT,
            "POLICY_RAISE_SLOT_START": RAISE_SLOT_START,
            "POLICY_RAISE_SLOT_COUNT": RAISE_SLOT_COUNT,
            "POLICY_ALL_IN_SLOT": SLOT_ALL_IN,
        }
        for name, expected in expected_constants.items():
            match = re.search(rf"{name}: [^=]+=\s*(\d+);", source)
            self.assertIsNotNone(match, f"missing solver constant {name} in {solver_path}")
            self.assertEqual(
                int(match.group(1)),
                int(expected),
                f"solver constant mismatch for {name}",
            )

    def test_preflop_no_to_call_uses_check_and_four_open_slots(self):
        legal_slots, actions = build_slot_action_map(
            "preflop",
            {
                "to_call_bb": 0.0,
                "pot_bb": 1.5,
                "hero_stack_bb": 100.0,
                "hero_street_contrib_bb": 0.0,
            },
        )
        self.assertEqual(legal_slots, [SLOT_CHECK, 3, 4, 5, 6])
        self.assertEqual(actions[3]["type"], "raise")
        self.assertEqual(actions[6]["type"], "raise")

    def test_preflop_facing_raise_uses_five_reraise_slots(self):
        legal_slots, actions = build_slot_action_map(
            "preflop",
            {
                "to_call_bb": 2.0,
                "pot_bb": 5.0,
                "hero_stack_bb": 98.0,
                "hero_street_contrib_bb": 0.0,
            },
        )
        self.assertEqual(
            legal_slots,
            [SLOT_FOLD, SLOT_CALL, 7, 8, 9, 10, 11],
        )
        self.assertTrue(all(actions[slot]["type"] == "raise" for slot in [7, 8, 9, 10, 11]))

    def test_turn_facing_raise_uses_ordered_raise_slots(self):
        legal_slots, actions = build_slot_action_map(
            "turn",
            {
                "to_call_bb": 5.0,
                "pot_bb": 15.0,
                "hero_stack_bb": 95.0,
                "hero_street_contrib_bb": 2.0,
            },
        )
        self.assertEqual(
            legal_slots,
            [SLOT_FOLD, SLOT_CALL, 7, 8, 9, 10, 11],
        )
        sizes = [float(actions[slot]["sizeBb"]) for slot in [7, 8, 9, 10, 11]]
        self.assertEqual(sizes, sorted(sizes), "raise slots should be ordered by total size")


if __name__ == "__main__":
    unittest.main()
