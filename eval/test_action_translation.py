import math
import unittest
from types import SimpleNamespace

from pokerenv.common import PlayerAction

from eval.run_league import build_action_from_intent


def _valid_actions(low: float, high: float):
    return {
        "actions_list": [
            PlayerAction.CHECK,
            PlayerAction.FOLD,
            PlayerAction.CALL,
            PlayerAction.BET,
        ],
        "bet_range": (float(low), float(high)),
    }


class ActionTranslationTests(unittest.TestCase):
    def test_preflop_open_is_hard_clamped(self):
        table = SimpleNamespace(bet_to_match=0.0)
        player = SimpleNamespace(bet_this_street=0.0)
        action, metadata = build_action_from_intent(
            "raise",
            table,
            player,
            _valid_actions(2.0, 99.0),
            pot_before_action=1.5,
            street="preflop",
            preflop_raise_count_before_action=0,
            pre_stack_bb=100.0,
            apply_hero_preflop_sizing=True,
            open_size_bb=97.5,
            threebet_size_bb=9.0,
            fourbet_size_bb=22.0,
        )

        self.assertEqual(action.action_type, PlayerAction.BET)
        self.assertGreaterEqual(float(action.bet_amount), 2.0)
        self.assertLessEqual(float(action.bet_amount), 15.0)
        self.assertTrue(bool(metadata.get("preflop_raise_clamped")))
        self.assertFalse(bool(metadata.get("is_allin")))
        self.assertEqual(metadata.get("preflop_node"), "OPEN")

    def test_postflop_raise_is_clamped_below_extreme_size(self):
        table = SimpleNamespace(bet_to_match=30.0)
        player = SimpleNamespace(bet_this_street=6.0)
        action, metadata = build_action_from_intent(
            "raise",
            table,
            player,
            _valid_actions(31.0, 100.0),
            pot_before_action=20.0,
            street="flop",
            preflop_raise_count_before_action=0,
            pre_stack_bb=100.0,
            apply_hero_preflop_sizing=False,
            open_size_bb=2.5,
            threebet_size_bb=9.0,
            fourbet_size_bb=22.0,
        )

        self.assertEqual(action.action_type, PlayerAction.BET)
        self.assertTrue(math.isclose(float(action.bet_amount), 49.0, rel_tol=0, abs_tol=1e-9))
        self.assertTrue(bool(metadata.get("postflop_raise_clamped")))
        self.assertTrue(bool(metadata.get("translation_adjusted")))
        self.assertFalse(bool(metadata.get("is_allin")))

    def test_requested_bet_size_is_respected_when_legal(self):
        table = SimpleNamespace(bet_to_match=0.0)
        player = SimpleNamespace(bet_this_street=0.0)
        action, metadata = build_action_from_intent(
            "bet33",
            table,
            player,
            _valid_actions(5.0, 100.0),
            pot_before_action=27.0,
            street="turn",
            preflop_raise_count_before_action=0,
            pre_stack_bb=100.0,
            apply_hero_preflop_sizing=False,
            open_size_bb=2.5,
            threebet_size_bb=9.0,
            fourbet_size_bb=22.0,
            requested_size_bb=9.5,
        )

        self.assertEqual(action.action_type, PlayerAction.BET)
        self.assertTrue(math.isclose(float(action.bet_amount), 9.5, rel_tol=0, abs_tol=1e-9))
        self.assertTrue(math.isclose(float(metadata.get("target_bet_amount")), 9.5, rel_tol=0, abs_tol=1e-9))

    def test_requested_bet75_size_is_respected_when_legal(self):
        table = SimpleNamespace(bet_to_match=0.0)
        player = SimpleNamespace(bet_this_street=0.0)
        action, metadata = build_action_from_intent(
            "bet75",
            table,
            player,
            _valid_actions(1.0, 30.0),
            pot_before_action=40.0,
            street="river",
            preflop_raise_count_before_action=0,
            pre_stack_bb=26.8,
            apply_hero_preflop_sizing=False,
            open_size_bb=2.5,
            threebet_size_bb=9.0,
            fourbet_size_bb=22.0,
            requested_size_bb=26.8,
        )

        self.assertEqual(action.action_type, PlayerAction.BET)
        self.assertTrue(math.isclose(float(action.bet_amount), 26.8, rel_tol=0, abs_tol=1e-9))
        self.assertTrue(math.isclose(float(metadata.get("target_bet_amount")), 26.8, rel_tol=0, abs_tol=1e-9))
        self.assertFalse(bool(metadata.get("clipped_to_bounds")))

    def test_preflop_5bet_node_converts_raise_to_allin(self):
        table = SimpleNamespace(bet_to_match=22.0)
        player = SimpleNamespace(bet_this_street=0.0)
        action, metadata = build_action_from_intent(
            "raise",
            table,
            player,
            _valid_actions(22.0, 91.0),
            pot_before_action=31.5,
            street="preflop",
            preflop_raise_count_before_action=3,
            pre_stack_bb=100.0,
            apply_hero_preflop_sizing=True,
            open_size_bb=2.5,
            threebet_size_bb=9.0,
            fourbet_size_bb=22.0,
        )

        self.assertEqual(action.action_type, PlayerAction.BET)
        self.assertTrue(math.isclose(float(action.bet_amount), 91.0, rel_tol=0, abs_tol=1e-9))
        self.assertTrue(bool(metadata.get("is_allin")))
        self.assertEqual(metadata.get("preflop_node"), "5BET")

    def test_exact_hit_postflop_raise_uses_requested_size_when_relaxed(self):
        table = SimpleNamespace(bet_to_match=30.0)
        player = SimpleNamespace(bet_this_street=6.0)
        action, metadata = build_action_from_intent(
            "raise",
            table,
            player,
            _valid_actions(31.0, 100.0),
            pot_before_action=20.0,
            street="flop",
            preflop_raise_count_before_action=0,
            pre_stack_bb=100.0,
            apply_hero_preflop_sizing=False,
            open_size_bb=2.5,
            threebet_size_bb=9.0,
            fourbet_size_bb=22.0,
            requested_size_bb=80.0,
            exact_hit_policy=True,
            preserve_exact_postflop_size=True,
            relax_exact_postflop_raise_guardrail=True,
        )

        self.assertEqual(action.action_type, PlayerAction.BET)
        self.assertTrue(math.isclose(float(action.bet_amount), 80.0, rel_tol=0, abs_tol=1e-9))
        self.assertTrue(bool(metadata.get("exact_requested_size_used")))
        self.assertTrue(bool(metadata.get("postflop_raise_guardrail_relaxed")))
        self.assertFalse(bool(metadata.get("postflop_raise_clamped")))

    def test_exact_hit_postflop_raise_clamps_without_relaxed_guardrail(self):
        table = SimpleNamespace(bet_to_match=30.0)
        player = SimpleNamespace(bet_this_street=6.0)
        action, metadata = build_action_from_intent(
            "raise",
            table,
            player,
            _valid_actions(31.0, 100.0),
            pot_before_action=20.0,
            street="flop",
            preflop_raise_count_before_action=0,
            pre_stack_bb=100.0,
            apply_hero_preflop_sizing=False,
            open_size_bb=2.5,
            threebet_size_bb=9.0,
            fourbet_size_bb=22.0,
            requested_size_bb=80.0,
            exact_hit_policy=True,
            preserve_exact_postflop_size=True,
            relax_exact_postflop_raise_guardrail=False,
        )

        self.assertEqual(action.action_type, PlayerAction.BET)
        # With guardrail enabled and to_call>0, raise gets capped to pot/stack guardrail.
        self.assertTrue(math.isclose(float(action.bet_amount), 49.0, rel_tol=0, abs_tol=1e-9))
        self.assertTrue(bool(metadata.get("exact_requested_size_used")))
        self.assertFalse(bool(metadata.get("postflop_raise_guardrail_relaxed")))
        self.assertTrue(bool(metadata.get("postflop_raise_clamped")))


if __name__ == "__main__":
    unittest.main()
