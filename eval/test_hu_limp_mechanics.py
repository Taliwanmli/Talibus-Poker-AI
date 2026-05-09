#!/usr/bin/env python3
"""Regression tests for HU preflop limp mechanics in the eval compat patch."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pokerenv.common import Action, GameState, PlayerAction
from pokerenv.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_league as rl  # noqa: E402


def _player_by_identifier(table: Table, identifier: int):
    for player in table.players:
        if int(player.identifier) == int(identifier):
            return player
    raise ValueError(f"player {identifier} not found")


class TestHuLimpMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rl.patch_pokerenv_compat()

    def _new_table(self, seed: int) -> Table:
        table = Table(2, stack_low=100, stack_high=101, hand_history_location=None)
        table.seed(seed)
        return table

    def test_sb_limp_keeps_preflop_and_gives_bb_node(self) -> None:
        table = self._new_table(seed=42)
        obs = table.reset()
        sb_id = int(obs[0])
        sb_player = _player_by_identifier(table, sb_id)
        sb_valid = table._get_valid_actions(sb_player)
        self.assertIn(PlayerAction.CALL, sb_valid["actions_list"])

        obs_after_limp, _rewards, done, _info = table.step(Action(PlayerAction.CALL, 0))
        self.assertFalse(done)
        self.assertEqual(table.street, GameState.PREFLOP)

        bb_id = int(obs_after_limp[0])
        self.assertNotEqual(bb_id, sb_id)
        bb_player = _player_by_identifier(table, bb_id)
        bb_valid = table._get_valid_actions(bb_player)
        self.assertIn(PlayerAction.CHECK, bb_valid["actions_list"])
        self.assertIn(PlayerAction.BET, bb_valid["actions_list"])

        table.step(Action(PlayerAction.CHECK, 0))
        self.assertEqual(table.street, GameState.FLOP)

    def test_sb_limp_then_bb_bet_gives_sb_response_node(self) -> None:
        table = self._new_table(seed=42)
        obs = table.reset()
        sb_id = int(obs[0])
        obs_after_limp, _rewards, done, _info = table.step(Action(PlayerAction.CALL, 0))
        self.assertFalse(done)
        self.assertEqual(table.street, GameState.PREFLOP)

        bb_id = int(obs_after_limp[0])
        bb_player = _player_by_identifier(table, bb_id)
        bb_valid = table._get_valid_actions(bb_player)
        self.assertIn(PlayerAction.BET, bb_valid["actions_list"])
        min_bet = float(bb_valid["bet_range"][0])
        obs_after_bb_bet, _rewards, done, _info = table.step(Action(PlayerAction.BET, min_bet))
        self.assertFalse(done)
        self.assertEqual(table.street, GameState.PREFLOP)
        self.assertEqual(int(obs_after_bb_bet[0]), sb_id)

        sb_player = _player_by_identifier(table, sb_id)
        sb_valid = table._get_valid_actions(sb_player)
        self.assertIn(PlayerAction.FOLD, sb_valid["actions_list"])
        self.assertIn(PlayerAction.CALL, sb_valid["actions_list"])

    def test_forced_limp_policy_produces_bb_preflop_actions(self) -> None:
        table = self._new_table(seed=42)
        bb_preflop_actions = 0
        for _ in range(100):
            obs = table.reset()
            sb_id = int(obs[0])
            sb_player = _player_by_identifier(table, sb_id)
            sb_valid = table._get_valid_actions(sb_player)
            self.assertIn(PlayerAction.CALL, sb_valid["actions_list"])

            obs_after_limp, _rewards, done, _info = table.step(Action(PlayerAction.CALL, 0))
            self.assertFalse(done)
            self.assertEqual(table.street, GameState.PREFLOP)

            bb_id = int(obs_after_limp[0])
            bb_player = _player_by_identifier(table, bb_id)
            bb_valid = table._get_valid_actions(bb_player)
            self.assertIn(PlayerAction.CHECK, bb_valid["actions_list"])
            self.assertIn(PlayerAction.BET, bb_valid["actions_list"])
            bb_preflop_actions += 1

            table.step(Action(PlayerAction.CHECK, 0))
            self.assertEqual(table.street, GameState.FLOP)

        self.assertGreater(bb_preflop_actions, 0)


if __name__ == "__main__":
    unittest.main()
