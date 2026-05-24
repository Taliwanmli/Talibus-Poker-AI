import unittest
from pathlib import Path
from unittest.mock import patch

from eval.run_league_parallel import launch_workers, merge_report_payloads


class MergeReportPayloadsTests(unittest.TestCase):
    def test_weighted_bb_and_strategy_store_aggregation(self):
        part_a = {
            "config": {"hands": 100, "seed": 42},
            "summary": {
                "raw": {"hands": 100, "bb_per_100": 10.0, "total_bb": 10.0},
                "net": {"hands": 100, "bb_per_100": 8.0, "total_bb": 8.0},
                "fallback_count": 2,
                "status_counts": {"ok": 100, "unavailable": 0},
                "diagnostics": {
                    "street_decisions": {"preflop": 100, "flop": 30},
                    "strategy_store": {
                        "enabled": True,
                        "file_path": "eval/store_a.jsonl",
                        "node_count": 10,
                        "store_hit_count": 30,
                        "store_miss_count": 70,
                        "store_hit_rate": 0.3,
                        "exact_hits": 20,
                        "exact_misses": 80,
                        "fuzzy_hits": 10,
                        "fuzzy_misses": 70,
                        "quality_exact_hits": 28,
                        "quality_approx_hits": 2,
                        "top_missing_spot_keys": [
                            {"spot_key": "A", "count": 2},
                            {"spot_key": "B", "count": 1},
                        ],
                        "top_fuzzy_miss_groups": [
                            {"group": "G1", "count": 4},
                            {"group": "G2", "count": 2},
                        ],
                    },
                },
            },
        }
        part_b = {
            "config": {"hands": 300, "seed": 1_000_045},
            "summary": {
                "raw": {"hands": 300, "bb_per_100": -20.0, "total_bb": -60.0},
                "net": {"hands": 300, "bb_per_100": -10.0, "total_bb": -30.0},
                "fallback_count": 3,
                "status_counts": {"ok": 290, "unavailable": 10},
                "diagnostics": {
                    "street_decisions": {"preflop": 300, "flop": 120},
                    "strategy_store": {
                        "enabled": True,
                        "file_path": "eval/store_b.jsonl",
                        "node_count": 20,
                        "store_hit_count": 20,
                        "store_miss_count": 280,
                        "store_hit_rate": 20 / 300.0,
                        "exact_hits": 5,
                        "exact_misses": 295,
                        "fuzzy_hits": 15,
                        "fuzzy_misses": 280,
                        "quality_exact_hits": 12,
                        "quality_approx_hits": 8,
                        "top_missing_spot_keys": [
                            {"spot_key": "A", "count": 3},
                            {"spot_key": "C", "count": 5},
                        ],
                        "top_fuzzy_miss_groups": [
                            {"group": "G1", "count": 3},
                            {"group": "G3", "count": 7},
                        ],
                    },
                },
            },
        }

        merged = merge_report_payloads([part_a, part_b], top_missing_n=2)
        raw = merged["summary"]["raw"]
        strategy_store = merged["summary"]["diagnostics"]["strategy_store"]

        self.assertEqual(raw["hands"], 400)
        # Weighted by hand count:
        # (10 * 100 + (-20) * 300) / 400 = -12.5
        self.assertAlmostEqual(float(raw["bb_per_100"]), -12.5, places=9)

        self.assertEqual(strategy_store["store_hit_count"], 50)
        self.assertEqual(strategy_store["store_miss_count"], 350)
        self.assertAlmostEqual(float(strategy_store["store_hit_rate"]), 0.125, places=9)
        self.assertEqual(strategy_store["exact_hits"], 25)
        self.assertEqual(strategy_store["exact_misses"], 375)
        self.assertEqual(strategy_store["fuzzy_hits"], 25)
        self.assertEqual(strategy_store["fuzzy_misses"], 350)
        self.assertEqual(strategy_store["quality_exact_hits"], 40)
        self.assertEqual(strategy_store["quality_approx_hits"], 10)
        self.assertAlmostEqual(float(strategy_store["quality_exact_hit_rate"]), 0.8, places=9)
        self.assertAlmostEqual(float(strategy_store["fuzzy_hit_rate"]), 25 / 375.0, places=9)
        self.assertAlmostEqual(float(strategy_store["overall_hit_rate"]), 0.125, places=9)

        top_missing = strategy_store["top_missing_spot_keys"]
        self.assertEqual(len(top_missing), 2)
        self.assertEqual(top_missing[0]["spot_key"], "A")
        self.assertEqual(top_missing[0]["count"], 5)
        self.assertEqual(top_missing[1]["spot_key"], "C")
        self.assertEqual(top_missing[1]["count"], 5)

        top_fuzzy_miss = strategy_store["top_fuzzy_miss_groups"]
        self.assertEqual(len(top_fuzzy_miss), 2)
        self.assertEqual(top_fuzzy_miss[0]["group"], "G1")
        self.assertEqual(top_fuzzy_miss[0]["count"], 7)
        self.assertEqual(top_fuzzy_miss[1]["group"], "G3")
        self.assertEqual(top_fuzzy_miss[1]["count"], 7)

    def test_merge_aggregates_fidelity_mode_and_translation_counters(self):
        part_a = {
            "config": {"hands": 50, "seed": 7, "hero_mode": "worker"},
            "summary": {
                "raw": {"hands": 50, "bb_per_100": 0.0, "total_bb": 0.0},
                "net": {"hands": 50, "bb_per_100": 0.0, "total_bb": 0.0},
                "fallback_count": 0,
                "status_counts": {"ok": 50},
                "diagnostics": {
                    "street_decisions": {"preflop": 50},
                    "street_ok": {"preflop": 50},
                    "street_unavailable": {"preflop": 3},
                    "street_fallbacks": {"preflop": 2},
                    "mode_config": {
                        "hero_mode_effective": "worker",
                        "preflop_selection_mode_effective": "sample",
                        "postflop_selection_mode_effective": "hybrid",
                    },
                    "policy_execution_fidelity": {
                        "hero_mode_effective": "worker",
                        "selection_source_counts": {
                            "worker_executed": 30,
                            "evaluator_mix_reselection": 5,
                        },
                        "selection_source_by_street": {
                            "preflop": {"worker_executed": 30}
                        },
                        "selection_source_by_street_exact_hit": {
                            "preflop": {"worker_executed": 10}
                        },
                        "selection_source_by_street_non_exact_hit": {
                            "preflop": {"worker_executed": 20}
                        },
                        "chosen_vs_executed_match_count": 28,
                        "chosen_vs_executed_total": 30,
                        "chosen_vs_recommended_match_count": 29,
                        "chosen_vs_recommended_total": 30,
                        "chosen_vs_argmax_match_count": 15,
                        "chosen_vs_argmax_total": 30,
                        "exact_hit_and_translated_count": 4,
                        "exact_hit_and_unmodified_count": 6,
                        "exact_hit_translation_total": 10,
                    },
                    "action_translation_audit": {
                        "translation_adjusted_count": 7,
                        "preflop_raise_clamped_count": 2,
                        "postflop_raise_clamped_count": 1,
                    },
                    "strategy_store": {
                        "enabled": True,
                        "store_hit_count": 20,
                        "store_miss_count": 30,
                        "quality_exact_hits": 18,
                        "quality_approx_hits": 2,
                        "store_load_ms": 15.0,
                        "store_file_size_bytes": 1000,
                    },
                },
            },
        }
        part_b = {
            "config": {"hands": 50, "seed": 8, "hero_mode": "worker"},
            "summary": {
                "raw": {"hands": 50, "bb_per_100": 0.0, "total_bb": 0.0},
                "net": {"hands": 50, "bb_per_100": 0.0, "total_bb": 0.0},
                "fallback_count": 0,
                "status_counts": {"ok": 50},
                "diagnostics": {
                    "street_decisions": {"preflop": 50},
                    "street_ok": {"preflop": 50},
                    "street_unavailable": {"preflop": 1},
                    "street_fallbacks": {"preflop": 1},
                    "policy_execution_fidelity": {
                        "hero_mode_effective": "worker",
                        "selection_source_counts": {"worker_recommended": 4},
                        "selection_source_by_street": {
                            "preflop": {"worker_recommended": 4}
                        },
                        "selection_source_by_street_exact_hit": {
                            "preflop": {"worker_recommended": 1}
                        },
                        "selection_source_by_street_non_exact_hit": {
                            "preflop": {"worker_recommended": 3}
                        },
                        "chosen_vs_executed_match_count": 3,
                        "chosen_vs_executed_total": 4,
                        "chosen_vs_recommended_match_count": 4,
                        "chosen_vs_recommended_total": 4,
                        "chosen_vs_argmax_match_count": 1,
                        "chosen_vs_argmax_total": 4,
                        "exact_hit_and_translated_count": 2,
                        "exact_hit_and_unmodified_count": 3,
                        "exact_hit_translation_total": 5,
                    },
                    "action_translation_audit": {
                        "translation_adjusted_count": 2,
                        "preflop_raise_clamped_count": 1,
                        "postflop_raise_clamped_count": 3,
                    },
                    "strategy_store": {
                        "enabled": True,
                        "store_hit_count": 25,
                        "store_miss_count": 25,
                        "quality_exact_hits": 22,
                        "quality_approx_hits": 3,
                        "store_load_ms": 11.0,
                        "store_file_size_bytes": 1200,
                    },
                },
            },
        }

        merged = merge_report_payloads([part_a, part_b], top_missing_n=5)
        diagnostics = merged["summary"]["diagnostics"]
        fidelity = diagnostics["policy_execution_fidelity"]
        audit = diagnostics["action_translation_audit"]
        preflop_fallback = diagnostics["preflop_fallback_unavailable"]
        strategy_store = diagnostics["strategy_store"]

        self.assertEqual(fidelity["selection_source_counts"]["worker_executed"], 30)
        self.assertEqual(fidelity["selection_source_counts"]["worker_recommended"], 4)
        self.assertEqual(fidelity["chosen_vs_executed_match_count"], 31)
        self.assertEqual(fidelity["chosen_vs_executed_total"], 34)
        self.assertEqual(fidelity["exact_hit_and_unmodified_count"], 9)
        self.assertEqual(fidelity["exact_hit_translation_total"], 15)
        self.assertEqual(preflop_fallback["unavailable"], 4)
        self.assertEqual(preflop_fallback["fallbacks"], 3)
        self.assertEqual(audit["translation_adjusted_count"], 9)
        self.assertEqual(audit["raise_clamped_count"], 7)
        self.assertAlmostEqual(float(strategy_store["store_load_ms"]), 13.0, places=6)
        self.assertEqual(int(strategy_store["store_file_size_bytes"]), 1200)
        self.assertEqual(diagnostics["mode_config"]["hero_mode_effective"], "worker")


class LaunchWorkersTests(unittest.TestCase):
    def test_launch_workers_forwards_policy_cmd(self):
        captured: list[list[str]] = []

        class _Proc:
            def wait(self) -> int:
                return 0

        def _fake_popen(cmd, **_kwargs):  # type: ignore[no-untyped-def]
            captured.append(list(cmd))
            return _Proc()

        repo_root = Path("C:/repo")
        with patch("eval.run_league_parallel.subprocess.Popen", side_effect=_fake_popen):
            launch_workers(
                repo_root=repo_root,
                hands_per_worker=[100],
                base_seed=42,
                trace_hands=0,
                report_json=Path("report.json"),
                quiet=True,
                hero_mode="worker",
                policy_cmd="cargo run -p player --bin blueprint_policy_worker --release",
            )

        self.assertEqual(len(captured), 1)
        cmd = captured[0]
        self.assertIn("--policy-cmd", cmd)
        policy_idx = cmd.index("--policy-cmd")
        self.assertEqual(
            cmd[policy_idx + 1],
            "cargo run -p player --bin blueprint_policy_worker --release",
        )


if __name__ == "__main__":
    unittest.main()
