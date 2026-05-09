import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.run_smoke_eval import (
    default_jobs,
    extract_metrics,
    prune_runs,
    resolve_policy_cmd,
    resolve_policy_source,
    resolve_hero_mode,
    select_runs_to_keep,
)


class RunSmokeEvalTests(unittest.TestCase):
    def test_default_jobs_uses_eval_jobs_env(self):
        with patch.dict(os.environ, {"EVAL_JOBS": "7"}, clear=False):
            self.assertEqual(default_jobs(), 7)

    def test_default_jobs_uses_cpu_minus_two_when_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("eval.run_smoke_eval.os.cpu_count", return_value=12):
                self.assertEqual(default_jobs(), 10)

    def test_extract_metrics_prefers_net_and_includes_store_counts(self):
        payload = {
            "summary": {
                "raw": {"bb_per_100": -1.0},
                "net": {"bb_per_100": 2.5},
                "fallback_count": 7,
                "status_counts": {"missing_fields": 3},
                "diagnostics": {
                    "strategy_store": {
                        "store_hit_count": 11,
                        "store_miss_count": 19,
                        "quality_exact_hits": 9,
                        "quality_approx_hits": 2,
                    }
                },
            }
        }

        metrics = extract_metrics(
            payload,
            run_id="run_a",
            mode="smoke",
            hands=1000,
            seed=42,
            timestamp_utc="2026-02-19T00:00:00+00:00",
        )

        self.assertEqual(metrics["eval_mode"], "smoke")
        self.assertEqual(metrics["bb_per_100"], 2.5)
        self.assertEqual(metrics["fallback_count"], 7)
        self.assertEqual(metrics["missing_fields_count"], 3)
        self.assertFalse(metrics["store_enabled"])
        self.assertEqual(metrics["store_hit_count"], 11)
        self.assertEqual(metrics["store_miss_count"], 19)
        self.assertEqual(metrics["store_quality_exact_hits"], 9)
        self.assertEqual(metrics["store_quality_approx_hits"], 2)

    def test_extract_metrics_sets_store_enabled_true(self):
        payload = {
            "summary": {
                "raw": {"bb_per_100": 1.0},
                "diagnostics": {
                    "strategy_store": {
                        "enabled": True,
                        "store_hit_count": 5,
                        "store_miss_count": 7,
                    }
                },
            }
        }
        metrics = extract_metrics(
            payload,
            run_id="run_b",
            mode="confirm",
            hands=2000,
            seed=7,
            timestamp_utc="2026-02-19T00:00:00+00:00",
        )
        self.assertEqual(metrics["eval_mode"], "confirm")
        self.assertTrue(metrics["store_enabled"])

    def test_resolve_hero_mode_auto_worker_when_sampling_enabled(self):
        mode, source = resolve_hero_mode(
            cli_value=None,
            env_value=None,
            preflop_selection_mode="sample",
            policy_selection_mode="hybrid",
        )
        self.assertEqual(mode, "worker")
        self.assertEqual(source, "auto_worker")

    def test_resolve_hero_mode_auto_argmax_when_both_deterministic(self):
        mode, source = resolve_hero_mode(
            cli_value=None,
            env_value=None,
            preflop_selection_mode="argmax",
            policy_selection_mode="argmax",
        )
        self.assertEqual(mode, "argmax")
        self.assertEqual(source, "auto_argmax")

    def test_resolve_hero_mode_cli_override(self):
        mode, source = resolve_hero_mode(
            cli_value="sample",
            env_value="argmax",
            preflop_selection_mode="argmax",
            policy_selection_mode="argmax",
        )
        self.assertEqual(mode, "sample")
        self.assertEqual(source, "cli")

    def test_resolve_policy_source_prefers_cli_then_env_then_default(self):
        self.assertEqual(resolve_policy_source("blueprint", None), "blueprint")
        self.assertEqual(resolve_policy_source(None, "node"), "node")
        self.assertEqual(resolve_policy_source(None, None), "node")

    def test_resolve_policy_cmd_uses_blueprint_default_when_requested(self):
        cmd, source = resolve_policy_cmd(
            cli_value=None,
            env_value=None,
            policy_source="blueprint",
        )
        self.assertEqual(
            cmd,
            "cargo run --manifest-path solver/Cargo.toml -p player --bin blueprint_policy_worker --release",
        )
        self.assertEqual(source, "policy_source_blueprint")

    def test_resolve_policy_cmd_prefers_cli_over_env(self):
        cmd, source = resolve_policy_cmd(
            cli_value="custom_worker --foo",
            env_value="env_worker --bar",
            policy_source="node",
        )
        self.assertEqual(cmd, "custom_worker --foo")
        self.assertEqual(source, "cli")

    def test_retention_keeps_latest_best_and_last_n(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            rows = [
                ("20260101_000000", -10.0),
                ("20260101_000100", 4.0),
                ("20260101_000200", 2.0),
                ("20260101_000300", 1.0),
                ("20260101_000400", -1.0),
            ]
            for run_id, bb in rows:
                run_dir = runs_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "report_smoke.json").write_text("{}", encoding="utf-8")
                (run_dir / "metrics.json").write_text(
                    json.dumps({"bb_per_100": bb}),
                    encoding="utf-8",
                )

            run_dirs = sorted([entry for entry in runs_dir.iterdir() if entry.is_dir()])
            keep = select_runs_to_keep(run_dirs, keep_last=3)
            keep_names = {path.name for path in keep}

            # keep_last=3 => 000200, 000300, 000400
            # plus best => 000100
            self.assertEqual(
                keep_names,
                {
                    "20260101_000100",
                    "20260101_000200",
                    "20260101_000300",
                    "20260101_000400",
                },
            )

            deleted = prune_runs(runs_dir=runs_dir, keep_last=3)
            self.assertEqual(deleted, ["20260101_000000"])
            remaining = sorted(path.name for path in runs_dir.iterdir() if path.is_dir())
            self.assertEqual(
                remaining,
                [
                    "20260101_000100",
                    "20260101_000200",
                    "20260101_000300",
                    "20260101_000400",
                ],
            )

    def test_retention_prunes_old_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            complete = runs_dir / "20260101_000100"
            complete.mkdir(parents=True, exist_ok=True)
            (complete / "report_smoke.json").write_text("{}", encoding="utf-8")
            (complete / "metrics.json").write_text(
                json.dumps({"bb_per_100": 1.0}),
                encoding="utf-8",
            )

            stale_incomplete = runs_dir / "20260101_000101"
            stale_incomplete.mkdir(parents=True, exist_ok=True)
            (stale_incomplete / "temp.txt").write_text("x", encoding="utf-8")

            latest_incomplete = runs_dir / "20260101_000102"
            latest_incomplete.mkdir(parents=True, exist_ok=True)

            deleted = prune_runs(runs_dir=runs_dir, keep_last=1)
            self.assertEqual(deleted, ["20260101_000101"])
            remaining = sorted(path.name for path in runs_dir.iterdir() if path.is_dir())
            self.assertEqual(remaining, ["20260101_000100", "20260101_000102"])


if __name__ == "__main__":
    unittest.main()
