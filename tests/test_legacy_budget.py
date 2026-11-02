"""Budget parsing uses only the standard library, on both supported interpreters."""

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import run_algorithm
from src.legacy_cli import resolve_training_budget


ROOT = Path(__file__).resolve().parents[1]


def resolve(arguments):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--episode_length", type=int, default=80)
    parser.add_argument("--num_env_steps", type=int, default=10e6)
    parsed = parser.parse_args(arguments)
    return resolve_training_budget(arguments, parsed.num_episodes,
                                   parsed.episode_length, parsed.num_env_steps)


class TrainingBudgetTests(unittest.TestCase):
    def test_default_budget_preserves_existing_episode_count(self):
        budget = resolve([])
        self.assertEqual((budget["num_episodes"], budget["num_env_steps"]), (1000, 80000))
        self.assertIsNone(budget["requested_num_env_steps"])
        self.assertEqual(budget["training_step_unit"], "joint_slot")

    def test_episode_only_budget(self):
        budget = resolve(["--num_episodes", "2"])
        self.assertEqual(budget["effective_num_env_steps"], 160)
        self.assertEqual(budget["training_budget_source"], "episodes")

    def test_steps_only_derives_complete_episodes(self):
        budget = resolve(["--num_env_steps", "160"])
        self.assertEqual(budget["num_episodes"], 2)
        self.assertEqual(budget["training_budget_source"], "steps")

    def test_matching_explicit_budgets(self):
        budget = resolve(["--num_env_steps", "160", "--num_episodes", "2"])
        self.assertEqual(budget["training_budget_source"], "steps_and_episodes")
        self.assertEqual(budget["requested_num_episodes"], 2)

    def test_conflicting_explicit_budgets_fail(self):
        with self.assertRaisesRegex(ValueError, "Conflicting training budget"):
            resolve(["--num_env_steps", "160", "--num_episodes", "3"])

    def test_partial_and_too_short_episode_budgets_fail(self):
        for steps in (1, 79, 81, 200):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "multiple"):
                resolve(["--num_env_steps", str(steps)])

    def test_nonpositive_budgets_fail(self):
        for name in ("--num_env_steps", "--num_episodes", "--episode_length"):
            for value in ("0", "-1"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "positive"):
                    resolve([name, value])

    def test_equals_spelling_and_last_duplicate_value(self):
        budget = resolve(["--num_env_steps=80", "--num_env_steps=160", "--num_episodes=1", "--num_episodes=2"])
        self.assertEqual((budget["num_episodes"], budget["num_env_steps"]), (2, 160))

    def test_launcher_does_not_inject_a_phantom_budget(self):
        defaults = run_algorithm.with_tg_mappo_defaults([])
        self.assertNotIn("--num_env_steps", defaults)
        explicit = run_algorithm.with_tg_mappo_defaults(["--num_env_steps=160"])
        self.assertEqual(explicit.count("--num_env_steps=160"), 1)
        self.assertNotIn("--num_env_steps", explicit)
        self.assertNotIn("--num_env_steps 200", (ROOT / "run_training.cmd").read_text())

    def test_real_cli_rejects_conflicts_and_abbreviations_before_environment_import(self):
        with tempfile.TemporaryDirectory() as temp:
            for extra, message in ((["--num_env_steps=160", "--num_episodes=3"], "Conflicting training budget"),
                                   (["--num_env_s=160"], "unrecognized arguments")):
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
                result = subprocess.run([sys.executable, "-m", "buoy_network.main",
                                         "--grid_size", "5", "--num_agents", "25", "--episode_length", "80",
                                         "--output_root", str(Path(temp) / "must_not_exist")] + extra,
                                        cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertNotIn("gym", result.stderr)
                self.assertFalse((Path(temp) / "must_not_exist").exists())


if __name__ == "__main__":
    unittest.main()
