"""Stage 3: start fresh Python processes and inspect their actual output files."""

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.buoy_network.research.data.storage import DatasetStore

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "tests.research_plugin_fixture"


class ProcessIntegrationTests(unittest.TestCase):
    def run_cli(self, arguments, expected=0, module=False):
        prefix = [sys.executable, "-m", "src.buoy_network.research"] if module else [
            sys.executable, str(ROOT / "run_research.py")
        ]
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(prefix + arguments, cwd=ROOT, env=environment,
                                capture_output=True, text=True, encoding="utf-8", timeout=45)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout) if expected == 0 and result.stdout.startswith("{") else result

    def test_help_module_entry_and_legacy_algorithm_list(self):
        self.assertIn("collect", self.run_cli(["--help"]).stdout)
        self.assertIn("implemented", self.run_cli(["describe"], module=True))
        result = subprocess.run([sys.executable, str(ROOT / "run_algorithm.py"), "--list-algorithms"],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tg-mappo", result.stdout)

    def test_real_graph_policy_training_and_frozen_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            common = ["--policy", "graph_mappo", "--policy-hidden-size", "16", "--device", "cpu",
                      "--grid-size", "3", "--num-moving-buoys", "0", "--task-horizon", "8",
                      "--task-ttl", "8", "--rollout-length", "4", "--output-root", temp]
            training = self.run_cli(["train-policy", "--episodes", "2", "--ppo-epochs", "1"] + common)
            metrics = training["report"]["metrics"]
            self.assertGreater(metrics["actor_parameter_delta"], 0)
            self.assertGreater(metrics["critic_parameter_delta"], 0)
            self.assertGreater(metrics["optimizer_steps"], 0)
            checkpoint = Path(training["report"]["checkpoint"])
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            args = ["evaluate", "--policy-checkpoint", str(checkpoint), "--episodes", "2", "--seed", "71"] + common
            first, second = self.run_cli(args), self.run_cli(args)
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertTrue(first["model_versions"]["policy"].endswith(":greedy"))
            self.assertEqual(digest, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertFalse((Path(first["output"]) / "data").exists())

    def test_graph_checkpoint_mismatch_is_clean_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            common = ["--policy", "graph_mappo", "--policy-hidden-size", "16", "--device", "cpu",
                      "--grid-size", "3", "--num-moving-buoys", "0", "--task-horizon", "2",
                      "--task-ttl", "2", "--rollout-length", "2", "--episodes", "1", "--ppo-epochs", "1"]
            training = self.run_cli(["train-policy", "--output-root", str(Path(temp) / "train")] + common)
            destination = Path(temp) / "must_not_exist"
            result = self.run_cli(["evaluate", "--policy-checkpoint", training["report"]["checkpoint"],
                "--output-root", str(destination)] + common + ["--policy-hidden-size", "32"], 2)
            self.assertIn("Incompatible policy architecture", result.stderr)
            self.assertFalse(destination.exists())

    def test_collect_25_nodes_three_seeds_and_audit_saved_data(self):
        with tempfile.TemporaryDirectory() as temp:
            report = self.run_cli(["collect", "--config", "configs/research_v1.json", "--device", "cpu",
                                   "--episodes", "3", "--seed", "41", "--output-root", temp])
            root = Path(report["output"])
            store = DatasetStore(root / "data")
            manifest = store.manifest()
            self.assertEqual(manifest["num_nodes"], 25)
            self.assertEqual(set(manifest["episode_seeds"].values()), {41, 42, 43})
            with (root / "research_metrics.csv").open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 3)
            for split in ("train", "validation", "test"):
                records = list(store.iter_rows(split))
                packets = list(store.iter_rows(split, packets=True))
                seed = records[0]["seed"]
                metric = next(row for row in metrics if int(row["seed"]) == seed)
                self.assertEqual(len(packets), 24)
                self.assertEqual(len(records), int(metric["environment_steps"]))
                self.assertEqual(records[0]["env_step"], 0)
                self.assertEqual(records[-1]["state_tp1"]["env_step"], len(records))
                self.assertTrue(records[-1]["env_terminated"])
                self.assertEqual(len(records[-1]["state_tp1"]["packets"]), 0)
                self.assertAlmostEqual(float(metric["task_return"]), sum(r["task_reward"] for r in records))
                for transition in records:
                    for local, action in zip(transition["obs_t"]["agents"],
                                             transition["action_batch"]["actions"]):
                        self.assertTrue(local["available_actions"][action])
                self.assertTrue(all(p["outcome"] is not None for p in packets))

    def test_evaluate_is_reproducible_and_does_not_archive_training_data(self):
        with tempfile.TemporaryDirectory() as temp:
            options = ["evaluate", "--device", "cpu", "--grid-size", "4", "--episodes", "3",
                       "--policy", "random", "--num-moving-buoys", "16", "--mobility-std", "0.3",
                       "--loss-probability", "0.4", "--task-horizon", "16", "--task-ttl", "12",
                       "--rollout-length", "3", "--output-root", temp]
            first, second = self.run_cli(options), self.run_cli(options)
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertFalse((Path(first["output"]) / "data").exists())

    def test_160_slot_isolated_task_continues_across_rollout_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            report = self.run_cli(["smoke", "--device", "cpu", "--grid-size", "3",
                                   "--num-moving-buoys", "0", "--comm-range", "0.01",
                                   "--task-horizon", "160", "--task-ttl", "320", "--rollout-length", "7",
                                   "--output-root", temp])
            store = DatasetStore(Path(report["output"]) / "data")
            records = list(store.iter_rows("train"))
            self.assertEqual(len(records), 160)
            for index, record in enumerate(records):
                self.assertEqual(record["env_step"], index)
                self.assertEqual(record["rollout_truncated"], (index + 1) % 7 == 0 and index < 159)
                self.assertEqual(record["env_terminated"], index == 159)
                self.assertTrue(all(action == 9 for action in record["action_batch"]["actions"]))
            self.assertTrue(all(p["end_reason"] == "task_deadline" for p in store.iter_rows("train", packets=True)))

    def test_36_node_moving_network_and_total_loss_network(self):
        with tempfile.TemporaryDirectory() as temp:
            report = self.run_cli(["smoke", "--device", "cpu", "--grid-size", "6",
                                   "--num-moving-buoys", "36", "--mobility-std", "0.4",
                                   "--loss-probability", "0.5", "--task-horizon", "24", "--task-ttl", "20",
                                   "--rollout-length", "5", "--output-root", temp])
            store = DatasetStore(Path(report["output"]) / "data")
            self.assertEqual(store.manifest()["num_nodes"], 36)
            first = next(store.iter_rows("train"))
            self.assertNotEqual(first["state_t"]["nodes"], first["state_tp1"]["nodes"])
            failed = self.run_cli(["smoke", "--device", "cpu", "--loss-probability", "1.0",
                                   "--task-ttl", "6", "--task-horizon", "20", "--output-root", temp])
            self.assertEqual(failed["metrics"][0]["delivery_rate"], 0.0)
            self.assertEqual(failed["metrics"][0]["drop_rate"], 1.0)
            self.assertIsNone(failed["metrics"][0]["successful_mean_delay_slots"])

    def test_collection_train_hooks_checkpoints_and_combined_models(self):
        # These trainers are explicitly non-learning fixtures: this tests wiring only.
        with tempfile.TemporaryDirectory() as temp:
            common = ["--device", "cpu", "--grid-size", "3", "--num-moving-buoys", "0",
                      "--task-horizon", "8", "--task-ttl", "8", "--rollout-length", "2",
                      "--output-root", temp]
            collected = self.run_cli(["collect", "--episodes", "3"] + common)
            dataset = str(Path(collected["output"]) / "data")
            wm = self.run_cli(["train-wm", "--plugin", PLUGIN, "--wm-trainer", "fixture_wm",
                               "--dataset", dataset] + common)
            pm = self.run_cli(["train-pm", "--plugin", PLUGIN, "--pm-trainer", "fixture_pm",
                               "--dataset", dataset] + common)
            for trained in (wm, pm):
                self.assertEqual(trained["report"]["metrics"]["test_fixture_only"], 1)
                self.assertGreater(trained["report"]["metrics"]["train_rows"], 0)
                self.assertTrue(Path(trained["report"]["checkpoint"]).is_file())
            models = ["--plugin", PLUGIN, "--use-world-model", "true", "--wm-backend", "fixture_wm",
                      "--wm-checkpoint", wm["report"]["checkpoint"], "--use-progress-reward", "true",
                      "--pm-backend", "fixture_pm", "--pm-checkpoint", pm["report"]["checkpoint"]]
            combined = self.run_cli(["smoke"] + common + models)
            self.assertEqual(combined["model_versions"]["world_model"], "fixture_wm_v1")
            self.assertEqual(combined["model_versions"]["progress_model"], "fixture_pm_v1")
            store = DatasetStore(Path(combined["output"]) / "data")
            records = list(store.iter_rows("train"))
            self.assertTrue(all(r["wm_features"]["model_version"] == "fixture_wm_v1" for r in records))
            self.assertTrue(any(r["shaping_reward"] != 0.0 for r in records))
            policy = self.run_cli(["train-policy", "--policy-trainer", "fixture_policy"] + common + models)
            self.assertGreater(policy["report"]["metrics"]["environment_steps"], 0)

    def test_unavailable_trainers_fail_cleanly_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "absent"
            for command in ("train-wm", "train-pm", "train-policy"):
                with self.subTest(command=command):
                    options = ["--policy-trainer", "not_implemented"] if command == "train-policy" else []
                    result = self.run_cli([command, "--device", "cpu", "--output-root", str(destination)] + options, 2)
                    self.assertIn("not implemented/registered", result.stderr)
                    self.assertFalse(destination.exists())

    def test_corrupt_dataset_and_checkpoint_fail_cleanly_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            dataset = base / "data"
            dataset.mkdir()
            (dataset / "data_manifest.json").write_text("[]", encoding="utf-8")
            destination = base / "absent"
            self.run_cli(["train-wm", "--plugin", PLUGIN, "--wm-trainer", "fixture_wm",
                          "--dataset", str(dataset), "--device", "cpu", "--output-root", str(destination)], 2)
            checkpoint = base / "bad.json"
            checkpoint.write_text('{"bad": true}', encoding="utf-8")
            self.run_cli(["smoke", "--plugin", PLUGIN, "--use-world-model", "true",
                          "--wm-backend", "fixture_wm", "--wm-checkpoint", str(checkpoint),
                          "--device", "cpu", "--output-root", str(destination)], 2)
            self.assertFalse(destination.exists())

    def test_existing_run_is_preserved_and_typo_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            args = ["smoke", "--device", "cpu", "--grid-size", "3", "--num-moving-buoys", "0",
                    "--output-root", temp, "--run-name", "same_name"]
            report = self.run_cli(args)
            path = Path(report["output"]) / "run_report.json"
            original = path.read_bytes()
            self.run_cli(args, 2)
            self.assertEqual(path.read_bytes(), original)
            self.run_cli(["check-config", "--device", "cpu", "--wm-horizn", "3"], 2)


if __name__ == "__main__":
    unittest.main()
