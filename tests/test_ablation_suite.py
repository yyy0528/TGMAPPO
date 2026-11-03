"""Protocol/aggregation/job resume tests; no real model fitting here."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.buoy_network.ablation_suite import validate, budget, jobs, summarize, snapshot, suite_lock, run_suite, atomic, read


def configuration(root):
    path = root/"artifact.json"
    path.write_text("{}", encoding="utf-8")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema": "original_tg_ablation_v1", "purpose": "pilot", "python": sys.executable,
        "groups": ["A", "B", "C", "D", "E", "F"], "train_seeds": [11,22,33], "eval_seeds": [7001],
        "train_episodes": 2, "eval_episodes": 2, "episode_length": 80, "ppo_epoch": 1, "num_mini_batch": 2,
        "num_moving_buoys": 5, "random_channel": True, "target_delivery_rate": .8, "target_window": 2,
        "job_timeout_seconds": 30, "wm_checkpoint": str(path), "wm_sha256": checksum,
        "pm_checkpoint": str(path), "pm_sha256": checksum, "auxiliary_reports": {key:str(path) for key in ("wm", "pm", "data")}}


class SuiteFunctions(unittest.TestCase):
    def test_budget_and_commands_keep_training_and_evaluation_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = validate(configuration(root))
            self.assertEqual(budget(config)["total_training_steps"], 2880)
            planned = jobs(config, root)
            self.assertEqual(len(planned), 36)
            for job in planned:
                args = job["base_command"]
                self.assertEqual("--tg_evaluate" in args, job["kind"] == "evaluation")
                self.assertEqual("--wm_feature_source" in args, job["group"] == "F")
                self.assertEqual("--actor_extra_width" in args, job["group"] == "E")

    def test_invalid_seeds_or_model_versions_fail_before_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            config = configuration(Path(temp))
            for changed in (dict(config, eval_seeds=[11]), dict(config, eval_seeds=[101]),
                            dict(config, train_seeds=[11,11,22]), dict(config, pm_sha256="wrong"),
                            dict(config, purpose="formal"), dict(config, num_mini_batch=3)):
                with self.assertRaises(ValueError):
                    validate(changed)

    def test_summary_uses_training_seeds_as_statistical_units(self):
        with tempfile.TemporaryDirectory() as temp:
            config = configuration(Path(temp))
            config["groups"], config["eval_seeds"] = ["A"], [7001,7002]
            planned = jobs(config, Path(temp))
            for job in planned:
                job["state"] = "completed"
                job["attempts"] = [{"seconds": 1.}]
            report = {"resources": {}, "profile": {}}
            def inspect(job, _):
                delivered = config["train_seeds"].index(job["train_seed"])+1
                if job["kind"] == "training":
                    return report, [{"delivery_rate": .8}]*2
                return report, [{"generated_packets": 4, "delivered_packets": delivered,
                    "success_hops": [2]*delivered, "success_delays": [3]*delivered, "retry_attempts": 1, "loop_transfers": 0}]
            with patch("src.buoy_network.ablation_suite.inspect_job", side_effect=inspect):
                result = summarize({"config": config, "jobs": planned})
            item = result["groups"][0]
            self.assertEqual(item["independent_training_n"], 3)
            self.assertEqual(item["metrics"]["delivery_rate"]["mean"], .5)
            self.assertEqual(item["metrics"]["delivery_rate"]["sample_std"], .25)
            self.assertEqual(result["by_training_seed"][0]["target_first_training_steps"], 160)
            planned[0]["state"] = "failed"
            with self.assertRaises(ValueError):
                summarize({"config": config, "jobs": planned})


class SuiteModules(unittest.TestCase):
    def test_snapshot_bytes_and_single_writer_lock(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/"results")) as temp:
            root = Path(temp)
            recorded = snapshot(root)
            for path, checksum in recorded["files_sha256"].items():
                self.assertEqual(hashlib.sha256((root/"source_snapshot"/path).read_bytes()).hexdigest(), checksum)
            self.assertIn("run_algorithm.py", recorded["files_sha256"])
            with suite_lock(root):
                with self.assertRaises(OSError):
                    with suite_lock(root):
                        self.fail("two writers acquired one suite")
            with suite_lock(root):
                pass

    def test_failed_job_resume_preserves_previous_logs(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/"results")) as temp:
            root = Path(temp)
            config = configuration(root)
            (root/"job_logs").mkdir()
            (root/"source_snapshot").mkdir()
            job = {"key": "failed_job", "kind": "training", "group": "A", "train_seed": 11,
                   "state": "pending", "attempts": [], "base_command": [sys.executable, "-c", "raise SystemExit(7)"]}
            atomic(root/"manifest.json", {"config": config, "source": {"files_sha256": {}}, "jobs": [job]})
            for count in (1,2):
                with self.assertRaisesRegex(ValueError, "Job failed"):
                    run_suite(root)
                manifest = read(root/"manifest.json")
                self.assertEqual(manifest["state"], "failed")
                self.assertEqual(len(manifest["jobs"][0]["attempts"]), count)
                self.assertEqual(manifest["jobs"][0]["attempts"][-1]["exit_code"], 7)
            self.assertEqual(len(list((root/"job_logs").glob("*.log"))), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
