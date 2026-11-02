"""Real legacy-environment data checks; run with the Python 3.6 dependencies."""

import contextlib
import io
import json
import random
import os
import subprocess
import sys
import tempfile
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
from test_legacy_targeted_fixes import make_environment, ScriptedAlgorithm, Runner
from buoy_network.transition_data import snapshot, plain, TransitionRecorder, routing_actions
from buoy_network.dataset import audit_run, iter_wm_samples, run_roots, read_json, read_rows
from src.legacy_cli import reject_research_options


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SnapshotFunctionTests(unittest.TestCase):
    def test_rule_actions_are_legal_and_use_a_private_rng(self):
        env = make_environment()
        env.packet_generation()
        before = random.getstate()
        for behavior in ("random", "geographic"):
            actions, logs = routing_actions(env, behavior, random.Random(41))
            again = routing_actions(env, behavior, random.Random(41))
            self.assertEqual((actions, logs), again)
            self.assertEqual(random.getstate(), before)
            for node, (action, logp) in enumerate(zip(actions[0], logs[0])):
                self.assertTrue(env.available_actions()[0, node, action[0]])
                if behavior == "geographic":
                    self.assertEqual(logp[0], 0)
        self.assertEqual(routing_actions(env, "geographic", random.Random(1))[0][0][18][0], 24)
        with self.assertRaises(ValueError):
            routing_actions(env, "unknown", random.Random(1))

    def test_only_explicit_data_options_pass_the_legacy_guard(self):
        reject_research_options(["--export_wm_data", "true", "--wm_data_split", "train",
                                 "--wm_export_max_steps", "100", "--wm_export_max_bytes", "10000",
                                 "--wm_export_max_packets", "100"])
        for option in ("--wm_checkpoint", "--pm_unknown", "--wm_unknown", "--use_world_model"):
            with self.assertRaises(ValueError):
                reject_research_options([option, "x"])

    def test_snapshot_is_detached_and_does_not_consume_rng(self):
        env = make_environment()
        env.packet_generation()
        before = random.getstate()
        state = snapshot(env, "episode", 0, np.zeros((25, 25)))
        self.assertEqual(random.getstate(), before)
        self.assertEqual(len(state["actor_inputs"]["obs"]), 25)
        self.assertEqual(len(state["actor_inputs"]["obs"][0]), 26)
        self.assertNotIn("lost_rate_truth", json.dumps(state["actor_inputs"]))
        env.buoys[0].position[0] = 999
        self.assertEqual(state["actor_inputs"]["node_features"][0][0], 0)
        self.assertEqual(plain(np.asarray([1, 2])), [1, 2])

    def test_explicit_scenario_seed_changes_channel_and_default_preserves_legacy(self):
        env = make_environment()
        default_rates = [node.lost_rate for node in env.buoys]
        env.all_args.scenario_seed = 100
        env.build_env()
        changed = [node.lost_rate for node in env.buoys]
        self.assertNotEqual(default_rates, changed)
        env.all_args.scenario_seed = None
        env.build_env()
        self.assertEqual(default_rates, [node.lost_rate for node in env.buoys])


class RecorderModuleTests(unittest.TestCase):
    def run_script(self, temp, script, enabled=True, loss=False, max_steps=20000):
        env = make_environment(temp, length=len(script))
        env.all_args.export_wm_data = enabled
        env.all_args.wm_export_max_steps = max_steps
        for node in env.buoys:
            node.random_channel = loss
            if loss:
                node.lost_rate_list = [1.0]
        algorithm = ScriptedAlgorithm(script)
        agents = [SimpleNamespace(buoy=node, RL=algorithm, step=0) for node in env.buoys]
        with contextlib.redirect_stdout(io.StringIO()):
            Runner(agents, env).run()
        return Path(temp) / "moves_0" / "wm_data"

    def test_wait_is_not_a_failed_attempt_and_unresolved_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_script(temp, [{}, {}])
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(len(rows(root / "snapshots.jsonl")), 3)
            records = rows(root / "transitions.jsonl")
            self.assertFalse(records[0]["env_terminated"])
            self.assertTrue(records[1]["env_terminated"])
            self.assertTrue(all(e["ack"] is None and not e["attempted"] for e in records[0]["events"]))
            traces = rows(root / "packet_traces.jsonl")
            self.assertEqual(len(traces), 24)
            self.assertTrue(all(t["outcome"] is None and t["censored"] for t in traces))
            self.assertTrue(all(len(t["prefix"]) == 3 for t in traces))

    def test_actual_delivery_is_archived_before_legacy_packet_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_script(temp, [{18: 24}, {}])
            records = rows(root / "transitions.jsonl")
            event = records[0]["events"][18]
            self.assertTrue(event["ack"])
            self.assertTrue(event["delivered_this_step"])
            self.assertEqual(event["queue_at_attempt"]["before"], [1, 0])
            self.assertEqual(event["queue_at_attempt"]["after"], [0, 1])
            self.assertAlmostEqual(sum(event["reward_components"].values()), event["reward_legacy"])
            trace = next(t for t in rows(root / "packet_traces.jsonl") if t["packet_key"][1] == 18)
            self.assertEqual(trace["route"], [18, 24])
            self.assertEqual(trace["outcome"], 1)
            self.assertEqual(trace["delivered_step"], 1)
            self.assertFalse(trace["censored"])
            self.assertEqual(len(trace["prefix"]), 2)
            audited = audit_run(root)
            self.assertEqual(audited["steps"], 2)
            self.assertEqual(audited["delivered"], 1)
            self.assertEqual(audited["unresolved"], 23)

    def test_ack_failure_does_not_make_final_packet_outcome_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_script(temp, [{0: 1}], loss=True)
            event = rows(root / "transitions.jsonl")[0]["events"][0]
            self.assertTrue(event["attempted"])
            self.assertFalse(event["ack"])
            self.assertEqual(event["queue_at_attempt"]["before"], event["queue_at_attempt"]["after"])
            self.assertTrue(all(t["outcome"] is None for t in rows(root / "packet_traces.jsonl")))

    def test_export_can_be_disabled_and_budget_failure_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_script(temp, [{}], enabled=False)
            self.assertFalse(root.exists())
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "step budget"):
                self.run_script(temp, [{}, {}], max_steps=1)
            root = Path(temp) / "moves_0" / "wm_data"
            self.assertFalse(json.loads((root / "manifest.json").read_text())["complete"])

    def test_reader_has_only_past_actions_and_current_inputs_with_wait_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_script(temp, [{0: 1, 1: 2}, {}, {}])
            audit_run(root)
            samples = list(iter_wm_samples(root, history_length=2))
            first = next(sample for sample in samples if sample["agent_id"] == 0 and sample["step"] == 0)
            self.assertEqual(first["inputs"]["candidate_action"], 1)
            self.assertIsNone(first["inputs"]["history"][0]["own_previous_action"])
            self.assertIsNone(first["inputs"]["history"][0]["own_previous_ack"])
            self.assertFalse(first["inputs"]["history"][0]["own_previous_attempt"])
            next_sender = next(sample for sample in samples if sample["agent_id"] == 1 and sample["step"] == 1)
            self.assertTrue(next_sender["inputs"]["history"][-1]["own_previous_ack"])
            self.assertTrue(next_sender["inputs"]["history"][-1]["own_previous_attempt"])
            self.assertTrue(first["targets"]["ack_loss_mask"])
            waited = next(sample for sample in samples if sample["agent_id"] == 1 and sample["step"] == 1)
            self.assertEqual([frame["step"] for frame in waited["inputs"]["history"]], [0, 1])
            self.assertEqual(waited["inputs"]["history"][-1]["own_previous_action"], 2)
            self.assertFalse(waited["targets"]["ack_loss_mask"])
            self.assertIsNone(waited["targets"]["ack"])
            self.assertNotIn("diagnostic", json.dumps(waited["inputs"]))
            self.assertNotIn("outcome", json.dumps(waited["inputs"]))
            self.assertEqual(list(iter_wm_samples(root, split="test")), [])
            with self.assertRaises(ValueError):
                list(iter_wm_samples(root, history_length=0))

    def test_audit_rejects_corruption_and_reader_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_script(temp, [{0: 1}, {}])
            original = (root / "transitions.jsonl").read_text(encoding="utf-8")
            records = rows(root / "transitions.jsonl")
            changes = [lambda r: r[0].update(state_tp1_ref="bad:0"),
                       lambda r: r[0]["behavior"]["actions"][0].__setitem__(0, 0),
                       lambda r: r[0].update(reward_legacy=999),
                       lambda r: r[0]["events"][0].update(ack=None),
                       lambda r: r[0].update(step=2)]
            for change in changes:
                corrupt = copy.deepcopy(records)
                change(corrupt)
                (root / "transitions.jsonl").write_text("\n".join(json.dumps(r) for r in corrupt)+"\n", encoding="utf-8")
                # Keep byte accounting valid so semantic validators are exercised.
                manifest = read_json(root / "manifest.json")
                manifest["jsonl_bytes"] = sum(p.stat().st_size for p in root.glob("*.jsonl"))
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises((ValueError, KeyError)):
                    audit_run(root)
            (root / "transitions.jsonl").write_text(original, encoding="utf-8")
            manifest = read_json(root / "manifest.json")
            manifest["complete"] = False
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                list(iter_wm_samples(root))
            (root / "transitions.jsonl").write_text('{"value": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                list(read_rows(root / "transitions.jsonl"))

    def test_split_manifest_rejects_seed_reuse_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.run_script(temp, [{}])
            manifest = read_json(root / "manifest.json")
            manifest["scenario_seed_override"] = True
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            collection = Path(temp)
            run = {"scenario_seed": manifest["scenario_seed"], "split": "train",
                   "data_path": str(root.relative_to(collection))}
            other_manifest = dict(manifest, split="test")
            second = Path(temp)/"second"
            second.mkdir()
            (second/"manifest.json").write_text(json.dumps(other_manifest), encoding="utf-8")
            collection = Path(temp)
            data = {"schema_version": "legacy_tg_collection_v1", "complete": True,
                    "runs": [run, dict(run, split="test", data_path="second")]}
            (collection / "dataset_manifest.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed reused"):
                list(run_roots(collection))
            data["runs"] = [run, dict(run)]
            (collection / "dataset_manifest.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source run reused"):
                list(run_roots(collection))
            data["runs"] = [dict(run, data_path="../outside")]
            (collection / "dataset_manifest.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                list(run_roots(collection))

    def test_packet_and_byte_budgets_leave_incomplete_dataset(self):
        for budget in ("max_packets", "max_bytes"):
            with tempfile.TemporaryDirectory() as temp:
                env = make_environment(temp)
                env.packet_generation()
                recorder = TransitionRecorder(Path(temp)/"limited", env.all_args, **{budget: 1})
                try:
                    with self.assertRaisesRegex(ValueError, "budget"):
                        recorder.start_episode(env, 0, np.zeros((25, 25)))
                finally:
                    recorder.close()
                self.assertFalse(read_json(Path(temp)/"limited"/"manifest.json")["complete"])


class ExportProcessIntegrationTests(unittest.TestCase):
    def run_command(self, args, expected=0):
        result = subprocess.run([sys.executable] + args, cwd=str(ROOT),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        self.assertEqual(result.returncode, expected, (result.stdout, result.stderr))
        return result

    def test_export_toggle_preserves_original_training_metrics_and_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            for mode in ("false", "true"):
                self.run_command(["run_algorithm.py", "--algorithm", "tg-mappo",
                    "--seed", "0", "--num_episodes", "2", "--episode_length", "8",
                    "--num_env_steps", "16", "--ppo_epoch", "1", "--num_mini_batch", "2",
                    "--num_moving_buoys", "5", "--export_wm_data", mode,
                    "--output-root", str(Path(temp)/mode)])
            before = next((Path(temp)/"false").rglob("TG_MAPPO_metrics.csv")).read_bytes()
            after = next((Path(temp)/"true").rglob("TG_MAPPO_metrics.csv")).read_bytes()
            self.assertEqual(before, after)
            root = next((Path(temp)/"true").rglob("wm_data"))
            report = audit_run(root)
            self.assertEqual(report["steps"], 16)
            self.assertEqual(report["packets"], 48)
            self.assertEqual(report["snapshots"], 18)
            self.run_command(["scripts/audit_tg_dataset.py", str(root),
                              "--output", str(Path(temp)/"audit.json")])

    def test_collection_uses_independent_splits_and_rejects_budget_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            self.run_command(["scripts/collect_tg_data.py", "--episodes", "1", "--episode-length", "4",
                              "--behaviors", "tg-mappo", "--seeds", "12", "41", "73", "--output-root", temp])
            path = next(Path(temp).rglob("dataset_manifest.json")).parent
            manifest = read_json(path/"dataset_manifest.json")
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["total_steps"], 12)
            self.assertEqual([run["split"] for run in manifest["runs"]], ["train", "validation", "test"])
            for split in ("train", "validation", "test"):
                runs = list(run_roots(path, split))
                self.assertEqual(len(runs), 1)
                self.assertEqual(audit_run(runs[0])["steps"], 4)
                self.assertTrue(list(iter_wm_samples(path, split)))
            rejected = Path(temp)/"must_not_exist"
            self.run_command(["scripts/collect_tg_data.py", "--episodes", "1", "--episode-length", "4",
                              "--max-steps", "1", "--output-root", str(rejected)], expected=2)
            self.assertFalse(rejected.exists())

    def test_rule_runs_never_update_ppo_and_export_valid_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            for behavior in ("random", "geographic"):
                env = make_environment(str(Path(temp)/behavior), length=4)
                env.all_args.export_wm_data = True
                env.all_args.data_behavior = behavior
                algorithm = ScriptedAlgorithm([{}] * 4)
                algorithm.train = lambda: self.fail("Rules must not update PPO")
                agents = [SimpleNamespace(buoy=node, RL=algorithm, step=0) for node in env.buoys]
                with contextlib.redirect_stdout(io.StringIO()):
                    Runner(agents, env).run()
                root = Path(temp)/behavior/"moves_0"/"wm_data"
                self.assertEqual(audit_run(root)["steps"], 4)
                self.assertEqual(read_json(root/"manifest.json")["behavior"], behavior)
                self.assertTrue(list(iter_wm_samples(root)))


if __name__ == "__main__":
    unittest.main()
