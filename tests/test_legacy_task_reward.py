"""Task reward functions -> artifact/modules -> real TG/WM/PM PPO."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/"src", ROOT/"tests"):
    sys.path.insert(0, str(p))
import numpy as np
import torch
from buoy_network.task_reward import reward_contract, TaskRewardLedger
from buoy_network.task_reward_audit import audit_reward_csv
from buoy_network.wm_runtime import tensor_digest
from test_legacy_pm_reward import learner_at, checkpoint
from test_legacy_wm_actor import run, checkpoint as wm_checkpoint


def spec(**kwargs):
    return reward_contract(SimpleNamespace(tg_reward_version="task_v2", **kwargs))


class RewardFunctions(unittest.TestCase):
    def test_wait_cost_and_closed_form(self):
        ledger = TaskRewardLedger(spec(), range(24), 80)
        rows = [ledger.advance(t, 12345., t == 80) for t in range(1,81)]
        self.assertAlmostEqual(sum(r["reward_base"] for r in rows), -2.)
        self.assertEqual(rows[-1]["reward_window_end"], -1.)
        self.assertEqual(rows[-1]["unresolved_after"], 24)

    def test_delivery_once_pre_slot_time_and_empty_packets(self):
        ledger = TaskRewardLedger(spec(), [0,1], 4)
        ledger.delivered(0)
        with self.assertRaises(ValueError):
            ledger.delivered(0)
        row = ledger.advance(1, -17.833, False)
        self.assertEqual(row["reward_delivery"], .5)
        self.assertEqual(row["reward_delay"], -.25)
        self.assertEqual(row["reward_base"], .25)
        with self.assertRaises(ValueError):
            ledger.delivered(0)
        with self.assertRaises(ValueError):
            ledger.delivered(999)
        ledger.delivered(1)
        rest = [ledger.advance(t, -999., t == 4) for t in range(2,5)]
        self.assertEqual(rest[0]["reward_delivery"], .5)
        self.assertEqual(rest[-1]["reward_window_end"], 0)
        self.assertEqual(rest[-1]["reward_base"], 0)
        empty = TaskRewardLedger(spec(), [], 1).advance(1, 1., True)
        self.assertEqual(empty["reward_base"], 0.)

    def test_faster_delivery_outranks_wait_discounted_and_undiscounted(self):
        def rollout(delivery_time):
            ledger = TaskRewardLedger(spec(), [0], 80)
            rows = []
            for t in range(1,81):
                if t == delivery_time:
                    ledger.delivered(0)
                rows.append(ledger.advance(t, 0., t == 80)["reward_base"])
            return sum(rows), sum(.99**t*r for t,r in enumerate(rows))
        for column in (0,1):
            self.assertGreater(rollout(1)[column], rollout(40)[column])
            self.assertGreater(rollout(40)[column], rollout(80)[column])
            self.assertGreater(rollout(80)[column], rollout(None)[column])

    def test_invalid_weights_boundaries_and_legacy_identity(self):
        for key,value in (("tg_delivery_weight", 0), ("tg_delay_weight", -1),
                          ("tg_unfinished_weight", -1), ("tg_delay_weight", float("nan")),
                          ("tg_delivery_weight", float("inf"))):
            with self.assertRaises(ValueError):
                spec(**{key:value})
        with self.assertRaises(ValueError):
            TaskRewardLedger(spec(), [0,0], 4)
        ledger = TaskRewardLedger(spec(), [0], 4)
        with self.assertRaises(ValueError):
            ledger.advance(1, 0, True)
        row = ledger.advance(1, 0, False, rollout_truncated=True)
        self.assertEqual(row["reward_window_end"], 0.)
        self.assertEqual(row["unresolved_after"], 1)
        with self.assertRaises(ValueError):
            ledger.advance(3, 0, False)
        legacy = TaskRewardLedger(reward_contract(SimpleNamespace()), [0], 1)
        self.assertEqual(legacy.advance(1, -17.833, True)["reward_base"], -17.833)


class RewardModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_checkpoint_contract_old_compatibility_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.manual_seed(17)
            env, learner = learner_at(root/"train")
            env.all_args.tg_reward_version = "task_v2"
            run(env, learner)
            path = root/"train/moves_0/checkpoints/tg_policy_latest.pt"
            _, legacy = learner_at(root/"legacy")
            with self.assertRaisesRegex(ValueError, "reward version/weights"):
                legacy.load_policy(path)
            env2, restored = learner_at(root/"restore")
            env2.all_args.tg_reward_version = "task_v2"
            restored.load_policy(path)
            self.assertEqual(tensor_digest(learner.policy.actor.state_dict()), tensor_digest(restored.policy.actor.state_dict()))
            env2.all_args.tg_delay_weight = 2.
            with self.assertRaisesRegex(ValueError, "reward version/weights"):
                restored.load_policy(path)
            # Simulate pre-versioning v1 checkpoint: same networks/config, absent reward metadata.
            payload = torch.load(str(path), map_location="cpu")
            del payload["reward_contract"]
            old = root/"old.pt"
            torch.save(payload, str(old))
            legacy.load_policy(old)

    def test_suite_v2_uses_one_reward_for_all_groups_and_rejects_drift(self):
        from buoy_network.ablation_suite import validate, jobs
        from test_ablation_suite import configuration
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = configuration(root)
            config.update(schema="original_tg_ablation_v2", reward_contract=spec())
            config = validate(config)
            for job in jobs(config, root):
                command = job["base_command"]
                self.assertEqual(command[command.index("--tg_reward_version")+1], "task_v2")
                for key in ("delivery_weight", "delay_weight", "unfinished_weight"):
                    self.assertEqual(float(command[command.index("--tg_"+key)+1]), 1.)
            config["reward_contract"]["delay_weight"] = -1.
            with self.assertRaises(ValueError):
                validate(config)


class RewardIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_task_reward_pm_wm_buffer_gae_and_independent_reward_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pm, wm = root/"pm.pt", root/"wm.pt"
            checkpoint(pm)
            wm_checkpoint(wm)
            env, learner = learner_at(root/"train", pm, wm_path=wm)
            env.all_args.tg_reward_version = "task_v2"
            inserted = []
            original_insert, original_compute = learner.insert, learner.compute
            def insert(data):
                values = np.asarray(data[5], dtype=np.float32)
                self.assertTrue(np.all(values == values[0,0,0]))
                inserted.append(float(values[0,0,0]))
                original_insert(data)
            def compute():
                original_compute()
                b = learner.buffer
                np.testing.assert_allclose(b.rewards[:,0,0,0], inserted, atol=1e-7)
                gae = np.zeros_like(b.value_preds[0])
                for t in reversed(range(4)):
                    delta = b.rewards[t]+b.gamma*b.value_preds[t+1]*b.masks[t+1]-b.value_preds[t]
                    gae = delta+b.gamma*b.gae_lambda*b.masks[t+1]*gae
                    np.testing.assert_allclose(b.returns[t], gae+b.value_preds[t], atol=1e-5)
            learner.insert, learner.compute = insert, compute
            run(env, learner)
            logs = root/"train/moves_0/logs"
            audited = audit_reward_csv(logs, spec())
            self.assertEqual(audited["slots"], 4)
            with (logs/"TG_reward_components.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_allclose(inserted, [float(r["reward_train"]) for r in rows], atol=1e-6)
            self.assertTrue(learner.pm_update_records[0]["actor_changed"])
            self.assertTrue(learner.pm_update_records[0]["critic_changed"])
            self.assertTrue(learner.wm_update_records[0]["wm_frozen"])
            self.assertLess(learner.pm_runtime.summaries[0]["boundary_error"], 1e-7)
            text = (logs/"TG_reward_components.csv").read_text(encoding="utf-8")
            # Independent event audit must reject a corrupted reward channel.
            (logs/"TG_reward_components.csv").write_text(text.replace(rows[0]["reward_base"], "12345", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_reward_csv(logs, spec())

    def test_alpha_zero_keeps_task_v2_baseline_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pm = root/"pm.pt"
            checkpoint(pm)
            results = []
            for name,path in (("off",None), ("zero",pm)):
                torch.manual_seed(17)
                env, learner = learner_at(root/name, path, alpha=0)
                env.all_args.tg_reward_version = "task_v2"
                run(env, learner)
                results.append((tensor_digest(learner.policy.actor.state_dict()), tensor_digest(learner.policy.critic.state_dict()),
                                (root/name/"moves_0/logs/TG_physical_metrics.jsonl").read_bytes()))
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
