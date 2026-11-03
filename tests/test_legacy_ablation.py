"""Physical metrics and original-TG ablation controls (Python 3.6)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/"src", ROOT/"tests"):
    sys.path.insert(0, str(directory))
import numpy as np
import torch
from buoy_network.physical_metrics import PacketMetrics, percentile95
from buoy_network.wm_runtime import FrozenWMRuntime, tensor_digest
from test_legacy_targeted_fixes import make_environment, make_learner
from test_legacy_wm_actor import run, checkpoint, inputs


class AblationFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_hops_delays_retry_loops_and_censor_denominators(self):
        p = SimpleNamespace(id=0, route=[0], agent_id=0, destination=2, if_sink=False)
        other = SimpleNamespace(id=1, route=[1], agent_id=1, destination=2, if_sink=False)
        metric = PacketMetrics([p, other])
        for step, sender, receiver, success in ((1,0,1,True), (2,1,2,False), (3,1,0,True), (4,0,1,True), (5,1,2,True)):
            metric.attempt(0, sender, receiver, success)
            if success:
                p.agent_id, p.if_sink = receiver, receiver == 2
                p.route.append(receiver)
                metric.transferred(p, step)
        row = metric.summary(0, 5)
        self.assertEqual((row["generated_packets"], row["delivered_packets"], row["unresolved_packets"]), (2,1,1))
        self.assertEqual((row["success_hops_mean"], row["success_delay_mean"], row["success_delay_p95"]), (4,5,5))
        self.assertEqual((row["attempts"], row["ack_failures"], row["retry_attempts"], row["loop_transfers"]), (5,1,1,2))
        self.assertIsNone(row["true_drop_rate"])
        self.assertEqual(percentile95(list(range(1,21))), 19)
        empty = PacketMetrics([other]).summary(0, 80)
        self.assertEqual(empty["delivery_rate"], 0)
        self.assertIsNone(empty["success_delay_mean"])

    def test_transfer_without_ack_is_rejected(self):
        p = SimpleNamespace(id=0, route=[0], agent_id=0, destination=2, if_sink=False)
        metric = PacketMetrics([p])
        p.route.append(1)
        p.agent_id = 1
        with self.assertRaises(ValueError):
            metric.transferred(p, 1)

    def test_history_control_never_calls_learned_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"wm.pt"
            checkpoint(path, 3)
            runtime = FrozenWMRuntime(path, 3, "history")
            arrays = inputs()
            with patch.object(runtime.model, "forward", side_effect=AssertionError("learned WM used")):
                runtime.predict(*arrays, step=0)
                runtime.record_actions(np.ones((1,3,1), dtype=np.int64))
                runtime.record_attempt(0,1,True)
                arrays[1][0,0,0,2] += .2
                arrays[1][0,0,1,2] += .1
                result = runtime.predict(*arrays, step=1)
            self.assertAlmostEqual(result[0,0,1,0], 2/3, places=6)
            np.testing.assert_allclose(result[0,0,1,1:], [.2,.1], atol=1e-6)
            self.assertFalse(result[0,0,0].any())
            self.assertFalse(runtime.metadata()["learned_forward_used"])


class AblationModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_expansion_neutral_rng_and_real_training(self):
        with tempfile.TemporaryDirectory() as temp:
            env = make_environment(Path(temp)/"base")
            torch.manual_seed(17)
            base = make_learner(env)
            rng = torch.get_rng_state().clone()
            env = make_environment(Path(temp)/"extra")
            env.all_args.actor_extra_width = 17
            torch.manual_seed(17)
            extra = make_learner(env)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(tensor_digest(base.policy.critic.state_dict()), tensor_digest(extra.policy.critic.state_dict()))
            for key, value in base.policy.actor.state_dict().items():
                self.assertTrue(torch.equal(value, extra.policy.actor.state_dict()[key]), key)
            self.assertEqual(sum(p.numel() for p in extra.policy.actor.actor_extra.parameters()), 2257)
            before = tensor_digest(extra.policy.actor.actor_extra.state_dict())
            run(env, extra)
            self.assertNotEqual(before, tensor_digest(extra.policy.actor.actor_extra.state_dict()))
            rows = [json.loads(line) for line in (Path(temp)/"extra"/"moves_0"/"logs"/"TG_physical_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["generated_packets"], 24)

    def test_history_control_real_ppo_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/"wm.pt"
            checkpoint(path)
            env = make_environment(root/"train")
            env.all_args.use_wm_actor = True
            env.all_args.wm_actor_checkpoint = str(path)
            env.all_args.wm_feature_source = "history"
            learner = make_learner(env)
            with patch.object(learner.wm_runtime.model, "forward", side_effect=AssertionError("learned forward")):
                run(env, learner)
            self.assertTrue(learner.wm_update_records[0]["fusion_changed"])
            saved = root/"train"/"moves_0"/"checkpoints"/"tg_policy_latest.pt"
            other_env = make_environment(root/"load")
            other_env.all_args.use_wm_actor = True
            other_env.all_args.wm_actor_checkpoint = str(path)
            other = make_learner(other_env)
            with self.assertRaisesRegex(ValueError, "feature source"):
                other.load_policy(saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
