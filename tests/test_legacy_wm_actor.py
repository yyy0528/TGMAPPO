"""Function -> module -> real original-TG PPO checks (Python 3.6/Torch 1.8).

Synthetic weights isolate plumbing; production WM/2x80 runs are recorded separately.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src", ROOT / "tests"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from buoy_network.world_model import GraphWorldModel, encode_inputs, encode_sample
from buoy_network.wm_runtime import ARCHITECTURE, FrozenWMRuntime, tensor_digest
from buoy_network.runner import Runner
from test_legacy_targeted_fixes import make_environment, make_learner


def checkpoint(path, n=25):
    model = GraphWorldModel(16)
    payload = {"architecture": ARCHITECTURE, "node_count": n, "history_length": 4,
               "normalization_fit_split": "train", "hidden": 16, "state_dict": model.state_dict()}
    torch.save(payload, str(path))
    return payload


def inputs(n=3):
    nodes = np.array([[i, 0, 0.5, int(i == n-1)] for i in range(n)], dtype=np.float32)
    distances = np.abs(np.arange(n)[:, None]-np.arange(n)[None, :]).astype(np.float32)
    obs = np.zeros((1, n, n+1), dtype=np.float32)
    obs[0, :, :n] = 0.5
    obs[0, :, n] = np.arange(n)
    return (obs, np.tile(nodes, (1, n, 1, 1)), np.tile(distances, (1, n, 1, 1)),
            np.tile(np.array([0, 1, 0, 1], dtype=np.float32), (1, n, 1)))


def learner_at(root, wm_path=None, episodes=1):
    env = make_environment(root, length=4)
    env.all_args.num_episodes = episodes
    env.all_args.use_wm_actor = wm_path is not None
    env.all_args.wm_actor_checkpoint = str(wm_path or "")
    return env, make_learner(env)


def run(env, learner):
    agents = [SimpleNamespace(buoy=buoy, RL=learner, step=0) for buoy in env.buoys]
    with contextlib.redirect_stdout(io.StringIO()):
        Runner(agents, env).run()


class FrozenInputFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_inference_encoder_never_needs_future_targets(self):
        obs, nodes, distances, _ = inputs()
        record = {"agent_id": 0, "inputs": {"candidate_action": 1, "history": [{
            "step": 0, "node_features": nodes[0, 0].tolist(), "distances": distances[0, 0].tolist(),
            "own_obs": obs[0, 0].tolist(), "own_previous_action": None}]}}
        before = encode_inputs(record)
        record["targets"] = {"next_node_features": nodes[0, 0].tolist(), "ack": True, "ack_loss_mask": True}
        supervised = encode_sample(record)
        for key in before:
            self.assertTrue(torch.equal(before[key], supervised[key]), key)
        record["targets"] = {"nonsense": object()}
        self.assertTrue(torch.equal(before["candidate"], encode_inputs(record)["candidate"]))

    def test_frozen_load_preserves_rng_and_rejects_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"wm.pt"
            payload = checkpoint(path, 3)
            rng = torch.get_rng_state().clone()
            runtime = FrozenWMRuntime(path, 3)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(runtime.model.training)
            self.assertFalse(any(p.requires_grad for p in runtime.model.parameters()))
            for field, value in (("node_count", 4), ("history_length", 0), ("normalization_fit_split", "test")):
                bad = dict(payload, **{field: value})
                torch.save(bad, str(path))
                with self.assertRaises(ValueError):
                    FrozenWMRuntime(path, 3)
            payload["state_dict"]["node_std"][0] = 0
            torch.save(payload, str(path))
            with self.assertRaises(ValueError):
                FrozenWMRuntime(path, 3)
            payload["state_dict"]["node_std"][0] = float("nan")
            torch.save(payload, str(path))
            with self.assertRaises(ValueError):
                FrozenWMRuntime(path, 3)

    def test_online_history_matches_offline_encoder_and_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"wm.pt"
            checkpoint(path, 3)
            runtime = FrozenWMRuntime(path, 3)
            arrays = inputs()
            runtime.predict(*arrays, step=0)
            actions = np.ones((1, 3, 1), dtype=np.int64)
            runtime.record_actions(actions)
            runtime.record_attempt(0, 1, True)
            predicted = runtime.predict(*arrays, step=1)
            record = {"agent_id": 0, "inputs": {"candidate_action": 1,
                       "history": list(runtime.histories[0])}}
            encoded = encode_inputs(record)
            self.assertAlmostEqual(encoded["candidate"][5].item(), 2/3, places=6)
            with torch.no_grad():
                expected = runtime.model({key: value.unsqueeze(0) for key,value in encoded.items()})
            actual = np.r_[expected["ack_logit"].sigmoid().numpy(), np.clip(expected["queue"].numpy()[0], -1, 1)]
            np.testing.assert_allclose(predicted[0, 0, 1], actual, atol=1e-6)
            self.assertFalse(predicted[0, 0, 0].any())
            self.assertEqual(predicted[0, 0, 3, 0], 0)
            self.assertEqual(predicted[0, 0, 3, 2], 0)
            with self.assertRaises(ValueError):
                runtime.predict(*arrays, step=1)
            with self.assertRaises(ValueError):
                runtime.record_attempt(0, 2, False)
            runtime.reset()
            self.assertFalse(any(runtime.histories))
            self.assertEqual(runtime.previous_ack, [None]*3)
            arrays[-1][..., :3] = 0
            self.assertFalse(runtime.predict(*arrays, step=0).any())
            runtime.assert_frozen()
            runtime.reset()
            arrays[0][0, 0, 0] = float("nan")
            with self.assertRaises(ValueError):
                runtime.predict(*arrays, step=0)
            self.assertEqual(runtime.next_step, 0)


class ActorCacheModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_neutral_initialization_masks_and_real_prediction_influence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"wm.pt"
            checkpoint(path)
            torch.manual_seed(17)
            _, baseline = learner_at(temp)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(17)
            _, learner = learner_at(temp, path)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in baseline.policy.actor.state_dict().items():
                self.assertTrue(torch.equal(value, learner.policy.actor.state_dict()[key]), key)
            self.assertEqual(tensor_digest(baseline.policy.critic.state_dict()), tensor_digest(learner.policy.critic.state_dict()))
            features = torch.randn(25, learner.policy.actor.hidden_size)
            available = torch.zeros(25, 26)
            available[:, [1, 2, 25]] = 1
            wm = torch.zeros(25, 26, 3)
            actor = learner.policy.actor
            expected = baseline.policy.actor.act.action_out(features, available)
            distribution = actor._wm_distribution(features, available, wm)
            self.assertTrue(torch.allclose(expected.probs, distribution.probs, atol=1e-7))
            self.assertTrue((distribution.probs[available == 0] == 0).all())
            with torch.no_grad():
                actor.wm_fusion[-1].weight.fill_(1)
            original = actor._wm_distribution(features, available, wm).probs
            wm[:, 1, 0] = 1
            changed = actor._wm_distribution(features, available, wm).probs
            self.assertGreater(float((original-changed).abs().max()), 1e-6)
            available[:, :25] = 0
            wait = actor._wm_distribution(features, available, wm)
            self.assertTrue((wait.mode() == 25).all())
            self.assertTrue(torch.isfinite(wait.entropy()).all())
            with self.assertRaises(ValueError):
                actor._wm_distribution(features, available, None)

    def test_minibatch_features_follow_the_same_shuffled_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"wm.pt"
            checkpoint(path)
            _, learner = learner_at(temp, path)
            buffer = learner.buffer
            advantages = np.ones_like(buffer.rewards)
            with self.assertRaises(ValueError):
                next(buffer.feed_forward_generator(advantages, 2))
            for step in range(4):
                for node in range(25):
                    value = step*25+node
                    buffer.obs[step, 0, node, 0] = value
                    buffer.wm_features[step, 0, node, :, 0] = value
            buffer.wm_feature_ready[:] = True
            seen = []
            for sample in buffer.feed_forward_generator(advantages, 2):
                self.assertEqual(len(sample), 17)
                np.testing.assert_array_equal(sample[-1][:, 0, 0], sample[1][:, 0])
                seen.extend(sample[1][:, 0].tolist())
            self.assertEqual(sorted(seen), list(range(100)))
            buffer.reset_episode()
            self.assertFalse(buffer.wm_feature_ready.any())
            self.assertFalse(buffer.wm_features.any())

    def test_policy_reload_validates_mode_environment_and_wm_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"logs").mkdir()
            path = root/"wm.pt"
            checkpoint(path)
            _, learner = learner_at(temp, path)
            learner.save_policy(root)
            policy = root/"checkpoints"/"tg_policy_latest.pt"
            _, restored = learner_at(temp, path)
            restored.load_policy(policy)
            self.assertEqual(tensor_digest(learner.policy.actor.state_dict()), tensor_digest(restored.policy.actor.state_dict()))
            _, baseline = learner_at(temp)
            with self.assertRaisesRegex(ValueError, "mode"):
                baseline.load_policy(policy)
            restored.all_args.num_moving_buoys += 1
            with self.assertRaises(ValueError):
                restored.load_policy(policy)
            restored.all_args.num_moving_buoys -= 1
            restored.all_args.gnn_use_ReLU = not restored.all_args.gnn_use_ReLU
            with self.assertRaises(ValueError):
                restored.load_policy(policy)
            restored.all_args.gnn_use_ReLU = not restored.all_args.gnn_use_ReLU
            restored.wm_runtime.file_sha256 = "different"
            with self.assertRaisesRegex(ValueError, "different frozen WM"):
                restored.load_policy(policy)


class WMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_wait_training_is_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"wm.pt"
            checkpoint(path)
            env, learner = learner_at(Path(temp)/"run", path)
            env.all_args.comm_range = env.all_args.max_edge_dist = 0.01
            env.buoy_net.comm_range = 0.01
            env.buoy_net.update_topology(np.zeros((25, 25)))
            run(env, learner)
            self.assertFalse(learner.buffer.wm_features.any())
            self.assertTrue((learner.buffer.actions == 25).all())
            self.assertTrue(all(np.isfinite(v) for v in learner.wm_update_records[0]["metrics"].values()))
            learner.wm_runtime.assert_frozen()

    def test_real_ppo_uses_only_cache_and_frozen_eval_never_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/"wm.pt"
            checkpoint(path)
            env, learner = learner_at(root/"train", path, episodes=2)
            before = tensor_digest(learner.wm_runtime.model.state_dict())
            real_train = learner.train

            def cached_train():
                with patch.object(learner.wm_runtime, "predict", side_effect=AssertionError("WM recomputed in PPO")):
                    return real_train()

            learner.train = cached_train
            run(env, learner)
            self.assertEqual(len(learner.wm_update_records), 2)
            for record in learner.wm_update_records:
                self.assertLessEqual(record["ratio_max_error"], 1e-5)
                self.assertTrue(record["actor_changed"])
                self.assertTrue(record["critic_changed"])
                self.assertTrue(record["fusion_changed"])
            self.assertEqual(before, tensor_digest(learner.wm_runtime.model.state_dict()))
            self.assertTrue(all(p.grad is None for p in learner.wm_runtime.model.parameters()))
            features = torch.randn(25, learner.policy.actor.hidden_size)
            mask = torch.from_numpy(learner.buffer.available_actions[0, 0])
            cached = learner.buffer.wm_features[0, 0]
            with torch.no_grad():
                with_wm = learner.policy.actor._wm_distribution(features, mask, cached).probs
                without_wm = learner.policy.actor._wm_distribution(features, mask, np.zeros_like(cached)).probs
            self.assertGreater(float((with_wm-without_wm).abs().max()), 0)
            wm_ids = {id(p) for p in learner.wm_runtime.model.parameters()}
            for optimizer in (learner.policy.actor_optimizer, learner.policy.critic_optimizer):
                self.assertFalse(any(id(p) in wm_ids for group in optimizer.param_groups for p in group["params"]))
            saved = list((root/"train").rglob("tg_policy_latest.pt"))[0]
            env, evaluated = learner_at(root/"eval", path)
            env.all_args.tg_evaluate = True
            evaluated.load_policy(saved)
            evaluated.compute = lambda: self.fail("evaluation computed training returns")
            evaluated.train = lambda: self.fail("evaluation called PPO")
            run(env, evaluated)
            report_path = list((root/"eval").rglob("TG_policy_report.json"))[0]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertTrue(report["actor_unchanged"] and report["critic_unchanged"])
            self.assertTrue(report["wm"]["frozen"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
