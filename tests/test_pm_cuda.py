"""Real CUDA placement, same-snapshot parity and native PM/PPO integration."""
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
import numpy as np
import torch
from buoy_network.progress_runtime import FrozenPMRuntime
from buoy_network.pm_action_credit import collect_action_targets, packet_action_scores, target_distribution
from buoy_network.wm_runtime import tensor_digest
from test_legacy_pm_reward import checkpoint
from test_pm_action_credit import build
from test_legacy_targeted_fixes import make_environment, make_learner
from test_legacy_wm_actor import run


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable in this interpreter")
class CudaProgressModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        path = root / "pm.pt"
        checkpoint(path)
        env, learner = build(root / "cpu", path, "learned_set", value_mode="base_residual")
        env.packet_generation()
        learner.warmup()
        cpu = learner.pm_runtime
        cpu.start_episode(env, 0)
        before, cuda_before = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        gpu = FrozenPMRuntime(path, 25, 4, .99, .1, "task_return", "base_residual",
                              cpu.task_contract, device="cuda")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_before, torch.cuda.get_rng_state()))
        gpu.start_episode(env, 0)
        return env, learner, cpu, gpu

    def test_potential_hidden_buffers_and_frozen_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            env, _, cpu, gpu = self.fixture(Path(temp))
            self.assertEqual(tensor_digest(cpu.model.state_dict()), tensor_digest(gpu.model.state_dict()))
            self.assertAlmostEqual(cpu.phi, gpu.phi, places=5)
            self.assertTrue(all(x.is_cuda for x in list(gpu.model.parameters()) + list(gpu.model.buffers())))
            self.assertTrue(all(x.is_cuda for x in gpu.hidden.values()))
            for step in range(1, 5):
                a = cpu.advance(env, step, -.1, step == 4)
                b = gpu.advance(env, step, -.1, step == 4)
                self.assertAlmostEqual(a["reward_train"], b["reward_train"], places=5)
            self.assertEqual(gpu.phi, 0.)
            gpu.assert_frozen()
            self.assertEqual(gpu.metadata()["inference_device"], "cuda:0")

    def test_action_targets_match_cpu_without_mutating_history_or_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            env, learner, cpu, gpu = self.fixture(Path(temp))
            expected = collect_action_targets(cpu, env, learner.buffer, 0)
            hidden = {k: v.clone() for k, v in gpu.hidden.items()}
            prefix = copy.deepcopy(gpu.prefix_state)
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone(),
                   random.getstate(), np.random.get_state())
            queues = [b.packet for b in env.buoys]
            with patch.object(gpu.model, "step", wraps=gpu.model.step) as calls:
                actual = collect_action_targets(gpu, env, learner.buffer, 0)
            self.assertEqual(calls.call_count, 1)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
            self.assertTrue(all(torch.equal(v, gpu.hidden[k]) for k, v in hidden.items()))
            self.assertEqual(prefix, gpu.prefix_state)
            self.assertEqual(queues, [b.packet for b in env.buoys])
            self.assertTrue(torch.equal(rng[0], torch.get_rng_state()))
            self.assertTrue(torch.equal(rng[1], torch.cuda.get_rng_state()))
            self.assertEqual(rng[2], random.getstate())
            np.testing.assert_equal(rng[3], np.random.get_state())
            gpu.assert_frozen()

    def test_batch_matches_per_packet_gpu_and_skips_terminal_or_zero_model_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            env, learner, _, gpu = self.fixture(Path(temp))
            b = learner.buffer
            actor = {"obs": b.obs[0, 0], "node_features": b.node_obs[0, 0, 0], "distances": b.adj[0, 0, 0]}
            packets = {}
            for packet in sorted(env.packets, key=lambda p: p.queue_order):
                packets.setdefault(packet.agent_id, packet)
            expected = np.zeros_like(b.available_actions[0, 0])
            for node, legal in enumerate(b.available_actions[0, 0]):
                if b.active_masks[0, 0, node, 0] <= 0:
                    expected[node, -1] = 1.
                else:
                    q = packet_action_scores(gpu, actor, node, packets[node], legal, 0)
                    expected[node] = target_distribution(q, legal, b.pm_action_spec['temperature'])
            np.testing.assert_allclose(collect_action_targets(gpu, env, b, 0), expected, atol=2e-6, rtol=2e-5)
            with patch.object(gpu.model, "step", side_effect=AssertionError("No future model query expected")):
                b.pm_action_spec['mode'] = 'zero_set'
                immediate = collect_action_targets(gpu, env, b, 0)
                np.testing.assert_allclose(immediate.sum(-1), 1., atol=1e-6)
                b.pm_action_spec['mode'] = 'learned_set'
                gpu.step = 3
                for name in ('obs', 'node_obs', 'adj', 'available_actions', 'active_masks'):
                    getattr(b, name)[3] = getattr(b, name)[0]
                terminal = collect_action_targets(gpu, env, b, 3)
                np.testing.assert_allclose(terminal.sum(-1), 1., atol=1e-6)
                b.active_masks[3] = 0
                inactive = collect_action_targets(gpu, env, b, 3)
                self.assertTrue(np.all(inactive[:, -1] == 1.))

    def test_native_gpu_pm_ppo_keeps_simulator_and_buffer_on_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "pm.pt"
            checkpoint(path)
            env = make_environment(root / "run", length=4)
            a = env.all_args
            a.tg_device = "cuda"
            a.use_pm_reward = True
            a.pm_reward_checkpoint, a.pm_reward_alpha = str(path), .1
            a.pm_potential_mode, a.pm_value_mode = "task_return", "base_residual"
            a.tg_reward_version, a.pm_action_credit = "task_v2", "learned_set"
            learner = make_learner(env)
            runtime = learner.pm_runtime
            original, devices = runtime.model.step, []
            def observed(features, hidden):
                self.assertTrue(features.is_cuda and hidden.is_cuda)
                logits, following = original(features, hidden)
                self.assertTrue(logits.is_cuda and following.is_cuda)
                devices.append(str(features.device))
                return logits, following
            with patch.object(runtime.model, "step", observed):
                run(env, learner)
            self.assertTrue(devices)
            self.assertTrue(all(p.is_cuda for m in (learner.policy.actor, learner.policy.critic)
                                for p in m.parameters()))
            self.assertIsInstance(learner.buffer.obs, np.ndarray)
            self.assertIsInstance(learner.buffer.pm_action_targets, np.ndarray)
            self.assertFalse(any(isinstance(x, torch.Tensor) for values in env.observe_graph() for x in values))
            update = learner.training_records[-1]
            self.assertEqual(update["actor_optimizer_steps"], 2)
            self.assertEqual(update["critic_head_optimizer_steps"], 2)
            self.assertTrue(learner.pm_update_records[-1]["pm_frozen"])


if __name__ == "__main__":
    unittest.main()
