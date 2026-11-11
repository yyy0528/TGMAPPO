"""Fixed temperature must be on-policy, mask-safe and artifact-bound."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tests')]
from onpolicy.algorithms.utils.distributions import FixedCategorical
from onpolicy.algorithms.utils.policy_temperature import temperature_from_args, temper_distribution
from test_legacy_targeted_fixes import make_environment, make_learner
from test_legacy_wm_actor import checkpoint, run


class TemperatureFunctions(unittest.TestCase):
    def test_identity_mask_mode_entropy_and_gradients(self):
        x = torch.tensor([[.1, .7, float('-inf')]], requires_grad=True)
        base = FixedCategorical(logits=x)
        self.assertIs(temper_distribution(base, 1.), base)
        cold = temper_distribution(base, .5)
        self.assertTrue(torch.equal(cold.mode(), base.mode()))
        self.assertEqual(float(cold.probs[0, 2]), 0.)
        self.assertLess(float(cold.entropy()), float(base.entropy()))
        torch.testing.assert_allclose(cold.probs[:, :2], torch.softmax(x[:, :2]/.5, -1))
        cold.log_probs(torch.tensor([[1]])).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_temperature(self):
        self.assertEqual(temperature_from_args(SimpleNamespace()), 1.)
        for t in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                temperature_from_args(SimpleNamespace(tg_policy_temperature=t))


class TemperatureNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def native(self, device, wm):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = make_environment(root/'run', length=4)
            args = env.all_args
            args.tg_device = device
            args.tg_policy_temperature = .5
            if wm:
                checkpoint(root/'wm.pt')
                args.use_wm_actor = True
                args.wm_actor_checkpoint = str(root/'wm.pt')
                args.wm_actor_input = 'residual_scaled'
            model = make_learner(env)
            env.reset();env.packet_generation();model.warmup()
            result = model.collect(0)
            b = model.buffer
            replay, _ = model.policy.actor.evaluate_actions(
                b.obs[0, 0], b.node_obs[0, 0], b.adj[0, 0], b.agent_id[0, 0],
                b.rnn_states[0, 0], result[1][0], b.masks[0, 0],
                available_actions=b.available_actions[0, 0], active_masks=b.active_masks[0, 0],
                wm_features=b.wm_features[0, 0] if wm else None)
            np.testing.assert_allclose(replay.detach().cpu().numpy(), result[2][0], atol=1e-6, rtol=1e-6)
            run(env, model)
            self.assertTrue(all(np.isfinite(v) for u in model.training_records for v in u.values()))
            other = make_environment(root/'eval', length=4)
            other.all_args.tg_device = device
            other.all_args.tg_policy_temperature = .5
            if wm:
                other.all_args.use_wm_actor = True
                other.all_args.wm_actor_checkpoint = str(root/'wm.pt')
                other.all_args.wm_actor_input = 'residual_scaled'
            restored = make_learner(other)
            cp = root/'run/moves_0/checkpoints/tg_policy_latest.pt'
            restored.load_policy(cp)
            self.assertEqual(restored.policy.actor.policy_temperature, .5)
            restored.all_args.tg_policy_temperature = 1.
            with self.assertRaises(ValueError):restored.load_policy(cp)
            with self.assertRaises(ValueError):restored.save_policy(root/'invalid')

    def test_cpu_tg(self):self.native('cpu', False)
    def test_cpu_wm(self):self.native('cpu', True)
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_tg(self):self.native('cuda', False)
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_wm(self):self.native('cuda', True)


if __name__ == '__main__':unittest.main()
