"""Decision-time PM context: functions, shuffled PPO cache, real saved policies."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import numpy as np
import torch
from onpolicy.algorithms.utils.critic_progress import context_mode, extend_head, join_context, cache_context
from onpolicy.algorithms.utils.popart import PopArt
from test_legacy_targeted_fixes import make_environment, make_learner
from test_legacy_pm_reward import checkpoint
from test_legacy_wm_actor import checkpoint as wm_checkpoint, run


def build(root, pm, mode, wm=None, clock='disabled'):
    env = make_environment(root, length=4)
    a = env.all_args
    a.use_pm_reward = True
    a.pm_reward_checkpoint, a.pm_reward_alpha = str(pm), .1
    a.pm_potential_mode, a.pm_value_mode, a.tg_reward_version = 'task_return', 'learned_shaped', 'task_v2'
    a.pm_critic_context, a.tg_critic_time_mode = mode, clock
    if wm is not None:
        a.use_wm_actor, a.wm_actor_checkpoint = True, str(wm)
    return env, make_learner(env)


class ContextFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_public_launcher_accepts_context_and_rejects_unknown_PM_options(self):
        import run_algorithm
        for args in (['--pm_critic_context', 'potential'], ['--pm_critic_context=disabled']):
            run_algorithm.reject_research_options(args)
        with self.assertRaises(ValueError):
            run_algorithm.reject_research_options(['--pm_unknown_option', 'true'])

    def test_zero_extension_preserves_rng_output_and_PopArt_optimizer_references(self):
        for head in (PopArt(3, 1), torch.nn.Linear(3, 1)):
            before = torch.get_rng_state().clone()
            extended = extend_head(head)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            x = torch.tensor([[1., 2., 3.], [-1., 1., -2.]])
            phi = torch.tensor([[.3], [-.2]], requires_grad=True)
            y = join_context(x, phi, 'potential')
            self.assertTrue(torch.allclose(head(x), extended(y), atol=1e-7))
            optimizer = torch.optim.Adam(extended.parameters(), lr=.001)
            weight_id = id(extended.weight)
            if isinstance(extended, PopArt):
                extended.update(torch.tensor([[1.], [-2.]]))
            extended(y).sum().backward()
            self.assertIsNone(phi.grad)
            self.assertNotEqual(float(extended.weight.grad[0, -1]), 0.)
            optimizer.step()
            self.assertEqual(id(extended.weight), weight_id)
            self.assertIn(extended.weight, optimizer.state)

    def test_context_mode_and_input_contracts(self):
        self.assertEqual(context_mode(SimpleNamespace()), 'disabled')
        for args in (dict(pm_critic_context='invalid'), dict(pm_critic_context='potential'),
                     dict(pm_critic_context='potential', use_pm_reward=True, pm_value_mode='base_residual'),
                     dict(pm_critic_context='potential', use_pm_reward=True, use_recurrent_policy=True)):
            with self.assertRaises(ValueError):
                context_mode(SimpleNamespace(**args))
        for phi in (None, [[float('nan')]], [[1., 2.]]):
            with self.assertRaises(ValueError):
                join_context(torch.ones(1, 3), phi, 'potential')
        self.assertEqual(float(join_context(torch.ones(1, 3), [[-.2]], 'constant')[0, -1]), 1.)

    def test_context_cache_is_immutable_per_time_and_rejects_invalid_values(self):
        b = SimpleNamespace(episode_length=2, critic_progress=np.full((3, 1, 2, 1), np.nan, dtype=np.float32))
        cache_context(b, 0, .123); cache_context(b, 0, .123)
        np.testing.assert_array_equal(b.critic_progress[0], np.float32(.123))
        for step, value in ((0, .456), (3, 1.), (1, float('nan'))):
            with self.assertRaises(ValueError):
                cache_context(b, step, value)


class ContextModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shuffled_sample_uses_matching_decision_context_and_reset_clears_it(self):
        with tempfile.TemporaryDirectory() as temp:
            pm = Path(temp)/'pm.pt'; checkpoint(pm)
            _, l = build(Path(temp)/'run', pm, 'potential')
            b = l.buffer
            for t in range(5):
                b.obs[t, :, :, 0] = t
                cache_context(b, t, t/10.)
            for sample in b.feed_forward_generator(np.zeros_like(b.rewards), 2):
                np.testing.assert_allclose(sample[-1][:, 0], sample[1][:, 0]/10.)
            b.reset_episode()
            self.assertTrue(np.isnan(b.critic_progress).all())
            with self.assertRaises(ValueError):
                next(b.feed_forward_generator(np.zeros_like(b.rewards), 2))

    def test_real_PM_PPO_cache_time_head_training_and_checkpoint_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pm = root/'pm.pt'; checkpoint(pm)
            env, l = build(root/'run', pm, 'potential')
            train = l.trainer.train
            def observed(b, *a, **kw):
                expected = [r['phi_t'] for r in l.pm_runtime.episode_records]+[0.]
                np.testing.assert_allclose(b.critic_progress[:, 0, 0, 0], expected, rtol=1e-6)
                return train(b, *a, **kw)
            l.trainer.train = observed
            run(env, l)
            l.pm_runtime.assert_frozen()
            self.assertNotEqual(float(l.policy.critic.v_out.weight[0, -1]), 0.)
            self.assertEqual(l.training_records[-1]['critic_head_optimizer_steps'], 2)
            path = root/'run/moves_0/checkpoints/tg_policy_latest.pt'
            _, restored = build(root/'reload', pm, 'potential'); restored.load_policy(path)
            for mode in ('disabled', 'constant'):
                _, incompatible = build(root/mode, pm, mode)
                with self.assertRaisesRegex(ValueError, 'context configuration'):
                    incompatible.load_policy(path)

    def test_combined_WM_clock_and_PM_context_real_PPO(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pm, wm = root/'pm.pt', root/'wm.pt'
            checkpoint(pm); wm_checkpoint(wm)
            env, l = build(root/'run', pm, 'constant', wm, 'remaining')
            run(env, l)
            self.assertEqual(l.profile['ppo_updates'], 1)
            l.pm_runtime.assert_frozen(); l.wm_runtime.assert_frozen()


if __name__ == '__main__':
    unittest.main(verbosity=2)
