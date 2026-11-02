"""Real Torch policy tests, independent of legacy PyG/Gym dependencies."""

import math
import unittest
from dataclasses import replace

import torch

from src.buoy_network.research.bootstrap import default_registry
from src.buoy_network.research.config import ExperimentConfig
from src.buoy_network.research.contracts import CandidateFeatures
from src.buoy_network.research.envs.reference import ReferenceGridEnvironment
from src.buoy_network.research.policies.graph import GraphMAPPOPolicy


def graph_config(**overrides):
    changes = {"environment.grid_size": 3, "environment.num_moving_buoys": 0,
               "environment.task_horizon": 8, "environment.task_ttl": 8,
               "policy.backend": "graph_mappo", "policy.hidden_size": 16,
               "policy.rollout_length": 4}
    changes.update(overrides)
    return ExperimentConfig().with_overrides(changes)


class GraphPolicyFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_policy(self, **overrides):
        cfg = graph_config(**overrides)
        env = ReferenceGridEnvironment(cfg.environment)
        obs = env.reset(12, "graph_test")
        return cfg, env, obs, default_registry().build("policy", "graph_mappo", cfg, "cpu")

    def test_real_sampling_mask_replay_values_and_rng(self):
        cfg, env, obs, policy = self.make_policy()
        policy.reset(41)
        action = policy.sample(obs, None)
        state = policy.generator.get_state().clone()
        replay = policy.evaluate_actions(obs, action, None)
        self.assertEqual(policy.values(obs, None), action.values)
        self.assertTrue(torch.equal(state, policy.generator.get_state()))
        for local, selected, before, after in zip(obs.agents, action.actions, action.log_probs, replay):
            self.assertTrue(local.available_actions[selected])
            self.assertAlmostEqual(before, after, places=6)
        policy.reset(41)
        self.assertEqual(policy.sample(obs, None), action)

    def test_actor_is_local_and_critic_has_joint_observation_context(self):
        _, _, obs, policy = self.make_policy()
        before, values = policy.distribution((obs,), (None,))
        other = obs.agents[1]
        features = list(other.node_features)
        features[1] = (*features[1][:2], 0.9, features[1][3])
        agents = list(obs.agents)
        agents[1] = replace(other, node_features=tuple(features))
        changed, changed_values = policy.distribution((replace(obs, agents=tuple(agents)),), (None,))
        self.assertTrue(torch.equal(before.probs[0, 0], changed.probs[0, 0]))
        self.assertFalse(torch.equal(values[0, 0], changed_values[0, 0]))

    def test_extreme_negative_legal_logits_still_exclude_illegal_actions(self):
        for deterministic in (False, True):
            _, _, obs, policy = self.make_policy(**{"policy.deterministic": deterministic})
            with torch.no_grad():
                policy.network.actor_head[-1].weight.zero_()
                policy.network.actor_head[-1].bias.fill_(-1e10)
            distribution, _ = policy.distribution((obs,), (None,))
            available = torch.tensor([[local.available_actions for local in obs.agents]])
            self.assertEqual(distribution.probs[~available].sum().item(), 0.0)
            self.assertTrue(torch.isfinite(distribution.entropy()).all())
            for _ in range(5):
                action = policy.sample(obs, None)
                self.assertTrue(all(local.available_actions[selected]
                                    for local, selected in zip(obs.agents, action.actions)))
                self.assertTrue(all(math.isfinite(value) for value in action.log_probs))
                for agent, (selected, reported) in enumerate(zip(action.actions, action.log_probs)):
                    actual = distribution.probs[0, agent, selected].log().item()
                    self.assertAlmostEqual(reported, actual, places=6)

    def test_invisible_features_cannot_leak_into_actor_or_critic(self):
        _, _, obs, policy = self.make_policy()
        before, values = policy.distribution((obs,), (None,))
        local = obs.agents[0]
        invisible = next(i for i, visible in enumerate(local.visible_mask) if not visible)
        features = list(local.node_features)
        features[invisible] = (1000.0, 2000.0, 3000.0, 1.0)
        agents = (replace(local, node_features=tuple(features)),) + obs.agents[1:]
        after, after_values = policy.distribution((replace(obs, agents=agents),), (None,))
        self.assertTrue(torch.equal(before.probs, after.probs))
        self.assertTrue(torch.equal(values, after_values))

    def test_all_wait_graph_and_cached_features_are_finite(self):
        _, _, obs, policy = self.make_policy(**{"environment.comm_range": 0.01})
        action = policy.sample(obs, None)
        self.assertEqual(action.actions, (9,) * 9)
        self.assertEqual(action.log_probs, (0.0,) * 9)
        features = CandidateFeatures(tuple(tuple((0.0,) * 6 for _ in range(10)) for _ in range(9)), "test_v1")
        self.assertEqual(policy.evaluate_actions(obs, action, features), action.log_probs)
        self.assertTrue(all(math.isfinite(value) for value in action.values))

    def test_invalid_observation_features_and_versions_fail(self):
        _, _, obs, policy = self.make_policy()
        with self.assertRaises(ValueError):
            policy.distribution((), ())
        with self.assertRaises(ValueError):
            policy.sample(replace(obs, agents=obs.agents[:-1]), None)
        with self.assertRaises(ValueError):
            policy.sample(obs, CandidateFeatures((), "bad"))
        action = policy.sample(obs, None)
        with self.assertRaisesRegex(ValueError, "different policy version"):
            policy.evaluate_actions(obs, replace(action, policy_version="stale"), None)

    def test_configuration_dimensions_are_positive(self):
        for name in ("policy.hidden_size", "policy.graph_layers"):
            with self.assertRaises(ValueError):
                graph_config(**{name: 0})


if __name__ == "__main__":
    unittest.main()
