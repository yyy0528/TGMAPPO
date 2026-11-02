"""Real gradient updates, finite WAIT-only updates and task/rollout boundaries."""

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from test_research_graph_policy import graph_config
from src.buoy_network.research.bootstrap import build_runner, default_registry
from src.buoy_network.research.runtime.artifacts import RunArtifacts
from src.buoy_network.research.training.base import TrainingContext
from src.buoy_network.research.training.ppo import GraphMAPPOTrainer


class PPOFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_trainer(self, **overrides):
        changes = {"policy.ppo_epochs": 2, "policy.minibatch_steps": 2}
        changes.update(overrides)
        cfg = graph_config(**changes)
        registry = default_registry()
        runner = build_runner(cfg, "cpu", registry)
        trainer = registry.build("trainer", "graph_mappo", cfg, "cpu")
        trainer.policy = runner.policy
        trainer.optimizer = torch.optim.Adam(trainer.policy.network.parameters(), lr=cfg.policy.learning_rate)
        return cfg, runner, trainer

    def test_actual_actor_and_critic_update_and_unchanged_rollout(self):
        cfg, runner, trainer = self.make_trainer()
        runner.reset("update", 12)
        rollout = runner.collect_rollout()
        old_actions = tuple(record.action_batch for record in rollout.transitions)
        metrics = trainer.update(rollout)
        self.assertGreater(metrics["actor_parameter_delta"], 0)
        self.assertGreater(metrics["critic_parameter_delta"], 0)
        self.assertLess(metrics["initial_ratio_max_error"], 1e-5)
        self.assertEqual(tuple(record.action_batch for record in rollout.transitions), old_actions)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        with self.assertRaisesRegex(ValueError, "fresh rollout"):
            trainer.update(rollout)

    def test_wait_only_after_active_update_preserves_actor_including_adam_momentum(self):
        cfg, runner, trainer = self.make_trainer()
        runner.reset("active", 12)
        trainer.update(runner.collect_rollout())
        isolated_cfg = cfg.with_overrides({"environment.comm_range": 0.01})
        isolated = build_runner(isolated_cfg, "cpu", default_registry())
        isolated.policy = trainer.policy
        isolated.reset("isolated", 13)
        rollout = isolated.collect_rollout()
        self.assertTrue(all(record.action_batch.actions == (9,) * 9 for record in rollout.transitions))
        metrics = trainer.update(rollout)
        self.assertEqual(metrics["actor_parameter_delta"], 0.0)
        self.assertGreater(metrics["critic_parameter_delta"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_long_task_keeps_environment_and_updates_policy_at_rollout_boundary(self):
        _, runner, trainer = self.make_trainer(**{"environment.comm_range": 0.01})
        runner.reset("long", 12)
        first = runner.collect_rollout()
        trainer.update(first)
        second = runner.collect_rollout()
        self.assertTrue(first.transitions[-1].rollout_truncated)
        self.assertEqual(first.transitions[-1].state_tp1, second.transitions[0].state_t)
        self.assertNotEqual(first.transitions[0].action_batch.policy_version,
                            second.transitions[0].action_batch.policy_version)
        self.assertTrue(second.transitions[-1].env_terminated)
        trainer.update(second)

    def test_policy_diagnostics_use_only_active_decisions(self):
        _, runner, trainer = self.make_trainer(**{"policy.learning_rate": 0.02, "policy.minibatch_steps": 4})
        runner.reset("diagnostics", 12)
        rollout = runner.collect_rollout()
        by_step = {record.env_step: record for record in rollout.transitions}
        original = trainer.policy.distribution
        expected, calls = [], []

        def observe_batch(observations, features):
            distribution, values = original(observations, features)
            if calls:
                records = [by_step[observation.env_step] for observation in observations]
                actions = torch.tensor([record.action_batch.actions for record in records])
                old = torch.tensor([record.action_batch.log_probs for record in records])
                mask = torch.tensor([[agent.active and any(agent.available_actions[:-1])
                                      for agent in observation.agents] for observation in observations])
                with torch.no_grad():
                    log_ratio = distribution.log_prob(actions) - old
                    ratio = log_ratio.exp()
                    expected.append((((ratio - 1) - log_ratio)[mask].mean().item(),
                                     ((ratio - 1).abs() > trainer.config.policy.clip_param)[mask].float().mean().item()))
            calls.append(True)
            return distribution, values

        trainer.policy.distribution = observe_batch
        metrics = trainer.update(rollout)
        self.assertAlmostEqual(metrics["approx_kl"], sum(row[0] for row in expected) / len(expected), places=8)
        self.assertAlmostEqual(metrics["clip_fraction"], sum(row[1] for row in expected) / len(expected), places=8)

    def test_fit_saves_real_weights_and_reports_real_parameter_deltas(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = graph_config(**{"runtime.episodes": 2, "runtime.output_root": temp,
                "policy.ppo_epochs": 1, "policy.minibatch_steps": 2})
            registry = default_registry()
            trainer = registry.build("trainer", "graph_mappo", cfg, "cpu")
            artifacts = RunArtifacts.create(cfg, "cpu", "train-policy")
            report = trainer.fit(TrainingContext(cfg, "cpu", artifacts, registry))
            payload = torch.load(report.checkpoint, map_location="cpu", weights_only=True)
            self.assertTrue(payload["model_state"])
            self.assertEqual(payload["architecture_version"], trainer.policy.architecture_version)
            self.assertGreater(report.metrics["actor_parameter_delta"], 0)
            self.assertGreater(report.metrics["critic_parameter_delta"], 0)
            self.assertTrue((artifacts.root / "model_metrics.csv").is_file())
            self.assertTrue(Path(report.checkpoint).with_suffix(".metadata.json").is_file())

    def test_invalid_baseline_modes_and_ppo_settings_are_rejected(self):
        for changes in ({"policy.backend": "geographic"},
                        {"world_model.enabled": True, "world_model.checkpoint": "unused.pt"}):
            with self.assertRaises(ValueError):
                GraphMAPPOTrainer(graph_config(**changes), "cpu")
        for name, value in (("policy.ppo_epochs", 0), ("policy.minibatch_steps", 0),
                            ("policy.learning_rate", 0), ("policy.clip_param", 0),
                            ("policy.entropy_coef", -1), ("policy.max_grad_norm", 0)):
            with self.assertRaises(ValueError):
                graph_config(**{name: value})


if __name__ == "__main__":
    unittest.main()
