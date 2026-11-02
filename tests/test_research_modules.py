"""Stage 2: verify module behavior and independent physical/data oracles."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_research_architecture import ConstantProgress, RecordingWorldModel, config, runner_for
from src.buoy_network.research.bootstrap import default_registry
from src.buoy_network.research.cli import _train
from src.buoy_network.research.contracts import TrainingReport
from src.buoy_network.research.data.storage import DatasetStore
from src.buoy_network.research.envs.reference import ReferenceGridEnvironment
from src.buoy_network.research.models.disabled import DisabledProgressModel, DisabledWorldModel
from src.buoy_network.research.policies.rules import RulePolicy
from src.buoy_network.research.training.base import Trainer
from src.buoy_network.research.training.runner import EpisodeRunner


def assert_physical_episode(case, runner, records):
    traces = {trace.key: trace for trace in runner.environment.packet_traces()}
    n = runner.config.environment.num_nodes
    for transition in records:
        before, after = transition.state_t, transition.state_tp1
        case.assertEqual(after.generated, len(after.packets) + after.delivered + after.dropped)
        queue_ids = [packet_id for node in after.nodes for packet_id in node.queue]
        case.assertEqual(len(queue_ids), len(set(queue_ids)))
        case.assertEqual(set(queue_ids), {packet.key.packet_id for packet in after.packets})
        for packet in after.packets:
            case.assertIn(packet.key.packet_id, after.nodes[packet.node_id].queue)
            case.assertGreater(packet.ttl, 0)
        executed_keys = [event.key for event in transition.executed_events]
        case.assertEqual(len(executed_keys), len(set(executed_keys)))
        for packet in before.packets:
            # Every packet gets exactly one history entry in every unresolved slot.
            step = traces[packet.key].history[before.env_step]
            case.assertEqual((step.env_step, step.node_id), (before.env_step, packet.node_id))
            after_packet = next((p for p in after.packets if p.key == packet.key), None)
            if after_packet is not None:
                case.assertIn(after_packet.hops - packet.hops, (0, 1))
        for sender, (node, local, action) in enumerate(zip(
                before.nodes, transition.obs_t.agents, transition.action_batch.actions)):
            case.assertTrue(local.available_actions[action])
            case.assertEqual(local.agent_id, sender)
            for receiver in range(n):
                visible = math.dist(node.position, before.nodes[receiver].position) <= runner.config.environment.comm_range
                case.assertEqual(local.visible_mask[receiver], visible)
                expected = bool(node.queue) and sender != n - 1 and receiver != sender and visible
                case.assertEqual(local.available_actions[receiver], expected)
            if not node.queue:
                case.assertEqual(action, n)
        for event in transition.executed_events:
            case.assertEqual(event.receiver, transition.action_batch.actions[event.sender])
            case.assertEqual(event.attempted, event.receiver != n)
            if event.success:
                case.assertTrue(event.attempted)
        case.assertAlmostEqual(transition.train_reward, transition.task_reward
                               + transition.earp_reward + transition.shaping_reward)
    final = records[-1].state_tp1
    case.assertTrue(records[-1].env_terminated)
    case.assertEqual(len(final.packets), 0)
    case.assertEqual(sum(t.outcome is True for t in traces.values()), final.delivered)
    case.assertEqual(sum(t.outcome is False for t in traces.values()), final.dropped)


class EnvironmentModuleMatrix(unittest.TestCase):
    def test_node_scales_and_physical_masks(self):
        for grid in (2, 3, 4, 5, 6):
            with self.subTest(nodes=grid * grid):
                cfg = config(**{"environment.grid_size": grid, "environment.task_horizon": 12,
                                "environment.task_ttl": 12})
                runner = runner_for(cfg)
                records = runner.run_episode("scale", 12)
                assert_physical_episode(self, runner, records)

    def test_dynamic_topology_random_loss_and_seed_variation(self):
        for seed in (0, 12, 31):
            for policy in ("random", "geographic"):
                with self.subTest(seed=seed, policy=policy):
                    cfg = config(**{"environment.num_moving_buoys": 9,
                                    "environment.mobility_std": 0.6,
                                    "environment.loss_probability": 0.4,
                                    "policy.backend": policy})
                    runner = runner_for(cfg)
                    records = runner.run_episode("dynamic", seed)
                    assert_physical_episode(self, runner, records)
                    self.assertNotEqual(records[0].state_t.nodes, records[0].state_tp1.nodes)

    def test_total_channel_failure_and_isolated_nodes(self):
        for changes in ({"environment.loss_probability": 1.0},
                        {"environment.comm_range": 0.01}):
            with self.subTest(changes=changes):
                runner = runner_for(config(**changes))
                records = runner.run_episode("failure", 12)
                assert_physical_episode(self, runner, records)
                self.assertEqual(records[-1].state_tp1.delivered, 0)
                self.assertEqual(records[-1].state_tp1.dropped, 8)

    def test_success_delay_and_hops_match_independent_trace_counts(self):
        runner = runner_for()
        records = runner.run_episode("metrics_oracle", 12)
        successful = [trace for trace in runner.environment.packet_traces() if trace.outcome]
        expected_delays = sorted(len(trace.history) for trace in successful)
        expected_hops = [sum(step.event in ("forwarded", "delivered") for step in trace.history)
                         for trace in successful]
        metrics = runner.metrics(records, 0)
        self.assertAlmostEqual(metrics["successful_mean_hops"], sum(expected_hops) / len(successful))
        self.assertAlmostEqual(metrics["successful_mean_delay_slots"], sum(expected_delays) / len(successful))
        self.assertEqual(metrics["successful_p95_delay_slots"],
                         expected_delays[math.ceil(len(successful) * 0.95) - 1])


class LearningDataModuleTests(unittest.TestCase):
    def test_multistep_gae_matches_discounted_sum_oracle(self):
        cfg = config(**{"policy.rollout_length": 8, "policy.gamma": 0.9, "policy.gae_lambda": 1.0})
        runner = runner_for(cfg)
        runner.reset("gae_oracle", 12)
        rollout = runner.collect_rollout()
        advantages, returns = rollout.compute_gae(0.9, 1.0)
        rewards = [transition.train_reward for transition in rollout.transitions]
        for index in range(len(rewards)):
            expected = sum(0.9 ** offset * reward for offset, reward in enumerate(rewards[index:]))
            self.assertAlmostEqual(returns[index][0], expected)
            self.assertAlmostEqual(advantages[index][0], expected)

    def test_combined_frozen_model_contract_across_short_rollouts(self):
        cfg = config(**{"policy.rollout_length": 1, "progress.enabled": True,
                        "progress.checkpoint": "injected_fixture"})
        wm = RecordingWorldModel()
        runner = EpisodeRunner(cfg, ReferenceGridEnvironment(cfg.environment), RulePolicy("geographic"),
                               wm, ConstantProgress())
        records = runner.run_episode("combined", 12)
        assert_physical_episode(self, runner, records)
        self.assertTrue(all(r.wm_features is not None for r in records))
        self.assertTrue(all(r.rollout_truncated for r in records[:-1]))
        self.assertAlmostEqual(sum(cfg.policy.gamma ** i * r.shaping_reward for i, r in enumerate(records)),
                               -cfg.progress.alpha * 0.5)

    def test_storage_rejects_mixed_node_counts_across_episodes(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DatasetStore(temp)
            for grid in (3, 4):
                runner = runner_for(config(**{"environment.grid_size": grid}))
                records = runner.run_episode("grid_{}".format(grid), grid)
                store.write_episode(records, runner.environment.packet_traces())
            with self.assertRaises(ValueError):
                store.write_manifest({"train": ["grid_3"], "validation": ["grid_4"], "test": []},
                                     "research_v1")

    def test_storage_rejects_same_scenario_seed_in_different_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DatasetStore(temp)
            for episode_id in ("a", "b", "c"):
                runner = runner_for()
                records = runner.run_episode(episode_id, 12)
                store.write_episode(records, runner.environment.packet_traces())
            with self.assertRaises(ValueError):
                store.write_manifest({"train": ["a"], "validation": ["b"], "test": ["c"]},
                                     "research_v1")

    def test_offline_training_rejects_dataset_node_count_before_fit(self):
        calls = []

        class DataTrainer(Trainer):
            def fit(self, context):
                calls.append("fit")
                return TrainingReport("world_model", "module_fixture", {}, "")

        registry = default_registry()
        registry.register("trainer", "supervised_wm", lambda cfg, device: DataTrainer())
        with tempfile.TemporaryDirectory() as temp:
            store = DatasetStore(Path(temp) / "data")
            for index in range(3):
                episode_id = "episode_{}".format(index)
                runner = runner_for()
                records = runner.run_episode(episode_id, 12 + index)
                store.write_episode(records, runner.environment.packet_traces())
            store.write_manifest({"train": ["episode_0"], "validation": ["episode_1"],
                                  "test": ["episode_2"]}, "research_v1")
            cfg = config(**{"environment.grid_size": 5, "runtime.dataset": str(store.root),
                            "runtime.output_root": str(Path(temp) / "output")})
            with self.assertRaises(ValueError):
                _train("train-wm", cfg, "cpu", registry)
            self.assertEqual(calls, [])
            self.assertFalse((Path(temp) / "output").exists())


if __name__ == "__main__":
    unittest.main()
