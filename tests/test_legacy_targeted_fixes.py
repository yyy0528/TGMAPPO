"""Targeted regressions for the installed Python 3.6 TG-MAPPO environment."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import torch
    import gym
    from buoy_network.main import parse_args
    from buoy_network.environment import buoy_environment
    from buoy_network.agent import DeepQNetwork
    from buoy_network.runner import Runner
    from onpolicy.config import get_config, graph_config
except ImportError as exc:
    raise unittest.SkipTest("Run with the legacy dependencies and PYTHONPATH=src: " + str(exc))

ROOT = Path(__file__).resolve().parents[1]


def make_environment(output_root="results", length=4):
    argv = ["--grid_size", "5", "--num_agents", "25", "--num_episodes", "1",
            "--episode_length", str(length), "--n_rollout_threads", "1",
            "--num_mini_batch", "2", "--ppo_epoch", "1", "--use_cent_obs", "False",
            "--max_edge_dist", "1.5", "--output_root", str(output_root), "--num_moving_buoys", "0"]
    args, parser = parse_args(argv, get_config())
    args, parser = graph_config(argv, parser)
    buoys = [[10, [i, j]] for i in range(5) for j in range(5)]
    environment = buoy_environment(buoys, args)
    environment.build_env()
    return environment


def make_learner(environment):
    return DeepQNetwork(
        max(environment.get_n_action(i) for i in range(environment.buoy_number)),
        max(environment.get_n_obs(i) for i in range(environment.buoy_number)),
        env=environment,
    )


class ScriptedAlgorithm:
    """Deterministic test-only actions; real PPO is tested separately below."""

    def __init__(self, script):
        self.script = script
        self.buffer = SimpleNamespace(memory_list=np.zeros(len(script)))
        self.policy = SimpleNamespace(lr_decay=lambda *args: None)

    def warmup(self):
        pass

    def collect(self, step):
        values = np.zeros((1, 25, 1), dtype=np.float32)
        actions = np.full((1, 25, 1), 25, dtype=np.int64)
        for sender, receiver in self.script[step].items():
            actions[0, sender, 0] = receiver
        states = np.zeros((1, 25, 1, 64), dtype=np.float32)
        return values, actions, values.copy(), states, states.copy(), None

    def insert(self, data):
        pass

    def compute(self):
        pass

    def train(self):
        return {}


class GeometryAndChannelFixes(unittest.TestCase):
    def test_huber_large_negative_errors_have_symmetric_nonzero_gradients(self):
        from onpolicy.utils.util import huber_loss
        errors = torch.tensor([-20., -2., -.5, 0., .5, 2., 20.], requires_grad=True)
        losses = huber_loss(errors, 1.)
        self.assertTrue(torch.allclose(losses, torch.tensor([19.5, 1.5, .125, 0, .125, 1.5, 19.5])))
        losses.sum().backward()
        self.assertTrue(torch.allclose(errors.grad, torch.tensor([-1., -1., -.5, 0, .5, 1., 1.])))

    def test_graph_distance_is_physical_and_changes_after_motion(self):
        env = make_environment()
        _, before = env.graph_observation(env.buoys[0])
        self.assertAlmostEqual(float(before[1, 2]), 1.0)
        env.buoys[1].position = [100.0, 100.0]
        _, after = env.graph_observation(env.buoys[0])
        self.assertGreater(float(after[1, 2]), 100)
        self.assertFalse(np.array_equal(before, after))

    def test_node_features_are_physical_with_a_valid_entity_type(self):
        env = make_environment()
        features, _ = env.graph_observation(env.buoys[0])
        self.assertEqual(features.shape, (25, 4))
        np.testing.assert_allclose(features[6, :2], env.buoys[6].position)
        self.assertEqual(set(features[:, -1]), {0.0, 1.0})

    def test_channel_refresh_and_rotation_preserve_configured_loss(self):
        env = make_environment()
        for buoy in env.buoys:
            buoy.lost_rate_list = [0.6, 0.7, 0.8]
            buoy.lost_idx = 0
            buoy.node_refresh()
            self.assertEqual(buoy.lost_rate, 0.6)
            buoy.lost_idx = 0
        for expected in (0.6, 0.7, 0.8, 0.6):
            env.buoy_net.reset_lost_rate()
            self.assertTrue(all(buoy.lost_rate == expected for buoy in env.buoys))

    def test_lossless_channel_override_survives_reset_and_rotation(self):
        env = make_environment()
        for buoy in env.buoys:
            buoy.random_channel = False
            buoy.lost_rate_list = [0.6, 0.8]
            buoy.node_refresh()
            self.assertEqual(buoy.lost_rate, 0.0)
        env.buoy_net.reset_lost_rate()
        self.assertTrue(all(buoy.lost_rate == 0.0 for buoy in env.buoys))

    def test_gnn_includes_links_at_the_physical_radius(self):
        from onpolicy.algorithms.utils.gnn import TransformerConvNet
        edges, weights = TransformerConvNet.process_adj(
            torch.tensor([[0.0, 1.5, 1.6], [1.5, 0.0, 0.0], [1.6, 0.0, 0.0]]), 1.5)
        self.assertEqual(edges.t().tolist(), [[0, 1], [1, 0]])
        self.assertEqual(weights.flatten().tolist(), [1.5, 1.5])

    def test_topology_refresh_keeps_link_statistics_by_receiver(self):
        env = make_environment()
        node = env.buoys[0]
        receiver = node.neighbor[-1]
        index = node.neighbor.index(receiver)
        node.send_number[index] = 7
        node.success_number[index] = 4
        node.success_rate_neighbor[receiver] = 0.37
        env.buoy_net.update_topology(np.zeros((25, 25)))
        index = node.neighbor.index(receiver)
        self.assertEqual(node.send_number[index], 7)
        self.assertEqual(node.success_number[index], 4)
        self.assertEqual(node.success_rate_neighbor[receiver], 0.37)


class ActionAndRolloutFixes(unittest.TestCase):
    def test_a_packet_cannot_use_two_hops_in_one_joint_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            env = make_environment(temp, length=1)
            for buoy in env.buoys:
                buoy.random_channel = False

            algorithm = ScriptedAlgorithm([{0: 1, 1: 2, 2: 3, 3: 4}])
            agents = [SimpleNamespace(buoy=buoy, RL=algorithm, step=0) for buoy in env.buoys]
            with contextlib.redirect_stdout(io.StringIO()):
                Runner(agents, env).run()
            packets = {packet.id: packet for packet in env.packets}
            for source in range(4):
                self.assertEqual(packets[source].route, [source, source + 1])
            self.assertEqual(packets[4].route, [4])
            self.assertEqual(packets[4].stay_time, 2)
            self.assertEqual(sum(buoy.packet for buoy in env.buoys), 24)

    def test_new_arrivals_stay_behind_already_queued_packets(self):
        with tempfile.TemporaryDirectory() as temp:
            env = make_environment(temp, length=2)
            for buoy in env.buoys:
                buoy.random_channel = False
            algorithm = ScriptedAlgorithm([{0: 1}, {1: 2}])
            agents = [SimpleNamespace(buoy=buoy, RL=algorithm, step=0) for buoy in env.buoys]
            with contextlib.redirect_stdout(io.StringIO()):
                Runner(agents, env).run()
            packets = {packet.id: packet for packet in env.packets}
            self.assertEqual(packets[0].route, [0, 1])
            self.assertEqual(packets[1].route, [1, 2])

    def test_global_action_masks_include_wait_for_inactive_nodes(self):
        env = make_environment()
        env.packet_generation()
        mask = env.available_actions()
        self.assertEqual(mask.shape, (1, 25, 26))
        self.assertEqual(env.action_space[0].n, 26)
        self.assertEqual(env.get_avail_action(0), mask[0, 0].tolist())
        self.assertTrue(mask[0, 0, 1])
        self.assertFalse(mask[0, 0, 0])
        self.assertEqual(list(np.flatnonzero(mask[0, env.sink_id])), [25])
        env.buoys[0].packet = 0
        self.assertEqual(list(np.flatnonzero(env.available_actions()[0, 0])), [25])

    def test_warmup_resets_recurrent_state_and_writes_real_initial_observations(self):
        env = make_environment()
        env.packet_generation()
        learner = make_learner(env)
        learner.buffer.rnn_states.fill(1)
        learner.buffer.masks.fill(0)
        learner.buffer.step = 2
        learner.warmup()
        self.assertEqual(learner.buffer.step, 0)
        self.assertTrue(np.all(learner.buffer.rnn_states == 0))
        self.assertTrue(np.all(learner.buffer.masks[0] == 1))
        expected = env.observe_graph()
        np.testing.assert_allclose(learner.buffer.obs[0, 0], expected[0])
        np.testing.assert_allclose(learner.buffer.available_actions[0], env.available_actions())

    def test_buffer_rejects_overflow_and_old_scalar_masks_before_mutation(self):
        env = make_environment()
        buffer = make_learner(env).buffer
        with self.assertRaisesRegex(ValueError, "action masks"):
            buffer.insert(*([None] * 15), available_actions=np.zeros((1, 25, 1)))
        self.assertEqual(buffer.filled_steps, 0)
        buffer.filled_steps = env.all_args.episode_length
        with self.assertRaisesRegex(RuntimeError, "full"):
            buffer.insert(*([None] * 15))

    def test_sampling_and_probability_replay_use_the_same_current_mask(self):
        env = make_environment()
        env.packet_generation()
        learner = make_learner(env)
        learner.warmup()
        _, actions, old_logp, _, _, _ = learner.collect(0)
        mask = learner.buffer.available_actions[0]
        for node, action in enumerate(actions[0, :, 0]):
            self.assertTrue(mask[0, node, int(action)])
        with torch.no_grad():
            replay, _ = learner.policy.actor.evaluate_actions(
                learner.buffer.obs[0, 0], learner.buffer.node_obs[0, 0],
                learner.buffer.adj[0, 0], learner.buffer.agent_id[0, 0],
                learner.buffer.rnn_states[0, 0], actions[0], learner.buffer.masks[0, 0],
                available_actions=mask[0], active_masks=learner.buffer.active_masks[0, 0],
            )
        np.testing.assert_allclose(replay.cpu().numpy(), old_logp[0], atol=1e-5)

    def test_runner_keeps_wait_and_exactly_t_transitions(self):
        with tempfile.TemporaryDirectory() as temp:
            env = make_environment(temp, length=4)

            class WaitAlgorithm:
                def __init__(self):
                    self.buffer = SimpleNamespace(memory_list=np.zeros(5))
                    self.policy = SimpleNamespace(lr_decay=lambda *args: None)
                    self.collected = []
                    self.inserted = []
                    self.warmups = self.computes = self.trains = 0

                def warmup(self):
                    self.warmups += 1

                def collect(self, step):
                    self.collected.append(step)
                    values = np.zeros((1, 25, 1), dtype=np.float32)
                    actions = np.full((1, 25, 1), 25, dtype=np.int64)
                    states = np.zeros((1, 25, 1, 64), dtype=np.float32)
                    return values, actions, values.copy(), states, states.copy(), None

                def insert(self, data):
                    self.inserted.append(data)

                def compute(self):
                    self.computes += 1

                def train(self):
                    self.trains += 1
                    return {}

            algorithm = WaitAlgorithm()
            agents = [SimpleNamespace(buoy=buoy, RL=algorithm, step=0) for buoy in env.buoys]
            runner = Runner(agents, env)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run()
            self.assertEqual(algorithm.collected, [0, 1, 2, 3])
            self.assertEqual(len(algorithm.inserted), 4)
            self.assertEqual((env.current_step, env.current_time_step), (4, 4))
            self.assertEqual((algorithm.warmups, algorithm.computes, algorithm.trains), (1, 1, 1))
            self.assertTrue(all(packet.route == [packet.route[0]] for packet in env.packets))
            self.assertEqual(np.asarray(algorithm.inserted[0][6]).shape, (1, 25, 26))
            self.assertTrue(np.asarray(algorithm.inserted[-1][7]).all())


class NeuralIntegrationFixes(unittest.TestCase):
    def test_real_two_episode_ppo_updates_and_replays_each_probability(self):
        with tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(19)
            env = make_environment(temp, length=4)
            env.all_args.num_episodes = 2
            env.all_args.num_moving_buoys = 25
            learner = make_learner(env)
            before_actor = [p.detach().clone() for p in learner.policy.actor.parameters()]
            before_critic = [p.detach().clone() for p in learner.policy.critic.parameters()]
            original_train = learner.train
            original_collect = learner.collect
            sampled, trained = [], []

            def checked_collect(step):
                np.testing.assert_array_equal(learner.buffer.available_actions[step], env.available_actions())
                np.testing.assert_allclose(learner.buffer.node_obs[step, 0], env.observe_graph()[2])
                result = original_collect(step)
                sampled.append(result[1].copy())
                return result

            def checked_train():
                buffer = learner.buffer
                self.assertEqual(buffer.filled_steps, 4)
                self.assertTrue((buffer.masks[-1] == 0).all())
                self.assertTrue(np.isfinite(buffer.returns).all())
                for step in range(4):
                    np.testing.assert_array_equal(buffer.actions[step], sampled[len(sampled) - 4 + step])
                    with torch.no_grad():
                        replay, _ = learner.policy.actor.evaluate_actions(
                            buffer.obs[step, 0], buffer.node_obs[step, 0], buffer.adj[step, 0],
                            buffer.agent_id[step, 0], buffer.rnn_states[step, 0],
                            buffer.actions[step, 0], buffer.masks[step, 0],
                            available_actions=buffer.available_actions[step, 0],
                            active_masks=buffer.active_masks[step, 0])
                    np.testing.assert_allclose(replay.cpu().numpy(), buffer.action_log_probs[step, 0], atol=1e-5)
                trained.append(original_train())
                for value in trained[-1].values():
                    self.assertTrue(np.isfinite(float(value)))
                return trained[-1]

            learner.collect = checked_collect
            learner.train = checked_train
            agents = [SimpleNamespace(buoy=buoy, RL=learner, step=0) for buoy in env.buoys]
            runner = Runner(agents, env)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run()
            self.assertEqual((len(sampled), len(trained)), (8, 2))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_actor, learner.policy.actor.parameters())))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_critic, learner.policy.critic.parameters())))
            self.assertTrue(any(buoy.position != buoy.position_init for buoy in env.buoys))

    def test_all_inactive_wait_rollout_has_finite_losses_and_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            env = make_environment(temp, length=2)
            env.all_args.comm_range = 0.01
            env.all_args.max_edge_dist = 0.01
            env.buoy_net.comm_range = 0.01
            env.buoy_net.update_topology(np.zeros((25, 25)))
            learner = make_learner(env)
            original_train = learner.train
            metrics = []

            def checked_train():
                self.assertFalse(learner.buffer.active_masks[:-1].any())
                self.assertTrue((learner.buffer.actions == 25).all())
                metrics.append(original_train())
                return metrics[-1]

            learner.train = checked_train
            agents = [SimpleNamespace(buoy=buoy, RL=learner, step=0) for buoy in env.buoys]
            with contextlib.redirect_stdout(io.StringIO()):
                Runner(agents, env).run()
            self.assertEqual(len(metrics), 1)
            for key, value in metrics[0].items():
                self.assertTrue(np.isfinite(float(value)), key)
            for network in (learner.policy.actor, learner.policy.critic):
                self.assertTrue(all(torch.isfinite(p).all().item() for p in network.parameters()))


class ArgumentFixes(unittest.TestCase):
    def test_research_options_are_rejected_before_any_output_or_training(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "must_not_exist"
            result = subprocess.run([
                sys.executable, str(ROOT / "run_algorithm.py"), "--algorithm", "tg-mappo",
                "--num_episodes", "0", "--env_version", "research_v1",
                "--use_world_model", "true", "--output_root", str(destination),
            ], cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("run_research.py", result.stderr)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
