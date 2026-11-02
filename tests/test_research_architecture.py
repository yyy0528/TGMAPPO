"""Contract/algorithm semantics; runs with the standard library unittest runner."""

from __future__ import annotations

import contextlib
import io
import json
import random
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from src.buoy_network.research.bootstrap import build_runner, default_registry
from src.buoy_network.research.cli import main
from src.buoy_network.research.config import ExperimentConfig
from src.buoy_network.research.contracts import CandidateFeatures, TrainingReport
from src.buoy_network.research.data.buffers import EpisodeReplay, RolloutBuffer, validate_transition
from src.buoy_network.research.data.storage import DatasetStore, episode_split
from src.buoy_network.research.envs.reference import ReferenceGridEnvironment
from src.buoy_network.research.models.base import ProgressModel, WorldModel
from src.buoy_network.research.models.disabled import DisabledProgressModel, DisabledWorldModel
from src.buoy_network.research.models.rewards import PotentialRewardShaper
from src.buoy_network.research.policies.rules import RulePolicy
from src.buoy_network.research.registry import ComponentRegistry, ComponentUnavailable
from src.buoy_network.research.runtime.checkpoints import CheckpointMetadata
from src.buoy_network.research.training.base import Trainer
from src.buoy_network.research.training.runner import EpisodeRunner

ROOT = Path(__file__).resolve().parents[1]


def config(**overrides):
    values = {
        "environment.grid_size": 3, "environment.num_moving_buoys": 0,
        "environment.task_horizon": 8, "environment.task_ttl": 8,
        "policy.rollout_length": 2, "runtime.device": "cpu",
    }
    values.update(overrides)
    return ExperimentConfig().with_overrides(values)


def runner_for(cfg=None):
    return build_runner(cfg or config(), "cpu", default_registry())


class ConstantProgress(ProgressModel):
    version = "test_pm_v1"

    def score_prefix(self, trace):
        if trace.outcome is not None:
            raise AssertionError("An unresolved scorer must not receive future labels")
        return 0.5


class RecordingWorldModel(WorldModel):
    version = "test_wm_v1"

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def predict_candidates(self, local_history):
        self.inputs.append(local_history)
        result = CandidateFeatures(tuple(tuple(
            (float(legal), 0.0, 0.0, 0.0, 0.0, 0.0)
            for legal in agent.observations[-1].available_actions
        ) for agent in local_history.agents), self.version)
        self.outputs.append(result)
        return result


class ConstantValuePolicy(RulePolicy):
    def values(self, observation, features):
        return (2.0,) * len(observation.agents)


class ConfigurationTests(unittest.TestCase):
    def test_example_config_loads_and_cli_override_wins(self):
        cfg = ExperimentConfig.load(ROOT / "configs/research_v1.json")
        self.assertEqual(cfg.environment.num_nodes, 25)
        self.assertEqual(cfg.with_overrides({"runtime.seed": 98}).runtime.seed, 98)

    def test_unknown_fields_wrong_types_and_nonfinite_values_fail(self):
        cases = [
            {"enviroment": {}}, {"environment": {"grid_szie": 3}},
            {"progress": {"enabled": "false"}}, {"runtime": {"seed": True}},
            {"world_model": {"learning_rate": float("nan")}},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for value in cases:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        ExperimentConfig.load(path)

    def test_invalid_research_semantics_are_rejected(self):
        for key, value in (
            ("world_model.horizon", 3), ("environment.task_ttl", 0),
            ("policy.gamma", 1.2), ("environment.num_moving_buoys", 10),
            ("world_model.enabled", True), ("data.validation_fraction", 0.4),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                config(**{key: value})

    def test_cli_can_override_file_values_before_semantic_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            # A training-enabled config may be checked with WM explicitly off.
            path.write_text(json.dumps({"world_model": {"enabled": True}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(["check-config", "--config", str(path), "--device", "cpu",
                             "--use-world-model", "false"])
            self.assertEqual(code, 0)
            self.assertFalse(json.loads(output.getvalue())["config"]["world_model"]["enabled"])

    def test_importing_cli_does_not_load_torch_or_old_runner(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import src.buoy_network.research.cli; "
            "assert 'torch' not in sys.modules; "
            "assert 'src.buoy_network.runner' not in sys.modules",
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class EnvironmentTests(unittest.TestCase):
    def test_physical_visibility_and_immutable_snapshot(self):
        env = ReferenceGridEnvironment(config().environment)
        observation = env.reset(12, "physical")
        local = observation.agents[0]
        self.assertTrue(local.visible_mask[4])  # distance sqrt(2)
        self.assertFalse(local.visible_mask[8])
        self.assertEqual(local.node_features[8], (0.0,) * 4)
        self.assertTrue(all(row[8] == 0 for row in local.adjacency))
        self.assertFalse(local.available_actions[0])
        self.assertEqual(observation.agents[env.sink].available_actions,
                         (False,) * env.n + (True,))
        snapshot = env.export_state()
        env.step_joint((env.n,) * env.n)
        self.assertEqual(snapshot.env_step, 0)
        self.assertEqual(snapshot.packets[0].ttl, 8)
        self.assertEqual(env.export_state().packets[0].ttl, 7)
        with self.assertRaises(FrozenInstanceError):
            snapshot.env_step = 3

    def test_one_hop_per_slot_and_packet_conservation(self):
        env = ReferenceGridEnvironment(config().environment)
        env.reset(12, "one_hop")
        # Node 1 forwards its ORIGINAL head while node 0's packet arrives at 1.
        actions = (1, 2, 5, 4, 5, 8, 7, 8, 9)
        result = env.step_joint(actions)
        self.assertEqual(result.executed_actions, actions)
        packet_zero = next(p for p in result.state.packets if p.key.packet_id == 0)
        self.assertEqual((packet_zero.node_id, packet_zero.hops), (1, 1))
        self.assertTrue(all(p.hops <= 1 for p in result.state.packets))
        self.assertEqual(result.state.generated,
                         len(result.state.packets) + result.state.delivered + result.state.dropped)
        self.assertTrue(all(len(trace.history) == 1 for trace in env.packet_traces()))

    def test_illegal_actions_are_rejected_without_advancing_time(self):
        env = ReferenceGridEnvironment(config().environment)
        env.reset(0, "illegal")
        actions = [env.n] * env.n
        actions[0] = 8  # out of physical range
        with self.assertRaises(ValueError):
            env.step_joint(actions)
        self.assertEqual(env.export_state().env_step, 0)

    def test_ttl_failure_and_task_deadline_are_real_outcomes(self):
        for ttl, horizon, reason in ((1, 8, "ttl"), (8, 1, "task_deadline")):
            cfg = config(**{"environment.task_ttl": ttl, "environment.task_horizon": horizon})
            env = ReferenceGridEnvironment(cfg.environment)
            env.reset(12, reason)
            result = env.step_joint((env.n,) * env.n)
            self.assertTrue(result.env_terminated)
            self.assertEqual(result.state.dropped, result.state.generated)
            self.assertTrue(all(trace.outcome is False and trace.end_reason == reason
                                for trace in env.packet_traces()))
            with self.assertRaises(RuntimeError):
                env.step_joint((env.n,) * env.n)

    def test_seeded_mobility_and_channel_are_reproducible(self):
        cfg = config(**{"environment.num_moving_buoys": 3, "environment.mobility_std": 0.2,
                        "environment.loss_probability": 0.5})
        snapshots = []
        for _ in range(2):
            runner = runner_for(cfg)
            snapshots.append(runner.run_episode("repro", 42))
        self.assertEqual(snapshots[0], snapshots[1])

    def test_censored_packets_keep_unknown_outcomes(self):
        env = ReferenceGridEnvironment(config().environment)
        env.reset(0, "censored")
        env.step_joint((env.n,) * env.n)
        self.assertTrue(all(trace.outcome is None and trace.end_step is None
                            for trace in env.packet_traces()))


class RolloutTests(unittest.TestCase):
    def test_probability_replay_for_both_rule_policies(self):
        for name in ("random", "geographic"):
            runner = runner_for(config(**{"policy.backend": name}))
            records = runner.run_episode("probability", 2)
            for record in records:
                evaluated = runner.policy.evaluate_actions(
                    record.obs_t, record.action_batch, record.wm_features
                )
                self.assertEqual(evaluated, record.action_batch.log_probs)
                validate_transition(record)

    def test_truncation_keeps_environment_history_and_bootstrap(self):
        cfg = config(**{"policy.rollout_length": 1, "world_model.enabled": True,
                        "world_model.checkpoint": "injected-test-model"})
        wm = RecordingWorldModel()
        runner = EpisodeRunner(cfg, ReferenceGridEnvironment(cfg.environment),
                               ConstantValuePolicy("geographic"), wm, DisabledProgressModel())
        runner.reset("chunks", 12)
        first = runner.collect_rollout()
        self.assertTrue(first.transitions[-1].rollout_truncated)
        self.assertFalse(first.transitions[-1].env_terminated)
        self.assertEqual(first.bootstrap_values, (2.0,) * 9)
        advantages, _ = first.compute_gae(cfg.policy.gamma, cfg.policy.gae_lambda)
        expected = first.transitions[0].train_reward + cfg.policy.gamma * 2.0 - 2.0
        self.assertAlmostEqual(advantages[0][0], expected)
        with self.assertRaises(ValueError):
            runner.reset("bad_reset", 12)
        second = runner.collect_rollout()
        self.assertEqual(first.transitions[-1].state_tp1, second.transitions[0].state_t)
        history = wm.inputs[2]
        self.assertEqual(history.env_step, 1)
        for agent in history.agents:
            self.assertEqual(agent.env_steps, (0, 1))
            self.assertEqual(agent.past_actions,
                             (first.transitions[0].action_batch.actions[agent.agent_id],))
        self.assertIs(first.transitions[0].wm_features, wm.outputs[0])
        self.assertNotEqual(first.transitions[0].obs_t.env_step, second.transitions[0].obs_t.env_step)

    def test_terminal_bootstrap_and_buffer_overflow(self):
        runner = runner_for()
        runner.reset("terminal", 12)
        while True:
            rollout = runner.collect_rollout()
            if rollout.transitions[-1].env_terminated:
                self.assertEqual(rollout.bootstrap_values, (0.0,) * 9)
                break
        records = runner_for().run_episode("overflow", 12)
        buffer = RolloutBuffer(1)
        buffer.append(records[0])
        with self.assertRaisesRegex(ValueError, "overflow"):
            buffer.append(records[1])

    def test_pm_version_is_fixed_across_rollout_boundaries(self):
        cfg = config(**{"policy.rollout_length": 1})
        pm = ConstantProgress()
        runner = EpisodeRunner(cfg, ReferenceGridEnvironment(cfg.environment),
                               RulePolicy("geographic"), DisabledWorldModel(), pm)
        runner.reset("fixed_pm", 12)
        runner.collect_rollout()
        pm.version = "test_pm_v2"
        with self.assertRaisesRegex(ValueError, "inside a task"):
            runner.collect_rollout()

    def test_reward_boundary_identity_and_alpha_zero(self):
        gamma, alpha = 0.9, 0.2
        shaper = PotentialRewardShaper(gamma, alpha)
        phi = (0.2, 0.5, 0.4, 0.0)
        terms = [shaper.apply(0.0, phi[t], phi[t + 1], t == 2)[1] for t in range(3)]
        self.assertAlmostEqual(sum(gamma ** t * term for t, term in enumerate(terms)),
                               -alpha * phi[0])
        _, truncated = shaper.apply(0.0, 0.2, 0.5, False)
        _, terminal = shaper.apply(0.0, 0.2, 0.5, True)
        self.assertAlmostEqual(truncated, alpha * (gamma * 0.5 - 0.2))
        self.assertAlmostEqual(terminal, -alpha * 0.2)
        self.assertEqual(PotentialRewardShaper(gamma, 0.0).apply(3.0, 0.2, 0.8, False), (3.0, 0.0))

    def test_pm_shaping_flows_into_gae_and_preserves_boundary(self):
        cfg = config(**{"progress.enabled": True, "progress.checkpoint": "test-injection"})
        runner = EpisodeRunner(cfg, ReferenceGridEnvironment(cfg.environment),
                               RulePolicy("geographic"), DisabledWorldModel(), ConstantProgress())
        records = runner.run_episode("shaping", 12)
        discounted = sum(cfg.policy.gamma ** i * r.shaping_reward for i, r in enumerate(records))
        self.assertAlmostEqual(discounted, -cfg.progress.alpha * 0.5)
        self.assertTrue(any(record.shaping_reward != 0 for record in records))
        for record in records:
            self.assertAlmostEqual(record.train_reward, record.task_reward + record.shaping_reward)

    def test_auxiliary_replay_evicts_whole_episodes(self):
        first = runner_for().run_episode("first", 12)
        second = runner_for().run_episode("second", 13)
        replay = EpisodeReplay(max(len(first), len(second)))
        replay.add_episode(first)
        replay.add_episode(second)
        window = replay.sample_window(2, random.Random(1))
        self.assertEqual({record.episode_id for record in window}, {"second"})
        self.assertEqual(replay.steps, len(second))
        with self.assertRaises(ValueError):
            replay.add_episode(second)


class StorageAndExtensionTests(unittest.TestCase):
    def test_splits_do_not_overlap_and_small_datasets_are_explicit(self):
        splits = episode_split([str(i) for i in range(7)], 12, 0.7, 0.15)
        sets = [set(ids) for ids in splits.values()]
        self.assertTrue(all(sets))
        self.assertEqual(set.union(*sets), {str(i) for i in range(7)})
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(episode_split(["a"], 12, 0.7, 0.15)["validation"], [])

    def test_checkpoint_metadata_rejects_dimension_and_environment_changes(self):
        cfg = config()
        metadata = CheckpointMetadata.for_config(cfg, "world_model", "wm_test")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            metadata.save(path)
            loaded = CheckpointMetadata.load(path)
            loaded.validate(cfg, "world_model")
            with self.assertRaisesRegex(ValueError, "num_nodes"):
                loaded.validate(config(**{"environment.grid_size": 4}), "world_model")
            with self.assertRaisesRegex(ValueError, "component"):
                loaded.validate(cfg, "progress_model")
            with self.assertRaises(FileExistsError):
                metadata.save(path)

    def test_registry_refuses_silent_replacement_and_missing_components(self):
        registry = ComponentRegistry()
        registry.register("policy", "test", lambda cfg, device: (cfg.runtime.seed, device))
        self.assertEqual(registry.build("policy", "test", config(), "cpu"), (12, "cpu"))
        with self.assertRaises(ValueError):
            registry.register("policy", "test", lambda cfg, device: None)
        with self.assertRaises(ComponentUnavailable):
            registry.require("trainer", "not_implemented")

    def test_collection_storage_and_cli_work_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            args = ["collect", "--device", "cpu", "--grid-size", "3", "--episodes", "3",
                    "--num-moving-buoys", "0", "--output-root", temp, "--run-name", "integration"]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(args), 0)
            report = json.loads(output.getvalue())
            root = Path(report["output"])
            store = DatasetStore(root / "data")
            manifest = store.manifest()
            for split in ("train", "validation", "test"):
                self.assertEqual(len(manifest["splits"][split]), 1)
                rows = list(store.iter_rows(split))
                packets = list(store.iter_rows(split, packets=True))
                self.assertEqual(rows[0]["env_step"], 0)
                self.assertTrue(rows[-1]["env_terminated"])
                self.assertTrue(all(packet["outcome"] is not None for packet in packets))
            self.assertTrue((root / "research_metrics.csv").is_file())
            self.assertTrue((root / "config.json").is_file())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)  # a run name cannot overwrite existing output

    def test_unimplemented_trainer_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "must_not_exist"
            with contextlib.redirect_stderr(io.StringIO()) as error:
                code = main(["train-policy", "--policy-trainer", "not_implemented",
                             "--device", "cpu", "--output-root", str(destination)])
            self.assertEqual(code, 2)
            self.assertIn("not implemented/registered", error.getvalue())
            self.assertFalse(destination.exists())

    def test_cli_booleans_are_explicit_and_typos_are_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["check-config", "--device", "cpu",
                                   "--use_world_model", "false"]), 0)
        self.assertFalse(json.loads(output.getvalue())["config"]["world_model"]["enabled"])
        for options in (["--wm-horizn", "1"], ["--use-world-model", "maybe"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["check-config"] + options)
            self.assertEqual(error.exception.code, 2)

    def test_custom_trainer_can_be_registered_through_public_extension_hook(self):
        module = types.ModuleType("architecture_test_extension")

        class LifecycleTrainer(Trainer):
            def fit(self, context):
                metadata = CheckpointMetadata.for_config(context.config, "policy", "lifecycle_test")
                metadata.save(context.artifacts.root / "checkpoints" / "metadata.json")
                return TrainingReport("policy", "lifecycle_test", {"extension_called": 1},
                                      "checkpoints/metadata.json")

        def register_components(registry):
            registry.register("trainer", "lifecycle_test", lambda cfg, device: LifecycleTrainer())

        module.register_components = register_components
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {module.__name__: module}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(["train-policy", "--plugin", module.__name__, "--policy-trainer",
                             "lifecycle_test", "--device", "cpu", "--output-root", temp])
            self.assertEqual(code, 0)
            root = Path(json.loads(output.getvalue())["output"])
            self.assertTrue((root / "training_report.json").is_file())
            CheckpointMetadata.load(root / "checkpoints" / "metadata.json").validate(
                ExperimentConfig(), "policy"
            )


if __name__ == "__main__":
    unittest.main()
