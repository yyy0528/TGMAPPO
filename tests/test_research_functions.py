"""Stage 1: call concrete functions with small valid and invalid inputs."""

import argparse
import contextlib
import io
import json
import math
import random
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from test_research_architecture import (
    ConstantProgress, RecordingWorldModel, config, runner_for,
)
from src.buoy_network.research.bootstrap import build_runner, default_registry, load_extensions
from src.buoy_network.research.cli import _train, boolean, build_parser, execute, main
from src.buoy_network.research.config import ExperimentConfig
from src.buoy_network.research.contracts import CandidateFeatures, TrainingReport
from src.buoy_network.research.data.buffers import (
    EpisodeReplay, RolloutBatch, RolloutBuffer, validate_transition,
)
from src.buoy_network.research.data.storage import DatasetStore, episode_split
from src.buoy_network.research.envs.base import ResearchEnvironment
from src.buoy_network.research.envs.reference import ReferenceGridEnvironment
from src.buoy_network.research.models.base import ProgressModel, WorldModel
from src.buoy_network.research.models.disabled import DisabledProgressModel, DisabledWorldModel
from src.buoy_network.research.models.rewards import PotentialRewardShaper
from src.buoy_network.research.policies.base import RoutingPolicy
from src.buoy_network.research.policies.rules import RulePolicy
from src.buoy_network.research.registry import ComponentRegistry, ComponentUnavailable
from src.buoy_network.research.runtime.artifacts import RunArtifacts
from src.buoy_network.research.runtime.checkpoints import CheckpointMetadata
from src.buoy_network.research.runtime.device import resolve_device
from src.buoy_network.research.training.base import Trainer
from src.buoy_network.research.training.runner import EpisodeRunner


def records_for(episode_id="functions"):
    runner = runner_for()
    records = runner.run_episode(episode_id, 12)
    return runner, records


class ConfigFunctions(unittest.TestCase):
    def test_defaults_serialization_validation_and_override(self):
        cfg = ExperimentConfig.load()
        self.assertIs(cfg.validate(), cfg)
        self.assertEqual(cfg.environment.num_nodes, 25)
        self.assertEqual(cfg.to_dict()["runtime"]["seed"], 12)
        self.assertEqual(cfg.with_overrides({"runtime.seed": 19}).runtime.seed, 19)

    def test_override_typo_has_a_clear_value_error(self):
        with self.assertRaises(ValueError):
            config().with_overrides({"runtime.seeed": 19})

    def test_load_nonobject_and_section_type(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            for value in ([], {"runtime": []}):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ExperimentConfig.load(path)


class EnvironmentFunctions(unittest.TestCase):
    def test_constructor_reset_observe_graph_snapshot_actions_and_close(self):
        env = ReferenceGridEnvironment(config().environment)
        with self.assertRaises(RuntimeError):
            env.observe()
        observation = env.reset(12, "direct")
        self.assertIs(env.observe(), observation)
        self.assertEqual(env._distance(0, 1), 1.0)
        self.assertEqual(env._build_observation(), observation)
        self.assertEqual(len(env.available_actions()), 9)
        result = env.step_joint((9,) * 9)
        self.assertEqual(result.state, env.export_state())
        self.assertTrue(all(trace.outcome is None for trace in env.packet_traces()))
        env.close()

    def test_finish_archives_result_and_preserves_conservation(self):
        env = ReferenceGridEnvironment(config().environment)
        env.reset(12, "finish")
        event = env._finish(0, False, "test_drop")
        self.assertEqual(event.reason, "test_drop")
        self.assertEqual(env.packet_traces()[0].outcome, False)
        state = env.export_state()
        self.assertEqual(state.generated, len(state.packets) + state.delivered + state.dropped)

    def test_abstract_interfaces_cannot_be_mistaken_for_implementations(self):
        for interface in (ResearchEnvironment, RoutingPolicy, WorldModel, ProgressModel, Trainer):
            with self.subTest(interface=interface.__name__), self.assertRaises(TypeError):
                interface()


class PolicyAndModelFunctions(unittest.TestCase):
    def test_rule_policy_distribution_sampling_values_and_probability(self):
        env = ReferenceGridEnvironment(config().environment)
        observation = env.reset(12, "rule")
        for name in ("random", "geographic"):
            policy = RulePolicy(name)
            policy.reset(12)
            distribution = policy._distribution(observation.agents[0])
            self.assertAlmostEqual(sum(distribution.values()), 1.0)
            batch = policy.sample(observation, None)
            self.assertEqual(policy.evaluate_actions(observation, batch, None), batch.log_probs)
            self.assertEqual(policy.values(observation, None), (0.0,) * 9)

    def test_unknown_rule_policy_is_not_silently_geographic(self):
        with self.assertRaises(ValueError):
            RulePolicy("randmo")

    def test_disabled_models_and_unresolved_team_potential(self):
        env = ReferenceGridEnvironment(config().environment)
        env.reset(12, "potential")
        traces = env.packet_traces()
        wm, pm = DisabledWorldModel(), DisabledProgressModel()
        self.assertIsNone(wm.predict_candidates(None))
        self.assertEqual(pm.score_prefix(traces[0]), 0.0)
        self.assertEqual(pm.packet_potential(traces[0]), 0.0)
        self.assertEqual(pm.team_potential(traces, 8), 0.0)
        learned = ConstantProgress()
        self.assertEqual(learned.packet_potential(traces[0]), 0.5)
        self.assertEqual(learned.team_potential(traces, 8), 0.5)
        self.assertEqual(learned.packet_potential(replace(traces[0], outcome=True)), 0.0)

    def test_reward_shaper_normal_terminal_and_invalid_inputs(self):
        shaper = PotentialRewardShaper(0.9, 0.2)
        reward, shaping = shaper.apply(1.0, 0.2, 0.4, False)
        self.assertAlmostEqual(reward, 1.032)
        self.assertAlmostEqual(shaping, 0.032)
        self.assertAlmostEqual(shaper.apply(1.0, 0.2, 0.4, True)[0], 0.96)
        for gamma, alpha in ((0, 0.1), (1.1, 0.1), (0.9, -1)):
            with self.assertRaises(ValueError):
                PotentialRewardShaper(gamma, alpha)
        with self.assertRaises(ValueError):
            shaper.apply(float("nan"), 0.0, 0.0, False)


class BufferFunctions(unittest.TestCase):
    def test_transition_validation_buffer_append_finish_and_gae(self):
        _, records = records_for()
        record = replace(records[0], rollout_truncated=True)
        validate_transition(record)
        buffer = RolloutBuffer(1)
        buffer.append(record)
        batch = buffer.finish((2.0,) * 9)
        advantages, returns = batch.compute_gae(0.9, 0.95)
        expected = record.train_reward + 1.8
        self.assertAlmostEqual(advantages[0][0], expected)
        self.assertAlmostEqual(returns[0][0], expected)

    def test_gae_rejects_nan_discount_and_bootstrap(self):
        _, records = records_for()
        record = replace(records[0], rollout_truncated=True)
        for gamma, bootstrap in ((float("nan"), (0.0,) * 9),
                                 (0.9, (float("nan"),) * 9)):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                RolloutBatch((record,), bootstrap).compute_gae(gamma, 0.95)

    def test_transition_rejects_mismatched_agent_order(self):
        _, records = records_for()
        record = records[0]
        agents = list(record.obs_t.agents)
        agents[0] = replace(agents[0], agent_id=8)
        with self.assertRaises(ValueError):
            validate_transition(replace(record, obs_t=replace(record.obs_t, agents=tuple(agents))))

    def test_replay_add_and_window_are_callable(self):
        _, records = records_for()
        replay = EpisodeReplay(100)
        replay.add_episode(records)
        window = replay.sample_window(2, random.Random(12))
        self.assertEqual(window[1].env_step, window[0].env_step + 1)
        with self.assertRaises(ValueError):
            replay.sample_window(100, random.Random(12))


class StorageFunctions(unittest.TestCase):
    def test_split_rejects_impossible_fractions(self):
        for fractions in ((-0.1, 0.1), (0.9, 0.9), (float("nan"), 0.1)):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                episode_split(["one", "two", "three"], 12, *fractions)

    def test_dataset_write_manifest_and_read(self):
        runner, records = records_for()
        with tempfile.TemporaryDirectory() as temp:
            store = DatasetStore(temp)
            self.assertEqual(store._episode_path("functions").name, "functions")
            with self.assertRaises(ValueError):
                store._episode_path("../bad")
            store.write_episode(records, runner.environment.packet_traces())
            store.write_manifest({"train": ["functions"], "validation": [], "test": []},
                                 "research_v1")
            self.assertEqual(store.manifest()["env_version"], "research_v1")
            self.assertEqual(len(list(store.iter_rows("train"))), len(records))
            self.assertEqual(len(list(store.iter_rows("train", packets=True))), 8)

    def test_corrupt_transition_is_rejected_before_creating_episode(self):
        runner, records = records_for()
        with tempfile.TemporaryDirectory() as temp:
            store = DatasetStore(temp)
            bad = (replace(records[0], schema_version=999),) + records[1:]
            with self.assertRaises(ValueError):
                store.write_episode(bad, runner.environment.packet_traces())
            self.assertFalse((Path(temp) / "episodes" / "functions").exists())

    def test_manifest_requires_complete_episode_files(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DatasetStore(temp)
            store._episode_path("empty").mkdir(parents=True)
            with self.assertRaises(ValueError):
                store.write_manifest({"train": ["empty"], "validation": [], "test": []},
                                     "research_v1")

    def test_malformed_manifest_has_clear_validation_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data_manifest.json"
            for value in ([], {"schema_version": 1},
                          {"schema_version": 1, "env_version": "research_v1", "splits": []}):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        DatasetStore(temp).manifest()

    def test_checkpoint_load_and_validate_reports_missing_fields(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            metadata = CheckpointMetadata.for_config(cfg, "policy", "function_v1")
            metadata.save(path)
            CheckpointMetadata.load(path).validate(cfg, "policy")
            raw = asdict(metadata)
            del raw["num_actions"]
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                CheckpointMetadata.load(path)


class RuntimeFunctions(unittest.TestCase):
    def test_cpu_auto_cuda_and_unavailable_device_branches(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        with patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(resolve_device("auto"), "cpu")
            with self.assertRaises(ValueError):
                resolve_device("cuda")
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        with patch("importlib.util.find_spec", return_value=object()), \
                patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(resolve_device("auto"), "cuda")
            self.assertEqual(resolve_device("cuda"), "cuda")
        with self.assertRaises(ValueError):
            resolve_device("wrong")

    def test_artifact_create_json_metrics_and_containment(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = config(**{"runtime.output_root": temp, "runtime.run_name": "function_run"})
            artifacts = RunArtifacts.create(cfg, "cpu", "functions")
            artifacts.write_json("nested/result.json", {"value": 3})
            artifacts.write_metrics([])
            self.assertTrue((artifacts.root / "research_metrics.csv").is_file())
            self.assertEqual(json.loads((artifacts.root / "nested/result.json").read_text())["value"], 3)
            with self.assertRaises(ValueError):
                artifacts.write_json("../escape.json", {})


class BootstrapAndCliFunctions(unittest.TestCase):
    def test_registry_register_require_build_describe(self):
        registry = ComponentRegistry()
        registry.register("test", "factory", lambda cfg, device: device)
        self.assertTrue(callable(registry.require("test", "factory")))
        self.assertEqual(registry.build("test", "factory", config(), "cpu"), "cpu")
        self.assertEqual(registry.describe(), {"test": ["factory"]})
        with self.assertRaises(TypeError):
            registry.register("test", "bad", None)
        with self.assertRaises(ComponentUnavailable):
            registry.require("test", "missing")

    def test_default_bootstrap_and_extension_hook(self):
        registry = default_registry()
        module = types.ModuleType("function_test_extension")
        module.register_components = lambda registry: registry.register(
            "test", "extension", lambda cfg, device: 12
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            load_extensions(registry, [module.__name__])
        self.assertEqual(registry.build("test", "extension", config(), "cpu"), 12)
        runner = build_runner(config(), "cpu", registry)
        runner.close()

    def test_boolean_parser_execute_and_main(self):
        self.assertTrue(boolean("TRUE"))
        self.assertFalse(boolean("false"))
        with self.assertRaises(argparse.ArgumentTypeError):
            boolean("yes")
        args = build_parser().parse_args(["describe"])
        self.assertIn("implemented", execute(args))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["check-config", "--device", "cpu"]), 0)
        self.assertTrue(json.loads(output.getvalue())["valid"])

    def test_execute_smoke_entry_and_its_gae_callback(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["smoke", "--device", "cpu", "--grid-size", "3",
                         "--num-moving-buoys", "0", "--task-horizon", "3", "--output-root", temp])
            self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["episodes"], 1)

    def test_train_dispatch_uses_registered_trainer(self):
        class DirectTrainer(Trainer):
            def fit(self, context):
                return TrainingReport("policy", "function_test", {"called": 1}, "")

        registry = default_registry()
        registry.register("trainer", "function_test", lambda cfg, device: DirectTrainer())
        with tempfile.TemporaryDirectory() as temp:
            cfg = config(**{"policy.trainer": "function_test", "runtime.output_root": temp})
            report = _train("train-policy", cfg, "cpu", registry)
            self.assertEqual(report["report"]["metrics"]["called"], 1)


class RunnerFunctions(unittest.TestCase):
    def test_reset_history_rollout_episode_metrics_and_close(self):
        cfg = config(**{"world_model.history_len": 1})
        runner = runner_for(cfg)
        runner.reset("one_step_history", 12)
        runner.collect_rollout()
        for agent in runner.local_history().agents:
            self.assertEqual(len(agent.observations), 1)
            self.assertEqual(agent.past_actions, ())
        while not runner.terminated:
            runner.collect_rollout()
        records = runner.run_episode("metrics", 13)
        self.assertEqual(runner.metrics(records, 0)["environment_steps"], len(records))
        runner.close()

    def test_invalid_policy_values_do_not_advance_environment(self):
        class InvalidValues(RulePolicy):
            def values(self, observation, features):
                return (float("nan"),) * len(observation.agents)

        cfg = config()
        runner = EpisodeRunner(cfg, ReferenceGridEnvironment(cfg.environment),
                               InvalidValues("geographic"), DisabledWorldModel(), DisabledProgressModel())
        runner.reset("bad_values", 12)
        with self.assertRaises(ValueError):
            runner.collect_rollout()
        self.assertEqual(runner.environment.export_state().env_step, 0)

    def test_invalid_wm_features_do_not_advance_environment(self):
        class InvalidFeatures(WorldModel):
            version = "invalid"

            def predict_candidates(self, local_history):
                return CandidateFeatures((), self.version)

        cfg = config()
        runner = EpisodeRunner(cfg, ReferenceGridEnvironment(cfg.environment),
                               RulePolicy("geographic"), InvalidFeatures(), DisabledProgressModel())
        runner.reset("bad_features", 12)
        with self.assertRaises(ValueError):
            runner.collect_rollout()
        self.assertEqual(runner.environment.export_state().env_step, 0)

    def test_wm_version_must_stay_fixed_during_bootstrap(self):
        class ChangesAtBootstrap(RecordingWorldModel):
            def predict_candidates(self, history):
                if len(self.inputs) == 1:
                    self.version = "changed_during_bootstrap"
                return super().predict_candidates(history)

        cfg = config(**{"policy.rollout_length": 1})
        runner = EpisodeRunner(cfg, ReferenceGridEnvironment(cfg.environment),
                               RulePolicy("geographic"), ChangesAtBootstrap(), DisabledProgressModel())
        runner.reset("bad_version", 12)
        with self.assertRaises(ValueError):
            runner.collect_rollout()


if __name__ == "__main__":
    unittest.main()
