"""Actual learned-weight round trips, validation and read-only evaluation."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from test_research_graph_policy import graph_config
from src.buoy_network.research.bootstrap import build_runner, default_registry
from src.buoy_network.research.cli import main
from src.buoy_network.research.runtime.artifacts import RunArtifacts
from src.buoy_network.research.training.base import TrainingContext


class PolicyCheckpointFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def train(self, directory):
        cfg = graph_config(**{"runtime.output_root": str(directory), "runtime.episodes": 1,
                             "policy.ppo_epochs": 1})
        registry = default_registry()
        trainer = registry.build("trainer", "graph_mappo", cfg, "cpu")
        artifacts = RunArtifacts.create(cfg, "cpu", "train-policy")
        report = trainer.fit(TrainingContext(cfg, "cpu", artifacts, registry))
        return cfg, trainer.policy, Path(report.checkpoint)

    def test_real_weight_roundtrip_preserves_values_probabilities_and_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, trained, path = self.train(temp)
            loaded_cfg = cfg.with_overrides({"policy.checkpoint": str(path)})
            runner = build_runner(loaded_cfg, "cpu", default_registry())
            obs = runner.environment.reset(47, "roundtrip")
            trained.reset(47)
            runner.policy.reset(47)
            self.assertEqual(trained.sample(obs, None), runner.policy.sample(obs, None))
            self.assertEqual(trained.version, runner.policy.version)
            before = {name: value.clone() for name, value in runner.policy.network.state_dict().items()}
            runner.run_episode("evaluation", 71)
            self.assertTrue(all(torch.equal(value, runner.policy.network.state_dict()[name])
                                for name, value in before.items()))

    def test_bad_dimensions_architecture_and_nan_weights_fail_without_partial_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, policy, path = self.train(temp)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for kind in ("dimension", "architecture", "nan", "names"):
                data = torch.load(path, map_location="cpu", weights_only=True)
                if kind == "dimension":
                    data["metadata"]["num_nodes"] = 4
                elif kind == "architecture":
                    data["hidden_size"] = 1000
                elif kind == "nan":
                    next(iter(data["model_state"].values())).fill_(float("nan"))
                else:
                    data["model_state"].pop(next(iter(data["model_state"])))
                bad = Path(temp) / (kind + ".pt")
                torch.save(data, bad)
                before = {name: value.clone() for name, value in policy.network.state_dict().items()}
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Invalid graph policy checkpoint"):
                    policy.load_checkpoint(bad)
                self.assertTrue(all(torch.equal(value, policy.network.state_dict()[name])
                                    for name, value in before.items()))

    def test_corrupt_checkpoint_and_missing_evaluation_weights_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            bad = directory / "bad.pt"
            bad.write_text("not a torch checkpoint", encoding="utf-8")
            destination = directory / "must_not_exist"
            common = ["evaluate", "--policy", "graph_mappo", "--device", "cpu", "--output-root", str(destination)]
            for extra in ([], ["--policy-checkpoint", str(bad)]):
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    code = main(common + extra)
                self.assertEqual(code, 2)
                self.assertFalse(destination.exists())
                self.assertNotIn("Traceback", errors.getvalue())

    def test_checkpoint_cli_evaluation_does_not_change_weight_file_or_collect_data(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, checkpoint = self.train(Path(temp) / "train")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            config_path = Path(temp) / "config.json"
            config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(["evaluate", "--config", str(config_path), "--policy-checkpoint", str(checkpoint),
                             "--episodes", "2", "--seed", "71", "--device", "cpu", "--output-root", str(Path(temp) / "eval")])
            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["episodes"], 2)
            self.assertTrue(report["model_versions"]["policy"].endswith(":greedy"))
            self.assertEqual(digest, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertFalse((Path(report["output"]) / "data").exists())
            with self.assertRaisesRegex(ValueError, "resume/warm-start"):
                default_registry().build("trainer", "graph_mappo",
                    cfg.with_overrides({"policy.checkpoint": str(checkpoint)}), "cpu")

    def test_truncated_binary_checkpoints_are_clean_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for index, binary in enumerate((b"\x80\x02K", b"\x80\x02J")):
                path = directory / ("truncated_{}.pt".format(index))
                path.write_bytes(binary)
                destination = directory / ("must_not_exist_{}".format(index))
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    code = main(["evaluate", "--policy", "graph_mappo", "--policy-checkpoint", str(path),
                                 "--device", "cpu", "--output-root", str(destination)])
                self.assertEqual(code, 2)
                self.assertIn("Cannot decode checkpoint", errors.getvalue())
                self.assertNotIn("Traceback", errors.getvalue())
                self.assertFalse(destination.exists())

    def test_greedy_evaluation_has_point_mass_probabilities_and_cannot_train(self):
        cfg = graph_config(**{"policy.deterministic": True})
        runner = build_runner(cfg, "cpu", default_registry())
        obs = runner.environment.reset(12, "greedy")
        action = runner.policy.sample(obs, None)
        self.assertEqual(action.log_probs, (0.0,) * 9)
        self.assertEqual(runner.policy.evaluate_actions(obs, action, None), action.log_probs)
        runner.policy.reset(999)
        self.assertEqual(runner.policy.sample(obs, None), action)
        with self.assertRaisesRegex(ValueError, "stochastic sampling"):
            default_registry().build("trainer", "graph_mappo", cfg, "cpu")

    def test_explicit_evaluation_mode_in_json_is_preserved_and_cli_can_override(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, checkpoint = self.train(Path(temp) / "train")
            data = cfg.to_dict()
            data["policy"]["deterministic"] = False
            config_path = Path(temp) / "stochastic.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            common = ["evaluate", "--config", str(config_path), "--policy-checkpoint", str(checkpoint),
                      "--episodes", "1", "--device", "cpu", "--output-root", str(Path(temp) / "eval")]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(common), 0)
            self.assertFalse(json.loads(output.getvalue())["model_versions"]["policy"].endswith(":greedy"))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(common + ["--policy-deterministic", "true"]), 0)
            self.assertTrue(json.loads(output.getvalue())["model_versions"]["policy"].endswith(":greedy"))
            with self.assertRaises(ValueError):
                graph_config(**{"policy.deterministic": 0})


if __name__ == "__main__":
    unittest.main()
