"""Original TG PM function/module/PPO tests; synthetic checkpoints isolate plumbing."""
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/"src", ROOT/"tests"):
    sys.path.insert(0, str(directory))
import numpy as np
import torch
from buoy_network.progress_model import ProgressModel, ARCHITECTURE, FEATURES, load_packets
from buoy_network.progress_runtime import FrozenPMRuntime, shaping_reward
from buoy_network.wm_runtime import tensor_digest
from buoy_network.dataset import read_rows
from test_legacy_targeted_fixes import make_environment, make_learner
from test_legacy_wm_actor import run, checkpoint as wm_checkpoint


def checkpoint(path, horizon=4):
    model = ProgressModel(horizon, 16)
    payload = {"architecture": ARCHITECTURE, "features": list(FEATURES), "horizon": horizon,
               "hidden": 16, "node_count": 25, "state_dict": model.state_dict(),
               "normalization_fit_split": "train", "validation_gate_passed": True,
               "validation": {"survival_nll": 1., "prior_survival_nll": 2., "window_brier": .1, "prior_window_brier": .2}}
    torch.save(payload, str(path))
    return payload


def learner_at(root, pm_path=None, alpha=1., wm_path=None):
    env = make_environment(root, length=4)
    args = env.all_args
    args.use_popart = args.use_valuenorm = False
    args.use_pm_reward = pm_path is not None
    args.pm_reward_checkpoint, args.pm_reward_alpha = str(pm_path or ""), alpha
    args.use_wm_actor, args.wm_actor_checkpoint = wm_path is not None, str(wm_path or "")
    return env, make_learner(env)


class ShapingFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_boundary_discount_identity_no_clipping_and_truncation(self):
        phi, gamma, alpha = [.2, .6, .3, .7, 0], .99, 1.5
        rewards = [shaping_reward(10, phi[t], phi[t+1], gamma, alpha, terminated=t == 3)[1] for t in range(4)]
        self.assertAlmostEqual(sum(gamma**t*r for t,r in enumerate(rewards)), -alpha*phi[0], places=12)
        self.assertLess(rewards[1], 0)
        self.assertEqual(shaping_reward(10, .2, .8, gamma, 0)[0], 10)
        self.assertEqual(shaping_reward(0, .2, .8, gamma, alpha, terminated=True)[2], 0)
        self.assertEqual(shaping_reward(0, .2, .8, gamma, alpha, truncated=True)[2], .8)
        with self.assertRaises(ValueError):
            shaping_reward(0, 0, 0, gamma, alpha, terminated=True, truncated=True)

    def test_loader_preserves_rng_and_rejects_unvalidated_models(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"pm.pt"
            payload = checkpoint(path)
            before = torch.get_rng_state().clone()
            runtime = FrozenPMRuntime(path, 25, 4, .99, 1.)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertFalse(any(p.requires_grad for p in runtime.model.parameters()))
            for field, value in (("validation_gate_passed", False), ("normalization_fit_split", "validation"), ("horizon", 80)):
                changed = dict(payload, **{field: value})
                torch.save(changed, str(path))
                with self.assertRaises(ValueError):
                    FrozenPMRuntime(path, 25, 4, .99, 1.)
            payload["state_dict"]["feature_std"][0] = 0
            torch.save(payload, str(path))
            with self.assertRaises(ValueError):
                FrozenPMRuntime(path, 25, 4, .99, 1.)
            runtime.alpha = 2
            with self.assertRaises(ValueError):
                runtime.assert_frozen()


class PMModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exported_packet_history_matches_online_runtime_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env, learner = learner_at(root/"export")
            env.all_args.export_wm_data = True
            run(env, learner)
            data_root = root/"export"/"moves_0"/"wm_data"
            packets = load_packets(data_root, "train")
            traces = list(read_rows(data_root/"packet_traces.jsonl"))
            states = list(read_rows(data_root/"snapshots.jsonl"))
            path = root/"pm.pt"
            checkpoint(path)
            runtime = FrozenPMRuntime(path, 25, 4, .99, 1)
            captured = []
            original_step = runtime.model.step
            def capture(features, hidden):
                captured.append(features.clone())
                return original_step(features, hidden)
            runtime.model.step = capture
            def observed(step):
                alive = []
                for trace in traces:
                    if trace["outcome"] == 1 and trace["last_observed_step"] <= step:
                        continue
                    prefixes = trace["prefix"][:step+1]
                    route = [prefixes[0]["node"]]
                    for prefix in prefixes[1:]:
                        if prefix["node"] != route[-1]:
                            route.append(prefix["node"])
                    alive.append(SimpleNamespace(id=trace["packet_key"][1], agent_id=route[-1],
                        destination=trace["destination"], route=route, if_sink=False))
                actor = states[step]["actor_inputs"]
                return SimpleNamespace(packets=alive, observe_graph=lambda: (actor["obs"], None,
                    [actor["node_features"]], [actor["distances"]]))
            runtime.start_episode(observed(0), 0)
            for step in range(4):
                expected = [p["features"][step] for p in packets if p["length"] > step]
                self.assertTrue(torch.allclose(captured[-1], torch.stack(expected), atol=1e-6))
                for trace in traces:
                    if trace["outcome"] == 1 and trace["last_observed_step"] == step+1:
                        runtime.delivered(trace["packet_key"][1])
                runtime.advance(observed(step+1), step+1, 0., step == 3)
            self.assertLess(runtime.summaries[0]["boundary_error"], 1e-7)
            self.assertEqual(runtime.phi, 0)

    def test_pm_policy_metadata_is_required_and_old_baseline_still_loads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/"pm.pt"
            checkpoint(path)
            env, trained = learner_at(root/"train", path)
            run(env, trained)
            policy = root/"train"/"moves_0"/"checkpoints"/"tg_policy_latest.pt"
            _, restored = learner_at(root/"reload", path)
            restored.load_policy(policy)
            self.assertEqual(tensor_digest(trained.policy.actor.state_dict()), tensor_digest(restored.policy.actor.state_dict()))
            restored.pm_runtime.alpha = 2
            with self.assertRaises(ValueError):
                restored.load_policy(policy)
            _, baseline = learner_at(root/"baseline")
            with self.assertRaisesRegex(ValueError, "PM mode"):
                baseline.load_policy(policy)


class PMIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_alpha_zero_exactly_recovers_baseline_weights_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/"pm.pt"
            checkpoint(path)
            results = []
            for name, source in (("off", None), ("zero", path)):
                torch.manual_seed(17)
                env, learner = learner_at(root/name, source, alpha=0)
                run(env, learner)
                results.append((tensor_digest(learner.policy.actor.state_dict()), tensor_digest(learner.policy.critic.state_dict()),
                                (root/name/"moves_0"/"logs"/"TG_MAPPO_metrics.csv").read_bytes()))
            self.assertEqual(results[0], results[1])

    def test_combined_frozen_models_shaped_gae_and_no_update_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, wm_path = root/"pm.pt", root/"wm.pt"
            checkpoint(path)
            wm_checkpoint(wm_path)
            env, learner = learner_at(root/"train", path, wm_path=wm_path)
            real_compute = learner.compute
            def checked_compute():
                real_compute()
                buffer = learner.buffer
                gae = np.zeros_like(buffer.value_preds[0])
                for step in reversed(range(4)):
                    delta = buffer.rewards[step]+buffer.gamma*buffer.value_preds[step+1]*buffer.masks[step+1]-buffer.value_preds[step]
                    gae = delta+buffer.gamma*buffer.gae_lambda*buffer.masks[step+1]*gae
                    np.testing.assert_allclose(buffer.returns[step], gae+buffer.value_preds[step], atol=1e-5)
                self.assertAlmostEqual(learner.pm_runtime.gamma, buffer.gamma)
                self.assertNotEqual(learner.pm_runtime.gamma, learner.gamma)
            learner.compute = checked_compute
            run(env, learner)
            self.assertTrue(learner.pm_update_records[0]["actor_changed"])
            self.assertTrue(learner.pm_update_records[0]["critic_changed"])
            self.assertTrue(learner.wm_update_records[0]["wm_frozen"])
            for runtime in (learner.pm_runtime, learner.wm_runtime):
                runtime.assert_frozen()
                self.assertTrue(all(p.grad is None for p in runtime.model.parameters()))
                model_ids = {id(p) for p in runtime.model.parameters()}
                self.assertFalse(any(id(p) in model_ids for optimizer in (learner.policy.actor_optimizer, learner.policy.critic_optimizer)
                                     for group in optimizer.param_groups for p in group["params"]))
            rows = list(csv.DictReader((root/"train"/"moves_0"/"logs"/"TG_PM_rewards.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 4)
            self.assertTrue(any(float(row["reward_progress"]) != 0 for row in rows))
            env, evaluated = learner_at(root/"eval", path, wm_path=wm_path)
            env.all_args.tg_evaluate = True
            evaluated.load_policy(root/"train"/"moves_0"/"checkpoints"/"tg_policy_latest.pt")
            evaluated.train = lambda: self.fail("Evaluation called PPO")
            run(env, evaluated)
            report = json.loads((root/"eval"/"moves_0"/"logs"/"TG_policy_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertTrue(report["actor_unchanged"] and report["critic_unchanged"])
            self.assertTrue(report["pm"]["frozen"] and report["wm"]["frozen"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
