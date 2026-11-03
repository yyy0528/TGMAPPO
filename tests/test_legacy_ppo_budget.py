"""PPO budget counting and read-only diagnostics, then original-TG integration."""
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src",ROOT/"tests"):
    sys.path.insert(0,str(p))
import numpy as np
import torch
from onpolicy.algorithms.utils.ppo_diagnostics import ppo_update_diagnostics
from buoy_network.wm_runtime import tensor_digest
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_wm_actor import run,checkpoint as wm_checkpoint
from test_legacy_pm_reward import checkpoint as pm_checkpoint


class BudgetFunctions(unittest.TestCase):
    def test_known_ratios_and_clipping(self):
        x = torch.tensor([[0.],[math.log(1.3)],[math.log(.7)]])
        result = ppo_update_diagnostics(x,torch.ones_like(x),.2)
        self.assertAlmostEqual(result["clip_fraction"],2/3.,places=6)
        self.assertAlmostEqual(result["approx_kl"],sum((r-1)-math.log(r) for r in (1,1.3,.7))/3,places=6)
        self.assertEqual(ppo_update_diagnostics(torch.zeros(4,1),torch.ones(4,1),.2),{"approx_kl":0.,"clip_fraction":0.})

    def test_masks_empty_and_multiple_action_components(self):
        x = torch.tensor([[0.,math.log(1.5)],[float("nan"),float("nan")]])
        mask = torch.tensor([[1.],[0.]])
        result = ppo_update_diagnostics(x,mask,.2)
        self.assertEqual(result["clip_fraction"],.5)
        self.assertEqual(ppo_update_diagnostics(x,torch.zeros_like(mask),.2),{"approx_kl":0.,"clip_fraction":0.})
        with self.assertRaises(ValueError):
            ppo_update_diagnostics(x,mask,.2,use_active_masks=False)

    def test_no_rng_gradient_or_input_changes(self):
        x = torch.tensor([[.1],[-.1]],requires_grad=True)
        before,rng = x.detach().clone(),torch.get_rng_state().clone()
        ppo_update_diagnostics(x,torch.ones_like(x),.2)
        self.assertTrue(torch.equal(x,before))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(x.grad)
        x.square().sum().backward()
        self.assertTrue(torch.allclose(x.grad,2*x.detach()))

    def test_invalid_metric_inputs(self):
        for x,mask,clip in ((torch.zeros(2),torch.ones(2,1),.2),
                            (torch.zeros(2,1),torch.ones(3,1),.2),
                            (torch.zeros(2,1),torch.ones(2,1),-.1),
                            (torch.zeros(2,1),torch.ones(2,1),float("nan")),
                            (torch.tensor([[float("inf")]]),torch.ones(1,1),.2)):
            with self.assertRaises(ValueError):
                ppo_update_diagnostics(x,mask,clip)


class BudgetModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_baseline_diagnostic_protocol_and_ablation_guards(self):
        from buoy_network.ablation_suite import read,validate,budget,jobs
        configs = [validate(read(ROOT/("configs/tg_ppo_budget_epoch{}.json".format(e)))) for e in (1,4)]
        a,b = configs
        self.assertEqual({k:v for k,v in a.items() if k != "ppo_epoch"},{k:v for k,v in b.items() if k != "ppo_epoch"})
        for config in configs:
            self.assertEqual(budget(config)["total_training_steps"],4800)
            plan = jobs(config,ROOT/"results/test_budget_plan")
            self.assertEqual(len(plan),6)
            self.assertTrue(all(j["group"] == "A" and "--use_linear_lr_decay" in j["base_command"] for j in plan))
            with self.assertRaises(ValueError):
                validate(dict(config,purpose="pilot"))
            with self.assertRaises(ValueError):
                validate(dict(config,groups=["A","B"]))

    def test_diagnostics_preserve_actual_policy_optimizer_results(self):
        with tempfile.TemporaryDirectory() as temp:
            results = []
            for enabled in (False,True):
                random.seed(11)
                np.random.seed(11)
                torch.manual_seed(11)
                env = make_environment(Path(temp)/str(enabled),length=4)
                env.all_args.num_episodes = 2
                env.all_args.use_linear_lr_decay = True
                learner = make_learner(env)
                context = (patch("onpolicy.algorithms.graph_mappo.ppo_update_diagnostics",
                    return_value={"approx_kl":0.,"clip_fraction":0.}) if not enabled else
                    patch("onpolicy.algorithms.graph_mappo.ppo_update_diagnostics",wraps=ppo_update_diagnostics))
                with context:
                    run(env,learner)
                results.append((tensor_digest(learner.policy.actor.state_dict()),tensor_digest(learner.policy.critic.state_dict())))
            self.assertEqual(results[0],results[1])


class BudgetIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_ppo_counts_popart_once_per_rollout_and_fixed_interactions(self):
        with tempfile.TemporaryDirectory() as temp:
            for epochs in (1,4):
                env = make_environment(Path(temp)/str(epochs),length=4)
                env.all_args.num_episodes = 2
                env.all_args.ppo_epoch = epochs
                env.all_args.use_linear_lr_decay = True
                learner = make_learner(env)
                calls = []
                original = learner.trainer.value_normalizer.update
                def observed(targets):
                    calls.append(len(targets))
                    return original(targets)
                learner.trainer.value_normalizer.update = observed
                run(env,learner)
                self.assertEqual(len(calls),2)
                self.assertEqual(learner.profile["collect_calls"],8)
                for i,row in enumerate(learner.training_records,1):
                    self.assertEqual(row["actor_optimizer_steps"],i*epochs*2)
                    self.assertEqual(row["critic_head_optimizer_steps"],i*epochs*2)
                    self.assertTrue(math.isfinite(row["approx_kl"]) and row["approx_kl"] >= 0)
                    self.assertTrue(0 <= row["clip_fraction"] <= 1)

    def test_four_epoch_wm_pm_cached_features_and_frozen_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wm,pm = root/"wm.pt",root/"pm.pt"
            wm_checkpoint(wm)
            pm_checkpoint(pm)
            env = make_environment(root/"run",length=4)
            a = env.all_args
            a.ppo_epoch,a.use_linear_lr_decay,a.tg_reward_version = 4,True,"task_v2"
            a.use_wm_actor,a.wm_actor_checkpoint = True,str(wm)
            a.use_pm_reward,a.pm_reward_checkpoint = True,str(pm)
            learner = make_learner(env)
            run(env,learner)
            self.assertEqual(learner.training_records[0]["actor_optimizer_steps"],8)
            self.assertTrue(learner.wm_update_records[0]["wm_frozen"])
            self.assertTrue(learner.pm_update_records[0]["gae_uses_shaped_rewards"])
            learner.pm_runtime.assert_frozen()
            learner.wm_runtime.assert_frozen()


if __name__ == "__main__":
    unittest.main(verbosity=2)
