"""LR switch semantics, frozen evaluation, protocol and original PPO tests."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src",ROOT/"tests"):
    sys.path.insert(0,str(p))
import numpy as np
import torch
from buoy_network.learning_rate import schedule_spec,advance_schedule,optimizer_rates
from buoy_network.task_reward import reward_contract
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_wm_actor import run,checkpoint as wm_checkpoint
from test_legacy_pm_reward import checkpoint as pm_checkpoint


class LRFunctions(unittest.TestCase):
    def test_spec_records_separate_rates_and_rejects_invalid_inputs(self):
        args = SimpleNamespace(lr=.002,critic_lr=.004,use_linear_lr_decay=False)
        self.assertEqual(schedule_spec(args)["kind"],"fixed")
        self.assertEqual(schedule_spec(args)["critic_initial_lr"],.004)
        for key,value in (("lr",float("nan")),("critic_lr",-1),("use_linear_lr_decay","false")):
            bad = dict(vars(args),**{key:value})
            with self.assertRaises(ValueError):
                schedule_spec(SimpleNamespace(**bad))

    def test_switch_off_and_evaluation_never_call_scheduler(self):
        calls = []
        policy = SimpleNamespace(lr_decay=lambda e,n:calls.append((e,n)))
        args = SimpleNamespace(lr=.002,critic_lr=.004,use_linear_lr_decay=False)
        advance_schedule(policy,args,1,2)
        args.use_linear_lr_decay = True
        advance_schedule(policy,args,1,2,training=False)
        self.assertEqual(calls,[])
        advance_schedule(policy,args,1,2)
        self.assertEqual(calls,[(1,2)])
        for e,n in ((-1,2),(2,2),(0,0),(.5,2),(0,True)):
            with self.assertRaises(ValueError):
                advance_schedule(policy,args,e,n)

    def test_original_linear_formula_and_last_epoch_endpoint(self):
        from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
        policy = GR_MAPPOPolicy.__new__(GR_MAPPOPolicy)
        policy.lr,policy.critic_lr = .002,.004
        policy.actor_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],lr=.002)
        policy.critic_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],lr=.004)
        args = SimpleNamespace(lr=.002,critic_lr=.004,use_linear_lr_decay=True)
        for e in range(4):
            advance_schedule(policy,args,e,4)
            rates = optimizer_rates(policy)
            self.assertEqual(rates["actor"],[.002-.002*(e/4.)])
            self.assertEqual(rates["critic"],[.004-.004*(e/4.)])
        self.assertEqual(optimizer_rates(policy)["actor"],[.0005])

    def test_cli_default_fixed_and_explicit_linear(self):
        from onpolicy.config import get_config
        from buoy_network.main import parse_args
        self.assertFalse(parse_args([],get_config())[0].use_linear_lr_decay)
        self.assertTrue(parse_args(["--use_linear_lr_decay"],get_config())[0].use_linear_lr_decay)


class LRModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_suite_schedule_is_explicit_and_old_schemas_preserve_linear(self):
        from buoy_network.ablation_suite import validate,jobs
        from test_ablation_suite import configuration
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = configuration(root)
            self.assertTrue(all("--use_linear_lr_decay" in j["base_command"] for j in jobs(validate(old),root)))
            for kind in ("fixed","linear"):
                config = dict(old,schema="original_tg_ablation_v3",learning_rate_schedule=kind,
                              reward_contract=reward_contract(SimpleNamespace(tg_reward_version="task_v2")))
                plan = jobs(validate(config),root)
                self.assertEqual(len(plan),36)
                for job in plan:
                    self.assertEqual("--use_linear_lr_decay" in job["base_command"],kind == "linear")
            config["learning_rate_schedule"] = "typo"
            with self.assertRaises(ValueError):
                validate(config)
            del config["learning_rate_schedule"]
            with self.assertRaises(ValueError):
                validate(config)

    def test_real_runner_logs_rates_in_training_checkpoint_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for enabled in (False,True):
                env = make_environment(root/str(enabled),length=4)
                env.all_args.num_episodes = 3
                env.all_args.use_linear_lr_decay = enabled
                env.all_args.lr,env.all_args.critic_lr = .0002,.0004
                learner = make_learner(env)
                run(env,learner)
                for e,row in enumerate(learner.training_records):
                    expected_a = .0002-.0002*(e/3.) if enabled else .0002
                    expected_c = .0004-.0004*(e/3.) if enabled else .0004
                    self.assertEqual(row["actor_lr"],expected_a)
                    self.assertEqual(row["critic_lr"],expected_c)
                output = root/str(enabled)/"moves_0"
                report = json.loads((output/"logs/TG_policy_report.json").read_text(encoding="utf-8"))
                payload = torch.load(str(output/"checkpoints/tg_policy_latest.pt"),map_location="cpu")
                self.assertEqual(payload["learning_rate_schedule"],report["learning_rate_schedule"])
                self.assertEqual(report["learning_rate_schedule"]["kind"],"linear" if enabled else "fixed")


class LRIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_wm_pm_both_schedules_and_frozen_evaluation_never_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wm,pm = root/"wm.pt",root/"pm.pt"
            wm_checkpoint(wm)
            pm_checkpoint(pm)
            for enabled in (False,True):
                def make(name):
                    env = make_environment(root/name,length=4)
                    a = env.all_args
                    a.num_episodes,a.tg_reward_version,a.use_linear_lr_decay = 2,"task_v2",enabled
                    a.use_wm_actor,a.wm_actor_checkpoint = True,str(wm)
                    a.use_pm_reward,a.pm_reward_checkpoint = True,str(pm)
                    return env,make_learner(env)
                env,learner = make("train"+str(enabled))
                run(env,learner)
                self.assertEqual([r["critic_head_optimizer_steps"] for r in learner.training_records],[2,4])
                learner.wm_runtime.assert_frozen()
                learner.pm_runtime.assert_frozen()
                env2,restored = make("eval"+str(enabled))
                env2.all_args.tg_evaluate = True
                restored.load_policy(root/("train"+str(enabled))/"moves_0/checkpoints/tg_policy_latest.pt")
                before = optimizer_rates(restored.policy)
                def forbidden(*a):
                    self.fail("Frozen evaluation changed schedule or trained")
                restored.policy.lr_decay = restored.train = forbidden
                run(env2,restored)
                self.assertEqual(optimizer_rates(restored.policy),before)
                report = json.loads((root/("eval"+str(enabled))/"moves_0/logs/TG_policy_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["source_learning_rate_schedule"]["kind"],"linear" if enabled else "fixed")
                self.assertEqual(report["profile"]["ppo_updates"],0)
                self.assertTrue(report["actor_unchanged"] and report["critic_unchanged"])

    def test_legacy_unrecorded_schedule_is_not_invented_on_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = make_environment(root/"train",length=4)
            learner = make_learner(env)
            run(env,learner)
            path = root/"train/moves_0/checkpoints/tg_policy_latest.pt"
            payload = torch.load(str(path),map_location="cpu")
            del payload["learning_rate_schedule"]
            payload["training_implementation"] = "tg_ppo_popart_v2"
            old = root/"old.pt"
            torch.save(payload,str(old))
            restored = make_learner(make_environment(root/"restore",length=4))
            restored.load_policy(old)
            self.assertIsNone(restored.evaluation_metadata["source_learning_rate_schedule"])
            self.assertEqual(restored.evaluation_metadata["source_training_implementation"],"tg_ppo_popart_v2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
