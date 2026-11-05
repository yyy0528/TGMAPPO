"""Independent calibration decisions, frozen runtime, then original PPO."""
import copy
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np
import torch
from buoy_network.pm_reliability import SCHEMA,select_gate,load_gate
from buoy_network.task_reward import reward_contract
from buoy_network.wm_runtime import tensor_digest
from types import SimpleNamespace


def episodes(good=True,horizon=4,nodes=25):
    r=np.full(horizon,-.1);mc=np.zeros(horizon)
    for t in reversed(range(horizon)):mc[t]=r[t]+(.99*mc[t+1] if t+1<horizon else 0)
    phi=np.r_[mc if good else -mc,0.]
    return [dict(scene=scene,episode=ep,rewards=r.tolist(),values=np.zeros((horizon,nodes)).tolist(),phi=phi.tolist())
            for scene in (4101,4102) for ep in (0,1)]


def gate_report(path,pm,learner,good=True):
    from buoy_network.policy_artifacts import signature
    data=episodes(good)
    payload={'schema':SCHEMA,'phase':'independent_initial_policy_calibration',
        'horizon':4,'nodes':25,'gamma':.99,'maximum_alpha':1.,
        'pm_checkpoint_sha256':hashlib.sha256(Path(pm).read_bytes()).hexdigest(),
        'reward_contract':reward_contract(learner.all_args),'potential_mode':'task_return',
        'value_mode':'learned_shaped','weights_unchanged':True,'optimizer_updates':0,
        'training_seed':17,'initial_actor_sha256':tensor_digest(learner.policy.actor.state_dict()),
        'initial_critic_sha256':tensor_digest(learner.policy.critic.state_dict()),
        'policy_signature':signature(learner),'episodes':data,'synthetic_test_fixture':True,
        'decision':select_gate(data,4,25,.99,1.)}
    Path(path).write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
    return payload


def fresh(root,pm=None,gate=None):
    from test_legacy_targeted_fixes import make_environment,make_learner
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    env=make_environment(root);a=env.all_args
    a.seed=17;a.tg_reward_version='task_v2';a.num_episodes=2
    a.use_pm_reward=pm is not None;a.pm_reward_checkpoint=str(pm or '')
    a.pm_reward_alpha=1.;a.pm_potential_mode='task_return' if pm else 'remaining'
    a.pm_reliability_report=str(gate or '')
    return env,make_learner(env)


class ReliabilityFunctions(unittest.TestCase):
    def test_good_prior_enabled_bad_or_zero_prior_disabled(self):
        self.assertEqual(select_gate(episodes(True),4,25,.99,1.)['effective_alpha'],1.)
        self.assertEqual(select_gate(episodes(False),4,25,.99,1.)['effective_alpha'],0.)
        data=episodes(True)
        for row in data:row['phi']=[0.]*5
        self.assertFalse(select_gate(data,4,25,.99,1.)['enabled'])

    def test_one_bad_episode_rejects_an_average_improvement(self):
        data=episodes(True)
        data[0]['phi']=[-.1*x for x in data[0]['phi']]
        result=select_gate(data,4,25,.99,1.)
        self.assertLess(result['pm_prior_mse'],result['baseline_mse'])
        self.assertFalse(result['enabled'])

    def test_missing_duplicate_nonfinite_or_terminal_data_rejected(self):
        for mutate in ('few','duplicate','nan','terminal','shape'):
            data=episodes(True)
            if mutate=='few':data=data[:2]
            elif mutate=='duplicate':data[0]=copy.deepcopy(data[1])
            elif mutate=='nan':data[0]['values'][0][0]=float('nan')
            elif mutate=='terminal':data[0]['phi'][-1]=1
            else:data[0]['rewards'].pop()
            with self.assertRaises(ValueError):select_gate(data,4,25,.99,1.)


class ReliabilityModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_report_recomputed_and_model_phase_tampering_rejected(self):
        from test_legacy_pm_reward import checkpoint
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);p=root/'pm.pt';checkpoint(p)
            _,initial=fresh(root/'init',p);report=root/'gate.json'
            original=gate_report(report,p,initial)
            spec=(report,initial.pm_runtime.file_sha256,4,25,.99,1.,reward_contract(initial.all_args))
            self.assertEqual(load_gate(*spec)['effective_alpha'],1.)
            for field,value in (('optimizer_updates',1),('phase','PPO_rollout'),('gamma',.9),('pm_checkpoint_sha256','0'*64)):
                report.write_text(json.dumps(dict(original,**{field:value})),encoding='utf-8')
                with self.assertRaises(ValueError):load_gate(*spec)
            bad=copy.deepcopy(original);bad['decision']['effective_alpha']=0
            report.write_text(json.dumps(bad),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'decision'):load_gate(*spec)

    def test_runtime_gate_is_frozen_and_report_file_changes_rejected(self):
        from test_legacy_pm_reward import checkpoint
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);p=root/'pm.pt';checkpoint(p)
            _,initial=fresh(root/'init',p);report=root/'gate.json';gate_report(report,p,initial,False)
            _,learner=fresh(root/'gated',p,report)
            self.assertEqual(learner.pm_runtime.alpha,0.)
            learner.pm_runtime.reliability['enabled']=True
            with self.assertRaises(ValueError):learner.pm_runtime.assert_frozen()
            _,learner=fresh(root/'gated_again',p,report)
            report.write_text(report.read_text(encoding='utf-8')+' ',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'report changed'):learner.pm_runtime.metadata()


class ReliabilityIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_real_ppo_gate_off_equals_baseline_gate_on_equals_always_PM(self):
        from test_legacy_pm_reward import checkpoint
        from test_legacy_wm_actor import run
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);p=root/'pm.pt';checkpoint(p)
            _,initial=fresh(root/'init',p)
            good,bad=root/'good.json',root/'bad.json'
            gate_report(good,p,initial,True);gate_report(bad,p,initial,False)
            states={}
            for name,pm,gate in (('off',None,None),('always',p,None),('gated_off',p,bad),('gated_on',p,good)):
                env,learner=fresh(root/name,pm,gate);run(env,learner)
                states[name]={n:tensor_digest(getattr(learner.policy,n).state_dict()) for n in ('actor','critic')}
                self.assertEqual(int(learner.policy.critic_optimizer.state[learner.policy.critic.v_out.weight]['step']),4)
                if gate:
                    learner.pm_runtime.assert_frozen()
                    learner.load_policy(root/name/'moves_0/checkpoints/tg_policy_latest.pt')
            self.assertEqual(states['off'],states['gated_off'])
            self.assertEqual(states['always'],states['gated_on'])

    def test_initialization_and_checkpoint_calibration_mismatches_rejected(self):
        from test_legacy_pm_reward import checkpoint
        from test_legacy_wm_actor import run
        from test_legacy_targeted_fixes import make_learner
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);p=root/'pm.pt';checkpoint(p)
            _,initial=fresh(root/'init',p);g=root/'good.json';gate_report(g,p,initial)
            env,learner=fresh(root/'train',p,g);run(env,learner)
            _,plain=fresh(root/'plain',p)
            with self.assertRaisesRegex(ValueError,'reliability'):
                plain.load_policy(root/'train/moves_0/checkpoints/tg_policy_latest.pt')
            env.all_args.seed=18
            with self.assertRaisesRegex(ValueError,'initialization'):make_learner(env)

    def test_actual_independent_calibration_is_sampled_and_never_updates(self):
        from test_legacy_pm_reward import checkpoint
        from buoy_network.pm_calibration import collect_initial
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);p=root/'pm.pt';checkpoint(p)
            _,initial=fresh(root/'init',p)
            config=vars(initial.all_args).copy()
            parts=[collect_initial(config,root/('scene_'+str(scene)),17,scene) for scene in (4101,4102)]
            self.assertEqual(parts[0]['initial_actor_sha256'],parts[1]['initial_actor_sha256'])
            self.assertEqual(parts[0]['initial_actor_sha256'],tensor_digest(initial.policy.actor.state_dict()))
            self.assertTrue(all(x['optimizer_updates']==0 and x['weights_unchanged'] and x['collect_calls']==8 for x in parts))
            for scene in (4101,4102):
                actual=json.loads((root/('scene_'+str(scene))/'moves_0/logs/TG_MAPPO_environment.json').read_text(encoding='utf-8'))['config']
                self.assertEqual(actual['num_env_steps'],8)
                self.assertEqual(actual['effective_num_env_steps'],8)
            data=[row for x in parts for row in x['episodes']]
            select_gate(data,4,25,.99,1.)


if __name__=='__main__':unittest.main(verbosity=2)
