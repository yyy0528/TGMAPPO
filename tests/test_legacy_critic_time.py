import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'src',ROOT/'tests'):
    sys.path.insert(0,str(path))
from onpolicy.algorithms.utils.critic_time import time_mode,task_clock,CriticTime
from buoy_network.wm_runtime import tensor_digest
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_wm_actor import run,checkpoint as wm_checkpoint
from test_legacy_pm_reward import checkpoint as pm_checkpoint


def learner(root,mode,episodes=1):
    random.seed(33)
    np.random.seed(33)
    torch.manual_seed(33)
    env = make_environment(root,length=4)
    env.all_args.num_episodes = episodes
    env.all_args.tg_critic_time_mode = mode
    env.all_args.tg_reward_version = 'task_v2'
    return env,make_learner(env)


class TimeFunctions(unittest.TestCase):
    def test_clock_decision_and_terminal_indices(self):
        x=task_clock(4,2,3)
        self.assertEqual(x.shape,(5,2,3,1))
        np.testing.assert_array_equal(x[:,0,0,0],[1,.75,.5,.25,0])
        x[0,0,0]=0
        self.assertEqual(x[0,1,0],1)

    def test_initial_identity_rng_and_trainable_branch(self):
        rng=torch.get_rng_state().clone()
        model=CriticTime(3,'remaining')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        h=torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True)
        t=torch.tensor([[1.],[0.]],requires_grad=True)
        self.assertTrue(torch.equal(model(h,t),h))
        model(h,t).sum().backward()
        self.assertIsNone(t.grad)
        self.assertIsNotNone(model.modulation.grad)
        self.assertEqual(model.modulation.numel(),6)

    def test_remaining_changes_output_constant_ignores_clock(self):
        h=torch.zeros(2,3)
        t=torch.tensor([[1.],[0.]])
        outputs=[]
        for mode in ('remaining','constant'):
            model=CriticTime(3,mode)
            with torch.no_grad():
                model.modulation[1].fill_(1)
            outputs.append(model(h,t))
        self.assertFalse(torch.equal(outputs[0][0],outputs[0][1]))
        self.assertTrue(torch.equal(outputs[1][0],outputs[1][1]))

    def test_invalid_modes_shapes_and_ranges(self):
        self.assertEqual(time_mode(SimpleNamespace()),'disabled')
        for args in (SimpleNamespace(tg_critic_time_mode='bad'),
                     SimpleNamespace(tg_critic_time_mode='remaining',use_recurrent_policy=True)):
            with self.assertRaises(ValueError):
                time_mode(args)
        for dims in ((0,1,1),(True,1,1),(4,0,1)):
            with self.assertRaises(ValueError):
                task_clock(*dims)
        model=CriticTime(3,'remaining')
        for t in (None,torch.zeros(2),torch.tensor([[1.1],[0.]]),torch.tensor([[float('nan')],[0.]])):
            with self.assertRaises(ValueError):
                model(torch.zeros(2,3),t)


class TimeModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_original_initialization_and_equal_capacity_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            states=[]
            for mode in ('disabled','constant','remaining'):
                _,model=learner(Path(temp)/mode,mode)
                states.append((tensor_digest(model.policy.actor.state_dict()),model.policy.critic.state_dict()))
            self.assertEqual(len({x[0] for x in states}),1)
            for key,value in states[0][1].items():
                self.assertTrue(all(torch.equal(value,x[1][key]) for x in states[1:]),key)
            self.assertEqual(set(states[1][1]),set(states[2][1]))

    def test_buffer_clock_alignment_reset_and_shuffled_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            _,model=learner(Path(temp),'remaining')
            b=model.buffer
            for step in range(4):
                b.obs[step].fill(step)
            samples=list(b.feed_forward_generator(np.zeros_like(b.rewards),num_mini_batch=2))
            for sample in samples:
                self.assertEqual(len(sample),17)
                np.testing.assert_array_equal(sample[-1][:,0],1-sample[1][:,0]/4)
            b.critic_time.fill(0)
            b.reset_episode()
            np.testing.assert_array_equal(b.critic_time[:,0,0,0],[1,.75,.5,.25,0])

    def test_checkpoint_clock_mode_guard_and_old_missing_field(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for mode in ('disabled','constant','remaining'):
                env,model=learner(root/mode,mode)
                run(env,model)
                path=root/mode/('moves_'+str(env.all_args.num_moving_buoys))/'checkpoints/tg_policy_latest.pt'
                _,loaded=learner(root/(mode+'_load'),mode)
                loaded.load_policy(path)
                self.assertEqual(tensor_digest(model.policy.critic.state_dict()),tensor_digest(loaded.policy.critic.state_dict()))
                _,wrong=learner(root/(mode+'_wrong'),'constant' if mode!='constant' else 'remaining')
                with self.assertRaisesRegex(ValueError,'clock'):
                    wrong.load_policy(path)
                if mode=='disabled':
                    payload=torch.load(str(path))
                    payload.pop('critic_time_mode')
                    old=root/'old.pt'
                    torch.save(payload,str(old))
                    loaded.load_policy(old)


class TimeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_first_rollout_and_actor_update_identical_clock_branch_learns(self):
        with tempfile.TemporaryDirectory() as temp:
            actors,physics=[] ,[]
            for mode in ('disabled','constant','remaining'):
                root=Path(temp)/mode
                env,model=learner(root,mode)
                run(env,model)
                actors.append(tensor_digest(model.policy.actor.state_dict()))
                physics.append((root/('moves_'+str(env.all_args.num_moving_buoys))/'logs/TG_physical_metrics.jsonl').read_bytes())
                self.assertEqual(model.training_records[-1]['actor_optimizer_steps'],2)
                if mode!='disabled':
                    self.assertGreater(float(model.policy.critic.time_conditioning.modulation.abs().sum()),0)
            self.assertEqual(len(set(actors)),1)
            self.assertEqual(len(set(physics)),1)

    def test_time_with_frozen_wm_pm_and_four_epoch_ppo(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            wm,pm=root/'wm.pt',root/'pm.pt'
            wm_checkpoint(wm)
            pm_checkpoint(pm)
            env=make_environment(root/'run',length=4)
            a=env.all_args
            a.tg_critic_time_mode,a.tg_reward_version,a.ppo_epoch='remaining','task_v2',4
            a.use_wm_actor,a.wm_actor_checkpoint=True,str(wm)
            a.use_pm_reward,a.pm_reward_checkpoint=True,str(pm)
            model=make_learner(env)
            run(env,model)
            model.wm_runtime.assert_frozen()
            model.pm_runtime.assert_frozen()
            self.assertEqual(model.training_records[-1]['critic_head_optimizer_steps'],8)


if __name__=='__main__':
    unittest.main(verbosity=2)
