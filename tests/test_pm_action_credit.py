"""PM action target tests: hand calculations, no leakage/mutation, original PPO integration."""
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np
import torch
from buoy_network.pm_action_credit import packet_action_scores,target_distribution,collect_action_targets
from onpolicy.algorithms.utils.pm_action_loss import action_credit_settings,teacher_loss
from buoy_network.progress_runtime import FrozenPMRuntime
from buoy_network.wm_runtime import tensor_digest
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_pm_reward import checkpoint
from test_legacy_wm_actor import checkpoint as wm_checkpoint,run

def build(root,pm,mode='disabled',coef=.1,wm=None,extra=False):
    env=make_environment(root,length=4)
    a=env.all_args
    a.use_pm_reward=True;a.pm_reward_checkpoint=str(pm);a.pm_reward_alpha=.1
    a.pm_potential_mode='task_return';a.pm_value_mode='learned_shaped';a.tg_reward_version='task_v2'
    a.pm_action_credit=mode;a.pm_action_credit_coef=coef;a.pm_action_temperature=.2
    if extra:
        a.pm_critic_context='potential';a.tg_critic_time_mode='remaining'
    if wm is not None:
        a.use_wm_actor=True;a.wm_actor_checkpoint=str(wm)
    return env,make_learner(env)

class ActionFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_masked_temperature_stability_and_invalid_inputs(self):
        q=np.array([1000.,1001.,-999999.]);mask=[1,1,0]
        p=target_distribution(q,mask,1.)
        np.testing.assert_allclose(p,[1/(1+np.e),np.e/(1+np.e),0],atol=1e-7)
        for scores,legal,temp in ((q,[0,0,0],1),(q,[1,1,1],0),([float('nan')],[1],1)):
            with self.assertRaises(ValueError):target_distribution(scores,legal,temp)

    def test_teacher_gradient_is_hand_softmax_minus_target_and_target_detached(self):
        raw=torch.tensor([[.2,.4,-2.],[.1,.3,.2]],requires_grad=True)
        target=torch.tensor([[.75,.25,0.],[0.,0.,1.]],requires_grad=True)
        mask=torch.tensor([[1.],[0.]])
        loss=teacher_loss(torch.log_softmax(raw,-1),target,mask)
        loss.backward()
        expected=torch.softmax(raw.detach(),-1)-target.detach()
        expected[1]=0
        self.assertTrue(torch.allclose(raw.grad,expected,atol=1e-7))
        self.assertIsNone(target.grad)
        value=teacher_loss(torch.tensor([[-.2,float('-inf')]]),[[1.,0.]],[[0.]])
        self.assertEqual(float(value),0.)

    def test_settings_and_public_guard(self):
        import run_algorithm
        run_algorithm.reject_research_options(['--pm_action_credit','learned','--pm_action_credit_coef=.1','--pm_action_temperature','.2'])
        self.assertEqual(action_credit_settings(SimpleNamespace())['mode'],'disabled')
        for kwargs in (dict(pm_action_credit='bad'),dict(pm_action_credit='learned'),dict(pm_action_temperature=0)):
            with self.assertRaises(ValueError):action_credit_settings(SimpleNamespace(**kwargs))

    def test_terminal_scores_are_exact_packet_reward_and_no_future_queries(self):
        runtime=SimpleNamespace(potential_mode='task_return',step=3,started=True,n=2,horizon=4,
            hidden={7:torch.zeros(2)},prefix_state={7:{}},gamma=.99,task_contract={'delivery_weight':1.,'delay_weight':1.,'unfinished_weight':1.})
        actor={'distances':np.array([[0,1],[1,0]]),'obs':np.array([[0,1],[1,0]])}
        packet=SimpleNamespace(id=7,route=[0],destination=1)
        q=packet_action_scores(runtime,actor,0,packet,[0,1,1],3)
        self.assertAlmostEqual(float(q[1]),.75)
        self.assertAlmostEqual(float(q[2]),-1.25)

class ActionModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_lookahead_preserves_PM_history_environment_and_all_RNGs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pm=root/'pm.pt';checkpoint(pm)
            env,l=build(root/'run',pm,'learned')
            env.packet_generation();l.warmup();l.pm_runtime.start_episode(env,0)
            before=copy.deepcopy(l.pm_runtime.hidden);prefix=copy.deepcopy(l.pm_runtime.prefix_state)
            weights=tensor_digest(l.pm_runtime.model.state_dict())
            counts=[p.packet for p in env.buoys]
            rng=torch.get_rng_state().clone();prng=random.getstate();nrng=np.random.get_state()
            targets=collect_action_targets(l.pm_runtime,env,l.buffer,0)
            np.testing.assert_allclose(targets.sum(-1),1.,atol=1e-6)
            self.assertTrue(np.all(targets[l.buffer.available_actions[0,0]==0]==0))
            self.assertTrue(all(torch.equal(v,l.pm_runtime.hidden[k]) for k,v in before.items()))
            self.assertEqual(prefix,l.pm_runtime.prefix_state)
            self.assertEqual(weights,tensor_digest(l.pm_runtime.model.state_dict()))
            self.assertEqual(counts,[p.packet for p in env.buoys])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(prng,random.getstate());np.testing.assert_equal(nrng,np.random.get_state())

    def test_coefficient_zero_matches_original_policy_critic_losses_and_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pm=root/'pm.pt';checkpoint(pm)
            learners=[]
            for mode in ('disabled','learned'):
                torch.manual_seed(22);random.seed(22);np.random.seed(22)
                env,l=build(root/mode,pm,mode,0.)
                run(env,l);learners.append(l)
            for network in ('actor','critic'):
                self.assertEqual(tensor_digest(getattr(learners[0].policy,network).state_dict()),
                                 tensor_digest(getattr(learners[1].policy,network).state_dict()))
            a,b=learners[0].training_records,learners[1].training_records
            for x,y in zip(a,b):
                self.assertEqual(x,{k:v for k,v in y.items() if k!='pm_action_target_loss'})
            self.assertEqual((root/'disabled/moves_0/logs/TG_physical_metrics.jsonl').read_bytes(),
                             (root/'learned/moves_0/logs/TG_physical_metrics.jsonl').read_bytes())

    def test_shuffled_targets_keep_their_original_observation_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pm=root/'pm.pt';checkpoint(pm)
            _,l=build(root/'run',pm,'learned')
            b=l.buffer
            b.pm_action_targets.fill(0)
            for t in range(4):
                b.obs[t,:,:,0]=t
                b.pm_action_targets[t,:,:,t]=1.
                b.pm_action_ready[t]=True
            for sample in b.feed_forward_generator(np.zeros_like(b.rewards),2):
                np.testing.assert_array_equal(sample[-1].argmax(-1),sample[1][:,0])
            b.reset_episode()
            self.assertFalse(b.pm_action_ready.any())
            with self.assertRaises(ValueError):
                next(b.feed_forward_generator(np.zeros_like(b.rewards),2))

    def test_real_auxiliary_PPO_updates_only_policy_and_checkpoint_semantics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pm=root/'pm.pt';checkpoint(pm)
            env,l=build(root/'run',pm,'learned')
            run(env,l)
            self.assertGreater(l.training_records[0]['pm_action_target_loss'],0.)
            self.assertTrue(l.buffer.pm_action_ready.all())
            l.pm_runtime.assert_frozen()
            path=root/'run/moves_0/checkpoints/tg_policy_latest.pt'
            _,loaded=build(root/'load',pm,'learned');loaded.load_policy(path)
            _,bad=build(root/'bad',pm,'zero_model')
            with self.assertRaisesRegex(ValueError,'action credit configuration'):bad.load_policy(path)

    def test_WM_clock_critic_context_and_action_target_sample_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pm=root/'pm.pt';wm=root/'wm.pt';checkpoint(pm);wm_checkpoint(wm)
            env,l=build(root/'run',pm,'learned',wm=wm,extra=True)
            run(env,l);l.pm_runtime.assert_frozen();l.wm_runtime.assert_frozen()
            self.assertGreater(l.training_records[0]['pm_action_target_loss'],0.)

if __name__=='__main__':unittest.main(verbosity=2)
