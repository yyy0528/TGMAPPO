"""PM value coordinates, PopArt targets, and real baseline-equivalent PPO."""
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
from buoy_network.pm_value import coordinate_audit,prepare_residual_rollout
from buoy_network.wm_runtime import tensor_digest
from onpolicy.algorithms.utils.popart import PopArt
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_pm_reward import checkpoint
from test_legacy_wm_actor import run,checkpoint as wm_checkpoint


class ValueFunctions(unittest.TestCase):
    def test_same_cached_trajectory_td_gae_identity_all_lambdas(self):
        rng=np.random.RandomState(17)
        rewards=rng.randn(80,3,1)*.1
        values=rng.randn(81,3,1)
        phi=rng.rand(81,3,1);phi[-1]=0
        masks=np.ones_like(values);masks[-1]=0
        for alpha in (0,.1,1.):
            shaped=rewards+alpha*(.99*phi[1:]-phi[:-1])
            for lam in (0,.95,1.):
                a=coordinate_audit(rewards,shaped,values,phi,masks,.99,lam,alpha)
                np.testing.assert_allclose(a['base_delta'],a['aligned_delta'],atol=2e-14)
                np.testing.assert_allclose(a['base_gae'],a['aligned_gae'],atol=2e-14)
        self.assertGreater(np.max(np.abs(a['base_gae']-a['legacy_same_numeric_value_gae'])),.1)

    def test_monte_carlo_targets_shift_but_advantages_do_not(self):
        r=np.array([[.1],[.2],[-.4]])
        v=np.array([[.8],[.6],[.2],[0.]])
        phi=np.array([[.7],[.3],[.1],[0.]])
        m=np.ones_like(v);m[-1]=0
        a=coordinate_audit(r,r+.99*phi[1:]-phi[:-1],v,phi,m,.99,1,1)
        base_return=a['base_gae']+v[:-1]
        shaped_return=a['aligned_gae']+a['shaped_values'][:-1]
        np.testing.assert_allclose(shaped_return,base_return-phi[:-1],atol=1e-14)

    def test_inconsistent_cache_or_nonfinite_values_rejected(self):
        r=np.zeros((3,1));v=np.zeros((4,1));m=np.ones_like(v);m[-1]=0
        with self.assertRaises(ValueError):coordinate_audit(r,r+1,v,v,m,.99,.95,1)
        v[0]=np.nan
        with self.assertRaises(ValueError):coordinate_audit(r,r,v,v,m,.99,.95,1)


class ValueModules(unittest.TestCase):
    def test_popart_audit_uses_raw_values_and_keeps_cache_immutable(self):
        torch.set_num_threads(1)
        normalizer=PopArt(2,1)
        normalizer.update(torch.tensor([[90.],[100.],[110.]]))
        shape=(4,1,2,1)
        base=np.full(shape,.1,dtype=np.float32)
        phi=np.array([.8,.7,.4,.2,0.])
        shaped=base+(.99*phi[1:]-phi[:-1]).reshape(4,1,1,1)
        masks=np.ones((5,1,2,1),dtype=np.float32);masks[-1]=0
        b=SimpleNamespace(episode_length=4,_use_gae=True,_use_popart=True,_use_valuenorm=False,
            rewards=shaped.astype(np.float32),masks=masks,bad_masks=np.ones_like(masks),
            value_preds=np.full_like(masks,.2),gamma=.99,gae_lambda=.95)
        rows=[dict(step=i,terminated=i==3,rollout_truncated=False,phi_t=phi[i],phi_tp1=phi[i+1],
                   reward_env=float(base[i,0,0,0]),reward_train=float(shaped[i,0,0,0])) for i in range(4)]
        runtime=SimpleNamespace(episode_records=rows,alpha=1)
        old=b.rewards.copy();state=tensor_digest(normalizer.state_dict())
        override,report=prepare_residual_rollout(b,runtime,normalizer)
        np.testing.assert_array_equal(override,base)
        np.testing.assert_array_equal(b.rewards,old)
        self.assertEqual(state,tensor_digest(normalizer.state_dict()))
        self.assertLess(report['raw_gae_identity_max_error'],1e-10)
        rows[1]['phi_t']+=.01
        with self.assertRaisesRegex(ValueError,'cache'):prepare_residual_rollout(b,runtime,normalizer)

    def test_buffer_override_does_not_relabel_collected_rewards(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            env=make_environment(temp);learner=make_learner(env);b=learner.buffer
            b.rewards[:]=2.;b.masks[-1]=0
            old=b.rewards.copy()
            b.compute_returns(np.zeros_like(b.value_preds[-1]),learner.trainer.value_normalizer,
                              rewards_override=np.zeros_like(b.rewards))
            np.testing.assert_array_equal(b.rewards,old)
            with self.assertRaises(ValueError):b.compute_returns(np.zeros_like(b.value_preds[-1]),rewards_override=np.zeros(1))


class ValueIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_actual_ppo_residual_matches_baseline_with_popart_and_plain_values(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);path=root/'pm.pt';checkpoint(path)
            for popart,valuenorm in ((False,False),(True,False),(False,True)):
                outputs=[]
                for mode in ('off','learned_shaped','base_residual'):
                    random.seed(17);np.random.seed(17);torch.manual_seed(17)
                    label=str(popart)+str(valuenorm)+mode
                    env=make_environment(root/label)
                    args=env.all_args
                    args.use_popart=popart;args.use_valuenorm=valuenorm;args.use_linear_lr_decay=False
                    args.num_episodes=2;args.use_pm_reward=mode!='off'
                    args.pm_reward_checkpoint=str(path) if mode!='off' else ''
                    args.pm_value_mode=mode if mode!='off' else 'learned_shaped'
                    learner=make_learner(env)
                    run(env,learner)
                    outputs.append({n:tensor_digest(getattr(learner.policy,n).state_dict()) for n in ('actor','critic')})
                    if valuenorm:
                        outputs[-1]['normalizer']=tensor_digest(learner.trainer.value_normalizer.state_dict())
                    if mode=='base_residual':
                        self.assertEqual(int(learner.policy.critic_optimizer.state[learner.policy.critic.v_out.weight]['step']),4)
                        for record in learner.pm_update_records:
                            self.assertFalse(record['gae_uses_shaped_rewards'])
                            self.assertLess(record['value_coordinates']['raw_gae_identity_max_error'],1e-9)
                        learner.pm_runtime.assert_frozen()
                        saved=root/label/'moves_0/checkpoints/tg_policy_latest.pt'
                        learner.load_policy(saved)
                self.assertEqual(outputs[0],outputs[2])
                self.assertNotEqual(outputs[0]['actor'],outputs[1]['actor'])

    def test_wm_pm_popart_and_value_mode_checkpoint_guard(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);p=root/'pm.pt';w=root/'wm.pt';checkpoint(p);wm_checkpoint(w)
            outcomes=[]
            for mode in ('off','base_residual'):
                random.seed(22);np.random.seed(22);torch.manual_seed(22)
                env=make_environment(root/mode)
                args=env.all_args;args.use_wm_actor=True;args.wm_actor_checkpoint=str(w)
                args.use_pm_reward=mode!='off';args.pm_reward_checkpoint=str(p) if mode!='off' else ''
                args.pm_value_mode=mode if mode!='off' else 'learned_shaped'
                learner=make_learner(env);run(env,learner)
                outcomes.append({n:tensor_digest(getattr(learner.policy,n).state_dict()) for n in ('actor','critic')})
            self.assertEqual(outcomes[0],outcomes[1])
            learner.pm_runtime.value_mode='learned_shaped'
            # Build a valid legacy-mode runtime instead of mutating its freeze signature.
            args.pm_value_mode='learned_shaped'
            legacy=make_learner(env)
            with self.assertRaisesRegex(ValueError,'value coordinate'):
                legacy.load_policy(root/'base_residual/moves_0/checkpoints/tg_policy_latest.pt')


if __name__=='__main__':unittest.main(verbosity=2)
