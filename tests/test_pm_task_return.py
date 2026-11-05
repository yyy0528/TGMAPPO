"""Independent event enumeration before module and PPO integration tests."""
import sys
import random
import tempfile
import unittest
from pathlib import Path
import torch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tests')]
from buoy_network.pm_return import expected_task_return
from buoy_network.task_reward import reward_contract
from types import SimpleNamespace


class ReturnFunctions(unittest.TestCase):
    def test_expected_return_matches_independent_outcome_enumeration(self):
        h = 4
        contract = reward_contract(SimpleNamespace(tg_reward_version='task_v2',
            tg_delivery_weight=2., tg_delay_weight=.7, tg_unfinished_weight=1.3))
        hazard = [.2, .4, .3, .8]
        logits = torch.logit(torch.tensor(hazard, dtype=torch.float64)) if hasattr(torch, 'logit') else torch.log(torch.tensor(hazard)/(1-torch.tensor(hazard)))
        for gamma in (.5, .99, 1.):
            for left in range(h+1):
                survival, expected = 1., 0.
                for k in range(1, left+1):
                    reward = 2*gamma**(k-1)-.7/h*sum(gamma**j for j in range(k))
                    expected += survival*hazard[k-1]*reward
                    survival *= 1-hazard[k-1]
                if left:
                    expected += survival*(-1.3*gamma**(left-1)-.7/h*sum(gamma**j for j in range(left)))
                actual = expected_task_return(logits, left, h, gamma, contract)
                self.assertAlmostEqual(float(actual['task_return']), expected, places=7)

    def test_delivery_slot_cost_terminal_and_extreme_hazards(self):
        c = reward_contract(SimpleNamespace(tg_reward_version='task_v2'))
        logits = torch.tensor([[80.]*80, [-80.]*80, [-80.]*80])
        result = expected_task_return(logits, torch.tensor([80,80,0]), 80, .99, c)
        self.assertAlmostEqual(float(result['task_return'][0]), 1-1/80, places=10)
        self.assertAlmostEqual(float(result['task_return'][1]), -sum(.99**k for k in range(80))/80-.99**79, places=10)
        self.assertEqual(float(result['task_return'][2]), 0.)
        self.assertTrue(all(torch.isfinite(x).all() for x in result.values()))

    def test_invalid_contract_window_or_nonfinite_rejected(self):
        c = reward_contract(SimpleNamespace(tg_reward_version='task_v2'))
        for left in (-1, .5, 5, float('nan')):
            with self.assertRaises(ValueError): expected_task_return(torch.zeros(4), left, 4, .99, c)
        for values, contract in ((torch.full((4,), float('nan')), c), (torch.zeros(4), {'version':'legacy_v1'})):
            with self.assertRaises(ValueError): expected_task_return(values, 4, 4, .99, contract)


class ReturnModules(unittest.TestCase):
    def test_runtime_team_potential_matches_enumerated_remaining_packets(self):
        from buoy_network.progress_runtime import FrozenPMRuntime, shaping_reward
        from test_legacy_pm_reward import checkpoint
        from test_legacy_targeted_fixes import make_environment
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp); path=root/'pm.pt'; checkpoint(path)
            env=make_environment(root/'env')
            env.packet_generation()
            c=reward_contract(SimpleNamespace(tg_reward_version='task_v2'))
            runtime=FrozenPMRuntime(path,25,4,.99,1,'task_return',task_contract=c)
            captured=[]
            original=runtime.model.step
            def observed(features, hidden):
                logits, following=original(features,hidden);captured.append(logits.clone())
                return logits,following
            runtime.model.step=observed
            runtime.start_episode(env,0)
            expected=expected_task_return(captured[0],torch.full((runtime.initial_packets,),4,dtype=torch.long),4,.99,c)
            self.assertAlmostEqual(runtime.phi,float(expected['task_return'].mean()),places=12)
            rewards=[shaping_reward(0,x,y,.99,1,terminated=t==2)[1]
                     for t,(x,y) in enumerate(zip([runtime.phi,-.7,-.3],[-.7,-.3,0.]))]
            self.assertAlmostEqual(sum(.99**t*r for t,r in enumerate(rewards)),-runtime.phi,places=12)
            runtime.finished=set(runtime.packet_ids)
            self.assertEqual(runtime.potential(SimpleNamespace(packets=[]),2),0)
            runtime.task_contract['delay_weight']=2
            with self.assertRaises(ValueError): runtime.assert_frozen()

    def test_wrong_task_contract_is_rejected(self):
        from buoy_network.progress_runtime import FrozenPMRuntime
        for c in (None,{'version':'legacy_v1'}):
            with self.assertRaisesRegex(ValueError,'contract'):
                FrozenPMRuntime('not-loaded.pt',25,80,.99,1,'task_return',task_contract=c)


class ReturnIntegration(unittest.TestCase):
    def test_actual_ppo_updates_popart_and_reload_guards(self):
        from buoy_network.wm_runtime import tensor_digest
        from test_legacy_pm_reward import checkpoint
        from test_legacy_targeted_fixes import make_environment, make_learner
        from test_legacy_wm_actor import run
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);path=root/'pm.pt';checkpoint(path)
            outputs=[]
            for mode in ('remaining','task_return'):
                random.seed(31);np.random.seed(31);torch.manual_seed(31)
                env=make_environment(root/mode);args=env.all_args
                args.tg_reward_version='task_v2';args.use_pm_reward=True
                args.pm_reward_checkpoint=str(path);args.pm_reward_alpha=1.
                args.pm_potential_mode=mode;args.num_episodes=2
                learner=make_learner(env);run(env,learner)
                learner.pm_runtime.assert_frozen()
                self.assertEqual(int(learner.policy.critic_optimizer.state[learner.policy.critic.v_out.weight]['step']),4)
                self.assertTrue(all(r['gae_uses_shaped_rewards'] and r['actor_changed'] for r in learner.pm_update_records))
                self.assertTrue(all(s['boundary_error']<1e-7 for s in learner.pm_runtime.summaries))
                outputs.append(tensor_digest(learner.policy.actor.state_dict()))
                if mode=='task_return':
                    saved=root/mode/'moves_0/checkpoints/tg_policy_latest.pt'
                    learner.load_policy(saved)
                    args.pm_potential_mode='remaining'
                    old=make_learner(env)
                    with self.assertRaisesRegex(ValueError,'potential mode'): old.load_policy(saved)
            self.assertNotEqual(*outputs)


if __name__ == '__main__': unittest.main(verbosity=2)
