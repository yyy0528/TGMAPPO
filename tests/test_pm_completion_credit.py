"""Versioned PM delivery credit: boundary algebra, runtime and actual PPO."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from buoy_network.progress_runtime import completion_potential,shaping_reward,FrozenPMRuntime
from test_legacy_pm_reward import checkpoint,learner_at
from test_legacy_wm_actor import run


class CreditFunctions(unittest.TestCase):
    def test_delivery_does_not_remove_completed_credit(self):
        before=.95/24
        old=shaping_reward(0,before,0,.99,1)[1]
        new=shaping_reward(0,before,completion_potential(0,1,24),.99,1)[1]
        self.assertLess(old,0)
        self.assertGreater(new,0)

    def test_terminal_discount_identity_remains_exact(self):
        phi=[.8,.7,.9,1.,0.]
        rewards=[shaping_reward(0,phi[t],phi[t+1],.99,1,terminated=t==3)[1] for t in range(4)]
        self.assertAlmostEqual(sum(.99**t*r for t,r in enumerate(rewards)),-phi[0],places=12)

    def test_invalid_mass_rejected(self):
        for args in ((1,2,1),(-1,0,1),(float('nan'),0,1)):
            with self.assertRaises(ValueError):
                completion_potential(*args)


class CreditModules(unittest.TestCase):
    def test_all_delivered_retains_credit_until_terminal(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            p=Path(temp)/'pm.pt'; checkpoint(p)
            runtime=FrozenPMRuntime(p,25,4,.99,1,'completion_credit')
            runtime.packet_ids={0};runtime.finished={0};runtime.initial_packets=1
            env=SimpleNamespace(packets=[])
            self.assertEqual(runtime.potential(env,2),1)
            self.assertEqual(runtime.potential(env,4),0)
            runtime.potential_mode='remaining'
            with self.assertRaises(ValueError):runtime.assert_frozen()


class CreditIntegration(unittest.TestCase):
    def test_real_ppo_freeze_metadata_reload_and_mode_rejection(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp);p=root/'pm.pt';checkpoint(p)
            env,learner=learner_at(root/'train',p)
            # Build again using the new argument so the actual runner uses v2.
            from test_legacy_targeted_fixes import make_learner
            env.all_args.pm_potential_mode='completion_credit'
            learner=make_learner(env)
            run(env,learner)
            self.assertTrue(learner.pm_update_records[0]['gae_uses_shaped_rewards'])
            self.assertTrue(learner.pm_update_records[0]['actor_changed'])
            self.assertEqual(learner.pm_runtime.metadata()['potential_mode'],'completion_credit')
            self.assertLess(learner.pm_runtime.summaries[0]['boundary_error'],1e-7)
            saved=root/'train/moves_0/checkpoints/tg_policy_latest.pt'
            learner.load_policy(saved)
            _,old=learner_at(root/'old',p)
            with self.assertRaisesRegex(ValueError,'potential mode'):old.load_policy(saved)


if __name__=='__main__':unittest.main(verbosity=2)
