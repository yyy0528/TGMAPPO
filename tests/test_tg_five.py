"""Registered five-group budgets, guards, capacity and shared initialization."""
import copy,json,random,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import numpy as np,torch
from tg_five import validate,budget,capacity,VARIANTS,comparable_initial
from test_legacy_wm_actor import checkpoint as wm_checkpoint
from test_legacy_pm_reward import checkpoint as pm_checkpoint
from test_legacy_targeted_fixes import make_environment,make_learner

class FiveProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_registered_full_and_probe_budgets(self):
        full=validate(json.loads((ROOT/'configs/tg_five_gpu_300.json').read_text()))
        self.assertEqual(budget(full),dict(training_models=15,training_steps=360000,evaluation_steps=115200))
        probe=validate(json.loads((ROOT/'configs/tg_five_gpu_probe.json').read_text()))
        self.assertEqual(budget(probe),dict(training_models=5,training_steps=800,evaluation_steps=1200))
    def test_rejects_wrong_groups_fresh_seeds_and_budget(self):
        c=json.loads((ROOT/'configs/tg_five_gpu_300.json').read_text())
        for key,value in [('groups',['TG','PM']),('seeds',[3041]),('episodes',299),('device','cpu'),('interval',100)]:
            invalid=copy.deepcopy(c);invalid[key]=value
            with self.assertRaises(ValueError):validate(invalid)
    def test_wm_capacity_is_reported_not_hidden(self):
        self.assertEqual(capacity('TG'),(21956,18615))
        self.assertEqual(capacity('WM'),(24165,18615))
        self.assertEqual(capacity('PM_WM'),capacity('WM'))
    def test_five_groups_share_backbone_critic_and_rng_with_zero_wm_residual(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);wm=root/'wm.pt';pm=root/'pm.pt';wm_checkpoint(wm);pm_checkpoint(pm);fingerprints=[]
            for group,v in VARIANTS.items():
                random.seed(22);np.random.seed(22);torch.manual_seed(22)
                env=make_environment(root/group,length=4);a=env.all_args;a.tg_reward_version='task_v2'
                a.use_wm_actor=v['wm'];a.wm_actor_checkpoint=str(wm) if v['wm'] else ''
                a.use_pm_reward=v['pm'];a.pm_reward_checkpoint=str(pm) if v['pm'] else '';a.pm_reward_alpha=.1
                a.pm_potential_mode='task_return' if v['pm'] else 'remaining'
                a.pm_value_mode='base_residual' if v['pm'] else 'learned_shaped';a.pm_action_credit=v['mode']
                model=make_learner(env);fingerprints.append(comparable_initial(model))
            self.assertTrue(all(x==fingerprints[0] for x in fingerprints))

if __name__=='__main__':unittest.main()
