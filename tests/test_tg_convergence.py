"""Convergence measurement checks: hand thresholds, censoring, AUC and read-only state hashes."""
import copy,json,random,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np,torch
from scripts.tg_convergence import checkpoint_grid,trailing_mean,threshold_time,normalized_auc,physical_return,learner_fingerprint,nested_digest
def point(e,y):return dict(episode=e,delivery=y,training_steps=e*80,training_seconds=e*2)
class ConvergenceFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_checkpoint_grid_and_invalid_budgets(self):
        self.assertEqual(checkpoint_grid(300,20),list(range(0,301,20)))
        self.assertEqual(checkpoint_grid(3,2),[0,2,3])
        for a,b in [(0,20),(300,0),(True,20)]:
            with self.assertRaises(ValueError):checkpoint_grid(a,b)
    def test_trailing_mean_never_reads_future_samples(self):
        self.assertEqual(trailing_mean([1,0,0,1],3),[None,None,1/3,1/3])
    def test_threshold_requires_consecutive_points_and_reports_confirmation(self):
        pts=[point(e,y) for e,y in [(0,.2),(20,.85),(40,.7),(60,.9),(80,.8)]]
        r=threshold_time(pts,.8)
        self.assertEqual(r['first_observed_episode'],20)
        self.assertEqual((r['sustained']['first_episode'],r['sustained']['confirmed_episode']),(60,80))
        self.assertIsNone(threshold_time([point(20,.9),point(60,.9)],.8)['sustained'])
    def test_unconfirmed_target_is_censored_not_zero(self):
        r=threshold_time([point(0,.1),point(300,1.)],.8,complete=True)
        self.assertEqual(r['first_observed_episode'],300);self.assertIsNone(r['sustained'])
        self.assertEqual(r['status'],'not_confirmed_by_budget')
    def test_auc_and_original_task_return_match_hand_calculation(self):
        self.assertAlmostEqual(normalized_auc([point(0,0),point(20,.5),point(40,1)]),.5)
        self.assertIsNone(normalized_auc([point(0,0)]))
        c=dict(delivery_weight=1,delay_weight=1,unfinished_weight=1)
        self.assertAlmostEqual(physical_return(dict(horizon=2,generated_packets=2,success_delays=[2],unresolved_packets=1),c),-.995)
    def test_fingerprints_preserve_rng_and_detect_optimizer_state_mutation(self):
        a,b=torch.nn.Linear(2,2),torch.nn.Linear(2,1)
        ao,bo=torch.optim.Adam(a.parameters()),torch.optim.Adam(b.parameters())
        model=SimpleNamespace(policy=SimpleNamespace(actor=a,critic=b,actor_optimizer=ao,critic_optimizer=bo),
            trainer=SimpleNamespace(value_normalizer=None,scaler=SimpleNamespace(state_dict=lambda:{})))
        first=learner_fingerprint(model)
        self.assertEqual(first,learner_fingerprint(model))
        self.assertEqual(nested_digest({'b':2,'a':1}),nested_digest({'a':1,'b':2}))
        a(torch.ones(1,2)).sum().backward();ao.step()
        second=learner_fingerprint(model);self.assertNotEqual(first['parameters_and_optimizer'],second['parameters_and_optimizer'])
        self.assertEqual(first['rng'],second['rng'])
        np.random.rand();self.assertNotEqual(second['rng'],learner_fingerprint(model)['rng'])
class ConvergenceModules(unittest.TestCase):
    def test_registered_budget_is300_and_known_seeds_only(self):
        c=json.loads((ROOT/'configs/tg_pm_convergence_300.json').read_text())
        self.assertEqual(c['episodes']*c['horizon'],24000)
        self.assertEqual(c['seeds'],[11,22,33])
        self.assertEqual(c['groups'],['TG','Immediate','PM'])
    def test_unapproved_fresh_seed_protocol_fails_before_creating_runs(self):
        from scripts.run_tg_convergence import prepare
        c=json.loads((ROOT/'configs/tg_pm_convergence_300.json').read_text());c['seeds']=[3041]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.json';p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'Unexpected long protocol'):prepare(p)
if __name__=='__main__':unittest.main(verbosity=2)
