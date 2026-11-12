import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_tg_sensitivity import VARIANTS,AXES,SEEDS,assert_one_factor,summarize,analyze,validate_resources


def rows():
    return [dict(variant=v,seed=s,coordinates=c,episodes=20,delivery=x,
        training_delivery=[x]*20,training_delivery_mean=x,reused=v in ['center','aux02'],checkpoint='frozen.pt',success_delay=5.,success_hops=3.,eval_loops=0)
        for v,c in VARIANTS.items() for s,x in [(11,.5),(22,.75),(33,1.)]]


class SensitivityPilotTests(unittest.TestCase):
    def test_cuda_names_are_equivalent_without_accepting_cpu_or_wrong_capacity(self):
        report=dict(learning_rate_schedule=dict(kind='fixed'),resources=dict(device='cuda',actor_trainable=24165,critic_trainable=18615))
        validate_resources(report);report['resources']['device']='cuda:0';validate_resources(report)
        for device in ['cpu','cuda:1']:
            report['resources']['device']=device
            with self.assertRaises(ValueError):validate_resources(report)
        report['resources']['device']='cuda';report['resources']['actor_trainable']=1
        with self.assertRaises(ValueError):validate_resources(report)

    def test_one_factor_grid_and_independent_seed_statistics(self):
        self.assertEqual(len(VARIANTS),7);self.assertEqual(len(AXES),3)
        for v,c in VARIANTS.items():assert_one_factor(v,c)
        with self.assertRaises(ValueError):assert_one_factor('alpha0',dict(alpha=0.,aux=0.,wm_scale=1.))
        result=summarize(rows());self.assertEqual(len(result),7)
        self.assertEqual(result[0]['n'],3);self.assertEqual(result[0]['mean'],.75);self.assertEqual(result[0]['sample_sd'],.25)
        self.assertEqual(sum(x['reused'] for x in result),6)
        for bad in [rows()[:-1],rows()[:-1]+[rows()[-2]]]:
            with self.assertRaises(ValueError):summarize(bad)
        bad=copy.deepcopy(rows());bad[0]['episodes']=300
        with self.assertRaises(ValueError):summarize(bad)
        bad=copy.deepcopy(rows());bad[0]['coordinates']['alpha']=1.
        with self.assertRaises(ValueError):summarize(bad)

    def test_csv_and_plots_do_not_claim_stable_or_blind_improvement(self):
        parent=ROOT/'results/thesis_test_temp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(parent)) as directory:
            root=Path(directory);analyze(root,dict(rows=rows()))
            r=json.loads((root/'report.json').read_text());self.assertEqual(r['models'],21);self.assertFalse(r['selected_new_default']);self.assertFalse(r['stable300_improvement_verified']);self.assertFalse(r['blind_test_used']);self.assertFalse(r['all_thesis_experiments_completed'])
            self.assertEqual(len((root/'per_model.csv').read_text().splitlines()),22)
            for name in ['sensitivity20','training_curves20']:
                for ext in ['png','svg','pdf']:self.assertGreater((root/(name+'.'+ext)).stat().st_size,1000)


if __name__=='__main__':unittest.main()
