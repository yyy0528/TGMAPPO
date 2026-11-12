import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_component_failure_extension import summarize,analyze


def rows():
    return [dict(algorithm=g,mode=m,condition=c,seed=s,scene=9402,status='completed',delivery=v,
        optimizer_updates=0,weights_unchanged=True,frozen_replay_exact=m=='factual',wait_decisions=3,
        waits_outside_teacher_set=2,loop_transfers=0,wm_changes_action=1,wm_introduces_wait=0,node_groups={'4':dict(wait=3)})
        for c in ['loss_high','walk_fast'] for g,m in [('PM','factual'),('PM_WM','factual'),('PM_WM','no_wm_residual')] for s,v in [(11,.5),(22,.75),(33,1.)]]


class FailureExtensionTests(unittest.TestCase):
    def test_three_seed_statistics_and_incomplete_duplicate_scene_rejection(self):
        r=rows();cells=summarize(r);self.assertEqual(len(cells),6)
        self.assertEqual(cells[0]['seeds'],[11,22,33]);self.assertEqual(cells[0]['mean'],.75);self.assertEqual(cells[0]['sample_sd'],.25)
        self.assertEqual(cells[0]['wait_nodes'],[('4',9)])
        with self.assertRaises(ValueError):summarize(r[:-1])
        r[-1]=dict(r[-2])
        with self.assertRaises(ValueError):summarize(r)
        r=rows();r[0]['scene']=9401
        with self.assertRaises(ValueError):summarize(r)

    def test_frozen_checks_and_figures(self):
        r=rows();r[0]['frozen_replay_exact']=False
        with self.assertRaises(ValueError):summarize(r)
        r=rows();r[0]['optimizer_updates']=1
        with self.assertRaises(ValueError):summarize(r)
        parent=ROOT/'results/thesis_test_temp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(parent)) as directory:
            root=Path(directory);analyze(root,dict(rows=rows(),base='frozen_base'))
            report=json.loads((root/'report.json').read_text());self.assertTrue(report['engineering_passed']);self.assertTrue(report['offline_test_reused'])
            self.assertFalse(report['all_thesis_experiments_completed'])
            for ext in ['png','svg','pdf']:self.assertGreater((root/('failure_scene9402.'+ext)).stat().st_size,1000)
            self.assertEqual(len((root/'case_metrics.csv').read_text().splitlines()),19)


if __name__=='__main__':unittest.main()
