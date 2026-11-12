"""A small fixture checks mean/SD, per-seed exports and figure generation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_thesis_experiments import summarize, extra_args, compatibility_reference, BASELINES


class ReportingTests(unittest.TestCase):
    def test_reference_path_does_not_depend_on_snapshot_cwd(self):
        from unittest.mock import patch
        base=ROOT/'results/five_fixture/time'
        probe=ROOT/'results/probe_fixture/time'
        expected=ROOT/'results/probe_fixture/time/jobs/TG_22/output'
        def fake_read(path):
            if path==base/'protocol.json':return dict(probe_reference='results/probe_fixture/time')
            self.assertEqual(path,probe/'manifest.json')
            return dict(jobs=[dict(group='TG',output=str(expected))])
        with patch('run_thesis_experiments.read',side_effect=fake_read):
            self.assertEqual(compatibility_reference(dict(reused=[dict(algorithm='TG',source=str(base))]),'TG'),expected)

    def test_selected_wm_is_shared_normalized_in_both_ablation_groups(self):
        frozen=Path('frozen')
        for group in ('WM','PM_WM'):
            args=extra_args(group,frozen)
            self.assertEqual(args[args.index('--wm_actor_input')+1],'residual_scaled')
            self.assertEqual(args[args.index('--wm_actor_calibration')+1],str(frozen/'calibration.json'))
            self.assertNotIn('action_delta',args)
        self.assertIn('learned_set',extra_args('PM_WM',frozen))
        self.assertIn('zero_set',extra_args('Immediate',frozen))

    def test_report_statistics_csv_and_figures(self):
        temp=ROOT/'results/thesis_test_temp';temp.mkdir(parents=True,exist_ok=True)
        models=[dict(algorithm='TG',seed=seed,source='synthetic_test_fixture',
                     points=[dict(episode=0,delivery=0.),dict(episode=300,delivery=value)])
                for seed,value in [(11,.2),(22,.4),(33,.6)]]
        models.append(dict(algorithm='Greedy',seed=11,source='synthetic_test_fixture',points=[dict(episode=300,delivery=1.)]))
        with tempfile.TemporaryDirectory(dir=str(temp)) as folder:
            root=Path(folder)
            summarize(root,dict(reused=models,jobs=[],later_stages=[]))
            report=json.loads((root/'report.json').read_text())
            self.assertAlmostEqual(report['aggregate'][0]['mean'],.4)
            self.assertAlmostEqual(report['aggregate'][0]['sample_sd'],.2)
            self.assertFalse(report['all_thesis_experiments_completed'])
            self.assertNotIn('Greedy',BASELINES)
            self.assertEqual(report['excluded_algorithms'],['Greedy'])
            self.assertNotIn('Greedy',(root/'per_seed.csv').read_text())
            self.assertNotIn('Greedy',(root/'learning_curves.csv').read_text())
            self.assertNotIn('Greedy',[r['algorithm'] for r in report['aggregate']])
            self.assertEqual(len((root/'per_seed.csv').read_text().splitlines()),4)
            for name in ('comparison','convergence'):
                for ext in ('png','svg','pdf'):
                    self.assertGreater((root/(name+'.'+ext)).stat().st_size,1000)


if __name__=='__main__':unittest.main()
