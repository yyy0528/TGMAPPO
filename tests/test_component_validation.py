import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from buoy_network.component_validation import calibration,observed_task_return,packet_weighted_mean,survival_diagnostics
from validate_frozen_components import FrozenForwardView


class MetricFunctions(unittest.TestCase):
    def test_calibration_weighting_and_invalid_probability(self):
        r=calibration([.1,.4],[0,1],[9,1]);self.assertAlmostEqual(r['ece'],.15)
        self.assertEqual(r['weight_total'],10)
        with self.assertRaises(ValueError):calibration([1.2],[1])
        self.assertEqual(calibration([1.+1e-12],[1])['ece'],0.)

    def test_finite_return_delivery_and_unfinished_hand_calculation(self):
        values,known=observed_task_return(torch.tensor([2,4,2]),torch.tensor([True,False,False]),4,1.)
        torch.testing.assert_close(values[0,:2],torch.tensor([.5,.75],dtype=torch.float64))
        torch.testing.assert_close(values[1],torch.tensor([-2.,-1.75,-1.5,-1.25],dtype=torch.float64))
        self.assertEqual(known[0].tolist(),[True,True,False,False]);self.assertFalse(known[2].any())

    def test_packet_mean_does_not_overweight_long_trace(self):
        values=torch.tensor([[1.,1.,1.],[3.,999.,999.]])
        mask=torch.tensor([[True,True,True],[True,False,False]])
        self.assertEqual(packet_weighted_mean(values,mask),2.)
        self.assertIsNone(packet_weighted_mean(values,torch.zeros_like(mask)))

    def test_early_censor_excluded_from_completion_and_task_return(self):
        data=dict(horizon=4,lengths=torch.tensor([2,4,2]),events=torch.tensor([True,False,False]))
        r=survival_diagnostics(torch.zeros(3,4,4),torch.zeros(4),data)
        self.assertEqual(r['identifiable_packets'],2);self.assertEqual(r['unknown_prefixes_excluded'],2)
        self.assertAlmostEqual(r['completion_calibration']['weight_total'],2.)


class FrozenForwardTests(unittest.TestCase):
    def test_view_matches_native_cpu_and_preserves_weights(self):
        from buoy_network.progress_model import ProgressModel,evaluate
        from buoy_network.wm_runtime import tensor_digest
        torch.manual_seed(71);m=ProgressModel(4,8);m.requires_grad_(False)
        data=dict(horizon=4,lengths=torch.tensor([2,4,3]),events=torch.tensor([True,False,False]),features=torch.randn(3,4,14))
        before=tensor_digest(m.state_dict());native=evaluate(m,data,batch_size=2);view=FrozenForwardView(m)
        actual=evaluate(view,data,batch_size=2)
        self.assertEqual(native,actual);self.assertEqual(torch.cat(view.cache).shape,(3,4,4));self.assertEqual(before,tensor_digest(m.state_dict()))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            gpu=copy.deepcopy(m).to('cuda');gpu_view=FrozenForwardView(gpu);result=evaluate(gpu_view,data,batch_size=2)
            for key in ['survival_nll','window_brier','prior_survival_nll','prior_window_brier']:self.assertAlmostEqual(result[key],native[key],places=5)
            self.assertEqual(before,tensor_digest(gpu.state_dict()))


class ReportingTests(unittest.TestCase):
    def test_reports_figures_and_three_seed_cell_statistics(self):
        from run_component_validation import analyze
        temp=ROOT/'results/thesis_test_temp';temp.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temp)) as folder:
            root=Path(folder);(root/'offline').mkdir();case=root/'case';case.mkdir()
            w=dict(ack_brier=.2,ack_history_link_brier=.25,queue_mse=.001,queue_persistence_mse=.002,motion_mse=.001,motion_clamped_walk_mse=.0009)
            p=dict(survival_nll=1.,prior_survival_nll=2.,window_brier=.1,prior_window_brier=.2,task_return_mse=.1,prior_task_return_mse=.3)
            (root/'offline/report.json').write_text(json.dumps(dict(splits={s:dict(wm=w,pm=p) for s in ['validation','test']})))
            rows=[]
            for condition in ['baseline','loss_high','walk_fast']:
                groups=[('PM','factual'),('PM_WM','factual')]+([('PM_WM','no_wm_residual')] if condition!='baseline' else [])
                for group,mode in groups:
                    for seed,value in [(11,.5),(22,.75),(33,1.)]:
                        rows.append(dict(algorithm=group,seed=seed,condition=condition,mode=mode,delivery=value,frozen_replay_exact=mode=='factual'))
            self.assertEqual(len(rows),24)
            ledger=[dict(final_positions=[[i//5,i%5] for i in range(25)],packets={'0':dict(route=[0,5],delay=None),'1':dict(route=[1,7,24],delay=3)})]
            (case/'packet_ledger.json').write_text(json.dumps(ledger))
            jobs=[dict(algorithm='PM_WM',seed=11,condition='loss_high',mode='factual',output=str(case))]
            analyze(root,dict(rows=rows,jobs=jobs))
            result=json.loads((root/'report.json').read_text());self.assertFalse(result['all_thesis_experiments_completed'])
            self.assertAlmostEqual(result['case_aggregate'][0]['mean'],.75);self.assertAlmostEqual(result['case_aggregate'][0]['sample_sd'],.25)
            self.assertTrue(result['factual_replays_exact'])
            for name in ['wm_validation','pm_validation','failure_comparison','packet_routes']:
                for ext in ['png','svg','pdf']:self.assertGreater((root/(name+'.'+ext)).stat().st_size,1000)


if __name__=='__main__':unittest.main()
