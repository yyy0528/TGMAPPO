import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from buoy_network.thesis_probe_checks import long_probe_allowed,verify_native_prefix
from onpolicy.algorithms.utils.pm_action_loss import teacher_loss


class RepairChecks(unittest.TestCase):
    def test_long_gate_cannot_promote_bad_short_probe(self):
        self.assertTrue(long_probe_allowed([1.,1.,1.]))
        self.assertFalse(long_probe_allowed([1.,1.,.9]))
        self.assertFalse(long_probe_allowed([1.,1.]))
        self.assertFalse(long_probe_allowed([.96,.96,.96]))

    def test_stronger_coefficient_doubles_real_top_set_gradient(self):
        gradients=[]
        for coefficient in [.1,.2]:
            logits=torch.tensor([[1.,0.,2.]],requires_grad=True)
            target=torch.tensor([[.8,.15,.05]])
            loss=coefficient*teacher_loss(torch.log_softmax(logits,-1),target,objective='top_set')
            loss.backward();gradients.append(logits.grad)
        torch.testing.assert_close(gradients[1],gradients[0]*2)
        self.assertGreater(float(gradients[0][0,2]),0.)

    def test_prefix_checks_reject_weight_and_physical_drift(self):
        temp=ROOT/'results/thesis_test_temp';temp.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temp)) as folder:
            root=Path(folder);a,b=root/'new',root/'reference'
            payload=dict(actor=dict(w=torch.ones(2)),critic=dict(w=torch.ones(1)))
            for path in [a,b]:
                (path/'checkpoints').mkdir(parents=True);(path/'logs').mkdir()
                torch.save(payload,path/'checkpoints/tg_policy_latest.pt')
                (path/'logs/TG_physical_metrics.jsonl').write_text('physical\n')
                (path/'logs/TG_policy_report.json').write_text(json.dumps(dict(training_updates=[1,2])))
            verify_native_prefix(a,a,b)
            (a/'logs/TG_physical_metrics.jsonl').write_text('changed\n')
            with self.assertRaises(ValueError):verify_native_prefix(a,a,b)
            (a/'logs/TG_physical_metrics.jsonl').write_text('physical\n')
            payload['actor']['w'].zero_();torch.save(payload,a/'checkpoints/tg_policy_latest.pt')
            with self.assertRaises(ValueError):verify_native_prefix(a,a,b)


if __name__=='__main__':unittest.main()
