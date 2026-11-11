"""The conservative scale floor must not change centers, weights or fitting split."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from derive_tg_wm_scale_floor import derive


class ConservativeScales(unittest.TestCase):
    def fixture(self):
        c = dict(schema='wm_increment_train_calibration_v1',fit_split='train',future_labels_used=False,
                 wm_sha256='model',samples=50,mean=[.03,-.01,.02],scale=[.08,.009,.008])
        model = SimpleNamespace(ack_prior=torch.tensor(.5),queue_scale=torch.tensor([.03,.04]))
        return c,model

    def test_floor_preserves_center_input_and_model(self):
        c,model=self.fixture();before=copy.deepcopy(c);q=model.queue_scale.clone()
        result=derive(c,model,'model','source')
        self.assertEqual(c,before);self.assertEqual(result['mean'],c['mean'])
        self.assertTrue(torch.equal(q,model.queue_scale))
        torch.testing.assert_allclose(torch.tensor(result['scale']),torch.tensor([.5,.03,.04]))
        self.assertEqual(result['source_calibration_sha256'],'source')
        self.assertFalse(result['future_labels_used'])

    def test_large_existing_scale_is_not_reduced(self):
        c,model=self.fixture();c['scale']=[.8,.2,.4]
        torch.testing.assert_allclose(torch.tensor(derive(c,model,'model','source')['scale']),torch.tensor(c['scale']))

    def test_invalid_split_labels_hash_and_scales_are_rejected(self):
        c,model=self.fixture()
        for key,value in [('fit_split','validation'),('future_labels_used',True),('wm_sha256','other'),('scale',[0,.1,.2]),('scale',[float('nan'),.1,.2])]:
            broken=dict(c);broken[key]=value
            with self.assertRaises(ValueError):derive(broken,model,'model','source')


if __name__=='__main__':unittest.main()
