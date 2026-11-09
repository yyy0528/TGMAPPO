"""Increment-only WM inputs: no new parameters, no WAIT-ID leakage, native PPO."""
import sys,tempfile,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np,torch
from onpolicy.algorithms.utils.wm_features import incremental_features,training_scales,load_calibration
from test_legacy_wm_actor import checkpoint,run
from test_legacy_targeted_fixes import make_environment,make_learner

class InputFunctions(unittest.TestCase):
    def test_zero_increment_is_neutral_and_wait_id_is_not_a_probability(self):
        obs=torch.tensor([[.3,.7,24.]])
        x=torch.tensor([[[.3,0,0],[.7,0,0],[0,0,0]]])
        mask=torch.ones(1,3);scale=torch.tensor([.5,.02,.04])
        y=incremental_features(x,obs,mask,scale)
        self.assertTrue(torch.equal(y,torch.zeros_like(y)))
        self.assertTrue(torch.equal(x,torch.tensor([[[.3,0,0],[.7,0,0],[0,0,0]]])))
    def test_scale_mask_clamp_and_gradients(self):
        obs=torch.tensor([[.3,.7,8.]])
        x=torch.tensor([[[.4,.02,-.04],[.9,100,100],[0,.01,0]]],requires_grad=True)
        y=incremental_features(x,obs,torch.tensor([[1,0,1]]),torch.tensor([.5,.02,.04]))
        torch.testing.assert_allclose(y,torch.tensor([[[.2,1.,-1.],[0,0,0],[0,.5,0]]]))
        y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad[0,0,1]),50.)
        with self.assertRaises(ValueError):incremental_features(x,obs,torch.ones(1,3),torch.zeros(3))
    def test_centered_null_is_constant_across_legal_candidates(self):
        obs=torch.tensor([[.3,.7,24.]])
        x=torch.tensor([[[.3,0,0],[.7,0,0],[0,0,0]]])
        y=incremental_features(x,obs,torch.ones(1,3),torch.tensor([.1,.02,.03]),torch.tensor([.01,-.02,.02]))
        self.assertTrue(torch.equal(y[:,0],y[:,1]) and torch.equal(y[:,1],y[:,2]))
    def test_calibration_rejects_wrong_model_and_non_training_split(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'cal.json';v=dict(schema='wm_increment_train_calibration_v1',fit_split='train',samples=5,wm_sha256='model',mean=[0,0,0],scale=[1,1,1])
            p.write_text(json.dumps(v));self.assertEqual(load_calibration(p,'model')[0],v)
            with self.assertRaises(ValueError):load_calibration(p,'other')
            v['fit_split']='validation';p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):load_calibration(p,'model')

class InputNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def native(self,device):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'wm.pt';checkpoint(path)
            env=make_environment(root/'run',length=4);a=env.all_args
            a.tg_device=device;a.use_wm_actor=True;a.wm_actor_checkpoint=str(path);a.wm_actor_input='residual_scaled'
            model=make_learner(env);scale=model.policy.actor.wm_input_scale.clone()
            self.assertEqual(sum(p.numel() for p in model.policy.actor.parameters()),24165)
            run(env,model)
            self.assertTrue(torch.equal(scale,model.policy.actor.wm_input_scale))
            self.assertTrue(model.wm_update_records[-1]['wm_frozen'])
            self.assertGreater(model.training_records[-1]['wm_fusion_grad_norm'],0)
            env2=make_environment(root/'eval',length=4);b=env2.all_args
            b.tg_device=device;b.use_wm_actor=True;b.wm_actor_checkpoint=str(path);b.wm_actor_input='residual_scaled'
            restored=make_learner(env2);restored.load_policy(root/'run/moves_0/checkpoints/tg_policy_latest.pt')
            self.assertTrue(torch.equal(scale,restored.policy.actor.wm_input_scale))
            env3=make_environment(root/'wrong',length=4);c=env3.all_args;c.use_wm_actor=True;c.wm_actor_checkpoint=str(path)
            with self.assertRaises(ValueError):make_learner(env3).load_policy(root/'run/moves_0/checkpoints/tg_policy_latest.pt')
    def test_native_cpu_and_artifact_guards(self):self.native('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_native_cuda_and_artifact_guards(self):self.native('cuda')

if __name__=='__main__':unittest.main()
