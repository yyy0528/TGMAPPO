"""Candidate identity, neutral forecasts, true gradients and native WM PPO."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
from onpolicy.algorithms.utils.wm_fusion import fusion_mode, action_conditioned_residual
from test_legacy_targeted_fixes import make_environment, make_learner
from test_legacy_wm_actor import checkpoint, run


class FusionFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_mode_contract(self):
        self.assertEqual(fusion_mode(SimpleNamespace()),'shared')
        for args in (SimpleNamespace(wm_actor_fusion='bad'),
                     SimpleNamespace(wm_actor_fusion='action_delta',use_wm_actor=False),
                     SimpleNamespace(wm_actor_fusion='action_delta',use_wm_actor=True,wm_actor_input='raw')):
            with self.assertRaises(ValueError):fusion_mode(args)

    def test_common_forecasts_give_exact_zero_and_ignore_illegal_values(self):
        net=torch.nn.Sequential(torch.nn.Linear(7,5),torch.nn.Tanh(),torch.nn.Linear(5,1))
        h=torch.randn(2,4);w=torch.randn(3,4);z=torch.randn(2,1,3).expand(-1,3,-1).clone()
        mask=torch.tensor([[1,1,0],[0,1,1]])
        z[0,2]=100;z[1,0]=-100
        result=action_conditioned_residual(net,h,z,mask,w)
        self.assertTrue(torch.equal(result,torch.zeros_like(result)))
        with self.assertRaises(ValueError):action_conditioned_residual(net,h,z,mask*0,w)

    def test_same_forecast_can_have_distinct_candidate_utility(self):
        net=torch.nn.Sequential(torch.nn.Linear(5,1),torch.nn.Tanh(),torch.nn.Linear(1,1))
        with torch.no_grad():
            net[0].weight.copy_(torch.tensor([[1.,0.,1.,0.,0.]]));net[0].bias.zero_()
            net[-1].weight.fill_(1.);net[-1].bias.zero_()
        h=torch.ones(1,2);w=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])
        z=torch.tensor([[[1.,0.,0.],[1.,0.,0.],[-2.,0.,0.]]])
        out=action_conditioned_residual(net,h,z,torch.ones(1,3),w)
        self.assertGreater(float((out[0,0]-out[0,1]).abs()),.001)

    def test_gradient_matches_finite_differences_including_action_weights(self):
        net=torch.nn.Sequential(torch.nn.Linear(7,4),torch.nn.Tanh(),torch.nn.Linear(4,1)).double()
        h=torch.randn(2,4,dtype=torch.double,requires_grad=True)
        z=torch.randn(2,3,3,dtype=torch.double,requires_grad=True)
        w=torch.randn(3,4,dtype=torch.double,requires_grad=True)
        mask=torch.tensor([[1,1,0],[0,1,1]])
        self.assertTrue(torch.autograd.gradcheck(lambda a,b,c:action_conditioned_residual(net,a,b,mask,c),(h,z,w),atol=1e-4))


class FusionNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def native(self,device):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint(root/'wm.pt')
            env=make_environment(root/'run',length=4);a=env.all_args
            a.tg_device=device;a.use_wm_actor=True;a.wm_actor_checkpoint=str(root/'wm.pt')
            a.wm_actor_input='residual_scaled';a.wm_actor_fusion='action_delta'
            model=make_learner(env)
            self.assertEqual(sum(p.numel() for p in model.policy.actor.parameters()),24165)
            run(env,model)
            self.assertTrue(model.wm_update_records[-1]['wm_frozen'])
            self.assertLessEqual(model.wm_update_records[-1]['ratio_max_error'],1e-4)
            self.assertGreater(model.training_records[-1]['wm_fusion_grad_norm'],0)
            restored_env=make_environment(root/'eval',length=4);b=restored_env.all_args
            b.tg_device=device;b.use_wm_actor=True;b.wm_actor_checkpoint=str(root/'wm.pt')
            b.wm_actor_input='residual_scaled';b.wm_actor_fusion='action_delta'
            restored=make_learner(restored_env);cp=root/'run/moves_0/checkpoints/tg_policy_latest.pt'
            restored.load_policy(cp)
            restored.all_args.wm_actor_fusion='shared'
            with self.assertRaises(ValueError):restored.load_policy(cp)

    def test_cpu_ppo_save_load(self):self.native('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_ppo_save_load(self):self.native('cuda')


if __name__=='__main__':unittest.main()
