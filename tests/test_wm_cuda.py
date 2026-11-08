"""Frozen WM CUDA parity and original WM / PM+WM PPO integration."""
import sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np,torch
from buoy_network.wm_runtime import FrozenWMRuntime,tensor_digest
from test_legacy_wm_actor import checkpoint,inputs,run
from test_legacy_pm_reward import checkpoint as pm_checkpoint
from test_legacy_targeted_fixes import make_environment,make_learner

@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
class WMCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.old_tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=cls.old_tf32
    def test_cpu_gpu_features_and_frozen_load_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'wm.pt';checkpoint(path,3)
            cpu=FrozenWMRuntime(path,3)
            before=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
            gpu=FrozenWMRuntime(path,3,device='cuda')
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertTrue(torch.equal(cuda,torch.cuda.get_rng_state()))
            self.assertTrue(all(p.is_cuda for p in gpu.model.parameters()))
            for step in range(3):
                x=cpu.predict(*inputs(),step);y=gpu.predict(*inputs(),step)
                np.testing.assert_allclose(x,y,atol=2e-6,rtol=2e-5)
                actions=np.ones((1,3,1),dtype=np.int64)
                for runtime in (cpu,gpu):
                    runtime.record_actions(actions);runtime.record_attempt(0,1,True)
            self.assertEqual(tensor_digest(cpu.model.state_dict()),tensor_digest(gpu.model.state_dict()))
            gpu.assert_frozen()
    def native(self,combo):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);wm=root/'wm.pt';pm=root/'pm.pt';checkpoint(wm);pm_checkpoint(pm)
            env=make_environment(root/'run',length=4);a=env.all_args
            a.tg_device='cuda';a.use_wm_actor=True;a.wm_actor_checkpoint=str(wm)
            a.tg_reward_version='task_v2'
            if combo:
                a.use_pm_reward=True;a.pm_reward_checkpoint=str(pm);a.pm_reward_alpha=.1
                a.pm_potential_mode='task_return';a.pm_value_mode='base_residual';a.pm_action_credit='learned_set'
            model=make_learner(env);run(env,model)
            self.assertEqual(model.wm_runtime.metadata()['inference_device'],'cuda:0')
            self.assertEqual(model.training_records[-1]['actor_optimizer_steps'],2)
            self.assertEqual(model.training_records[-1]['critic_head_optimizer_steps'],2)
            self.assertTrue(model.wm_update_records[-1]['wm_frozen'])
            self.assertLessEqual(model.wm_update_records[-1]['ratio_max_error'],1e-4)
            if combo:self.assertEqual(model.pm_runtime.metadata()['inference_device'],'cuda:0')
    def test_native_wm_gpu(self):self.native(False)
    def test_native_pm_wm_gpu(self):self.native(True)

if __name__=='__main__':unittest.main()
