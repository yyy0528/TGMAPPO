"""Function -> module tests before real original-environment smoke runs."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from buoy_network.thesis_baselines import Mixer, encode, gae, masked_distribution, mlp, BaselineLearner


class FunctionTests(unittest.TestCase):
    def test_terminal_gae_hand_calculation(self):
        adv, returns = gae(torch.tensor([[1.],[2.]]),torch.tensor([[.5],[.25]]),gamma=.9,lam=1.)
        torch.testing.assert_close(returns,torch.tensor([[2.8],[2.]]))
        torch.testing.assert_close(adv,returns-torch.tensor([[.5],[.25]]))

    def test_illegal_actions_have_zero_probability(self):
        d = masked_distribution(torch.tensor([[100.,0.,-1.]]),torch.tensor([[False,True,True]]))
        self.assertEqual(d.probs[0,0],0)
        self.assertTrue((d.sample((100,))!=0).all())

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            masked_distribution(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool))

    def test_observation_shape_and_no_mutation(self):
        obs = np.ones((1,3,4),np.float32)
        nodes = np.ones((1,3,3,4),np.float32)
        adj = np.ones((1,3,3,3),np.float32)
        x,s = encode(obs,nodes,adj)
        self.assertEqual(x.shape,(3,22))
        self.assertEqual(s.shape,(33,))
        self.assertTrue(np.equal(obs,1).all() and np.equal(nodes,1).all())


class ModuleTests(unittest.TestCase):
    def test_native_checkpoint_logs_created_before_saver(self):
        import tempfile
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from thesis_experiment_worker import save_point
        root=Path(__file__).resolve().parents[1]/'results/thesis_test_temp'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(root)) as temp:
            point=Path(temp)/'convergence/e0000'
            def native_style_save(model,directory):
                with (Path(directory)/'logs/report.json').open('x') as stream:
                    stream.write('{}')
            save_point(None,point,native_style_save)
            self.assertTrue((point/'logs/report.json').is_file())

    def test_qmix_joint_monotonicity_and_gradients(self):
        torch.manual_seed(7)
        mixer = Mixer(25,20)
        qs = torch.randn(8,25,requires_grad=True)
        state = torch.randn(8,20)
        total = mixer(qs,state)
        gradient = torch.autograd.grad(total.sum(),qs,retain_graph=True)[0]
        self.assertTrue((gradient>=0).all())
        self.assertTrue((gradient>0).any(dim=0).all())
        total.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in mixer.parameters()))
        self.assertTrue(torch.all(mixer(qs.detach()+.1,state)>=total.detach()))

    def test_optimizer_actually_changes_policy(self):
        policy = mlp(10,4)
        old = [p.clone() for p in policy.parameters()]
        optimizer = torch.optim.Adam(policy.parameters(),lr=.001)
        loss = policy(torch.ones(5,10)).square().mean()
        BaselineLearner.optimize(loss,optimizer,policy.parameters())
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old,policy.parameters())))

    def test_nonfinite_loss_is_rejected(self):
        policy = mlp(10,4)
        optimizer = torch.optim.Adam(policy.parameters())
        with self.assertRaises(ValueError):
            BaselineLearner.optimize(torch.tensor(float('nan')),optimizer,policy.parameters())


if __name__ == '__main__':
    unittest.main()
