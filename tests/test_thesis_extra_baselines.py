import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from buoy_network.thesis_baselines import SumMixer,CounterfactualCritic,counterfactual_advantage,td_lambda_returns


class ExtraBaselineTests(unittest.TestCase):
    def test_vdn_is_sum_and_all_agents_receive_gradient(self):
        q=torch.tensor([[1.,2.,3.]],requires_grad=True)
        result=SumMixer()(q,torch.zeros(1,7))
        self.assertEqual(float(result),6.)
        result.sum().backward()
        torch.testing.assert_close(q.grad,torch.ones_like(q))

    def test_coma_critic_excludes_own_action_but_uses_others(self):
        torch.manual_seed(31)
        critic=CounterfactualCritic(3,5,7)
        x,state=torch.randn(2,3,5),torch.randn(2,7)
        actions=torch.zeros(2,3,dtype=torch.long)
        original=critic(x,state,actions)
        for node in range(3):
            modified=actions.clone();modified[:,node]=1
            changed=critic(x,state,modified)
            torch.testing.assert_close(original[:,node],changed[:,node],rtol=0,atol=0)
            others=[i for i in range(3) if i!=node]
            self.assertGreater(float((changed[:,others]-original[:,others]).abs().max()),0)

    def test_coma_counterfactual_baseline_hand_computed(self):
        q=torch.tensor([[1.,3.,9.]])
        probabilities=torch.tensor([[.25,.75,0.]])
        a=counterfactual_advantage(q,probabilities,torch.tensor([1]))
        torch.testing.assert_close(a,torch.tensor([.5]))
        torch.testing.assert_close(((q-(q*probabilities).sum(-1,keepdim=True))*probabilities).sum(-1),torch.zeros(1))

    def test_td_lambda_terminal_and_bootstrap(self):
        rewards=torch.tensor([[1.],[2.]])
        q=torch.tensor([[10.],[5.]])
        torch.testing.assert_close(td_lambda_returns(rewards,q,.9,1.),torch.tensor([[2.8],[2.]]))
        torch.testing.assert_close(td_lambda_returns(rewards,q,.9,0.),torch.tensor([[5.5],[2.]]))


if __name__=='__main__':unittest.main()
