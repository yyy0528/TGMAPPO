"""Exercise actual q_update: terminal, legal mask, inactive next utilities."""
import sys,unittest,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from types import SimpleNamespace
from collections import deque
import numpy as np
import torch
from buoy_network.thesis_baselines import BaselineLearner,SumMixer


def model(done,interval=100):
    m=object.__new__(BaselineLearner)
    m.algorithm='VDN';m.device=torch.device('cpu');m.random=random.Random(7);m.all_args=SimpleNamespace(gamma=1.)
    m.actor=torch.nn.Linear(1,3);m.target=torch.nn.Linear(1,3)
    with torch.no_grad():
        m.actor.weight.zero_();m.actor.bias.copy_(torch.tensor([3.,4.,100.]))
        m.target.weight.zero_();m.target.bias.copy_(torch.tensor([10.,20.,1e6]))
    m.mixer=SumMixer();m.target_mixer=SumMixer();m.updates=0;m.q_target_interval=interval
    m.actor_optimizer=torch.optim.SGD(m.actor.parameters(),lr=0.)
    row=dict(x=np.zeros((2,1),np.float32),next_x=np.zeros((2,1),np.float32),action=[0,0],active=[1.,1.],
             next_active=[1.,0.],next_legal=[[True,True,False],[True,True,False]],reward=2.,done=done,state=[0.],next_state=[0.])
    m.replay=deque([row]*64);return m


class QTargetTests(unittest.TestCase):
    def test_terminal_target_ignores_all_bootstrap(self):
        m=model(True)
        self.assertEqual(m.q_update(),[16.,16.])

    def test_bootstrap_masks_illegal_and_inactive_values(self):
        m=model(False)
        # Current sum 6. Next sum 20 (illegal 1e6 excluded, inactive second node zero).
        self.assertEqual(m.q_update(),[256.,256.])

    def test_optional_target_sync_and_default(self):
        self.assertEqual(BaselineLearner.q_target_interval,100)
        m=model(True,2);m.q_update()
        for a,b in zip(m.actor.parameters(),m.target.parameters()):
            torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
