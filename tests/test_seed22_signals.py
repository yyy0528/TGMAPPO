import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from tg_seed22_signals import signal_arrays,aggregate,packet_waiting


class SignalFunctions(unittest.TestCase):
    def test_hand_computed_two_transition_GAE_and_active_normalization(self):
        base=np.array([1.,-1.]);phi=np.array([.2,.1,0.]);v=np.zeros((2,2))
        masks=np.array([[[1,1,1],[0,0,1]],[[1,1,1],[0,0,1]]])
        shaped=base+.1*(.5*phi[1:]-phi[:-1])
        result=signal_arrays(base,shaped,v,phi,masks,.5,1.,.1)
        np.testing.assert_allclose(result['raw'][:,0],[.48,-1.01])
        np.testing.assert_allclose(result['normalized'][:,0],[.745/(.745+1e-5),-.745/(.745+1e-5)])
        np.testing.assert_array_equal(result['normalized'][:,1],0)
        self.assertLess(result['value_coordinate_error'],1e-14)

    def test_zero_active_and_invalid_reward_time_alignment(self):
        v=np.zeros((2,2));m=np.zeros((2,2,3));m[:,:,2]=1
        result=signal_arrays([0,0],[0,0],v,[0,0,0],m,.99,.95,0)
        np.testing.assert_array_equal(result['normalized'],0)
        with self.assertRaises(ValueError):signal_arrays([0,0],[1,0],v,[0,0,0],m,.99,.95,.1)


class SignalModules(unittest.TestCase):
    def test_successful_other_delivery_does_not_label_WAIT_as_own_delivery(self):
        row=dict(action=25,wait=25,sink=24,own_delivery=False,revisit=False,other_deliveries=1,
                 sink_legal=True,advantage=1.,raw_advantage=.1,base_reward=.04,progress_reward=-.01,wait_score=.8,sink_score=-.2)
        result=aggregate([row])
        self.assertEqual(result['WAIT']['count'],1)
        self.assertEqual(result['Sink_ACK_success']['count'],0)
        self.assertEqual(result['Sink_legal_other']['count'],1)

    def test_packet_service_wait_and_FIFO_blocking_are_distinct(self):
        frames=[{'episode':0,'actions':[1,2],'mask':[[1,1,1],[0,0,1]]},
                {'episode':0,'actions':[2,2],'mask':[[1,1,1],[0,0,1]]}]
        states=[{'service_ids':[0,-1],'packet_nodes':[[0,0],[1,0]]},
                {'service_ids':[1,-1],'packet_nodes':[[1,0]]}]
        routes=[dict(episode=0,packet_id=0,route=[0,1],delivered=True,delivery_step=1),
                dict(episode=0,packet_id=1,route=[0],delivered=False,delivery_step=None)]
        rows=packet_waiting(frames,states,routes,2)
        self.assertEqual(rows[0]['forward_attempts'],1)
        self.assertEqual(rows[1]['queued_behind'],1)
        self.assertEqual(rows[1]['service_wait'],1)


if __name__=='__main__':unittest.main(verbosity=2)
