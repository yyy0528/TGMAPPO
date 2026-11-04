import copy
import unittest
import numpy as np
from src.buoy_network.credit_diagnostics import advantages, summarize


class CreditFunctions(unittest.TestCase):
    def test_matches_original_active_standardization_without_mutation(self):
        ret = np.array([[[1.],[2.]],[[4.],[8.]]],dtype=np.float32)
        val = np.ones_like(ret)
        mask = np.array([[[1.],[0.]],[[1.],[1.]]])
        before = [a.copy() for a in (ret,val,mask)]
        raw,z = advantages(ret,val,mask)
        expected = (ret-val-(ret-val)[mask>0].mean())/((ret-val)[mask>0].std()+1e-5)
        expected[mask==0] = 0
        np.testing.assert_array_equal(z,expected)
        np.testing.assert_array_equal(raw,ret-val)
        for a,b in zip((ret,val,mask),before):
            np.testing.assert_array_equal(a,b)

    def test_empty_and_constant_advantages(self):
        x = np.ones((2,1),dtype=np.float32)
        for mask in (np.zeros_like(x),np.ones_like(x)):
            np.testing.assert_array_equal(advantages(x,x,mask)[1],np.zeros_like(x))

    def test_rejects_shapes_and_nonfinite_values(self):
        for ret,val,mask in (([1],[1,2],[1]),([float('nan')],[0],[1])):
            with self.assertRaises(ValueError):
                advantages(ret,val,mask)


class CreditModules(unittest.TestCase):
    def test_joint_delivery_does_not_label_another_action_as_own_delivery(self):
        row = dict(action=5,sink=24,wait=25,sink_legal=True,ack=True,revisit=True,other_deliveries=1,
            advantage_raw=.2,advantage_normalized=1.,value=-.1,**{'return':.1},reward=.04,
            p_selected_change=.01,p_sink_change=-.01)
        before = copy.deepcopy(row)
        result = summarize([row])
        self.assertEqual(result['sink_chosen']['count'],0)
        self.assertEqual(result['revisit_with_other_delivery']['count'],1)
        self.assertEqual(result['successful_revisit']['positive_advantage_count'],1)
        self.assertEqual(result['sink_legal_not_chosen']['count'],1)
        self.assertEqual(row,before)

    def test_wait_and_failed_ack_are_separate(self):
        row = dict(action=25,sink=24,wait=25,sink_legal=False,ack=None,revisit=False,other_deliveries=0,
            advantage_raw=-1.,advantage_normalized=-1.,value=0.,**{'return':-1.},reward=-.1,
            p_selected_change=-.01,p_sink_change=0.)
        result = summarize([row,dict(row,action=5,ack=False)])
        self.assertEqual(result['wait']['count'],1)
        self.assertEqual(result['failed_attempt']['count'],1)
        self.assertEqual(result['all_active']['count'],2)
        self.assertEqual(summarize([])['all_active'],{'count':0})


if __name__ == '__main__':
    unittest.main(verbosity=2)
