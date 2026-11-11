"""Training cache audits and frozen evaluation reports have distinct contracts."""
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts'),
               str(Path(__file__).resolve().parents[1]/'src')]
from run_tg_temperature_probe import audit_wm_updates


class WMAuditContract(unittest.TestCase):
    def test_frozen_evaluation_needs_no_training_records(self):
        audit_wm_updates({'optimizer_updates':0}, 2, True)

    def test_frozen_evaluation_rejects_updates_or_nonempty_cache(self):
        for report in ({'optimizer_updates':1}, {'optimizer_updates':0,'wm_ppo_updates':[{}]}):
            with self.assertRaises(AssertionError):audit_wm_updates(report,2,True)

    def test_training_requires_complete_frozen_probability_replay(self):
        valid = {'wm_frozen':True,'ratio_max_error':0.}
        audit_wm_updates({'wm_ppo_updates':[valid,valid]},2,False)
        with self.assertRaises(KeyError):audit_wm_updates({},2,False)
        with self.assertRaises(AssertionError):audit_wm_updates({'wm_ppo_updates':[valid]},2,False)
        for row in ({'wm_frozen':False,'ratio_max_error':0.}, {'wm_frozen':True,'ratio_max_error':.01}):
            with self.assertRaises(AssertionError):audit_wm_updates({'wm_ppo_updates':[row]},1,False)


if __name__ == '__main__':unittest.main()
