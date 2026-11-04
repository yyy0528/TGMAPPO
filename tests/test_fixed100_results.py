"""Regression for frozen-report schema and incomplete fixed100 acceptance."""
import math
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_tg_fixed100_results import finite_tree,equivalent,audit

SUITE=ROOT/'results/tg_ablation_fixed100/20261104_142119_244782'


class ResultFunctions(unittest.TestCase):
    def test_nonfinite_nested_diagnostics_rejected(self):
        self.assertTrue(finite_tree({'metrics':[None,{'loss':.2,'note':'finite'}]}))
        self.assertFalse(finite_tree({'metrics':[{'loss':float('nan')}]}))
        self.assertFalse(finite_tree({'metrics':[{'gradient':float('inf')}]}))

    def test_derived_summary_tolerance_does_not_accept_missing_fields(self):
        self.assertTrue(equivalent({'rate':[.5]}, {'rate':[.5+1e-13]}))
        self.assertFalse(equivalent({'rate':[.5]}, {'rate':[.5001]}))
        self.assertFalse(equivalent({'rate':[.5]}, {}))


class ResultModules(unittest.TestCase):
    def test_real_partial_acceptance_handles_frozen_reports_without_update_field(self):
        result=audit(SUITE,partial=True)
        self.assertGreaterEqual(result['audited_completed_jobs'],6)
        self.assertTrue(any(j['kind']=='evaluation' for j in result['jobs']))
        if not result['complete']:
            self.assertTrue(result['pending_or_failed'])
            self.assertNotIn('summary',result)


class ResultIntegration(unittest.TestCase):
    def test_actual_incomplete_suite_cannot_be_accepted_as_complete(self):
        from src.buoy_network.ablation_suite import read
        if read(SUITE/'manifest.json')['state']=='completed':
            # When replayed after completion, use a scoped read-only metadata view.
            from unittest.mock import patch
            from scripts import verify_tg_fixed100_results as verifier
            original=verifier.driver.guard
            def view(root):
                m=original(root)
                m['state']='running'
                m['jobs'][-1]['state']='pending'
                return m
            with patch.object(verifier.driver,'guard',side_effect=view):
                with self.assertRaisesRegex(ValueError,'Cannot accept incomplete'):
                    audit(SUITE)
        else:
            with self.assertRaisesRegex(ValueError,'Cannot accept incomplete'):
                audit(SUITE)


if __name__=='__main__':
    unittest.main(verbosity=2)
