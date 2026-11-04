"""Background result processing waits, runs real children, and preserves failures."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import finalize_tg_fixed100 as f


def fixture(root,state):
    f.atomic(root/'manifest.json',{'state':state,'jobs':[], 'config':{'python':sys.executable}})


class FinalizerTests(unittest.TestCase):
    def test_running_suite_only_waits_and_never_starts_training_or_children(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp)
            fixture(root,'running')
            with patch.object(f.subprocess,'run') as launch:
                self.assertEqual(f.step(root)['state'],'waiting')
            launch.assert_not_called()
            self.assertFalse((root/'fixed100_acceptance.json').exists())

    def test_completed_child_pipeline_and_idempotent_receipt(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp)
            fixture(root,'completed')
            verify=[sys.executable,'-c','import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(\'{"complete":true}\')',
                    str(root/'fixed100_acceptance.json')]
            plot=[sys.executable,'-c','print("plot child succeeded")']
            with patch.object(f,'commands',return_value=[('verification',verify),('plots',plot)]):
                result=f.step(root)
                self.assertEqual(result['state'],'completed')
                self.assertEqual([s['exit_code'] for s in result['stages']],[0,0])
            with patch.object(f.subprocess,'run',side_effect=AssertionError('duplicate child')):
                self.assertEqual(f.step(root),result)

    def test_child_failure_stops_before_plots_and_preserves_log(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root=Path(temp)
            fixture(root,'completed')
            with patch.object(f,'commands',return_value=[('verification',[sys.executable,'-c','raise SystemExit(7)']),
                                                       ('plots',[sys.executable,'-c','print("unexpected")'])]):
                result=f.step(root)
            self.assertEqual(result['state'],'failed')
            self.assertEqual(len(result['stages']),1)
            self.assertEqual(result['stages'][0]['exit_code'],7)
            self.assertTrue(Path(result['stages'][0]['log']).is_file())


if __name__=='__main__':
    unittest.main(verbosity=2)
