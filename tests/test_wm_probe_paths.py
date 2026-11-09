"""A snapshotted controller must compare artifacts outside its source tree."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_tg_wm_input_probe import verify_raw_compatibility


class ProbePaths(unittest.TestCase):
    def test_snapshot_location_does_not_change_absolute_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, output = root / 'old_results', root / 'new_results'
            payload = dict(actor={'weight': torch.tensor([1.])},
                           critic={'weight': torch.tensor([2.])})
            for folder in (reference, output):
                (folder / 'checkpoints').mkdir(parents=True)
                (folder / 'logs').mkdir()
                torch.save(payload, folder / 'checkpoints/tg_policy_latest.pt')
                (folder / 'logs/TG_policy_report.json').write_text(
                    json.dumps({'training_updates': [{'update': 1}]}))
                (folder / 'logs/TG_physical_metrics.jsonl').write_bytes(b'{"delivery": 1}\n')
            with patch('run_tg_wm_input_probe.ROOT', root / 'source_snapshot'):
                verify_raw_compatibility(output, reference)
                (output / 'logs/TG_physical_metrics.jsonl').write_bytes(b'{"delivery": 0}\n')
                with self.assertRaises(AssertionError):
                    verify_raw_compatibility(output, reference)

    def test_relative_artifact_paths_are_rejected(self):
        with self.assertRaises(AssertionError):
            verify_raw_compatibility('relative_results', 'old_results')


if __name__ == '__main__':
    unittest.main()
