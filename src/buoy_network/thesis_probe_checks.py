"""Small gates for diagnostic pilots; no policy selection on final test scenes."""
from pathlib import Path
import json
import torch


def verify_native_prefix(checkpoint_dir,training_output,reference):
    point,reference=Path(checkpoint_dir),Path(reference)
    a=torch.load(point/'checkpoints/tg_policy_latest.pt',map_location='cpu',weights_only=True)
    b=torch.load(reference/'checkpoints/tg_policy_latest.pt',map_location='cpu',weights_only=True)
    if any(set(a[k])!=set(b[k]) or any(not torch.equal(a[k][name],b[k][name]) for name in a[k]) for k in ('actor','critic')):
        raise ValueError('Long-run prefix weights differ from the short probe')
    if (Path(training_output)/'logs/TG_physical_metrics.jsonl').read_bytes()!=(reference/'logs/TG_physical_metrics.jsonl').read_bytes():
        raise ValueError('Long-run prefix physical trajectory changed')
    current=json.loads((point/'logs/TG_policy_report.json').read_text())
    previous=json.loads((reference/'logs/TG_policy_report.json').read_text())
    if current['training_updates']!=previous['training_updates']:
        raise ValueError('Long-run prefix optimizer records changed')


def long_probe_allowed(values):
    return len(values)==3 and min(values)>=.95 and sum(values)/3>=.98
