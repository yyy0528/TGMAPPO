"""Read-only train/validation calibration against actual finite task returns."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import torch
from buoy_network.progress_model import load_packets, pack
from buoy_network.progress_runtime import FrozenPMRuntime
from buoy_network.pm_return import expected_task_return
from buoy_network.task_reward import reward_contract
from types import SimpleNamespace


@torch.no_grad()
def evaluate_split(model, packets, contract, gamma):
    data = pack(packets)
    h = data['horizon']
    groups = {}
    for start in range(0, len(packets), 24):
        ps = packets[start:start+24]
        lengths = data['lengths'][start:start+24]
        logits = model(data['features'][start:start+24], lengths)
        for i, p in enumerate(ps):
            n = p['length']
            if not p['event'] and n != h:
                raise ValueError('Cannot identify full-window returns for earlier censoring')
            left = h-torch.arange(n)
            target = []
            for t in range(n):
                duration = n-t
                delay = -contract['delay_weight']/h*sum(gamma**k for k in range(duration))
                target.append(delay+gamma**(duration-1)*(contract['delivery_weight'] if p['event'] else -contract['unfinished_weight']))
            target = torch.tensor(target, dtype=torch.float64)
            prediction = expected_task_return(logits[i, :n], left, h, gamma, contract)
            prior = expected_task_return(model.prior_logits.expand(n, -1), left, h, gamma, contract)
            scores = {'probability_as_value': prediction['probability'],
                      'task_return': prediction['task_return'], 'hazard_prior_task_return': prior['task_return'],
                      'zero': torch.zeros_like(target)}
            row = {'outcome': p['event'], 'prefixes': n, 'actual_mean': float(target.mean())}
            for name, value in scores.items():
                row[name+'_mse'] = float(((value-target)**2).mean())
                row[name+'_bias'] = float((value-target).mean())
            for group in ('all', p['behavior']):
                groups.setdefault(group, []).append(row)
    result = {}
    for group, rows in groups.items():
        keys = [k for k in rows[0] if k not in ('outcome', 'prefixes')]
        result[group] = dict({k: sum(r[k] for r in rows)/len(rows) for k in keys},
                             packets=len(rows), delivered=sum(r['outcome'] for r in rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='results/tg_wm_multiscene/20261102_235951_684782')
    parser.add_argument('--checkpoint', default='results/tg_pm_final_pretrain/20261103_142745_276042/pm_best.pt')
    parser.add_argument('--output-root', default='results/tg_pm_target_audit')
    args = parser.parse_args()
    torch.set_num_threads(1)
    runtime = FrozenPMRuntime(ROOT/args.checkpoint, 25, 80, .99, 1)
    contract = reward_contract(SimpleNamespace(tg_reward_version='task_v2'))
    report = {'checkpoint_sha256': runtime.file_sha256, 'gamma': .99, 'contract': contract,
              'weighting': 'equal packet, then equal observed prefixes per packet',
              'test_evaluated': False, 'fit_performed': False, 'splits': {}}
    for split in ('train', 'validation'):
        packets = load_packets(ROOT/args.dataset, split)
        report['splits'][split] = evaluate_split(runtime.model, packets, contract, .99)
        report['splits'][split]['scenario_seeds'] = sorted({p['scenario_seed'] for p in packets})
    runtime.assert_frozen()
    root = ROOT/args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    (root/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(str(root/'report.json'))
    for split in ('train', 'validation'):
        print(split+': '+json.dumps(report['splits'][split]['all']))


if __name__ == '__main__': main()
