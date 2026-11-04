"""Read live original-TG task progress without changing runs or requiring Torch."""
import argparse
import json
from pathlib import Path


def status(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    new = [j for j in manifest['jobs'] if j['group'] != 'A']
    result = {'state':manifest['state'], 'new_completed_jobs':sum(j['state']=='completed' for j in new),
              'new_total_jobs':len(new), 'failed_jobs':[j['key'] for j in new if j['state']=='failed'],
              'reused_A_jobs':sum(j['group']=='A' and j['state']=='completed' for j in manifest['jobs']),
              'active':[]}
    for job in new:
        if job['state'] != 'running':
            continue
        path = Path(job['output'])/'logs/TG_physical_metrics.jsonl'
        rows = []
        for line in path.read_bytes().splitlines() if path.exists() else []:
            try:
                rows.append(json.loads(line.decode('utf-8')))
            except (ValueError,UnicodeDecodeError):
                # A live writer may not yet have finished its final JSON line.
                pass
        window = rows[-20:]
        result['active'].append({'key':job['key'],'kind':job['kind'],
            'complete_episodes':len(rows),
            'planned_episodes':manifest['config']['train_episodes' if job['kind']=='training' else 'eval_episodes'],
            'recent_physical_delivery_mean':sum(r['delivery_rate'] for r in window)/len(window) if window else None,
            'recent_window_size':len(window)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite')
    args = parser.parse_args()
    print(json.dumps(status(args.suite),indent=2))


if __name__ == '__main__':
    main()
