"""Read-only PM decline evidence from the completed fixed100 experiment."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    suite=ROOT/'results/tg_ablation_fixed100/20261104_142119_244782'
    manifest=read(suite/'manifest.json')
    for name,h in manifest['source']['files_sha256'].items():
        assert sha(suite/'source_snapshot'/name)==h,'Original archived source changed'
    summary=read(suite/'summary.json')
    evidence={'suite':str(suite),'manifest_sha256':sha(suite/'manifest.json'),
              'policies':[r for r in summary['by_training_seed'] if r['group'] in ('A','C')], 'reward_diagnostics':[],
              'frozen_policy_diagnostics':[], 'causal_scope':'mechanism evidence, not proof of a unique cause or statistical significance'}
    for job in manifest['jobs']:
        if job['group']!='C' or job['kind']!='training':continue
        output=Path(job['output'])
        for name,h in job['artifacts_sha256'].items():assert sha(output/name)==h,'Old artifact changed'
        report=read(output/'logs/TG_policy_report.json')
        assert sha(report['checkpoint'])==report['checkpoint_sha256']
        with (output/'logs/TG_reward_components.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        delivery=[r for r in rows if float(r['reward_delivery'])>0 and r['terminated']=='False']
        evidence['reward_diagnostics'].append({'seed':job['train_seed'],'delivery_slots':len(delivery),
            'negative_progress_delivery_slots':sum(float(r['reward_progress'])<0 for r in delivery),
            'positive_base_becomes_nonpositive':sum(float(r['reward_base'])>0 and float(r['reward_train'])<=0 for r in delivery),
            'max_boundary_error':max(r['boundary_error'] for r in report['pm']['episodes']),
            'reward_sha256':sha(output/'logs/TG_reward_components.csv')})
    for seed in (22,33):
        path=ROOT/('results/tg_pm_diagnosis_20261104/C_'+str(seed))/'report.json'
        report=read(path)
        assert report['weights_unchanged'] and report['optimizer_updates']==0 and report['pilot_metrics_exact_match']
        evidence['frozen_policy_diagnostics'].append({'seed':seed,'report_sha256':sha(path),'analysis':report['analysis']})
    output=ROOT/'docs/evidence/TG_PM_DECLINE_20261104.json'
    output.write_text(json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Saved '+str(output))


if __name__=='__main__':main()
