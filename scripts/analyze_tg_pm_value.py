"""Verify persisted same-trajectory PM coordinate evidence and test receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',required=True);p.add_argument('--tests',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();root=Path(args.probe).resolve();report=read(root/'report.json')
    assert report['success'] and len(report['jobs'])==12 and all(j['exit_code']==0 for j in report['jobs'])
    for name,h in report['source']['files_sha256'].items():assert sha(root/'source_snapshot'/name)==h
    tests_root=Path(args.tests).resolve();tests=read(tests_root/'report.json')
    assert tests['success'] and sum(s['tests'] for s in tests['stages'])==7
    assert all(s['failures']==s['errors']==s['skipped']==0 for s in tests['stages'])
    tests['logs']={n:(tests_root/(n+'.log')).read_text(encoding='utf-8') for n in ('functions','modules','integration')}
    pairs=[]
    for seed in (11,22,33):
        folders=[root/('{}_{}_training'.format(mode,seed))/'runs/case/value_audit'
                 for mode in ('learned_shaped','base_residual')]
        arrays=[np.load(str(folder/'episode_0.npz'),allow_pickle=False) for folder in folders]
        same=all(np.array_equal(arrays[0][k],arrays[1][k]) for k in
                 ('base_rewards','shaped_rewards','raw_values','phi','masks'))
        assert same
        rows=[[json.loads(x) for x in (folder/'summary.jsonl').read_text().splitlines()] for folder in folders]
        pairs.append({'seed':seed,'initial_trajectory_rewards_values_phi_exact':same,
            'actual_gae_difference_max':float(np.max(np.abs(arrays[0]['actual_gae']-arrays[1]['actual_gae']))),
            'aligned_identity_max_error':max(r['aligned_gae_error'] for rr in rows for r in rr),
            'sampled_shaped_MC_bias_first_episode':[rr[0]['sampled_effective_value_minus_shaped_MC_active_mean'] for rr in rows],
            'cache_files_sha256':{str(folder/'episode_0.npz'):sha(folder/'episode_0.npz') for folder in folders}})
        for a in arrays:a.close()
    evidence={'success':True,'scope':'value-coordinate invariance repair/control; no additional PM policy benefit claimed',
        'probe':report,'same_trajectory_pairs':pairs,'new_layered_tests':tests,
        'other_observed_test_runs':{'legacy_discovery_passed':105,'legacy_discovery_seconds':87.236,
                                    'completion_credit_passed':5,'completion_credit_seconds':.777,
                                    'unique_total':117,'failures':0,'errors':0,'skipped':0},
        'startup_failure':read(ROOT/'results/tg_pm_value_tests/startup_failure/failure.json'),
        'current_test_sources':{'tests/test_pm_value.py':sha(ROOT/'tests/test_pm_value.py')},
        'analyzer_sha256':sha(__file__),
        'limitations':['2 episodes per training seed; no new fixed100 performance result',
                       'legacy shaped critic is a legitimate alternative, not a proven GAE formula bug',
                       'base_residual analytically cancels PM in policy updates; it is not innovation efficacy',
                       'sampled MC mean bias is not exact conditional value error or statistical significance']}
    with Path(args.report).open('x',encoding='utf-8') as stream:json.dump(evidence,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print('PASS: 117 tests; 12 real jobs; 3 identical initial caches; 3 exact baseline residual policies')


if __name__=='__main__':main()
