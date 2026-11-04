"""Audit completed original-TG fixed100 jobs and, when complete, compare A-F.

--partial only audits finished tasks. It never summarizes pending work as success.
No model fitting, checkpoint selection, decoding changes, or run resumption occurs.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from scripts import run_tg_fixed100_ablation as driver
from src.buoy_network.ablation_suite import read,digest,require,inspect_job,summarize


def finite_tree(value):
    if isinstance(value,dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value,list):
        return all(finite_tree(v) for v in value)
    return type(value) not in (float,int) or math.isfinite(value)


def equivalent(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if type(a) in (int,float) and type(b) in (int,float):
        return abs(a-b) <= 1e-12
    return a==b


def audit(root,partial=False):
    import torch
    root = Path(root).resolve()
    manifest = driver.guard(root)
    driver.check_executor(manifest)
    require(manifest['state'] in ('running','completed','failed'), 'Fixed100 has not started')
    rows = []
    for job in manifest['jobs']:
        if job['state'] != 'completed':
            continue
        report,physical = inspect_job(job,manifest['config'])
        if job['kind'] == 'training':
            require(finite_tree(report['training_updates']), 'Nonfinite update report: '+job['key'])
            driver.check_training(report,100)
            kwargs = {'map_location':'cpu'}
            if int(torch.__version__.split('.')[0]) >= 2:
                kwargs['weights_only'] = True
            payload = torch.load(report['checkpoint'],**kwargs)
            require(all(torch.isfinite(t).all().item() for n in ('actor','critic')
                        for t in payload[n].values()), 'Nonfinite model weights: '+job['key'])
        rows.append({'key':job['key'],'kind':job['kind'],'group':job['group'],
                     'train_seed':job['train_seed'],'reused':job['group']=='A',
                     'checkpoint_sha256':report['checkpoint_sha256'],
                     'physical_sha256':digest(Path(job['output'])/'logs/TG_physical_metrics.jsonl'),
                     'reward_sha256':digest(Path(job['output'])/'logs/TG_reward_components.csv'),
                     'environment_steps':report['profile']['collect_calls'],
                     'ppo_updates':report['profile']['ppo_updates'],
                     'generated':sum(x['generated_packets'] for x in physical),
                     'delivered':sum(x['delivered_packets'] for x in physical),
                     'loop_transfers':sum(x['loop_transfers'] for x in physical),
                     'resources':report['resources'],'profile':report['profile']})
    done = manifest['state']=='completed' and len(rows)==36
    require(partial or done, 'Cannot accept incomplete fixed100 suite')
    result = {'schema':'original_tg_fixed100_results_v1','suite':str(root),
              'complete':done,'audited_completed_jobs':len(rows),'jobs':rows,
              'pending_or_failed':[{'key':j['key'],'state':j['state']} for j in manifest['jobs']
                                   if j['state']!='completed'],
              'source':manifest['source'],'dependencies':read(root/'dependencies.json'),
              'execution_budget':manifest['workflow']['execution_budget'],
              'validator_sha256':digest(__file__)}
    if done:
        summary = summarize(manifest)
        require(equivalent(summary,read(root/'summary.json')), 'Summary differs from raw results')
        result['summary'] = summary
        paired = []
        by = {(r['group'],r['train_seed']):r for r in summary['by_training_seed']}
        for group in 'BCDEF':
            changes = [(by[(group,s)]['delivery_rate']-by[('A',s)]['delivery_rate'])*100
                       for s in manifest['config']['train_seeds']]
            paired.append({'group':group,'reference':'A','paired_delivery_percentage_points':changes,
                           'mean':statistics.mean(changes),'sample_sd':statistics.stdev(changes)})
        result['paired_development_differences'] = paired
        result['actual_new_steps'] = {
            kind:sum(r['environment_steps'] for r in rows if r['kind']==kind and not r['reused'])
            for kind in ('training','evaluation')}
    result['limitations'] = ['three development training seeds; scene 7001 is not blind',
        'no convergence/significance claim from this budget',
        'E matches extra trainable Actor capacity, not frozen-model total capacity',
        'one frozen auxiliary pretraining seed; PM reliability/true failure labels incomplete',
        'success-only hop/delay populations differ between policies; unresolved is not true drop',
        'CPU costs are observations, not controlled speed benchmarks']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--partial',action='store_true')
    p.add_argument('--report',required=True)
    args = p.parse_args()
    report = audit(args.suite,args.partial)
    with Path(args.report).open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print('PASS completed jobs: {}; full suite complete: {}'.format(report['audited_completed_jobs'],report['complete']))


if __name__ == '__main__':
    main()
