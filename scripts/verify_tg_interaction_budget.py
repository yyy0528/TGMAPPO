"""Independently audit fixed20/fixed100 original TG interaction-budget evidence."""
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.buoy_network.ablation_suite import read,digest,require,inspect_job,summarize
from src.buoy_network.task_reward_audit import audit_reward_csv
from run_tg_interaction_budget import audit_old
from verify_tg_ablation import equivalent


def payload(root):
    return torch.load(str(Path(root)/'checkpoints/tg_policy_latest.pt'),map_location='cpu',weights_only=True)


def equal_models(a,b):
    return all(set(a[n])==set(b[n]) and all(torch.equal(a[n][k],b[n][k]) for k in a[n]) for n in ('actor','critic'))


def lines(path):
    return Path(path).read_bytes().splitlines()


def physical(root):
    return [json.loads(x) for x in lines(Path(root)/'logs/TG_physical_metrics.jsonl')]


def csv_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def stat(values):
    values=[v for v in values if v is not None]
    return {'n_defined':len(values),'mean':statistics.mean(values) if values else None,
            'sample_sd':statistics.stdev(values) if len(values)>1 else None}


def first_threshold(rows,window,target):
    for end in range(window,len(rows)+1):
        if statistics.mean(r['delivery_rate'] for r in rows[end-window:end])>=target-1e-12:
            return end*80
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--report',required=True)
    a=p.parse_args()
    root=Path(a.suite).resolve()
    m=read(root/'manifest.json')
    c=m['config']
    require(m['state']=='completed' and len(m['jobs'])==6 and c['groups']==['A']
        and c['train_episodes']==100 and c['eval_episodes']==2 and c['eval_seeds']==[7001]
        and c['train_seeds']==[11,22,33] and c['learning_rate_schedule']=='fixed'
        and c['critic_time_mode']=='disabled' and c['ppo_epoch']==1 and c['num_mini_batch']==2,'Wrong completed budget protocol')
    for name,sha in m['source']['files_sha256'].items():
        require(digest(root/'source_snapshot'/name)==sha==digest(ROOT/name),'Experiment source changed: '+name)
    baseline=read(root/'baseline20.json')
    require(baseline==audit_old(ROOT/'results/tg_lr_budget_fixed'),'Preserved old baseline changed')
    for job in m['jobs']:
        require(job['state']=='completed','Incomplete/failed job')
        inspect_job(job,c)
    require(equivalent(summarize(m),read(root/'summary.json')),'Summary differs from raw jobs')
    old_attempts={j['key']:len(j['attempts']) for j in m['jobs']}
    summary_sha=digest(root/'summary.json')
    resumed=subprocess.run([c['python'],str(ROOT/'scripts/run_tg_interaction_budget.py'),'--resume',str(root),'--mode','run'],
        cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60)
    require(resumed.returncode==0 and old_attempts=={j['key']:len(j['attempts']) for j in read(root/'manifest.json')['jobs']}
        and summary_sha==digest(root/'summary.json'),'Resume reran or changed jobs')
    by_seed,prefixes=[],[]
    for seed in c['train_seeds']:
        old=next(r for r in baseline['policies'] if r['seed']==seed)
        tr=next(j for j in m['jobs'] if j['kind']=='training' and j['train_seed']==seed)
        ev=next(j for j in m['jobs'] if j['kind']=='evaluation' and j['train_seed']==seed)
        new_root=Path(tr['output'])
        report=read(new_root/'logs/TG_policy_report.json')
        require(report['wm'] is None and report['pm'] is None and report['critic_time_mode']=='disabled','Model information changed')
        require(report['resources']['actor_trainable']==21956 and report['resources']['critic_trainable']==18615,'Capacity changed')
        require(report['profile']['ppo_updates']==100 and report['profile']['collect_calls']==8000,'Wrong measured training budget')
        records=report['training_updates']
        require(len(records)==100 and report['learning_rate_schedule']['kind']=='fixed','Missing training records/fixed plan')
        for i,r in enumerate(records,1):
            require(r['actor_optimizer_steps']==r['critic_head_optimizer_steps']==i*2 and r['critic_head_in_optimizer'],'Optimizer budget incorrect')
            require(r['actor_lr']==.0005 and r['critic_lr']==.0003,'LR changed with total episodes')
            require(all(math.isfinite(r[k]) for k in ('value_loss','policy_loss','dist_entropy','actor_grad_norm','critic_grad_norm','ratio','approx_kl','clip_fraction')),'Nonfinite learning metric')
        prefix_path=new_root.parent/'prefix/prefix_at_20.json'
        prefix=read(prefix_path)
        require(prefix['reference_exact'] and prefix['torch_rng_unchanged'] and prefix['update']==20
            and prefix['environment_steps']==1600 and digest(prefix['checkpoint'])==prefix['checkpoint_sha256'],'Invalid prefix capture')
        saved=torch.load(prefix['checkpoint'],map_location='cpu',weights_only=True)
        require(equal_models(saved,payload(old['training'])),'First20 Actor/Critic differs from old fixed20')
        previous=read(Path(old['training'])/'logs/TG_policy_report.json')['training_updates']
        require(len(prefix['training_updates'])==20 and all(all(n[k]==v for k,v in o.items())
            for n,o in zip(prefix['training_updates'],previous)),'First20 losses/updates changed')
        train_physics=physical(new_root)
        require(len(train_physics)==100 and lines(new_root/'logs/TG_physical_metrics.jsonl')[:20]==
                lines(Path(old['training'])/'logs/TG_physical_metrics.jsonl'),'First20 physical episodes differ')
        require(csv_rows(new_root/'logs/TG_reward_components.csv')[:1600]==
                csv_rows(Path(old['training'])/'logs/TG_reward_components.csv'),'First20 rewards differ')
        audit_reward_csv(new_root/'logs',report['reward_contract'])
        frozen=read(Path(ev['output'])/'logs/TG_policy_report.json')
        require(frozen['checkpoint_sha256']==report['checkpoint_sha256'] and frozen['actor_unchanged']
            and frozen['critic_unchanged'] and frozen['optimizer_updates']==0,'Evaluation updated model')
        audit_reward_csv(Path(ev['output'])/'logs',frozen['reward_contract'])
        prefixes.append({'seed':seed,'actor_critic_exact':True,'physical_rewards_exact':True,'old_record_fields_exact':True,
            'new_measured_actor_steps':prefix['training_updates'][-1]['actor_optimizer_steps'],
            'prefix_sha256':prefix['checkpoint_sha256']})
        for episodes,train,eval_root in ((20,Path(old['training']),Path(old['evaluation'])),(100,new_root,Path(ev['output']))):
            diagnostic_path=root/'diagnostics'/('fixed{}_seed{}'.format(episodes,seed))/'report.json'
            diagnostic=read(diagnostic_path)
            expected=digest(train/'checkpoints/tg_policy_latest.pt')
            require(diagnostic['weights_unchanged'] and diagnostic['optimizer_updates']==0
                and diagnostic['pilot_metrics_exact_match'] and diagnostic['policy_sha256']==expected
                and diagnostic['reference_episodes']==episodes and diagnostic['training_seed']==seed
                and diagnostic['decode']=='greedy','Wrong frozen policy diagnosis')
            require(lines(Path(diagnostic['output'])/'moves_5/logs/TG_physical_metrics.jsonl')==
                lines(eval_root/'logs/TG_physical_metrics.jsonl'),'Instrumented replay changed original evaluation')
            analysis=diagnostic['analysis']
            ph=physical(eval_root)
            trains=physical(train)
            delivered=sum(r['delivered_packets'] for r in ph)
            generated=sum(r['generated_packets'] for r in ph)
            by_seed.append({'episodes':episodes,'seed':seed,'delivered':delivered,'generated':generated,
                'delivery_rate':delivered/generated,'loop_transfers':sum(r['loop_transfers'] for r in ph),
                'active_wait_fraction':analysis['active_wait_fraction'],
                'sink_legal':analysis['sink_legal_count'],'sink_chosen':analysis['sink_chosen_count'],
                'normalized_entropy':analysis['mean_normalized_entropy'],
                'success_hops_mean':statistics.mean(v for r in ph for v in r['success_hops']) if delivered else None,
                'success_delay_mean':statistics.mean(v for r in ph for v in r['success_delays']) if delivered else None,
                'last20_training_delivery':statistics.mean(r['delivery_rate'] for r in trains[-20:]),
                'first_target_window_steps':first_threshold(trains,c['target_window'],c['target_delivery_rate']),
                'training_curve':trains,'diagnostic':analysis,'diagnostic_sha256':digest(diagnostic_path),
                'train_output':str(train),'evaluation_output':str(eval_root)})
    tests=read(ROOT/'results/tg_interaction_budget_tests/20261104/report.json')
    require(tests['success'] and sum(s['tests'] for s in tests['stages'])==7
        and all(s['exit_code']==0 and not any(s[k] for k in ('failures','errors','skipped')) for s in tests['stages']),'New tool tests failed')
    aggregates={str(e):{field:stat([r[field] for r in by_seed if r['episodes']==e]) for field in
        ('delivery_rate','loop_transfers','active_wait_fraction','normalized_entropy','success_hops_mean','success_delay_mean','last20_training_delivery')}
        for e in (20,100)}
    receipt={'success':True,'scope':'original TG baseline fixed20/fixed100 development interaction-budget diagnostic',
        'suite':str(root),'new_tests':tests,'previous_139_tests_rerun':False,'prefixes':prefixes,
        'source_file_count':len(m['source']['files_sha256']),'source_git':m['source'],
        'summary_sha256':summary_sha,'old_baseline':baseline,'by_training_seed':by_seed,'aggregate':aggregates,
        'budget':{'new_training_steps':24000,'new_frozen_evaluation_steps':480,'old_training_reused_steps':4800,
                  'old_evaluation_reused_steps':480,'extra_frozen_diagnostic_steps':960,'training_evaluation_jobs':6,
                  'frozen_action_replays':6,'resume_new_jobs':0,'new_optimizer_steps_per_network_per_seed':200},
        'limitations':['three trained seeds and one previously used development scene',
            'more fresh rollouts also cause proportionally more optimizer steps; no equal-compute claim',
            'training episodes/agents are correlated; statistical unit is independent training seed',
            'first20 exact; later policies/noise trajectories naturally diverge from the frozen20 model',
            'no formal convergence/generalization or WM/PM benefit acceptance'],
        'validator_sha256':digest(Path(__file__))}
    with Path(a.report).open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print('PASS: 7 new tests, 3 exact fixed20 prefixes, 6 completed jobs, 6 exact frozen replays, fixed LR, resume0')


if __name__=='__main__':
    main()
