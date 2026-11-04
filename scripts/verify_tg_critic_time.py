"""Audit optional Critic clock controls, old-baseline parity and bounded diagnostics."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.buoy_network.ablation_suite import read,digest,inspect_job,summarize,require
from src.buoy_network.task_reward_audit import audit_reward_csv
from verify_tg_ablation import equivalent


def weights(root):
    return torch.load(str(Path(root)/'checkpoints/tg_policy_latest.pt'),map_location='cpu',weights_only=True)


def equal_models(a,b):
    return all(set(a[n])==set(b[n]) and all(torch.equal(a[n][k],b[n][k]) for k in a[n]) for n in ('actor','critic'))


def physical(root):
    return [json.loads(x) for x in (Path(root)/'logs/TG_physical_metrics.jsonl').read_text().splitlines()]


def audit_train(root,episodes,mode):
    root=Path(root)
    r=read(root/'logs/TG_policy_report.json')
    require(r['critic_time_mode']==mode and r['wm'] is None and r['pm'] is None,'Wrong clock/auxiliary mode')
    require(r['profile']['ppo_updates']==episodes and r['profile']['collect_calls']==episodes*80,'Wrong interactions')
    records=r['training_updates']
    require(len(records)==episodes,'Missing PPO records')
    for i,row in enumerate(records):
        require(row['actor_optimizer_steps']==row['critic_head_optimizer_steps']==(i+1)*2,'Wrong optimizer budget')
        require(row['critic_head_in_optimizer'] and all(math.isfinite(row[k]) for k in
            ('value_loss','policy_loss','dist_entropy','actor_grad_norm','critic_grad_norm','approx_kl','clip_fraction')),'Invalid learner values')
        for field,start in (('actor_lr',.0005),('critic_lr',.0003)):
            require(math.isclose(row[field],start-start*i/episodes,rel_tol=1e-12,abs_tol=1e-16),'Learning rate changed')
    audit_reward_csv(root/'logs',r['reward_contract'])
    state=weights(root)
    clock=state['critic'].get('time_conditioning.modulation')
    require((clock is None)==(mode=='disabled'),'Wrong Critic parameters')
    if clock is not None:
        require(clock.shape==(2,64) and float(clock.abs().sum())>0,'Clock module never learned')
    return r,{'parameters':r['resources'],'clock_parameter_l1':float(clock.abs().sum()) if clock is not None else 0.}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for mode in ('disabled','constant','remaining'):
        p.add_argument('--'+mode+'-suite',required=True)
    p.add_argument('--probe',required=True)
    p.add_argument('--report',required=True)
    a=p.parse_args()
    modes=('disabled','constant','remaining')
    roots={m:Path(getattr(a,m+'_suite')).resolve() for m in modes}
    manifests={m:read(r/'manifest.json') for m,r in roots.items()}
    source=manifests['disabled']['source']['files_sha256']
    old2=read(ROOT/'results/tg_ablation_lr_linear_pilot/20261103_211525_205795/manifest.json')
    summaries,baseline_parity={},[]
    for mode,root in roots.items():
        m=manifests[mode]
        require(m['state']=='completed' and len(m['jobs'])==6 and m['config']['groups']==['A'],'Incomplete clock smoke')
        require(m['source']['files_sha256']==source,'Different training sources')
        for name,sha in source.items():
            require(digest(root/'source_snapshot'/name)==sha==digest(ROOT/name),'Source changed: '+name)
        cfg=m['config']
        require(cfg['critic_time_mode']==mode and cfg['train_episodes']==2,'Wrong clock protocol')
        require({k:v for k,v in cfg.items() if k!='critic_time_mode'}==
                {k:v for k,v in manifests['disabled']['config'].items() if k!='critic_time_mode'},'More than clock changed')
        for job in m['jobs']:
            require(job['state']=='completed','Failed job')
            r,_=inspect_job(job,cfg)
            if job['kind']=='training':
                audit_train(job['output'],2,mode)
            else:
                require(r['actor_unchanged'] and r['critic_unchanged'] and r['optimizer_updates']==0,'Evaluation trained model')
        actual=summarize(m)
        require(equivalent(actual,read(root/'summary.json')),'Summary differs')
        checksum=digest(root/'summary.json')
        attempts={j['key']:len(j['attempts']) for j in m['jobs']}
        resume=subprocess.run([cfg['python'],str(ROOT/'scripts/run_tg_ablation.py'),'--mode','run','--resume',str(root)],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60)
        require(resume.returncode==0 and digest(root/'summary.json')==checksum and
                attempts=={j['key']:len(j['attempts']) for j in read(root/'manifest.json')['jobs']},'Resume changed jobs')
        summaries[mode]=actual
    for seed in (11,22,33):
        jobs={mode:next(j for j in m['jobs'] if j['kind']=='training' and j['train_seed']==seed) for mode,m in manifests.items()}
        reference=(Path(jobs['disabled']['output'])/'logs/TG_physical_metrics.jsonl').read_bytes()
        require(all((Path(j['output'])/'logs/TG_physical_metrics.jsonl').read_bytes()==reference for j in jobs.values()),'First two rollouts differ')
        for kind in ('training','evaluation'):
            current=next(j for j in manifests['disabled']['jobs'] if j['kind']==kind and j['train_seed']==seed)
            old=next(j for j in old2['jobs'] if j['kind']==kind and j['group']=='A' and j['train_seed']==seed)
            require((Path(current['output'])/'logs/TG_physical_metrics.jsonl').read_bytes()==
                    (Path(old['output'])/'logs/TG_physical_metrics.jsonl').read_bytes(),'Old physical baseline changed')
            if kind=='training':
                require(equal_models(weights(current['output']),weights(old['output'])),'Old default weights changed')
        baseline_parity.append({'seed':seed,'actor_critic_exact':True,'training_evaluation_physics_exact':True})
    probe=Path(a.probe).resolve()
    pm=read(probe/'manifest.json')
    require(pm['state']=='completed' and pm['train_seed']==33 and set(pm['modes'])==set(modes),'Incomplete bounded probe')
    medium={}
    for mode,entry in pm['modes'].items():
        jobs={j['kind']:j for j in entry['jobs']}
        require(entry['state']=='completed' and all(j['exit_code']==0 for j in jobs.values()),'Failed probe job')
        report,resources=audit_train(jobs['training']['output'],20,mode)
        ev=read(Path(jobs['evaluation']['output'])/'logs/TG_policy_report.json')
        require(ev['actor_unchanged'] and ev['critic_unchanged'] and ev['optimizer_updates']==0
                and ev['checkpoint_sha256']==report['checkpoint_sha256'],'Probe evaluation changed model')
        audit_reward_csv(Path(jobs['evaluation']['output'])/'logs',ev['reward_contract'])
        sample_path=probe/'calibration_verified'/mode/'report.json'
        sample=read(sample_path)
        require(sample['success'] and sample['decode']=='sample' and sample['mode']==mode
                and sample['weights_unchanged'] and sample['optimizer_updates']==0,'Invalid frozen calibration')
        errors=[]
        with (probe/'calibration_verified'/mode/'run/moves_5/logs/TG_reward_components.csv').open(newline='',encoding='utf-8') as f:
            sample_rewards={(int(r['episode']),int(r['step'])):float(r['reward_train']) for r in csv.DictReader(f)}
        for episode in (0,1):
            rows=[r for r in sample['observations'] if r['episode']==episode]
            require([r['step'] for r in rows]==list(range(80)),'Incomplete sampled task')
            total=0.
            for row in reversed(rows):
                require(math.isclose(row['reward'],sample_rewards[(episode,row['step'])],abs_tol=1e-12),'Recorded reward differs from runtime CSV')
                total=row['reward']+.99*total
                require(math.isclose(total,row['mc_return'],abs_tol=1e-12),'Wrong Monte Carlo return')
                require(all(math.isclose(t,1-row['step']/80,abs_tol=1e-7) for t in row['remaining']),'Wrong observed clock')
                errors.extend(v-total for v,active in zip(row['value'],row['active']) if active)
        require(math.isclose(statistics.mean(errors),sample['value_mc_bias'],abs_tol=1e-10) and
                math.isclose(math.sqrt(statistics.mean(x*x for x in errors)),sample['value_mc_rmse'],abs_tol=1e-10),'Calibration statistic differs')
        require(sample['script_sha256']==digest(ROOT/'scripts/diagnose_tg_critic_clock.py'),'Calibration script changed')
        medium[mode]={'greedy_physical':physical(jobs['evaluation']['output']),
            'sample_physical':sample['physical'],'value_mc_bias':sample['value_mc_bias'],'value_mc_rmse':sample['value_mc_rmse'],
            'calibration_sha256':digest(sample_path),'checkpoint_sha256':report['checkpoint_sha256'],
            'training_updates':report['training_updates'],**resources}
    require(medium['constant']['parameters']==medium['remaining']['parameters'],'Capacity control not matched')
    old20=ROOT/'results/tg_ppo_budget_epoch1/20261103_232339_569526'
    old20m=read(old20/'manifest.json')
    for kind in ('training','evaluation'):
        prev=next(j for j in old20m['jobs'] if j['kind']==kind and j['train_seed']==33)
        current=next(j for j in pm['modes']['disabled']['jobs'] if j['kind']==kind)
        require((Path(prev['output'])/'logs/TG_physical_metrics.jsonl').read_bytes()==
                (Path(current['output'])/'logs/TG_physical_metrics.jsonl').read_bytes(),'Old 20-episode physics changed')
        if kind=='training':
            require(equal_models(weights(prev['output']),weights(current['output'])),'Old 20-episode weights changed')
    tests=read(ROOT/'results/tg_critic_time_tests/20261104/report.json')
    reg=read(ROOT/'results/tg_critic_time_tests/20261104/regressions.json')
    count=sum(s['tests'] for s in tests['stages'])
    require(tests['success'] and reg['success'] and count==9,'New tests failed')
    regression_reports=[]
    for item in reg['reports']:
        r=read(item['path'])
        require(r['success'] and all(not any(s.get(k,0) for k in ('failures','errors','skipped')) for s in r['stages']),'Regression failed')
        count+=sum(s['tests'] for s in r['stages'])
        regression_reports.append({'path':item['path'],'sha256':digest(item['path']),'report':r})
    require(all(x['exit_code']==0 for x in reg['extra']),'Extra regression failed')
    count+=sum(s['tests'] for s in reg['extra'])
    require(count==139,'Wrong regression count')
    receipt={'success':True,'scope':'Critic-only clock plumbing and bounded baseline diagnostics; not formal innovation acceptance',
        'tests':count,'new_tests':tests,'regressions':reg,'regression_reports':regression_reports,
        'source_file_count':len(source),'smoke_suites':{m:str(r) for m,r in roots.items()},'smoke_summaries':summaries,
        'old_default_parity':baseline_parity,'old20_seed33_parity':True,'probe':str(probe),'medium_seed33':medium,
        'actual_budget':{'smoke_training_steps':1440,'smoke_greedy_steps':1440,'probe_training_steps':4800,
                         'probe_greedy_steps':480,'frozen_sample_steps':480,'initial_failed_sample_slots':1,
                         'train_eval_jobs':24,'sample_diagnostics':3,'resume_new_jobs':0},
        'initial_sampling_error':{'type':'TypeError','cause':'Diagnostic used tuple indexing on Runner reward list',
            'fix':'Convert reward list with np.asarray before indexing','failed_output':str(probe/'calibration/disabled')},
        'limitations':['medium probe uses only previously selected train seed33 and development scene7001',
            'sample MC outcomes are noisy and correlated; no convergence or significance claim',
            'same seeds do not force equal later channel/random frames once policies differ',
            'clock adds 128 Critic parameters, constant-input control matches capacity; Actor inputs/architecture unchanged',
            'WM/PM only compatibility tested; no new auxiliary performance comparison'],
        'validator_sha256':digest(Path(__file__))}
    with Path(a.report).open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False)
    print('PASS: 139 tests, 18 smoke + 6 bounded probe jobs, 3 frozen sampling diagnostics; old baseline exact')


if __name__=='__main__':
    main()
