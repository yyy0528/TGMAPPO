"""Frozen pre-PPO calibration, then new-seed equal-budget A/PM/gated-PM pilot."""
import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from buoy_network.ablation_suite import read,atomic,digest,snapshot,require
from buoy_network.pm_reliability import RULE
from check_tg_pm_task_return import model_equal,physical_rows,summarize


def run_job(root,m,job,command,timeout=1800):
    job.update(command=command,state='running');m['jobs'].append(job);atomic(root/'manifest.json',m)
    print('START '+job['tag'],flush=True)
    with (root/(job['tag']+'.log')).open('xb') as stream:
        p=subprocess.run(command,cwd=str(root/'source_snapshot'),stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    job['exit_code']=p.returncode
    require(p.returncode==0,'Task failed: '+job['tag']+'; see '+str(root/(job['tag']+'.log')))


def inspect(job,m):
    output=Path(job['output']);r=read(output/'logs/TG_policy_report.json')
    c=read(output/'logs/TG_MAPPO_environment.json')['config'];evaluation=job['kind']=='evaluation'
    episodes=2 if evaluation else 20
    require(c['num_episodes']==episodes and c['num_env_steps']==episodes*80
            and c['effective_num_env_steps']==episodes*80 and r['profile']['collect_calls']==episodes*80,'Wrong actual budget')
    require(c['ppo_epoch']==1 and c['num_mini_batch']==2 and not c['use_linear_lr_decay']
            and not c['use_wm_actor'] and c['actor_extra_width']==0 and c['tg_critic_time_mode']=='disabled'
            and c['num_moving_buoys']==5 and c['random_channel'] and c['tg_reward_version']=='task_v2','Protocol drift')
    if job['group']=='A':require(r['pm'] is None,'Baseline has PM')
    else:
        pm=r['pm'];require(pm['frozen'] and pm['potential_mode']=='task_return' and pm['value_mode']=='learned_shaped'
            and pm['checkpoint_sha256']==m['config']['pm_sha256'] and pm['gamma']==.99,'Wrong PM')
        alpha=m['calibrations'][str(job['seed'])]['decision']['effective_alpha'] if job['group']=='reliable' else 1.
        require(pm['alpha']==alpha,'Wrong calibrated alpha')
        if job['group']=='reliable':
            require(pm['reliability']['report_sha256']==m['calibrations'][str(job['seed'])]['report_sha256'],'Wrong calibration version')
        else:require(pm.get('reliability') is None,'Always-PM has a gate')
        require(all(e['boundary_error']<1e-7 and e['terminal_phi']==0 for e in pm['episodes']),'PM boundary error')
    if evaluation:
        require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged'],'Evaluation updated weights')
        require(r['checkpoint_sha256']==job['policy_sha256'],'Evaluation loaded a different checkpoint')
    else:
        require(len(r['training_updates'])==20 and r['profile']['ppo_updates']==20,'Wrong PPO count')
        for i,row in enumerate(r['training_updates'],1):
            require(row['actor_optimizer_steps']==row['critic_head_optimizer_steps']==2*i,'Wrong Adam budget')
            require(all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))),'Nonfinite training')
        require(digest(r['checkpoint'])==r['checkpoint_sha256'],'Checkpoint changed')
    physical=physical_rows(output)
    require(len(physical)==episodes and all(x['generated_packets']==x['delivered_packets']+x['unresolved_packets'] for x in physical),'Packet conservation')
    from buoy_network.task_reward_audit import audit_reward_csv
    audit_reward_csv(output/'logs',r['reward_contract'])
    with (output/'logs/TG_reward_components.csv').open(encoding='utf-8',newline='') as stream:rewards=list(csv.DictReader(stream))
    job.update(delivered=sum(x['delivered_packets'] for x in physical),generated=sum(x['generated_packets'] for x in physical),
        loops=sum(x['loop_transfers'] for x in physical),checkpoint_sha256=r['checkpoint_sha256'],
        mean_discounted_base_return=sum(.99**int(x['step'])*float(x['reward_base']) for x in rewards)/episodes)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compat-root',required=True)
    p.add_argument('--output-root',default='results/tg_pm_reliability')
    a=p.parse_args();config=read(ROOT/'configs/tg_pm_reliability_pilot.json')
    require(config['gate_rule']==RULE,'Calibration rule changed')
    require(not set(config['train_seeds'])&set(config['calibration_scenes']) and
            not set(config['eval_scenes'])&(set(config['train_seeds'])|set(config['calibration_scenes'])),'Scene overlap')
    require(digest(ROOT/config['pm_checkpoint'])==config['pm_sha256'],'PM weights changed')
    compat=read(Path(a.compat_root)/'manifest.json')
    require(compat['status']=='completed' and compat['stage']=='preflight','Successful default compatibility required')
    for path,checksum in compat['source']['files_sha256'].items():
        require(digest(Path(a.compat_root)/'source_snapshot'/path)==checksum and digest(ROOT/path)==checksum,'Source changed since compatibility: '+path)
    old=read(ROOT/'results/tg_ablation_lr_fixed_pilot/20261103_211528_024882/manifest.json')
    template=next(j for j in old['jobs'] if j['group']=='A' and j['kind']=='training' and j['train_seed']==11)
    root=ROOT/a.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True,exist_ok=False)
    m={'root':str(root),'stage':'initial_reliability_pilot','status':'running','config':config,
       'source':snapshot(root),'compat_root':str(Path(a.compat_root).resolve()),'calibrations':{},'jobs':[]}
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True)
    try:
        # Every calibration is completed/frozen before ANY PPO training job begins.
        for seed in config['train_seeds']:
            tag='calibration_'+str(seed);out=root/tag
            command=[sys.executable,str(root/'source_snapshot/scripts/calibrate_tg_pm.py'),
                '--environment-config',str(Path(template['output'])/'logs/TG_MAPPO_environment.json'),
                '--training-seed',str(seed),'--scenes']+[str(x) for x in config['calibration_scenes']]+[
                '--pm-checkpoint',str(ROOT/config['pm_checkpoint']),'--output',str(out)]
            job={'tag':tag,'kind':'calibration','seed':seed,'output':str(out)}
            run_job(root,m,job,command)
            report=read(out/'reliability.json')
            require(report['optimizer_updates']==0 and report['weights_unchanged'] and len(report['episodes'])==4,'Invalid calibration phase')
            m['calibrations'][str(seed)]={'report':str(out/'reliability.json'),'report_sha256':digest(out/'reliability.json'),
                'decision':report['decision'],'interaction_slots':320}
            job.update(state='completed',interaction_slots=320)
            atomic(root/'manifest.json',m);print('DONE '+tag+' alpha='+str(report['decision']['effective_alpha']),flush=True)
        for seed in config['train_seeds']:
            models={};evals={}
            for group in config['groups']:
                common=list(template['base_command']);common[1]=str(root/'source_snapshot/scripts/pm_value_worker.py')
                common+=['--snapshot',str(root/'source_snapshot')]
                for key,value in (('--seed',seed),('--scenario_seed',seed),('--num_episodes',20),('--num_env_steps',1600)):
                    common[common.index(key)+1]=str(value)
                if group!='A':common+=['--use_pm_reward','true','--pm_reward_checkpoint',str(ROOT/config['pm_checkpoint']),
                    '--pm_potential_mode','task_return','--pm_value_mode','learned_shaped','--pm_reward_alpha','1']
                if group=='reliable':common+=['--pm_reliability_report',m['calibrations'][str(seed)]['report']]
                for kind,scene in [('training',None)]+[('evaluation',s) for s in config['eval_scenes']]:
                    tag='{}_{}_{}{}'.format(group,seed,kind,'' if scene is None else '_'+str(scene));command=list(common)
                    if kind=='evaluation':
                        for key,value in (('--seed',scene),('--scenario_seed',scene),('--num_episodes',2),('--num_env_steps',160)):
                            command[command.index(key)+1]=str(value)
                        command+=['--tg_evaluate','true','--tg_policy_checkpoint',str(models[group])]
                    command+=['--output_root',str(root/tag),'--run-name','case']
                    job={'tag':tag,'kind':kind,'group':group,'seed':seed,'scene':scene,'output':str(root/tag/'runs/case/moves_5')}
                    if kind=='evaluation':job['policy_sha256']=digest(models[group])
                    run_job(root,m,job,command);report=inspect(job,m)
                    if kind=='training':models[group]=Path(report['checkpoint'])
                    else:evals[(group,scene)]=Path(job['output'])
                    if group=='reliable':
                        reference='always' if m['calibrations'][str(seed)]['decision']['enabled'] else 'A'
                        if kind=='training':
                            x,y=[torch.load(str(models[g]),map_location='cpu') for g in (group,reference)]
                            require(model_equal(x,y),'Frozen-gate PPO deviated from its declared branch')
                        else:
                            require(digest(Path(job['output'])/'logs/TG_physical_metrics.jsonl')==
                                digest(evals[(reference,scene)]/'logs/TG_physical_metrics.jsonl'),'Gate evaluation differs from selected branch')
                        job['declared_branch_exact']=reference
                    job['state']='completed';atomic(root/'manifest.json',m)
                    print('DONE '+tag+' delivered {}/{}'.format(job['delivered'],job['generated']),flush=True)
        # This summary helper ignores calibration jobs because their kind is distinct.
        m['summary']=summarize(m)
        means={g:sum(r['delivery_rate'] for r in m['summary'] if r['group']==g)/3 for g in config['groups']}
        returns={g:sum(r['mean_discounted_base_return'] for r in m['summary'] if r['group']==g)/3 for g in config['groups']}
        rows={(r['group'],r['seed']):r for r in m['summary']}
        wins=sum(rows[('reliable',s)]['delivery_rate']>rows[('A',s)]['delivery_rate'] for s in config['train_seeds'])
        nonworse=all(rows[('reliable',s)]['delivery_rate']>=rows[('A',s)]['delivery_rate'] for s in config['train_seeds'])
        m['development_gate']={'means':means,'base_returns':returns,'wins_vs_A':wins,'no_seed_worse_than_A':nonworse,
            'passed':means['reliable']>max(means['A'],means['always']) and wins>=2 and nonworse and returns['reliable']>returns['A'],
            'formal_efficacy_proven':False}
        m['status']='completed'
    except Exception as exc:
        m['status']='failed';m['error']=repr(exc)
        if m['jobs'] and m['jobs'][-1]['state']=='running':m['jobs'][-1]['state']='failed'
        atomic(root/'manifest.json',m);raise
    atomic(root/'manifest.json',m);print('COMPLETED '+str(root/'manifest.json'),flush=True)
    print(json.dumps(m['development_gate']),flush=True)


if __name__=='__main__':main()
