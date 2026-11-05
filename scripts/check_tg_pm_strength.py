"""One alpha=0.1 regression using exactly the old task-return source snapshot.

Only alpha and output/checkpoint paths change. Historical baseline and alpha=1
models are read-only references, with their original runtime hash closure checked.
"""
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from buoy_network.ablation_suite import read,atomic,digest,require
from buoy_network.task_reward_audit import audit_reward_csv
from analyze_tg_pm_task_return import physical_return


def setarg(command,key,value):
    if key in command:command[command.index(key)+1]=str(value)
    else:command.extend([key,str(value)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',default='results/tg_pm_strength');a=p.parse_args()
    config=read(ROOT/'configs/tg_pm_strength_probe.json');reference=ROOT/config['reference_suite'];old=read(reference/'manifest.json')
    require(old['status']=='completed','Reference experiment incomplete')
    root=ROOT/a.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True,exist_ok=False)
    for path,sha in old['source']['files_sha256'].items():require(digest(reference/'source_snapshot'/path)==sha,'Reference source changed')
    shutil.copytree(str(reference/'source_snapshot'),str(root/'source_snapshot'),ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for path,sha in old['source']['files_sha256'].items():require(digest(root/'source_snapshot'/path)==sha,'Copied source differs')
    m={'root':str(root),'status':'running','config':config,'source':old['source'],'runtime_source_origin':str(reference),
        'controller_sha256':digest(Path(__file__)),'reference_manifest_sha256':digest(reference/'manifest.json'),'jobs':[]}
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True)
    try:
        for seed in config['train_seeds']:
            checkpoint=None
            for kind,scene in [('training',None)]+[('evaluation',s) for s in config['eval_scenes']]:
                ref=next(j for j in old['jobs'] if j['group']=='task_return' and j['seed']==seed and j['kind']==kind and j['scene']==scene)
                tag='alpha01_{}_{}{}'.format(seed,kind,'' if scene is None else '_'+str(scene))
                command=list(ref['command']);command[1]=str(root/'source_snapshot/scripts/pm_value_worker.py')
                for key,value in (('--snapshot',root/'source_snapshot'),('--output_root',root/tag),('--pm_reward_alpha',config['alpha'])):setarg(command,key,value)
                if kind=='evaluation':setarg(command,'--tg_policy_checkpoint',checkpoint)
                job={'tag':tag,'seed':seed,'kind':kind,'scene':scene,'command':command,'state':'running',
                    'output':str(root/tag/'runs/case/moves_5')}
                m['jobs'].append(job);atomic(root/'manifest.json',m);print('START '+tag,flush=True)
                with (root/(tag+'.log')).open('xb') as stream:
                    process=subprocess.run(command,cwd=str(root/'source_snapshot'),stdout=stream,stderr=subprocess.STDOUT,timeout=2400)
                job['exit_code']=process.returncode;require(process.returncode==0,'Task failed: '+tag)
                output=Path(job['output']);r=read(output/'logs/TG_policy_report.json')
                actual=read(output/'logs/TG_MAPPO_environment.json')['config']
                previous=read(Path(ref['output'])/'logs/TG_MAPPO_environment.json')['config']
                ignore={'output_root','pm_reward_alpha','tg_policy_checkpoint'}
                require({k:v for k,v in actual.items() if k not in ignore}=={k:v for k,v in previous.items() if k not in ignore},'Non-alpha training configuration changed')
                require(actual['pm_reward_alpha']==.1 and r['pm']['alpha']==.1 and r['pm']['potential_mode']=='task_return'
                    and r['pm']['value_mode']=='learned_shaped' and r['pm']['frozen'],'Wrong alpha/model mode')
                require(r['pm']['checkpoint_sha256']==read(Path(ref['output'])/'logs/TG_policy_report.json')['pm']['checkpoint_sha256'],'PM checkpoint changed')
                require(all(x['boundary_error']<1e-7 and x['terminal_phi']==0 for x in r['pm']['episodes']),'Boundary error')
                episodes=20 if kind=='training' else 2
                require(r['profile']['collect_calls']==episodes*80,'Wrong environment budget')
                if kind=='training':
                    require(len(r['training_updates'])==20,'Wrong PPO count')
                    for i,u in enumerate(r['training_updates'],1):
                        require(u['actor_optimizer_steps']==u['critic_head_optimizer_steps']==2*i,'Wrong Adam count')
                        require(all(math.isfinite(v) for v in u.values() if isinstance(v,(int,float))),'Nonfinite training')
                    checkpoint=Path(r['checkpoint']);require(digest(checkpoint)==r['checkpoint_sha256'],'Checkpoint changed')
                else:require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged']
                    and digest(checkpoint)==r['checkpoint_sha256'],'Evaluation updated or loaded wrong weights')
                physical=[json.loads(x) for x in (output/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
                audit_reward_csv(output/'logs',r['reward_contract'])
                require(len(physical)==episodes,'Wrong episode metrics')
                job.update(state='completed',delivered=sum(x['delivered_packets'] for x in physical),
                    generated=sum(x['generated_packets'] for x in physical),loops=sum(x['loop_transfers'] for x in physical),
                    base_return=physical_return(physical,r['reward_contract']),report_sha256=digest(output/'logs/TG_policy_report.json'))
                atomic(root/'manifest.json',m);print('DONE '+tag+' delivered {}/{}'.format(job['delivered'],job['generated']),flush=True)
        rows=[]
        for seed in config['train_seeds']:
            for group in ('A','task_return','alpha01'):
                selected=[j for j in (m['jobs'] if group=='alpha01' else old['jobs'])
                    if j['kind']=='evaluation' and j['seed']==seed and (group=='alpha01' or j['group']==group)]
                rates=[j['delivered']/j['generated'] for j in selected]
                returns=[j['base_return'] if group=='alpha01' else j['mean_discounted_base_return'] for j in selected]
                rows.append({'group':group,'seed':seed,'delivery_rate':sum(rates)/len(rates),'base_return':sum(returns)/len(returns)})
        means={g:sum(x['delivery_rate'] for x in rows if x['group']==g)/3 for g in ('A','task_return','alpha01')}
        returns={g:sum(x['base_return'] for x in rows if x['group']==g)/3 for g in ('A','task_return','alpha01')}
        per={(x['group'],x['seed']):x for x in rows}
        wins=sum(per[('alpha01',s)]['delivery_rate']>per[('A',s)]['delivery_rate'] for s in config['train_seeds'])
        nonworse=all(per[('alpha01',s)]['delivery_rate']>=per[('A',s)]['delivery_rate'] for s in config['train_seeds'])
        m.update(status='completed',summary=rows,development_gate={'means':means,'base_returns':returns,'wins_vs_A':wins,
            'no_seed_worse_than_A':nonworse,'passed':means['alpha01']>max(means['A'],means['task_return']) and
            returns['alpha01']>max(returns['A'],returns['task_return']) and wins>=2 and nonworse,'formal_efficacy_proven':False})
    except Exception as exc:
        m['status']='failed';m['error']=repr(exc)
        if m['jobs'] and m['jobs'][-1]['state']=='running':m['jobs'][-1]['state']='failed'
        atomic(root/'manifest.json',m);raise
    atomic(root/'manifest.json',m);print('COMPLETED '+str(root/'manifest.json'),flush=True)
    print(json.dumps(m['development_gate']),flush=True)


if __name__=='__main__':main()
