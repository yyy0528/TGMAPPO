"""Compatibility first, then one prespecified equal-budget PM return pilot.

Every task uses a new archived source and output. No optimizer resume or retries.
The 20-episode baseline is rerun and must equal historical A weights/physical logs.
"""
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
from buoy_network.ablation_suite import read, atomic, digest, snapshot, require

CONFIG=ROOT/'configs/tg_pm_task_return_pilot.json'
OLD=ROOT/'results/tg_ablation_lr_fixed_pilot/20261103_211528_024882/manifest.json'


def model_equal(a,b):
    return all(set(a[n])==set(b[n]) and all(torch.equal(a[n][k],b[n][k]) for k in a[n]) for n in ('actor','critic'))


def physical_rows(output):
    return [json.loads(x) for x in (output/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]


def frozen_source(preflight):
    require(preflight['status']=='completed' and preflight['stage']=='preflight','Successful preflight required')
    for path, checksum in preflight['source']['files_sha256'].items():
        require(digest(Path(preflight['root'])/'source_snapshot'/path)==checksum,'Preflight archive changed: '+path)
        require(digest(ROOT/path)==checksum,'Source/config changed since preflight: '+path)


def inspect(job, config, episodes, evaluation=False):
    output=Path(job['output']);report=read(output/'logs/TG_policy_report.json')
    actual=read(output/'logs/TG_MAPPO_environment.json')['config']
    require(actual['num_episodes']==episodes and report['profile']['collect_calls']==episodes*80,'Wrong interaction budget')
    require(actual['ppo_epoch']==1 and actual['num_mini_batch']==2 and not actual['use_linear_lr_decay']
            and actual['tg_critic_time_mode']=='disabled' and actual['random_channel']
            and actual['num_moving_buoys']==5 and actual['tg_reward_version']=='task_v2','Protocol drift')
    require(not actual['use_wm_actor'] and actual['actor_extra_width']==0,'Wrong Actor variant')
    if job['group']!='A':
        pm=report['pm']; require(pm['frozen'] and pm['potential_mode']==job['group']
            and pm['value_mode']=='learned_shaped' and pm['alpha']==1.0 and pm['gamma']==.99
            and pm['checkpoint_sha256']==config['pm_sha256'],'Wrong PM variant')
        require(all(r['boundary_error']<1e-7 and r['terminal_phi']==0 for r in pm['episodes']),'PM boundary failure')
    else:require(report['pm'] is None,'Baseline has a PM')
    if evaluation:
        require(report['optimizer_updates']==0 and report['actor_unchanged'] and report['critic_unchanged'],'Evaluation updated weights')
        require(report['checkpoint_sha256']==job['policy_sha256'],'Wrong evaluation checkpoint')
    else:
        updates=report['training_updates'];require(len(updates)==episodes,'Missing PPO updates')
        for i,row in enumerate(updates,1):
            require(row['actor_optimizer_steps']==row['critic_head_optimizer_steps']==i*2,'Wrong Adam budget')
            require(all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))),'Nonfinite training diagnostics')
        require(digest(report['checkpoint'])==report['checkpoint_sha256'],'Training checkpoint changed')
    rows=physical_rows(output)
    require(len(rows)==episodes and all(r['generated_packets']==r['delivered_packets']+r['unresolved_packets'] for r in rows),'Packet conservation')
    from buoy_network.task_reward_audit import audit_reward_csv
    audit_reward_csv(output/'logs',report['reward_contract'])
    job['delivered']=sum(r['delivered_packets'] for r in rows)
    job['generated']=sum(r['generated_packets'] for r in rows)
    job['loops']=sum(r['loop_transfers'] for r in rows)
    job['checkpoint_sha256']=report.get('checkpoint_sha256')
    with (output/'logs/TG_reward_components.csv').open(encoding='utf-8',newline='') as stream:
        rewards=list(csv.DictReader(stream))
    job['mean_discounted_base_return']=sum(.99**int(r['step'])*float(r['reward_base']) for r in rewards)/episodes
    return report


def exact_reference(output, reference):
    a,b=[torch.load(str(p/'checkpoints/tg_policy_latest.pt'),map_location='cpu') for p in (output,reference)]
    require(model_equal(a,b),'Legacy Actor/Critic changed')
    for f in ('TG_physical_metrics.jsonl','TG_reward_components.csv'):
        require(digest(output/'logs'/f)==digest(reference/'logs'/f),'Legacy physical/reward changed')


def summarize(manifest):
    rows=[]
    for group in manifest['config']['groups']:
        for seed in manifest['config']['train_seeds']:
            jobs=[j for j in manifest['jobs'] if j['kind']=='evaluation' and j['group']==group and j['seed']==seed]
            if not jobs:continue
            rows.append({'group':group,'seed':seed,'delivered':sum(j['delivered'] for j in jobs),
                'generated':sum(j['generated'] for j in jobs),'loops':sum(j['loops'] for j in jobs),
                'delivery_rate':sum(j['delivered'] for j in jobs)/sum(j['generated'] for j in jobs),
                'mean_discounted_base_return':sum(j['mean_discounted_base_return'] for j in jobs)/len(jobs)})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('preflight','pilot'))
    p.add_argument('--preflight-root')
    p.add_argument('--output-root',default='results/tg_pm_task_return')
    args=p.parse_args()
    config=read(CONFIG)
    require(digest(ROOT/config['pm_checkpoint'])==config['pm_sha256'],'PM weight mismatch')
    if args.stage=='pilot':
        require(args.preflight_root is not None,'Provide completed preflight directory')
        frozen_source(read(Path(args.preflight_root)/'manifest.json'))
    old=read(OLD)
    root=ROOT/args.output_root/(args.stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True,exist_ok=False)
    manifest={'root':str(root),'stage':args.stage,'config':config,'source':snapshot(root),
              'status':'running','jobs':[],'scope':'development pilot; not formal or 100-episode acceptance'}
    atomic(root/'manifest.json',manifest);print('OUTPUT '+str(root),flush=True)
    train_episodes=2 if args.stage=='preflight' else config['train_episodes']
    scenes=[7001] if args.stage=='preflight' else config['eval_scenes']
    try:
        for seed in config['train_seeds']:
            for group in config['groups']:
                ref=next(j for j in old['jobs'] if j['train_seed']==seed and j['group']==('A' if group=='A' else 'C') and j['kind']=='training')
                common=list(ref['base_command']);common[1]=str(root/'source_snapshot/scripts/pm_value_worker.py')
                common+=['--snapshot',str(root/'source_snapshot')]
                if group!='A':common+=['--pm_potential_mode',group,'--pm_value_mode','learned_shaped']
                # Replace the one declared episode/step pair, preserving original defaults.
                common[common.index('--num_episodes')+1]=str(train_episodes)
                common[common.index('--num_env_steps')+1]=str(train_episodes*80)
                checkpoint=None
                tasks=[('training',None)]+([('evaluation',s) for s in scenes] if args.stage=='pilot' or group=='task_return' else [])
                for kind,scene in tasks:
                    tag='{}_{}_{}{}'.format(group,seed,kind,'' if scene is None else '_'+str(scene))
                    command=list(common)
                    if kind=='evaluation':
                        for key,value in (('--num_episodes',2),('--num_env_steps',160),('--seed',scene),('--scenario_seed',scene)):
                            command[command.index(key)+1]=str(value)
                        command+=['--tg_evaluate','true','--tg_policy_checkpoint',str(checkpoint)]
                    command+=['--output_root',str(root/tag),'--run-name','case']
                    job={'tag':tag,'group':group,'seed':seed,'kind':kind,'scene':scene,'command':command,
                         'output':str(root/tag/'runs/case/moves_5'),'state':'running'}
                    if checkpoint is not None:job['policy_sha256']=digest(checkpoint)
                    manifest['jobs'].append(job);atomic(root/'manifest.json',manifest)
                    print('START '+tag,flush=True)
                    with (root/(tag+'.log')).open('xb') as stream:
                        process=subprocess.run(command,cwd=str(root/'source_snapshot'),stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
                    job['exit_code']=process.returncode
                    require(process.returncode==0,'Task failed: '+tag+'; see '+str(root/(tag+'.log')))
                    report=inspect(job,config,train_episodes if kind=='training' else 2,kind=='evaluation')
                    if kind=='training':
                        checkpoint=Path(report['checkpoint'])
                        if args.stage=='preflight' and group in ('A','remaining'):
                            exact_reference(Path(job['output']),Path(ref['output']));job['legacy_exact']=True
                        if args.stage=='pilot' and group=='A':
                            original=ROOT/'results/tg_lr_budget_fixed/training/runs'/('A_seed_{}_e20_fixed'.format(seed))/'moves_5'
                            exact_reference(Path(job['output']),original);job['legacy_fixed20_exact']=True
                    job['state']='completed';atomic(root/'manifest.json',manifest)
                    print('DONE '+tag+' delivered {}/{}'.format(job['delivered'],job['generated']),flush=True)
        manifest['summary']=summarize(manifest)
        if args.stage=='pilot':
            means={g:sum(r['delivery_rate'] for r in manifest['summary'] if r['group']==g)/3 for g in config['groups']}
            returns={g:sum(r['mean_discounted_base_return'] for r in manifest['summary'] if r['group']==g)/3 for g in config['groups']}
            per={(r['group'],r['seed']):r for r in manifest['summary']}
            improved=sum(per[('task_return',s)]['delivery_rate']>per[('A',s)]['delivery_rate'] for s in config['train_seeds'])
            manifest['development_gate']={'means':means,'base_returns':returns,'improved_training_seeds':improved,
                'passed':means['task_return']>max(means['A'],means['remaining']) and improved>=2 and returns['task_return']>returns['A'],
                'formal_efficacy_proven':False}
        manifest['status']='completed'
    except Exception as exc:
        manifest['status']='failed';manifest['error']=repr(exc)
        if manifest['jobs'] and manifest['jobs'][-1]['state']=='running':manifest['jobs'][-1]['state']='failed'
        atomic(root/'manifest.json',manifest);raise
    atomic(root/'manifest.json',manifest)
    print('COMPLETED '+str(root/'manifest.json'),flush=True)
    if 'development_gate' in manifest:print(json.dumps(manifest['development_gate']),flush=True)


if __name__=='__main__':main()
