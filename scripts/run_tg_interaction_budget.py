"""Prepare/resume a fixed-LR original-TG 100-episode baseline diagnostic."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.buoy_network.ablation_suite import read,digest,require,validate,create_suite,atomic,run_suite,budget
from src.buoy_network.task_reward_audit import audit_reward_csv


def audit_old(root):
    root=Path(root).resolve()
    commands=read(root/'commands.json')
    result=[]
    for seed in (11,22,33):
        entry=next(r for r in commands['training'] if r['seed']==seed)
        train=Path(entry['output'])
        actual=read(train/'logs/TG_MAPPO_environment.json')['config']
        report=read(train/'logs/TG_policy_report.json')
        require(actual['seed']==actual['scenario_seed']==seed and actual['num_episodes']==20
            and actual['episode_length']==80 and actual['ppo_epoch']==1 and actual['num_mini_batch']==2
            and actual['num_moving_buoys']==5 and actual['random_channel'] is True
            and not actual['use_linear_lr_decay'] and not actual['use_wm_actor'] and not actual['use_pm_reward']
            and actual.get('tg_critic_time_mode','disabled')=='disabled','Wrong preserved fixed20 protocol')
        require(report['learning_rate_schedule']['kind']=='fixed' and len(report['training_updates'])==20,'Missing fixed20 update records')
        require(digest(report['checkpoint'])==report['checkpoint_sha256'],'Old weights changed')
        for i,row in enumerate(report['training_updates']):
            require(row['critic_head_optimizer_steps']==(i+1)*2 and row['critic_head_in_optimizer']
                and row['actor_lr']==.0005 and row['critic_lr']==.0003,'Old fixed20 optimizer/LR mismatch')
        evaluation=root/'evaluation/runs'/('A_seed_{}_e20_fixed_scene7001'.format(seed))/'moves_5'
        ev=read(evaluation/'logs/TG_policy_report.json')
        require(ev['actor_unchanged'] and ev['critic_unchanged'] and ev['optimizer_updates']==0
            and ev['checkpoint_sha256']==report['checkpoint_sha256'],'Old evaluation was not frozen')
        audit_reward_csv(train/'logs',report['reward_contract'])
        audit_reward_csv(evaluation/'logs',ev['reward_contract'])
        rows=[json.loads(x) for x in (evaluation/'logs/TG_physical_metrics.jsonl').read_text().splitlines()]
        result.append({'seed':seed,'training':str(train),'evaluation':str(evaluation),
            'checkpoint':report['checkpoint'],'checkpoint_sha256':report['checkpoint_sha256'],
            'training_physics_sha256':digest(train/'logs/TG_physical_metrics.jsonl'),
            'evaluation_physics_sha256':digest(evaluation/'logs/TG_physical_metrics.jsonl'),
            'delivered':sum(r['delivered_packets'] for r in rows),'generated':sum(r['generated_packets'] for r in rows),
            'old_actor_optimizer_counter_available':'actor_optimizer_steps' in report['training_updates'][-1]})
    return {'success':True,'scope':'preserved fixed20; no retraining','policies':result}


def prepare(config,old_root,output_root):
    config=validate(config)
    require(config['train_episodes']==100 and config['learning_rate_schedule']=='fixed'
        and config.get('critic_time_mode')=='disabled' and config['groups']==['A']
        and config['train_seeds']==[11,22,33] and config['ppo_epoch']==1 and config['num_mini_batch']==2,
        'Require agreed fixed100 baseline protocol')
    old=audit_old(old_root)
    root=create_suite(config,output_root)
    atomic(root/'baseline20.json',old)
    manifest=read(root/'manifest.json')
    for job in manifest['jobs']:
        if job['kind']=='training':
            reference=next(r for r in old['policies'] if r['seed']==job['train_seed'])
            job['base_command'][1]=str(root/'source_snapshot/scripts/interaction_budget_worker.py')
            job['base_command']+=['--snapshot',str(root/'source_snapshot'),'--capture-update','20',
                                  '--prefix-reference',reference['checkpoint']]
    manifest['comparison']='old fixed20 vs new fixed100, prefix checkpoint must match exactly'
    atomic(root/'manifest.json',manifest)
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--resume')
    p.add_argument('--old-root',default='results/tg_lr_budget_fixed')
    p.add_argument('--output-root',default='results/tg_interaction_fixed100')
    p.add_argument('--mode',choices=('audit','plan','run'),default='plan')
    a=p.parse_args()
    if a.mode=='audit':
        print(json.dumps(audit_old(a.old_root),indent=2))
        return
    if a.resume:
        if a.mode=='run':
            run_suite(a.resume)
        else:
            print(json.dumps(read(Path(a.resume)/'manifest.json')['budget'],indent=2))
    else:
        if not a.config:
            p.error('--config required')
        config=validate(read(a.config))
        if a.mode=='plan':
            print(json.dumps({'new':budget(config),'old20':audit_old(a.old_root)},indent=2))
        else:
            root=prepare(config,a.old_root,a.output_root)
            print('Suite: '+str(root),flush=True)
            run_suite(root)


if __name__=='__main__':
    main()
