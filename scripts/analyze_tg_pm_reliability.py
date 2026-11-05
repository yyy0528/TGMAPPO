"""Independent raw calibration, PPO budget and declared-branch acceptance."""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from buoy_network.ablation_suite import read,digest,require
from buoy_network.pm_reliability import load_gate
from buoy_network.task_reward import reward_contract
from buoy_network.task_reward_audit import audit_reward_csv
from types import SimpleNamespace
from analyze_tg_pm_task_return import physical_return
from check_tg_pm_task_return import model_equal


def cached_gae(job,alpha):
    root=Path(job['output']).parent/'value_audit';rows=[]
    for episode in range(20):
        with np.load(str(root/('episode_{}.npz'.format(episode)))) as a:
            require(all(np.isfinite(a[k]).all() for k in a.files),'Nonfinite cache')
            require(np.all(a['phi'][-1]==0) and np.all(a['masks'][-1]==0),'Terminal mismatch')
            expected=a['base_rewards']+alpha*(.99*a['phi'][1:]-a['phi'][:-1])
            require(np.allclose(expected,a['shaped_rewards'],atol=1e-7,rtol=1e-6),'Calibration alpha not used in rewards')
            delta=a['shaped_rewards']+.99*a['raw_values'][1:]*a['masks'][1:]-a['raw_values'][:-1]
            gae=np.zeros_like(delta)
            for t in reversed(range(len(delta))):
                gae[t]=delta[t]+(.99*.95*gae[t+1]*a['masks'][t+1] if t+1<len(delta) else 0)
            error=float(np.max(np.abs(gae-a['actual_gae'])))
            require(error<1e-5,'Actual GAE mismatch')
            rows.append({'episode':episode,'gae_recompute_max_error':error,'gae_std':float(gae.std())})
    return rows


def audit(root):
    m=read(root/'manifest.json');c=m['config'];contract=reward_contract(SimpleNamespace(tg_reward_version='task_v2'))
    require(m['status']=='completed' and len(m['jobs'])==39 and all(j['state']=='completed' and j['exit_code']==0 for j in m['jobs']),'Incomplete experiment')
    for path,sha in m['source']['files_sha256'].items():require(digest(root/'source_snapshot'/path)==sha,'Source archive changed')
    gates={};calibration_sources=[]
    for seed in c['train_seeds']:
        info=m['calibrations'][str(seed)];require(digest(info['report'])==info['report_sha256'],'Calibration report changed')
        gates[seed]=load_gate(info['report'],c['pm_sha256'],80,25,.99,1.,contract)
        raw=read(info['report']);parts=[]
        for path in raw['source_parts']:
            part=read(path);out=Path(path).parent
            require(part['weights_unchanged'] and part['optimizer_updates']==0 and part['collect_calls']==160,'Calibration updated policy or wrong budget')
            require(part['initial_actor_sha256']==raw['initial_actor_sha256'] and part['initial_critic_sha256']==raw['initial_critic_sha256'],'Calibration initialization mismatch')
            require(part['training_seed']==seed and part['scene'] in c['calibration_scenes'],'Calibration scene/seed mismatch')
            logs=out/'moves_5/logs';actual=read(logs/'TG_MAPPO_environment.json')['config']
            require(actual['num_episodes']==2 and actual['num_env_steps']==actual['effective_num_env_steps']==160,'Calibration metadata budget mismatch')
            require(digest(logs/'TG_physical_metrics.jsonl')==part['physical_sha256'],'Calibration physical file changed')
            audit_reward_csv(logs,contract)
            with (logs/'TG_reward_components.csv').open(encoding='utf-8',newline='') as s:rewards=list(csv.DictReader(s))
            with (logs/'TG_PM_rewards.csv').open(encoding='utf-8',newline='') as s:pm=list(csv.DictReader(s))
            for ep,row in enumerate(part['episodes']):
                require(row['rewards']==[float(x['reward_base']) for x in rewards[ep*80:(ep+1)*80]],'Calibration labels differ from physical rewards')
                require(row['phi']==[float(x['phi_t']) for x in pm[ep*80:(ep+1)*80]]+[0.],'Calibration potentials differ')
            parts.extend(part['episodes']);calibration_sources.append({'path':path,'sha256':digest(path),'scene':part['scene'],'slots':160})
        require(parts==raw['episodes'],'Gate is not derived from declared independent calibration cache')
    rows=[];artifacts=[];values=[]
    for job in m['jobs']:
        if job['kind']=='calibration':continue
        out=Path(job['output']);r=read(out/'logs/TG_policy_report.json');a=read(out/'logs/TG_MAPPO_environment.json')['config']
        ep=20 if job['kind']=='training' else 2;seed=job['seed'];group=job['group']
        require(a['seed']==a['scenario_seed']==(seed if job['kind']=='training' else job['scene']),'Wrong scenario seed')
        require(a['num_episodes']==ep and a['num_env_steps']==ep*80 and r['profile']['collect_calls']==ep*80,'Wrong interaction count')
        require(a['ppo_epoch']==1 and a['num_mini_batch']==2 and not a['use_linear_lr_decay']
                and not a['use_wm_actor'] and a['actor_extra_width']==0 and a['tg_critic_time_mode']=='disabled'
                and a['num_moving_buoys']==5 and a['random_channel'],'PPO/environment protocol mismatch')
        require(r['reward_contract']==contract and r['resources']['actor_trainable']==21956 and r['resources']['critic_trainable']==18615,'Reward/capacity mismatch')
        if group=='A':require(r['pm'] is None,'Baseline contains PM')
        else:
            pm=r['pm'];alpha=gates[seed]['effective_alpha'] if group=='reliable' else 1.
            require(pm['alpha']==alpha and pm['frozen'] and pm['potential_mode']=='task_return'
                and pm['value_mode']=='learned_shaped' and pm['gamma']==.99 and pm['checkpoint_sha256']==c['pm_sha256'],'Wrong PM')
            require(all(s['terminal_phi']==0 and s['boundary_error']<1e-7 for s in pm['episodes']),'Potential boundary error')
            require((pm.get('reliability') is not None)==(group=='reliable'),'Wrong gate mode')
            if group=='reliable':require(pm['reliability']['report_sha256']==gates[seed]['report_sha256'],'Wrong calibration loaded')
            if job['kind']=='training':values.append({'group':group,'seed':seed,'alpha':alpha,'rollouts':cached_gae(job,alpha)})
        physical=[json.loads(x) for x in (out/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(physical)==ep and all(x['generated_packets']==x['delivered_packets']+x['unresolved_packets'] for x in physical),'Packet conservation')
        audit_reward_csv(out/'logs',contract)
        base_return=physical_return(physical,contract)
        require(abs(base_return-job['mean_discounted_base_return'])<1e-10,'Base return differs from physical timestamps')
        if job['kind']=='training':
            require(digest(r['checkpoint'])==r['checkpoint_sha256'] and len(r['training_updates'])==20,'Wrong training artifact/count')
            for i,update in enumerate(r['training_updates'],1):
                require(update['actor_optimizer_steps']==update['critic_head_optimizer_steps']==2*i,'Adam budget mismatch')
                require(update['actor_lr']==.0005 and update['critic_lr']==.0003,'LR mismatch')
                require(all(math.isfinite(v) for v in update.values() if isinstance(v,(int,float))),'Nonfinite update')
        else:
            training=next(x for x in m['jobs'] if x['kind']=='training' and x['group']==group and x['seed']==seed)
            require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged']
                and digest(Path(training['output'])/'checkpoints/tg_policy_latest.pt')==r['checkpoint_sha256'],'Frozen evaluation changed model or wrong weights')
            rows.append({'group':group,'seed':seed,'scene':job['scene'],'delivery_rate':job['delivered']/job['generated'],
                'base_return':base_return,'loops':job['loops']})
        if group=='reliable':
            branch='always' if gates[seed]['enabled'] else 'A'
            reference=next(x for x in m['jobs'] if x['kind']==job['kind'] and x.get('group')==branch and x['seed']==seed and x.get('scene')==job['scene'])
            if job['kind']=='training':
                x,y=[torch.load(str(Path(j['output'])/'checkpoints/tg_policy_latest.pt'),map_location='cpu') for j in (job,reference)]
                require(model_equal(x,y),'Gate training branch differs')
            require(digest(out/'logs/TG_physical_metrics.jsonl')==digest(Path(reference['output'])/'logs/TG_physical_metrics.jsonl'),'Gate physical branch differs')
        artifacts.append({'tag':job['tag'],'report_sha256':digest(out/'logs/TG_policy_report.json'),
            'physical_sha256':digest(out/'logs/TG_physical_metrics.jsonl')})
    by_seed=[]
    for group in c['groups']:
        for seed in c['train_seeds']:
            selected=[r for r in rows if r['group']==group and r['seed']==seed]
            require({x['scene'] for x in selected}==set(c['eval_scenes']),'Missing evaluation scenes')
            by_seed.append({'group':group,'seed':seed,'delivery_rate':statistics.mean(x['delivery_rate'] for x in selected),
                'base_return':statistics.mean(x['base_return'] for x in selected),'scene_results':selected})
    means={g:statistics.mean(x['delivery_rate'] for x in by_seed if x['group']==g) for g in c['groups']}
    returns={g:statistics.mean(x['base_return'] for x in by_seed if x['group']==g) for g in c['groups']}
    per={(x['group'],x['seed']):x for x in by_seed}
    wins=sum(per[('reliable',s)]['delivery_rate']>per[('A',s)]['delivery_rate'] for s in c['train_seeds'])
    nonworse=all(per[('reliable',s)]['delivery_rate']>=per[('A',s)]['delivery_rate'] for s in c['train_seeds'])
    gate={'means':means,'base_returns':returns,'wins_vs_A':wins,'no_seed_worse_than_A':nonworse,
        'passed':means['reliable']>max(means['A'],means['always']) and wins>=2 and nonworse and returns['reliable']>returns['A']}
    require(gate['passed']==m['development_gate']['passed'],'Efficacy decision differs')
    return {'engineering_passed':True,'audited_tasks':39,'root':str(root),'manifest_sha256':digest(root/'manifest.json'),
        'auditor_sha256':digest(Path(__file__)),'gates':gates,'calibration_sources':calibration_sources,
        'by_training_seed':by_seed,'development_gate':gate,'training_value_diagnostics':values,'artifacts':artifacts,
        'budget':{'unique_calibration_steps':960,'PPO_training_steps':14400,'evaluation_steps':4320,
            'per_variant_training_steps':1600,'additional_calibration_steps_per_initialization':320,
            'calibration_shared_across_variants':True},
        'formal_efficacy_proven':False,'full_interaction_efficiency_proven':False,
        'limitations':['initial-value calibration is not an uncertainty model or a routing guarantee',
            'three training seeds, development scenes, 20 PPO episodes only',
            'A is also charged calibration cost; no A24 full interaction-budget control run']}


def plot(report,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=['A','always','reliable'];labels=['TG','PM always','PM calibrated']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,label in ((axes[0],'delivery_rate','Delivered / generated'),(axes[1],'base_return','Discounted environment return')):
        for seed,color in zip((44,55,66),('#3971a4','#d28332','#4d9666')):
            data=[next(x[metric] for x in report['by_training_seed'] if x['group']==g and x['seed']==seed) for g in groups]
            ax.plot(range(3),data,'o-',label='Seed {} (gate alpha={})'.format(seed,report['gates'][seed]['effective_alpha']),color=color)
        ax.set_xticks(range(3));ax.set_xticklabels(labels);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Development: 20 PPO episodes/policy + 4 independent calibration episodes/initialization')
    fig.tight_layout(rect=[0,0,1,.94]);folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(str(folder/'pm_reliability_comparison.png'),dpi=160);fig.savefig(str(folder/'pm_reliability_comparison.svg'))
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--output')
    a=p.parse_args();root=Path(a.root).resolve();report=audit(root);plot(report,root)
    report['figures']={str(p.relative_to(root)):digest(p) for p in (root/'figures').iterdir() if p.is_file()}
    output=Path(a.output) if a.output else root/'acceptance.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['development_gate'],indent=2));print('REPORT '+str(output))


if __name__=='__main__':main()
