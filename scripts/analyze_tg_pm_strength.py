"""Independent single-variable alpha audit against immutable physical references."""
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
from buoy_network.ablation_suite import read,digest,require
from buoy_network.task_reward_audit import audit_reward_csv
from analyze_tg_pm_task_return import physical_return


def cache_audit(output,alpha,episodes=20):
    result=[]
    for ep in range(episodes):
        with np.load(str(output.parent/'value_audit'/('episode_{}.npz'.format(ep)))) as a:
            require(all(np.isfinite(a[k]).all() for k in a.files),'Nonfinite cached values')
            require(np.all(a['phi'][-1]==0) and np.all(a['masks'][-1]==0),'Bad terminal cache')
            expected=a['base_rewards']+alpha*(.99*a['phi'][1:]-a['phi'][:-1])
            require(np.allclose(expected,a['shaped_rewards'],rtol=1e-6,atol=1e-7),'Wrong shaping strength')
            delta=a['shaped_rewards']+.99*a['raw_values'][1:]*a['masks'][1:]-a['raw_values'][:-1]
            gae=np.zeros_like(delta)
            for t in reversed(range(len(delta))):gae[t]=delta[t]+(.99*.95*gae[t+1]*a['masks'][t+1] if t+1<len(delta) else 0)
            error=float(np.max(np.abs(gae-a['actual_gae'])))
            require(error<1e-5,'GAE recompute failed')
            result.append({'episode':ep,'actual_GAE_error':error,'GAE_std':float(gae.std())})
    return result


def audit(root):
    m=read(root/'manifest.json');reference=Path(m['runtime_source_origin']);old=read(reference/'manifest.json')
    require(m['status']=='completed' and len(m['jobs'])==12 and all(j['state']=='completed' and j['exit_code']==0 for j in m['jobs']),'Incomplete strength probe')
    require(digest(reference/'manifest.json')==m['reference_manifest_sha256'],'Reference manifest changed')
    accepted=read(reference/'acceptance.json');fingerprints={x['tag']:x for x in accepted['artifacts']}
    require(accepted['manifest_sha256']==m['reference_manifest_sha256'],'Reference acceptance/manifest mismatch')
    for path,sha in m['source']['files_sha256'].items():
        require(digest(reference/'source_snapshot'/path)==digest(root/'source_snapshot'/path)==sha,'Runtime source differs')
    groups={'A':[],'task_return':[],'alpha01':[]};artifacts=[];diagnostics=[];initial=[]
    for group,jobs in (('A',[j for j in old['jobs'] if j['group']=='A']),
                       ('task_return',[j for j in old['jobs'] if j['group']=='task_return']),('alpha01',m['jobs'])):
        for j in jobs:
            out=Path(j['output']);r=read(out/'logs/TG_policy_report.json')
            if group!='alpha01':
                require(digest(out/'logs/TG_policy_report.json')==fingerprints[j['tag']]['report_sha256']
                    and digest(out/'logs/TG_physical_metrics.jsonl')==fingerprints[j['tag']]['physical_sha256'],'Historical reference artifact changed')
            else:
                require(digest(out/'logs/TG_policy_report.json')==j['report_sha256'],'Strength report changed')
                ref=next(x for x in old['jobs'] if x['group']=='task_return' and x['kind']==j['kind'] and x['seed']==j['seed'] and x['scene']==j['scene'])
                actual=read(out/'logs/TG_MAPPO_environment.json')['config'];previous=read(Path(ref['output'])/'logs/TG_MAPPO_environment.json')['config']
                ignored={'output_root','tg_policy_checkpoint','pm_reward_alpha'}
                require({k:v for k,v in actual.items() if k not in ignored}=={k:v for k,v in previous.items() if k not in ignored}
                    and actual['pm_reward_alpha']==.1,'Another training variable changed')
            episodes=20 if j['kind']=='training' else 2
            require(r['profile']['collect_calls']==episodes*80 and r['resources']['actor_trainable']==21956
                and r['resources']['critic_trainable']==18615,'Budget/capacity differs')
            if group!='A':
                alpha=.1 if group=='alpha01' else 1.
                require(r['pm']['alpha']==alpha and r['pm']['frozen'] and r['pm']['potential_mode']=='task_return'
                    and r['pm']['value_mode']=='learned_shaped','Wrong PM alpha/mode')
                require(all(x['terminal_phi']==0 and x['boundary_error']<1e-7 for x in r['pm']['episodes']),'Potential boundary error')
            physical=[json.loads(x) for x in (out/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            require(len(physical)==episodes and all(x['generated_packets']==x['delivered_packets']+x['unresolved_packets'] for x in physical),'Packet conservation')
            audit_reward_csv(out/'logs',r['reward_contract'])
            if j['kind']=='training':
                require(digest(r['checkpoint'])==r['checkpoint_sha256'] and len(r['training_updates'])==20,'Wrong training artifact')
                for i,u in enumerate(r['training_updates'],1):
                    require(u['actor_optimizer_steps']==u['critic_head_optimizer_steps']==2*i and u['actor_lr']==.0005 and u['critic_lr']==.0003,'PPO budget/LR changed')
                    require(all(math.isfinite(v) for v in u.values() if isinstance(v,(int,float))),'Nonfinite update')
                if group=='alpha01':
                    diagnostics.append({'seed':j['seed'],'rollouts':cache_audit(out,.1)})
                    with np.load(str(out.parent/'value_audit/episode_0.npz')) as low,np.load(str(Path(ref['output']).parent/'value_audit/episode_0.npz')) as high:
                        for key in ('base_rewards','raw_values','phi','masks'):
                            require(np.array_equal(low[key],high[key]),'First trajectory differs before PPO: '+key)
                        error=float(np.max(np.abs(low['actual_gae']-(.9*high['base_gae']+.1*high['actual_gae']))))
                        require(error<1e-5,'Initial GAE change not explained by alpha')
                        initial.append({'seed':j['seed'],'base_values_potential_masks_exact':True,
                            'linear_alpha_GAE_error':error,'interpretation':'same trajectory; later policies/trajectories naturally diverge'})
            else:
                source=next(x for x in jobs if x['kind']=='training' and x['seed']==j['seed'])
                require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged']
                    and r['checkpoint_sha256']==digest(Path(source['output'])/'checkpoints/tg_policy_latest.pt'),'Bad frozen evaluation')
                groups[group].append({'seed':j['seed'],'scene':j['scene'],
                    'delivery_rate':sum(x['delivered_packets'] for x in physical)/sum(x['generated_packets'] for x in physical),
                    'delivered':sum(x['delivered_packets'] for x in physical),'generated':sum(x['generated_packets'] for x in physical),
                    'base_return':physical_return(physical,r['reward_contract']),'loops':sum(x['loop_transfers'] for x in physical)})
            artifacts.append({'tag':j['tag'],'group':group,'report_sha256':digest(out/'logs/TG_policy_report.json'),
                'physical_sha256':digest(out/'logs/TG_physical_metrics.jsonl')})
    rows=[]
    for group,scenes in groups.items():
        for seed in (11,22,33):
            selected=[x for x in scenes if x['seed']==seed]
            require({x['scene'] for x in selected}=={8001,8002,8003},'Evaluation scene mismatch')
            rows.append({'group':group,'seed':seed,'delivery_rate':statistics.mean(x['delivery_rate'] for x in selected),
                'base_return':statistics.mean(x['base_return'] for x in selected),'scenes':selected})
    means={g:statistics.mean(x['delivery_rate'] for x in rows if x['group']==g) for g in groups}
    returns={g:statistics.mean(x['base_return'] for x in rows if x['group']==g) for g in groups}
    per={(x['group'],x['seed']):x for x in rows}
    wins=sum(per[('alpha01',s)]['delivery_rate']>per[('A',s)]['delivery_rate'] for s in (11,22,33))
    nonworse=all(per[('alpha01',s)]['delivery_rate']>=per[('A',s)]['delivery_rate'] for s in (11,22,33))
    gate={'means':means,'base_returns':returns,'wins_vs_A':wins,'no_seed_worse_than_A':nonworse,
        'passed':means['alpha01']>max(means['A'],means['task_return']) and returns['alpha01']>max(returns['A'],returns['task_return']) and wins>=2 and nonworse}
    require(gate['passed']==m['development_gate']['passed'],'Efficacy decision differs')
    return {'engineering_passed':True,'new_tasks':12,'reference_tasks':24,'root':str(root),
        'manifest_sha256':digest(root/'manifest.json'),'auditor_sha256':digest(Path(__file__)),
        'by_training_seed':rows,'development_gate':gate,'initial_controls':initial,'training_GAE':diagnostics,
        'artifacts':artifacts,'formal_efficacy_proven':False,
        'budget':{'new_training_slots':4800,'new_frozen_evaluation_slots':1440,'reference_training_repeated':False},
        'limitations':['targeted previously-used development seeds/scenes, not blind generalization',
            'three training seeds and 20 episodes only; no fixed100/1000 confirmation',
            'old archived source isolates alpha; current entry compatibility is checked separately']}


def plot(report,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=['A','task_return','alpha01'];labels=['TG','PM alpha=1','PM alpha=0.1']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,label in ((axes[0],'delivery_rate','Delivered / generated'),(axes[1],'base_return','Discounted environment return')):
        for seed,color in zip((11,22,33),('#3971a4','#d28332','#4d9666')):
            data=[next(x[metric] for x in report['by_training_seed'] if x['group']==g and x['seed']==seed) for g in groups]
            ax.plot(range(3),data,'o-',color=color,label='Training seed '+str(seed))
        ax.set_xticks(range(3));ax.set_xticklabels(labels);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Targeted development: identical source, 20 PPO episodes/seed, one alpha change')
    fig.tight_layout(rect=[0,0,1,.94]);folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(str(folder/'pm_strength_comparison.png'),dpi=160);fig.savefig(str(folder/'pm_strength_comparison.svg'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--output')
    a=p.parse_args();root=Path(a.root).resolve();report=audit(root);plot(report,root)
    report['figures']={str(p.relative_to(root)):digest(p) for p in (root/'figures').iterdir() if p.is_file()}
    out=Path(a.output) if a.output else root/'acceptance.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['development_gate'],indent=2));print('REPORT '+str(out))


if __name__=='__main__':main()
