"""Audit completed PM return pilots from physical events, cached values and hashes."""
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


def physical_return(rows, contract):
    """Compute directly from delivered timestamps; no shaped reward CSV input."""
    values=[]
    for r in rows:
        h=r['horizon'];p=r['generated_packets']
        def cost(duration):return contract['delay_weight']/h*sum(.99**k for k in range(duration))
        delivered=sum(contract['delivery_weight']*.99**(d-1)-cost(d) for d in r['success_delays'])
        remaining=r['unresolved_packets']*(-cost(h)-contract['unfinished_weight']*.99**(h-1))
        values.append((delivered+remaining)/max(p,1))
    return statistics.mean(values)


def same_first_rollout(manifest):
    result=[]
    for seed in manifest['config']['train_seeds']:
        def cache(group):
            j=next(j for j in manifest['jobs'] if j['kind']=='training' and j['group']==group and j['seed']==seed)
            return np.load(str(Path(j['output']).parent/'value_audit/episode_0.npz'))
        with cache('remaining') as old,cache('task_return') as new:
            for key in ('base_rewards','raw_values','masks'):
                require(np.array_equal(old[key],new[key]),'First cached trajectory differs: '+key)
            base=new['base_rewards'];mc=np.zeros_like(base)
            for t in reversed(range(len(base))):mc[t]=base[t]+(.99*mc[t+1] if t+1<len(base) else 0)
            v=old['raw_values'][:-1]
            row={'seed':seed,'base_rewards_values_masks_exact':True,
                 'actual_gae_change_max':float(np.max(np.abs(old['actual_gae']-new['actual_gae']))),
                 'old_base_value_prior_sampled_MC_mse':float(((v+old['phi'][:-1]-mc)**2).mean()),
                 'new_base_value_prior_sampled_MC_mse':float(((v+new['phi'][:-1]-mc)**2).mean()),
                 'no_PM_same_numeric_value_sampled_MC_mse':float(((v-mc)**2).mean()),
                 'scope':'sampled first training rollout, all agent copies; not exact conditional V or independent samples'}
            require(row['actual_gae_change_max']>1e-5,'Candidate canceled from GAE')
            result.append(row)
    return result


def audit(root):
    m=read(root/'manifest.json');c=m['config']
    require(m['status']=='completed','Cannot accept an incomplete/failed experiment')
    expected=12 if m['stage']=='preflight' else 36
    require(len(m['jobs'])==expected and all(j['state']=='completed' and j['exit_code']==0 for j in m['jobs']),'Incomplete tasks')
    for path,checksum in m['source']['files_sha256'].items():
        require(digest(root/'source_snapshot'/path)==checksum,'Archived source changed: '+path)
    require(digest(ROOT/c['pm_checkpoint'])==c['pm_sha256'],'PM file changed')
    per_seed={}
    entries=[]
    for j in m['jobs']:
        out=Path(j['output']);r=read(out/'logs/TG_policy_report.json')
        actual=read(out/'logs/TG_MAPPO_environment.json')['config']
        episodes=(2 if m['stage']=='preflight' or j['kind']=='evaluation' else 20)
        require(actual['num_episodes']==episodes and actual['num_env_steps']==episodes*80
                and r['profile']['collect_calls']==episodes*80,'Wrong interaction count')
        expected_seed=j['seed'] if j['kind']=='training' else j['scene']
        require(actual['seed']==actual['scenario_seed']==expected_seed and not actual['use_wm_actor']
                and actual['actor_extra_width']==0 and actual['ppo_epoch']==1 and actual['num_mini_batch']==2
                and actual['tg_critic_time_mode']=='disabled' and not actual['use_linear_lr_decay']
                and actual['num_moving_buoys']==5 and actual['random_channel'],'Experimental control mismatch')
        require(r['resources']['actor_trainable']==21956 and r['resources']['critic_trainable']==18615,'Capacity mismatch')
        require(r['reward_contract']['version']=='task_v2' and all(r['reward_contract'][k]==1 for k in
                ('delivery_weight','delay_weight','unfinished_weight')),'Reward contract drift')
        if j['kind']=='training':
            require(digest(r['checkpoint'])==r['checkpoint_sha256'],'Checkpoint changed')
            require(len(r['training_updates'])==episodes,'PPO count mismatch')
            for i,row in enumerate(r['training_updates'],1):
                require(row['actor_optimizer_steps']==row['critic_head_optimizer_steps']==i*2,'Wrong actual Adam budget')
                require(row['actor_lr']==.0005 and row['critic_lr']==.0003,'LR drift')
            if j['group']=='A' or m['stage']=='preflight' and j['group']=='remaining':
                require(j.get('legacy_exact') or j.get('legacy_fixed20_exact'),'Missing legacy compatibility proof')
                if m['stage']=='preflight':
                    legacy=read(ROOT/'results/tg_ablation_lr_fixed_pilot/20261103_211528_024882/manifest.json')
                    original=Path(next(x['output'] for x in legacy['jobs'] if x['kind']=='training'
                        and x['group']==('A' if j['group']=='A' else 'C') and x['train_seed']==j['seed']))
                else:
                    original=ROOT/'results/tg_lr_budget_fixed/training/runs'/('A_seed_{}_e20_fixed'.format(j['seed']))/'moves_5'
                a,b=[torch.load(str(x/'checkpoints/tg_policy_latest.pt'),map_location='cpu') for x in (out,original)]
                require(all(set(a[n])==set(b[n]) and all(torch.equal(a[n][k],b[n][k]) for k in a[n])
                            for n in ('actor','critic')),'Historical parameters differ')
                for filename in ('TG_physical_metrics.jsonl','TG_reward_components.csv'):
                    require(digest(out/'logs'/filename)==digest(original/'logs'/filename),'Historical trajectory/reward differs')
        else:
            require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged'],'Frozen evaluation changed weights')
            train=next(x for x in m['jobs'] if x['kind']=='training' and x['group']==j['group'] and x['seed']==j['seed'])
            checkpoint=Path(train['output'])/'checkpoints/tg_policy_latest.pt'
            require(digest(checkpoint)==r['checkpoint_sha256'],'Wrong evaluation checkpoint')
        if j['group']!='A':
            pm=r['pm'];require(pm['frozen'] and pm['checkpoint_sha256']==c['pm_sha256']
                and pm['potential_mode']==j['group'] and pm['value_mode']=='learned_shaped'
                and pm['alpha']==1 and pm['gamma']==.99,'PM configuration/freeze mismatch')
            require(all(abs(s['boundary_error'])<1e-7 and s['terminal_phi']==0 for s in pm['episodes']),'Potential boundary error')
        physical=[json.loads(x) for x in (out/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(physical)==episodes and all(x['generated_packets']==x['delivered_packets']+x['unresolved_packets'] for x in physical),'Packet conservation')
        value=physical_return(physical,r['reward_contract'])
        require(abs(value-j['mean_discounted_base_return'])<1e-10,'CSV return differs from physical timestamps')
        if j['kind']=='evaluation':
            key=(j['group'],j['seed'])
            per_seed.setdefault(key,[]).append({'delivery_rate':j['delivered']/j['generated'],'return':value,
                'scene':j['scene'],'loops':j['loops']})
        entries.append({'tag':j['tag'],'report_sha256':digest(out/'logs/TG_policy_report.json'),
            'physical_sha256':digest(out/'logs/TG_physical_metrics.jsonl'),'return_from_physical_events':value})
    summary=[]
    for (group,seed),rows in sorted(per_seed.items()):
        summary.append({'group':group,'seed':seed,'delivery_rate':statistics.mean(x['delivery_rate'] for x in rows),
            'base_return':statistics.mean(x['return'] for x in rows),'scene_results':rows})
    gate=None
    if m['stage']=='pilot':
        means={g:statistics.mean(x['delivery_rate'] for x in summary if x['group']==g) for g in c['groups']}
        returns={g:statistics.mean(x['base_return'] for x in summary if x['group']==g) for g in c['groups']}
        by={(r['group'],r['seed']):r for r in summary}
        wins=sum(by[('task_return',s)]['delivery_rate']>by[('A',s)]['delivery_rate'] for s in c['train_seeds'])
        gate={'mean_delivery':means,'mean_base_return':returns,'wins_vs_A':wins,
              'passed':means['task_return']>max(means['A'],means['remaining']) and wins>=2 and returns['task_return']>returns['A']}
        require(gate['passed']==m['development_gate']['passed'],'Development decision mismatch')
    return {'engineering_passed':True,'stage':m['stage'],'audited_tasks':expected,'root':str(root),
            'auditor_sha256':digest(Path(__file__)),
            'manifest_sha256':digest(root/'manifest.json'),'first_rollout_controls':same_first_rollout(m),
            'by_training_seed':summary,'development_gate':gate,'formal_efficacy_proven':False,
            'artifacts':entries,'limitations':['20-episode development only; no fixed100 rerun',
                'three training seeds; scenes not independent training repetitions',
                'frozen behavior PM may be miscalibrated on new policies','CPU only']}


def plot(report,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=['A','remaining','task_return'];labels=['TG','TG+PM probability','TG+PM task return']
    rows=report['by_training_seed']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,label in ((axes[0],'delivery_rate','Delivered / generated'),(axes[1],'base_return','Discounted environment return')):
        for seed,color in zip((11,22,33),('#3971a4','#d28332','#4d9666')):
            values=[next(x[metric] for x in rows if x['group']==g and x['seed']==seed) for g in groups]
            ax.plot(range(3),values,'o-',color=color,label='Training seed '+str(seed),alpha=.8)
        ax.set_xticks(range(3));ax.set_xticklabels(labels,rotation=12);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Development: 20 episodes/policy; 3 training seeds; 3 evaluation scenes')
    fig.tight_layout(rect=[0,0,1,.94]);folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(str(folder/'pm_task_return_comparison.png'),dpi=160)
    fig.savefig(str(folder/'pm_task_return_comparison.svg'))
    plt.close(fig)
    with (folder/'by_training_seed.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=('group','seed','delivery_rate','base_return'))
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--output')
    args=p.parse_args();root=Path(args.root).resolve()
    report=audit(root)
    if report['stage']=='pilot':plot(report,root)
    output=Path(args.output) if args.output else root/'acceptance.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('engineering_passed','audited_tasks','development_gate','formal_efficacy_proven')},indent=2))
    print('REPORT '+str(output))


if __name__=='__main__':main()
