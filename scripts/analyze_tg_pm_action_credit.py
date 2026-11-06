"""Independent audit of original TG + frozen PM action-target experiments."""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from buoy_network.ablation_suite import read,atomic,digest,require
from buoy_network.task_reward_audit import audit_reward_csv
from analyze_tg_pm_strength import cache_audit

def audit(root):
    import numpy as np
    m=read(root/'manifest.json')
    require(m['status']=='completed','Experiment not completed')
    require(all(j['state']=='completed' and j['exit_code']==0 for j in m['jobs']),'Incomplete/failed task')
    for p,h in m['reference_manifests'].items():require(digest(p)==h,'Historical manifest changed')
    for p,h in m['source']['files_sha256'].items():require(digest(root/'source_snapshot'/p)==h,'Runtime source changed')
    artifacts=[];gae=[];wall={};slots={'training':0,'evaluation':0}
    for j in m['jobs']:
        out=Path(j['output']);r=read(out/'logs/TG_policy_report.json')
        require(digest(out/'logs/TG_policy_report.json')==j['report_sha256']
            and digest(out/'logs/TG_physical_metrics.jsonl')==j['physical_sha256'],'Job fingerprint mismatch')
        require(r['pm_action_credit']==dict(mode=j['mode'],coefficient=float(j['coefficient']),temperature=.2),'Wrong teacher objective')
        require(r['profile']['collect_calls']==j['episodes']*80 and r['resources']['actor_trainable']==21956
            and r['resources']['critic_trainable']==18615,'Unfair budget/capacity')
        audit_reward_csv(out/'logs',r['reward_contract'])
        slots[j['kind']]+=j['episodes']*80
        if j['kind']=='training':
            require(digest(r['checkpoint'])==r['checkpoint_sha256'] and len(r['training_updates'])==j['episodes'],'Training artifact invalid')
            for i,u in enumerate(r['training_updates'],1):
                require(u['actor_optimizer_steps']==u['critic_head_optimizer_steps']==2*i
                    and u['actor_lr']==.0005 and u['critic_lr']==.0003,'PPO/LR budget changed')
            if j['mode']!='disabled':
                require(len(r['pm_action_cache_audits'])==j['episodes'] and all(x['unchanged'] for x in r['pm_action_cache_audits']),'Teacher cache changed')
                for ep in range(j['episodes']):
                    with np.load(str(out.parent/'value_audit'/('episode_'+str(ep)+'.npz'))) as data:
                        target=data['pm_action_targets'];mask=data['pm_available_actions']
                        require(np.allclose(target.sum(-1),1.,atol=1e-5) and np.all(target[mask==0]==0),'Illegal teacher distribution')
                        require(digest_bytes(target.tobytes())==r['pm_action_cache_audits'][ep]['sha256'],'Stored teacher differs from actual PPO cache')
                if j['coefficient']>0:require(all(u['pm_action_target_loss']>0 for u in r['training_updates']),'Teacher loss not trained')
            if r.get('pm') is not None:gae.append({'tag':j['tag'],'rollouts':cache_audit(out,.1,j['episodes'])})
            if j['episodes']==20:
                key=('holdout' if j['tag'].startswith('holdout') else 'development')+'_'+j['mode']
                wall.setdefault(key,[]).append({'seed':j['seed'],'wall_seconds':(datetime.fromisoformat(j['finished'])-datetime.fromisoformat(j['started'])).total_seconds(),
                    'collect_seconds':r['profile']['collect_seconds'],'ppo_seconds':r['profile']['ppo_seconds']})
        else:
            require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged']
                    and digest(r['source'])==r['checkpoint_sha256'],'Evaluation is not frozen')
        artifacts.append({'tag':j['tag'],'report_sha256':j['report_sha256'],'physical_sha256':j['physical_sha256']})
    def summarize(prefix,mode,seeds):
        rows=[]
        for seed in seeds:
            jobs=[j for j in m['jobs'] if j['tag'].startswith(prefix) and j['mode']==mode and j['kind']=='evaluation' and j['seed']==seed]
            require(len(jobs)==3,'Wrong scene count')
            rows.append({'seed':seed,'delivery':statistics.mean(j['delivered']/j['generated'] for j in jobs),
                'base_return':statistics.mean(j['base_return'] for j in jobs),
                'delivered_each_scene':[j['delivered'] for j in jobs]})
        return {'per_seed':rows,'mean_delivery':statistics.mean(x['delivery'] for x in rows),
            'sample_SD':statistics.stdev(x['delivery'] for x in rows),
            'mean_base_return':statistics.mean(x['base_return'] for x in rows)}
    dev={g:summarize('dev_',g,(11,22,33)) for g in ('zero_model','learned')}
    old=read(ROOT/'results/tg_pm_task_return/pilot_20261105_140555_016580/manifest.json')
    low=read(ROOT/'results/tg_pm_strength/20261105_154758_858806/manifest.json')
    refs={'A':{},'old_PM':{}}
    for name,jobs in (('A',[j for j in old['jobs'] if j['group']=='A']),('old_PM',low['jobs'])):
        for seed in (11,22,33):
            selected=[j for j in jobs if j['seed']==seed and j['kind']=='evaluation']
            refs[name][str(seed)]=statistics.mean(j['delivered']/j['generated'] for j in selected)
    d=m['development'];require(d['A']==refs['A'] and d['old_PM']==refs['old_PM'],'Historical comparison changed')
    for mode in dev:
        require(all(abs(row['delivery']-d[mode][str(row['seed'])]['delivery'])<1e-12 for row in dev[mode]['per_seed']),'Development aggregate differs')
    values={str(r['seed']):r['delivery'] for r in dev['learned']['per_seed']}
    passed=dev['learned']['mean_delivery']>max(dev['zero_model']['mean_delivery'],statistics.mean(refs['old_PM'].values()))
    passed=passed and dev['learned']['mean_delivery']>=statistics.mean(refs['A'].values())+.05
    passed=passed and all(values[s]>=max(refs['A'][s],refs['old_PM'][s]) for s in values) and sum(values[s]>refs['old_PM'][s] for s in values)>=2
    require(passed==d['passed'],'Development gate mismatch')
    held=None
    if passed:
        seeds=(1041,1042,1043,1044,1045)
        held={g:summarize('holdout_',g,seeds) for g in ('disabled','zero_model','learned')}
        vals={g:{r['seed']:r['delivery'] for r in held[g]['per_seed']} for g in held}
        accepted=all(vals['learned'][s]>vals['disabled'][s] and vals['learned'][s]>=vals['zero_model'][s] for s in seeds)
        accepted=accepted and held['learned']['mean_delivery']>held['zero_model']['mean_delivery'] and held['learned']['mean_delivery']>=held['disabled']['mean_delivery']+.05
        require(accepted==m['holdout']['passed'],'Holdout gate mismatch')
    else:require(m['holdout']['status']=='not_started_sealed','Holdout was used after development failure')
    return {'engineering_passed':True,'development':dev,'historical_delivery':refs,'development_gate_passed':passed,
       'holdout':held,'holdout_gate_passed':m['holdout'].get('passed',False),'root':str(root),
       'manifest_sha256':digest(root/'manifest.json'),'artifacts':artifacts,'cache_audits':gae,
       'new_environment_slots':slots,'timings':wall,'formal_long_budget_efficacy_proven':False,
       'limitations':['Fixed 20-rollout budget; no long-budget, adversarial or topology-size generalization proof.',
          'Packet lookahead holds topology/other queues; teacher auxiliary objective is biased, not exact COMA or a policy-invariance theorem.']}
def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()

def plot(r,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    groups=['A','old_PM','zero_model','learned'];labels=['TG','Old PM','Immediate only','PM action teacher']
    for seed in (11,22,33):
        y=[r['historical_delivery'][g][str(seed)] for g in groups[:2]]
        y += [next(x['delivery'] for x in r['development'][g]['per_seed'] if x['seed']==seed) for g in groups[2:]]
        axes[0].plot(range(4),[100*x for x in y],marker='o',label='Seed '+str(seed))
    axes[0].set_xticks(range(4));axes[0].set_xticklabels(labels,rotation=15);axes[0].set_title('Known development seeds');axes[0].set_ylabel('Delivery rate (%)');axes[0].legend()
    if r['holdout']:
        groups2=['disabled','zero_model','learned']
        for seed in (1041,1042,1043,1044,1045):
            y=[100*next(x['delivery'] for x in r['holdout'][g]['per_seed'] if x['seed']==seed) for g in groups2]
            axes[1].plot(range(3),y,marker='o',label=str(seed))
        axes[1].set_xticks(range(3));axes[1].set_xticklabels(['TG','Immediate only','PM action teacher'],rotation=15);axes[1].legend(fontsize=8)
    else:axes[1].text(.5,.5,'Holdout kept sealed',ha='center',va='center',transform=axes[1].transAxes)
    axes[1].set_title('Predeclared new training seeds')
    for ax in axes:ax.grid(alpha=.2);ax.set_ylim(0,105)
    fig.tight_layout();folder=root/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(str(folder/('pm_action_credit.'+ext)),dpi=160)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args()
    root=Path(a.root).resolve();r=audit(root);plot(r,root)
    r['auditor_sha256']=digest(Path(__file__));r['figures']={str(p.relative_to(root)):digest(p) for p in (root/'figures').iterdir()}
    atomic(root/'acceptance.json',r)
    print('Engineering PASS; development {}; holdout {}'.format(r['development_gate_passed'],r['holdout_gate_passed']))
    print('REPORT '+str(root/'acceptance.json'))
if __name__=='__main__':main()
