"""Audit paired convergence curves; incomplete results never imply completion or non-convergence."""
import argparse,csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tg_convergence import *
def analyze(root,partial=False,destination=None):
    root=Path(root);m=read(root/'manifest.json');spec=read(root/'protocol.json')
    require(digest(root/'protocol.json')==m['protocol_sha256'],'Protocol changed')
    if not partial:
        require(all(j['state']=='completed' for j in m['jobs']),'Cannot audit incomplete experiment')
        for name,h in m['source']['files_sha256'].items():require(digest(root/'source_snapshot'/name)==h,'Archived source changed')
        for p,h in m['reference_manifests'].items():require(digest(p)==h,'Reference changed')
    models=[];train_rows=[];eval_rows=[];diagnostics=[];initial={}
    from buoy_network.task_reward_audit import audit_reward_csv
    for j in m['jobs']:
        path=Path(j['work_dir'])/'convergence/progress.json'
        if not path.exists():continue
        state=read(path);points=[]
        zero=next((p for p in state['points'] if p['episode']==0),None)
        if zero is not None:initial.setdefault(str(j['seed']),{})[j['group']]=zero['parent_fingerprints']
        for p in state['points']:
            require(p['complete'] and p['parent_fingerprints_unchanged'],'Measurement mutated training')
            require(p['training_steps']==p['episode']*spec['horizon'] and p['actor_optimizer_steps']==p['critic_optimizer_steps']==2*p['episode'],'Budget changed')
            require(p['torch_threads']==spec['torch_threads'],'CPU threads changed')
            if not partial:
                require(digest(p['checkpoint'])==p['checkpoint_sha256'],'Checkpoint changed')
                for ev in p['evaluations']:
                    actual=audit_frozen(ev['output'],p['checkpoint'],spec)
                    require(actual['policy_report_sha256']==ev['policy_report_sha256'] and actual['physical_sha256']==ev['physical_sha256']
                        and abs(actual['delivery']-ev['delivery'])<1e-12 and abs(actual['base_return']-ev['base_return'])<1e-12,'Frozen evaluation mismatch')
            delivery=statistics.mean(ev['delivery'] for ev in p['evaluations'])
            value=statistics.mean(ev['base_return'] for ev in p['evaluations'])
            require(len(p['evaluations'])==len(spec['scenes']) and sorted(x['scene'] for x in p['evaluations'])==sorted(spec['scenes']),'Missing scenes')
            point={'episode':p['episode'],'training_steps':p['training_steps'],'training_seconds':p['training_seconds'],
                   'delivery':delivery,'base_return':value}
            points.append(point)
            eval_rows.append(dict(point,group=j['group'],seed=j['seed']))
        physical=Path(j['output'])/'logs/TG_physical_metrics.jsonl'
        if physical.exists():
            rows=[json.loads(x) for x in physical.read_text(encoding='utf-8').splitlines()]
            values=[x['delivery_rate'] for x in rows];smooth=trailing_mean(values,spec['train_window'])
            for k,(x,avg) in enumerate(zip(rows,smooth)):
                train_rows.append({'group':j['group'],'seed':j['seed'],'episode':k+1,'training_steps':(k+1)*80,
                    'delivery':x['delivery_rate'],'moving_mean':avg,'base_return':physical_return(x,{'delivery_weight':1.,'delay_weight':1.,'unfinished_weight':1.}),
                    'loops':x['loop_transfers']})
        else:rows=[]
        if state['points']:
            latest=Path(state['points'][-1]['checkpoint']).parent.parent/'logs/TG_policy_report.json'
            for u in read(latest)['training_updates']:diagnostics.append(dict(u,group=j['group'],seed=j['seed'],episode=u['update']))
        complete=state['status']=='completed'
        if not partial:
            r=read(Path(j['output'])/'logs/TG_policy_report.json')
            require(complete and len(rows)==spec['episodes'] and [x['episode'] for x in points]==spec['checkpoints'],'Missing training/checkpoints')
            require(r['resources']['actor_trainable']==21956 and r['resources']['critic_trainable']==18615
                and r['resources']['device']=='cpu' and r['pm_action_credit']==dict(mode=j['mode'],coefficient=.1,temperature=.2),'Capacity/objective mismatch')
            cfg=read(Path(j['output'])/'logs/TG_MAPPO_environment.json')['config']
            require(cfg['num_episodes']==spec['episodes'] and cfg['num_env_steps']==spec['episodes']*80 and cfg['seed']==cfg['scenario_seed']==j['seed']
                and cfg['ppo_epoch']==1 and cfg['num_mini_batch']==2 and not cfg['use_linear_lr_decay'] and not cfg['use_wm_actor'],'Learning settings changed')
            require(r['profile']['collect_calls']==spec['episodes']*80 and len(r['training_updates'])==spec['episodes'],'Training budget mismatch')
            for n,u in enumerate(r['training_updates'],1):
                require(u['actor_optimizer_steps']==u['critic_head_optimizer_steps']==2*n and u['actor_lr']==.0005 and u['critic_lr']==.0003
                    and u['critic_head_in_optimizer'],'Optimizer mismatch')
                require(all(math.isfinite(v) for v in u.values() if isinstance(v,(int,float))),'Nonfinite training')
            import torch
            native=torch.load(r['checkpoint'],map_location='cpu')
            final=torch.load(state['points'][-1]['checkpoint'],map_location='cpu')
            require(nested_digest(native)==nested_digest(final),'Final evaluation checkpoint differs from native final policy')
            audit_reward_csv(Path(j['output'])/'logs',r['reward_contract'])
            require(any(p['episode']==j['reference_episodes'] and p['reference_exact'] for p in state['points']),'Missing reference prefix proof')
            if r.get('pm') is not None:
                require(r['pm']['frozen'] and r['pm']['checkpoint_sha256']==spec['frozen_pm_sha256'] and r['pm']['value_mode']=='base_residual','PM changed')
                require(len(r['pm_action_cache_audits'])==spec['episodes'] and all(x['unchanged'] for x in r['pm_action_cache_audits']),'Targets changed')
                from analyze_tg_pm_action_credit import value_cache_audit
                import numpy as np
                value_cache_audit(Path(j['output']),'base_residual',spec['episodes'])
                for ep in range(spec['episodes']):
                    with np.load(str(Path(j['output']).parent/'value_audit'/('episode_'+str(ep)+'.npz'))) as cache:
                        target=cache['pm_action_targets'];mask=cache['pm_available_actions']
                        require(np.allclose(target.sum(-1),1,atol=1e-5) and np.all(target[mask==0]==0),'Invalid/illegal cached action targets')
                        require(hashlib.sha256(target.tobytes()).hexdigest()==r['pm_action_cache_audits'][ep]['sha256'],'Cached targets differ from actual PPO')
        tail=points[-spec['tail_points']:] if len(points)>=spec['tail_points'] else []
        plateau=bool(complete and tail and max(x['delivery'] for x in tail)-min(x['delivery'] for x in tail)<=spec['tail_range']
                     and statistics.mean(x['delivery'] for x in tail)>=.8)
        models.append({'group':j['group'],'seed':j['seed'],'state':state['status'],'trained_episodes':state['trained_episodes'],
                       'points':points,'observed_auc':normalized_auc(points),'auc_horizon_episodes':points[-1]['episode'] if points else 0,
                       'thresholds':{str(t):threshold_time(points,t,spec['consecutive_points'],spec['interval'],complete) for t in spec['thresholds']},
                       'high_delivery_tail_plateau_observed':plateau if complete and tail else None,
                       'final_delivery':points[-1]['delivery'] if complete else None,'final_base_return':points[-1]['base_return'] if complete else None})
    aggregate={}
    for group in spec['groups']:
        selected=[x for x in models if x['group']==group];final=[x for x in selected if x['state']=='completed']
        curve=[]
        for n in spec['checkpoints']:
            pts=[p for x in selected for p in x['points'] if p['episode']==n]
            if pts:curve.append(dict(episode=n,**summarize([p['delivery'] for p in pts])))
        aggregate[group]={'completed_models':len(final),'planned_models':len(spec['seeds']),'curve':curve,
                          'final_delivery':summarize([x['final_delivery'] for x in final]),
                          'final_base_return':summarize([x['final_base_return'] for x in final]),
                          'auc':summarize([x['observed_auc'] for x in final]),
                          'confirmed_80':[{ 'seed':x['seed'],**x['thresholds']['0.8']} for x in selected]}
    initial_checks={}
    for seed in spec['seeds']:
        values=initial.get(str(seed),{});complete_initial=len(values)==len(spec['groups'])
        equal=(len({json.dumps(v,sort_keys=True) for v in values.values()})==1) if complete_initial else None
        initial_checks[str(seed)]={'groups':len(values),'identical':equal}
        if not partial:require(equal is True,'Initial parameters/optimizer/RNG differ across groups')
    if not partial:require(len(models)==len(m['jobs']),'Missing models')
    report={'status':'partial' if partial else 'completed','engineering_passed':None if partial else True,
            'source_manifest_status':m['status'],'protocol':spec,'models':models,'aggregate':aggregate,'initial_state_checks':initial_checks,
            'completed_models':sum(x['state']=='completed' for x in models),'planned_models':len(m['jobs']),
            'scope':'Known training seeds'+str(spec['seeds'])+'; finite-budget descriptive convergence comparison; fresh3041-3045 remain sealed.',
            'metric_definitions':{'time_to_target':'First observed checkpoint and first two consecutive qualifying checkpoints; resolution is the protocol checkpoint interval.',
              'tail_plateau':'Last five frozen-evaluation points: range<=5 percentage points and mean>=80%; not asymptotic convergence.',
              'wall_time':'Parent process elapsed excludes checkpoint IO, isolated evaluation subprocesses and live plot subprocesses; includes other parent bookkeeping.',
              'auc':'Trapezoid frozen greedy delivery integral / observed episode horizon; only completed-budget horizons compared in group summaries.',
              'uncertainty':'Sample SD across training seeds; scenes are averaged within each seed, not extra independent seeds.'},
            'offline_pm_train_validation_slots':2400,'manifest_sha256':digest(root/'manifest.json'),'analyzer_sha256':digest(Path(__file__))}
    out=Path(destination) if destination is not None else root/('live' if partial else 'analysis');out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('training_curves.csv',train_rows),('evaluation_curves.csv',eval_rows)]:
        if rows:
            with (out/filename).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if diagnostics:
        columns=['group','seed','episode']+sorted(set(k for u in diagnostics for k in u)-{'group','seed','episode'})
        with (out/'optimization_diagnostics.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(diagnostics)
    plot(report,train_rows,out)
    report['figure_sha256']=digest(out/'convergence.png')
    atomic(out/'report.json',report)
    if not partial:
        if destination is None:atomic(root/'report.json',report);markdown(report,root)
        else:markdown(report,out)
    return report

def plot(report,training,out):
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={'TG':'#286f9a','Immediate':'#9a7833','PM':'#2c8b60'}
    for group,agg in report['aggregate'].items():
        curve=agg['curve'];color=colors[group]
        if curve:
            xs=[p['episode'] for p in curve];ys=np.array([p['mean']*100 for p in curve]);sd=np.array([(p['sample_sd'] or 0)*100 for p in curve])
            axes[0,0].plot(xs,ys,'-o',ms=3,color=color,label=group);axes[0,0].fill_between(xs,ys-sd,ys+sd,color=color,alpha=.12)
        selected=[x for x in report['models'] if x['group']==group]
        for model in selected:
            pts=model['points']
            axes[0,1].plot([x['training_seconds']/60 for x in pts],[100*x['delivery'] for x in pts],color=color,alpha=.5,marker='o',ms=2,
                           label=group if model is selected[0] else None)
            axes[1,1].plot([x['episode'] for x in pts],[x['base_return'] for x in pts],color=color,alpha=.5,marker='o',ms=2,
                           label=group if model is selected[0] else None)
        for seed in report['protocol']['seeds']:
            rows=[x for x in training if x['group']==group and x['seed']==seed and x['moving_mean'] is not None]
            if rows:axes[1,0].plot([x['episode'] for x in rows],[x['moving_mean']*100 for x in rows],color=color,alpha=.5,marker='o',ms=2,
                                  label=group if seed==report['protocol']['seeds'][0] else None)
    titles=['Frozen greedy delivery: mean +/- training-seed SD','Frozen delivery vs parent training time','Stochastic training: trailing{} delivery per seed'.format(report['protocol']['train_window']),'Frozen original task return per seed']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
        handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=8)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_ylim(0,105);ax.axhline(80,color='#666',ls='--',lw=.7)
    for ax in (axes[0,0],axes[1,0],axes[1,1]):ax.set_xlim(0,report['protocol']['episodes']);ax.set_xlabel('Completed training episodes (80 joint slots each)')
    axes[0,1].set_xlabel('Training minutes; evaluations/checkpoint IO excluded')
    fig.suptitle('{} study: {} / {} models completed; group and seed counts in report'.format(report['status'].upper(),report['completed_models'],report['planned_models']))
    fig.savefig(str(out/'convergence.png'),dpi=150);fig.savefig(str(out/'convergence.svg'));plt.close(fig)
def markdown(r,root):
    lines=['# TG / immediate / PM convergence comparison','', 'Status: completed. This is a known-seed{}-episode comparison, not a fresh-seed generalization test.'.format(r['protocol']['episodes']),'',
           '|Group|Final frozen delivery mean|Sample SD|Frozen delivery AUC|','|---|---:|---:|---:|']
    for g,a in r['aggregate'].items():
        lines.append('|{}|{:.2f}%|{:.2f} pp|{:.4f}|'.format(g,a['final_delivery']['mean']*100,(a['final_delivery']['sample_sd'] or 0)*100,a['auc']['mean']))
    lines+=['','|Group|Seed|First observed80% episode|Confirmed80% episode|Training minutes at confirmation|Tail plateau|','|---|---:|---:|---:|---:|---|']
    for model in r['models']:
        target=model['thresholds']['0.8'];confirmed=target['sustained']
        lines.append('|{}|{}|{}|{}|{}|{}|'.format(model['group'],model['seed'],
            target['first_observed_episode'] if target['first_observed_episode'] is not None else 'not observed',
            confirmed['confirmed_episode'] if confirmed else 'not confirmed',
            '{:.2f}'.format(confirmed['training_seconds']/60) if confirmed else '-',model['high_delivery_tail_plateau_observed']))
    lines+=['','Detailed50%/80%/90% threshold times and first sustained block starts are in report.json. Non-reached thresholds remain censored; they are never replaced with zero.',
            'Training exploration curves and frozen greedy curves are distinct metrics. Reaching80% at two checkpoints is an observed criterion, not proof of mathematical convergence.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--partial',action='store_true');p.add_argument('--output-dir');a=p.parse_args()
    r=analyze(Path(a.root).resolve(),a.partial,a.output_dir);print('{} {}/{} models'.format(r['status'],r['completed_models'],r['planned_models']),flush=True)
if __name__=='__main__':main()
