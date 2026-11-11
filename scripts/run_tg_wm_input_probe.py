"""Bounded three-seed WM input-fix validation against immutable raw-input controls."""
import argparse,json,subprocess,sys,hashlib,statistics
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tg_convergence import read,atomic,digest,setarg,physical_return
from buoy_network.ablation_suite import snapshot,suite_lock

def invoke(command,log,timeout=7200):
    with Path(log).open('xb') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if r.returncode:raise ValueError('Failed command; see '+str(log))

def audit(output,episodes,mode,calibration_sha=None,evaluation=False,fusion='shared'):
    import math
    output=Path(output);r=read(output/'logs/TG_policy_report.json')
    assert r['resources']['actor_trainable']==24165 and r['resources']['critic_trainable']==18615 and r['resources']['device']=='cuda'
    assert r['wm']['frozen'] and r['wm']['checkpoint_sha256']=='2919dff60d3128321170410e632e7bddd476a4f09c766b92108bfe9cb73daca9'
    assert r['wm_actor_input']==mode and r['profile']['collect_calls']==episodes*80
    assert r.get('wm_actor_fusion','shared')==fusion
    if calibration_sha:assert r['wm_actor_calibration']['sha256']==calibration_sha
    if evaluation:assert r['actor_unchanged'] and r['critic_unchanged'] and r['optimizer_updates']==0
    else:
        assert len(r['training_updates'])==episodes
        assert all(u['actor_optimizer_steps']==u['critic_head_optimizer_steps']==2*(i+1) for i,u in enumerate(r['training_updates']))
        assert all(math.isfinite(v) for u in r['training_updates'] for v in u.values() if isinstance(v,(int,float)))
    rows=[json.loads(x) for x in (output/'logs/TG_physical_metrics.jsonl').read_text().splitlines()]
    assert len(rows)==episodes
    from buoy_network.task_reward_audit import audit_reward_csv
    audit_reward_csv(output/'logs',r['reward_contract'])
    return dict(delivery=sum(x['delivered_packets'] for x in rows)/sum(x['generated_packets'] for x in rows),
                base_return=statistics.mean(physical_return(x,r['reward_contract']) for x in rows),report_sha256=digest(output/'logs/TG_policy_report.json'))

def verify_raw_compatibility(output, reference):
    # The controller may run inside a snapshot. Artifact paths must come from
    # the manifest rather than be resolved relative to this script's location.
    import torch
    output,reference=Path(output),Path(reference)
    assert output.is_absolute() and reference.is_absolute(), 'Compatibility paths must be absolute'
    x=torch.load(reference/'checkpoints/tg_policy_latest.pt',map_location='cpu',weights_only=True)
    y=torch.load(output/'checkpoints/tg_policy_latest.pt',map_location='cpu',weights_only=True)
    assert all(set(x[k])==set(y[k]) and all(torch.equal(x[k][n],y[k][n]) for n in x[k]) for k in ('actor','critic'))
    assert read(reference/'logs/TG_policy_report.json')['training_updates']==read(output/'logs/TG_policy_report.json')['training_updates']
    assert (reference/'logs/TG_physical_metrics.jsonl').read_bytes()==(output/'logs/TG_physical_metrics.jsonl').read_bytes()


def prepare(calibration,episodes,prefix_reference=None,comparison_reference=None,fusion='shared'):
    assert episodes in (20,100,300)
    assert fusion in ('shared','action_delta')
    base=ROOT/'results/tg_five_gpu300/20261108_140256_956180';assert read(base/'manifest.json')['status']=='completed'
    c=read(calibration);assert c['fit_split']=='train' and c['future_labels_used'] is False
    if episodes==300:
        assert prefix_reference and read(Path(prefix_reference)/'manifest.json')['status']=='completed','Completed three-seed20 probe required'
        assert digest(calibration)==read(Path(prefix_reference)/'manifest.json')['calibration_sha256'],'Calibration changed since short probe'
        assert read(Path(prefix_reference)/'manifest.json').get('fusion','shared')==fusion,'Fusion mode changed since short probe'
    root=ROOT/'results/tg_wm_input_probe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    frozen=root/'frozen';frozen.mkdir()
    wm=ROOT/'results/tg_wm_multiscene_gate_selection/20261103_000951_565881/wm_best.pt'
    (frozen/'wm.pt').write_bytes(wm.read_bytes());(frozen/'calibration.json').write_bytes(Path(calibration).read_bytes())
    m=dict(status='prepared',root=str(root),source=snapshot(root),episodes=episodes,seeds=[11,22,33],scenes=[8001,8002,8003],
        reference_root=str(base),raw_compat_reference=str(ROOT/'results/tg_five_gpu_probe/20261108_135245_396245/jobs/WM_22/training/runs/case/moves_5'),
        calibration_sha256=digest(frozen/'calibration.json'),wm_sha256=digest(wm),prefix_reference=prefix_reference,fusion=fusion,jobs=[])
    if comparison_reference:
        reference_path=Path(comparison_reference).resolve()
        reference_manifest=read(reference_path/'manifest.json')
        reference_report=read(reference_path/'report.json')
        assert reference_manifest['status']=='completed' and reference_report['episodes']==episodes
        assert digest(reference_path/'report.json')==reference_manifest['report_sha256']
        assert len(reference_report['rows'])==len(m['seeds'])
        assert {r['seed'] for r in reference_report['rows']}==set(m['seeds'])
        assert all(0<=r['delivery']<=1 for r in reference_report['rows'])
        m['comparison_reference']=dict(root=str(reference_path),report_sha256=reference_manifest['report_sha256'])
    python=r'D:\anaconda3\envs\buoy_gpu\python.exe'
    for seed in [22,11,22,33]:
        compat=len(m['jobs'])==0;tag='compat_raw22' if compat else 'scaled_'+str(seed);work=root/tag;work.mkdir()
        n=2 if compat else episodes;mode='raw' if compat else 'residual_scaled'
        cmd=[python,'-s','-u',str(root/'source_snapshot/scripts/tg_eval_worker.py'),'--algorithm','tg-mappo','--tg_device','cuda',
             '--use_wm_actor','true','--wm_actor_checkpoint',str(frozen/'wm.pt'),'--wm_actor_input',mode,'--tg_reward_version','task_v2',
             '--num_episodes',str(n),'--num_env_steps',str(n*80),'--episode_length','80','--ppo_epoch','1','--num_mini_batch','2',
             '--num_moving_buoys','5','--seed',str(seed),'--scenario_seed',str(seed),'--output_root',str(work/'train'),'--run-name','case']
        if not compat:cmd+=['--wm_actor_calibration',str(frozen/'calibration.json')]
        if not compat and fusion!='shared':cmd+=['--wm_actor_fusion',fusion]
        m['jobs'].append(dict(tag=tag,seed=seed,episodes=n,mode=mode,fusion='shared' if compat else fusion,command=cmd,state='pending',work=str(work)))
        if episodes==300 and not compat:
            previous=next(j for j in read(Path(prefix_reference)/'manifest.json')['jobs'] if j['seed']==seed and j['mode']=='residual_scaled')
            m['jobs'][-1]['prefix20_reference']=str(Path(previous['work'])/'train/runs/case/moves_5')
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True);return root

def run(root):
    root=Path(root).resolve()
    with suite_lock(root):
        m=read(root/'manifest.json');assert m['status']=='prepared'
        for name,h in m['source']['files_sha256'].items():assert digest(root/'source_snapshot'/name)==h
        m['status']='running';atomic(root/'manifest.json',m)
        try:
            for j in m['jobs']:
                j['state']='running';atomic(root/'manifest.json',m);print('START '+j['tag'],flush=True)
                work=Path(j['work']);command=j['command']
                if j['episodes']==300:
                    atomic(work/'job.json',j)
                    command=command[:3]+[str(root/'source_snapshot/scripts/tg_wm_input_worker.py'),'--job-spec',str(work/'job.json')]+command[4:]
                invoke(command,work/'train.log');out=work/'train/runs/case/moves_5'
                j['train_metrics']=audit(out,j['episodes'],j['mode'],m['calibration_sha256'] if j['mode']!='raw' else None,fusion=j.get('fusion','shared'))
                if j['tag']=='compat_raw22':
                    verify_raw_compatibility(out,m['raw_compat_reference'])
                    j['legacy_exact']=True
                else:
                    j['evaluations']=[]
                    if j['episodes']==300:
                        state=read(work/'convergence/progress.json')
                        assert state['status']=='completed' and state['prefix20_exact'] and len(state['points'])==16
                        j['evaluations']=state['points'][-1]['evaluations']
                    for scene in ([] if j['episodes']==300 else m['scenes']):
                        cmd=list(j['command'])
                        for key,value in [('--num_episodes',2),('--num_env_steps',160),('--seed',scene),('--scenario_seed',scene),('--output_root',work/('eval_'+str(scene))),('--tg_evaluate','true'),('--tg_policy_checkpoint',out/'checkpoints/tg_policy_latest.pt')]:setarg(cmd,key,value)
                        invoke(cmd,work/('eval_'+str(scene)+'.log'))
                        metrics=audit(work/('eval_'+str(scene))/'runs/case/moves_5',2,j['mode'],m['calibration_sha256'],True,fusion=j.get('fusion','shared'))
                        j['evaluations'].append(dict(scene=scene,**metrics))
                    j['mean_delivery']=statistics.mean(x['delivery'] for x in j['evaluations'])
                    j['mean_base_return']=statistics.mean(x['base_return'] for x in j['evaluations'])
                j['state']='completed';atomic(root/'manifest.json',m);print('DONE '+j['tag']+' delivery '+str(j.get('mean_delivery')),flush=True)
            reference=read(Path(m['reference_root'])/'report.json');rows=[]
            for j in m['jobs'][1:]:
                old=next(x for x in reference['models'] if x['group']=='WM' and x['seed']==j['seed']);baseline=next(x for x in reference['models'] if x['group']=='TG' and x['seed']==j['seed'])
                raw=next(x for x in old['points'] if x['episode']==m['episodes']);tg=next(x for x in baseline['points'] if x['episode']==m['episodes'])
                rows.append(dict(seed=j['seed'],delivery=j['mean_delivery'],raw_delivery=raw['delivery'],tg_delivery=tg['delivery'],base_return=j['mean_base_return'],raw_base_return=raw['base_return']))
            if m.get('comparison_reference'):
                comparison=Path(m['comparison_reference']['root'])/'report.json'
                assert digest(comparison)==m['comparison_reference']['report_sha256']
                previous=read(comparison)
                for row in rows:
                    row['reference_delivery']=next(r['delivery'] for r in previous['rows'] if r['seed']==row['seed'])
            report=dict(status='completed',episodes=m['episodes'],rows=rows,mean_delivery=statistics.mean(x['delivery'] for x in rows),mean_raw_delivery=statistics.mean(x['raw_delivery'] for x in rows),
                mean_tg_delivery=statistics.mean(x['tg_delivery'] for x in rows),scope='Known three-seed development comparison; controls are immutable same-budget checkpoints, raw2 replay is exact; no unseen-seed or long-budget claim beyond actual episodes')
            report['mean_delivery_improved']=report['mean_delivery']>report['mean_raw_delivery']
            report['no_seed_delivery_regression']=all(x['delivery']>=x['raw_delivery'] for x in rows)
            report['mean_above_tg']=report['mean_delivery']>report['mean_tg_delivery']
            report['no_seed_regression_vs_tg']=all(x['delivery']>=x['tg_delivery'] for x in rows)
            report['engineering_passed']=True
            report['unseen_seed_validation_completed']=False
            report['wm_actor_fusion']=m.get('fusion','shared')
            if m.get('comparison_reference'):
                report['comparison_reference']=m['comparison_reference']
                report['mean_reference_delivery']=statistics.mean(r['reference_delivery'] for r in rows)
                report['mean_above_reference']=report['mean_delivery']>report['mean_reference_delivery']
                report['no_seed_regression_vs_reference']=all(r['delivery']>=r['reference_delivery'] for r in rows)
            plot_comparison(root,m,reference,report)
            atomic(root/'report.json',report);m.update(status='completed',report_sha256=digest(root/'report.json'));atomic(root/'manifest.json',m);print(json.dumps(report),flush=True)
        except Exception as e:
            m.update(status='failed',error=repr(e));atomic(root/'manifest.json',m);raise

def plot_comparison(root,m,reference,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np,csv
    labels=['TG','WM raw','WM fixed'];keys=['tg_delivery','raw_delivery','delivery'];colors=['#286f9a','#8a62a6','#2c8b60']
    if m.get('comparison_reference'):
        labels=['TG','WM raw','WM previous','WM current'];keys=['tg_delivery','raw_delivery','reference_delivery','delivery'];colors=['#286f9a','#8a62a6','#9d7a43','#2c8b60']
    means=[statistics.mean(x[k] for x in report['rows']) for k in keys]
    std=[statistics.stdev(x[k] for x in report['rows']) for k in keys]
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True);ax.bar(range(len(keys)),np.array(means)*100,yerr=np.array(std)*100,capsize=5,color=colors)
    ax.set_xticks(range(len(keys)));ax.set_xticklabels(labels);ax.set_ylim(0,110);ax.set_ylabel('Frozen delivery (%)');ax.set_title('WM input fix: {} episodes, 3 known training seeds'.format(m['episodes']));ax.grid(axis='y',alpha=.2)
    for ext in ('png','svg','pdf'):fig.savefig(str(root/('comparison.'+ext)),dpi=160)
    plt.close(fig)
    with (root/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(report['rows'][0]));w.writeheader();w.writerows(report['rows'])
    if m['episodes']==300:
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        for group,color in [('TG','#286f9a'),('WM','#8a62a6')]:
            curve=reference['aggregate'][group]['curve'];x=[p['episode'] for p in curve];y=np.array([p['mean'] for p in curve]);sd=np.array([p['sample_sd'] or 0 for p in curve])
            ax.plot(x,y*100,'-o',label='WM raw' if group=='WM' else group,color=color);ax.fill_between(x,(y-sd)*100,(y+sd)*100,alpha=.12,color=color)
        models=[read(Path(j['work'])/'convergence/progress.json') for j in m['jobs'][1:]];curve=[]
        for n in range(0,301,20):
            values=[statistics.mean(x['delivery'] for x in next(p for p in s['points'] if p['episode']==n)['evaluations']) for s in models];curve.append((n,statistics.mean(values),statistics.stdev(values)))
        previous_curve=[]
        if m.get('comparison_reference'):
            previous_manifest=read(Path(m['comparison_reference']['root'])/'manifest.json')
            previous_models=[read(Path(j['work'])/'convergence/progress.json') for j in previous_manifest['jobs'] if j['mode']=='residual_scaled']
            assert len(previous_models)==3 and all(s['status']=='completed' for s in previous_models)
            for n in range(0,301,20):
                values=[statistics.mean(x['delivery'] for x in next(p for p in s['points'] if p['episode']==n)['evaluations']) for s in previous_models]
                previous_curve.append((n,statistics.mean(values),statistics.stdev(values)))
            px=[p[0] for p in previous_curve];py=np.array([p[1] for p in previous_curve]);ps=np.array([p[2] for p in previous_curve])
            ax.plot(px,py*100,'-o',color='#9d7a43',label='WM previous');ax.fill_between(px,(py-ps)*100,(py+ps)*100,color='#9d7a43',alpha=.12)
        with (root/'convergence.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['episode','fixed_mean','fixed_sample_sd','raw_mean','tg_mean']+(['previous_mean','previous_sample_sd'] if previous_curve else []))
            for n,mean,sd in curve:
                raw=next(p['mean'] for p in reference['aggregate']['WM']['curve'] if p['episode']==n)
                tg=next(p['mean'] for p in reference['aggregate']['TG']['curve'] if p['episode']==n)
                prior=next((p for p in previous_curve if p[0]==n),None)
                w.writerow([n,mean,sd,raw,tg]+([prior[1],prior[2]] if prior else []))
        x=[p[0] for p in curve];y=np.array([p[1] for p in curve]);sd=np.array([p[2] for p in curve]);ax.plot(x,y*100,'-o',color='#2c8b60',label='WM current');ax.fill_between(x,(y-sd)*100,(y+sd)*100,color='#2c8b60',alpha=.12)
        report['curve_auc']={'current_wm':float(np.trapz(y,x)/300)}
        for group,key in [('TG','tg'),('WM','raw_wm')]:
            points=reference['aggregate'][group]['curve'];report['curve_auc'][key]=float(np.trapz([p['mean'] for p in points],[p['episode'] for p in points])/300)
        if previous_curve:report['curve_auc']['previous_wm']=float(np.trapz([p[1] for p in previous_curve],[p[0] for p in previous_curve])/300)
        ax.set_ylim(0,105);ax.set_xlabel('Training episodes');ax.set_ylabel('Frozen delivery (%)');ax.grid(alpha=.2);ax.legend();ax.set_title('WM comparison at equal budgets; mean +/- training-seed SD')
        for ext in ('png','svg','pdf'):fig.savefig(str(root/('convergence.'+ext)),dpi=160)
        plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare');p.add_argument('--run');p.add_argument('--prefix-reference');p.add_argument('--comparison-reference');p.add_argument('--fusion',choices=('shared','action_delta'),default='shared');p.add_argument('--episodes',type=int,default=20);a=p.parse_args()
    if a.prepare:prepare(a.prepare,a.episodes,a.prefix_reference,a.comparison_reference,a.fusion)
    elif a.run:run(a.run)
    else:p.error('Use --prepare calibration.json or --run root')

if __name__=='__main__':main()
