"""Controlled diagnostic probes: faster Q target sync; stronger PM auxiliary.

No environment/reward/model-architecture changes. Defaults and archived reports
stay unchanged. All Q short probes retain the historical300 exploration clock.
"""
import argparse,copy,csv,json,subprocess,sys,statistics
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tg_convergence import read,atomic,digest
from buoy_network.ablation_suite import snapshot,suite_lock
from buoy_network.thesis_probe_checks import long_probe_allowed
PYTHON=r'D:\anaconda3\envs\buoy_gpu\python.exe'
BASE=ROOT/'results/thesis_experiments/20261112_011930_763352'
EXTRA=ROOT/'results/thesis_extra_baselines/20261112_104503_699371'


def prepare():
    root=ROOT/'results/thesis_repair_probe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    base,extra=read(BASE/'manifest.json'),read(EXTRA/'manifest.json')
    source=snapshot(root)
    manifest=dict(status='prepared',root=str(root),source=source,jobs=[],references={str(p/'report.json'):digest(p/'report.json') for p in (BASE,EXTRA)},
        protocol=dict(q_change='target sync100 to20 optimizer updates only; epsilon horizon fixed24000',
        pm_change='auxiliary coefficient0.1 to0.2 only',seeds=[11,22,33],scenes=[8001,8002,8003],
        short_episodes=20,long_scope='seed11 only if mean20>=.98 and each>=.95; not three-seed stability proof',device='cuda',simulation='cpu'),
        rows=[])
    def add(tag,oldjob,episodes,options=None,reference=None,long=False):
        job=read(oldjob['spec']);work=root/'jobs'/tag
        job.update(work=str(work),episodes=episodes,points=list(range(0,301,20)) if long else [episodes],
            scenes=[8001] if episodes==2 else [8001,8002,8003],eval_episodes=1 if episodes==2 else 2)
        if options:job['baseline_options']=options
        if job['algorithm']=='PM_WM':job['extra_args']+=['--pm_action_credit_coef','.2']
        if long:job['prefix_reference']=str(root/'jobs/PM_WM_11_short/moves_5')
        path=root/'specs'/(tag+'.json');path.parent.mkdir(exist_ok=True);atomic(path,job)
        manifest['jobs'].append(dict(tag=tag,algorithm=job['algorithm'],seed=job['seed'],episodes=episodes,work=str(work),
            spec=str(path),spec_sha256=digest(path),state='pending',reference=reference,
            command=[PYTHON,'-s','-u',str(root/'source_snapshot/scripts/thesis_experiment_worker.py'),'--job',str(path)]))
    for group,old in [('VDN',extra),('QMIX',base)]:
        j=next(j for j in old['jobs'] if j['phase']=='preflight' and j['algorithm']==group)
        add(group+'_compat',j,2,reference=str(Path(j['work'])/'moves_5'))
    for group,old in [('VDN',extra),('QMIX',base)]:
        for seed in [11,22,33]:
            j=next(j for j in old['jobs'] if j['phase']=='main' and j['algorithm']==group and j['seed']==seed)
            add(group+'_'+str(seed)+'_short',j,20,dict(q_target_interval=20,q_exploration_steps=24000))
    for seed in [11,22,33]:
        j=next(j for j in base['jobs'] if j['phase']=='main' and j['algorithm']=='PM_WM' and j['seed']==seed)
        add('PM_WM_'+str(seed)+'_short',j,20)
        if seed==11:longjob=j
    add('PM_WM_11_long',longjob,300,long=True)
    # References use the original e20 points, not short runs with a different epsilon schedule.
    for j in manifest['jobs']:
        if j['episodes']!=20:continue
        old=extra if j['algorithm']=='VDN' else base
        ref=next(x for x in old['jobs'] if x['phase']=='main' and x['algorithm']==j['algorithm'] and x['seed']==j['seed'])
        progress=Path(ref['work'])/'progress.json';point=next(x for x in read(progress)['points'] if x['episode']==20)
        j.update(reference20_delivery=statistics.mean(x['delivery'] for x in point['evaluations']),reference_progress=str(progress),reference_progress_sha256=digest(progress))
    atomic(root/'manifest.json',manifest);print('OUTPUT '+str(root),flush=True)
    return root


def run(root):
    root=Path(root).resolve()
    with suite_lock(root):
        m=read(root/'manifest.json')
        if m['status']!='prepared':raise ValueError('Refuse duplicate or failed-run retry')
        for p,h in m['source']['files_sha256'].items():
            if digest(root/'source_snapshot'/p)!=h:raise ValueError('Source changed '+p)
        for p,h in m['references'].items():
            if digest(p)!=h:raise ValueError('Reference changed '+p)
        m['status']='running';atomic(root/'manifest.json',m)
        try:
            for j in m['jobs']:
                if j['episodes']==300:
                    values=[r['delivery'] for r in m['rows'] if r['algorithm']=='PM_WM']
                    if not long_probe_allowed(values):
                        j.update(state='not_started_gate_failed',reason='Three-seed20 safety gate failed; no long training');continue
                if digest(j['spec'])!=j['spec_sha256']:raise ValueError('Job changed')
                j['state']='running';atomic(root/'manifest.json',m);print('START '+j['tag'],flush=True)
                work=Path(j['work']);work.mkdir(parents=True)
                with (work/'execution.log').open('xb') as f:r=subprocess.run(j['command'],stdout=f,stderr=subprocess.STDOUT,cwd=str(root/'source_snapshot'),timeout=14400)
                if r.returncode:raise ValueError('Job failed '+j['tag']+'; '+str(work/'execution.log'))
                state=read(work/'progress.json')
                if state['status']!='completed':raise ValueError('Worker incomplete')
                if j['reference']:
                    import torch
                    old=Path(j['reference']);a=torch.load(old/'checkpoints/baseline.pt',map_location='cpu',weights_only=True);b=torch.load(work/'moves_5/checkpoints/baseline.pt',map_location='cpu',weights_only=True)
                    if any(not torch.equal(a[k][n],b[k][n]) for k in ('actor','critic') for n in a[k]):raise ValueError('Default Q compatibility changed')
                    if (old/'logs/TG_physical_metrics.jsonl').read_bytes()!=(work/'moves_5/logs/TG_physical_metrics.jsonl').read_bytes():raise ValueError('Default physical prefix changed')
                    j['legacy_exact']=True
                point=state['points'][-1];delivery=statistics.mean(x['delivery'] for x in point['evaluations'])
                j.update(state='completed',progress_sha256=digest(work/'progress.json'),delivery=delivery)
                if j['episodes']==20:
                    if digest(j['reference_progress'])!=j['reference_progress_sha256']:raise ValueError('Control progress changed')
                    m['rows'].append(dict(algorithm=j['algorithm'],seed=j['seed'],episodes=20,delivery=delivery,control=j['reference20_delivery']))
                atomic(root/'manifest.json',m);print('DONE '+j['tag']+' delivery '+str(delivery),flush=True)
            with (root/'pilot.csv').open('x',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['algorithm','seed','episodes','delivery','control']);writer.writeheader();writer.writerows(m['rows'])
            import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
            for ax,group in zip(axes,['VDN','QMIX','PM_WM']):
                rows=[r for r in m['rows'] if r['algorithm']==group]
                ax.bar(['Control','Candidate'],[statistics.mean(r[k] for r in rows)*100 for k in ('control','delivery')]);ax.set_ylim(0,105);ax.set_title(group+' at20 episodes');ax.set_ylabel('Delivery (%)')
            for ext in ['png','pdf','svg']:fig.savefig(root/('pilot.'+ext),dpi=160)
            plt.close(fig)
            report=dict(status='completed',engineering_passed=True,rows=m['rows'],long_job=next(j for j in m['jobs'] if j['episodes']==300),
                stable_three_seed300_improvement_verified=False,scope='Development candidate checks; no archived result overwritten')
            atomic(root/'report.json',report);m.update(status='completed',report_sha256=digest(root/'report.json'));atomic(root/'manifest.json',m)
        except Exception as error:
            if 'j' in locals() and j['state']=='running':j['state']='failed'
            m.update(status='failed',error=repr(error));atomic(root/'manifest.json',m);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.run)
