"""Sequential, gated thesis experiments; never silently retry completed training.

First: all original-env smoke jobs. Then: baseline300 and the aligned WM+PM
ablation. Existing TG/Immediate/PM/selected-WM300 artifacts are verified/reused.
Later scientific stages are explicitly pending, not advertised as completed.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from buoy_network.ablation_suite import snapshot, suite_lock
from tg_convergence import atomic, read, digest

PYTHON = r'D:\anaconda3\envs\buoy_gpu\python.exe'
BASE = ROOT/'results/tg_five_gpu300/20261108_140256_956180'
SELECTED = ROOT/'results/tg_wm_input_probe/20261109_122413_853474'
BASELINES = ['IPPO','QMIX','MAPPO','IQL','VDN','COMA']
SEEDS = [11,22,33]
SCENES = [8001,8002,8003]


def extra_args(group, frozen):
    result = []
    if group in ('WM','PM_WM'):
        result += ['--use_wm_actor','true','--wm_actor_checkpoint',str(frozen/'wm.pt'),
                   '--wm_actor_input','residual_scaled','--wm_actor_calibration',str(frozen/'calibration.json')]
    if group in ('PM','Immediate','PM_WM'):
        result += ['--use_pm_reward','true','--pm_reward_checkpoint',str(frozen/'pm.pt'),
                   '--pm_reward_alpha','.1','--pm_potential_mode','task_return','--pm_value_mode','base_residual',
                   '--pm_action_credit','zero_set' if group=='Immediate' else 'learned_set']
    return result


def verify_reference(root):
    manifest = read(root/'manifest.json')
    if manifest['status']!='completed' or digest(root/'report.json')!=manifest['report_sha256']:
        raise ValueError('Incomplete/changed reference '+str(root))
    return manifest,read(root/'report.json')


def compatibility_reference(manifest, group):
    base=Path(next(r['source'] for r in manifest['reused'] if r['algorithm']=='TG'))
    reference=Path(read(base/'protocol.json')['probe_reference'])
    if not reference.is_absolute():
        reference=base.parents[2]/reference
    return Path(next(j['output'] for j in read(reference/'manifest.json')['jobs'] if j['group']==group))


def verify_native_probe(job, manifest):
    import torch
    old=compatibility_reference(manifest,job['algorithm']);new=Path(job['work'])/'moves_5'
    a=torch.load(old/'checkpoints/tg_policy_latest.pt',map_location='cpu',weights_only=True)
    b=torch.load(new/'checkpoints/tg_policy_latest.pt',map_location='cpu',weights_only=True)
    if any(not torch.equal(a[key][name],b[key][name]) for key in ('actor','critic') for name in a[key]):
        raise ValueError('Original short compatibility weights changed '+job['algorithm'])
    if (old/'logs/TG_physical_metrics.jsonl').read_bytes()!=(new/'logs/TG_physical_metrics.jsonl').read_bytes():
        raise ValueError('Original short physical replay changed '+job['algorithm'])
    job['legacy_exact']=True


def prepare(recovery=None):
    base, report = verify_reference(BASE)
    selected, wmreport = verify_reference(SELECTED)
    active = read(ROOT/'configs/tg_wm_active.json')
    if active['selected_version']!='shared_residual_scaled_300' or Path(active['study_root']).name != SELECTED.name:
        raise ValueError('The user-selected WM version changed')
    root = ROOT/'results/thesis_experiments'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    source = snapshot(root)
    frozen = root/'frozen';frozen.mkdir()
    for name, original in [('wm.pt',SELECTED/'frozen/wm.pt'),('calibration.json',SELECTED/'frozen/calibration.json'),('pm.pt',BASE/'frozen_models/pm.pt')]:
        (frozen/name).write_bytes(original.read_bytes())
    protocol = dict(schema='original_env_thesis_v1',python=PYTHON,train_seeds=SEEDS,development_scenes=SCENES,
        episodes=300,horizon=80,interval=20,eval_episodes=2,device='cuda',simulation='cpu',
        known_development_seeds=True,greedy_evaluation=True,ppo_epoch=1,num_mini_batch=2,
        baseline_encoding='observed_graph_mlp_v1',q_replay_capacity=3000,q_batch=32,q_updates_per_episode=2,
        q_target_interval=100,hyperparameter_tuning_completed=False,
        note='New baseline implementations, not old baselines/main.py. Development comparison, not blind thesis confirmation.',
        later_stages=[dict(name='unseen_scenes_and_loss_mobility_load',state='pending',reason='After main comparison; scenario intervention interfaces require separate tests'),
                      dict(name='WM_PM_component_validation',state='pending',reason='After main comparison; audit selected-model datasets and censored labels'),
                      dict(name='sensitivity',state='pending',reason='After main comparison; pre-register ranges and equal budgets'),
                      dict(name='cost_and_failure_analysis',state='collecting',reason='Timing/physical metrics collected in all jobs; final analysis after completion')])
    atomic(root/'protocol.json',protocol)
    manifest = dict(status='prepared',root=str(root),source=source,created=datetime.now().isoformat(),
        protocol_sha256=digest(root/'protocol.json'),frozen_sha256={str(p):digest(p) for p in frozen.iterdir()},
        references={str(p/'report.json'):digest(p/'report.json') for p in (BASE,SELECTED)},jobs=[],reused=[],
        phase='preflight',later_stages=protocol['later_stages'])
    for phase, groups, seeds, episodes in [('preflight',BASELINES+['TG','Immediate','PM','WM','PM_WM'],[22],2),
                                          ('main',BASELINES+['PM_WM'],SEEDS,300)]:
        for group in groups:
            for seed in seeds:
                tag='{}_{}_{}'.format(phase,group,seed)
                work = root/'jobs'/tag
                spec=dict(algorithm=group,seed=seed,episodes=episodes,
                          points=[0,2] if phase=='preflight' else list(range(0,301,20)),
                          scenes=[8001] if phase=='preflight' else SCENES,
                          eval_episodes=1 if phase=='preflight' else 2,work=str(work),extra_args=extra_args(group,frozen))
                if group=='Greedy' and phase=='main':
                    # A rule has no training budget. Evaluate once at the nominal final point.
                    spec.update(episodes=1,points=[1])
                path=root/'specs'/(tag+'.json');path.parent.mkdir(exist_ok=True)
                atomic(path,spec)
                manifest['jobs'].append(dict(tag=tag,phase=phase,algorithm=group,seed=seed,state='pending',
                    spec=str(path),spec_sha256=digest(path),work=str(work),
                    command=[PYTHON,'-s','-u',str(root/'source_snapshot/scripts/thesis_experiment_worker.py'),'--job',str(path)]))
    for group in ('TG','Immediate','PM'):
        for seed in SEEDS:
            model=next(r for r in report['models'] if r['group']==group and r['seed']==seed)
            oldjob=next(j for j in base['jobs'] if j['group']==group and j['seed']==seed)
            manifest['reused'].append(dict(algorithm=group,seed=seed,points=model['points'],source=str(BASE),output=oldjob['output']))
    for job in selected['jobs']:
        if job['mode']=='residual_scaled':
            progress = Path(job['work'])/'convergence/progress.json'
            manifest['reused'].append(dict(algorithm='WM',seed=job['seed'],points=read(progress)['points'],
                source=str(SELECTED),output=str(Path(job['work'])/'train/runs/case/moves_5'),progress=str(progress),progress_sha256=digest(progress)))
    if recovery:
        oldroot=Path(recovery).resolve();old=read(oldroot/'manifest.json')
        if old['status']!='failed':raise ValueError('Recovery requires an explicitly failed study')
        # Carry only completed baseline probes with identical learning code.
        key='src\\buoy_network\\thesis_baselines.py'
        if old['source']['files_sha256'][key]!=source['files_sha256'][key]:
            raise ValueError('Baseline implementation changed; cannot carry smoke results')
        manifest['recovered_from']=str(oldroot)
        manifest['recovery_reason']=old.get('error','Explicit failed-study recovery; completed probes verified before reuse')
        for job in manifest['jobs']:
            previous=next(j for j in old['jobs'] if j['tag']==job['tag'])
            if job['phase']=='preflight' and job['algorithm'] in BASELINES+['TG','Immediate','PM'] and previous['state']=='completed':
                progress=Path(previous['work'])/'progress.json'
                if digest(progress)!=previous['progress_sha256'] or read(progress)['status']!='completed':
                    raise ValueError('Carried probe changed')
                if job['algorithm'] not in BASELINES:
                    verify_native_probe(previous,old)
                job.update(previous)
                job['carried_from']=str(oldroot)
    atomic(root/'manifest.json',manifest)
    print('OUTPUT '+str(root),flush=True)
    return root


def summarize(root, manifest, excluded_algorithms=('Greedy',)):
    """Per-seed scatter + mean/SD; never combine different WM configurations."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows, curves = [], []
    models = list(manifest['reused'])
    for job in manifest['jobs']:
        if job['phase']=='main' and job['state']=='completed':
            models.append(dict(algorithm=job['algorithm'],seed=job['seed'],points=read(Path(job['work'])/'progress.json')['points'],source=job['work']))
    for model in models:
        if model['algorithm'] in excluded_algorithms:
            continue
        for point in model['points']:
            value = point.get('delivery')
            if value is None:
                value = statistics.mean(x['delivery'] for x in point['evaluations'])
            curves.append(dict(algorithm=model['algorithm'],seed=model['seed'],episode=point['episode'],delivery=value))
        last=curves[-1]
        rows.append(dict(algorithm=model['algorithm'],seed=model['seed'],delivery=last['delivery'],source=model['source']))
    with (root/'per_seed.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (root/'learning_curves.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    groups = [g for g in ['Greedy','IQL','VDN','QMIX','COMA','IPPO','MAPPO','TG','Immediate','WM','PM','PM_WM'] if any(r['algorithm']==g for r in rows)]
    aggregate = []
    for group in groups:
        values=[r['delivery'] for r in rows if r['algorithm']==group]
        aggregate.append(dict(algorithm=group,n=len(values),mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None))
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    ax.bar(groups,[r['mean']*100 for r in aggregate],yerr=[(r['sample_sd'] or 0)*100 for r in aggregate],capsize=3)
    for i,group in enumerate(groups):
        values=[r['delivery']*100 for r in rows if r['algorithm']==group]
        ax.scatter([i]*len(values),values,c='black',s=15,zorder=4)
    ax.set_ylabel('Frozen delivery (%)');ax.set_ylim(0,125);ax.tick_params(axis='x',rotation=25)
    ax.set_title('Known-scene development comparison; mean and training-seed SD')
    for ext in ('png','svg','pdf'):fig.savefig(root/('comparison.'+ext),dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
    for group in groups:
        if group=='Greedy':continue
        points=sorted({r['episode'] for r in curves if r['algorithm']==group})
        ys=[];sds=[]
        for point in points:
            values=[r['delivery'] for r in curves if r['algorithm']==group and r['episode']==point]
            ys.append(statistics.mean(values)*100);sds.append(statistics.stdev(values)*100 if len(values)>1 else 0)
        ax.plot(np.asarray(points)*80,ys,label=group)
        ax.fill_between(np.asarray(points)*80,np.asarray(ys)-sds,np.asarray(ys)+sds,alpha=.08)
    ax.set_xlabel('Training joint environment steps');ax.set_ylabel('Frozen delivery (%)');ax.set_ylim(0,105);ax.legend(ncol=2)
    ax.set_title('Convergence at equal interaction budgets; fixed terminal checkpoints')
    for ext in ('png','svg','pdf'):fig.savefig(root/('convergence.'+ext),dpi=160)
    plt.close(fig)
    atomic(root/'report.json',dict(status='main_completed',engineering_passed=True,aggregate=aggregate,
        excluded_algorithms=list(excluded_algorithms),comparison_scope='Reported algorithms only; original full results retained',
        all_thesis_experiments_completed=False,hyperparameter_tuning_completed=False,later_stages=manifest['later_stages']))


def report_only(study):
    """Re-export a scoped comparison, preserving archived results and models."""
    study=Path(study).resolve();manifest=read(study/'manifest.json')
    if manifest['status']!='main_completed':
        raise ValueError('Require a completed main study')
    output=study/'reports'/('without_greedy_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    summarize(output,manifest)
    atomic(output/'provenance.json',dict(source_study=str(study),source_manifest_sha256=digest(study/'manifest.json'),
        source_report_sha256=digest(study/'report.json'),excluded_algorithms=['Greedy'],
        reason='User requested removing Greedy from the main comparison; raw results preserved',training_started=False))
    print('OUTPUT '+str(output))
    return output


def run(root, preflight_only=False):
    root=Path(root).resolve()
    with suite_lock(root):
        manifest=read(root/'manifest.json')
        if manifest['status'] not in ('prepared','preflight_passed'):
            raise ValueError('Refuse duplicate/failed job replay; inspect manifest first')
        if digest(root/'protocol.json')!=manifest['protocol_sha256']:
            raise ValueError('Protocol changed')
        for name,checksum in manifest['source']['files_sha256'].items():
            if digest(root/'source_snapshot'/name)!=checksum:raise ValueError('Source changed: '+name)
        for path,checksum in {**manifest['references'],**manifest['frozen_sha256']}.items():
            if digest(path)!=checksum:raise ValueError('Frozen input/reference changed: '+path)
        from audit_thesis_references import audit
        audit(root)
        manifest.update(status='running',pid=os.getpid());atomic(root/'manifest.json',manifest)
        try:
            for job in manifest['jobs']:
                if job['state']=='completed':continue
                if preflight_only and job['phase']=='main':break
                if job['phase']=='main' and any(j['state']!='completed' for j in manifest['jobs'] if j['phase']=='preflight'):
                    raise ValueError('Preflight is incomplete')
                if digest(job['spec'])!=job['spec_sha256']:raise ValueError('Job spec changed')
                work=Path(job['work']);work.mkdir(parents=True,exist_ok=True)
                job.update(state='running',started=datetime.now().isoformat())
                manifest['phase']=job['phase'];atomic(root/'manifest.json',manifest)
                print('START '+job['tag'],flush=True)
                with (work/'execution.log').open('xb') as stream:
                    result=subprocess.run(job['command'],stdout=stream,stderr=subprocess.STDOUT,cwd=str(root/'source_snapshot'),timeout=28800)
                if result.returncode:raise ValueError('Task failed '+job['tag']+'; '+str(work/'execution.log'))
                progress=read(work/'progress.json')
                if progress['status']!='completed':raise ValueError('Missing completed progress '+job['tag'])
                job.update(state='completed',finished=datetime.now().isoformat(),progress_sha256=digest(work/'progress.json'))
                # Exact original-TG short compatibility gate before reusing old models.
                if job['phase']=='preflight' and job['algorithm'] in ('TG','PM','Immediate'):
                    verify_native_probe(job,manifest)
                atomic(root/'manifest.json',manifest);print('DONE '+job['tag'],flush=True)
            if preflight_only:
                manifest['status']='preflight_passed'
            else:
                summarize(root,manifest)
                manifest.update(status='main_completed',report_sha256=digest(root/'report.json'),all_thesis_experiments_completed=False)
            atomic(root/'manifest.json',manifest)
        except Exception as error:
            if 'job' in locals() and job.get('state')=='running':job['state']='failed'
            manifest.update(status='failed',error=repr(error));atomic(root/'manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--run');group.add_argument('--report-only')
    parser.add_argument('--recover-from')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:report_only(args.report_only)
    elif args.prepare:prepare(args.recover_from)
    else:run(args.run,args.preflight_only)
