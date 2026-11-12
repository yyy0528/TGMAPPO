"""Frozen unseen-seed and one-factor scenario evaluation, not model tuning."""
import argparse,copy,csv,json,statistics,subprocess,sys,os,time
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tg_convergence import read,atomic,digest
from buoy_network.ablation_suite import snapshot,suite_lock
from buoy_network.scenario_evaluation import PROFILES
PYTHON=r'D:\anaconda3\envs\buoy_gpu\python.exe'
SOURCE=ROOT/'results/thesis_extra_baselines/20261112_104503_699371'
GROUPS=['IQL','VDN','QMIX','COMA','IPPO','MAPPO','TG','Immediate','WM','PM','PM_WM']
ROBUST=['TG','WM','PM','PM_WM','IPPO']
SCENES=[9401,9402,9403]


def original_progress(model):
    if model.get('progress'):
        path=Path(model['progress'])
        if digest(path)!=model['progress_sha256']:raise ValueError('Changed reference progress')
    else:
        m=read(Path(model['source'])/'manifest.json')
        job=next(j for j in m['jobs'] if j['group']==model['algorithm'] and j['seed']==model['seed'])
        path=Path(job['work_dir'])/'convergence/progress.json'
        if digest(path)!=job['progress_sha256']:raise ValueError('Changed original progress')
    progress=read(path)
    if progress['status']!='completed' or progress['trained_episodes']!=300:raise ValueError('Require fixed300 model')
    return path,progress


def evaluation_path(point,scene):
    row=next(r for r in point['evaluations'] if r['scene']==scene)
    if row.get('output'):return Path(row['output'])
    cp=Path(point['checkpoint']).parent.parent
    for path in [cp/('scene_'+str(scene))/'moves_5',cp/('scene_'+str(scene))/'runs/case/moves_5']:
        if (path/'logs/TG_physical_metrics.jsonl').exists():return path
    raise ValueError('Cannot resolve archived evaluation path')


def catalog():
    m=read(SOURCE/'manifest.json')
    if m['status']!='main_completed' or digest(SOURCE/'report.json')!=m['report_sha256']:
        raise ValueError('Original comparison incomplete/changed')
    models=copy.deepcopy(m['reused'])
    for j in m['jobs']:
        if j['phase']=='main':
            progress=Path(j['work'])/'progress.json'
            models.append(dict(algorithm=j['algorithm'],seed=j['seed'],source=str(SOURCE),
                output=str(Path(j['work'])/'moves_5'),progress=str(progress),progress_sha256=j['progress_sha256']))
    if len(models)!=33 or {r['algorithm'] for r in models}!=set(GROUPS):raise ValueError('Require 11 groups, 3 policies each')
    result=[]
    for model in models:
        path,state=original_progress(model);point=state['points'][-1]
        if point['episode']!=300 or digest(point['checkpoint'])!=point['checkpoint_sha256']:raise ValueError('Selected policy changed')
        model.update(progress=str(path),progress_sha256=digest(path),checkpoint=point['checkpoint'],checkpoint_sha256=point['checkpoint_sha256'],
            identity_reference=str(evaluation_path(point,8001)),
            development_delivery=statistics.mean(r['delivery'] for r in point['evaluations']))
        result.append(model)
    return sorted(result,key=lambda r:(GROUPS.index(r['algorithm']),r['seed']))


def prepare():
    models=catalog();now=datetime.now(ZoneInfo('Asia/Shanghai'))
    root=ROOT/'results/thesis_generalization'/now.strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    source=snapshot(root);policies=root/'frozen_policies';policies.mkdir();aux=root/'frozen_aux';aux.mkdir()
    frozen={};copied={}
    def freeze(path):
        original=Path(path).resolve();sha=digest(original)
        if sha not in copied:
            target=aux/(sha[:16]+original.suffix);target.write_bytes(original.read_bytes());copied[sha]=str(target)
            frozen[str(target)]=sha
        return copied[sha]
    protocol=dict(schema='original_env_frozen_generalization_v1',created=now.isoformat(),groups=GROUPS,
        robustness_groups=ROBUST,training_seeds=[11,22,33],new_evaluation_scenes=SCENES,
        horizon=80,evaluation_episodes=2,training_steps=0,device='cuda',simulation='cpu',
        profiles=PROFILES,selection='fixed300, original coefficient0.1, chosen shared normalized WM; no repaired candidate',
        scope='Unseen channel/motion seeds in the SAME25-node grid; not unseen topology or independent retraining seeds',
        limitations=['Known weak/untuned baselines retained','Load is an initial burst, not continuous arrivals',
                     'Walk step changes inside original plus/minus0.1 displacement bound','Test results must not be used to tune and re-label as blind test'],
        budget=dict(identity_tasks=11,variant_smoke_tasks=5,unseen_tasks=99,robustness_tasks=225,
                    formal_tasks=324,formal_environment_steps=51840,preflight_environment_steps=2160),
        estimated_formal_hours=[1.5,3.0])
    atomic(root/'protocol.json',protocol)
    m=dict(status='prepared',phase='preflight',root=str(root),source=source,protocol_sha256=digest(root/'protocol.json'),
        created=now.isoformat(),references={str(SOURCE/'report.json'):digest(SOURCE/'report.json')},models=[],jobs=[],frozen_sha256=frozen,
        test_scenes_consumed=False,rows=[],all_thesis_experiments_completed=False)
    for model in models:
        g,seed=model['algorithm'],model['seed'];tag=g+'_'+str(seed)
        target=policies/(tag+'.pt');target.write_bytes(Path(model['checkpoint']).read_bytes());frozen[str(target)]=model['checkpoint_sha256']
        cfg=read(Path(model['output'])/'logs/TG_MAPPO_environment.json')['config']
        args=[]
        if cfg.get('use_wm_actor'):
            args+=['--use_wm_actor','true','--wm_actor_checkpoint',freeze(cfg['wm_actor_checkpoint']),
                '--wm_actor_input',cfg.get('wm_actor_input','raw')]
            if cfg.get('wm_actor_calibration'):args+=['--wm_actor_calibration',freeze(cfg['wm_actor_calibration'])]
        if cfg.get('use_pm_reward'):
            args+=['--use_pm_reward','true','--pm_reward_checkpoint',freeze(cfg['pm_reward_checkpoint']),
                '--pm_reward_alpha',str(cfg['pm_reward_alpha']),'--pm_potential_mode',cfg['pm_potential_mode'],
                '--pm_value_mode',cfg['pm_value_mode'],'--pm_action_credit',cfg.get('pm_action_credit','disabled'),
                '--pm_action_credit_coef',str(cfg.get('pm_action_credit_coef',.1)),
                '--pm_action_temperature',str(cfg.get('pm_action_temperature',.2))]
        job=dict(algorithm=g,seed=seed,episodes=300,points=[],scenes=[],eval_episodes=2,
                 work=str(root/'unused_training'/tag),extra_args=args)
        path=root/'model_specs'/(tag+'.json');path.parent.mkdir(exist_ok=True);atomic(path,job)
        model.update(frozen_checkpoint=str(target),job=str(path),job_sha256=digest(path))
        m['models'].append(model)
    def add(model,phase,condition,scene,episodes,reference=None):
        tag='{}_{}_{}_{}_{}'.format(phase,condition,model['algorithm'],model['seed'],scene);work=root/'jobs'/tag
        spec=dict(algorithm=model['algorithm'],seed=model['seed'],condition=condition,scene=scene,episodes=episodes,
            work=str(work),settings=PROFILES[condition],checkpoint=model['frozen_checkpoint'],checkpoint_sha256=model['checkpoint_sha256'],
            job=model['job'],reference=reference)
        path=root/'specs'/(tag+'.json');path.parent.mkdir(exist_ok=True);atomic(path,spec)
        m['jobs'].append(dict(tag=tag,phase=phase,algorithm=model['algorithm'],seed=model['seed'],condition=condition,scene=scene,
            state='pending',work=str(work),spec=str(path),spec_sha256=digest(path),
            command=[PYTHON,'-s','-u',str(root/'source_snapshot/scripts/thesis_generalization_worker.py'),str(path)]))
    for model in m['models']:
        if model['seed']==11:add(model,'preflight','baseline',8001,2,model['identity_reference'])
    model=next(r for r in m['models'] if r['algorithm']=='PM_WM' and r['seed']==11)
    for condition in PROFILES:
        if condition!='baseline':add(model,'preflight',condition,8001,1)
    for model in m['models']:
        for scene in SCENES:add(model,'unseen','baseline',scene,2)
    for condition in PROFILES:
        if condition=='baseline':continue
        for model in m['models']:
            if model['algorithm'] in ROBUST:
                for scene in SCENES:add(model,'robustness',condition,scene,2)
    m['frozen_sha256']=frozen
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True)
    return root


def analyze(root,m):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=m['rows']
    with (root/'per_scene.csv').open('x',newline='',encoding='utf-8') as stream:
        fields=['algorithm','seed','condition','scene','generated','delivered','delivery','unresolved_fraction',
            'success_hops','success_delay','loop_transfers','ack_failures','wall_seconds','optimizer_updates','weights_unchanged',
            'physical_sha256','checkpoint_sha256','identity_replay_exact']
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    aggregates=[];per_model=[]
    for condition in PROFILES:
        groups=GROUPS if condition=='baseline' else ROBUST
        for group in groups:
            values=[]
            for seed in [11,22,33]:
                cases=[r for r in rows if r['condition']==condition and r['algorithm']==group and r['seed']==seed]
                if len(cases)!=3 or {r['scene'] for r in cases}!=set(SCENES):raise ValueError('Incomplete crossed evaluation cells')
                generated=sum(r['generated'] for r in cases);delivered=sum(r['delivered'] for r in cases)
                value=delivered/generated;values.append(value)
                per_model.append(dict(condition=condition,algorithm=group,seed=seed,delivery=value,generated=generated,delivered=delivered,
                    success_delay=sum(r['success_delay']*r['delivered'] for r in cases if r['success_delay'] is not None)/delivered if delivered else None,
                    success_hops=sum(r['success_hops']*r['delivered'] for r in cases if r['success_hops'] is not None)/delivered if delivered else None))
            aggregates.append(dict(condition=condition,algorithm=group,mean=statistics.mean(values),sample_sd=statistics.stdev(values),training_policies=3))
    with (root/'per_model.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(per_model[0]));w.writeheader();w.writerows(per_model)
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    x=np.arange(len(GROUPS));dev=[];new=[];dev_sd=[];new_sd=[]
    for group in GROUPS:
        old=[r['development_delivery'] for r in m['models'] if r['algorithm']==group]
        current=next(r for r in aggregates if r['condition']=='baseline' and r['algorithm']==group)
        dev.append(statistics.mean(old)*100);dev_sd.append(statistics.stdev(old)*100);new.append(current['mean']*100);new_sd.append(current['sample_sd']*100)
    ax.bar(x-.18,dev,.36,yerr=dev_sd,label='Prior development scenes',capsize=2)
    ax.bar(x+.18,new,.36,yerr=new_sd,label='Unseen channel/motion scenes',capsize=2)
    ax.set_xticks(x);ax.set_xticklabels(GROUPS,rotation=35);ax.set_ylabel('Delivery (%)');ax.set_ylim(0,135);ax.legend()
    ax.set_title('Fixed300 policies; mean and SD over3 training seeds; no updates')
    for ext in ['png','svg','pdf']:fig.savefig(root/('unseen_comparison.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for ax,conditions,xs,title in zip(axes,[['loss_low','baseline','loss_high'],['walk_slow','baseline','walk_fast'],['baseline','load_double']],
        [[.5,1.,1.5],[.05,.1,.2],[1,2]],['Channel loss multiplier (capped1)','Random walk step (grid units/slot)','Initial packets per source']):
        for group in ROBUST:
            stats=[next(r for r in aggregates if r['condition']==c and r['algorithm']==group) for c in conditions]
            ax.errorbar(xs,[r['mean']*100 for r in stats],yerr=[r['sample_sd']*100 for r in stats],marker='o',capsize=2,label=group)
        ax.set_xlabel(title);ax.set_ylim(0,115);ax.grid(alpha=.2)
    axes[0].set_ylabel('Delivery (%)');axes[-1].legend(fontsize=8)
    for ext in ['png','svg','pdf']:fig.savefig(root/('robustness.'+ext),dpi=160)
    plt.close(fig)
    report=dict(status='completed',engineering_passed=True,training_updates=0,formal_tasks=len(rows),aggregate=aggregates,
        scope=read(root/'protocol.json')['scope'],all_thesis_experiments_completed=False,
        unseen_evaluation_scenes=SCENES,hyperparameters_changed=False,
        uncertainty='Sample SD over3 training-policy means; scene/episode repetitions are not extra training seeds')
    atomic(root/'report.json',report)
    lines=['# Frozen unseen-scene and robustness results','',report['scope'],'',
        '| Condition | Method | Delivery mean | Training-seed SD |','|---|---|---:|---:|']
    for r in aggregates:lines.append('| {} | {} | {:.2f}% | {:.2f} pp |'.format(r['condition'],r['algorithm'],r['mean']*100,r['sample_sd']*100))
    lines+=['','No training or hyperparameter changes. Other thesis stages are not marked complete.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(root):
    root=Path(root).resolve()
    with suite_lock(root):
        m=read(root/'manifest.json')
        if m['status']!='prepared':raise ValueError('Refuse duplicate or failed-run retry')
        if digest(root/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
        for name,h in m['source']['files_sha256'].items():
            if digest(root/'source_snapshot'/name)!=h:raise ValueError('Archived source changed')
        for path,h in {**m['frozen_sha256'],**m['references']}.items():
            if digest(path)!=h:raise ValueError('Frozen model/reference changed: '+path)
        for model in m['models']:
            if digest(model['job'])!=model['job_sha256']:raise ValueError('Model spec changed')
        m.update(status='running',pid=os.getpid(),started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        atomic(root/'manifest.json',m);print('RUNNING '+str(root),flush=True)
        try:
            for j in m['jobs']:
                if j['phase']!='preflight':
                    if any(x['state']!='completed' for x in m['jobs'] if x['phase']=='preflight'):raise ValueError('Preflight incomplete')
                    m['test_scenes_consumed']=True
                if digest(j['spec'])!=j['spec_sha256']:raise ValueError('Evaluation spec changed')
                m['phase']=j['phase'];j.update(state='running',started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
                atomic(root/'manifest.json',m);print('START '+j['tag'],flush=True)
                started=time.perf_counter()
                log=root/'logs'/(j['tag']+'.log');log.parent.mkdir(exist_ok=True)
                with log.open('xb') as stream:
                    result=subprocess.run(j['command'],stdout=stream,stderr=subprocess.STDOUT,cwd=str(root/'source_snapshot'),timeout=1200)
                if result.returncode:raise ValueError('Evaluation failed '+j['tag']+'; '+str(log))
                row=read(Path(j['work'])/'result.json')
                if row['status']!='completed' or row['optimizer_updates']!=0 or not row['weights_unchanged']:raise ValueError('Frozen evaluation audit failed')
                j.update(state='completed',finished=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),elapsed_seconds=time.perf_counter()-started,
                    result_sha256=digest(Path(j['work'])/'result.json'),delivery=row['delivery'])
                if j['phase']!='preflight':m['rows'].append(row)
                completed=[x['elapsed_seconds'] for x in m['jobs'] if x['state']=='completed'][-30:]
                remaining=sum(x['state']=='pending' for x in m['jobs'])
                m['remaining_tasks']=remaining;m['estimated_remaining_seconds']=remaining*statistics.mean(completed)
                m['estimated_finish_cn']=(datetime.now(ZoneInfo('Asia/Shanghai'))+timedelta(seconds=m['estimated_remaining_seconds'])).isoformat()
                atomic(root/'manifest.json',m);print('DONE '+j['tag']+' delivery '+str(row['delivery']),flush=True)
            if len(m['rows'])!=324:raise ValueError('Incomplete formal evaluation matrix')
            m['phase']='postprocessing';atomic(root/'manifest.json',m)
            analyze(root,m)
            for path,h in m['frozen_sha256'].items():
                if digest(path)!=h:raise ValueError('Frozen artifact changed during study')
            m.update(status='completed',finished=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),report_sha256=digest(root/'report.json'))
            atomic(root/'manifest.json',m);print('COMPLETED '+str(root),flush=True)
        except Exception as error:
            if 'j' in locals() and j['state']=='running':j['state']='failed'
            m.update(status='failed',error=repr(error));atomic(root/'manifest.json',m);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.run)
