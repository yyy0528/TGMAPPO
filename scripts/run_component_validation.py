"""Frozen component tests plus paired failure traces; no policy fitting."""
import argparse,csv,json,os,statistics,subprocess,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tg_convergence import read,atomic,digest
from buoy_network.ablation_suite import snapshot,suite_lock
PYTHON=r'D:\anaconda3\envs\buoy_gpu\python.exe'
GENERALIZATION=ROOT/'results/thesis_generalization/20261112_145410_224809'
DATASET=ROOT/'results/tg_wm_multiscene/20261102_235951_684782'


def prepare():
    previous=read(GENERALIZATION/'manifest.json')
    if previous['status']!='completed' or digest(GENERALIZATION/'report.json')!=previous['report_sha256']:
        raise ValueError('Completed unchanged scenario study required')
    root=ROOT/'results/component_validation'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    source=snapshot(root);frozen=root/'frozen';frozen.mkdir()
    original=dict(wm=ROOT/'results/tg_wm_multiscene_gate_selection/20261103_000951_565881/wm_best.pt',
                  pm=ROOT/'results/tg_pm_final_pretrain/20261103_142745_276042/pm_best.pt')
    components={}
    refs={str(GENERALIZATION/'report.json'):digest(GENERALIZATION/'report.json'),
          str(DATASET/'dataset_manifest.json'):digest(DATASET/'dataset_manifest.json')}
    dataset_manifest=read(DATASET/'dataset_manifest.json')
    if not dataset_manifest['complete']:raise ValueError('Dataset incomplete')
    for run in dataset_manifest['runs']:
        if run['split'] in ('validation','test'):
            folder=DATASET/run['data_path']
            for name in ['manifest.json','snapshots.jsonl','transitions.jsonl','packet_traces.jsonl']:
                refs[str(folder/name)]=digest(folder/name)
    for name,path in original.items():
        dst=frozen/(name+'.pt');dst.write_bytes(path.read_bytes());components[name]=str(dst);components[name+'_sha256']=digest(dst)
    protocol=dict(schema='frozen_component_validation_v1',created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        device='cuda',simulation='cpu',training_updates=0,offline_splits=['validation','test'],validation_scene=101,test_scene=202,
        factual_cases=18,intervention_cases=6,replay_environment_steps=3840,seeds=[11,22,33],scene=9401,
        conditions=['baseline','loss_high','walk_fast'],mode_selection='fixed300 original coefficient0.1, selected shared normalized WM',
        limitations=['Test202 is one scene, not many independent scenes','Teacher advice is approximate, not an optimal-action label',
            'Inference intervention is not retrained ablation or historical training-gradient attribution','No tuning/normalization fitting/model selection'],
        expected_finish_cn=['2026-11-12T17:20:00+08:00','2026-11-12T17:50:00+08:00'])
    atomic(root/'protocol.json',protocol)
    m=dict(status='prepared',root=str(root),source=source,references=refs,protocol_sha256=digest(root/'protocol.json'),
        frozen_sha256={components[k]:components[k+'_sha256'] for k in ('wm','pm')},jobs=[],rows=[],test202_evaluated=False,
        all_thesis_experiments_completed=False)
    specs=root/'specs';specs.mkdir()
    spec=specs/'offline.json';output=root/'offline';atomic(spec,dict(components,dataset=str(DATASET),output=str(output)))
    m['jobs'].append(dict(tag='offline_components',spec=str(spec),spec_sha256=digest(spec),state='pending',output=str(output),
        command=[PYTHON,'-s','-u',str(root/'source_snapshot/scripts/validate_frozen_components.py'),str(spec)]))
    def add(group,seed,condition,mode):
        old=next(j for j in previous['jobs'] if j['phase']!='preflight' and j['algorithm']==group and j['seed']==seed and j['condition']==condition and j['scene']==9401)
        original_spec=read(old['spec']);checkpoint=original_spec['checkpoint']
        tag='{}_{}_{}_{}'.format(group,seed,condition,mode);output=root/'cases'/tag
        s=dict(algorithm=group,seed=seed,condition=condition,mode=mode,scene=9401,
            checkpoint=checkpoint,checkpoint_sha256=digest(checkpoint),job=original_spec['job'],
            settings=original_spec['settings'],reference=str(Path(old['work'])/'evaluation/moves_5'),output=str(output))
        for path in [checkpoint,original_spec['job'],str(Path(old['work'])/'result.json')]:m['references'][path]=digest(path)
        path=specs/(tag+'.json');atomic(path,s)
        m['jobs'].append(dict(tag=tag,algorithm=group,seed=seed,condition=condition,mode=mode,spec=str(path),spec_sha256=digest(path),
            output=str(output),state='pending',command=[PYTHON,'-s','-u',str(root/'source_snapshot/scripts/trace_frozen_components.py'),str(path)]))
    for condition in protocol['conditions']:
        for group in ['PM','PM_WM']:
            for seed in [11,22,33]:add(group,seed,condition,'factual')
    for condition in ['loss_high','walk_fast']:
        for seed in [11,22,33]:add('PM_WM',seed,condition,'no_wm_residual')
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True);return root


def analyze(root,m):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    import numpy as np
    offline=read(root/'offline/report.json');rows=m['rows']
    fields=['algorithm','seed','condition','mode','delivery','active_decisions','wait_decisions','teacher_disagreement',
        'waits_outside_teacher_set','wm_changes_action','wm_introduces_wait','loop_transfers','unresolved','frozen_replay_exact','optimizer_updates']
    with (root/'case_metrics.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,metric,base,title in zip(axes,['ack_brier','queue_mse','motion_mse'],['ack_history_link_brier','queue_persistence_mse','motion_clamped_walk_mse'],['WM ACK Brier','WM queue delta MSE','Unused motion head MSE']):
        for x,split in enumerate(['validation','test']):
            values=offline['splits'][split]['wm'];ax.bar(x-.18,values[base],.36,color='#888888',label='Simple baseline' if x==0 else None)
            ax.bar(x+.18,values[metric],.36,color='#2878a5',label='Frozen WM' if x==0 else None)
        ax.set_xticks([0,1]);ax.set_xticklabels(['Scene101 val','Scene202 test']);ax.set_title(title);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    for ext in ['png','svg','pdf']:fig.savefig(root/('wm_validation.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,metric,base in zip(axes,['survival_nll','window_brier','task_return_mse'],['prior_survival_nll','prior_window_brier','prior_task_return_mse']):
        for x,split in enumerate(['validation','test']):
            v=offline['splits'][split]['pm'];ax.bar(x-.18,v[base],.36,color='#888888');ax.bar(x+.18,v[metric],.36,color='#3b9160')
        ax.set_xticks([0,1]);ax.set_xticklabels(['Scene101 val','Scene202 test']);ax.set_title(metric);ax.set_ylim(bottom=0)
    for ext in ['png','svg','pdf']:fig.savefig(root/('pm_validation.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    aggregate=[]
    for ax,condition in zip(axes,['baseline','loss_high','walk_fast']):
        groups=[('PM','factual'),('PM_WM','factual')]+([('PM_WM','no_wm_residual')] if condition!='baseline' else [])
        labels=[];means=[];sd=[]
        for group,mode in groups:
            cases=[r for r in rows if r['condition']==condition and r['algorithm']==group and r['mode']==mode]
            if len(cases)!=3:raise ValueError('Missing three-seed case cells')
            values=[r['delivery'] for r in cases];labels.append(group if mode=='factual' else 'PM_WM residual off')
            means.append(statistics.mean(values)*100);sd.append(statistics.stdev(values)*100)
            aggregate.append(dict(condition=condition,algorithm=group,mode=mode,mean=statistics.mean(values),sample_sd=statistics.stdev(values)))
        ax.bar(labels,means,yerr=sd,capsize=3);ax.tick_params(axis='x',rotation=25);ax.set_ylim(0,120);ax.set_title(condition+' / scene9401')
    axes[0].set_ylabel('Frozen delivery (%)')
    for ext in ['png','svg','pdf']:fig.savefig(root/('failure_comparison.'+ext),dpi=160)
    plt.close(fig)
    # Representative routes: fixed seed11, episode0, all unfinished packets plus two shortest delivered examples.
    selected=next(j for j in m['jobs'] if j.get('algorithm')=='PM_WM' and j.get('seed')==11 and j.get('condition')=='loss_high' and j.get('mode')=='factual')
    ledger=read(Path(selected['output'])/'packet_ledger.json')[0];positions=ledger['final_positions'];packets=ledger['packets']
    fig,ax=plt.subplots(figsize=(6,6),constrained_layout=True);ax.scatter([p[0] for p in positions],[p[1] for p in positions],c='#b0b0b0')
    for i,p in enumerate(positions):ax.text(p[0]+.03,p[1]+.03,str(i),fontsize=8)
    failed=[(k,p) for k,p in packets.items() if p['delay'] is None]
    delivered=sorted([(k,p) for k,p in packets.items() if p['delay'] is not None],key=lambda kv:(len(kv[1]['route']),int(kv[0])))[:2]
    for k,p in failed+delivered:
        route=p['route'];xs=[positions[n][0] for n in route];ys=[positions[n][1] for n in route]
        ax.plot(xs,ys,'-o',alpha=.7,color='#ba4035' if p['delay'] is None else '#33885b',label='packet'+str(k)+(' unfinished' if p['delay'] is None else ' delivered'))
        if len(route)==1:ax.scatter(xs,ys,marker='x',s=70,c='#ba4035')
    ax.set_title('Seed11 high-loss example; routes drawn on final node coordinates');ax.set_aspect('equal');ax.legend(fontsize=7)
    for ext in ['png','svg','pdf']:fig.savefig(root/('packet_routes.'+ext),dpi=160)
    plt.close(fig)
    result=dict(status='completed',engineering_passed=True,training_updates=0,offline=offline,case_aggregate=aggregate,
        factual_replays_exact=all(r['frozen_replay_exact'] for r in rows if r['mode']=='factual'),
        frozen_interventions=6,all_thesis_experiments_completed=False,
        scope='One reserved offline scene202 and known scenario9401 mechanism checks; no tuning or stable-gain claim')
    atomic(root/'report.json',result)
    lines=['# Frozen WM/PM validation and failure analysis','','No training, fitting, checkpoint selection or parameter changes.','',
        '## Offline metrics','', '| Split | WM ACK Brier / history | Queue MSE / zero | PM NLL / prior | PM task-return MSE / prior |', '|---|---|---|---|---|']
    for split in ['validation','test']:
        w=offline['splits'][split]['wm'];p=offline['splits'][split]['pm']
        lines.append('| {} | {:.6f} / {:.6f} | {:.6f} / {:.6f} | {:.6f} / {:.6f} | {:.6f} / {:.6f} |'.format(split,w['ack_brier'],w['ack_history_link_brier'],w['queue_mse'],w['queue_persistence_mse'],p['survival_nll'],p['prior_survival_nll'],p['task_return_mse'],p['prior_task_return_mse']))
    lines+=['','## Frozen routing cases','','| Condition | Policy / mode | Delivery mean | Sample SD |','|---|---|---:|---:|']
    for r in aggregate:lines.append('| {} | {} / {} | {:.2f}% | {:.2f} pp |'.format(r['condition'],r['algorithm'],r['mode'],r['mean']*100,r['sample_sd']*100))
    lines+=['','Teacher advice is an approximate one-step score, not ground-truth optimality. Residual-off is an inference intervention, not a retrained ablation.','The motion head is not used by the Actor. Prefix repetitions are correlated; offline test202 is only one scenario.','Other sensitivity, capacity controls, new training seeds and full compute-cost studies remain incomplete.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(root):
    root=Path(root).resolve()
    with suite_lock(root):
        m=read(root/'manifest.json')
        if m['status']!='prepared':raise ValueError('Refuse duplicate or failed-run retry')
        if digest(root/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
        for name,h in m['source']['files_sha256'].items():
            if digest(root/'source_snapshot'/name)!=h:raise ValueError('Source changed')
        for path,h in {**m['references'],**m['frozen_sha256']}.items():
            if digest(path)!=h:raise ValueError('Data/model/reference changed: '+path)
        m.update(status='running',pid=os.getpid(),started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat());atomic(root/'manifest.json',m)
        try:
            for j in m['jobs']:
                if digest(j['spec'])!=j['spec_sha256']:raise ValueError('Job spec changed')
                j['state']='running';atomic(root/'manifest.json',m);print('START '+j['tag'],flush=True)
                log=root/'logs'/(j['tag']+'.log');log.parent.mkdir(exist_ok=True)
                with log.open('xb') as f:p=subprocess.run(j['command'],stdout=f,stderr=subprocess.STDOUT,cwd=str(root/'source_snapshot'),timeout=1800)
                if p.returncode:raise ValueError('Task failed '+j['tag']+'; '+str(log))
                report=read(Path(j['output'])/'report.json')
                if report['status']!='completed' or report['training_updates' if j['tag']=='offline_components' else 'optimizer_updates']!=0 or not report['weights_unchanged']:
                    raise ValueError('Frozen component/task audit failed')
                if j['tag']=='offline_components':m['test202_evaluated']=True
                else:m['rows'].append(report)
                j.update(state='completed',report_sha256=digest(Path(j['output'])/'report.json'));atomic(root/'manifest.json',m);print('DONE '+j['tag'],flush=True)
            if len(m['rows'])!=24:raise ValueError('Missing paired routing cases')
            analyze(root,m)
            for path,h in m['frozen_sha256'].items():
                if digest(path)!=h:raise ValueError('Frozen component changed')
            m.update(status='completed',finished=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),report_sha256=digest(root/'report.json'));atomic(root/'manifest.json',m)
        except Exception as e:
            if 'j' in locals() and j['state']=='running':j['state']='failed'
            m.update(status='failed',error=repr(e));atomic(root/'manifest.json',m);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.run)
