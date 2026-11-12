"""Known failure-scene follow-up; reuses offline202 and the frozen trace observer."""
import argparse,csv,json,os,statistics,subprocess,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tg_convergence import read,atomic,digest
from buoy_network.ablation_suite import snapshot,suite_lock
PYTHON=r'D:\anaconda3\envs\buoy_gpu\python.exe'
GENERALIZATION=ROOT/'results/thesis_generalization/20261112_145410_224809'


def prepare(base):
    base=Path(base).resolve();bm=read(base/'manifest.json');offline=base/'offline/report.json'
    if read(offline)['status']!='completed' or not bm['test202_evaluated']:raise ValueError('Offline test must already be complete; never rerun it here')
    gm=read(GENERALIZATION/'manifest.json')
    if gm['status']!='completed' or digest(GENERALIZATION/'report.json')!=gm['report_sha256']:raise ValueError('Scenario study changed')
    if digest(ROOT/'scripts/trace_frozen_components.py')!=bm['source']['files_sha256']['scripts\\trace_frozen_components.py']:raise ValueError('Trace observer must stay unchanged')
    root=ROOT/'results/component_failure_extension'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    m=dict(status='prepared',root=str(root),source=snapshot(root),base=str(base),jobs=[],rows=[],
        references={str(offline):digest(offline),str(GENERALIZATION/'report.json'):digest(GENERALIZATION/'report.json')})
    protocol=dict(scene=9402,seeds=[11,22,33],conditions=['loss_high','walk_fast'],factual_cases=12,interventions=6,
        environment_steps=2880,training_updates=0,device='cuda',simulation='cpu',offline_test_reused=True,
        selection='Known scene9402 selected after reading prior generalization results: all three high-loss PM_WM models trail PM. Diagnostic, not blind test.',
        expected_finish_cn=['2026-11-12T17:20:00+08:00','2026-11-12T17:50:00+08:00'])
    atomic(root/'protocol.json',protocol);m['protocol_sha256']=digest(root/'protocol.json')
    for condition in protocol['conditions']:
        for group,mode in [('PM','factual'),('PM_WM','factual'),('PM_WM','no_wm_residual')]:
            for seed in protocol['seeds']:
                old=next(j for j in gm['jobs'] if j['phase']!='preflight' and j['algorithm']==group and j['seed']==seed and j['condition']==condition and j['scene']==9402)
                previous=read(old['spec']);checkpoint=previous['checkpoint'];tag='{}_{}_{}_{}'.format(group,seed,condition,mode);out=root/'cases'/tag
                spec=root/'specs'/(tag+'.json')
                atomic(spec,dict(algorithm=group,seed=seed,condition=condition,mode=mode,scene=9402,checkpoint=checkpoint,
                    checkpoint_sha256=digest(checkpoint),job=previous['job'],settings=previous['settings'],
                    reference=str(Path(old['work'])/'evaluation/moves_5'),output=str(out)))
                for p in [checkpoint,previous['job'],str(Path(old['work'])/'result.json')]:m['references'][p]=digest(p)
                m['jobs'].append(dict(tag=tag,algorithm=group,seed=seed,condition=condition,mode=mode,scene=9402,spec=str(spec),spec_sha256=digest(spec),output=str(out),state='pending',
                    command=[PYTHON,'-s','-u',str(root/'source_snapshot/scripts/trace_frozen_components.py'),str(spec)]))
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True)
    return root


def summarize(rows):
    if len(rows)!=18:raise ValueError('Expected18 cases')
    keys={(r['condition'],r['algorithm'],r['mode'],r['seed'],r['scene']) for r in rows}
    expected={(c,g,m,s,9402) for c in ['loss_high','walk_fast'] for g,m in [('PM','factual'),('PM_WM','factual'),('PM_WM','no_wm_residual')] for s in [11,22,33]}
    if keys!=expected:raise ValueError('Unexpected/missing/duplicate failure cells')
    if any(r['optimizer_updates']!=0 or not r['weights_unchanged'] or r['status']!='completed' for r in rows):raise ValueError('Evaluation freeze failed')
    if not all(r['frozen_replay_exact'] for r in rows if r['mode']=='factual'):raise ValueError('Factual replay mismatch')
    cells=[]
    for c in ['loss_high','walk_fast']:
        for g,m in [('PM','factual'),('PM_WM','factual'),('PM_WM','no_wm_residual')]:
            rs=sorted([r for r in rows if (r['condition'],r['algorithm'],r['mode'])==(c,g,m)],key=lambda r:r['seed']);values=[r['delivery'] for r in rs]
            nodes={}
            for r in rs:
                for n,v in r['node_groups'].items():nodes[n]=nodes.get(n,0)+v['wait']
            cells.append(dict(condition=c,algorithm=g,mode=m,scene=9402,mean=statistics.mean(values),sample_sd=statistics.stdev(values),
                seeds=[r['seed'] for r in rs],delivery_by_seed=values,wait_decisions=sum(r['wait_decisions'] for r in rs),
                waits_outside_teacher_set=sum(r['waits_outside_teacher_set'] for r in rs),loop_transfers=sum(r['loop_transfers'] for r in rs),
                wm_changes_action=sum(r['wm_changes_action'] for r in rs),wm_introduces_wait=sum(r['wm_introduces_wait'] for r in rs),
                wait_nodes=sorted(nodes.items(),key=lambda x:(-x[1],int(x[0])))))
    return cells


def analyze(root,m):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    cells=summarize(m['rows'])
    fields=['algorithm','seed','condition','mode','scene','delivery','wait_decisions','waits_outside_teacher_set','wm_changes_action','wm_introduces_wait','loop_transfers','frozen_replay_exact','optimizer_updates']
    with (root/'case_metrics.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(m['rows'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,c in zip(axes,['loss_high','walk_fast']):
        rows=[r for r in cells if r['condition']==c]
        ax.bar(range(3),[r['mean']*100 for r in rows],yerr=[r['sample_sd']*100 for r in rows],capsize=4,color=['#3b9160','#2878a5','#aaa'])
        ax.set_xticks(range(3));ax.set_xticklabels(['PM','PM_WM','PM_WM residual off'],rotation=12);ax.set_ylim(0,110);ax.set_ylabel('Delivery %; mean +/- sample SD');ax.set_title('Known scene9402: '+c)
    for ext in ['png','svg','pdf']:fig.savefig(root/('failure_scene9402.'+ext),dpi=160)
    plt.close(fig)
    result=dict(status='completed',engineering_passed=True,case_aggregate=cells,training_updates=0,
        factual_replays_exact=True,offline_test_reused=True,offline_report=str(Path(m['base'])/'offline/report.json'),
        scope='Known worst high-loss scene9402 diagnostic, not additional blind generalization or retrained ablation',
        all_thesis_experiments_completed=False)
    atomic(root/'report.json',result)
    lines=['# Known failure-scene9402 follow-up','','Training updates0; reuses prior offline test202 without rereading/refitting it.',
        'Selected after prior generalization showed every high-loss PM_WM seed below PM in9402. This is diagnostic selection, not a blind test.','',
        '| Condition | Policy / mode | Delivery mean | Sample SD | WAIT | WAIT outside approximate teacher set |',
        '|---|---|---:|---:|---:|---:|']
    for r in cells:lines.append('| {} | {} / {} | {:.2f}% | {:.2f}pp | {} | {} |'.format(r['condition'],r['algorithm'],r['mode'],r['mean']*100,r['sample_sd']*100,r['wait_decisions'],r['waits_outside_teacher_set']))
    lines+=['','Residual-off changes only inference, not Actor training history. Teacher scores are approximations, not optimal-action labels.',
        'wm_changes_action and wm_introduces_wait compare full-vs-base argmax at visited states; in residual-off these are hypothetical differences, not executed changes.',
        'Node/packet details and online WM errors remain in each case report and decisions.jsonl. Broader sensitivity/capacity/compute experiments are still pending.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(root):
    root=Path(root).resolve()
    with suite_lock(root):
        m=read(root/'manifest.json')
        if m['status']!='prepared':raise ValueError('Refuse duplicate/failed retry')
        if read(Path(m['base'])/'manifest.json')['status']!='completed':raise ValueError('First diagnostic queue must finish before extension')
        if digest(root/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
        for p,h in m['references'].items():
            if digest(p)!=h:raise ValueError('Reference changed '+p)
        for p,h in m['source']['files_sha256'].items():
            if digest(root/'source_snapshot'/p)!=h:raise ValueError('Source changed')
        m.update(status='running',pid=os.getpid(),started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat());atomic(root/'manifest.json',m)
        try:
            for j in m['jobs']:
                if digest(j['spec'])!=j['spec_sha256']:raise ValueError('Spec changed')
                j['state']='running';atomic(root/'manifest.json',m);print('START '+j['tag'],flush=True)
                log=root/'logs'/(j['tag']+'.log');log.parent.mkdir(exist_ok=True)
                with log.open('xb') as f:p=subprocess.run(j['command'],stdout=f,stderr=subprocess.STDOUT,cwd=str(root/'source_snapshot'),timeout=1800)
                if p.returncode:raise ValueError('Task failed '+j['tag']+' '+str(log))
                r=read(Path(j['output'])/'report.json')
                if r['status']!='completed' or r['optimizer_updates']!=0 or not r['weights_unchanged']:raise ValueError('Freeze failed')
                m['rows'].append(r);j.update(state='completed',report_sha256=digest(Path(j['output'])/'report.json'));atomic(root/'manifest.json',m);print('DONE '+j['tag'],flush=True)
            analyze(root,m)
            for p,h in m['references'].items():
                if digest(p)!=h:raise ValueError('Reference changed '+p)
            m.update(status='completed',finished=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),report_sha256=digest(root/'report.json'));atomic(root/'manifest.json',m)
        except Exception as e:
            if 'j' in locals() and j['state']=='running':j['state']='failed'
            m.update(status='failed',error=repr(e));atomic(root/'manifest.json',m);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-for');g.add_argument('--run');a=p.parse_args()
    if a.prepare_for:prepare(a.prepare_for)
    else:run(a.run)
