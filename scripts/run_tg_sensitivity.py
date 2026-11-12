"""One-factor sensitivity pilot: archived controls, gated new20-episode runs."""
import argparse,copy,csv,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tg_convergence import read,atomic,digest,setarg
from buoy_network.ablation_suite import snapshot,suite_lock
PYTHON=r'D:\anaconda3\envs\buoy_gpu\python.exe'
SEEDS=[11,22,33];SCENES=[8001,8002,8003]
CENTER=dict(alpha=.1,aux=.1,wm_scale=1.)
VARIANTS={'center':CENTER,'alpha0':dict(CENTER,alpha=0.),'alpha02':dict(CENTER,alpha=.2),
          'aux0':dict(CENTER,aux=0.),'aux02':dict(CENTER,aux=.2),
          'wm0':dict(CENTER,wm_scale=0.),'wm2':dict(CENTER,wm_scale=2.)}
AXES={'alpha':['alpha0','center','alpha02'],'aux':['aux0','center','aux02'],'wm_scale':['wm0','center','wm2']}


def validate_resources(report):
    r=report['resources']
    if (report['learning_rate_schedule']['kind']!='fixed' or r['device'] not in ('cuda','cuda:0')
            or r['actor_trainable']!=24165 or r['critic_trainable']!=18615):
        raise ValueError('Wrong LR/device/capacity')


def assert_one_factor(name,values):
    if name not in VARIANTS or values!=VARIANTS[name]:raise ValueError('Unknown/changed sensitivity coordinates')
    changed=[k for k in CENTER if values[k]!=CENTER[k]]
    if len(changed)!=(0 if name=='center' else 1):raise ValueError('Not a one-factor configuration')


def summarize(rows):
    expected={(v,s) for v in VARIANTS for s in SEEDS}
    if len(rows)!=21 or {(r['variant'],r['seed']) for r in rows}!=expected:raise ValueError('Missing/duplicate sensitivity cells')
    result=[]
    for v,coords in VARIANTS.items():
        assert_one_factor(v,coords);group=[r for r in rows if r['variant']==v]
        for r in group:
            if r['coordinates']!=coords or r['episodes']!=20:raise ValueError('Mixed coordinates/training budget')
            if not math.isfinite(r['delivery']) or not 0<=r['delivery']<=1:raise ValueError('Invalid delivery')
        result.append(dict(variant=v,coordinates=coords,training_seeds=SEEDS,n=3,mean=statistics.mean(r['delivery'] for r in group),sample_sd=statistics.stdev(r['delivery'] for r in group),
            training_delivery_mean=statistics.mean(r['training_delivery_mean'] for r in group),
            reused=sum(r['reused'] for r in group),scenes=SCENES,scope='20-episode development sensitivity, not300 or unseen generalization'))
    return result


def analyze(root,m):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    cells=summarize(m['rows']);lookup={r['variant']:r for r in cells}
    fields=['variant','seed','episodes','delivery','success_delay','success_hops','eval_loops','training_delivery_mean','reused','checkpoint']
    with (root/'per_model.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(m['rows'])
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,(key,names) in zip(axes,AXES.items()):
        vals=[lookup[n] for n in names];x=[r['coordinates'][key] for r in vals]
        ax.errorbar(x,[r['mean']*100 for r in vals],yerr=[r['sample_sd']*100 for r in vals],marker='o',capsize=4)
        ax.set_xticks(x);ax.set_ylim(0,110);ax.set_xlabel(key);ax.set_title({'alpha':'PM shaping alpha','aux':'PM action auxiliary coefficient','wm_scale':'WM logit residual multiplier'}[key]);ax.set_ylabel('Frozen delivery %; mean +/- sample SD')
    for ext in ['png','svg','pdf']:fig.savefig(root/('sensitivity20.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,(key,names) in zip(axes,AXES.items()):
        for name in names:
            rs=[r for r in m['rows'] if r['variant']==name];ys=[statistics.mean(r['training_delivery'][i] for r in rs) for i in range(20)]
            ax.plot(range(1,21),ys,label='{}={}'.format(key,VARIANTS[name][key]))
        ax.set_ylim(0,1.05);ax.set_xlabel('Training episode');ax.set_ylabel('Training delivery (exploratory policy)');ax.legend(fontsize=8)
    for ext in ['png','svg','pdf']:fig.savefig(root/('training_curves20.'+ext),dpi=160)
    plt.close(fig)
    result=dict(status='completed',engineering_passed=True,cells=cells,models=21,new_models=15,reused_models=6,preflight_jobs=5,
        training_episodes_per_model=20,training_seeds=SEEDS,development_scenes=SCENES,blind_test_used=False,selected_new_default=False,
        stable300_improvement_verified=False,all_thesis_experiments_completed=False,
        scope='Single-factor short-budget development pilot; not full300 sensitivity or new training-seed generalization')
    atomic(root/'report.json',result)
    lines=['# TG+WM+PM sensitivity pilot (20 episodes)','','Seven unique settings, three training seeds each. Only one factor changes per setting.',
        'Archived center and auxiliary0.2 results reused after hash/config/budget/freeze audits. No test202 or scenes9401-9403 used.',
        'Center already has100% development delivery at20; ceilings cannot establish a best setting or300 stability.','',
        '| Setting | Alpha | Aux | WM scale | Frozen delivery mean | Sample SD | Training delivery mean | Reused models |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in cells:lines.append('| {} | {} | {} | {} | {:.2f}% | {:.2f}pp | {:.2f}% | {} |'.format(r['variant'],r['coordinates']['alpha'],r['coordinates']['aux'],r['coordinates']['wm_scale'],r['mean']*100,r['sample_sd']*100,r['training_delivery_mean']*100,r['reused']))
    lines+=['','Training curves use observed training episodes, not frozen-policy convergence AUC. Scene repetitions are not extra training seeds.',
        'WM scale0 keeps WM/PM modules and allocated parameters, but zeroes the learned WM residual throughout training and evaluation. It is not the earlier inference-only intervention.',
        'PM shaping alpha and action auxiliary coefficient are different mechanisms. No coordinate selected on a blind test or promoted to default.',
        'Capacity-matched controls, full compute costs, full-budget sensitivity and independent confirmation remain pending.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def pin(m,path):
    path=Path(path).resolve();m['references'][str(path)]=digest(path)


def prepare():
    oldroot=ROOT/'results/thesis_experiments/20261112_011930_763352';repairroot=ROOT/'results/thesis_repair_probe/20261112_131735_713721'
    old=read(oldroot/'manifest.json');repair=read(repairroot/'manifest.json')
    if old['status']!='main_completed' or repair['status']!='completed':raise ValueError('Archived references incomplete')
    for p,manifest in [(oldroot,old),(repairroot,repair)]:
        if digest(p/'report.json')!=manifest['report_sha256']:raise ValueError('Archived report changed')
    root=ROOT/'results/tg_sensitivity20'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    m=dict(status='prepared',root=str(root),source=snapshot(root),references={},jobs=[],reused=[],rows=[])
    for p in [oldroot/'manifest.json',oldroot/'report.json',repairroot/'manifest.json',repairroot/'report.json']:pin(m,p)
    frozen=root/'frozen';frozen.mkdir()
    for name in ['wm.pt','pm.pt','calibration.json']:
        src=oldroot/'frozen'/name;dst=frozen/name;dst.write_bytes(src.read_bytes());pin(m,dst);pin(m,src)
    protocol=dict(schema='tg_sensitivity20_v1',center=CENTER,variants=VARIANTS,axes=AXES,training_seeds=SEEDS,development_scenes=SCENES,
        train_episodes=20,horizon=80,eval_episodes=2,ppo_epoch=1,minibatches=2,device='cuda',simulation='cpu',
        total_effective_models=21,reused_models=6,new_models=15,preflight_jobs=5,
        new_training_steps=24800,new_frozen_evaluation_steps=7600,
        expected_finish_cn=['2026-11-12T19:50:00+08:00','2026-11-12T20:30:00+08:00'],
        monitoring='Current-chat live checks through completion/failure; no scheduled job automation',
        forbidden_selection_scenes=[202,9401,9402,9403],no_auto_promotion=True,
        limitations=['20 episodes only; not300 stability','Existing center endpoint100%, compare training curves and harms without claiming best value','Only known development scenes and training seeds'])
    atomic(root/'protocol.json',protocol);m['protocol_sha256']=digest(root/'protocol.json')
    def add(variant,seed,phase,template,reference=None):
        values=VARIANTS[variant];assert_one_factor(variant,values);tag='{}_{}_{}'.format(phase,variant,seed);work=root/'jobs'/tag
        spec=read(template['spec']);spec.update(algorithm='PM_WM',seed=seed,work=str(work),episodes=2 if phase=='preflight' else 20,
            points=[2] if phase=='preflight' else [20],scenes=[8001] if phase=='preflight' else SCENES,eval_episodes=1 if phase=='preflight' else 2)
        args=spec['extra_args']
        for k,v in [('--wm_actor_checkpoint',frozen/'wm.pt'),('--pm_reward_checkpoint',frozen/'pm.pt'),('--wm_actor_calibration',frozen/'calibration.json'),
            ('--pm_reward_alpha',values['alpha']),('--pm_action_credit_coef',values['aux']),('--wm_actor_residual_scale',values['wm_scale'])]:setarg(args,k,v)
        spec.pop('prefix_reference',None);path=root/'specs'/(tag+'.json');atomic(path,spec)
        m['jobs'].append(dict(tag=tag,phase=phase,variant=variant,seed=seed,coordinates=values,work=str(work),spec=str(path),spec_sha256=digest(path),state='pending',reference=reference,
            command=[PYTHON,'-s','-u',str(root/'source_snapshot/scripts/thesis_experiment_worker.py'),'--job',str(path)]))
    pref=next(j for j in old['jobs'] if j['phase']=='preflight' and j['algorithm']=='PM_WM')
    for variant in ['center','alpha0','aux0','wm0','wm2']:
        add(variant,22,'preflight',pref,reference=str(Path(pref['work'])/'moves_5') if variant=='center' else None)
    for variant in VARIANTS:
        for seed in SEEDS:
            template=next(j for j in old['jobs'] if j['phase']=='main' and j['algorithm']=='PM_WM' and j['seed']==seed)
            if variant in ['center','aux02']:
                ref=template if variant=='center' else next(j for j in repair['jobs'] if j['tag']=='PM_WM_{}_short'.format(seed))
                progress=read(Path(ref['work'])/'progress.json');point=next(p for p in progress['points'] if p['episode']==20)
                d=dict(variant=variant,seed=seed,coordinates=VARIANTS[variant],work=ref['work'],point=point,spec=ref['spec'],state='pending_audit')
                m['reused'].append(d)
                paths=[Path(ref['work'])/'progress.json',ref['spec'],point['checkpoint'],Path(point['checkpoint']).parent.parent/'logs/TG_policy_report.json',
                    Path(ref['work'])/'moves_5/logs/TG_MAPPO_environment.json',Path(ref['work'])/'moves_5/logs/TG_physical_metrics.jsonl']
                cp=Path(point['checkpoint']).parent.parent
                for scene in SCENES:
                    paths.extend([cp/('scene_'+str(scene))/'moves_5/logs'/n for n in ['TG_policy_report.json','TG_physical_metrics.jsonl','TG_MAPPO_environment.json','TG_reward_components.csv','TG_PM_rewards.csv']])
                for p in paths:pin(m,p)
            else:add(variant,seed,'pilot',template)
    if len(m['jobs'])!=20 or len(m['reused'])!=6:raise ValueError('Wrong queue size')
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True);return root


def audit(d,episodes,reused=False):
    from thesis_experiment_worker import inspect
    from buoy_network.task_reward_audit import audit_reward_csv
    values=d['coordinates'];assert_one_factor(d['variant'],values);work=Path(d['work']);spec=read(d['spec'])
    point=d['point'] if reused else read(work/'progress.json')['points'][-1]
    cp=Path(point['checkpoint']);pointdir=cp.parent.parent;r=read(pointdir/'logs/TG_policy_report.json')
    cfg=read(work/'moves_5/logs/TG_MAPPO_environment.json')['config']
    if point['episode']!=episodes or point['training_steps']!=episodes*80 or digest(cp)!=point['checkpoint_sha256']:raise ValueError('Wrong checkpoint budget/hash')
    if any(float(cfg.get(k,1. if k=='wm_actor_residual_scale' else .1))!=values[v] for k,v in [('pm_reward_alpha','alpha'),('pm_action_credit_coef','aux'),('wm_actor_residual_scale','wm_scale')]):raise ValueError('Another/wrong factor')
    if not cfg['use_wm_actor'] or not cfg['use_pm_reward'] or cfg['episode_length']!=80 or cfg['seed']!=d['seed'] or cfg['use_linear_lr_decay'] or cfg['ppo_epoch']!=1 or cfg['num_mini_batch']!=2:raise ValueError('Wrong original training protocol')
    validate_resources(r)
    if r['profile']['collect_calls']!=episodes*80 or r['profile']['ppo_updates']!=episodes or len(r['training_updates'])!=episodes:raise ValueError('PPO interaction budget mismatch')
    if r['pm']['alpha']!=values['alpha'] or r['pm']['value_mode']!='base_residual' or r['pm_action_credit']!=dict(mode='learned_set',coefficient=values['aux'],temperature=.2) or r.get('wm_actor_residual_scale',1.)!=values['wm_scale']:raise ValueError('Wrong learner coordinates')
    for aux in ['pm','wm']:
        if not r[aux]['frozen'] or not r[aux]['inference_device'].startswith('cuda') or digest(r[aux]['checkpoint'])!=r[aux]['checkpoint_sha256']:raise ValueError('Auxiliary not frozen on GPU')
    for i,u in enumerate(r['training_updates'],1):
        if u['actor_optimizer_steps']!=2*i or u['critic_head_optimizer_steps']!=2*i or any(not math.isfinite(v) for v in u.values() if isinstance(v,(int,float))):raise ValueError('Optimizer count/nonfinite record')
    physical=[json.loads(line) for line in (work/'moves_5/logs/TG_physical_metrics.jsonl').read_text().splitlines()][:episodes]
    if len(physical)!=episodes:raise ValueError('Missing training physical prefix')
    expected_scenes=[8001] if episodes==2 else SCENES
    if [e['scene'] for e in point['evaluations']]!=expected_scenes:raise ValueError('Unexpected evaluation scenes')
    evaluations=[]
    for scene in expected_scenes:
        out=pointdir/('scene_'+str(scene))/'moves_5';er=read(out/'logs/TG_policy_report.json');ecfg=read(out/'logs/TG_MAPPO_environment.json')['config']
        metric=inspect(out,'PM_WM',1 if episodes==2 else 2,True)
        if er['checkpoint_sha256']!=digest(cp) or er.get('wm_actor_residual_scale',1.)!=values['wm_scale'] or er['pm']['alpha']!=values['alpha'] or er['pm_action_credit']['coefficient']!=values['aux']:raise ValueError('Evaluation changed experimental factor')
        evaluations.append(metric)
    delivered=sum(x['delivered'] for x in evaluations);generated=sum(x['generated'] for x in evaluations)
    def successful(key):return sum(x[key]*x['delivered'] for x in evaluations if x[key] is not None)/delivered if delivered else None
    return dict(variant=d['variant'],seed=d['seed'],coordinates=values,episodes=episodes,delivery=delivered/generated,
        success_delay=successful('success_delay_mean'),success_hops=successful('success_hops_mean'),eval_loops=sum(x['loop_transfers'] for x in evaluations),
        training_delivery=[p['delivery_rate'] for p in physical],training_delivery_mean=statistics.mean(p['delivery_rate'] for p in physical),
        reused=reused,checkpoint=str(cp),checkpoint_sha256=digest(cp),evaluations=evaluations)


def run(root):
    root=Path(root).resolve()
    with suite_lock(root):
        m=read(root/'manifest.json')
        if m['status']!='prepared':raise ValueError('Refuse duplicate/failed retry')
        if digest(root/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
        for p,h in m['source']['files_sha256'].items():
            if digest(root/'source_snapshot'/p)!=h:raise ValueError('Source changed')
        for p,h in m['references'].items():
            if digest(p)!=h:raise ValueError('Reference changed '+p)
        m.update(status='running_preflight',pid=os.getpid(),started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat());atomic(root/'manifest.json',m)
        try:
            for j in m['jobs']:
                if j['phase']=='pilot' and m['status']!='running_pilot':
                    if not all(x['state']=='completed' for x in m['jobs'] if x['phase']=='preflight'):raise ValueError('Preflight not complete')
                    for d in m['reused']:
                        row=audit(d,20,True);m['rows'].append(row);d['state']='verified_reused'
                    m['status']='running_pilot';atomic(root/'manifest.json',m);print('PREFLIGHT PASSED; REUSED6; START PILOT',flush=True)
                if digest(j['spec'])!=j['spec_sha256']:raise ValueError('Spec changed')
                j.update(state='running',started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat());atomic(root/'manifest.json',m);print('START '+j['tag'],flush=True)
                work=Path(j['work']);work.mkdir(parents=True)
                start=time.perf_counter()
                with (work/'execution.log').open('xb') as stream:p=subprocess.run(j['command'],stdout=stream,stderr=subprocess.STDOUT,cwd=str(root/'source_snapshot'),timeout=2400)
                if p.returncode:raise ValueError('Task failed '+j['tag']+'; '+str(work/'execution.log'))
                progress=read(work/'progress.json')
                if progress['status']!='completed':raise ValueError('Worker incomplete')
                row=audit(j,2 if j['phase']=='preflight' else 20)
                if j['reference']:
                    import torch
                    original=Path(j['reference']);new=work/'moves_5'
                    a=torch.load(original/'checkpoints/tg_policy_latest.pt',weights_only=True,map_location='cpu');b=torch.load(new/'checkpoints/tg_policy_latest.pt',weights_only=True,map_location='cpu')
                    if any(not torch.equal(a[k][n],b[k][n]) for k in ('actor','critic') for n in a[k]):raise ValueError('Default Actor/Critic changed')
                    if (original/'logs/TG_physical_metrics.jsonl').read_bytes()!=(new/'logs/TG_physical_metrics.jsonl').read_bytes():raise ValueError('Default physical changed')
                    if read(original/'logs/TG_policy_report.json')['training_updates']!=read(new/'logs/TG_policy_report.json')['training_updates']:raise ValueError('Default optimizer records changed')
                    j['legacy_exact']=True
                if j['phase']=='pilot':m['rows'].append(row)
                j.update(state='completed',wall_seconds=time.perf_counter()-start,progress_sha256=digest(work/'progress.json'),delivery=row['delivery']);atomic(root/'manifest.json',m);print('DONE '+j['tag']+' delivery '+str(row['delivery']),flush=True)
                samples=[x['wall_seconds'] for x in m['jobs'] if x['phase']=='pilot' and x['state']=='completed']
                if samples:
                    left=sum(x['state']=='pending' for x in m['jobs']);seconds=statistics.mean(samples[-5:])*left+30
                    from datetime import timedelta
                    m['estimated_remaining_seconds']=seconds;m['estimated_finish_cn']=(datetime.now(ZoneInfo('Asia/Shanghai'))+timedelta(seconds=seconds)).isoformat();atomic(root/'manifest.json',m)
            analyze(root,m)
            for p,h in m['references'].items():
                if digest(p)!=h:raise ValueError('Reference changed during pilot')
            m.update(status='completed',finished=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),report_sha256=digest(root/'report.json'));atomic(root/'manifest.json',m);print('COMPLETE',flush=True)
        except Exception as e:
            if 'j'in locals() and j['state']=='running':j['state']='failed'
            m.update(status='failed',error=repr(e));atomic(root/'manifest.json',m);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.run)
