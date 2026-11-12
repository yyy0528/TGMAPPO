"""Frozen original-env routing replay. No optimizer steps or model changes."""
import argparse,json,os,sys,math,statistics,subprocess
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
TEMP=ROOT/'results/thesis_diagnostic_temp';TEMP.mkdir(parents=True,exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(TEMP)
from tg_convergence import read,atomic,digest


def case(spec):
    from unittest.mock import patch
    import torch,numpy as np
    from buoy_network.thesis_baselines import BaselineLearner
    from buoy_network.agent import DeepQNetwork
    from buoy_network.physical_metrics import PacketMetrics
    from onpolicy.algorithms.graph_actor_critic import GR_Actor
    from thesis_experiment_worker import run
    s=read(spec);job=read(s['job']);out=Path(s['output']);out.mkdir(parents=True,exist_ok=False)
    native=job['algorithm']=='PM_WM'
    cls=DeepQNetwork if native else BaselineLearner
    collect,insert,summary,attempt=cls.collect,cls.insert,PacketMetrics.summary,PacketMetrics.attempt
    distribution=GR_Actor._wm_distribution
    owner={};records=[];episodes=[];transitions=[];current={};episode=-1;wm_scores={}
    def wm_dist(self,features,available,wm,obs=None):
        result=distribution(self,features,available,wm,obs)
        with torch.no_grad():
            base=self.act.action_out.linear(features)/self.policy_temperature
            base=base.masked_fill(available<=0,float('-inf'))
            wm_scores['base']=base.detach().cpu().numpy()
            wm_scores['full']=result.logits.detach().cpu().numpy()
        return result
    def collected(self,step):
        nonlocal episode,current
        if step==0:episode+=1
        owner['model']=self
        mask=self.env.available_actions()[0]>0;active=self.env.active_masks()[0,:,0]>0
        packets={}
        for p in sorted(self.env.packets,key=lambda p:p.queue_order):
            if not p.if_sink:packets.setdefault(p.agent_id,p)
        result=collect(self,step);actions=result[1][0,:,0]
        shadow=None
        if native:
            from buoy_network.pm_action_credit import collect_action_targets
            shadow=collect_action_targets(self.pm_runtime,self.env,self.buffer,step)
        if not native:
            with torch.no_grad():scores=self.actor(self.t(self.x)).cpu().numpy()
        else:scores=wm_scores['full']
        current={}
        for node in np.flatnonzero(active):
            n=self.env.buoy_number;a=int(actions[node]);p=packets[node]
            legal=np.flatnonzero(mask[node]);forward=legal[legal!=n]
            best_forward=int(forward[np.argmax(scores[node,forward])])
            r=dict(episode=episode,step=step,node=int(node),packet=p.id,route=list(p.route),queue=int(self.env.buoys[node].packet),
                action=a,wait=a==n,sink_legal=bool(mask[node,n-1]),sink_chosen=a==n-1,
                best_forward=best_forward,wait_margin=float(scores[node,n]-scores[node,best_forward]),ack=None)
            if native:
                target=shadow[node]
                good=(target>0)&(target>=target.max()*math.exp(-.1))
                r.update(base_action=int(np.where(mask[node],wm_scores['base'][node],-np.inf).argmax()),
                    teacher_action=int(target.argmax()),teacher_wait_good=bool(good[n]),teacher_chosen_good=bool(good[a]),
                    teacher_wait_loggap=float(math.log(max(float(target.max()),1e-35)/max(float(target[n]),1e-35))),
                    wm_feature=self.buffer.wm_features[step,0,node,a].tolist())
            records.append(r);current[int(node)]=r
        return result
    def attempted(self,packet,sender,receiver,success):
        attempt(self,packet,sender,receiver,success)
        if sender in current:current[sender]['ack']=bool(success)
    def inserted(self,data):
        if not native:
            transitions.append(dict(self.pending,reward=float(np.asarray(data[5])[0,0,0]),
                next_x=None,next_state=None,done=bool(np.asarray(data[7]).all())))
        return insert(self,data)
    def summarized(self,ep,horizon):
        r=summary(self,ep,horizon)
        episodes.append(dict(episode=ep,delivered=r['delivered_packets'],unresolved=r['unresolved_packets'],
            packets={str(k):v for k,v in self.packets.items()}))
        return r
    evaluation=dict(output=str(out/'evaluation'),scene=s['scene'],episodes=2,checkpoint=s['checkpoint'])
    with patch.object(cls,'collect',collected),patch.object(cls,'insert',inserted),patch.object(PacketMetrics,'attempt',attempted),patch.object(PacketMetrics,'summary',summarized),patch.object(GR_Actor,'_wm_distribution',wm_dist):
        run(s['job'],evaluation)
    actual=out/'evaluation/moves_5/logs/TG_physical_metrics.jsonl'
    expected=Path(s['reference'])/'logs/TG_physical_metrics.jsonl'
    if actual.read_bytes()!=expected.read_bytes():raise ValueError('Frozen physical trajectory changed')
    waits=[r for r in records if r['wait']]
    groups={}
    for node in sorted({r['node'] for r in records}):
        rows=[r for r in records if r['node']==node]
        groups[str(node)]=dict(active=len(rows),wait=sum(r['wait'] for r in rows),
            sink_available=sum(r['sink_legal'] for r in rows),sink_missed=sum(r['sink_legal'] and not r['sink_chosen'] for r in rows),
            mean_wait_margin=statistics.mean(r['wait_margin'] for r in rows))
    result=dict(algorithm=job['algorithm'],seed=job['seed'],scene=s['scene'],checkpoint=s['checkpoint'],checkpoint_sha256=digest(s['checkpoint']),
        frozen_replay_exact=True,active_decisions=len(records),wait_decisions=len(waits),node_groups=groups,
        unresolved=sum(e['unresolved'] for e in episodes),delivered=sum(e['delivered'] for e in episodes))
    if native:
        result.update(wm_changes_action=sum(r['base_action']!=r['action'] for r in records),
            waits_introduced_by_wm=sum(r['wait'] and r['base_action']!=25 for r in records),
            wait_teacher_good=sum(r['teacher_wait_good'] for r in waits),
            wait_teacher_best=sum(r['teacher_action']==25 for r in waits),
            waits_outside_teacher_set=sum(not r['teacher_chosen_good'] for r in waits))
    (out/'decisions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    atomic(out/'packet_ledger.json',episodes);atomic(out/'diagnosis.json',result)
    print(json.dumps(result),flush=True)


def study():
    output=ROOT/'results/thesis_policy_audit'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True)
    cases=[]
    sources=[('VDN',[11,22,33],ROOT/'results/thesis_extra_baselines/20261112_104503_699371'),
        ('QMIX',[11,33],ROOT/'results/thesis_experiments/20261112_011930_763352'),
        ('PM_WM',[11],ROOT/'results/thesis_experiments/20261112_011930_763352')]
    for group,seeds,source in sources:
        manifest=read(source/'manifest.json')
        for seed in seeds:
            job=next(j for j in manifest['jobs'] if j['phase']=='main' and j['algorithm']==group and j['seed']==seed)
            progress=read(Path(job['work'])/'progress.json')
            for point in progress['points']:
                if point['episode'] not in ([280,300] if group=='PM_WM' else [300]):continue
                scenes=[8001,8002,8003] if group=='PM_WM' and point['episode']==300 else [8001]
                for scene in scenes:
                    work=Path(job['work'])/'convergence'/('e%04d'%point['episode'])/('scene_'+str(scene))/'moves_5'
                    tag='{}_{}_e{}_s{}'.format(group,seed,point['episode'],scene)
                    path=output/(tag+'.json')
                    atomic(path,dict(job=job['spec'],scene=scene,checkpoint=point['checkpoint'],reference=str(work),output=str(output/tag)))
                    cases.append(dict(tag=tag,spec=str(path),state='pending'))
    m=dict(status='running',root=str(output),cases=cases);atomic(output/'manifest.json',m);print('OUTPUT '+str(output),flush=True)
    try:
        for c in cases:
            c['state']='running';atomic(output/'manifest.json',m)
            with (output/(c['tag']+'.log')).open('xb') as f:
                p=subprocess.run([sys.executable,'-s','-u',str(Path(__file__).resolve()),'--case',c['spec']],stdout=f,stderr=subprocess.STDOUT,timeout=900)
            if p.returncode:raise ValueError('Frozen audit failed: '+c['tag'])
            c['state']='completed';atomic(output/'manifest.json',m);print('DONE '+c['tag'],flush=True)
        m['status']='completed';atomic(output/'manifest.json',m)
    except Exception as e:
        m.update(status='failed',error=repr(e));atomic(output/'manifest.json',m);raise


def resume(root):
    root=Path(root).resolve();m=read(root/'manifest.json')
    if m['status']!='failed':raise ValueError('Only a failed diagnostic can resume')
    m['previous_error']=m.pop('error',None);m['status']='running';atomic(root/'manifest.json',m)
    try:
        for c in m['cases']:
            if c['state']=='completed':continue
            old=read(c['spec']);retry=root/(c['tag']+'_retry_'+datetime.now().strftime('%H%M%S'))
            old['output']=str(retry);spec=retry.with_suffix('.json');atomic(spec,old)
            c.update(spec=str(spec),state='running');atomic(root/'manifest.json',m)
            with retry.with_suffix('.log').open('xb') as f:
                p=subprocess.run([sys.executable,'-s','-u',str(Path(__file__).resolve()),'--case',str(spec)],stdout=f,stderr=subprocess.STDOUT,timeout=900)
            if p.returncode:raise ValueError('Diagnostic failed: '+str(spec))
            c['state']='completed';atomic(root/'manifest.json',m);print('DONE '+c['tag'],flush=True)
        m['status']='completed';atomic(root/'manifest.json',m)
    except Exception as e:
        m.update(status='failed',error=repr(e));atomic(root/'manifest.json',m);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--resume');a=p.parse_args()
    if a.case:case(a.case)
    elif a.resume:resume(a.resume)
    else:study()
