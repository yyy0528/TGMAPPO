"""Read-only paired PM action diagnosis; all extra work waits for training to end."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def require(v,msg):
    if not v:raise ValueError(msg)

def worker(root,out,group,seed):
    snapshot=root/'source_snapshot'
    sys.path[:0]=[str(snapshot/'scripts'),str(snapshot/'src'),str(snapshot)]
    import diagnose_tg_policy as d
    import numpy as np
    import torch
    original=d.DeepQNetwork.collect;handles=[]
    def collect(self,step):
        captured={}
        if group=='PM':
            h=self.policy.actor.act.action_out.register_forward_hook(lambda m,i,value:captured.update(pi=value.probs.detach().cpu().numpy().copy()))
        else:h=None
        try:result=original(self,step)
        finally:
            if h is not None:h.remove()
        if group!='PM':return result
        from buoy_network.pm_action_credit import collect_action_targets
        targets=collect_action_targets(self.pm_runtime,self.env,self.buffer,step)
        if not hasattr(self,'_paired_actors'):
            self._paired_actors={}
            manifest=read(root/'manifest.json')
            for mode,label in [('disabled','TG'),('zero_regret','Immediate')]:
                job=next(j for j in manifest['jobs'] if j['tag']=='holdout_'+mode+'_'+str(seed)+'_train')
                actor=copy.deepcopy(self.policy.actor)
                actor.load_state_dict(torch.load(str(Path(job['output'])/'checkpoints/tg_policy_latest.pt'),map_location='cpu')['actor'])
                actor.eval();self._paired_actors[label]=actor
        b=self.buffer
        alternatives={}
        with torch.no_grad():
            for label,actor in self._paired_actors.items():
                action,_,_=actor(b.obs[step,0],b.node_obs[step,0],b.adj[step,0],b.agent_id[step,0],
                    b.rnn_states[step,0],b.masks[step,0],available_actions=b.available_actions[step,0],deterministic=True)
                alternatives[label]=action[:,0].cpu().numpy().astype(int)
        heads={}
        for packet in sorted(self.env.packets,key=lambda p:p.queue_order):
            if not packet.if_sink:heads.setdefault(packet.agent_id,packet)
        if not hasattr(self,'_action_view_stream'):
            self._action_view_stream=(out/'pm_actions.jsonl').open('x',encoding='utf-8');handles.append(self._action_view_stream)
        for node,packet in sorted(heads.items()):
            mask=b.available_actions[step,0,node]>0
            legal=np.flatnonzero(mask);target=targets[node]
            require(np.all(target[mask]>=1e-30),'Target support underflow')
            log=np.log(target[mask].astype(np.float64));gaps=float(b.pm_action_spec['temperature'])*(log-log.max())
            chosen=int(result[1][0,node,0])
            row={'episode':self.pm_runtime.episode,'step':step,'node':node,'packet':packet.id,'route':list(packet.route),
                'chosen':chosen,'PM_top':int(legal[np.argmax(target[mask])]),'TG_same_input':int(alternatives['TG'][node]),
                'Immediate_same_input':int(alternatives['Immediate'][node]),'legal':legal.tolist(),
                'pi':captured['pi'][node].tolist(),'teacher':target.tolist(),
                'q_minus_max':dict((str(int(a)),float(v)) for a,v in zip(legal,gaps))}
            self._action_view_stream.write(json.dumps(row,allow_nan=False)+'\n')
        return result
    d.DeepQNetwork.collect=collect
    try:
        report=d.worker(root/'action_diagnosis_view',out,group,seed,'trained','greedy')
    finally:
        d.DeepQNetwork.collect=original
        for h in handles:h.close()
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--seed',type=int,default=2042)
    p.add_argument('--scene',type=int,default=9201);p.add_argument('--wait',action='store_true');p.add_argument('--worker',choices=('TG','Immediate','PM'));a=p.parse_args()
    root=Path(a.root).resolve();require(ROOT/'results/tg_pm_action_credit' in root.parents,'Unexpected root')
    if a.worker:
        worker(root,root/('action_diagnosis_'+str(a.seed))/a.worker,a.worker,a.seed);return
    folder=root/('action_diagnosis_'+str(a.seed));folder.mkdir(exist_ok=False)
    progress={'status':'waiting','seed':a.seed,'scene':a.scene,'jobs':[]};write(folder/'manifest.json',progress)
    print('OUTPUT '+str(folder),flush=True)
    try:
        if a.wait:
            while read(root/'manifest.json')['status']=='running':time.sleep(15)
            while (root/'known_case1042/manifest.json').exists() and read(root/'known_case1042/manifest.json')['status'] in ('waiting','running'):time.sleep(15)
        m=read(root/'manifest.json');require(m['status']=='completed','Parent incomplete')
        require(not (root/'action_diagnosis_view').exists(),'Existing view is never overwritten')
        view=root/'action_diagnosis_view';view.mkdir()
        jobs=[]
        for mode,label in [('disabled','TG'),('zero_regret','Immediate'),('learned_regret','PM')]:
            for kind in ['training','evaluation']:
                job=next(j for j in m['jobs'] if j['seed']==a.seed and j['mode']==mode and j['kind']==kind and j['tag'].startswith('holdout_') and (kind=='training' or j['scene']==a.scene))
                jobs.append(dict(job,group=label,train_seed=a.seed))
        write(view/'manifest.json',{'source':m['source'],'jobs':jobs})
        progress['status']='running'
        for group in ['TG','Immediate','PM']:
            cmd=[sys.executable,str(Path(__file__).resolve()),str(root),'--seed',str(a.seed),'--scene',str(a.scene),'--worker',group]
            print('START '+group,flush=True)
            with (folder/(group+'.log')).open('xb') as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
            job={'group':group,'command':cmd,'exit_code':proc.returncode};progress['jobs'].append(job);write(folder/'manifest.json',progress)
            require(proc.returncode==0,'Diagnostic replay failed: '+group)
            report=read(folder/group/'report.json')
            require(report['weights_unchanged'] and report['optimizer_updates']==0 and report['pilot_metrics_exact_match'],'Replay is not read-only/exact')
            print('DONE '+group,flush=True)
        from collections import Counter
        rows=[json.loads(x) for x in (folder/'PM/pm_actions.jsonl').read_text(encoding='utf-8').splitlines()]
        routes=[json.loads(x) for x in (folder/'PM/packet_routes.jsonl').read_text(encoding='utf-8').splitlines()]
        unresolved={(r['episode'],r['packet_id']) for r in routes if not r['delivered']}
        wait_action=len(rows[0]['pi'])-1 if rows else None
        waits=[r for r in rows if r['chosen']==wait_action]
        summary={'active_rows':len(rows),'wait_rows':len(waits),'wait_rows_with_PM_wait_top':sum(r['PM_top']==wait_action for r in waits),
                 'waits_PM_would_forward':sum(r['PM_top']!=wait_action for r in waits),
                 'chosen_equals_PM_top':sum(r['chosen']==r['PM_top'] for r in rows),
                 'chosen_differs_TG_same_input':sum(r['chosen']!=r['TG_same_input'] for r in rows),
                 'unresolved_head_waits_by_node':dict(Counter(str(r['node']) for r in waits if (r['episode'],r['packet']) in unresolved)),
                 'unresolved_routes':[r for r in routes if not r['delivered']],
                 'scope':'Observed candidate trajectory; same-input policies are comparisons, not true causal Q or extra training.'}
        write(folder/'summary.json',summary)
        progress.update(status='completed',engineering_passed=True,extra_environment_slots=480,optimizer_updates=0)
    except Exception as exc:
        progress.update(status='failed',error=repr(exc));write(folder/'manifest.json',progress);raise
    write(folder/'manifest.json',progress);print('DIAGNOSIS COMPLETED',flush=True)
if __name__=='__main__':main()
