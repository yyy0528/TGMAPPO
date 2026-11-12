"""Paired factual replay and explicit inference-only WM residual intervention."""
import argparse,json,math,os,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT)]
from thesis_experiment_worker import run
from tg_convergence import read,atomic,digest


def trace(spec):
    import numpy as np,torch
    from buoy_network.agent import DeepQNetwork
    from buoy_network.environment import buoy_environment
    from buoy_network.scenario_evaluation import scenario_context
    from buoy_network.physical_metrics import PacketMetrics
    from buoy_network.pm_action_credit import collect_action_targets
    from buoy_network.component_validation import calibration
    from buoy_network.wm_runtime import tensor_digest
    from onpolicy.algorithms.graph_actor_critic import GR_Actor
    from onpolicy.algorithms.utils.distributions import Categorical,FixedCategorical
    from onpolicy.algorithms.utils.wm_features import incremental_features
    s=read(spec);out=Path(s['output']);out.mkdir(parents=True,exist_ok=False)
    if digest(s['checkpoint'])!=s['checkpoint_sha256']:raise ValueError('Policy changed')
    collect,insert=DeepQNetwork.collect,DeepQNetwork.insert
    category,wm=Categorical.forward,GR_Actor._wm_distribution
    attempted,summary=PacketMetrics.attempt,PacketMetrics.summary
    owner={};captured={};records=[];packets=[];ack=[];queue=[];current={};episode=-1
    def category_observed(self,x,available_actions=None):
        result=category(self,x,available_actions)
        captured['probabilities']=result.probs.detach().cpu().numpy()
        captured['base_action']=captured['probabilities'].argmax(-1)
        captured['full_action']=captured['base_action']
        return result
    def wm_observed(self,features,available,wm_features,obs=None):
        original=wm(self,features,available,wm_features,obs)
        with torch.no_grad():
            base_logits=(self.act.action_out.linear(features)/self.policy_temperature).masked_fill(available<=0,float('-inf'))
            base=FixedCategorical(logits=base_logits)
            selected=base if s['mode']=='no_wm_residual' else original
            captured['base_action']=base.probs.argmax(-1).cpu().numpy()
            captured['full_action']=original.probs.argmax(-1).cpu().numpy()
            captured['probabilities']=selected.probs.cpu().numpy()
            raw=torch.as_tensor(wm_features,device=features.device,dtype=features.dtype)
            transformed=incremental_features(raw,obs,available,self.wm_input_scale,self.wm_input_center)
            clipped=(transformed.abs()>=5).float().mean(-1)
            captured['clipped']=clipped.cpu().numpy()
        return selected
    def observed_collect(self,step):
        nonlocal episode,current
        if step==0:episode+=1
        owner['model']=self
        before=self.env.available_actions()[0]>0;active=self.env.active_masks()[0,:,0]>0
        head={}
        for packet in sorted(self.env.packets,key=lambda p:p.queue_order):
            if not packet.if_sink:head.setdefault(packet.agent_id,packet)
        result=collect(self,step)
        shadow=collect_action_targets(self.pm_runtime,self.env,self.buffer,step)
        current={}
        for node in np.flatnonzero(active):
            action=int(result[1][0,node,0]);n=self.env.buoy_number;packet=head[node]
            teacher=shadow[node];good=(teacher>0)&(teacher>=teacher.max()*math.exp(-.1))
            row=dict(episode=episode,step=step,node=int(node),packet=packet.id,action=action,wait=action==n,
                sink_legal=bool(before[node,n-1]),sink_chosen=action==n-1,queue=int(self.env.buoys[node].packet),
                route=list(packet.route),probabilities=captured['probabilities'][node].tolist(),
                base_action=int(captured['base_action'][node]),full_action=int(captured['full_action'][node]),
                teacher_best=int(teacher.argmax()),teacher_chosen_good=bool(good[action]),teacher_wait_good=bool(good[n]),
                ack=None,wm_feature=None,prior_ack=None)
            if self.wm_runtime is not None:
                row['wm_feature']=self.buffer.wm_features[step,0,node,action].tolist()
                row['prior_ack']=float(self.buffer.obs[step,0,node,action]) if action!=n else None
                row['clip_fraction']=float(captured['clipped'][node,action])
                row['current_queues']=[float(self.buffer.node_obs[step,0,0,node,2]),float(self.buffer.node_obs[step,0,0,node if action==n else action,2])]
            records.append(row);current[int(node)]=row
        return result
    def observed_attempt(self,packet,sender,receiver,success):
        attempted(self,packet,sender,receiver,success)
        row=current.get(sender)
        if row is not None:
            if row['packet']!=packet:raise ValueError('Decision and service packet mismatch')
            row['ack']=bool(success)
            if row['wm_feature'] is not None:ack.append([row['wm_feature'][0],row['prior_ack'],float(success)])
    def observed_insert(self,data):
        following=np.asarray(data[2])[0,0]
        for node,row in current.items():
            if row['wm_feature'] is None:continue
            target=node if row['wait'] else row['action']
            queue.append([row['wm_feature'][1],float(following[node,2])-row['current_queues'][0]])
            if not row['wait']:queue.append([row['wm_feature'][2],float(following[target,2])-row['current_queues'][1]])
        return insert(self,data)
    def observed_summary(self,ep,horizon):
        result=summary(self,ep,horizon)
        packets.append(dict(episode=ep,physical=result,packets=self.packets,
            final_positions=[list(map(float,n.position)) for n in owner['model'].env.buoys]))
        return result
    evaluation=dict(output=str(out/'evaluation'),scene=s['scene'],episodes=2,checkpoint=s['checkpoint'])
    with scenario_context(buoy_environment,s['settings']) as scenario,patch.object(DeepQNetwork,'collect',observed_collect),patch.object(DeepQNetwork,'insert',observed_insert),patch.object(Categorical,'forward',category_observed),patch.object(GR_Actor,'_wm_distribution',wm_observed),patch.object(PacketMetrics,'attempt',observed_attempt),patch.object(PacketMetrics,'summary',observed_summary):
        run(s['job'],evaluation)
    logs=out/'evaluation/moves_5/logs';native=read(logs/'TG_policy_report.json')
    if native['optimizer_updates']!=0 or not native['actor_unchanged'] or not native['critic_unchanged']:raise ValueError('Frozen policy changed')
    physical=logs/'TG_physical_metrics.jsonl';expected=Path(s['reference'])/'logs/TG_physical_metrics.jsonl'
    exact=physical.read_bytes()==expected.read_bytes()
    if s['mode']=='factual' and not exact:raise ValueError('Factual diagnostic changed archived trajectory')
    if digest(s['checkpoint'])!=s['checkpoint_sha256']:raise ValueError('Policy file modified')
    generated=sum(x['physical']['generated_packets'] for x in packets);delivered=sum(x['physical']['delivered_packets'] for x in packets)
    waits=[r for r in records if r['wait']]
    result=dict(status='completed',algorithm=s['algorithm'],seed=s['seed'],condition=s['condition'],scene=s['scene'],mode=s['mode'],
        delivery=delivered/generated,delivered=delivered,generated=generated,unresolved=generated-delivered,
        active_decisions=len(records),wait_decisions=len(waits),
        teacher_disagreement=sum(not r['teacher_chosen_good'] for r in records),
        waits_outside_teacher_set=sum(not r['teacher_chosen_good'] for r in waits),
        waits_teacher_best=sum(r['teacher_best']==25 for r in waits),
        wm_changes_action=sum(r['full_action']!=r['base_action'] for r in records),
        wm_introduces_wait=sum(r['full_action']==25 and r['base_action']!=25 for r in records),
        loop_transfers=sum(x['physical']['loop_transfers'] for x in packets),
        frozen_replay_exact=exact,optimizer_updates=0,weights_unchanged=True,
        scope='Teacher disagreement is not a true optimal-action label; no-WM mode is an inference intervention, not retrained ablation')
    result['wm_online']=dict(ack_attempts=len(ack),queue_samples=len(queue))
    if ack:
        a=np.asarray(ack);result['wm_online'].update(ack_brier=float(np.mean((a[:,0]-a[:,2])**2)),
            history_brier=float(np.mean((a[:,1]-a[:,2])**2)),ack_calibration=calibration(a[:,0],a[:,2]))
    if queue:
        a=np.asarray(queue);result['wm_online'].update(queue_mse=float(np.mean((a[:,0]-a[:,1])**2)),queue_zero_mse=float(np.mean(a[:,1]**2)))
    result['node_groups']={str(n):dict(active=sum(r['node']==n for r in records),wait=sum(r['node']==n and r['wait'] for r in records)) for n in sorted({r['node'] for r in records})}
    (out/'decisions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    atomic(out/'packet_ledger.json',packets);atomic(out/'scenario_audit.json',scenario);atomic(out/'report.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args();trace(a.spec)
