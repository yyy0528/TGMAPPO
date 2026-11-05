"""Pure summaries of frozen rollouts; not historical PPO-update attribution."""
import numpy as np
from src.buoy_network.credit_diagnostics import advantages


def signal_arrays(rewards, shaped, values, phi, masks, gamma, lam, alpha):
    base,train,v,p,legal=[np.asarray(x) for x in (rewards,shaped,values,phi,masks)]
    t,n=v.shape
    if (base.shape!=(t,) or train.shape!=(t,) or p.shape!=(t+1,) or legal.shape!=(t,n,n+1)
            or p[-1]!=0 or not 0<gamma<=1 or not 0<=lam<=1 or alpha<0
            or not all(np.isfinite(x).all() for x in (base,train,v,p,legal))):
        raise ValueError('Invalid frozen credit arrays')
    base,train,v,p=[x.astype(np.float64) for x in (base,train,v,p)]
    if not np.allclose(train,base+alpha*(gamma*p[1:]-p[:-1]),rtol=1e-6,atol=1e-7):
        raise ValueError('Frozen reward/potential time mismatch')
    following=np.vstack((v[1:],np.zeros_like(v[:1])))
    delta=train[:,None]+gamma*following-v
    gae=np.zeros_like(delta)
    for k in reversed(range(t)):gae[k]=delta[k]+(gamma*lam*gae[k+1] if k<t-1 else 0)
    active=legal[:,:,:n].any(-1)
    _,z=advantages(gae,np.zeros_like(gae),active)
    effective=v+alpha*p[:-1,None]
    next_effective=np.vstack((effective[1:],np.zeros_like(effective[:1])))
    base_delta=base[:,None]+gamma*next_effective-effective
    return {'raw':gae,'normalized':z,'active':active,'delta':delta,
            'value_coordinate_error':float(np.max(np.abs(base_delta-delta))),
            'critic_agent_spread':float(np.ptp(v,axis=1).max()),
            'advantage_agent_spread':float(np.ptp(gae,axis=1).max())}


def aggregate(rows):
    predicates={'all_active':lambda r:True,'WAIT':lambda r:r['action']==r['wait'],
        'forward':lambda r:r['action']!=r['wait'],
        'Sink_chosen':lambda r:r['action']==r['sink'],
        'Sink_ACK_success':lambda r:r['own_delivery'],
        'Sink_legal_other':lambda r:r['sink_legal'] and r['action']!=r['sink'],
        'successful_revisit':lambda r:r['revisit'],
        'revisit_with_other_delivery':lambda r:r['revisit'] and r['other_deliveries']>0}
    result={}
    for name,predicate in predicates.items():
        selected=[r for r in rows if predicate(r)]
        summary={'count':len(selected)}
        if selected:
            for field in ('advantage','raw_advantage','base_reward','progress_reward','wait_score','sink_score'):
                summary[field+'_mean']=float(np.mean([r[field] for r in selected]))
            summary['negative_advantage_fraction']=sum(r['advantage']<0 for r in selected)/len(selected)
        result[name]=summary
    return result


def packet_waiting(frames,states,routes,wait):
    """Account for service WAIT, missing links, FIFO blocking and forwarding."""
    result={}
    for route in routes:
        key=(route['episode'],route['packet_id'])
        result[key]=dict(route,service_wait=0,forced_wait=0,queued_behind=0,forward_attempts=0,alive_slots=0)
    for frame,state in zip(frames,states):
        service=state['service_ids'];episode=frame['episode']
        for packet,node in state['packet_nodes']:
            row=result[(episode,packet)];row['alive_slots']+=1
            if service[node]!=packet:row['queued_behind']+=1
            elif frame['actions'][node]!=wait:row['forward_attempts']+=1
            elif any(frame['mask'][node][:wait]):row['service_wait']+=1
            else:row['forced_wait']+=1
    for row in result.values():
        total=sum(row[k] for k in ('service_wait','forced_wait','queued_behind','forward_attempts'))
        expected=row['delivery_step'] if row['delivered'] else sum(f['episode']==row['episode'] for f in frames)
        if total!=row['alive_slots'] or total!=expected:
            raise ValueError('Packet waiting ledger does not match delivery time')
    return list(result.values())
