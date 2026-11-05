"""Read-only sampled failed policy: cached Sink decisions and team GAE signs.

No PPO, extra policy sampling, weight updates, or action replacement. The source
snapshot's existing trace worker supplies the original policy/environment path.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--suite',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=11)
    a=p.parse_args();snapshot=Path(a.snapshot).resolve()
    sys.path[:0]=[str(snapshot),str(snapshot/'src'),str(snapshot/'scripts')]
    import numpy as np
    from buoy_network.agent import DeepQNetwork
    from diagnose_tg_policy import worker
    original=DeepQNetwork.collect;cache=[]
    def observed(self,step):
        result=original(self,step)
        values=(self.trainer.value_normalizer.denormalize(result[0])
            if self.buffer._use_popart or self.buffer._use_valuenorm else result[0])
        cache.append({'step':step,'values':np.asarray(values,dtype=np.float64).reshape(-1),
                      'phi':float(self.pm_runtime.phi)})
        return result
    output=Path(a.output).resolve()
    with patch.object(DeepQNetwork,'collect',observed):
        report=worker(Path(a.suite).resolve(),output,'task_return',a.seed,'trained','sample')
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    require(report['weights_unchanged'] and report['optimizer_updates']==0,'Read-only worker updated weights')
    frames=[json.loads(x) for x in (output/'slots.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(frames)==len(cache)==160,'Incomplete credit trace')
    episodes=[];groups={}
    for ep in (0,1):
        rows=frames[ep*80:(ep+1)*80];captured=cache[ep*80:(ep+1)*80]
        values=np.asarray([x['values'] for x in captured]);phi=np.asarray([x['phi'] for x in captured]+[0.])
        shaped=np.asarray([x['reward_train'] for x in rows]);base=np.asarray([x['reward_env'] for x in rows])
        require(np.allclose(shaped,base+.99*phi[1:]-phi[:-1],atol=1e-6),'Reward/score capture mismatch')
        following=np.vstack((values[1:],np.zeros_like(values[:1])))
        delta=shaped[:,None]+.99*following-values;gae=np.zeros_like(delta)
        for t in reversed(range(80)):gae[t]=delta[t]+(.99*.95*gae[t+1] if t<79 else 0)
        masks=np.asarray([x['mask'] for x in rows]);active=masks[:,:,:25].any(-1)
        normalized=(gae-gae[active].mean())/(gae[active].std()+1e-5)
        actions=np.asarray([x['actions'] for x in rows]);sink_legal=masks[:,:,24]>0
        delivered=np.zeros_like(active)
        for t,row in enumerate(rows):
            for event in row['events']:
                if event['receiver']==24 and event['ack']:delivered[t,event['sender']]=True
        choices={'Sink_chosen':active&sink_legal&(actions==24),
                 'Sink_ACK_success':delivered,
                 'Sink_legal_other_action':active&sink_legal&(actions!=24)}
        for name,mask in choices.items():
            groups.setdefault(name,[]).extend(normalized[mask].tolist())
        episodes.append({'episode':ep,'critic_max_agent_spread':float(np.ptp(values,axis=1).max()),
            'advantage_max_agent_spread':float(np.ptp(gae,axis=1).max()),
            'Sink_ACK_success':int(delivered.sum()),
            'Sink_ACK_success_negative_normalized_GAE':int((delivered&(normalized<0)).sum()),
            'Sink_ACK_success_negative_team_train_reward':int((delivered&(shaped[:,None]<0)).sum())})
        np.savez_compressed(str(output/('credit_episode_'+str(ep)+'.npz')),raw_values=values,phi=phi,
            base_rewards=base,shaped_rewards=shaped,normalized_gae=normalized,actions=actions,active=active,Sink_legal=sink_legal)
    result={'scope':'sampled frozen failed policy, not an efficacy evaluation or causal attribution',
        'weights_unchanged':True,'optimizer_updates':0,'source_snapshot':str(snapshot),
        'decode':'sample','episodes':episodes,
        'actions':{k:{'count':len(v),'mean_normalized_gae':float(np.mean(v)) if v else None,
            'negative_fraction':sum(x<0 for x in v)/len(v) if v else None} for k,v in groups.items()},
        'physical':{k:report['analysis'][k] for k in ('generated','delivered','sink_legal_count','sink_chosen_count','revisit_transfers')},
        'limitation':'Team rewards/global critic give common signals; a good local action may have a negative sampled team advantage without a GAE implementation bug.'}
    (output/'sink_credit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
