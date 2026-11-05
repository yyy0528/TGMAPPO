"""Frozen original-TG actions plus raw values and FIFO service evidence."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--view',required=True)
    p.add_argument('--group',required=True);p.add_argument('--decode',choices=('greedy','sample'),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=('trained','initial'),default='trained')
    p.add_argument('--shadow-checkpoint')
    a=p.parse_args();snapshot=Path(a.snapshot).resolve()
    sys.path[:0]=[str(snapshot),str(snapshot/'src'),str(snapshot/'scripts'),str(ROOT)]
    import numpy as np
    from buoy_network.agent import DeepQNetwork
    import diagnose_tg_policy as d
    spec=importlib.util.spec_from_file_location('seed22_signals',str(ROOT/'scripts/tg_seed22_signals.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    original=DeepQNetwork.collect;insert=DeepQNetwork.insert;cache=[];inputs=[];context={}
    shadow=None
    def observed(self,step):
        nonlocal shadow
        b=self.buffer
        if a.shadow_checkpoint and step==0:
            from buoy_network.progress_runtime import FrozenPMRuntime
            from buoy_network.task_reward import reward_contract
            shadow=FrozenPMRuntime(a.shadow_checkpoint,self.env.buoy_number,b.episode_length,b.gamma,1.,
                'task_return','learned_shaped',reward_contract(self.all_args))
            shadow.start_episode(self.env,0)
        inputs.append({k:np.asarray(getattr(b,k)[step,0]).copy() for k in
            ('obs','node_obs','adj','agent_id','rnn_states','masks','available_actions')})
        service=[-1]*self.env.buoy_number;packets=[]
        for packet in sorted(self.env.packets,key=lambda x:x.queue_order):
            if not packet.if_sink:
                packets.append([packet.id,packet.agent_id])
                if service[packet.agent_id]<0:service[packet.agent_id]=packet.id
        result=original(self,step)
        raw=(self.trainer.value_normalizer.denormalize(result[0]) if b._use_popart or b._use_valuenorm else result[0])
        runtime=self.pm_runtime
        cache.append({'step':step,'values':np.asarray(raw,dtype=np.float64).reshape(-1).tolist(),
            'phi':float(runtime.phi) if runtime is not None else 0.,'service_ids':service,'packet_nodes':packets})
        if shadow is not None:cache[-1]['shadow_phi']=float(shadow.phi)
        context.update(alpha=runtime.alpha if runtime is not None else 0.,gamma=b.gamma,lam=b.gae_lambda)
        context.update(horizon=b.episode_length,episodes=self.all_args.num_episodes)
        return result
    def observed_insert(self,data):
        if shadow is not None:
            alive={p.id for p in self.env.packets if not p.if_sink}
            for key in sorted(shadow.packet_ids-shadow.finished-alive):shadow.delivered(key)
            step=cache[-1]['step']+1
            shadow.advance(self.env,step,float(np.asarray(data[5])[0,0,0]),step==shadow.horizon)
        return insert(self,data)
    output=Path(a.output).resolve()
    with patch.object(DeepQNetwork,'collect',observed),patch.object(DeepQNetwork,'insert',observed_insert):
        report=d.worker(Path(a.view).resolve(),output,a.group,22,a.stage,a.decode)
    if not report['weights_unchanged'] or report['optimizer_updates']!=0:
        raise ValueError('Frozen replay modified policy')
    frames=[json.loads(x) for x in (output/'slots.jsonl').read_text(encoding='utf-8').splitlines()]
    routes=[json.loads(x) for x in (output/'packet_routes.jsonl').read_text(encoding='utf-8').splitlines()]
    h,eps=context['horizon'],context['episodes']
    if len(cache)!=len(frames) or len(frames)!=h*eps:raise ValueError('Incomplete trace')
    rows=[];episodes=[];seen={(r['episode'],r['packet_id']):{r['route'][0]} for r in routes}
    for ep in range(eps):
        fs=frames[ep*h:(ep+1)*h];cs=cache[ep*h:(ep+1)*h]
        signal=helper.signal_arrays([f['reward_env'] for f in fs],[f['reward_train'] for f in fs],
            [x['values'] for x in cs],[x['phi'] for x in cs]+[0.],[f['mask'] for f in fs],
            context['gamma'],context['lam'],context['alpha'])
        for t,frame in enumerate(fs):
            events={e['sender']:e for e in frame['events']}
            deliveries=sum(e['receiver']==24 and e['ack'] for e in frame['events'])
            for node in np.flatnonzero(signal['active'][t]):
                node=int(node);action=frame['actions'][node];event=events.get(node)
                revisit=bool(event and event['ack'] and event['receiver'] in seen[(ep,event['packet_id'])])
                own=bool(event and event['ack'] and event['receiver']==24)
                probs=frame['probabilities'][node];adv=float(signal['normalized'][t,node])
                rows.append({'episode':ep,'step':t,'sender':node,'packet_id':cs[t]['service_ids'][node],
                    'action':action,'wait':25,'sink':24,'sink_legal':bool(frame['mask'][node][24]),
                    'own_delivery':own,'revisit':revisit,'other_deliveries':deliveries-int(own),
                    'advantage':adv,'raw_advantage':float(signal['raw'][t,node]),
                    'base_reward':frame['reward_env'],'progress_reward':frame['reward_train']-frame['reward_env'],
                    'wait_score':adv*(int(action==25)-probs[25]),
                    'sink_score':adv*(int(action==24)-probs[24])})
            for event in frame['events']:
                if event['ack']:seen[(ep,event['packet_id'])].add(event['receiver'])
        episodes.append({k:signal[k] for k in ('value_coordinate_error','critic_agent_spread','advantage_agent_spread')})
        np.savez_compressed(str(output/('signals_episode_'+str(ep)+'.npz')),raw_advantages=signal['raw'],normalized_advantages=signal['normalized'])
    packet_rows=helper.packet_waiting(frames,cache,routes,25)
    (output/'credit_rows.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows),encoding='utf-8')
    (output/'packet_waiting.json').write_text(json.dumps(packet_rows,indent=2),encoding='utf-8')
    (output/'raw_credit_cache.json').write_text(json.dumps({'context':context,'cache':cache},allow_nan=False),encoding='utf-8')
    np.savez_compressed(str(output/'actor_inputs.npz'),**{k:np.asarray([x[k] for x in inputs]) for k in inputs[0]})
    result={'group':a.group,'decode':a.decode,'weights_unchanged':True,'optimizer_updates':0,
        'alpha':context['alpha'],'scene':report['eval_scene'],'episodes':episodes,'groups':helper.aggregate(rows),
        'per_sender':{str(n):helper.aggregate([r for r in rows if r['sender']==n]) for n in range(25)},
        'limitations':['Frozen rollout GAE is recomputed; these are not historical PPO updates.',
            'Score terms are ratio=1 likelihood terms, excluding entropy and shared-parameter Jacobian.',
            'Different policies visit different states; action/advantage group means are observational.']}
    (output/'signals.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('SIGNALS '+str(output/'signals.json'),flush=True)


if __name__=='__main__':main()
