"""Frozen Actors score the same alpha=0.1 greedy observations; execute nothing."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args()
    root=Path(a.root).resolve();m=json.loads((root/'manifest.json').read_text());snapshot=Path(m['source_snapshot'])
    sys.path[:0]=[str(snapshot),str(snapshot/'src'),str(snapshot/'scripts')]
    import numpy as np
    import torch
    import diagnose_tg_policy as d
    view=d.read(root/'view/manifest.json');out=root/'same_inputs';out.mkdir(exist_ok=False)
    data=np.load(str(root/'alpha01_greedy/actor_inputs.npz'))
    frames=[json.loads(x) for x in (root/'alpha01_greedy/slots.jsonl').read_text().splitlines()]
    cache=d.read(root/'alpha01_greedy/raw_credit_cache.json')['cache']
    t,n=data['obs'].shape[:2]
    arrays=[data[k].reshape(t*n,*data[k].shape[2:]) for k in
            ('obs','node_obs','adj','agent_id','rnn_states','masks','available_actions')]
    probabilities={};fingerprints={}
    for group in ('A','task_return','alpha01'):
        training=next(j for j in view['jobs'] if j['group']==group and j['kind']=='training')
        evaluation=next(j for j in view['jobs'] if j['group']==group and j['kind']=='evaluation')
        config=d.read(Path(evaluation['output'])/'logs/TG_MAPPO_environment.json')['config']
        _,learner=d.build(config,out/('constructor_'+group))
        learner.load_policy(Path(training['output'])/'checkpoints/tg_policy_latest.pt')
        actor=learner.policy.actor;actor.eval();capture=d.DistributionCapture(actor)
        before=d.tensor_digest(actor.state_dict());rng=torch.get_rng_state().clone();pieces=[]
        try:
            with torch.no_grad():
                for start in range(0,t*n,100):
                    x=[value[start:start+100] for value in arrays]
                    actor(*x[:6],available_actions=x[6],deterministic=True)
                    pieces.append(capture.last.copy())
        finally:capture.close()
        if before!=d.tensor_digest(actor.state_dict()) or not torch.equal(rng,torch.get_rng_state()):
            raise ValueError('Counterfactual probability probe modified weights/RNG')
        probabilities[group]=np.concatenate(pieces).reshape(t,n,n+1)
        fingerprints[group]={'actor_state_sha256':before,'checkpoint_sha256':d.sha(Path(training['output'])/'checkpoints/tg_policy_latest.pt'),
                             'weights_unchanged':True,'RNG_unchanged':True}
    actual=np.asarray([f['probabilities'] for f in frames]);mask=data['available_actions']>0
    error=float(np.abs(actual-probabilities['alpha01']).max())
    actions=np.asarray([f['actions'] for f in frames]);active=mask[:,:,:n].any(-1)
    if error>1e-6 or not np.array_equal(probabilities['alpha01'].argmax(-1),actions):
        raise ValueError('Same-input own-policy probability replay differs')
    summaries={};rows=[];examples=[]
    for group,probs in probabilities.items():
        selected=probs.argmax(-1);low_wait=active&(actions==25);sink_legal=active&mask[:,:,24]
        margin=probs[:,:,25]-probs[:,:,:25].max(-1)
        summaries[group]={'active_decisions':int(active.sum()),'same_action':int((active&(selected==actions)).sum()),
            'low_WAIT_alt_forward':int((low_wait&(selected!=25)).sum()),
            'low_forward_alt_WAIT':int((active&(actions!=25)&(selected==25)).sum()),
            'sink_legal':int(sink_legal.sum()),'sink_chosen':int((sink_legal&(selected==24)).sum()),
            'WAIT_minus_best_forward_on_low_WAIT':float(margin[low_wait].mean())}
    for slot,node in np.argwhere(active):
        row={'episode':int(slot//80),'step':int(slot%80),'sender':int(node),
             'service_packet_id':cache[slot]['service_ids'][node],'actual_action':int(actions[slot,node]),
             'alternative_actions':{g:int(p[slot,node].argmax()) for g,p in probabilities.items()}}
        rows.append(row)
        if node==13 or row['service_packet_id']==13:
            row=dict(row,rankings={g:sorted([[int(k),float(p[slot,node,k])] for k in np.flatnonzero(mask[slot,node])],key=lambda x:-x[1])
                                   for g,p in probabilities.items()})
            examples.append(row)
    (out/'rows.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows),encoding='utf-8')
    np.savez_compressed(str(out/'probabilities.npz'),**probabilities)
    result={'own_probability_max_error':error,'executed_environment_steps':0,'optimizer_updates':0,
            'fingerprints':fingerprints,'summaries':summaries,'packet13_examples':examples,
            'scope':'same observed inputs, counterfactual action rankings only; no counterfactual delivery claim'}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summaries,indent=2));print('REPORT '+str(out/'report.json'))


if __name__=='__main__':main()
