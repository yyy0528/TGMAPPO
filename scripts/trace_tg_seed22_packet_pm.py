"""Read-only per-packet PM extraction on the exact low-strength greedy replay."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args()
    root=Path(a.root).resolve();manifest=json.loads((root/'manifest.json').read_text());snapshot=Path(manifest['source_snapshot'])
    sys.path[:0]=[str(snapshot),str(snapshot/'src'),str(snapshot/'scripts')]
    import torch
    import diagnose_tg_policy as d
    from buoy_network.progress_runtime import FrozenPMRuntime
    from buoy_network.pm_return import expected_task_return
    original=FrozenPMRuntime.potential;predictions=[]
    def observed(self,env,step):
        keys=sorted(p.id for p in env.packets if not p.if_sink)
        packets={p.id:p for p in env.packets if not p.if_sink};captured=[];forward=self.model.step
        def score(*args,**kwargs):
            logits,following=forward(*args,**kwargs)
            captured.append(logits.detach().clone());return logits,following
        with patch.object(self.model,'step',score):phi=original(self,env,step)
        if captured:
            with torch.no_grad():
                result=expected_task_return(captured[0],torch.full((len(keys),),self.horizon-step,dtype=torch.long),
                    self.horizon,self.gamma,self.task_contract)
            reconstructed=float(result['task_return'].sum())/self.initial_packets
            if abs(reconstructed-phi)>1e-12:raise ValueError('Per-packet extraction differs from team potential')
            if 13 in packets:
                i=keys.index(13);packet=packets[13]
                predictions.append({'episode':self.episode,'step':step,'node':packet.agent_id,'route':list(packet.route),
                    'value':float(result['task_return'][i]),'probability':float(result['probability'][i]),
                    'delivery':float(result['delivery'][i]),'delay':float(result['delay'][i]),
                    'unfinished':float(result['unfinished'][i]),'team_phi':phi,'initial_packets':self.initial_packets})
        return phi
    output=root/'packet_pm_replay'
    with patch.object(FrozenPMRuntime,'potential',observed):
        report=d.worker(root/'view',output,'alpha01',22,'trained','greedy')
    if not report['weights_unchanged'] or report['optimizer_updates']!=0 or not report['pilot_metrics_exact_match']:
        raise ValueError('Per-packet observation perturbed frozen replay')
    (output/'packet13_predictions.json').write_text(json.dumps(predictions,indent=2,allow_nan=False),encoding='utf-8')
    rows=[json.loads(x) for x in (root/'alpha01_greedy/credit_rows.jsonl').read_text().splitlines()]
    advantages={(r['episode'],r['step'],r['sender']):r['advantage'] for r in rows}
    frames=[json.loads(x) for x in (output/'slots.jsonl').read_text().splitlines()]
    scores={(r['episode'],r['step']):r for r in predictions};seen={0:{13},1:{13}};events=[]
    for frame in frames:
        ep,t=frame['episode'],frame['step']
        for event in frame['events']:
            if event['packet_id']!=13:continue
            before=scores[(ep,t)];following=scores.get((ep,t+1),{'value':0.,'team_phi':0.})
            own=.1*(.99*following['value']-before['value'])/before['initial_packets']
            team=frame['reward_train']-frame['reward_env'];revisit=event['ack'] and event['receiver'] in seen[ep]
            events.append(dict(event,episode=ep,step=t,revisit=bool(revisit),packet_value_before=before['value'],
                packet_value_after=following['value'],packet_shaping=own,team_shaping=team,
                other_packet_shaping=team-own,normalized_GAE=advantages[(ep,t,event['sender'])],
                other_deliveries=sum(e['ack'] and e['receiver']==24 for e in frame['events'])))
            if event['ack']:seen[ep].add(event['receiver'])
    result={'optimizer_updates':0,'weights_unchanged':True,'physical_exact':True,
        'events':events,'prediction_rows':len(predictions),
        'scope':'per-packet PM component on observed route, not causal credit or a counterfactual action value'}
    (output/'packet13_signals.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('REPORT '+str(output/'packet13_signals.json'))
    print(json.dumps([e for e in events if e['sender']==13],indent=2))


if __name__=='__main__':main()
