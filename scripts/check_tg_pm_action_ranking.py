"""Read-only packet action-ranking gate on the exact known seed22 trajectory."""
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

def main():
    root=ROOT/'results/tg_pm_action_rank'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True,exist_ok=False)
    diagnosis=ROOT/'results/tg_seed22_signals/20261105_165019_896744'
    source=json.loads((diagnosis/'manifest.json').read_text(encoding='utf-8'))
    snapshot=Path(source['source_snapshot'])
    sys.path[:0]=[str(snapshot),str(snapshot/'src'),str(snapshot/'scripts')]
    import numpy as np
    import torch
    import diagnose_tg_policy as d
    spec=importlib.util.spec_from_file_location('buoy_network.pm_action_credit',str(ROOT/'src/buoy_network/pm_action_credit.py'))
    credit=importlib.util.module_from_spec(spec);spec.loader.exec_module(credit)
    original=d.DeepQNetwork.collect;rows=[]
    def observed(self,step):
        b=self.buffer;runtime=self.pm_runtime;service={}
        for p in sorted(self.env.packets,key=lambda p:p.queue_order):
            if not p.if_sink:service.setdefault(p.agent_id,p)
        actor={'obs':b.obs[step,0], 'node_features':b.node_obs[step,0,0],'distances':b.adj[step,0,0]}
        current=[]
        for node in np.flatnonzero(b.active_masks[step,0,:,0]>0):
            node=int(node);p=service[node];mask=b.available_actions[step,0,node]
            q=credit.packet_action_scores(runtime,actor,node,p,mask,step)
            rank=sorted([[int(a),float(q[a])] for a in np.flatnonzero(mask)],key=lambda x:-x[1])
            current.append({'episode':runtime.episode,'step':step,'node':node,'packet':p.id,'route':list(p.route),'rank':rank})
        result=original(self,step)
        for row in current:row['actual_action']=int(result[1][0,row['node'],0])
        rows.extend(current)
        return result
    print('OUTPUT '+str(root),flush=True)
    with patch.object(d.DeepQNetwork,'collect',observed):
        report=d.worker(diagnosis/'view',root/'frozen','alpha01',22,'trained','greedy')
    examples=[r for r in rows if r['step']==0 and r['node']==13]
    gate=len(examples)==2 and all(dict(r['rank'])[19]>dict(r['rank'])[17] for r in examples)
    result={'success':report['weights_unchanged'] and report['optimizer_updates']==0 and report['pilot_metrics_exact_match'],
        'gate_passed':gate,'examples':examples,'rows':len(rows),'environment_steps':160,'optimizer_updates':0,
        'scope':'Approximate held-context action scores; no causal counterfactual or routing benefit guarantee',
        'script_sha256':d.sha(Path(__file__)),'scorer_sha256':d.sha(ROOT/'src/buoy_network/pm_action_credit.py')}
    if not result['success'] or not gate:raise ValueError('PM candidate action ranking gate failed')
    (root/'rank_rows.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows),encoding='utf-8')
    (root/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS '+str(root/'report.json'),flush=True)

if __name__=='__main__':main()
