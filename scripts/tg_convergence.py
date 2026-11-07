"""Read-only measurement helpers for original TG convergence studies."""
import hashlib,json,math,pickle,statistics
from pathlib import Path

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(value,message):
    if not value:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    import os
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(str(q),str(p))
def setarg(command,key,value):
    if key in command:command[command.index(key)+1]=str(value)
    else:command.extend([key,str(value)])
def getarg(command,key):return command[command.index(key)+1]
def checkpoint_grid(episodes,interval):
    require(type(episodes) is int and type(interval) is int and episodes>0 and interval>0,'Invalid checkpoint budget')
    return sorted(set([0,episodes]+list(range(interval,episodes+1,interval))))
def trailing_mean(values,window):
    require(type(window) is int and window>0,'Invalid window')
    return [None if i+1<window else sum(values[i+1-window:i+1])/window for i in range(len(values))]
def threshold_time(points,threshold,consecutive=2,interval=20,complete=False):
    require(0<=threshold<=1 and consecutive>=1,'Invalid threshold')
    points=sorted(points,key=lambda p:p['episode'])
    first=next((p for p in points if p['delivery']>=threshold),None)
    sustained=None
    for i in range(len(points)-consecutive+1):
        block=points[i:i+consecutive]
        if all(p['delivery']>=threshold for p in block) and all(b['episode']-a['episode']==interval for a,b in zip(block,block[1:])):
            sustained={'first_episode':block[0]['episode'],'confirmed_episode':block[-1]['episode'],
                       'training_steps':block[-1]['training_steps'],'training_seconds':block[-1]['training_seconds']};break
    return {'first_observed_episode':first['episode'] if first else None,'sustained':sustained,
            'status':('confirmed' if sustained else ('not_confirmed_by_budget' if complete else 'not_confirmed_yet')),
            'resolution_episodes':interval}
def normalized_auc(points):
    points=sorted(points,key=lambda p:p['episode'])
    if len(points)<2:return None
    require(points[0]['episode']==0,'AUC requires initial checkpoint')
    last=points[-1]['episode'];require(last>0,'Empty AUC horizon')
    return sum((b['episode']-a['episode'])*(a['delivery']+b['delivery'])/2 for a,b in zip(points,points[1:]))/last
def summarize(values):
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
            'sample_sd':statistics.stdev(values) if len(values)>1 else None}
def physical_return(row,contract,gamma=.99):
    h=row['horizon'];p=row['generated_packets']
    def cost(t):return contract['delay_weight']/h*sum(gamma**k for k in range(t))
    delivery=sum(contract['delivery_weight']*gamma**(d-1)-cost(d) for d in row['success_delays'])
    censor=row['unresolved_packets']*(-cost(h)-contract['unfinished_weight']*gamma**(h-1))
    return (delivery+censor)/max(p,1)
def rng_digest():
    import random,numpy as np,torch
    data=(random.getstate(),np.random.get_state(),torch.get_rng_state().cpu().numpy().tobytes(),
          [x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    return hashlib.sha256(pickle.dumps(data,protocol=4)).hexdigest()
def nested_digest(value):
    import torch
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:h.update(pickle.dumps(x,protocol=4))
    visit(value);return h.hexdigest()
def learner_fingerprint(model):
    data={'actor':model.policy.actor.state_dict(),'critic':model.policy.critic.state_dict(),
          'actor_optimizer':model.policy.actor_optimizer.state_dict(),'critic_optimizer':model.policy.critic_optimizer.state_dict(),
          'scaler':model.trainer.scaler.state_dict(),
          'normalizer':model.trainer.value_normalizer.state_dict() if model.trainer.value_normalizer is not None else None}
    return {'parameters_and_optimizer':nested_digest(data),'rng':rng_digest()}
def assert_reference(model,reference):
    import torch
    old=torch.load(str(Path(reference)/'checkpoints/tg_policy_latest.pt'),map_location='cpu')
    for name in ('actor','critic'):
        current=getattr(model.policy,name).state_dict()
        require(set(current)==set(old[name]) and all(torch.equal(current[k].cpu(),old[name][k]) for k in current),'Prefix20 differs: '+name)
    require(model.training_records==read(Path(reference)/'logs/TG_policy_report.json')['training_updates'],'Prefix20 PPO losses/counters differ')
def audit_frozen(output,checkpoint,spec):
    from buoy_network.task_reward_audit import audit_reward_csv
    output=Path(output);r=read(output/'logs/TG_policy_report.json')
    require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged'],'Evaluation mutated policy')
    require(r['checkpoint_sha256']==digest(checkpoint) and r['profile']['collect_calls']==spec['eval_episodes']*spec['horizon'],'Wrong evaluation source/budget')
    require(r['resources']['actor_trainable']==21956 and r['resources']['critic_trainable']==18615 and r['resources']['device']=='cpu','Wrong evaluation capacity/device')
    audit_reward_csv(output/'logs',r['reward_contract'])
    rows=[json.loads(x) for x in (output/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(rows)==spec['eval_episodes'],'Missing evaluation episodes')
    return {'delivery':sum(x['delivered_packets'] for x in rows)/sum(x['generated_packets'] for x in rows),
        'base_return':statistics.mean(physical_return(x,r['reward_contract']) for x in rows),
        'delivered':sum(x['delivered_packets'] for x in rows),'generated':sum(x['generated_packets'] for x in rows),
        'loops':sum(x['loop_transfers'] for x in rows),'output':str(output),'policy_report_sha256':digest(output/'logs/TG_policy_report.json'),
        'physical_sha256':digest(output/'logs/TG_physical_metrics.jsonl')}
