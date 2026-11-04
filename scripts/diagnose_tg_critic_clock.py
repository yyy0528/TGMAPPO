"""Frozen on-policy sampling diagnostics for a trained original-TG Critic."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-suite',required=True)
    p.add_argument('--training-output',required=True)
    p.add_argument('--evaluation-output',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    suite,training,evaluation,output=map(lambda x:Path(x).resolve(),
        (a.source_suite,a.training_output,a.evaluation_output,a.output))
    m=read(suite/'manifest.json')
    for name,digest in m['source']['files_sha256'].items():
        if sha(suite/'source_snapshot'/name)!=digest or sha(ROOT/name)!=digest:
            raise ValueError('Source changed: '+name)
    sys.path[:0]=[str(suite/'source_snapshot'),str(suite/'source_snapshot/src'),str(suite/'source_snapshot/scripts')]
    import numpy as np
    import torch
    import diagnose_tg_policy as d
    config=read(evaluation/'logs/TG_MAPPO_environment.json')['config']
    output.mkdir(parents=True,exist_ok=False)
    env,learner=d.build(config,output/'run')
    checkpoint=training/'checkpoints/tg_policy_latest.pt'
    learner.load_policy(checkpoint)
    before={k:d.tensor_digest(getattr(learner.policy,k).state_dict()) for k in ('actor','critic')}
    observations,context=[],{'episode':-1,'rows':[]}

    class Metrics(d.PacketMetrics):
        def __init__(self,packets):
            super().__init__(packets)
            context['episode']+=1
        def summary(self,episode,horizon):
            result=super().summary(episode,horizon)
            context['rows'].append(result)
            return result

    get_actions=learner.policy.get_actions
    def sample(*args,**kw):
        kw['deterministic']=False
        return get_actions(*args,**kw)
    learner.policy.get_actions=sample
    collect,insert=learner.collect,learner.insert
    def observed_collect(step):
        result=collect(step)
        b=learner.buffer
        values=learner.trainer.value_normalizer.denormalize(result[0][0])
        remaining=(b.critic_time[step,0,:,0].copy() if b.critic_time is not None else
                   np.full(env.buoy_number,1-step/b.episode_length,dtype=np.float32))
        observations.append({'episode':context['episode'],'step':step,'value':values[:,0].tolist(),
            'active':b.active_masks[step,0,:,0].tolist(),'remaining':remaining.tolist(),
            'actions':result[1][0,:,0].astype(int).tolist()})
        return result
    def observed_insert(data):
        observations[-1]['reward']=float(np.asarray(data[5])[0,0,0])
        return insert(data)
    learner.collect,learner.insert=observed_collect,observed_insert
    def fail(*args,**kw):
        raise AssertionError('Frozen sampling attempted PPO')
    learner.compute=learner.train=fail
    learner.save_policy=lambda run_dir:None
    with patch('buoy_network.runner.PacketMetrics',Metrics):
        d.Runner([d.Agent(b,learner) for b in env.buoys],env).run()
    after={k:d.tensor_digest(getattr(learner.policy,k).state_dict()) for k in before}
    if before!=after or learner.profile['ppo_updates']!=0:
        raise AssertionError('Sampling changed model')
    errors,targets,values=[],[],[]
    for episode in range(config['num_episodes']):
        rows=[r for r in observations if r['episode']==episode]
        if [r['step'] for r in rows]!=list(range(config['episode_length'])):
            raise ValueError('Incomplete task for Monte Carlo return')
        total=0.
        for row in reversed(rows):
            total=row['reward']+learner.buffer.gamma*total
            row['mc_return']=total
            for i,active in enumerate(row['active']):
                if active:
                    errors.append(row['value'][i]-total)
                    values.append(row['value'][i])
                    targets.append(total)
    report={'success':True,'mode':config.get('tg_critic_time_mode','disabled'),'decode':'sample',
        'scope':'two frozen on-policy sampled development episodes; correlated/noisy MC targets, no formal performance claim',
        'checkpoint_sha256':sha(checkpoint),'weights_unchanged':True,'optimizer_updates':0,
        'physical':context['rows'],'active_decisions':len(errors),'value_mc_bias':float(np.mean(errors)),
        'value_mc_rmse':float(np.sqrt(np.mean(np.square(errors)))),'value_mean':float(np.mean(values)),
        'mc_mean':float(np.mean(targets)),'observations':observations,'script_sha256':sha(Path(__file__))}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('PASS frozen sample '+report['mode']+' MC RMSE '+str(report['value_mc_rmse']),flush=True)


if __name__=='__main__':
    main()
