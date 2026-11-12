"""Add VDN/COMA with a two-episode gate; reuse all completed learning models."""
import argparse
import copy
from datetime import datetime
from pathlib import Path
from run_thesis_experiments import ROOT,PYTHON,SEEDS,SCENES,run
from tg_convergence import read,atomic,digest
from buoy_network.ablation_suite import snapshot


def prepare(previous):
    previous=Path(previous).resolve();old=read(previous/'manifest.json')
    if old['status']!='main_completed' or digest(previous/'report.json')!=old['report_sha256']:
        raise ValueError('Completed unchanged main study required')
    root=ROOT/'results/thesis_extra_baselines'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    source=snapshot(root)
    protocol=read(previous/'protocol.json')
    protocol.update(schema='original_env_thesis_extra_v1',new_baselines=['VDN','COMA'],
        source_study=str(previous),coma=dict(critic='Q(s, observed_i, joint_actions_except_i)',
        actor='counterfactual policy gradient, one fresh-rollout update per episode',
        critic_updates_per_episode=2,target_sync_episodes=20,td_lambda=.95,
        adaptation='Feedforward parameter-sharing original-global-observation adaptation; no tuned-optimality claim'),
        vdn=dict(mixer='sum of active-agent Q utilities',q_updates_per_episode=2))
    atomic(root/'protocol.json',protocol)
    m=dict(status='prepared',root=str(root),source=source,created=datetime.now().isoformat(),phase='preflight',
        protocol_sha256=digest(root/'protocol.json'),references={str(previous/'report.json'):digest(previous/'report.json')},
        frozen_sha256={},reused=copy.deepcopy(old['reused']),jobs=[],later_stages=old['later_stages'])
    for job in old['jobs']:
        if job['phase']!='main' or job['algorithm']=='Greedy':continue
        path=Path(job['work'])/'progress.json'
        if job['state']!='completed' or digest(path)!=job['progress_sha256']:
            raise ValueError('Changed/incomplete old model '+job['tag'])
        m['reused'].append(dict(algorithm=job['algorithm'],seed=job['seed'],points=read(path)['points'],source=str(previous),
            output=str(Path(job['work'])/'moves_5'),progress=str(path),progress_sha256=digest(path)))
    m['expected_reused_models']=len(m['reused'])
    if m['expected_reused_models']!=27:raise ValueError('Expected nine learning groups, three seeds each')
    specs=root/'specs';specs.mkdir()
    for phase,seeds,episodes in [('preflight',[22],2),('main',SEEDS,300)]:
        for group in ['VDN','COMA']:
            for seed in seeds:
                tag='{}_{}_{}'.format(phase,group,seed);work=root/'jobs'/tag
                spec=dict(algorithm=group,seed=seed,episodes=episodes,points=[0,2] if phase=='preflight' else list(range(0,301,20)),
                    scenes=[8001] if phase=='preflight' else SCENES,eval_episodes=1 if phase=='preflight' else 2,
                    work=str(work),extra_args=[])
                path=specs/(tag+'.json');atomic(path,spec)
                m['jobs'].append(dict(tag=tag,phase=phase,algorithm=group,seed=seed,state='pending',work=str(work),
                    spec=str(path),spec_sha256=digest(path),command=[PYTHON,'-s','-u',str(root/'source_snapshot/scripts/thesis_experiment_worker.py'),'--job',str(path)]))
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True);return root


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare');g.add_argument('--run');a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    else:run(a.run)
