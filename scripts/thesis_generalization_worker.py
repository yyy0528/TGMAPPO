"""Frozen checkpoint evaluation under a registered scenario; no training."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT)]
# Import worker first: it sets the workspace temp directory before native imports.
from thesis_experiment_worker import run
from tg_convergence import read,atomic,digest


def execute(spec):
    from buoy_network.environment import buoy_environment
    from buoy_network.scenario_evaluation import scenario_context
    s=read(spec);out=Path(s['work']);out.mkdir(parents=True,exist_ok=False)
    if digest(s['checkpoint'])!=s['checkpoint_sha256']:raise ValueError('Policy checkpoint changed')
    started=time.perf_counter()
    with scenario_context(buoy_environment,s['settings']) as audit:
        run(s['job'],dict(output=str(out/'evaluation'),scene=s['scene'],episodes=s['episodes'],checkpoint=s['checkpoint']))
    output=out/'evaluation/moves_5';progress=read(out/'evaluation/progress.json')
    if progress['status']!='completed':raise ValueError('Frozen worker incomplete')
    report=read(output/('logs/baseline_report.json' if (output/'logs/baseline_report.json').exists() else 'logs/TG_policy_report.json'))
    if report['optimizer_updates']!=0:raise ValueError('Unexpected optimizer update')
    if 'weights_unchanged' in report:
        if not report['weights_unchanged']:raise ValueError('Baseline weights changed')
    elif not report['actor_unchanged'] or not report['critic_unchanged']:
        raise ValueError('TG policy weights changed')
    if digest(s['checkpoint'])!=s['checkpoint_sha256']:raise ValueError('Original checkpoint modified')
    rows=[json.loads(line) for line in (output/'logs/TG_physical_metrics.jsonl').read_text().splitlines()]
    expected=24*s['settings']['packets_per_source']
    if len(rows)!=s['episodes'] or len(audit['initial_bursts'])!=s['episodes'] or any(r['generated_packets']!=expected for r in rows):
        raise ValueError('Wrong evaluation load/horizon budget')
    replay=None
    if s.get('reference'):
        replay=(output/'logs/TG_physical_metrics.jsonl').read_bytes()==(Path(s['reference'])/'logs/TG_physical_metrics.jsonl').read_bytes()
        if not replay:raise ValueError('Identity intervention changed archived physical trajectory')
    hops=[v for r in rows for v in r['success_hops']];delays=[v for r in rows for v in r['success_delays']]
    generated=sum(r['generated_packets'] for r in rows);delivered=sum(r['delivered_packets'] for r in rows)
    result=dict(status='completed',algorithm=s['algorithm'],seed=s['seed'],condition=s['condition'],scene=s['scene'],
        generated=generated,delivered=delivered,delivery=delivered/generated,unresolved_fraction=1-delivered/generated,
        success_hops=sum(hops)/len(hops) if hops else None,success_delay=sum(delays)/len(delays) if delays else None,
        loop_transfers=sum(r['loop_transfers'] for r in rows),ack_failures=sum(r['ack_failures'] for r in rows),
        wall_seconds=time.perf_counter()-started,physical_sha256=digest(output/'logs/TG_physical_metrics.jsonl'),
        checkpoint_sha256=s['checkpoint_sha256'],optimizer_updates=0,weights_unchanged=True,identity_replay_exact=replay)
    atomic(out/'scenario_audit.json',audit);atomic(out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args();execute(a.spec)
