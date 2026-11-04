"""Watch the already-running fixed100 suite; then audit and plot its results.

Never starts/resumes training or edits source, parameters, or rewards.
Stops with a recorded failure if training or postprocessing fails.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.buoy_network.ablation_suite import read,atomic,digest,require,suite_lock


def commands(root,manifest):
    python=manifest['config']['python']
    return [('verification',[python,str(ROOT/'scripts/verify_tg_fixed100_results.py'),
                             '--suite',str(root),'--report',str(root/'fixed100_acceptance.json')]),
            ('plots',[python,str(ROOT/'scripts/plot_tg_ablation.py'),str(root)])]


def step(root):
    root=Path(root).resolve()
    manifest=read(root/'manifest.json')
    directory=root/'postprocessing'
    directory.mkdir(exist_ok=True)
    receipt=directory/'report.json'
    if receipt.exists() and read(receipt)['state']=='completed':
        result=read(receipt)
        require(digest(root/'fixed100_acceptance.json')==result['acceptance_sha256'],
                'Completed acceptance changed')
        return result
    if manifest['state']=='failed':
        result={'state':'blocked','reason':'training/evaluation task failed',
                'failed_jobs':[j['key'] for j in manifest['jobs'] if j['state']=='failed']}
        atomic(receipt,result)
        return result
    if manifest['state']!='completed':
        return {'state':'waiting','completed_jobs':sum(j['state']=='completed' for j in manifest['jobs'])}
    result={'state':'running','stages':[]}
    for name,command in commands(root,manifest):
        if name=='verification' and (root/'fixed100_acceptance.json').exists():
            require(read(root/'fixed100_acceptance.json')['complete'] is True,'Incomplete existing acceptance')
            continue
        attempt=1+len(list(directory.glob(name+'_attempt_*.log')))
        log=directory/(name+'_attempt_{}.log'.format(attempt))
        with log.open('xb') as stream:
            execution=subprocess.run(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
        result['stages'].append({'name':name,'command':command,'exit_code':execution.returncode,
                                 'log':str(log),'log_sha256':digest(log)})
        if execution.returncode!=0:
            result['state']='failed'
            atomic(receipt,result)
            return result
    acceptance=read(root/'fixed100_acceptance.json')
    require(acceptance['complete'] is True,'Full experiment was not accepted')
    result['state']='completed'
    result['acceptance_sha256']=digest(root/'fixed100_acceptance.json')
    atomic(receipt,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite')
    p.add_argument('--poll-seconds',type=int,default=30)
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    root=Path(args.suite).resolve()
    require(ROOT in root.parents and args.poll_seconds>0,'Invalid suite/poll interval')
    directory=root/'postprocessing'
    directory.mkdir(exist_ok=True)
    with suite_lock(directory):
        while True:
            result=step(root)
            if result['state']!='waiting' or args.once:
                print(json.dumps(result,indent=2),flush=True)
                return 0 if result['state'] in ('waiting','completed') else 1
            time.sleep(args.poll_seconds)


if __name__=='__main__':
    sys.exit(main())
