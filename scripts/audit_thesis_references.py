"""Read-only audit of historical models/curves before any result reuse."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from tg_convergence import read, digest, atomic


def audit(study):
    study=Path(study).resolve()
    master=read(study/'manifest.json')
    records=[]
    for model in master['reused']:
        source=Path(model['source'])
        original=read(source/'manifest.json')
        if original['status']!='completed' or digest(source/'report.json')!=original['report_sha256']:
            raise ValueError('Changed/incomplete reference '+str(source))
        if model['algorithm']=='WM':
            path=Path(model['progress'])
            if digest(path)!=model['progress_sha256']:
                raise ValueError('Selected WM progress changed')
        else:
            old=next(j for j in original['jobs'] if j['group']==model['algorithm'] and j['seed']==model['seed'])
            path=Path(old['work_dir'])/'convergence/progress.json'
            if digest(path)!=old['progress_sha256']:
                raise ValueError('Original group progress changed '+str(path))
        progress=read(path)
        if progress['status']!='completed' or progress['trained_episodes']!=300 or len(progress['points'])!=16:
            raise ValueError('Reference is not full300')
        for point in progress['points']:
            if digest(point['checkpoint'])!=point['checkpoint_sha256']:
                raise ValueError('Reference checkpoint changed '+point['checkpoint'])
            if len(point['evaluations'])!=3:
                raise ValueError('Reference evaluation scenes missing')
        records.append(dict(algorithm=model['algorithm'],seed=model['seed'],source=str(source),
                            checkpoints_verified=16,progress_sha256=digest(path)))
    if len(records)!=12:
        raise ValueError('Expected TG/Immediate/PM/selected WM, three seeds each')
    result=dict(passed=True,read_only=True,records=records,checkpoints_verified=sum(r['checkpoints_verified'] for r in records))
    atomic(study/'reference_audit.json',result)
    print(json.dumps(dict(passed=True,models=len(records),checkpoints_verified=result['checkpoints_verified'])))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study');args=p.parse_args();audit(args.study)
