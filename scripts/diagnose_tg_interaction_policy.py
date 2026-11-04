"""Read-only old20/new100 policy views through the archived TG diagnosis tool."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference_view(manifest,old,seed,episodes):
    view=copy.deepcopy(manifest)
    if episodes==20:
        ref=next(r for r in old['policies'] if r['seed']==seed)
        for job in view['jobs']:
            if job['train_seed']==seed:
                job['output']=ref[job['kind']]
    else:
        if not all(j['state']=='completed' for j in view['jobs'] if j['train_seed']==seed):
            raise ValueError('First finish this seed training and frozen evaluation')
    return view


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--seed',type=int,choices=(11,22,33),required=True)
    p.add_argument('--episodes',type=int,choices=(20,100),required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    suite,output=Path(a.suite).resolve(),Path(a.output).resolve()
    manifest=read(suite/'manifest.json')
    view=reference_view(manifest,read(suite/'baseline20.json'),a.seed,a.episodes)
    for name,digest in manifest['source']['files_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Original experiment source changed: '+name)
    snapshot=suite/'source_snapshot'
    sys.path[:0]=[str(snapshot),str(snapshot/'src'),str(snapshot/'scripts')]
    import diagnose_tg_policy as d
    original=d.read
    def selected(path):
        return view if Path(path).resolve()==suite/'manifest.json' else original(path)
    # Only redirect reference paths in memory. Neither the suite nor old runs are rewritten.
    before=(suite/'manifest.json').read_bytes()
    with patch.object(d,'read',selected):
        report=d.worker(suite,output,'A',a.seed,'trained','greedy')
    if before!=(suite/'manifest.json').read_bytes():
        # A live suite can legitimately advance another seed. The adapter never writes it.
        current=read(suite/'manifest.json')
        if current['source']!=manifest['source'] or current['config']!=manifest['config']:
            raise ValueError('Experiment source/config changed during diagnosis')
    ref=next(j for j in view['jobs'] if j['kind']=='training' and j['train_seed']==a.seed)
    report.update(reference_episodes=a.episodes,reference_training_output=ref['output'],
        diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='frozen reference view only; source hash is observer implementation, original training provenance retained separately')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
