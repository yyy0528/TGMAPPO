"""Bounded seed33 20-episode probe from already-tested immutable clock suites."""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for mode in ('disabled','constant','remaining'):
        p.add_argument('--'+mode+'-suite',required=True)
    p.add_argument('--output-root',default='results/tg_critic_time_probe')
    a=p.parse_args()
    output=Path(a.output_root).resolve()/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    manifest={'scope':'seed33 only; clock feasibility, not convergence or multi-seed performance',
              'train_seed':33,'train_episodes':20,'episode_length':80,'modes':{},'state':'running'}
    print('Probe: '+str(output),flush=True)
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    for mode in ('disabled','constant','remaining'):
        suite=Path(getattr(a,mode+'_suite')).resolve()
        source=read(suite/'manifest.json')
        if source['state']!='completed' or source['config'].get('critic_time_mode')!=mode:
            raise ValueError('First finish the three-seed smoke suite for '+mode)
        jobs={j['kind']:j for j in source['jobs'] if j['train_seed']==33}
        item={'source_suite':str(suite),'state':'running','jobs':[]}
        manifest['modes'][mode]=item
        save()
        for kind in ('training','evaluation'):
            name=mode+'_seed33_e20'
            command=list(jobs[kind]['base_command'])
            if kind=='training':
                for key,value in (('--num_episodes','20'),('--num_env_steps','1600')):
                    command[command.index(key)+1]=value
            run_root=output/kind
            command+=['--output_root',str(run_root),'--run-name',name]
            if kind=='evaluation':
                command+=['--tg_policy_checkpoint',str(output/'training/runs'/name/'moves_5/checkpoints/tg_policy_latest.pt')]
            run_output=run_root/'runs'/name/'moves_5'
            entry={'kind':kind,'command':command,'output':str(run_output),'exit_code':None}
            item['jobs'].append(entry)
            save()
            print('START '+mode+' '+kind,flush=True)
            started=time.perf_counter()
            with (output/(mode+'_'+kind+'.log')).open('xb') as stream:
                result=subprocess.run(command,cwd=str(suite/'source_snapshot'),stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
            entry.update(exit_code=result.returncode,seconds=time.perf_counter()-started)
            save()
            if result.returncode:
                raise RuntimeError('Probe failed; preserved '+str(output/(mode+'_'+kind+'.log')))
            print('DONE '+mode+' '+kind,flush=True)
        item['state']='completed'
        save()
    manifest['state']='completed'
    save()
    print('COMPLETE '+str(output),flush=True)


if __name__=='__main__':
    main()
