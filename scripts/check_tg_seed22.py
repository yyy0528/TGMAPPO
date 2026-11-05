"""Six read-only seed22 replays; no policy training or code/model changes."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from buoy_network.ablation_suite import read,atomic,digest,require


def main():
    reference=ROOT/'results/tg_pm_task_return/pilot_20261105_140555_016580'
    low=ROOT/'results/tg_pm_strength/20261105_154758_858806'
    before,after=read(reference/'manifest.json'),read(low/'manifest.json')
    require(before['status']==after['status']=='completed','References incomplete')
    for path,sha in before['source']['files_sha256'].items():
        require(digest(reference/'source_snapshot'/path)==digest(low/'source_snapshot'/path)==sha,'Archived source changed')
    root=ROOT/'results/tg_seed22_signals'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True,exist_ok=False);view=root/'view';view.mkdir()
    jobs=[dict(j,train_seed=22) for j in before['jobs'] if j['seed']==22 and j['group'] in ('A','task_return')]
    jobs += [dict(j,group='alpha01',train_seed=22) for j in after['jobs'] if j['seed']==22]
    require(len(jobs)==12,'Incomplete seed22 reference jobs')
    atomic(view/'manifest.json',{'source':before['source'],'jobs':jobs,
        'scope':'read-only schema view; original manifests and models untouched'})
    m={'root':str(root),'status':'running','source_snapshot':str(reference/'source_snapshot'),
       'reference_manifests':{str(reference/'manifest.json'):digest(reference/'manifest.json'),str(low/'manifest.json'):digest(low/'manifest.json')},
       'jobs':[],'scope':'frozen seed22 actions and recomputed advantages, not new training'}
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True)
    try:
        for group in ('A','task_return','alpha01'):
            for decode in ('greedy','sample'):
                tag=group+'_'+decode;out=root/tag
                command=[sys.executable,str(ROOT/'scripts/diagnose_tg_seed22_signals.py'),
                    '--snapshot',str(reference/'source_snapshot'),'--view',str(view),'--group',group,'--decode',decode,'--output',str(out)]
                job={'tag':tag,'group':group,'decode':decode,'command':command,'output':str(out),'state':'running'}
                m['jobs'].append(job);atomic(root/'manifest.json',m);print('START '+tag,flush=True)
                with (root/(tag+'.log')).open('xb') as stream:
                    p=subprocess.run(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=300)
                job['exit_code']=p.returncode;require(p.returncode==0,'Diagnostic failed: '+tag)
                report=read(out/'report.json');signals=read(out/'signals.json')
                require(report['weights_unchanged'] and report['optimizer_updates']==0 and signals['optimizer_updates']==0,'Read-only invariant failed')
                if decode=='greedy':require(report['pilot_metrics_exact_match'],'Greedy physics mismatch')
                job.update(state='completed',report_sha256=digest(out/'report.json'),signals_sha256=digest(out/'signals.json'))
                atomic(root/'manifest.json',m)
                a=report['analysis'];print('DONE {} delivery={}/{} wait={:.2%} loops={}'.format(tag,a['delivered'],a['generated'],a['active_wait_fraction'],a['revisit_transfers']),flush=True)
        m['status']='completed'
    except Exception as exc:
        m['status']='failed';m['error']=repr(exc)
        if m['jobs'] and m['jobs'][-1]['state']=='running':m['jobs'][-1]['state']='failed'
        atomic(root/'manifest.json',m);raise
    atomic(root/'manifest.json',m);print('COMPLETED '+str(root/'manifest.json'),flush=True)


if __name__=='__main__':main()
