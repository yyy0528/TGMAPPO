"""Prepare/run fixed-parameter TG / immediate / frozen-PM 300-episode convergence study."""
import argparse,json,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tg_convergence import read,atomic,require,digest,setarg,checkpoint_grid
from buoy_network.ablation_suite import snapshot,suite_lock

def prepare(config_path):
    cfg=read(config_path)
    require(cfg['schema']=='tg_pm_convergence_v1' and cfg['groups']==['TG','Immediate','PM'],'Unsupported study schema/groups')
    require(cfg['horizon']==80 and cfg['torch_threads']==16 and cfg['eval_episodes']>=1,'Keep original task/device protocol')
    require((cfg['episodes'],cfg['reference_episodes']) in ((300,20),(2,2)),'Only full300 or bounded preflight supported')
    if cfg['episodes']==300:require(cfg['seeds']==[11,22,33] and cfg['interval']==20 and cfg['scenes']==[8001,8002,8003],'Unexpected long protocol')
    else:require(cfg['seeds']==[22] and cfg['interval']==1 and cfg['scenes']==[8001] and cfg['eval_episodes']==1,'Unexpected preflight')
    v3root=(ROOT/cfg['reference_root']).resolve();baseline=(ROOT/cfg['baseline_reference_root']).resolve()
    v3,old=read(v3root/'manifest.json'),read(baseline/'manifest.json')
    require(v3['status']=='completed' and v3['development']['passed'] and old['status']=='completed','Reference incomplete')
    for name,h in v3['source']['files_sha256'].items():
        if name.replace('\\','/').startswith('src/'):require(digest(ROOT/name)==h,'Learning source changed since v3: '+name)
    root=ROOT/cfg['output_root']/datetime.now().strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    source=snapshot(root)
    cfg.update(python=str(Path(cfg['python']).resolve()),root=str(root.resolve()),created=datetime.now().isoformat(),
               checkpoints=checkpoint_grid(cfg['episodes'],cfg['interval']),
               frozen_pm_sha256='f62f05b520cbaf665823ce38981c6c88f5e4f816dc01f85d3149aca7d3f81240')
    atomic(root/'protocol.json',cfg)
    manifest={'status':'prepared','root':str(root),'source':source,'protocol_sha256':digest(root/'protocol.json'),
              'reference_manifests':{str(v3root/'manifest.json'):digest(v3root/'manifest.json'),str(baseline/'manifest.json'):digest(baseline/'manifest.json')},
              'jobs':[],'budget':{'training_steps':len(cfg['groups'])*len(cfg['seeds'])*cfg['episodes']*80,
                'evaluation_steps':len(cfg['groups'])*len(cfg['seeds'])*len(cfg['checkpoints'])*len(cfg['scenes'])*cfg['eval_episodes']*80},
              'prefix_scope':'All new continuous runs start from initialization; stored policies are evaluation-only, not training resume.'}
    for seed in cfg['seeds']:
        for group,mode in [('TG','disabled'),('Immediate','zero_set'),('PM','learned_set')]:
            if cfg['reference_episodes']==20:
                train=(next(j for j in old['jobs'] if j['group']=='A' and j['seed']==seed and j['kind']=='training') if group=='TG' else
                       next(j for j in v3['jobs'] if j['tag']=='dev_'+mode+'_'+str(seed)+'_train'))
            else:
                tag='compat_A_22' if group=='TG' else 'smoke_'+mode+'_train'
                train=next(j for j in v3['jobs'] if j['tag']==tag)
            evaluation=(next(j for j in old['jobs'] if j['group']=='A' and j['seed']==seed and j['kind']=='evaluation' and j['scene']==8001) if group=='TG' else
                        next(j for j in v3['jobs'] if j['tag']=='dev_'+mode+'_'+str(seed)+'_eval_8001'))
            ref=Path(train['output']);rr=read(ref/'logs/TG_policy_report.json')
            require(digest(rr['checkpoint'])==rr['checkpoint_sha256'],'Reference checkpoint changed')
            tag=group+'_'+str(seed);work=root/'jobs'/tag;work.mkdir(parents=True)
            output=work/'training/runs/case/moves_5'
            training=list(train['command'][2:])
            for key,val in [('--snapshot',root/'source_snapshot'),('--num_episodes',cfg['episodes']),('--num_env_steps',cfg['episodes']*80),
                            ('--output_root',work/'training'),('--pm_action_credit',mode),('--pm_critic_context','disabled')]:setarg(training,key,val)
            ev=list(evaluation['command'][2:])
            if '--snapshot' in ev:
                i=ev.index('--snapshot');del ev[i:i+2]
            eval_cmd=[cfg['python'],'-u',str(root/'source_snapshot/run_algorithm.py')]+ev
            spec={'root':str(root),'work_dir':str(work),'group':group,'mode':mode,'seed':seed,'output':str(output),
                  'reference_episodes':cfg['reference_episodes'],'reference_output':str(ref),
                  'reference_checkpoint_sha256':rr['checkpoint_sha256'],'reference_report_sha256':digest(ref/'logs/TG_policy_report.json'),
                  'reference_physical_sha256':digest(ref/'logs/TG_physical_metrics.jsonl'),'eval_command':eval_cmd}
            atomic(work/'job.json',spec)
            cmd=[cfg['python'],'-u',str(root/'source_snapshot/scripts/tg_convergence_worker.py'),'--study-job',str(work/'job.json')]+training
            manifest['jobs'].append(dict(spec,tag=tag,command=cmd,state='pending',job_spec_sha256=digest(work/'job.json')))
    atomic(root/'manifest.json',manifest)
    print('OUTPUT '+str(root),flush=True);print(json.dumps(manifest['budget']),flush=True);return root

def run(root):
    root=Path(root).resolve()
    require(ROOT/'results' in root.parents,'Unexpected study root')
    with suite_lock(root):
        m=read(root/'manifest.json');cfg=read(root/'protocol.json')
        require(m['status'] in ('prepared','running','failed','postprocessing'),'Study already completed')
        require(digest(root/'protocol.json')==m['protocol_sha256'],'Study protocol changed')
        for name,h in m['source']['files_sha256'].items():require(digest(root/'source_snapshot'/name)==h,'Runtime snapshot changed')
        for p,h in m['reference_manifests'].items():require(digest(p)==h,'Reference manifest changed')
        m.update(status='running',controller_pid=__import__('os').getpid());atomic(root/'manifest.json',m)
        try:
            for j in m['jobs']:
                progress=Path(j['work_dir'])/'convergence/progress.json'
                if j['state']=='completed':
                    require(read(progress)['status']=='completed' and digest(Path(j['output'])/'logs/TG_policy_report.json')==j['policy_report_sha256'],'Completed result changed');continue
                require(j['state']=='pending','Failed/interrupted continuous training is not silently restarted: '+j['tag'])
                require(digest(Path(j['work_dir'])/'job.json')==j['job_spec_sha256'],'Job definition changed')
                j.update(state='running',started=datetime.now().isoformat());atomic(root/'manifest.json',m)
                print('START '+j['tag'],flush=True)
                with (Path(j['work_dir'])/'worker.log').open('xb') as stream:
                    proc=subprocess.run(j['command'],cwd=str(root/'source_snapshot'),stdout=stream,stderr=subprocess.STDOUT,timeout=cfg.get('job_timeout_seconds',14400))
                j['exit_code']=proc.returncode
                require(proc.returncode==0,'Training worker failed: '+j['tag']+'; '+str(Path(j['work_dir'])/'worker.log'))
                state=read(progress);r=read(Path(j['output'])/'logs/TG_policy_report.json')
                require(state['status']=='completed' and state['trained_episodes']==cfg['episodes']
                    and [p['episode'] for p in state['points']]==cfg['checkpoints'],'Incomplete continuous run')
                require(len(r['training_updates'])==cfg['episodes'] and r['profile']['collect_calls']==cfg['episodes']*80,'Incorrect final budget')
                j.update(state='completed',finished=datetime.now().isoformat(),policy_report_sha256=digest(Path(j['output'])/'logs/TG_policy_report.json'),
                         progress_sha256=digest(progress))
                atomic(root/'manifest.json',m);print('DONE '+j['tag'],flush=True)
            m['status']='postprocessing';atomic(root/'manifest.json',m)
            cmd=[cfg['python']]+(['-s'] if cfg.get('schema')=='tg_five_convergence_v1' else [])+[str(root/'source_snapshot/scripts/analyze_tg_convergence.py'),str(root)]
            log=root/'postprocessing.log'
            if log.exists():log=root/('postprocessing_retry_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
            m.setdefault('postprocessing_logs',[]).append(str(log));atomic(root/'manifest.json',m)
            with log.open('xb') as f:done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
            require(done.returncode==0,'Final independent analysis failed; see postprocessing.log')
            require(read(root/'report.json')['engineering_passed'],'Final engineering audit failed')
            m.update(status='completed',finished=datetime.now().isoformat(),report_sha256=digest(root/'report.json'))
            atomic(root/'manifest.json',m);print('STUDY COMPLETED '+str(root),flush=True)
        except Exception as exc:
            m.update(status='failed',error=repr(exc),failed_at=datetime.now().isoformat())
            if 'j' in locals() and j['state']=='running':j['state']='failed'
            atomic(root/'manifest.json',m);raise
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare');g.add_argument('--run')
    a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    else:run(a.run)
if __name__=='__main__':main()
