"""Continuous native TG training, RNG-neutral periodic checkpoints and isolated evaluations."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
from unittest.mock import patch
from tg_convergence import read,atomic,require,digest,setarg,checkpoint_grid,learner_fingerprint,assert_reference,audit_frozen

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--study-job',required=True);special,argv=p.parse_known_args()
    job=read(special.study_job);root=Path(job['root']);spec=read(root/'protocol.json');snapshot=root/'source_snapshot'
    sys.path[:0]=[str(snapshot/'scripts'),str(snapshot),str(snapshot/'src')]
    five=spec.get('schema')=='tg_five_convergence_v1'
    if five:os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    if five:
        torch.set_num_threads(spec['torch_threads'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    import pm_value_worker
    from buoy_network.agent import DeepQNetwork
    from buoy_network.policy_artifacts import save_policy
    out=Path(job['output']);runroot=out.parent;folder=Path(job['work_dir'])/'convergence';folder.mkdir(parents=True,exist_ok=False)
    points=checkpoint_grid(spec['episodes'],spec['interval'])
    state={'status':'running','group':job['group'],'seed':job['seed'],'trained_episodes':0,'points':[],
           'worker_sha256':digest(Path(__file__)),'training_resume_supported':False}
    started=time.perf_counter();excluded=0.;capture_io=0.;observed=set()
    warmup,save=DeepQNetwork.warmup,DeepQNetwork.save_policy
    fingerprint=lambda model:learner_fingerprint(model,auxiliaries=five)
    def check_reference(model,n):
        if job['reference_episodes'] is not None and n==job['reference_episodes']:
            assert_reference(model,job['reference_output'])
            expected=(Path(job['reference_output'])/'logs/TG_physical_metrics.jsonl').read_bytes()
            require((out/'logs/TG_physical_metrics.jsonl').read_bytes()==expected,'Reference prefix physical trajectory changed')
            state['reference_verified']=True
    def capture(model,n):
        nonlocal excluded,capture_io
        if n not in points or n in observed:return
        before=fingerprint(model);t0=time.perf_counter()
        cp=folder/('e%04d'%n);(cp/'logs').mkdir(parents=True)
        save_policy(model,cp)
        checkpoint=cp/'checkpoints/tg_policy_latest.pt'
        check_reference(model,n)
        capture_io+=time.perf_counter()-t0
        point={'episode':n,'training_steps':n*spec['horizon'],
               'training_seconds':time.perf_counter()-started-excluded-capture_io,
               'collect_seconds':model.profile['collect_seconds'],'ppo_seconds':model.profile['ppo_seconds'],
               'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
               'actor_optimizer_steps':int(model.training_records[-1]['actor_optimizer_steps']) if n else 0,
               'critic_optimizer_steps':int(model.training_records[-1]['critic_head_optimizer_steps']) if n else 0,
               'torch_threads':torch.get_num_threads(),'reference_exact':n==job['reference_episodes'],'evaluations':[]}
        require(point['torch_threads']==spec['torch_threads'],'CPU thread count changed')
        if five and n==0:
            from tg_five import comparable_initial
            point['initial_components']=comparable_initial(model)
        if five:
            require(all(p.is_cuda for network in (model.policy.actor,model.policy.critic) for p in network.parameters()),'Policy parameter left on CPU')
            point['compute_settings']={'deterministic':torch.are_deterministic_algorithms_enabled(),
                'cudnn_deterministic':torch.backends.cudnn.deterministic,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32':torch.backends.cudnn.allow_tf32,'amp':model.trainer.scaler.is_enabled(),
                'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
        for scene in spec['scenes']:
            cmd=list(job['eval_command'])
            for key,val in [('--num_episodes',spec['eval_episodes']),('--num_env_steps',spec['eval_episodes']*spec['horizon']),
                            ('--seed',scene),('--scenario_seed',scene),('--tg_policy_checkpoint',checkpoint),
                            ('--tg_evaluate','true'),('--output_root',cp/('scene_'+str(scene)))]:
                setarg(cmd,key,val)
            logfile=cp/('eval_'+str(scene)+'.log');begin=time.perf_counter()
            with logfile.open('xb') as stream:
                proc=subprocess.run(cmd,cwd=str(snapshot),stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            excluded+=time.perf_counter()-begin
            require(proc.returncode==0,'Frozen evaluation failed; see '+str(logfile))
            evout=cp/('scene_'+str(scene))/'runs/case'/('moves_'+str(spec['moving_buoys']))
            row=audit_frozen(evout,checkpoint,spec,job['group']);row['scene']=scene;row['command']=cmd;point['evaluations'].append(row)
        after=fingerprint(model);require(before==after,'Checkpoint/evaluation changed parent weights/optimizer/RNG/history')
        point.update(parent_fingerprints_unchanged=True,parent_fingerprints=before,complete=True)
        observed.add(n);state['points'].append(point)
        state.update(trained_episodes=n,excluded_evaluation_seconds=excluded,checkpoint_io_seconds=capture_io)
        atomic(folder/'progress.json',state)
        print('CHECKPOINT {} seed{} e{} delivery {:.4f}'.format(job['group'],job['seed'],n,
            sum(x['delivery'] for x in point['evaluations'])/len(point['evaluations'])),flush=True)
        begin=time.perf_counter()
        result=subprocess.run([sys.executable]+(['-s'] if five else [])+[str(snapshot/'scripts/analyze_tg_convergence.py'),str(root),'--partial'],
                              cwd=str(snapshot),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        excluded+=time.perf_counter()-begin
        require(result.returncode==0,'Partial convergence report failed: '+result.stdout.decode(errors='replace')[-1200:])
    def observed_warmup(self,*args,**kwargs):
        r=warmup(self,*args,**kwargs);n=self.profile['ppo_updates']
        if five:check_reference(self,n)
        state['trained_episodes']=n;atomic(folder/'progress.json',state)
        capture(self,n);return r
    def observed_save(self,run_dir):
        result=save(self,run_dir);capture(self,self.profile['ppo_updates']);return result
    try:
        atomic(folder/'progress.json',state)
        with patch.object(DeepQNetwork,'warmup',observed_warmup),patch.object(DeepQNetwork,'save_policy',observed_save):
            sys.argv=[sys.argv[0]]+argv
            pm_value_worker.main()
        require(observed==set(points),'Missing checkpoints')
        state.update(status='completed',trained_episodes=spec['episodes'],excluded_evaluation_seconds=excluded,
                     checkpoint_io_seconds=capture_io,training_seconds=time.perf_counter()-started-excluded-capture_io)
        atomic(folder/'progress.json',state);print('WORKER COMPLETED',flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));atomic(folder/'progress.json',state);raise
if __name__=='__main__':main()
