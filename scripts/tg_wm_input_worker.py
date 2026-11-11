"""Continuous repaired-WM training with RNG/history-neutral checkpoint evaluations."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from tg_convergence import read,atomic,setarg,digest,learner_fingerprint,assert_reference
from run_tg_wm_input_probe import audit
import run_algorithm as launcher
from buoy_network.main import main
from buoy_network.agent import DeepQNetwork
from buoy_network.policy_artifacts import save_policy

def entry():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--job-spec',required=True);a,args=p.parse_known_args()
    j=read(a.job_spec);work=Path(j['work']);m=read(work.parent/'manifest.json');folder=work/'convergence';folder.mkdir()
    points=list(range(0,j['episodes']+1,20));seen=set();state=dict(status='running',trained_episodes=0,points=[])
    warmup,save=DeepQNetwork.warmup,DeepQNetwork.save_policy
    def capture(model,n):
        if n not in points or n in seen:return
        before=learner_fingerprint(model,auxiliaries=True)
        cp=folder/('e%04d'%n);(cp/'logs').mkdir(parents=True);save_policy(model,cp)
        if n==20:
            ref=Path(j['prefix20_reference']);assert_reference(model,ref)
            physical=Path(j['work'])/'train/runs/case/moves_5/logs/TG_physical_metrics.jsonl'
            assert physical.read_bytes()==(ref/'logs/TG_physical_metrics.jsonl').read_bytes()
            state['prefix20_exact']=True
        row=dict(episode=n,checkpoint=str(cp/'checkpoints/tg_policy_latest.pt'),evaluations=[])
        for scene in m['scenes']:
            cmd=list(j['command'])
            for key,value in [('--num_episodes',2),('--num_env_steps',160),('--seed',scene),('--scenario_seed',scene),('--output_root',cp/('scene_'+str(scene))),('--tg_evaluate','true'),('--tg_policy_checkpoint',row['checkpoint'])]:setarg(cmd,key,value)
            with (cp/('eval_'+str(scene)+'.log')).open('wb') as stream:r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
            assert r.returncode==0,'Frozen checkpoint evaluation failed: '+str(cp)
            out=cp/('scene_'+str(scene))/'runs/case/moves_5'
            row['evaluations'].append(dict(scene=scene,**audit(out,2,j['mode'],m['calibration_sha256'],True,fusion=j.get('fusion','shared'))))
        assert before==learner_fingerprint(model,auxiliaries=True),'Checkpoint/evaluation mutated parent'
        row['parent_unchanged']=True;row['checkpoint_sha256']=digest(row['checkpoint']);state['points'].append(row);seen.add(n)
        atomic(folder/'progress.json',state);print('CHECKPOINT '+str(n),flush=True)
    def warmed(self,*args,**kwargs):
        result=warmup(self,*args,**kwargs);state['trained_episodes']=self.profile['ppo_updates'];atomic(folder/'progress.json',state);capture(self,self.profile['ppo_updates']);return result
    def saved(self,*args,**kwargs):
        result=save(self,*args,**kwargs);capture(self,self.profile['ppo_updates']);return result
    known,extra=launcher.build_parser().parse_known_args(args);out=launcher.create_run_root(known.output_root,known.run_name)
    try:
        with patch.object(DeepQNetwork,'warmup',warmed),patch.object(DeepQNetwork,'save_policy',saved):main(launcher.with_tg_mappo_defaults(extra+['--output_root',str(out)]))
        assert seen==set(points) and state.get('prefix20_exact');state.update(status='completed',trained_episodes=j['episodes']);atomic(folder/'progress.json',state)
    except Exception as e:
        state.update(status='failed',error=repr(e));atomic(folder/'progress.json',state);raise

if __name__=='__main__':entry()
