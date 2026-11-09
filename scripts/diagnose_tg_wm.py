"""Read-only replay of archived WM policies: online errors and feature interventions."""
import argparse,json,os,subprocess,sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,r):Path(p).write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')

def case(spec):
    c=read(spec);snapshot=Path(c['snapshot']);sys.path[:0]=[str(snapshot),str(snapshot/'src')]
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,torch,inspect
    from unittest.mock import patch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import run_algorithm as launcher
    from buoy_network.main import main
    from buoy_network.agent import DeepQNetwork
    from buoy_network.wm_runtime import FrozenWMRuntime
    from onpolicy.algorithms.graph_actor_critic import GR_Actor
    from onpolicy.algorithms.utils.distributions import FixedCategorical
    init,collect,insert=DeepQNetwork.__init__,DeepQNetwork.collect,DeepQNetwork.insert
    distribution,attempt=GR_Actor._wm_distribution,FrozenWMRuntime.record_attempt
    has_obs='obs' in inspect.signature(distribution).parameters
    owner={};ack=[];queue=[];decisions=[];raw_features=[]
    def initialized(self,*a,**kw):init(self,*a,**kw);owner['model']=self
    @torch.no_grad()
    def measured_distribution(self,features,available,wm_features,obs=None):
        def evaluate(values):return distribution(self,features,available,values,obs) if has_obs else distribution(self,features,available,values)
        result=evaluate(wm_features);model=owner['model']
        b=model.buffer;t=model._diag_step;mask=available>0
        raw=torch.as_tensor(wm_features,dtype=features.dtype,device=features.device)
        base=self.act.action_out.linear(features).masked_fill(~mask,float('-inf'))
        base_dist=FixedCategorical(logits=base)
        fused=raw
        if getattr(self,'wm_actor_input','raw')=='residual_scaled':
            from onpolicy.algorithms.utils.wm_features import incremental_features
            fused=incremental_features(raw,obs,available,self.wm_input_scale,self.wm_input_center)
        residual=self.wm_fusion(torch.cat((features[:,None,:].expand(-1,self.wm_action_count,-1),fused),-1)).squeeze(-1)
        perm=raw.clone()
        for node in range(len(raw)):
            legal=torch.where(mask[node,:-1])[0]
            if len(legal)>1:perm[node,legal]=raw[node,legal.roll(1)]
        shuffled=evaluate(perm)
        observed=raw.clone();observed[...,1:]=0
        observed[:,:-1,0]=torch.as_tensor(b.obs[t,0,:,:-1],device=features.device);observed[:,-1,0]=0
        observed_dist=evaluate(observed)
        for node in np.flatnonzero(b.active_masks[t,0,:,0]>0):
            legal=mask[node];q=base[node,legal];d=residual[node,legal]
            row=dict(step=t,node=int(node),base_logit_std=float(q.std(unbiased=False)),residual_logit_std=float(d.std(unbiased=False)),
                     tv_vs_no_fusion=float((result.probs[node]-base_dist.probs[node]).abs().sum()/2),
                     tv_vs_shuffled=float((result.probs[node]-shuffled.probs[node]).abs().sum()/2),
                     tv_vs_observed_only=float((result.probs[node]-observed_dist.probs[node]).abs().sum()/2),
                     argmax_changed=int(result.probs[node].argmax()!=base_dist.probs[node].argmax()))
            decisions.append(row)
            raw_features.extend(raw[node,:-1][mask[node,:-1]].cpu().tolist())
        return result
    def collected(self,step):
        self._diag_step=step;result=collect(self,step);b=self.buffer
        self._diag=dict(pred=b.wm_features[step,0].copy(),obs=b.obs[step,0].copy(),nodes=b.node_obs[step,0,0].copy(),
                        active=b.active_masks[step,0,:,0].copy(),actions=result[1][0,:,0].astype(int),attempts={})
        return result
    def attempted(self,sender,receiver,value):
        attempt(self,sender,receiver,value);d=owner['model']._diag;d['attempts'][sender]=(receiver,bool(value))
    def inserted(self,data):
        d=self._diag;following=np.asarray(data[2])[0,0];n=self.env.buoy_number
        for node in np.flatnonzero(d['active']>0):
            action=d['actions'][node];target=node if action==n else action;pred=d['pred'][node,action]
            actual=[following[node,2]-d['nodes'][node,2],following[target,2]-d['nodes'][target,2]]
            queue.append([float(pred[1]),float(actual[0]),0])
            if action!=n:queue.append([float(pred[2]),float(actual[1]),1])
        for node,(action,value) in d['attempts'].items():
            ack.append([float(d['pred'][node,action,0]),float(d['obs'][node,action]),float(value)])
        return insert(self,data)
    known,extra=launcher.build_parser().parse_known_args(c['arguments'])
    output=Path(launcher.create_run_root(known.output_root,known.run_name))
    with patch.object(DeepQNetwork,'__init__',initialized),patch.object(DeepQNetwork,'collect',collected),patch.object(DeepQNetwork,'insert',inserted),patch.object(GR_Actor,'_wm_distribution',measured_distribution),patch.object(FrozenWMRuntime,'record_attempt',attempted):
        main(launcher.with_tg_mappo_defaults(extra+['--output_root',str(output)]))
    a=np.asarray(ack);q=np.asarray(queue);f=np.asarray(raw_features)
    report=dict(seed=c['seed'],scene=c['scene'],scope='Known-scene same-policy read-only replay; interventions are same-observation diagnostics, not counterfactual rollout gains',
        ack_attempts=len(a),ack_brier_model=float(np.mean((a[:,0]-a[:,2])**2)),ack_brier_observation=float(np.mean((a[:,1]-a[:,2])**2)),
        queue_samples=len(q),queue_mse_model=float(np.mean((q[:,0]-q[:,1])**2)),queue_mse_zero=float(np.mean(q[:,1]**2)),
        queue_predicted_rms=float(np.mean(q[:,0]**2)**.5),queue_actual_rms=float(np.mean(q[:,1]**2)**.5),
        forwarding_feature_std=f.std(0).tolist(),forwarding_feature_mean=f.mean(0).tolist(),active_decisions=len(decisions),
        decisions={key:float(np.mean([r[key] for r in decisions])) for key in decisions[0] if key not in ('step','node')},
        physical_replay_exact=(output/'moves_5/logs/TG_physical_metrics.jsonl').read_bytes()==(Path(c['reference_output'])/'logs/TG_physical_metrics.jsonl').read_bytes(),
        wm_queue_training_scale=owner['model'].wm_runtime.model.queue_scale.cpu().tolist())
    assert report['physical_replay_exact'],'Diagnostics changed original trajectory'
    write(Path(c['output_root'])/'diagnosis.json',report)
    with (Path(c['output_root'])/'decision_signals.jsonl').open('w',encoding='utf-8') as stream:
        for row in decisions:stream.write(json.dumps(row)+'\n')
    print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-spec');p.add_argument('--study',default='results/tg_five_gpu300/20261108_140256_956180');a=p.parse_args()
    if a.case_spec:return case(a.case_spec)
    study=(ROOT/a.study).resolve();m=read(study/'manifest.json');assert m['status']=='completed'
    out=ROOT/'results/tg_wm_diagnosis'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    reports=[]
    for seed in (11,22,33):
        j=next(j for j in m['jobs'] if j['group']=='WM' and j['seed']==seed);state=read(Path(j['work_dir'])/'convergence/progress.json')
        ev=next(e for e in state['points'][-1]['evaluations'] if e['scene']==8001);args=list(ev['command'][ev['command'].index('--algorithm'):])
        target=out/('WM_'+str(seed));target.mkdir();args[args.index('--output_root')+1]=str(target)
        spec=dict(seed=seed,scene=8001,snapshot=str(study/'source_snapshot'),arguments=args,output_root=str(target),reference_output=ev['output']);write(target/'case.json',spec)
        with (target/'worker.log').open('wb') as f:r=subprocess.run([sys.executable,'-s','-u',str(Path(__file__).resolve()),'--case-spec',str(target/'case.json')],stdout=f,stderr=subprocess.STDOUT,timeout=600)
        if r.returncode:raise RuntimeError('Diagnostic failed: '+str(target/'worker.log'))
        reports.append(read(target/'diagnosis.json'));print('DONE WM_'+str(seed),flush=True)
    write(out/'report.json',dict(status='completed',source_study=str(study),cases=reports))
    print('OUTPUT '+str(out),flush=True)

if __name__=='__main__':main()
