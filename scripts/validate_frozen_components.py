"""Frozen WM/PM validation and first reserved202 test, without any fitting."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
TEMP=ROOT/'results/component_temp';TEMP.mkdir(parents=True,exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(TEMP)
from tg_convergence import read,atomic,digest


class FrozenForwardView:
    def __init__(self,model):
        self.model=model;self.horizon=model.horizon;self.prior_logits=model.prior_logits.detach().cpu();self.cache=[]
    def eval(self):self.model.eval();return self
    def __call__(self,features,lengths):
        import torch
        with torch.no_grad():
            result=self.model(features.to(self.model.feature_mean.device),lengths).cpu()
        self.cache.append(result);return result


def evaluate(spec):
    import torch
    from buoy_network.wm_runtime import load_frozen_wm,tensor_digest
    from buoy_network.world_model import load_split,batch,evaluate as wm_evaluate
    from buoy_network.progress_model import ProgressModel,load_packets,pack,evaluate as pm_evaluate
    from buoy_network.component_validation import calibration,survival_diagnostics
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    s=read(spec);out=Path(s['output']);out.mkdir(parents=True,exist_ok=False)
    for name in ('wm','pm'):
        if digest(s[name])!=s[name+'_sha256']:raise ValueError('Frozen model changed')
    wm,history_length=load_frozen_wm(s['wm'],25);wm.to('cuda');wm.requires_grad_(False)
    saved=torch.load(s['pm'],map_location='cpu',weights_only=True)
    pm=ProgressModel(saved['horizon'],saved['hidden']);pm.load_state_dict(saved['state_dict']);pm.to('cuda');pm.eval();pm.requires_grad_(False)
    before=dict(wm=tensor_digest(wm.state_dict()),pm=tensor_digest(pm.state_dict()))
    result=dict(status='running',splits={},test_scene=202,training_updates=0,model_selection_performed=False)
    started=time.perf_counter()
    for split in ['validation','test']:
        print('LOAD '+split,flush=True)
        data=load_split(s['dataset'],split,history_length)
        metrics=wm_evaluate(wm,data,'cuda')
        probs=[];labels=[]
        with torch.no_grad():
            for idx in torch.arange(len(data['length'])).split(256):
                pred=wm(batch(data,idx,'cuda'))['ack_logit'].sigmoid().cpu();mask=data['ack_mask'][idx]
                probs.append(pred[mask]);labels.append(data['ack'][idx][mask])
        metrics['ack_calibration']=calibration(torch.cat(probs),torch.cat(labels))
        metrics['motion_used_by_actor']=False
        packets=load_packets(s['dataset'],split);packed=pack(packets);view=FrozenForwardView(pm)
        pm_metrics=pm_evaluate(view,packed)
        pm_metrics.update(survival_diagnostics(torch.cat(view.cache),pm.prior_logits,packed))
        pm_metrics['scenario_seeds']=sorted({p['scenario_seed'] for p in packets})
        if pm_metrics['scenario_seeds']!=([101] if split=='validation' else [202]):raise ValueError('Unexpected evaluation split/scenes')
        pm_metrics['behavior_groups']={}
        for behavior in sorted({p['behavior'] for p in packets}):
            select=torch.tensor([p['behavior']==behavior for p in packets])
            subset={k:(v[select] if isinstance(v,torch.Tensor) else v) for k,v in packed.items()}
            subset_view=FrozenForwardView(pm);pm_metrics['behavior_groups'][behavior]=pm_evaluate(subset_view,subset)
        result['splits'][split]=dict(wm=metrics,pm=pm_metrics)
        atomic(out/'progress.json',result);print('EVALUATED '+split,flush=True)
    after=dict(wm=tensor_digest(wm.state_dict()),pm=tensor_digest(pm.state_dict()))
    if before!=after:raise ValueError('Component weights/normalization changed')
    for name in ('wm','pm'):
        if digest(s[name])!=s[name+'_sha256']:raise ValueError('Original component changed')
    result.update(status='completed',weights_unchanged=True,wall_seconds=time.perf_counter()-started,
        scope='One fixed validation scene101 and reserved test202 across3 behaviors; not many independent test scenes')
    atomic(out/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args();evaluate(a.spec)
