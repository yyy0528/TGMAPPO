"""Fit Actor prediction-input normalization only on frozen WM training inputs."""
import argparse,json,hashlib,sys,os
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from buoy_network.wm_runtime import load_frozen_wm
from buoy_network.world_model import encode_inputs,batch
from buoy_network.dataset import iter_wm_samples

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wm',required=True);p.add_argument('--dataset',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,h=load_frozen_wm(a.wm,25);model.to(a.device)
    records=[encode_inputs(sample,h) for sample in iter_wm_samples(a.dataset,'train',h)]
    data={k:torch.stack([r[k] for r in records]) for k in records[0]};features=[]
    with torch.no_grad():
        for ids in torch.arange(len(records)).split(256):
            d=batch(data,ids,a.device);y=model(d);wait=d['candidate'][:,4]>0
            ack=y['ack_logit'].sigmoid().masked_fill(wait,0)-d['candidate'][:,3]
            queue=y['queue'].clone();queue[:,1].masked_fill_(wait,0)
            features.append(torch.cat((ack[:,None],queue),-1).cpu())
    x=torch.cat(features);floor=torch.tensor([.01,.001,.001]);mean=x.mean(0);scale=torch.maximum(x.std(0,unbiased=False),floor)
    report=dict(schema='wm_increment_train_calibration_v1',fit_split='train',samples=len(records),mean=mean.tolist(),scale=scale.tolist(),clip=5.,
                wm_sha256=hashlib.sha256(Path(a.wm).read_bytes()).hexdigest(),dataset_manifest_sha256=hashlib.sha256((Path(a.dataset)/'dataset_manifest.json').read_bytes()).hexdigest(),
                future_labels_used=False,description='Frozen model predictions minus observed ACK; physical queue predictions. No validation/test rows or targets used.')
    out=ROOT/'results/tg_wm_input_calibration'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    (out/'calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('OUTPUT '+str(out/'calibration.json'),flush=True);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
