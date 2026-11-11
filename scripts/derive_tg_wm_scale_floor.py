"""Derive conservative WM input scales from immutable training statistics only."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]
import torch
from buoy_network.wm_runtime import load_frozen_wm
from onpolicy.algorithms.utils.wm_features import load_calibration, training_scales


def derive(calibration, model, wm_sha256, source_sha256):
    if (calibration.get('schema')!='wm_increment_train_calibration_v1'
            or calibration.get('fit_split')!='train' or calibration.get('future_labels_used') is not False
            or calibration.get('wm_sha256')!=wm_sha256 or calibration.get('samples',0)<1):
        raise ValueError('Expected matching train-input calibration with no future-label fitting')
    original = torch.tensor(calibration['scale'], dtype=torch.float32)
    mean = torch.tensor(calibration['mean'], dtype=torch.float32)
    if (original.shape!=(3,) or mean.shape!=(3,) or not torch.isfinite(original).all()
            or not torch.isfinite(mean).all() or not (original>0).all()):
        raise ValueError('Invalid source calibration')
    minimum = training_scales(model)
    result = copy.deepcopy(calibration)
    result.update(scale=torch.maximum(original,minimum).tolist(),
                  scale_rule='frozen_training_rms_floor_v1',
                  minimum_scale=minimum.tolist(),source_prediction_scale=original.tolist(),
                  source_calibration_sha256=source_sha256,
                  uses_frozen_training_target_statistics=True,
                  description='Train prediction scale floored by frozen WM ACK-prior SD and queue target RMS; center unchanged. A conservative input scale, not a calibrated uncertainty estimate.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wm',required=True)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--output-root',default='results/tg_wm_scale_floor')
    args = parser.parse_args()
    model,_ = load_frozen_wm(args.wm,25)
    wm_sha = hashlib.sha256(Path(args.wm).read_bytes()).hexdigest()
    calibration,metadata = load_calibration(args.calibration,wm_sha)
    report = derive(calibration,model,wm_sha,metadata['sha256'])
    output = Path(args.output_root).resolve()/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT '+str(output/'calibration.json'),flush=True)
    print(json.dumps(dict(mean=report['mean'],scale=report['scale'],minimum_scale=report['minimum_scale'])),flush=True)


if __name__=='__main__':main()
