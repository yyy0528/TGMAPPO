"""Fixed train-split scaling of the WM's incremental action information."""
import torch
import hashlib,json
from pathlib import Path

MODES = ('raw', 'residual_scaled')


def wm_input_mode(args):
    mode = getattr(args, 'wm_actor_input', 'raw')
    if mode not in MODES or (mode != 'raw' and not getattr(args, 'use_wm_actor', False)):
        raise ValueError('WM input mode requires an enabled WM Actor')
    return mode


def training_scales(model):
    # Compute this tiny fixed constant on CPU for portable checkpoint identity.
    p = model.ack_prior.detach().cpu()
    queue = model.queue_scale.detach().cpu()
    scale = torch.cat(((p*(1-p)).sqrt().clamp_min(.01).reshape(1), queue))
    if scale.shape != (3,) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid WM train-split normalization')
    return scale


def load_calibration(path, wm_sha256):
    p=Path(path);data=json.loads(p.read_text(encoding='utf-8'))
    if (data.get('schema')!='wm_increment_train_calibration_v1' or data.get('fit_split')!='train'
            or data.get('wm_sha256')!=wm_sha256 or data.get('samples',0)<1):
        raise ValueError('WM calibration must match frozen model and training split')
    mean=torch.tensor(data['mean']);scale=torch.tensor(data['scale'])
    if mean.shape!=(3,) or scale.shape!=(3,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale>0).all():
        raise ValueError('Invalid WM calibration statistics')
    return data,dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fit_split='train')


def incremental_features(prediction, obs, available, scale, center=None):
    if (prediction.ndim != 3 or prediction.shape[-1] != 3 or obs.shape != prediction.shape[:2]
            or available.shape != obs.shape or scale.shape != (3,)
            or not torch.isfinite(prediction).all() or not torch.isfinite(obs).all()
            or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError('Invalid incremental WM feature tensors')
    # The final observation column is the node ID, NOT a WAIT ACK probability.
    ack = torch.cat((obs[:, :-1], torch.zeros_like(obs[:, -1:])), -1)
    residual = torch.cat(((prediction[..., 0]-ack).unsqueeze(-1), prediction[..., 1:]), -1)
    center=torch.zeros_like(scale) if center is None else center
    if center.shape!=(3,) or not torch.isfinite(center).all():raise ValueError('Invalid WM feature center')
    return ((residual-center)/scale).clamp(-5., 5.).masked_fill(~(available > 0).unsqueeze(-1), 0.)
