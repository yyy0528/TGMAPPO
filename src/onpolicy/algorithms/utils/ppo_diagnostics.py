"""Read-only minibatch diagnostics; does not change PPO gradients or sampling."""
import math
import torch


@torch.no_grad()
def ppo_update_diagnostics(log_ratio, active_masks, clip_param, use_active_masks=True):
    if (log_ratio.ndim != 2 or active_masks.shape != (log_ratio.shape[0],1)
            or not math.isfinite(clip_param) or clip_param < 0):
        raise ValueError("Invalid PPO diagnostic shapes/clip parameter")
    selected = log_ratio.detach()[active_masks.expand_as(log_ratio) > 0] if use_active_masks else log_ratio.detach().reshape(-1)
    if not selected.numel():
        return {"approx_kl":0.,"clip_fraction":0.}
    if not torch.isfinite(selected).all():
        raise ValueError("Nonfinite active PPO log ratio")
    ratio = selected.exp()
    return {"approx_kl":float(((ratio-1)-selected).clamp(min=0).mean()),
            "clip_fraction":float(((ratio-1).abs() > clip_param).float().mean())}
