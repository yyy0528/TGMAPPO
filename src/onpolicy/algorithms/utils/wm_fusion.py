"""Candidate-conditioned forecast residuals with no prediction-free capacity."""
import math
import torch


def fusion_mode(args):
    mode = getattr(args, 'wm_actor_fusion', 'shared')
    if mode not in ('shared', 'action_delta'):
        raise ValueError('Unknown WM fusion mode')
    if mode == 'action_delta' and (not getattr(args, 'use_wm_actor', False)
                                  or getattr(args, 'wm_actor_input', 'raw') != 'residual_scaled'):
        raise ValueError('Action-conditioned WM requires enabled residual_scaled inputs')
    return mode


def residual_scale(args):
    value = float(getattr(args, 'wm_actor_residual_scale', 1.0))
    if not math.isfinite(value) or value < 0:
        raise ValueError('WM residual scale must be finite and nonnegative')
    if value != 1.0 and not getattr(args, 'use_wm_actor', False):
        raise ValueError('WM residual scaling requires WM enabled')
    return value


def action_conditioned_residual(fusion, features, predictions, available, action_weights):
    if (features.ndim != 2 or predictions.ndim != 3 or predictions.shape[-1] != 3
            or predictions.shape[:2] != available.shape or features.shape[0] != predictions.shape[0]
            or action_weights.shape != (predictions.shape[1], features.shape[1])):
        raise ValueError('Invalid candidate-conditioned WM shapes')
    if not all(torch.isfinite(x).all() for x in (features, predictions, action_weights)):
        raise ValueError('Nonfinite candidate-conditioned WM inputs')
    legal = available > 0
    if not legal.any(-1).all():
        raise ValueError('No legal candidate for WM fusion')
    # Subtract a reference first: a constant legal forecast becomes EXACT zero.
    first = legal.long().argmax(-1)
    reference = predictions.gather(1, first[:, None, None].expand(-1, 1, 3))
    delta = (predictions-reference).masked_fill(~legal.unsqueeze(-1), 0.)
    center = delta.sum(1, keepdim=True)/legal.sum(-1)[:, None, None]
    contrast = (delta-center).masked_fill(~legal.unsqueeze(-1), 0.)
    # Reuse the existing action-head rows; retain the full policy gradient path.
    scale = action_weights.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    context = features[:, None, :]*(action_weights/scale)[None, :, :]
    factual = fusion(torch.cat((context, contrast), -1)).squeeze(-1)
    neutral = fusion(torch.cat((context, torch.zeros_like(contrast)), -1)).squeeze(-1)
    return factual-neutral
