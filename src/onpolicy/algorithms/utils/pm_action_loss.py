"""Frozen PM action-target auxiliary objective, separate from team PPO/GAE."""
import math
import torch


def action_credit_settings(args):
    mode = getattr(args, 'pm_action_credit', 'disabled')
    coef = float(getattr(args, 'pm_action_credit_coef', .1))
    temperature = float(getattr(args, 'pm_action_temperature', .2))
    if mode not in ('disabled', 'zero_model', 'learned', 'zero_regret', 'learned_regret') or not math.isfinite(coef) or coef < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid PM action credit settings')
    if mode != 'disabled' and (not getattr(args, 'use_pm_reward', False)
            or getattr(args, 'pm_potential_mode', 'remaining') != 'task_return'
            or getattr(args, 'pm_value_mode', 'learned_shaped') not in ('learned_shaped','base_residual')
            or getattr(args, 'use_recurrent_policy', False) or getattr(args, 'use_naive_recurrent_policy', False)):
        raise ValueError('PM action credit requires feedforward task_return PM with declared value coordinates')
    return {'mode': mode, 'coefficient': coef, 'temperature': temperature}


def teacher_loss(logits, targets, active_masks=None, objective='cross_entropy'):
    target = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device).detach()
    if (target.shape != logits.shape or logits.ndim != 2 or not torch.isfinite(target).all()
            or (target < 0).any() or not torch.allclose(target.sum(-1), torch.ones_like(target[:, 0]), atol=1e-5, rtol=0)
            or not torch.isfinite(logits[target > 0]).all()):
        raise ValueError('Invalid cached PM target distribution')
    if objective=='cross_entropy':
        safe = logits.masked_fill(target == 0, 0.)
        row = -(target*safe).sum(-1, keepdim=True)
    elif objective=='expected_regret':
        log_target=target.clamp_min(1e-30).log().masked_fill(target<=0,float('-inf'))
        regret=(log_target.max(-1,keepdim=True)[0]-log_target).masked_fill(target<=0,0.).detach()
        row=(logits.exp()*regret).sum(-1,keepdim=True)
    else:
        raise ValueError('Unknown PM action objective')
    if active_masks is None:
        return row.mean()
    mask = torch.as_tensor(active_masks, dtype=logits.dtype, device=logits.device).detach()
    if mask.shape != row.shape or not torch.isfinite(mask).all() or not ((mask == 0)|(mask == 1)).all():
        raise ValueError('Invalid PM target active mask')
    return (row*mask).sum()/mask.sum().clamp(min=1.)
