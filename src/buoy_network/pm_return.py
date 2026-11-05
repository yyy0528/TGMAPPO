"""Convert censored delivery-time hazards into the declared task_v2 return.

This is an expectation under the frozen PM's behavior distribution, not the
current policy's true value. A delivery slot pays its waiting cost too. Window
survival is censored, but the declared finite-task unfinished cost is observable.
"""
import math
import torch
from torch.nn import functional as F


def expected_task_return(logits, remaining, horizon, gamma, contract):
    """Per unfinished packet; sum and divide by initial P for team potential.

    Offset zero predicts delivery on the *next* transition. No reward from an
    already delivered packet belongs to the future value. Terminal value is zero.
    """
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.dim() < 1 or logits.shape[-1] < 1 or not torch.isfinite(logits).all()
            or type(horizon) is not int or horizon <= 0 or not 0 < gamma <= 1
            or contract.get('version') != 'task_v2'
            or contract.get('normalization') != 'initial_packets_and_task_horizon_v1'):
        raise ValueError('Invalid PM task-return inputs/contract')
    weights = [contract.get(key) for key in ('delivery_weight', 'delay_weight', 'unfinished_weight')]
    if (any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in weights)
            or weights[0] <= 0 or weights[1] <= 0 or weights[2] < 0):
        raise ValueError('Invalid PM task-return weights')
    left = torch.as_tensor(remaining, device=logits.device)
    if (left.shape != logits.shape[:-1] or left.dtype == torch.bool
            or not torch.isfinite(left).all() or not torch.equal(left, left.long().to(left.dtype))
            or (left < 0).any() or (left > horizon).any() or (left > logits.shape[-1]).any()):
        raise ValueError('Invalid PM remaining windows')
    # Float64 avoids subtracting almost equal survival masses in long windows.
    values = logits.double()
    log_survival_after = -F.softplus(values).cumsum(-1)
    survival_before = torch.cat((torch.ones_like(values[..., :1]), log_survival_after[..., :-1].exp()), -1)
    offsets = torch.arange(values.shape[-1], device=values.device)
    valid = offsets < left.unsqueeze(-1)
    discounts = gamma ** offsets.double()
    mass = survival_before * torch.sigmoid(values) * valid
    terminal_survival = torch.where(left > 0,
        log_survival_after.gather(-1, (left.long()-1).clamp_min(0).unsqueeze(-1)).squeeze(-1).exp(),
        torch.zeros_like(left, dtype=torch.float64))
    delivery = weights[0] * (mass * discounts).sum(-1)
    delay = -weights[1] / horizon * (survival_before * valid * discounts).sum(-1)
    unfinished = -weights[2] * gamma ** (left.double()-1).clamp_min(0) * terminal_survival
    return {'task_return': delivery + delay + unfinished,
            'delivery': delivery, 'delay': delay, 'unfinished': unfinished,
            'probability': mass.sum(-1), 'terminal_survival': terminal_survival}
