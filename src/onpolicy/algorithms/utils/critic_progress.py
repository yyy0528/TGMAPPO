"""Optional decision-time frozen PM context for the shaped-value Critic only."""
import copy
import numpy as np
import torch
from torch import nn


def context_mode(args):
    mode = getattr(args, 'pm_critic_context', 'disabled')
    if mode not in ('disabled', 'constant', 'potential'):
        raise ValueError('Unknown PM Critic context mode')
    if mode != 'disabled' and (not getattr(args, 'use_pm_reward', False)
            or getattr(args, 'pm_value_mode', 'learned_shaped') != 'learned_shaped'
            or getattr(args, 'use_recurrent_policy', False) or getattr(args, 'use_naive_recurrent_policy', False)):
        raise ValueError('PM Critic context requires PM-enabled feedforward learned_shaped values')
    return mode


def extend_head(head):
    """Construct before the optimizer. Add one zero weight, preserving RNG/state."""
    result = copy.deepcopy(head)
    result.weight = nn.Parameter(torch.cat((head.weight.detach(), torch.zeros_like(head.weight[:, :1])), dim=1))
    if hasattr(result, 'input_shape'):
        result.input_shape += 1
    if hasattr(result, 'in_features'):
        result.in_features += 1
    return result


def join_context(features, potential, mode):
    if potential is None:
        raise ValueError('PM Critic requires cached decision-time Phi')
    value = torch.as_tensor(potential, dtype=features.dtype, device=features.device).detach()
    if value.shape != (features.shape[0], 1) or not torch.isfinite(value).all():
        raise ValueError('Invalid PM Critic context shape/values')
    if mode == 'constant':
        value = torch.ones_like(value)
    elif mode != 'potential':
        raise ValueError('Invalid enabled PM Critic context mode')
    return torch.cat((features, value), dim=-1)


def cache_context(buffer, step, phi):
    if buffer.critic_progress is None:
        return
    if type(step) is not int or not 0 <= step <= buffer.episode_length or not np.isfinite(phi):
        raise ValueError('Invalid PM decision time/context')
    current = buffer.critic_progress[step]
    value = np.float32(phi)
    if not np.isfinite(value) or (np.isfinite(current).any() and not np.all(current == value)):
        raise ValueError('PM context changed for a cached decision')
    current.fill(value)
