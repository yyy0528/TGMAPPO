"""Optional finite-task clock, visible only to the value function."""
import numpy as np
import torch
from torch import nn

MODES = ("disabled", "constant", "remaining")


def time_mode(args):
    mode = getattr(args,"tg_critic_time_mode","disabled")
    if mode not in MODES:
        raise ValueError("Unknown Critic time mode")
    if mode != "disabled" and (getattr(args,"use_recurrent_policy",False)
                               or getattr(args,"use_naive_recurrent_policy",False)):
        raise ValueError("Critic task clock currently requires feed-forward policies")
    return mode


def task_clock(horizon,threads,agents):
    if any(type(v) is not int or v < 1 for v in (horizon,threads,agents)):
        raise ValueError("Clock dimensions must be positive integers")
    clock = (1.-np.arange(horizon+1,dtype=np.float64)/horizon).astype(np.float32)
    return np.broadcast_to(clock[:,None,None,None],(horizon+1,threads,agents,1)).copy()


class CriticTime(nn.Module):
    """Zero-initialized feature modulation; the constant control has equal capacity."""
    def __init__(self,width,mode):
        super(CriticTime,self).__init__()
        if mode not in ("remaining","constant") or type(width) is not int or width < 1:
            raise ValueError("Invalid Critic clock module")
        self.mode = mode
        # No random initialization: preserve all original parameters and RNG state.
        self.modulation = nn.Parameter(torch.zeros(2,width))

    def forward(self,features,remaining):
        if remaining is None:
            raise ValueError("Enabled Critic clock requires decision-time values")
        clock = torch.as_tensor(remaining,dtype=features.dtype,device=features.device).detach()
        if (features.ndim != 2 or clock.shape != (features.shape[0],1)
                or features.shape[1] != self.modulation.shape[1]
                or not torch.isfinite(clock).all() or (clock < 0).any() or (clock > 1).any()):
            raise ValueError("Invalid Critic clock shape/range")
        if self.mode == "constant":
            clock = torch.ones_like(clock)
        return features*(1+clock*self.modulation[0])+clock*self.modulation[1]
