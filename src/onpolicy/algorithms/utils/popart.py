import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


class PopArt(torch.nn.Module):
    """
    Preserving Outputs Precisely while Adaptively Rescaling Targets
    https://deepmind.com/blog/article/preserving-outputs-precisely-while-adaptively-rescaling-targets
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        norm_axes=1,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device=torch.device("cpu"),
    ):
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(
            **self.tpdv
        )
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)

        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(
            **self.tpdv
        )
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()
        self.stddev.fill_(0.1)  # Same floor as the effective debiased variance below.

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        if input_vector.numel() == 0 or not torch.isfinite(input_vector).all():
            raise ValueError("PopArt requires nonempty finite targets")
        old_mean, old_var = self.debiased_mean_var()
        old_mean, old_stddev = old_mean.clone(), old_var.sqrt().clone()

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        new_mean, new_var = self.debiased_mean_var()
        new_stddev = new_var.sqrt()
        # Keep optimizer references and requires_grad flags. Never replace Parameters.
        self.weight.mul_((old_stddev/new_stddev).unsqueeze(-1))
        self.bias.copy_((old_stddev*self.bias+old_mean-new_mean)/new_stddev)
        self.stddev.copy_(new_stddev)

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

        return out


    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )

        out = out.cpu().numpy()

        return out


@contextmanager
def popart_rollout(trainer, buffer):
    """Call AFTER old-scale GAE/advantages, BEFORE any critic forward/backward.

    Fix statistics for every epoch/minibatch in one rollout. Cached values remain
    in their collection coordinates; remap_popart_values aligns value clipping.
    """
    if not trainer._use_popart:
        yield
        return
    if getattr(trainer, "_popart_old_stats", None) is not None:
        raise ValueError("Nested PopArt rollout normalization")
    normalizer = trainer.value_normalizer
    mean, var = normalizer.debiased_mean_var()
    snapshot = (mean.detach().clone(), var.detach().sqrt().clone())
    targets = buffer.returns[:-1]
    if trainer._use_value_active_masks:
        targets = targets[buffer.active_masks[:-1, ..., 0] > 0]
    else:
        targets = targets.reshape(-1, targets.shape[-1])
    if targets.size:
        normalizer.update(targets)
    trainer._popart_old_stats = snapshot
    try:
        yield
    finally:
        trainer._popart_old_stats = None


def remap_popart_values(trainer, value_preds):
    if not trainer._use_popart:
        return value_preds
    snapshot = getattr(trainer, "_popart_old_stats", None)
    if snapshot is None:
        raise ValueError("PopArt ppo_update requires normalization prepared by train")
    old_mean, old_std = snapshot
    mean, var = trainer.value_normalizer.debiased_mean_var()
    return (value_preds*old_std+old_mean-mean)/var.sqrt()
