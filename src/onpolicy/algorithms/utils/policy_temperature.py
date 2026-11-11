"""A fixed on-policy categorical temperature, shared by collection and PPO."""
import math


def temperature_from_args(args):
    value = float(getattr(args, 'tg_policy_temperature', 1.0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError('TG policy temperature must be finite and positive')
    return value


def temper_distribution(distribution, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Policy temperature must be finite and positive')
    if temperature == 1.0:
        return distribution
    from .distributions import FixedCategorical
    return FixedCategorical(logits=distribution.logits / temperature)
