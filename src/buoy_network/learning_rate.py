"""Explicit original-TG learning-rate switch; no RNG or optimizer-step changes."""
import math


def schedule_spec(args):
    enabled = getattr(args,"use_linear_lr_decay",False)
    if type(enabled) is not bool:
        raise ValueError("Learning-rate decay switch must be boolean")
    actor,critic = float(args.lr),float(args.critic_lr)
    if not all(math.isfinite(v) and v >= 0 for v in (actor,critic)):
        raise ValueError("Learning rates must be finite and nonnegative")
    return {"version":"tg_lr_switch_v1","kind":"linear" if enabled else "fixed",
            "actor_initial_lr":actor,"critic_initial_lr":critic,"interval":"episode",
            "linear_rule":"initial - initial * (zero_based_episode / total_episodes)"}


def advance_schedule(policy, args, episode, episodes, training=True):
    if type(episode) is not int or type(episodes) is not int or episodes <= 0 or not 0 <= episode < episodes:
        raise ValueError("Invalid learning-rate episode boundary")
    spec = schedule_spec(args)
    if training and spec["kind"] == "linear":
        policy.lr_decay(episode,episodes)


def optimizer_rates(policy):
    return {"actor":[float(g["lr"]) for g in policy.actor_optimizer.param_groups],
            "critic":[float(g["lr"]) for g in policy.critic_optimizer.param_groups]}
