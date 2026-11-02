"""Keep research-only flags from silently falling through the legacy launcher.

This module intentionally uses only the standard library and Python 3.6 syntax.
"""


def resolve_training_budget(arguments, num_episodes, episode_length, num_env_steps):
    """Resolve a single-env finite-task budget without silently discarding steps."""
    options = {item.split("=", 1)[0] for item in arguments if item.startswith("--")}
    steps_requested = "--num_env_steps" in options
    episodes_requested = "--num_episodes" in options
    if type(num_episodes) is not int or num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    if type(episode_length) is not int or episode_length < 1:
        raise ValueError("episode_length must be a positive integer")
    requested_episodes = num_episodes if episodes_requested else None
    if steps_requested:
        if type(num_env_steps) is not int or num_env_steps < 1:
            raise ValueError("num_env_steps must be a positive integer")
        if num_env_steps % episode_length:
            raise ValueError("num_env_steps must be a multiple of episode_length (complete task episodes)")
        derived_episodes = num_env_steps // episode_length
        if episodes_requested and num_episodes != derived_episodes:
            raise ValueError("Conflicting training budget: num_env_steps must equal num_episodes * episode_length")
        num_episodes = derived_episodes
    effective_steps = num_episodes * episode_length
    return {
        "num_episodes": num_episodes,
        "num_env_steps": effective_steps,
        "effective_num_env_steps": effective_steps,
        "requested_num_env_steps": num_env_steps if steps_requested else None,
        "requested_num_episodes": requested_episodes,
        "training_budget_source": ("steps_and_episodes" if episodes_requested else "steps")
        if steps_requested else "episodes",
        "training_step_unit": "joint_slot",
    }


def reject_research_options(arguments):
    research_names = {
        "env_version", "env_backend", "reward_mode", "use_earp", "use_world_model",
        "use_progress_reward", "task_horizon", "task_ttl", "rollout_length", "device",
        "dataset", "plugin", "config", "policy", "checkpoint", "policy_checkpoint", "resume",
    }
    unsupported = []
    data_options = {"wm_data_split", "wm_export_max_steps", "wm_export_max_bytes", "wm_export_max_packets"}
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        name = argument[2:].split("=", 1)[0].replace("-", "_")
        if name in research_names or (name.startswith(("wm_", "pm_")) and name not in data_options):
            unsupported.append(argument.split("=", 1)[0])
    if unsupported:
        raise ValueError(
            "Legacy TG-MAPPO does not support research options: {}. "
            "Use run_research.py; unavailable research components will report an explicit error."
            .format(", ".join(unsupported))
        )
