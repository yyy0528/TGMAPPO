"""Stable training entry point for the original baseline implementations."""

import argparse
import importlib
import random

import numpy as np
import torch

from src.environment import buoy_environment
from src.runner import Runner


ALGORITHMS = {
    "DQN": ("baselines.agents.agentDQN", True),
    "IQL": ("baselines.agents.agentIQL", True),
    "MAPPO": ("baselines.agents.agentMappo", True),
    "QMIX": ("baselines.agents.agentQmix", True),
    "QMIX-ATTN": ("baselines.agents.agentQmixAttn", True),
    "VDN": ("baselines.agents.agentVDN", True),
    "RBMADQ": ("baselines.agents.agent", False),
}

ALIASES = {
    "QMIX_ATTN": "QMIX-ATTN",
    "QMIXATTN": "QMIX-ATTN",
}


def normalize_algorithm(name):
    normalized = name.strip().upper()
    return ALIASES.get(normalized, normalized)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a baseline algorithm on its original buoy environment."
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        default="VDN",
        help="Baseline algorithm name (use --list-algorithms to view choices)",
    )
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List baseline algorithms and exit",
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--grid-size", "--grid_size", type=int, default=5)
    parser.add_argument("--episodes", "--num_episodes", type=int, default=1000)
    parser.add_argument("--episode-length", "--episode_length", type=int, default=80)
    parser.add_argument(
        "--num-moving-buoys", "--num_moving_buoys", type=int, default=25
    )
    parser.add_argument("--output-root", "--output_root", default="results")
    return parser


def run(
    algorithm_name,
    seed=12,
    grid_size=5,
    episodes=1000,
    episode_length=80,
    num_moving_buoys=25,
    output_root="results",
):
    algorithm = normalize_algorithm(algorithm_name)
    if algorithm not in ALGORITHMS:
        choices = ", ".join(ALGORITHMS)
        raise ValueError(
            "Unknown baseline algorithm: {}. Available: {}".format(
                algorithm_name, choices
            )
        )

    module_name, needs_agent_count = ALGORITHMS[algorithm]
    agent_class = importlib.import_module(module_name).Agent

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    buoys = [[10, [i, j]] for i in range(grid_size) for j in range(grid_size)]
    env = buoy_environment(buoys)
    env.build_env()
    agent_number = env.buoy_number
    agents = []

    for index in range(agent_number):
        n_actions = env.get_n_action(index)
        n_features = env.get_n_obs(index)
        agent_args = (env.buoys[index], n_actions, n_features)
        if needs_agent_count:
            agent_args += (agent_number,)
        agents.append(agent_class(*agent_args))

    Runner(
        agents,
        env,
        algorithm_name=algorithm,
        episodes=episodes,
        episode_length=episode_length,
        num_moving_buoys=num_moving_buoys,
        output_root=output_root,
        grid_size=grid_size,
    ).run()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list_algorithms:
        print("Available baseline algorithms: " + ", ".join(ALGORITHMS))
        return 0

    try:
        run(
            args.algorithm,
            seed=args.seed,
            grid_size=args.grid_size,
            episodes=args.episodes,
            episode_length=args.episode_length,
            num_moving_buoys=args.num_moving_buoys,
            output_root=args.output_root,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
