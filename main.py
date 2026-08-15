import argparse
import importlib
import random

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


def parse_args():
    parser = argparse.ArgumentParser(description="Buoy network algorithm experiment")
    parser.add_argument(
        "-a", "--algorithm",
        default="VDN",
        help="Algorithm name (use --list-algorithms to view available choices)",
    )
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List available algorithms and exit",
    )
    return parser.parse_args()


def normalize_algorithm(name):
    normalized = name.strip().upper()
    return ALIASES.get(normalized, normalized)


def load_agent(algorithm):
    module_name, needs_agent_count = ALGORITHMS[algorithm]
    return importlib.import_module(module_name).Agent, needs_agent_count


if __name__ == "__main__":
    args = parse_args()
    if args.list_algorithms:
        print("Available algorithms: " + ", ".join(ALGORITHMS))
        raise SystemExit(0)

    algorithm = normalize_algorithm(args.algorithm)
    if algorithm not in ALGORITHMS:
        choices = ", ".join(ALGORITHMS)
        raise SystemExit(f"Unknown algorithm: {args.algorithm}. Available: {choices}")

    Agent, needs_agent_count = load_agent(algorithm)
    random.seed(12)

    # 浮标的格式[能力，坐标]，虽然能能量没有用到，但这个形式保留了下来
    buoys = [[10, [i, j]] for i in range(5) for j in range(5)]

    env = buoy_environment(buoys)
    env.build_env()
    agent_number = env.buoy_number
    agents = []
    buoys_obj = env.buoys

    for i in range(agent_number):

        n_actions = env.get_n_action(i)
        n_features = env.get_n_obs(i)

        agent_args = (buoys_obj[i], n_actions, n_features)
        if needs_agent_count:
            agent_args += (agent_number,)
        agent_tmp = Agent(*agent_args)
        agents.append(agent_tmp)

    runner = Runner(agents, env, algorithm)

    runner.run()


