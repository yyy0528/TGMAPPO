import argparse
import os
import subprocess
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

ALGORITHM_CHOICES = [
    "all",
    "tg-mappo",
    "mappo",
    "dqn",
    "iql",
    "qmix",
    "qmix-attn",
    "vdn",
    "rbmadq",
]

BASELINE_ALGORITHM_MAP = {
    "dqn": "DQN",
    "iql": "IQL",
    "mappo": "MAPPO",
    "qmix": "QMIX",
    "qmix-attn": "QMIX-ATTN",
    "qmix_attn": "QMIX-ATTN",
    "vdn": "VDN",
    "rbmadq": "RBMADQ",
}

RUN_ALL_ORDER = [
    "tg-mappo",
    "dqn",
    "iql",
    "mappo",
    "qmix",
    "qmix-attn",
    "vdn",
    "rbmadq",
]

# The custom TG-MAPPO buoy runner is built for this configuration.  These are
# the same defaults historically supplied by run_training.cmd.  Explicit CLI
# arguments always take precedence.
TG_MAPPO_DEFAULT_ARGS = [
    ("--project_name", "TG-MAPPO"),
    ("--env_name", "GraphMPE"),
    ("--algorithm_name", "mappo"),
    ("--seed", "0"),
    ("--experiment_name", "TG-MAPPO"),
    ("--scenario_name", "navigation_graph"),
    ("--num_agents", "25"),
    ("--collision_rew", "5"),
    ("--n_training_threads", "1"),
    ("--n_rollout_threads", "1"),
    ("--num_mini_batch", "20"),
    ("--episode_length", "80"),
    ("--num_env_steps", "200"),
    ("--ppo_epoch", "10"),
    ("--use_ReLU", None),
    ("--gain", "0.01"),
    ("--lr", "5e-4"),
    ("--critic_lr", "3e-4"),
    ("--user_name", "marl"),
    ("--use_cent_obs", "False"),
    ("--graph_feat_type", "global"),
    ("--target_mini_batch_size", "20"),
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified entry point for TG-MAPPO and baseline algorithms."
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_CHOICES,
        default="tg-mappo",
        help="Algorithm to run, or 'all' to run every algorithm sequentially",
    )
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List the algorithms supported by the unified launcher and exit.",
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        default="results",
        help="Base directory under which a new timestamped run is created",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name; defaults to a timestamp",
    )
    return parser


def add_project_paths(env):
    python_path = env.get("PYTHONPATH", "")
    paths = [SRC_DIR, ROOT_DIR]
    filtered = [p for p in paths if p and p not in python_path.split(os.pathsep)]
    if filtered:
        env["PYTHONPATH"] = os.pathsep.join(filtered + ([python_path] if python_path else []))
    # Training libraries emit expected CPU/CUDA compatibility warnings even
    # when CUDA was not requested. Keep the unified console focused on metrics;
    # actual exceptions and non-zero exits are still shown normally.
    env["PYTHONWARNINGS"] = "ignore"
    return env


def create_run_root(output_root, run_name=None):
    if run_name:
        safe_name = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in run_name
        ).strip("_")
        if not safe_name:
            raise ValueError("--run-name must contain a letter or number")
    else:
        safe_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = os.path.abspath(os.path.join(output_root, "runs", safe_name))
    os.makedirs(run_root, exist_ok=False)
    return run_root


def with_tg_mappo_defaults(extra_args):
    """Add required TG-MAPPO defaults without overriding explicit options."""
    present = set()
    for arg in extra_args:
        if arg.startswith("--"):
            present.add(arg.split("=", 1)[0])

    merged = list(extra_args)
    for option, value in TG_MAPPO_DEFAULT_ARGS:
        if option in present:
            continue
        merged.append(option)
        if value is not None:
            merged.append(value)
    return merged


def launch_tg_mappo(extra_args):
    cmd = [sys.executable, "-m", "buoy_network.main"] + with_tg_mappo_defaults(extra_args)
    env = add_project_paths(os.environ.copy())
    return subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=False).returncode


def launch_baseline(algorithm_name, extra_args):
    cmd = [
        sys.executable,
        "-m",
        "baselines.main",
        "--algorithm",
        algorithm_name,
    ] + extra_args
    env = add_project_paths(os.environ.copy())
    return subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=False).returncode


def main():
    parser = build_parser()
    known_args, extra_args = parser.parse_known_args()

    if known_args.list_algorithms:
        print("Supported algorithms: " + ", ".join(ALGORITHM_CHOICES))
        return 0

    try:
        run_root = create_run_root(known_args.output_root, known_args.run_name)
    except (OSError, ValueError) as exc:
        print("Cannot create run output directory: {}".format(exc))
        return 2
    extra_args = list(extra_args) + ["--output_root", run_root]
    print("Run output: {}".format(run_root), flush=True)

    algorithm = known_args.algorithm.lower()

    if algorithm == "all":
        for item in RUN_ALL_ORDER:
            print("\n=== Running {} ===".format(item), flush=True)
            if item == "tg-mappo":
                returncode = launch_tg_mappo(extra_args)
            else:
                returncode = launch_baseline(
                    BASELINE_ALGORITHM_MAP[item], extra_args
                )
            if returncode:
                print("{} failed with exit code {}".format(item, returncode))
                return returncode
        return 0

    if algorithm == "tg-mappo":
        return launch_tg_mappo(extra_args)

    normalized_name = BASELINE_ALGORITHM_MAP.get(algorithm, algorithm.upper())
    return launch_baseline(normalized_name, extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
