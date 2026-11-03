import random
import numpy as np
import torch

from onpolicy.config import get_config
from src.legacy_cli import reject_research_options, resolve_training_budget

from distutils.util import strtobool


import  sys
# 获取参数
#主要功能是解析命令行参数，并将解析后的参数返回。
def parse_args(args, parser):

    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple_spread",
        help="Which scenario to run on",
    )

    parser.add_argument("--num_landmarks", type=int, default=3)

    parser.add_argument("--num_agents", type=int, default=2, help="number of players")

    parser.add_argument(
        "--num_obstacles", type=int, default=3, help="Number of obstacles"
    )

    # 替换MPE特定参数
    parser.add_argument("--num_buoys", type=int, default=5, help="浮标数量")
    parser.add_argument("--routers", type=int, default=3, help="路由节点数量")
    parser.add_argument("--comm_range", type=float, default=1.5, help="Physical communication radius")
    parser.add_argument("--random_channel", type=lambda x: bool(strtobool(x)), default=True,
                        help="Enable configured stochastic packet loss; false uses a lossless channel")
    # 添加新环境所需参数...


    #应该是关于智能体是否协作的标志
    parser.add_argument(
        "--collaborative",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Number of agents in the env",
    )

    #智能体的最大速度。注意，如果此值为 None，则在离散动作空间中最大速度为 2
    parser.add_argument(
        "--max_speed",
        type=float,
        default=2,
        help="Max speed for agents. NOTE that if this is None, "
        "then max_speed is 2 with discrete action space",
    )

    #其他智能体和障碍物碰撞时要扣除的奖励
    parser.add_argument(
        "--collision_rew",
        type=float,
        default=5,
        help="The reward to be negated for collisions with other "
        "agents and obstacles",
    )
    #如果智能体到达目标要添加的奖励
    parser.add_argument(
        "--goal_rew",
        type=float,
        default=5,
        help="The reward to be added if agent reaches the goal",
    )

    #用于判断智能体是否到达目标的最小距离阈值
    parser.add_argument(
        "--min_dist_thresh",
        type=float,
        default=0.05,
        help="The minimum distance threshold to classify whether "
        "agent has reached the goal or not",
    )

    #当智能体到达目标时，我们是使用 done=True，
    # 还是像 simple.py 或 simple_spread.py 中那样只返回 False
    parser.add_argument(
        "--use_dones",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether we want to use the 'done=True' "
        "when agent has reached the goal or just return False like "
        "the `simple.py` or `simple_spread.py`",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=5,
        help="the size of the buoys' grid ",

    )
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--num_moving_buoys", type=int, default=25)
    parser.add_argument("--scenario_seed", type=int, default=None,
                        help="Optional simulator RNG seed; omitted preserves legacy seed 2568")
    parser.add_argument("--export_wm_data", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--use_wm_actor", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--wm_actor_checkpoint", default="")
    parser.add_argument("--use_pm_reward", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--pm_reward_checkpoint", default="")
    parser.add_argument("--pm_reward_alpha", type=float, default=1.0)
    parser.add_argument("--tg_evaluate", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--tg_policy_checkpoint", default="")
    parser.add_argument("--data_behavior", choices=("tg-mappo", "random", "geographic"), default="tg-mappo")
    parser.add_argument("--wm_data_split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--wm_export_max_steps", type=int, default=20000)
    parser.add_argument("--wm_export_max_bytes", type=int, default=268435456)
    parser.add_argument("--wm_export_max_packets", type=int, default=10000)
    parser.add_argument("--output_root", type=str, default="results")

    all_args = parser.parse_known_args(args)[0]

    return all_args, parser
#这段代码的作用是定义和解析一系列与多智能体环境相关的命令行参数，
# 以便在程序运行时可以通过命令行灵活配置环境的各种属性和奖励机制等

def main(args):
    # 这段代码的作用是定义和解析一系列与多智能体环境相关的命令行参数，
    # 以便在程序运行时可以通过命令行灵活配置环境的各种属性和奖励机制等
    parser = get_config()
    parser.allow_abbrev = False
    try:
        reject_research_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    all_args, parser = parse_args(args, parser)
    from onpolicy.config import graph_config
    all_args, parser = graph_config(args, parser)
    all_args = parser.parse_args(args)
    try:
        budget = resolve_training_budget(args, all_args.num_episodes,
                                         all_args.episode_length, all_args.num_env_steps)
    except ValueError as exc:
        parser.error(str(exc))
    for name, value in budget.items():
        setattr(all_args, name, value)
    if all_args.num_agents != all_args.grid_size ** 2:
        parser.error("num_agents must equal grid_size squared")
    if (all_args.episode_length < 1 or all_args.num_episodes < 1
            or all_args.comm_range <= 0 or all_args.n_rollout_threads != 1):
        parser.error("Require positive episode settings/comm_range and one rollout thread")
    # The buoy graph and legal routing links must have the same physical radius.
    all_args.max_edge_dist = all_args.comm_range
    if bool(all_args.wm_actor_checkpoint) != all_args.use_wm_actor:
        parser.error("--use_wm_actor true requires --wm_actor_checkpoint; disabled mode accepts no WM path")
    if bool(all_args.pm_reward_checkpoint) != all_args.use_pm_reward:
        parser.error("--use_pm_reward true requires --pm_reward_checkpoint; disabled mode accepts no PM path")
    if not np.isfinite(all_args.pm_reward_alpha) or all_args.pm_reward_alpha < 0:
        parser.error("PM alpha must be finite and nonnegative")
    if all_args.use_pm_reward and (all_args.data_behavior != "tg-mappo" or all_args.export_wm_data
                                   or all_args.grid_size != 5 or all_args.comm_range != 1.5):
        parser.error("PM integration requires original 25-node TG, radius 1.5, raw export disabled")
    if bool(all_args.tg_policy_checkpoint) != all_args.tg_evaluate:
        parser.error("Policy checkpoint is evaluation-only; provide --tg_evaluate true and --tg_policy_checkpoint")
    if all_args.tg_evaluate and (all_args.export_wm_data or all_args.data_behavior != "tg-mappo"):
        parser.error("Frozen policy evaluation requires TG behavior and raw data export disabled")
    if all_args.use_wm_actor and (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
                                 or all_args.data_behavior != "tg-mappo" or all_args.export_wm_data
                                 or all_args.comm_range != 1.5 or all_args.grid_size != 5):
        parser.error("WM integration requires 25 nodes, radius 1.5, feedforward TG, raw export disabled")
    if all_args.data_behavior != "tg-mappo" and not all_args.export_wm_data:
        parser.error("Rule behavior is data-only and requires --export_wm_data true")
    if all_args.scenario_seed is not None and not 0 <= all_args.scenario_seed < 2 ** 32:
        parser.error("scenario_seed must be in [0, 2**32)")
    if all_args.export_wm_data and (
            min(all_args.wm_export_max_steps, all_args.wm_export_max_bytes,
                all_args.wm_export_max_packets) <= 0
            or all_args.effective_num_env_steps > all_args.wm_export_max_steps):
        parser.error("WM data budgets must be positive and cover the full requested training")
    print("{} budget: {} episodes x {} joint slots = {} environment steps ({})".format(
        "Evaluation" if all_args.tg_evaluate else "Training",
        all_args.num_episodes, all_args.episode_length,
        all_args.effective_num_env_steps, all_args.training_budget_source), flush=True)
    from .environment import buoy_environment
    from .runner import Runner
    from .agent import Agent, DeepQNetwork
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    # 浮标的格式[能力，坐标]，虽然能能量没有用到，但这个形式保留了下来
    buoys = [[10, [i, j]] for i in range(all_args.grid_size) for j in range(all_args.grid_size)]

    env = buoy_environment(buoys,all_args)
    env.build_env()
    agent_number = env.buoy_number
    agents = []
    buoys_obj = env.buoys
    n_actions_max = 0
    n_features_max = 0
    for i in range(agent_number):
        n_actions_tmp = env.get_n_action(i)
        n_features_tmp = env.get_n_obs(i)
        if n_features_tmp > n_features_max:
            n_features_max = n_features_tmp
        if n_actions_tmp > n_actions_max:
            n_actions_max = n_actions_tmp
    try:
        RL = DeepQNetwork(n_actions_max, n_features_max,
                          learning_rate=0.001,
                          reward_decay=0.9,
                          e_greedy=1,
                          replace_target_iter=200,
                          memory_size=100000, env=env)
        if all_args.tg_evaluate:
            RL.load_policy(all_args.tg_policy_checkpoint)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    for i in range(agent_number):
        n_actions = env.get_n_action(i)
        n_features = env.get_n_obs(i)
        agent_tmp = Agent(buoys_obj[i], RL)
        agents.append(agent_tmp)

    runner = Runner(agents, env)

    runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])


