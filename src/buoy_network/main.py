import random

from .environment import buoy_environment
from .runner import Runner
from .agent import Agent, DeepQNetwork

from onpolicy.config import get_config

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
    parser.add_argument("--comm_range", type=float, default=100.0, help="通信范围")
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

    all_args = parser.parse_known_args(args)[0]

    return all_args, parser
#这段代码的作用是定义和解析一系列与多智能体环境相关的命令行参数，
# 以便在程序运行时可以通过命令行灵活配置环境的各种属性和奖励机制等

def main(args):
    # 这段代码的作用是定义和解析一系列与多智能体环境相关的命令行参数，
    # 以便在程序运行时可以通过命令行灵活配置环境的各种属性和奖励机制等
    parser = get_config()
    all_args, parser = parse_args(args, parser)
    from onpolicy.config import graph_config
    all_args, parser = graph_config(args, parser)
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
    RL = DeepQNetwork(n_actions_max, n_features_max,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=200,
                      memory_size=100000, env=env)

    for i in range(agent_number):
        n_actions = env.get_n_action(i)
        n_features = env.get_n_obs(i)
        agent_tmp = Agent(buoys_obj[i], RL)
        agents.append(agent_tmp)

    runner = Runner(agents, env)

    runner.run()


random.seed(12)

if __name__ == "__main__":
    main(sys.argv[1:])


