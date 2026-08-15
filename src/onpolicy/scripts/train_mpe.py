#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

import os, sys




sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import print_args, print_box, connected_to_internet
from onpolicy.config import get_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv



#MPE创造了一个简单的多智能体粒子世界，
# 粒子们可以进行连续的观察和离散的动作。
from onpolicy.envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
)

"""Train script for MPEs."""


#############################################
#创建 训练所需环境
def make_train_env(all_args: argparse.Namespace):
    # 内部函数定义
    def get_env_fn(rank: int):
        # 创建环境的函数
        def init_env():
            # 根据环境名称选择创建不同类型的环境
            if all_args.env_name == "MPE":
                # 创建标准MPE环境实例
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                # 创建图结构MPE环境实例
                env = GraphMPEEnv(all_args)
            else:
                # 处理不支持的环境类型
                print(f"Can not support the {all_args.env_name} environment")
                # 抛出未实现错误
                raise NotImplementedError
            # 设置环境随机种子
            # 使用基础种子 + rank * 1000 确保不同环境有不同种子
            env.seed(all_args.seed + rank * 1000)
            # 返回创建的环境实例
            return env

        # 返回环境初始化函数（不是环境实例本身）
        return init_env

    # 根据线程数决定环境类型
    # 检查是否需要创建单线程环境
    if all_args.n_rollout_threads == 1:
        # 如果是图结构环境
        if all_args.env_name == "GraphMPE":
            # 创建单线程图环境向量包装器
            # 传入一个环境初始化函数列表（包含一个元素）
            return GraphDummyVecEnv([get_env_fn(0)])
        # 创建标准单线程环境向量包装器
        return DummyVecEnv([get_env_fn(0)])
    # 需要创建多线程环境
    else:
        # 如果是图结构环境
        if all_args.env_name == "GraphMPE":
            # 创建多线程图环境向量包装器
            # 创建环境初始化函数列表，数量等于线程数
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
        # 创建标准多线程环境向量包装器
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


#创建用于评估（evaluation）的多智能体环境实例
def make_eval_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )

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

    all_args = parser.parse_known_args(args)[0]

    return all_args, parser
#这段代码的作用是定义和解析一系列与多智能体环境相关的命令行参数，
# 以便在程序运行时可以通过命令行灵活配置环境的各种属性和奖励机制等



def main(args):
    #获取参数
    parser = get_config()
    all_args, parser = parse_args(args, parser)




    if all_args.env_name == "GraphMPE":
        from onpolicy.config import graph_config

        all_args, parser = graph_config(args, parser)


    #这表示在"rmappo"算法中，必须使用某种形式的循环策略（recurrent policy）
    if all_args.algorithm_name in ["rmappo"]:
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif all_args.algorithm_name in ["mappo"]:
        #表示在"mappo"算法中，不应使用循环策略
        assert (
            all_args.use_recurrent_policy == False
            and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError



    assert (
        all_args.share_policy == True
        and all_args.scenario_name == "simple_speaker_listener"
    ) == False, (
        "The simple_speaker_listener scenario can not use shared policy. "
        "Please check the config.py."
    )



    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print_box("Choose to use gpu...")
        device = torch.device("cuda:0")
        #设置PyTorch使用的线程数，
        # 这通常与CPU核心数有关，但在这里是根据配置参数设置。
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print_box("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)



    #这是一个配置选项，
    # 用于指示是否需要打印详细的配置信息
    if all_args.verbose:
        print_args(all_args)


    # run dir
    #这段代码的作用是创建一个用于存储实验结果的目录。
    # 目录的路径是根据实验的相关信息
    # （如环境名称、场景名称、算法名称和实验名称）动态生成的

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))



    # wandb
    #这段代码的主要目的是配置和初始化wandb，
    # 以便跟踪实验结果，
    # 并在没有互联网连接的情况下提供离线模式的支持。
    if all_args.use_wandb:
        # for supercloud when no internet_connection
        if not connected_to_internet():
            import json

            # save a json file with your wandb api key in your
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser("~") + "/keys.json") as json_file:
                key = json.load(json_file)
                my_wandb_api_key = key["my_wandb_api_key"]  # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB_SAVE_CODE"] = "true"

        print_box("Creating wandboard...")
        run = wandb.init(
            config=all_args,
            project=all_args.project_name,
            # project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            # group=all_args.scenario_name,
            dir=str(run_dir),
            # job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )


    if all_args.use_wandb:
        # for supercloud when no internet_connection
        if not connected_to_internet():
            import json

            # save a json file with your wandb api key in your
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser("~") + "/keys.json") as json_file:
                key = json.load(json_file)
                my_wandb_api_key = key["my_wandb_api_key"]  # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB_SAVE_CODE"] = "true"

        print_box("Creating wandboard...")
        run = wandb.init(
            config=all_args,
            project=all_args.project_name,
            # project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            # group=all_args.scenario_name,
            dir=str(run_dir),
            # job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )




    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)







    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents



    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }


    # run experiments
    #Runner 类本身不是神经网络，而是一个用于管理和运行实验的工具。
    # 它负责初始化环境、配置模型、运行训练和评估循环，
    # 以及记录和保存训练过程中的数据和模型。


    if all_args.share_policy:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.separated.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.separated.mpe_runner import MPERunner as Runner


    runner = Runner(config)

    #具体来说，它首先检查是否启用了详细输出（verbose），
    # 然后根据策略的类型打印演员（Actor）和评论家（Critic）网络的详细信息，
    # 最后运行实验。
    if all_args.verbose:
        print_box("Actor Network", 80)
        if type(runner.policy) == list:
            print_box(runner.policy[0].actor, 80)
            print_box("Critic Network", 80)
            print_box(runner.policy[0].critic, 80)
        else:
            print_box(runner.policy.actor, 80)
            print_box("Critic Network", 80)
            print_box(runner.policy.critic, 80)


    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()



if __name__ == "__main__":
    main(sys.argv[1:])
