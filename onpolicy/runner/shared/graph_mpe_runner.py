import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio

#这个函数 _t2n(x) 是一个在深度学习中常用的辅助函数，
#用于将 PyTorch 的张量（Tensor）转换为 NumPy 数组（ndarray）。
def _t2n(x):
    return x.detach().cpu().numpy()

#GMPERunner 类是一个用于在多智能体粒子环境（MPEs）中进行训练、评估和数据收集的类。
# 它继承自 Runner 类，
# 并实现了多个方法来处理数据收集、插入、计算回报、评估和渲染等任务。
class GMPERunner(Runner):
    """
    Runner class to perform training, evaluation and data
    collection for the MPEs. See parent class for details
    """

    dt = 0.1

    def __init__(self, config):
        super(GMPERunner, self).__init__(config)

  #run 方法是主要的训练循环
    def run(self):
        self.warmup()  #预填充经验回放缓冲区（Replay Buffer），确保训练开始时缓冲区有足够的数据

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        # This is where the episodes are actually run.
        for episode in range(episodes):
            # 学习率衰减
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):

                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obs reward and next obs  环境交互

                #################################
                #agent_id：当前执行动作的智能体 ID（在多智能体环境中）
                #node_obs：图中每个节点的特征，形状可能是[num_nodes, node_dim]
                #adj：邻接矩阵 (Adjacency Matrix)，表示图中节点间的连接关系
                ######################################


                obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(
                    actions_env
                )
                #数据组装与存储
                data = (
                    obs,
                    agent_id,
                    node_obs,
                    adj,
                    agent_id,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network  回 报计算
            self.compute()

            #网络更新            从缓冲区采样数据，更新策略网络（Actor）和价值网络（Critic）
            train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model      模型保存    定期保存模型参数，防止训练中断丢失进度
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information   日志输出
            if episode % self.log_interval == 0:
                end = time.time()

                env_infos = self.process_infos(infos)

                avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                print(
                    f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)



    #warmup 方法用于初始化环境和经验回放缓冲区。
    # 它首先重置环境，然后根据是否使用集中式评论家（centralized_V）
    # 来处理共享观测和共享智能体ID。最后，将这些数据复制到缓冲区的初始位置。
    def warmup(self):

        # reset env
        obs, agent_id, node_obs, adj = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()



    #collect 方法用于在每个时间步中收集数据。
    # 它首先准备卷积操作，然后从策略网络中获取动作、状态值、动作的对数概率和RNN状态。
    # 根据动作空间的类型（MultiDiscrete或Discrete），
    # 将动作转换为环境可以接受的格式
    @torch.no_grad()
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.node_obs[step]),
            np.concatenate(self.buffer.adj[step]),
            np.concatenate(self.buffer.agent_id[step]),
            np.concatenate(self.buffer.share_agent_id[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[
                    actions[:, :, i]
                ]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )


    #insert 方法用于将收集到的数据插入到经验回放缓冲区 buffer 中
    def insert(self, data):
        (
            obs,
            agent_id,
            node_obs,
            adj,
            agent_id,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # if centralized critic, then shared_obs is concatenation of obs from all agents
        if self.use_centralized_V:
            # TODO stack agent_id as well for agent specific information
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.insert(
            share_obs,
            obs,
            node_obs,
            adj,
            agent_id,
            share_agent_id,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )


    #compute 方法用于计算收集数据的回报
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        #调用训练器的准备方法，通常用于设置模型为评估模式（如禁用 Dropout）。

        #调用策略网络中的价值函数，估计状态价值。
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.node_obs[-1]),
            np.concatenate(self.buffer.adj[-1]),
            np.concatenate(self.buffer.share_agent_id[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    #eval 方法用于评估当前策略的性能。
    @torch.no_grad()
    def eval(self, total_num_steps: int):
        eval_episode_rewards = []
        eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                        self.eval_envs.action_space[0].high[i] + 1
                    )[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate(
                            (eval_actions_env, eval_uc_actions_env), axis=2
                        )
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(
                    np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2
                )
            else:
                raise NotImplementedError

            # Obser reward and next obs
            (
                eval_obs,
                eval_agent_id,
                eval_node_obs,
                eval_adj,
                eval_rewards,
                eval_dones,
                eval_infos,
            ) = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(
            "eval average episode rewards of agent: "
            + str(eval_average_episode_rewards)
        )
        self.log_env(eval_env_infos, total_num_steps)

   #render 方法用于可视化环境。如果 get_metrics 为 True，
    # 则只返回环境的指标，不进行渲染。
    # 否则，它会渲染环境并保存为 GIF 文件。
    @torch.no_grad()
    def render(self, get_metrics: bool = False):
        """
        Visualize the env.
        get_metrics: bool (default=False)
            if True, just return the metrics of the env and don't render.
        """
        envs = self.envs

        all_frames = []
        rewards_arr, success_rates_arr, num_collisions_arr, frac_episode_arr = (
            [],
            [],
            [],
            [],
        )

        for episode in range(self.all_args.render_episodes):
            obs, agent_id, node_obs, adj = envs.reset()
            if not get_metrics:
                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(node_obs),
                    np.concatenate(adj),
                    np.concatenate(agent_id),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[
                            actions[:, :, i]
                        ]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate(
                                (actions_env, uc_actions_env), axis=2
                            )
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, agent_id, node_obs, adj, rewards, dones, infos = envs.step(
                    actions_env
                )
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if not get_metrics:
                    if self.all_args.save_gifs:
                        image = envs.render("rgb_array")[0][0]
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render("human")

            env_infos = self.process_infos(infos)
            # print('_'*50)
            num_collisions = self.get_collisions(env_infos)
            frac, success = self.get_fraction_episodes(env_infos)
            rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            frac_episode_arr.append(np.mean(frac))
            success_rates_arr.append(success)
            num_collisions_arr.append(num_collisions)
            # print(np.mean(frac), success)
            # print("Average episode rewards is: " +
            # str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        print(rewards_arr)
        print(frac_episode_arr)
        print(success_rates_arr)
        print(num_collisions_arr)

        if not get_metrics:
            if self.all_args.save_gifs:
                imageio.mimsave(
                    str(self.gif_dir) + "/render.gif",
                    all_frames,
                    duration=self.all_args.ifi,
                )
