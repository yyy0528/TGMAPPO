import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from .replay_buffer import ReplayBuffer
from onpolicy.algorithms.graph_mappo import GR_MAPPO as TrainAlgo
from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy as Policy
import gym
from gym import spaces
from onpolicy.utils.graph_buffer import  GraphReplayBuffer
import wandb
import os
from typing import Dict
import numpy as np
import torch
import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio


# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter  # tensorboardX to work with macos
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.graph_buffer import GraphReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()
device = torch.device("cpu")

class Agent:
    def __init__(self, buoy, RL):
        self.step = 0
        self.network = None
        self.buoy = buoy
        self.sample_id = []

        self.RL = RL







class Network(nn.Module):
    """
    Network Structure
    """
    def __init__(self,
                 n_features,
                 n_actions,
                 n_neuron=100
                 ):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, n_neuron)

        self.fc2 = nn.Linear(n_neuron, n_neuron)
        self.fc3 = nn.Linear(n_neuron, 20)

        self.fc4 = nn.Linear(20, n_actions)

    def forward(self, obs, hidden_state=None):

        x = f.relu(self.fc1(obs))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        q = self.fc4(x)
        return q


class DeepQNetwork(nn.Module):
    """
    Q Learning Algorithm
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=1,
                 replace_target_iter=300,
                 memory_size=100000,
                 batch_size=128,
                 batch_size_a=64,
                 batch_size_b=64,
                 e_greedy_increment=0.0001,
                 env=None):
        super(DeepQNetwork, self).__init__()

        self.env = env
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = 1
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.batch_size_a = batch_size_a
        self.batch_size_b = batch_size_b
        self.epsilon_increment = e_greedy_increment
        self.all_args = env.all_args
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.device = device
        self.n_rollout_threads=1
        self.recurrent_N = self.env.all_args.recurrent_N
        self.hidden_size = self.env.all_args.hidden_size

        self.learn_step_counter = 0

        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.buffer = GraphReplayBuffer(
                self.env.all_args,
                self.env.buoy_number,
                self.env.observation_space[0],
                self.env.observation_space[0],
                self.env.node_observation_space[0],
                self.env.agent_id_observation_space[0],
                self.env.share_agent_id_observation_space[0],
                self.env.adj_observation_space[0],
                self.env.action_space[0],
            )
        self.memory_id = 0
        self.memory_id_b = 0

        self.policy = Policy(
            self.env.all_args,
            self.env.observation_space[0],
            self.env.observation_space[0],
            self.env.node_observation_space[0],
            self.env.edge_observation_space[0],
            self.env.action_space[0],
            device=device,
        )

        # algorithm
        self.trainer = TrainAlgo(self.env.all_args, self.policy, device=self.device)
        self.wm_runtime = None
        self.wm_update_records = []
        self.evaluation_metadata = None
        if getattr(self.all_args, "use_wm_actor", False):
            if self.all_args.use_recurrent_policy or self.all_args.use_naive_recurrent_policy:
                raise ValueError("WM Actor integration currently supports feedforward TG policy only")
            from .wm_runtime import FrozenWMRuntime
            self.wm_runtime = FrozenWMRuntime(self.all_args.wm_actor_checkpoint, self.env.buoy_number)


        # build two network: eval_net and target_net
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.eval_net = self.eval_net.to(device)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.target_net = self.target_net.to(device)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # 记录每一步的误差
        self.cost_his = []

    # 重写
    def store_transition(self, s, a, r, s_, done, back):
        if back:
            self.memory_id_b += 1
            if self.memory_id_b >= self.memory_size:
                self.replay_buffer.pop(back)
                self.replay_buffer.add(s, a, r, s_, done, self.memory_id_b, back)
            else:
                self.replay_buffer.add(s, a, r, s_, done, self.memory_id_b, back)
            return self.memory_id_b
        else:
            self.memory_id += 1
            if self.memory_id >= self.memory_size:
                self.replay_buffer.pop(back)
                self.replay_buffer.add(s, a, r, s_, done, self.memory_id, back)
            else:
                self.replay_buffer.add(s, a, r, s_, done, self.memory_id, back)
            return self.memory_id


    def choose_action(self, observation):
        observation = np.atleast_1d(observation)
        observation = observation[np.newaxis, :]

        # self.epsilon
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(observation)
            s = s.to(device)
            actions_value = self.eval_net(s)
            actions_value = actions_value.to('cpu')
            actions_value = actions_value.detach().numpy()

            action = [np.argmax(actions_value)][0]
        else:

            action = np.random.randint(0, self.n_actions)

        return action

    def choose_action_two(self, observation, avail_obs, avail_actions):

        observation = np.atleast_1d(observation)
        observation = observation[np.newaxis, :]
        observation = observation * avail_obs

        # self.epsilon
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(observation)
            s = s.to(device)
            actions_value = self.eval_net(s)
            actions_value = actions_value.to('cpu')
            actions_value = actions_value.detach().numpy()
            # avail_actions_np = np.array(avail_actions)
            # actions_value = actions_value*avail_actions_np
            action = [np.argmax(actions_value)][0]
        else:
            while True:
                action_id = np.random.randint(0, self.n_actions)
                action_tmp = avail_actions[action_id]
                if action_tmp == 0:
                    continue
                else:
                    action = action_id
                    break

            # n_a = avail_actions.count(1)
            # action = np.random.randint(0, n_a)
        return action

    def _replace_target_params(self):
        # 复制网络参数
        tau = 0.01

        # 获取评估网络和目标网络的参数
        params_eval = self.eval_net.state_dict()
        params_target = self.target_net.state_dict()

        # 进行软更新
        for key in params_eval:
            params_target[key] = params_eval[key] * tau + params_target[key] * (1 - tau)

        # 将更新后的参数应用到目标网络
        self.target_net.load_state_dict(params_target)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        # sample batch memory from all memory
        # batch_memory = self.replay_buffer.sample(self.batch_size)

        b_s, b_a, b_r, b_ns, b_d, _ = self.replay_buffer.sample(self.batch_size_a, self.batch_size_b)

        # run the nextwork
        s = torch.FloatTensor(b_s)
        s = s.to(device)
        s_ = torch.FloatTensor(b_ns)
        s_ = s_.to(device)
        q_eval = self.eval_net(s)

        # q_next = self.target_net(s_)
        # 双Q学习
        q_next = self.eval_net(s_)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        # 更新值
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = b_a
        reward = torch.FloatTensor(b_r)

        reward = reward.to(device)

        q_target[batch_index, eval_act_index] = reward + self.gamma * q_next.max(dim=1).values

        # train eval network
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        self.cost_his.append(loss.detach().to('cpu').numpy())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()

#collect 方法用于在每个时间步中收集数据。
    # 它首先准备卷积操作，然后从策略网络中获取动作、状态值、动作的对数概率和RNN状态。
    # 根据动作空间的类型（MultiDiscrete或Discrete），
    # 将动作转换为环境可以接受的格式
    @torch.no_grad()
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        wm_features = None
        if self.wm_runtime is not None:
            predicted = self.wm_runtime.predict(self.buffer.obs[step], self.buffer.node_obs[step],
                self.buffer.adj[step], self.buffer.available_actions[step], step)
            self.buffer.wm_features[step] = predicted.copy()
            self.buffer.wm_feature_ready[step] = True
            wm_features = np.concatenate(self.buffer.wm_features[step])
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
            available_actions=np.concatenate(self.buffer.available_actions[step]),
            deterministic=getattr(self.all_args, "tg_evaluate", False),
            wm_features=wm_features,
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
        if self.wm_runtime is not None:
            self.wm_runtime.record_actions(actions)
        # rearrange action
        if self.env.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[
                    actions[:, :, i]
                ]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.env.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(self.env.action_space[0].n)[actions], 2)
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
            action_masks,
            dones,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            memory_id,
            memory_id_pool,
        ) = data

        #print("dones,",dones, self.recurrent_N, self.hidden_size )
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.env.buoy_number, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # if centralized critic, then shared_obs is concatenation of obs from all agents

        share_obs = obs
        share_agent_id = agent_id

        #print("action+maks",action_masks)

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
            memory_id,
            memory_id_pool,
            bad_masks= None,
            active_masks = self.env.active_masks(),
            available_actions=action_masks
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
    def warmup(self):

        # reset env
        obs, agent_id, node_obs, adj = self.env.reset_graph()
        self.buffer.reset_episode()
        if self.wm_runtime is not None:
            self.wm_runtime.reset()

        # replay buffer


        share_obs = obs
        share_agent_id = agent_id
        #print("warnup obs", obs)
        #print("warnup share obs", self.buffer.share_obs[0])
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()
        self.buffer.available_actions[0] = self.env.available_actions()
        self.buffer.active_masks[0] = self.env.active_masks()
    def train(self):
        """Train policies with data in buffer."""
        if self.wm_runtime is not None:
            import hashlib
            from .wm_runtime import tensor_digest
            self.wm_runtime.assert_frozen()
            if not self.buffer.wm_feature_ready.all():
                raise ValueError("Missing cached WM features before PPO update")
            cached_digest = hashlib.sha256(self.buffer.wm_features.tobytes()).hexdigest()
            errors = []
            with torch.no_grad():
                for step in range(self.buffer.episode_length):
                    replay, _ = self.policy.actor.evaluate_actions(
                        self.buffer.obs[step, 0], self.buffer.node_obs[step, 0], self.buffer.adj[step, 0],
                        self.buffer.agent_id[step, 0], self.buffer.rnn_states[step, 0],
                        self.buffer.actions[step, 0], self.buffer.masks[step, 0],
                        available_actions=self.buffer.available_actions[step, 0],
                        active_masks=self.buffer.active_masks[step, 0],
                        wm_features=self.buffer.wm_features[step, 0])
                    errors.append(float((torch.exp(replay-torch.as_tensor(
                        self.buffer.action_log_probs[step, 0], device=self.device))-1).abs().max()))
            if not np.isfinite(errors).all() or max(errors) > 1e-4:
                raise ValueError("Cached WM behavior probability replay mismatch")
            actor_before = tensor_digest(self.policy.actor.state_dict())
            critic_before = tensor_digest(self.policy.critic.state_dict())
            fusion_before = tensor_digest(self.policy.actor.wm_fusion.state_dict())
        self.trainer.prep_training()      # 准备训练环境
        train_infos = self.trainer.train(self.buffer)
        if self.wm_runtime is not None:
            self.wm_runtime.assert_frozen()
            if hashlib.sha256(self.buffer.wm_features.tobytes()).hexdigest() != cached_digest:
                raise ValueError("PPO mutated cached WM features")
            self.wm_update_records.append({"ratio_max_error": max(errors), "cache_sha256": cached_digest,
                "actor_changed": actor_before != tensor_digest(self.policy.actor.state_dict()),
                "critic_changed": critic_before != tensor_digest(self.policy.critic.state_dict()),
                "fusion_changed": fusion_before != tensor_digest(self.policy.actor.wm_fusion.state_dict()),
                "wm_frozen": True, "metrics": {key: float(value) for key,value in train_infos.items()}})
        #从缓冲区采样数据，计算损失函数，并通过反向传播更新网络参数。

        #更新后处理缓冲区
        self.buffer.after_update()
        return train_infos

    def save_policy(self, run_dir):
        from .policy_artifacts import save_policy
        save_policy(self, run_dir)

    def load_policy(self, path):
        from .policy_artifacts import load_policy
        load_policy(self, path)
