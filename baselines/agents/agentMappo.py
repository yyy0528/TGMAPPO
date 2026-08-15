import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from src.replay_buffer import ReplayBuffer

device = torch.device("cpu")

class Agent:
    def __init__(self, buoy, n_actions, n_features, n_agents):
        self.step = 0
        self.network = None
        self.buoy = buoy
        self.sample_id = []

        self.RL = MAPPO(n_actions, n_features,
                        learning_rate=0.001,
                        reward_decay=0.9,
                        e_greedy=1,
                        replace_target_iter=200,
                        memory_size=100000,
                        n_agents=n_agents)

class Actor(nn.Module):
    def __init__(self, n_features, n_actions, n_neuron=100):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_features, n_neuron)
        self.fc2 = nn.Linear(n_neuron, n_neuron)
        self.fc3 = nn.Linear(n_neuron, n_actions)

    def forward(self, obs):
        x = f.relu(self.fc1(obs))
        x = f.relu(self.fc2(x))
        x = f.softmax(self.fc3(x), dim=-1)  # Output a probability distribution
        return x

class Critic(nn.Module):
    def __init__(self, n_features, n_neuron=100):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_features, n_neuron)
        self.fc2 = nn.Linear(n_neuron, n_neuron)
        self.fc3 = nn.Linear(n_neuron, 1)

    def forward(self, obs):
        x = f.relu(self.fc1(obs))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MAPPO(nn.Module):
    def __init__(self, n_actions, n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=1,
                 replace_target_iter=300,
                 memory_size=100000,
                 batch_size=128,
                 n_agents=1,
                 batch_size_a=64,
                 batch_size_b=64,
                 e_greedy_increment=0.0001,
                 multi=8):
        super(MAPPO, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = 1
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.batch_size_a = batch_size_a
        self.batch_size_b = batch_size_b
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.multi = multi

        self.memory_id = 0
        self.memory_id_b = 0
        self.learn_step_counter = 0

        self.replay_buffer = ReplayBuffer(self.memory_size)

        self.actor = Actor(n_features=self.n_features, n_actions=self.n_actions).to(device)
        self.critic = Critic(n_features=self.n_features).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.loss_function = nn.MSELoss()

        self.cost_his = []

    def store_transition(self, s, a, r, s_, done, back):
        self.replay_buffer.add(s, a, r, s_, done,self.memory_id_b, back)
        return self.memory_id

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            observation = np.atleast_1d(observation)
            observation = observation[np.newaxis, :]
            s = torch.FloatTensor(observation).to(device)
            actions_prob = self.actor(s).cpu().detach().numpy()
            action = np.random.choice(np.arange(self.n_actions), p=actions_prob[0])
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        tau = 0.01
        # # 由于MAPPO中只有一个Critic网络，我们不需要遍历，直接更新Critic网络的参数即可
        # params_eval = self.critic.state_dict()
        # params_target = self.critic.state_dict()  # 在MAPPO中，我们不需要目标Critic网络，所以这里只是示例
        # for key in params_eval:
        #     params_target[key] = params_eval[key] * tau + params_target[key] * (1 - tau)
        # self.critic.load_state_dict(params_target)

        params_eval = self.actor.state_dict()
        params_target = self.actor.state_dict()
        for key in params_eval:
            params_target[key] = params_eval[key] * tau + params_target[key] * (1 - tau)
        self.actor.load_state_dict(params_target)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()  # 确保这个函数存在且正确
            #print('\ntarget params replaced\n')

        # Sample data from replay buffer
        b_s, b_a, b_r, b_ns, b_d, _ = self.replay_buffer.sample(self.batch_size_a, self.batch_size_b)

        b_s = np.array(b_s, dtype=np.float32)
        b_ns = np.array(b_ns, dtype=np.float32)
        b_r = np.array(b_r, dtype=np.float32)
        b_a = np.array(b_a, dtype=np.int32)

        s = torch.FloatTensor(b_s).to(device)
        s_ = torch.FloatTensor(b_ns).to(device)
        r = torch.FloatTensor(b_r).to(device)
        b_d = torch.FloatTensor(b_d).to(device)  # 确保 b_d 也在 GPU 上

        # Critic update
        q_values = self.critic(s).squeeze()
        q_next_values = self.critic(s_).squeeze()
        q_target = r + self.gamma * q_next_values * (1 - b_d)  # 确保所有操作都在 GPU 上
        critic_loss = self.loss_function(q_values, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        actions_prob = self.actor(s).gather(1, torch.LongTensor(b_a).view(-1, 1).to(device)).squeeze()
        actor_loss = -torch.log(actions_prob) * actions_prob  # Policy gradient loss
        actor_loss = actor_loss.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.cost_his.append(critic_loss.detach().cpu().numpy())  # 确保 loss 也在 CPU 上

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()