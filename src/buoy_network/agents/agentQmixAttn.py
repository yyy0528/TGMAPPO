import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from ..replay_buffer import ReplayBuffer

device = torch.device("cpu")


class Agent:
    def __init__(self, buoy, n_actions, n_features, n_agents):
        self.step = 0
        self.network = None
        self.buoy = buoy
        self.sample_id = []

        self.RL = QMIX(n_actions, n_features, n_agents=64,
                       learning_rate=0.001,
                       reward_decay=0.9,
                       e_greedy=1,
                       replace_target_iter=200,
                       memory_size=100000)


class Network(nn.Module):
    """
    Network Structure with Attention Mechanism
    """

    def __init__(self, n_features, n_actions, n_neuron=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, n_neuron)
        self.attention = Attention(n_neuron, n_neuron)  # Add attention layer
        self.fc2 = nn.Linear(n_neuron, n_neuron)
        self.fc3 = nn.Linear(n_neuron, 20)
        self.fc4 = nn.Linear(20, n_actions)

    def forward(self, obs):
        if obs.shape[1] == self.fc1.in_features:
            x = f.relu(self.fc1(obs))
            x = self.attention(x)
            x = f.relu(self.fc2(x))
            x = f.relu(self.fc3(x))
            q = self.fc4(x)
        else:
            q = obs
        return q



class MixingNetwork(nn.Module):
    """
    Mixing Network with Attention Mechanism for QMIX
    """

    def __init__(self, n_agents, multi, hidden_dim=64):
        super(MixingNetwork, self).__init__()
        self.fc1 = nn.Linear(n_agents * multi, hidden_dim)
        self.attention = Attention(hidden_dim, hidden_dim)  # Add attention layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, q_values):
        if q_values.shape[1] == self.fc1.in_features:
            x = f.relu(self.fc1(q_values))
            x = self.attention(x)  # Apply attention mechanism here
            x = f.relu(self.fc2(x))
            q_total = self.fc3(x)
        else:
            q_total = q_values
        return q_total



class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, value)
        return output


class QMIX(nn.Module):
    """
    QMIX Algorithm
    """

    def __init__(self, n_actions, n_features, n_agents,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=1,
                 replace_target_iter=300,
                 memory_size=100000,
                 batch_size=128,
                 batch_size_a=64,
                 batch_size_b=64,
                 e_greedy_increment=0.0001,
                 multi = 8):
        super(QMIX, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_agents = n_agents
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = 1
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.batch_size_a = batch_size_a
        self.batch_size_b = batch_size_b
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.multi = multi

        self.learn_step_counter = 0

        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.memory_id = 0
        self.memory_id_b = 0

        # build two network: eval_net and target_net
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.multi = self.target_net.fc4.out_features
        self.mixing_net = MixingNetwork(n_agents=self.n_agents, multi=self.multi)

        self.eval_net = self.eval_net.to(device)
        self.target_net = self.target_net.to(device)
        self.mixing_net = self.mixing_net.to(device)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # 记录每一步的误差
        self.cost_his = []

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

    def choose_action(self, observations):
        observation = observations
        observation = np.atleast_1d(observation)
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            s = torch.FloatTensor(observation).to(device)
            actions_value = self.eval_net(s).cpu().detach().numpy()
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def _replace_target_params(self):
        tau = 0.01
        params_eval = self.eval_net.state_dict()
        params_target = self.target_net.state_dict()
        for key in params_eval:
            params_target[key] = params_eval[key] * tau + params_target[key] * (1 - tau)
        self.target_net.load_state_dict(params_target)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            #print('\ntarget params replaced\n')

        b_s, b_a, b_r, b_ns, b_d, _ = self.replay_buffer.sample(self.batch_size_a, self.batch_size_b)

        b_s = np.array(b_s, dtype=np.float32)
        b_ns = np.array(b_ns, dtype=np.float32)
        b_r = np.array(b_r, dtype=np.float32)
        b_a = np.array(b_a, dtype=np.int32)

        s = torch.FloatTensor(b_s).to(device)
        s_ = torch.FloatTensor(b_ns).to(device)
        r = torch.FloatTensor(b_r).to(device)

        q_eval = self.eval_net(s)
        q_next = self.target_net(s_)

        # q_eval = torch.cat(q_eval, dim=1)
        # q_next = torch.cat(q_next, dim=1)

        q_target = torch.zeros_like(q_eval).to(device)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        reward = r + self.gamma * q_next.max(dim=1).values

        q_target[batch_index, b_a] = reward

        q_total_eval = self.mixing_net(q_eval)
        q_total_target = self.mixing_net(q_target)

        loss = self.loss_function(q_total_eval, q_total_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.detach().cpu().numpy())

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
