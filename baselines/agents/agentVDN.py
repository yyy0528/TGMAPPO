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

        self.RL = DQN(n_actions, n_features,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=200,
                      memory_size=100000)

class Network(nn.Module):
    """
    Network Structure
    """

    def __init__(self, n_features, n_actions, n_neuron=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, n_neuron)
        self.fc2 = nn.Linear(n_neuron, n_neuron)
        self.fc3 = nn.Linear(n_neuron, 20)
        self.fc4 = nn.Linear(20, n_actions)

    def forward(self, obs):
        if obs.shape[1] == self.fc1.in_features:
            x = f.relu(self.fc1(obs))
            x = f.relu(self.fc2(x))
            x = f.relu(self.fc3(x))
            q = self.fc4(x)
        else:
            q = obs
        return q

class DQN(nn.Module):
    """
    VDN Algorithm (Value Decomposition Networks)
    """

    def __init__(self, n_actions, n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=1,
                 replace_target_iter=300,
                 memory_size=100000,
                 batch_size=128,
                 batch_size_a = 64,
                 batch_size_b = 64,
                 e_greedy_increment=0.0001):
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = 1
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory_id = 0
        self.learn_step_counter = 0
        self.batch_size_a = batch_size_a
        self.batch_size_b = batch_size_b

        self.replay_buffer = ReplayBuffer(self.memory_size)

        # Build a single network for both evaluation and target
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions).to(device)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions).to(device)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # Record the cost history
        self.cost_his = []

    def store_transition(self, s, a, r, s_, done, back):
        self.replay_buffer.add(s, a, r, s_, done, self.memory_id, back)
        return self.memory_id

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            observation = np.atleast_1d(observation)
            observation = observation[np.newaxis, :]
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

        # print(f"b_s:{b_s}")
        b_s = np.array(b_s, dtype=np.float32)
        # print(f"after:b_s:{b_s}")
        b_ns = np.array(b_ns, dtype=np.float32)
        b_r = np.array(b_r, dtype=np.float32)
        b_a = np.array(b_a, dtype=np.int32)

        s = torch.FloatTensor(b_s).to(device)
        s_ = torch.FloatTensor(b_ns).to(device)
        r = torch.FloatTensor(b_r).to(device)

        # Get Q values for each agent in the current state (s) and next state (s_)
        q_eval = self.eval_net(s)  # Shape: [batch_size, n_agents, n_actions]
        q_next = self.target_net(s_)  # Shape: [batch_size, n_agents, n_actions]

        # Sum the Q values for all agents (VDN)
        q_eval = torch.sum(q_eval.gather(0, torch.LongTensor(b_a).view(-1, 1).to(device)), dim=1).squeeze()
        q_next = torch.sum(q_next.max(0)[0].detach(), dim=0)

        q_target = r + self.gamma * q_next
        loss = self.loss_function(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.detach().cpu().numpy())

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()
