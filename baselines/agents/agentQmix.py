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
    QMIX Network Structure for Individual Agent Q Values
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

class MixingNetwork(nn.Module):
    """
    QMIX Mixing Network
    """

    def __init__(self, n_agents, state_dim, hypernet_embed_dim=128):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Hypernetworks for generating mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, n_agents * 32)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, 32 * 1)
        )
        # Hypernetwork for generating biases
        self.hyper_b1 = nn.Linear(state_dim, 32)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, agent_qs, states):
        """
        Combine individual agent Q values into a single Q value for the team
        """
        batch_size = agent_qs.size(0)
        states = states.view(batch_size, -1)

        # Generate weights and biases
        w1 = self.hyper_w1(states).view(batch_size, self.n_agents, 32)
        w2 = self.hyper_w2(states).view(batch_size, 32, 1)
        b1 = self.hyper_b1(states).view(batch_size, 1, 32)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        # Compute hidden layer
        hidden = f.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        # Compute final Q value
        team_q = torch.bmm(hidden, w2) + b2
        return team_q.squeeze(-1)

class DQN(nn.Module):
    """
    QMIX Algorithm (Value Mixing Networks)
    """

    def __init__(self, n_actions, n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=1,
                 replace_target_iter=300,
                 memory_size=100000,
                 batch_size=128,
                 batch_size_a=64,
                 batch_size_b=64,
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

        # Build evaluation and target networks
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions).to(device)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions).to(device)

        # QMIX Mixing Networks
        self.mixing_net_eval = MixingNetwork(n_agents=1, state_dim=n_features).to(device)
        self.mixing_net_target = MixingNetwork(n_agents=1, state_dim=n_features).to(device)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(list(self.eval_net.parameters()) + list(self.mixing_net_eval.parameters()), lr=self.lr)

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

        params_eval_mix = self.mixing_net_eval.state_dict()
        params_target_mix = self.mixing_net_target.state_dict()
        for key in params_eval_mix:
            params_target_mix[key] = params_eval_mix[key] * tau + params_target_mix[key] * (1 - tau)
        self.mixing_net_target.load_state_dict(params_target_mix)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        b_s, b_a, b_r, b_ns, b_d, _ = self.replay_buffer.sample(self.batch_size_a, self.batch_size_b)

        b_s = np.array(b_s, dtype=np.float32)
        b_ns = np.array(b_ns, dtype=np.float32)
        b_r = np.array(b_r, dtype=np.float32)
        b_a = np.array(b_a, dtype=np.int32)

        s = torch.FloatTensor(b_s).to(device)
        s_ = torch.FloatTensor(b_ns).to(device)
        r = torch.FloatTensor(b_r).to(device)

        # Get Q values for each agent in the current state (s) and next state (s_)
        q_eval = self.eval_net(s)  # Shape: [batch_size, n_agents, n_actions]
        q_next = self.target_net(s_)  # Shape: [batch_size, n_agents, n_actions]

        # Select Q values for actions taken
        q_eval = torch.gather(q_eval, 1, torch.LongTensor(b_a).view(-1, 1).to(device)).squeeze()

        # Compute the target Q value
        q_next_max = q_next.max(1)[0].detach()  # Max over actions
        team_q_next = self.mixing_net_target(q_next_max.unsqueeze(1), s_)  # Mix Q values
        q_target = r + self.gamma * team_q_next

        # Compute the team Q value for current state
        team_q_eval = self.mixing_net_eval(q_eval.unsqueeze(1), s)

        # Compute loss and update networks
        loss = self.loss_function(team_q_eval, q_target)
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
