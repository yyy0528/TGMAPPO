import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from src.replay_buffer import ReplayBuffer
device = torch.device("cpu")

class Agent:
    def __init__(self, buoy, n_actions, n_features):
        self.step = 0
        self.network = None
        self.buoy = buoy
        self.sample_id = []

        self.RL = DeepQNetwork(n_actions, n_features,
                               learning_rate=0.001,
                               reward_decay=0.9,
                               e_greedy=1,
                               replace_target_iter=200,
                               memory_size=100000)







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
                 e_greedy_increment=0.0001):
        super(DeepQNetwork, self).__init__()


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
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.memory_id = 0
        self.memory_id_b = 0



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
            print('\ntarget params replaced\n')

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