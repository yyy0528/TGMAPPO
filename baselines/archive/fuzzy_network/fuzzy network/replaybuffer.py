from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, fuzzy_state, action, reward, next_state, next_fuzzy_state, done):
        self.buffer.append((state, fuzzy_state, action, reward, next_state, next_fuzzy_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, fuzzy_state, action, reward, next_state, next_fuzzy_state, done = zip(*transitions)
        # return np.array(state), action, reward, np.array(next_state), done

        return state, fuzzy_state, action, reward, next_state, next_fuzzy_state, done

    # def add(self, state, action, reward, next_state, done, log_prob_behavior):
    #     self.buffer.append((
    #         state, action, reward, next_state, done, log_prob_behavior
    #     ))
    #
    # def sample(self, batch_size):
    #     transitions = random.sample(self.buffer, batch_size)
    #     states, actions, rewards, next_states, dones, log_probs_behavior = zip(*transitions)
    #
    #     # 计算目标策略（当前Actor）的动作概率密度
    #     # mu_current = current_actor(torch.FloatTensor(states))
    #     # dist_current = torch.distributions.Normal(mu_current, torch.tensor([0.2]))  # 假设固定噪声方差σ=0.2
    #     # log_probs_target = dist_current.log_prob(torch.FloatTensor(actions)).detach().numpy()
    #     #
    #     # # 计算重要性权重
    #     # log_weights = log_probs_target - np.array(log_probs_behavior)
    #     # weights = np.exp(log_weights - np.max(log_weights))  # 数值稳定
    #     # weights /= np.sum(weights)
    #
    #     return (
    #         np.array(states),
    #         np.array(actions),
    #         np.array(rewards),
    #         np.array(next_states),
    #         np.array(dones),
    #         np.array(log_probs_behavior),
    #         # weights
    #     )

    def size(self):
        return len(self.buffer)