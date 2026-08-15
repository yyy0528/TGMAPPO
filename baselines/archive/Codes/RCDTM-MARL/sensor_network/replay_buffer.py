import collections
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.buffer_back = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, memory_id, back):
        if back:
            self.buffer_back.append((state, action, reward, next_state, done, memory_id))
        else:
            self.buffer.append((state, action, reward, next_state, done, memory_id))

    def sample(self, batch_size, batch_size_b):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples to sample from the buffer")
        transitions = random.sample(self.buffer, batch_size)
        n_back = len(self.buffer_back)
        if n_back >= batch_size_b:
            transitions_b = random.sample(self.buffer_back, batch_size_b)
        else:
            transitions_b = random.sample(self.buffer_back, n_back)
            tmp = random.sample(self.buffer, batch_size_b - n_back)
            transitions_b = transitions_b + tmp

        transitions = transitions + transitions_b
        state, action, reward, next_state, done, memory_id = zip(*transitions)

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done), np.array(memory_id)

    def size(self):
        return len(self.buffer)

    def pop(self, back):
        if back:
            self.buffer_back.popleft()
        else:
            self.buffer.popleft()

    def get_buffer(self):
        return self.buffer