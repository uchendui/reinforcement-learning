import random
import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque()
        self.capacity = capacity

    def __len__(self):
        return self.size()

    def sample(self, n):
        """
        Sample transitions from the replay buffer
        :param n: How many transitions to sample
        :return: samples
        """
        assert (n <= len(self.buffer)), "Sample amount larger than replay buffer size"
        sample = random.sample(self.buffer, n)
        return list(map(np.array, zip(*sample)))

    def add(self, transition):
        """
        Add a transition of the form (prev_state, action, reward, next_state) to the replay buffer
        :param transition: numpy array containing transition
        :return: None
        """
        assert (len(transition) == 5), "Transitions passed to replay buffer must be of length 5"
        if len(self.buffer) >= self.capacity:
            self.buffer.pop()
        self.buffer.appendleft(transition)

    def size(self):
        return len(self.buffer)
