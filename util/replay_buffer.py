import random
import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=1000):
        """Represents a replay buffer that stores transitions. (prev_state, action, reward, next_state)

        Args:
            capacity: Number of samples the replay buffer can hold.
        """
        self.buffer = deque()
        self.capacity = capacity

    def __len__(self):
        return self.size()

    def sample(self, n):
        """Samples transitions from the replay buffer.

        Args:
            n: Number of transitions to sample

        Returns:
            samples: array of (states, actions, rewards, next_states, done)
        """
        assert (n <= len(self.buffer)), "Sample amount larger than replay buffer size"
        sample = random.sample(self.buffer, n)
        return list(map(np.array, zip(*sample)))

    def add(self, transition):
        """ Adds a transition of the form (prev_state, action, reward, next_state) to the replay buffer

        Args:
            transition:  array containing transition (obs, action, reward, new_obs, done)
        """
        assert (len(transition) == 5), "Transitions passed to replay buffer must be of length 5"
        if len(self.buffer) >= self.capacity:
            self.buffer.pop()
        self.buffer.appendleft(transition)

    def size(self):
        """Returns the number of transitions in the buffer"""
        return len(self.buffer)
