import random
import numpy as np
import heapq
from collections import deque
from itertools import groupby
from operator import itemgetter


class Buffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return self.size()

    def size(self):
        """Returns the number of transitions in the buffer"""
        return len(self.buffer)

    def sample(self, n):
        pass

    def save(self, path):
        bf = np.array(self.buffer)
        np.save(path, bf)

    def load(self, path):
        bf = np.load(path)
        self.buffer = bf


class ReplayBuffer(Buffer):
    def __init__(self, capacity):
        """Represents a replay buffer that stores transitions. (prev_state, action, reward, next_state)

        Args:
            capacity: Number of samples the replay buffer can hold.
        """
        super().__init__(capacity=capacity)
        self.buffer = deque()

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


class PrioritizedReplayBuffer(Buffer):
    def __init__(self, capacity=1000, batch_size=32, alpha=0.7):
        """Represents a replay buffer that stores transitions. (prev_state, action, reward, next_state, done, priority)

        Args:
            capacity: Number of samples the replay buffer can hold.
        """
        super().__init__(capacity)
        self.buffer = list()
        heapq.heapify(self.buffer)
        self.capacity = capacity
        self.batch_size = batch_size
        self.max_td = 1e-6
        self.alpha = alpha
        self.count = 0

    def add(self, transition):
        """ Adds a transition of the form (priority, prev_state, action, reward, next_state) to the replay buffer

        Args:
            transition:  array containing transition (obs, action, reward, new_obs, done)
        """
        assert (len(transition) == 5), "Transitions passed to replay buffer must be of length 5"

        t = [-self.max_td, self.count, *transition]
        if len(self.buffer) >= self.capacity:
            heapq.heappushpop(self.buffer, t)
        heapq.heappush(self.buffer, t)
        self.count += 1

    def sample(self, n):
        """Samples transitions from the replay buffer.

             Args:
                 n: Number of transitions to sample

             Returns:
                 samples: array of (states, actions, rewards, next_states, done)
             """
        siz = len(self.buffer)
        assert (n <= siz), "Sample amount larger than replay buffer size"

        power_s = [0] + sorted((siz * np.random.power(a=self.alpha, size=self.batch_size - 1)).astype(int))
        walls = sorted(list(set(power_s))) + [siz]
        samples = []
        indices = []

        for i, (k, group) in enumerate(groupby(power_s)):
            assert k == walls[i]
            inds = np.arange(start=k, stop=walls[i + 1])
            samp = np.random.choice(inds, size=len(list(group)))
            transitions = itemgetter(*samp)(self.buffer)
            indices.extend(samp)
            if type(transitions[0]) != list:
                transitions = (transitions,)
            samples.extend(transitions)
        return list(map(np.array, zip(*samples))), indices

    def update_priorities(self, indices, tds):
        """Updates the priorities of transitions at "indices" with "tds" at the corresponding index
        Args:
            indices: Indices of transitions in the replay buffer
            tds: Temporal difference errors calculated during previous update
        """
        for i, index in enumerate(indices):
            self.max_td = max(tds[i], self.max_td)
            self.buffer[index][0] = -tds[i]
        heapq.heapify(self.buffer)

    def size(self):
        return len(self.buffer)
