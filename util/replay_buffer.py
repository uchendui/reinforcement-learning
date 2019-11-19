import sys
import os
import math
import heapq
import pickle
import random
import numpy as np
from collections import deque, OrderedDict
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
        raise NotImplementedError

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

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
    def __init__(self, capacity=1000, batch_size=32, alpha=0.7, sort_freq=1000):
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
        self.sort_freq = sort_freq
        self._segments = None
        self._pdf = np.power(range(1, self.capacity + 1), self.alpha)
        self._pdf /= self._pdf.sum()
        self._cdf = np.cumsum(self._pdf)

    def add(self, transition):
        """ Adds a transition of the form (priority, prev_state, action, reward, next_state) to the replay buffer

        Args:
            transition:  array containing transition (obs, action, reward, new_obs, done)
        """
        assert (len(transition) == 5), "Transitions passed to replay buffer must be of length 5"

        t = [self.max_td, self.count, *transition]
        if len(self.buffer) >= self.capacity:
            x = heapq.heappushpop(self.buffer, t)
            t[1] = x[1]  # reassign the count
        else:
            heapq.heappush(self.buffer, t)
            self.count += 1
            # segments change only when alpha or N changes
            self._compute_segments()

    def sample(self, n, beta):
        """Samples transitions from the replay buffer.

             Args:
                 n: Number of transitions to sample

             Returns:
                 samples: array of (states, actions, rewards, next_states, done)
             """
        assert (n <= len(self)), "Sampling amount larger than replay buffer size"

        probabilities = []
        samples = []
        indices = []
        for i in range(1, self.batch_size + 1):
            begin = self._segments[i - 1]
            end = self._segments[i]
            end += 1 if begin == end else 0

            choice = np.random.randint(begin, end)
            samples.append(self.buffer[choice])
            probabilities.append(self._pdf[choice])
            indices.append(choice)
        weights = np.array(probabilities)
        weights = np.power(len(self) * weights, -beta)
        weights /= weights.max()
        return list(map(np.array, zip(*samples))), indices, weights

    def _compute_segments(self):
        """Computes the segments for rank based sampling.
        Inpsired by https://github.com/Damcy/prioritized-experience-replay/blob/master/rank_based.py
        """
        if len(self) < self.batch_size:
            return

        # Dictionary storing segment walls
        ends = OrderedDict({0: 0})

        # Compute segment walls
        # to avoid computing the pdf/cdf at sample time, we pre compute them
        # and use varying portions depending on the current size of the buffer
        step = self._cdf[len(self) - 1] / self.batch_size
        for cnt in range(1, self.batch_size + 1):
            prob = step * cnt
            inde = np.searchsorted(self._cdf, prob)
            ends[cnt] = inde
        self._segments = ends

    def heapify(self):
        """Balances the replay buffer."""
        heapq.heapify(self.buffer)

    def update_priorities(self, indices, tds):
        """Updates the priorities of transitions at "indices" with "tds" at the corresponding index
        Args:
            indices: Indices of transitions in the replay buffer
            tds: Temporal difference errors calculated during previous update
        """
        max_td = np.max(tds)
        self.max_td = np.maximum(self.max_td, max_td)
        for i, index in enumerate(indices):
            self.buffer[index][0] = tds[i]

    def __len__(self):
        return len(self.buffer)
