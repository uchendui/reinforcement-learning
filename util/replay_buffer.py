import random
import numpy as np
import heapq
from collections import deque


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
    def __init__(self, capacity=1000, batch_size=32):
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
        self.count = 0

    def add(self, transition):
        """ Adds a transition of the form (priority, prev_state, action, reward, next_state) to the replay buffer

        Args:
            transition:  array containing transition (obs, action, reward, new_obs, done)
        """
        assert (len(transition) == 5), "Transitions passed to replay buffer must be of length 5"

        t = (self.max_td, self.count, *transition)
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
        bin_sep = int(np.floor(siz / self.batch_size))
        samples = []
        for i in range(self.batch_size):
            loc = np.random.randint(i * bin_sep, (i + 1) * bin_sep)
            samples.append(self.buffer[loc])
        # samples = [) for i in
        #            range(self.batch_size)]

        # Process samples to remove the priority
        return list(map(np.array, zip(*samples)))

    def update_priorities(self, indices):
        pass

    def size(self):
        return len(self.buffer)
