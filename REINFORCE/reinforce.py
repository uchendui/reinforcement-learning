import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util.network import ReinforceNetworkBuilder


class TrainReinforce:
    def __init__(self,
                 env,
                 sess,
                 render=False,
                 max_episodes=1000,
                 print_freq=20,
                 load_path=None,
                 save_path=None,
                 ):
        self.max_episodes = max_episodes
        self.env = env
        self.input_dim = env.observation_space.shape
        self.output_dim = env.action_space.n
        self.sess = sess
        self.render = render
        self.rnb = ReinforceNetworkBuilder(self.input_dim,
                                           self.output_dim,
                                           layers=(64,),
                                           activations=(tf.nn.relu, tf.nn.softmax),
                                           scope='reinforce_network')

    def reinforce(self):
        for e in range(self.max_episodes):
            # Sample trajectories from our policy (run it on the robot)
            actions, states, q_values = self.sample()

            # Compute and apply the gradient of the log policy multiplied by our baseline
            self.update(states, actions, q_values)

    def update(self, states, actions, q_values):
        pass

    def sample(self):
        done = False
        obs = self.env.reset()
        actions = []
        states = []

        while not done:
            if self.render:
                self.env.render()
            obs, rew, done, _ = self.env.step()


def main():
    pass


if __name__ == '__main__':
    main()
