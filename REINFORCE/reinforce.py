import gym
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util.network import ReinforceNetworkBuilder


class Trajectory:
    def __init__(self, states, actions, rewards, num_actions):
        """
        Represents a single trajectory sampled from our policy

        Args:
            states: list of states visited during the trajectory (excluding final state)
            actions: list of actions taken from each state
            rewards: list of rewards received in the trajectory
            num_actions: number of actions
        """

        self.states = np.array(states)
        self.actions = self.to_one_hot(actions, num_actions)
        self.q_values = np.zeros(shape=len(self.actions))
        mean_reward = np.mean(rewards)
        std = np.std(rewards)
        std = 1 if std == 0 else std
        for i in range(len(states)):
            self.q_values[i] = (np.sum(rewards[i:]) - mean_reward) / std

    def to_one_hot(self, indices, num_actions):
        """
        Converts a list of indices to a one-hot representation.

        For example, to_one_hot((1,3),4) -> [[0,1,0,0], [0, 0, 0, 1]]
        Args:
            indices:
            num_actions: Number of possible actions. (length of one-hot vector)
        """
        return np.eye(num_actions)[indices]


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
        """Trains an agent via vanilla policy gradient with average reward baseline.
        Args:
            env: gym.Env where our agent resides.
            sess: tensorflow session
            render: True to render the environment, else False
            max_episodes: maximum number of episodes to train for
            print_freq: Displays logging information every 'print_freq' episodes
            load_path: (str) Path to load existing model from
            save_path: (str) Path to save model during training
        """
        self.max_episodes = max_episodes
        self.print_freq = print_freq
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.sess = sess
        self.render = render
        self.save_path = save_path
        self.rewards = []

        self.rnb = ReinforceNetworkBuilder(self.input_dim,
                                           self.output_dim,
                                           layers=(32, 32,),
                                           activations=(tf.nn.relu, tf.nn.relu, tf.nn.softmax),
                                           scope='reinforce_network')

        if load_path is not None:
            self.rnb.saver.restore(sess, load_path)
            print(f'Successfully loaded model from {load_path}')

    def act(self, observation):
        pred = self.sess.run(self.rnb.output_pred,
                             feed_dict={self.rnb.input_ph: np.reshape(observation, (1, self.input_dim))})
        return np.random.choice(range(self.output_dim), p=pred.flatten())

    def reinforce(self):
        """Trains an agent via vanilla policy gradient"""
        total_reward = 0
        mean_reward = None
        for e in range(self.max_episodes):
            # Sample trajectories from our policy (run it on the robot)
            traj, reward = self.sample()
            total_reward += reward
            self.rewards.append(reward)

            # Compute and apply the gradient of the log policy multiplied by our baseline
            self.update(traj.states, traj.actions, traj.q_values)

            if e % self.print_freq == 0 and e > 0:
                new_mean_reward = total_reward / self.print_freq
                total_reward = 0
                print(f"-------------------------------------------------------")
                print(f"Mean {self.print_freq} Episode Reward: {new_mean_reward}")
                # print(f"Exploration fraction: {eps}")
                print(f"Total Episodes: {e}")
                # print(f"Total timesteps: {t}")
                print(f"-------------------------------------------------------")

                # Model saving inspired by Open AI Baseline implementation
                if (mean_reward is None or new_mean_reward >= mean_reward) and self.save_path is not None:
                    print(f"Saving model due to mean reward increase:{mean_reward} -> {new_mean_reward}")
                    print(f'Location: {self.save_path}')
                    self.save()
                    mean_reward = new_mean_reward

    def update(self, states, actions, q_values):
        """Takes a single gradient step using a trajectory.
        Args:
            states: array of visited states
            actions: array
            q_values: array of q values corresponding to each state
        """
        self.sess.run([self.rnb.opt, self.rnb.loss], feed_dict={self.rnb.input_ph: states,
                                                                self.rnb.actions_ph: actions,
                                                                self.rnb.q_values_ph: q_values})

    def save(self):
        """Saves the network."""
        self.rnb.saver.save(self.sess, self.save_path)

    def load(self):
        """Loads the network."""
        self.rnb.saver.restore(self.sess, self.save_path)

    def sample(self):
        """Samples a single trajectory under the current policy."""
        done = False
        actions = []
        states = []
        rewards = []

        obs = self.env.reset()
        total_reward = 0
        ep_len = 0
        while not done:
            states.append(obs)
            if self.render:
                self.env.render()

            action = self.act(obs)
            obs, rew, done, _ = self.env.step(action)

            total_reward += rew
            ep_len += 1

            rewards.append(rew)
            actions.append(action)
        return Trajectory(states, actions, rewards, self.output_dim), total_reward

    def plot_rewards(self, path=None):
        """
        Plots a graph of the total rewards received per training episode
        :param path:
        :return:
        """
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close('all')


def main():
    with tf.Session() as sess:
        env_name = 'CartPole-v0'
        env = gym.make(env_name)
        reinforce = TrainReinforce(env,
                                   sess,
                                   render=False,
                                   max_episodes=5000,
                                   print_freq=10,
                                   save_path=f'checkpoints/{env_name}.ckpt')
        sess.run(tf.initialize_all_variables())
        reinforce.reinforce()
        reinforce.plot_rewards()


if __name__ == '__main__':
    main()
