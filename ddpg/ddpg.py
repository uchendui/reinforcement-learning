import os
import sys
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util.network import DDPGPolicyBuilder
from util.replay_buffer import ReplayBuffer
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'Pendulum-v0', 'Environment name')
flags.DEFINE_string('save_path', './checkpoints', 'Save  location for the model')
flags.DEFINE_string('log_path', '../logging/ddpg', 'Location to log training data')
flags.DEFINE_float('learning_rate', 0.001, 'network learning rate')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_float('target_update', 1e-3, 'Soft target update coefficient')
flags.DEFINE_integer('batch_size', 32, 'Number of training examples in the batch')
flags.DEFINE_integer('buffer_capacity', 50000, 'Number of transitions to keep in replay buffer')
flags.DEFINE_integer('max_steps', 100000, 'Maximum number of training steps')
flags.DEFINE_integer('max_episode_len', 10000, 'Maximum length of each episode')
flags.DEFINE_integer('print_freq', 1, 'Episodes between displaying log info')
flags.DEFINE_integer('action_repeat', 1, 'Number of times to repeat action')
flags.DEFINE_string('load_path', None, 'Load location for the model')
flags.DEFINE_float('eps_decay', -1e-4, 'decay schedule for epsilon')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')


class TrainDDPG:
    def __init__(self,
                 env,
                 sess,
                 learning_rate=1e-3,
                 seed=1234,
                 gamma=0.99,
                 render=False,
                 print_freq=20,
                 load_path=None,
                 save_path=None,
                 batch_size=32,
                 log_dir='logs/train',
                 max_steps=100000,
                 buffer_capacity=None,
                 max_episode_len=2000,
                 eps_decay_rate=-0.0001,
                 target_update_coef=1000,
                 ):
        """Trains an openai gym-like environment with deep q learning.
        Args:
            env: gym.Env where our agent resides
            seed: Random seed for reproducibility
            gamma: Discount factor
            max_eps: Starting exploration factor
            min_eps: Exploration factor to decay towards
            max_episode_len: Maximum length of an individual episode
            render: True to render the environment, else False
            print_freq: Displays logging information every 'print_freq' episodes
            load_path: (str) Path to load existing model from
            save_path: (str) Path to save model during training
            max_steps: maximum number of times to sample the environment
            buffer_capacity: How many state, action, next state, reward tuples the replay buffer should store
            max_episode_len: Maximum number of timesteps in an episode
            eps_decay_rate: lambda parameter in exponential decay for epsilon
            target_update_fraction: Fraction of max_steps update the target network
        """
        np.random.seed(seed)
        self.sess = sess
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.shape[0]
        self.max_steps = max_steps
        self.eps_decay_rate = eps_decay_rate
        self.max_episode_len = max_episode_len
        self.render = render
        self.print_freq = print_freq
        self.rewards = []
        self.metrics = []
        self.save_path = save_path
        self.load_path = load_path
        self.batch_size = batch_size
        self.num_updates = 0
        self.gamma = gamma
        self.buffer = ReplayBuffer(capacity=max_steps // 2 if buffer_capacity is None else buffer_capacity)
        self.target_update_freq = target_update_coef
        self.learning_rate = learning_rate

        with tf.variable_scope('ddpg'):
            self.ddpg = DDPGPolicyBuilder(self.input_dim, self.output_dim)
            summaries = self.ddpg.add_summaries()
            with tf.variable_scope('target_ddpg'):
                target = DDPGPolicyBuilder(self.input_dim, self.output_dim)
            self.ddpg.set_target(target)
        self.merged = tf.summary.merge(summaries)
        self.train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        if self.load_path is not None:
            self.load()

    def learn(self):
        """Learns via Deep-Q-Networks (DQN)"""
        obs = self.env.reset()
        mean_reward = None
        total_reward = 0
        ep = 0
        ep_len = 0
        rand_actions = 0
        for t in range(self.max_steps):
            if self.render:
                self.env.render()

            # Take action
            # TODO: Add noise to the action chosen
            # val = np.random.normal(0, 1)
            # eps = val * np.power(self.gamma, t)
            eps = np.random.normal()
            action = self.act(obs) + eps / 2
            action = np.clip(action, a_min=self.env.action_space.low, a_max=self.env.action_space.high)
            # Execute action in emulator and observe reward and next state
            new_obs, reward, done, info = self.env.step(action)
            total_reward += reward

            # Store transition s_t, a_t, r_t, s_t+1 in replay buffer
            self.buffer.add((obs, action, reward, new_obs, done))

            # Perform learning step
            self.update()

            obs = new_obs
            ep_len += 1
            if done or ep_len >= self.max_episode_len:
                ep += 1
                ep_len = 0
                rand_actions = 0
                self.rewards.append(total_reward)
                total_reward = 0
                obs = self.env.reset()

                if ep % self.print_freq == 0 and ep > 0:
                    new_mean_reward = np.mean(self.rewards[-self.print_freq - 1:])

                    print(f"-------------------------------------------------------")
                    print(f"Mean {self.print_freq} Episode Reward: {new_mean_reward}")
                    print(f"Total Episodes: {ep}")
                    print(f"Total timesteps: {t}")
                    print(f"-------------------------------------------------------")

                    # Add reward summary
                    summary = tf.Summary()
                    summary.value.add(tag=f'Mean {self.print_freq} Episode Reward',
                                      simple_value=new_mean_reward)
                    self.train_writer.add_summary(summary, self.num_updates)

                    # Model saving inspired by Open AI Baseline implementation
                    if (mean_reward is None or new_mean_reward >= mean_reward) and self.save_path is not None:
                        print(f"Saving model due to mean reward increase:{mean_reward} -> {new_mean_reward}")
                        print(f'Location: {self.save_path}')
                        self.save()
                        mean_reward = new_mean_reward

    def act(self, observation):
        """Takes an action given the observation.
        Args:
            observation: observation from the environment
        Returns:
            integer index of the selected action
        """
        pred = self.sess.run(self.ddpg.actor_pred,
                             feed_dict={self.ddpg.in_ph: np.reshape(observation, (1, self.input_dim))})
        return pred.item()

    def update(self):
        """Applies gradients to the Q network computed from a minibatch of self.batch_size."""
        if self.batch_size <= self.buffer.size():
            self.num_updates += 1

            # Soft target network update
            self.sess.run(self.ddpg.update_target_network_op)

            # Sample random minibatch of transitions from the replay buffer
            sample = self.buffer.sample(self.batch_size)
            states, action, reward, next_states, done = sample
            summary = self.ddpg.update(self.sess, self.merged, states, action, reward, next_states, done)
            self.train_writer.add_summary(summary, global_step=self.num_updates)

    def save(self):
        """Saves the Q network."""
        loc = f'{self.save_path}/{self.env.unwrapped.spec.id}'
        os.makedirs(loc, exist_ok=True)
        self.ddpg.saver.save(self.sess, self.save_path)
        print(f'Successfully saved model to {loc}')

    def load(self):
        """Loads the Q network."""
        loc = f'{self.load_path}/{self.env.name}'
        self.ddpg.saver.restore(self.sess, loc)
        print(f'Successfully loaded model from {loc}')

    def plot_rewards(self, path=None):
        """Plots rewards per episode.
        Args:
            path: Location to save the rewards plot. If None, image will be displayed with plt.show()
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
        env = gym.make(FLAGS.env_name)

        ddpg = TrainDDPG(env=env,
                         sess=sess,
                         learning_rate=FLAGS.learning_rate,
                         gamma=FLAGS.gamma,
                         print_freq=FLAGS.print_freq,
                         target_update_coef=FLAGS.target_update,
                         batch_size=FLAGS.batch_size,
                         seed=FLAGS.seed,
                         buffer_capacity=FLAGS.buffer_capacity,
                         render=FLAGS.render,
                         max_steps=FLAGS.max_steps,
                         eps_decay_rate=FLAGS.eps_decay,
                         max_episode_len=FLAGS.max_episode_len,
                         log_dir=FLAGS.log_path,
                         save_path=FLAGS.save_path,
                         load_path=FLAGS.load_path)
        sess.run(tf.global_variables_initializer())
        ddpg.learn()
        ddpg.plot_rewards()


if __name__ == '__main__':
    main()
