import os
import sys
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util.network import DQNPolicyBuilder
from util.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'CartPole-v0', 'Environment name')
flags.DEFINE_string('save_path', './checkpoints', 'Save  location for the model')
flags.DEFINE_string('log_path', '../logging/dqn', 'Location to log training data')
flags.DEFINE_float('learning_rate', 0.001, 'network learning rate')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_integer('target_update', 1000, 'Steps before we update target network')
flags.DEFINE_integer('buffer_sort', 10000, 'Steps before we sort the prioritized replay buffer')
flags.DEFINE_integer('batch_size', 32, 'Number of training examples in the batch')
flags.DEFINE_integer('buffer_capacity', 10000, 'Number of transitions to keep in replay buffer')
flags.DEFINE_integer('max_steps', 50000, 'Maximum number of training steps')
flags.DEFINE_integer('print_freq', 1, 'Episodes between displaying log info')
flags.DEFINE_integer('action_repeat', 1, 'Number of times to repeat action')
flags.DEFINE_string('load_path', None, 'Load location for the model')
flags.DEFINE_float('min_eps', 0.1, 'minimum for epsilon greedy exploration')
flags.DEFINE_float('max_eps', 1.0, 'maximum for epsilon greedy exploration')
flags.DEFINE_float('alpha', 0.7, 'Prioritized replay scaling factor. Determines how much prioritization is used')
flags.DEFINE_float('beta', 0.5, 'Prioritized replay importance sampling weighting. ')
flags.DEFINE_float('eps_decay', -1e-4, 'decay schedule for epsilon')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_boolean('prioritized', False, 'Use prioritized replay buffer.')


class TrainDQN:
    def __init__(self,
                 env,
                 sess,
                 learning_rate=1e-3,
                 seed=1234,
                 gamma=0.99,
                 alpha=0.7,
                 beta=0.5,
                 max_eps=1.0,
                 min_eps=0.1,
                 render=False,
                 print_freq=20,
                 load_path=None,
                 save_path=None,
                 batch_size=32,
                 log_dir='logs/train',
                 buffer_sort=1000,
                 max_steps=100000,
                 buffer_capacity=None,
                 eps_decay_rate=-0.0001,
                 prioritized=False,
                 target_update_freq=1000,
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
        self.env_name = env.unwrapped.spec.id
        self.prioritized = prioritized
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.max_steps = max_steps
        self.buffer_sort = buffer_sort
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.alpha = alpha
        self.beta = beta
        self.eps_decay_rate = eps_decay_rate
        self.max_episode_len = self.env.spec.max_episode_steps
        self.render = render
        self.print_freq = print_freq
        self.rewards = []
        self.metrics = []
        self.save_path = save_path
        self.load_path = load_path
        self.batch_size = batch_size
        self.num_updates = 0
        self.gamma = gamma
        if prioritized:
            self.buffer = PrioritizedReplayBuffer(
                capacity=max_steps // 2 if buffer_capacity is None else buffer_capacity, alpha=self.alpha, )
            self.beta_step = (1 - self.beta) / (self.max_steps - self.batch_size)
        else:
            self.buffer = ReplayBuffer(capacity=max_steps // 2 if buffer_capacity is None else buffer_capacity)

        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate

        with tf.variable_scope('q_network'):
            self.q_network = DQNPolicyBuilder(self.input_dim, self.output_dim)
            q_summaries = self.q_network.add_summaries()
            with tf.variable_scope('target_network'):
                target_network = DQNPolicyBuilder(self.input_dim, self.output_dim)
            self.q_network.set_target_network(target_network)
        self.merged = tf.summary.merge(q_summaries)
        self.train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

    def learn(self):
        """Learns via Deep-Q-Networks (DQN)"""
        obs = self.env.reset()
        mean_reward = None
        total_reward = 0
        ep = 0
        ep_len = 0
        rand_actions = 0
        for t in range(self.max_steps):
            # weight decay from https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
            eps = self.min_eps + (self.max_eps - self.min_eps) * np.exp(
                self.eps_decay_rate * t)
            if self.render:
                self.env.render()

            # Take exploratory action with probability epsilon
            if np.random.uniform() < eps:
                action = self.env.action_space.sample()
                rand_actions += 1
            else:
                action = self.act(obs)

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

                # Add reward summary
                summary = tf.Summary()
                summary.value.add(tag=f'Episode Reward', simple_value=total_reward)
                self.train_writer.add_summary(summary, self.num_updates)
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
                    print(f"Exploration fraction: {eps}")
                    print(f"Total Episodes: {ep}")
                    print(f"Total timesteps: {t}")
                    print(f"-------------------------------------------------------")

                    # Add reward summary
                    summary = tf.Summary()
                    summary.value.add(tag=f'Mean {self.print_freq} Episode Reward',
                                      simple_value=new_mean_reward)
                    summary.value.add(tag=f'Epsilon', simple_value=eps)
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
        pred = self.sess.run([self.q_network.out_pred],
                             feed_dict={self.q_network.in_ph: np.reshape(observation, (1, self.input_dim))})
        return np.argmax(pred)

    def update(self):
        """Applies gradients to the Q network computed from a minibatch of self.batch_size."""
        if self.batch_size <= self.buffer.size():
            self.num_updates += 1

            # Update the Q network with model parameters from the target network
            if self.num_updates % self.target_update_freq == 0:
                self.sess.run(self.q_network.update_target_network_op)
                print('Updated Target Network')

            # Sample random minibatch of transitions from the replay buffer
            if self.prioritized:
                # Periodically sort the replay buffer
                if self.num_updates % self.buffer_sort == 0:
                    self.buffer.heapify()

                # Linearly anneal prioritized replay
                self.beta += self.beta_step
                (
                    priorities, count, states, action, reward, next_states,
                    done), indices, is_weights = self.buffer.sample(
                    self.batch_size, self.beta)
                next_state_pred = self.gamma * self.sess.run(self.q_network.target_network.out_pred,
                                                             feed_dict=
                                                             self.q_network.target_network.get_input_feed_dict(
                                                                 next_states, action))

                # Adjust the targets for non-terminal states
                r = reward.reshape(-1, 1)
                targets = np.copy(r)
                loc = np.argwhere(done != True).flatten()
                if len(loc) > 0:
                    max_q = np.amax(next_state_pred, axis=1)
                    targets[loc] = np.add(
                        targets[loc],
                        max_q[loc].reshape(max_q[loc].shape[0], 1),
                        casting='unsafe')
                state_preds = self.sess.run(self.q_network.action_values,
                                            feed_dict=self.q_network.get_input_feed_dict(states, action))
                td_errors = np.abs(targets.flatten() - state_preds)
                self.buffer.update_priorities(indices, td_errors)
            else:
                states, action, reward, next_states, done = self.buffer.sample(self.batch_size)
                is_weights = np.ones(states.shape[0])
            summary = self.q_network.update(self.sess, self.merged, states, action, reward, next_states, done,
                                            is_weights=is_weights)
            self.train_writer.add_summary(summary, global_step=self.num_updates)

    def save(self):
        """Saves the Q network."""
        loc = f'{self.save_path}/{self.env_name}'
        os.makedirs(loc, exist_ok=True)
        save_loc = f'{loc}/{self.env_name}.ckpt'
        self.q_network.saver.save(self.sess, save_loc)

    def load(self):
        """Loads the Q network."""
        loc = f'{self.load_path}/{self.env_name}/{self.env_name}.ckpt'
        self.q_network.saver.restore(self.sess, loc)


def main():
    with tf.Session() as sess:
        env = gym.make(FLAGS.env_name)
        dqn = TrainDQN(env=env,
                       sess=sess,
                       learning_rate=FLAGS.learning_rate,
                       gamma=FLAGS.gamma,
                       print_freq=FLAGS.print_freq,
                       beta=FLAGS.beta,
                       alpha=FLAGS.alpha,
                       target_update_freq=FLAGS.target_update,
                       batch_size=FLAGS.batch_size,
                       seed=FLAGS.seed,
                       buffer_capacity=FLAGS.buffer_capacity,
                       render=FLAGS.render,
                       max_steps=FLAGS.max_steps,
                       min_eps=FLAGS.min_eps,
                       max_eps=FLAGS.max_eps,
                       prioritized=FLAGS.prioritized,
                       eps_decay_rate=FLAGS.eps_decay,
                       log_dir=FLAGS.log_path,
                       buffer_sort=FLAGS.buffer_sort,
                       save_path=FLAGS.save_path,
                       load_path=FLAGS.load_path)
        sess.run(tf.global_variables_initializer())
        dqn.learn()


if __name__ == '__main__':
    main()
