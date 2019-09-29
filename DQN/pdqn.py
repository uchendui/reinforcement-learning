import sys
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import DQN.sparsemountaincar
from util.network import QNetworkBuilder
from util.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'CartPole-v0', 'Environment name')
flags.DEFINE_string('save_path', './checkpoints', 'Save  location for the model')
flags.DEFINE_string('log_path', './logs', 'Location to log training data')
flags.DEFINE_float('learning_rate', 0.001, 'network learning rate')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_integer('target_update', 10000, 'Steps before we update target network')
flags.DEFINE_integer('batch_size', 32, 'Number of training examples in the batch')
flags.DEFINE_integer('buffer_capacity', 50, 'Number of transitions to keep in replay buffer')
flags.DEFINE_integer('max_steps', 10, 'Maximum number of training steps')
flags.DEFINE_integer('max_episode_len', 10, 'Maximum length of each episode')
flags.DEFINE_integer('print_freq', 1, 'Episodes between displaying log info')
flags.DEFINE_integer('action_repeat', 1, 'Number of times to repeat action')
flags.DEFINE_string('load_path', None, 'Load location for the model')
flags.DEFINE_float('min_eps', 0.1, 'minimum for epsilon greedy exploration')
flags.DEFINE_float('max_eps', 1.0, 'maximum for epsilon greedy exploration')
flags.DEFINE_float('eps_decay', -1e-4, 'decay schedule for epsilon')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')


class TrainDQN:
    def __init__(self,
                 env,
                 sess,
                 learning_rate=1e-3,
                 seed=1234,
                 gamma=0.99,
                 max_eps=1.0,
                 min_eps=0.1,
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
        self.env_name = env.spec.id
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.max_steps = max_steps
        self.max_eps = max_eps
        self.min_eps = min_eps
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
        self.buffer = PrioritizedReplayBuffer(capacity=max_steps // 2 if buffer_capacity is None else buffer_capacity)
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate

        with tf.variable_scope('q_network'):
            self.q_network = QNetworkBuilder(self.input_dim, self.output_dim, (64,))
        with tf.variable_scope('target_network'):
            self.target_network = QNetworkBuilder(self.input_dim, self.output_dim, (64,))
        self.update_target_network = [old.assign(new) for (new, old) in
                                      zip(tf.trainable_variables('q_network'),
                                          tf.trainable_variables('target_network'))]
        self._add_summaries(log_dir)

    def _add_summaries(self, log_dir):
        tf.summary.scalar('Loss', self.q_network.loss, )
        tf.summary.scalar('Mean Estimated Value', tf.reduce_mean(self.q_network.output_pred))
        # Merge all the summaries and write them out to log_dir
        self.merged = tf.summary.merge_all()
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
            # if reward > 0:
            #     print('Got the reward')
            total_reward += reward

            # Store transition s_t, a_t, r_t, s_t+1 in replay buffer
            self.buffer.add((obs, action, reward, new_obs, done))

            # Perform learning step
            self._update()

            obs = new_obs
            ep_len += 1
            if done or ep_len >= self.max_episode_len:
                #         print("Episode Length:", ep_len)
                #         print(f"Episode {ep} Reward:{total_reward}")
                #         print(f"Random Action Percent: {rand_actions/ep_len}")
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
                        self.save()
                        mean_reward = new_mean_reward

    def act(self, observation):
        """Takes an action given the observation.
        Args:
            observation: observation from the environment
        Returns:
            integer index of the selected action
        """
        pred = self.sess.run([self.q_network.output_pred],
                             feed_dict={self.q_network.input_ph: np.reshape(observation, (1, self.input_dim))})
        return np.argmax(pred)

    def _update(self):
        """Applies gradients to the Q network computed from a minibatch of self.batch_size."""
        if self.batch_size <= len(self.buffer):
            self.num_updates += 1

            # Update the Q network with model parameters from the target network
            if self.num_updates % self.target_update_freq == 0:
                self.sess.run(self.update_target_network)

            # Sample random minibatch of transitions from the replay buffer
            sample = self.buffer.sample(self.batch_size)
            (tds, num, states, action, reward, next_states, done), inds = sample
            # Calculate discounted predictions for the subsequent states using target network
            next_state_pred = self.gamma * self.sess.run(self.target_network.output_pred,
                                                         feed_dict={self.target_network.input_ph: next_states}, )
            state_pred = self.sess.run(self.q_network.output_pred,
                                       feed_dict={self.q_network.input_ph: states}, )

            # Adjust the targets for non-terminal states
            reward = reward.reshape(len(reward), 1)
            targets = reward
            loc = np.argwhere(done != True).flatten()
            if len(loc) > 0:
                max_q = np.amax(next_state_pred, axis=1)
                targets[loc] = np.add(
                    targets[loc],
                    max_q[loc].reshape(max_q[loc].shape[0], 1),
                    casting='unsafe')

            # Compute TD Error for updating the prioritized replay buffer
            rang = np.arange(len(action))
            curr_q = state_pred[rang, action]
            td_error = np.abs(targets.flatten() - curr_q)
            self.buffer.update_priorities(indices=inds, tds=td_error)

            # Update discount factor and train model on batch
            _, loss = self.sess.run([self.q_network.opt, self.q_network.loss],
                                    feed_dict={self.q_network.input_ph: states,
                                               self.q_network.target_ph: targets.flatten(),
                                               self.q_network.action_indices_ph: action})

    def save(self):
        """Saves the Q network."""
        loc = f'{self.save_path}/{self.env_name}/{self.env_name}.ckpt'
        self.q_network.saver.save(self.sess, loc)
        print(f'Successfully saved model to: {loc}')

    def load(self):
        """Loads the Q network."""
        loc = f'{self.load_path}/{self.env_name}/{self.env_name}.ckpt'
        self.q_network.saver.restore(self.sess, loc)
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
        env_name = FLAGS.env_name
        env = gym.make(env_name)

        dqn = TrainDQN(env=env,
                       sess=sess,
                       learning_rate=FLAGS.learning_rate,
                       gamma=FLAGS.gamma,
                       print_freq=FLAGS.print_freq,
                       target_update_freq=FLAGS.target_update,
                       batch_size=FLAGS.batch_size,
                       seed=FLAGS.seed,
                       buffer_capacity=FLAGS.buffer_capacity,
                       render=FLAGS.render,
                       max_steps=FLAGS.max_steps,
                       min_eps=FLAGS.min_eps,
                       max_eps=FLAGS.max_eps,
                       eps_decay_rate=FLAGS.eps_decay,
                       max_episode_len=FLAGS.max_episode_len,
                       log_dir=FLAGS.log_path,
                       save_path=FLAGS.save_path,
                       load_path=FLAGS.load_path)
        # save_path=f'checkpoints/{env_name}.ckpt')
        sess.run(tf.initialize_all_variables())
        dqn.learn()
        dqn.plot_rewards()


if __name__ == '__main__':
    main()
