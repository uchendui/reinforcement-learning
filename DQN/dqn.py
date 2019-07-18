import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util.network import QNetworkBuilder
from util.replay_buffer import ReplayBuffer


class TrainDQN:
    def __init__(self,
                 env,
                 sess,
                 lr=1e-3,
                 seed=1234,
                 gamma=0.99,
                 max_eps=1.0,
                 min_eps=0.1,
                 render=False,
                 print_freq=20,
                 load_path=None,
                 save_path=None,
                 max_steps=100000,
                 buffer_capacity=None,
                 max_episode_len=2000,
                 eps_decay_rate=-0.0001,
                 target_update_fraction=1e-2,
                 ):
        """
        Use DQN to train an open ai gym-like environment
        :param env:
        :param max_steps: Maximum number of steps to train for
        :param max_eps: Starting exploration factor
        :param min_eps: Exploration factor to decay towards
        :param max_episode_len: Maximum length of an individual episode
        :param render: True to render the environment, else False
        :param print_freq: Displays logging information every 'print_freq'
        episodes
        :return: None
        :type env: gym.Env
        """
        np.random.seed(seed)
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.sess = sess
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
        self.batch_size = 32
        self.num_updates = 0
        self.gamma = gamma
        self.buffer = ReplayBuffer(capacity=max_steps // 2 if buffer_capacity is None else buffer_capacity)
        self.target_update_freq = int(target_update_fraction * max_steps)
        self.learning_rate = lr
        self.q_network = QNetworkBuilder(self.input_dim, self.output_dim, (64,), scope='q_network')
        self.target_network = QNetworkBuilder(self.input_dim, self.output_dim, (64,), scope='target_network')
        self.update_target_network = [old.assign(new) for (new, old) in
                                      zip(tf.trainable_variables('q_network'),
                                          tf.trainable_variables('target_network'))]

        if load_path is not None:
            self.q_network.saver.restore(sess, load_path)
            print(f'Successfully loaded model from {load_path}')

    def learn(self):
        """
        Reinforcement learning via Deep Q Networks (DQN)
        """
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

                    # Model saving inspired by Open AI Baseline implementation
                    if (mean_reward is None or new_mean_reward >= mean_reward) and self.save_path is not None:
                        print(f"Saving model due to mean reward increase:{mean_reward} -> {new_mean_reward}")
                        print(f'Location: {self.save_path}')
                        # save_path = f"{self.save_path}_model"
                        self.save()
                        mean_reward = new_mean_reward

    def act(self, observation):
        """
        Take an action given the observation
        :param observation: observation of the environment
        :return:
        """

        pred = self.sess.run([self.q_network.output_pred],
                             feed_dict={self.q_network.input_ph: np.reshape(observation, (1, self.input_dim))})
        return np.argmax(pred)

    def update(self):
        """
        Update the Q network
        :return: None or evaluation metrics
        """
        if self.batch_size <= self.buffer.size():
            self.num_updates += 1

            # Update the Q network with model parameters from the target network
            if self.num_updates % self.target_update_freq == 0:
                self.sess.run(self.update_target_network)

            # Sample random minibatch of transitions from the replay buffer
            sample = self.buffer.sample(self.batch_size)
            states, action, reward, next_states, done = sample

            # Calculate discounted predictions for the subsequent states using target network
            next_state_pred = self.gamma * self.sess.run(self.target_network.output_pred,
                                                         feed_dict={self.target_network.input_ph: next_states}, )

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

            # Update discount factor and train model on batch
            _, loss = self.sess.run([self.q_network.opt, self.q_network.loss],
                                    feed_dict={self.q_network.input_ph: states,
                                               self.q_network.target_ph: targets.flatten(),
                                               self.q_network.action_indices_ph: action})

    def save(self):
        self.q_network.saver.save(self.sess, self.save_path)

    def load(self):
        self.q_network.saver.restore(self.sess, self.save_path)

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
        dqn = TrainDQN(env, sess, print_freq=10, target_update_fraction=0.01, render=False, max_steps=40000,
                       save_path=f'checkpoints/{env_name}.ckpt')
        sess.run(tf.initialize_all_variables())
        dqn.learn()
        dqn.plot_rewards()


if __name__ == '__main__':
    main()
