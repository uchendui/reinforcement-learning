import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from util.network import ReinforceNetworkBuilder


class Trajectory:
    def __init__(self, states, actions, rewards, num_actions):
        self.states = np.array(states)
        actions = np.array(actions).reshape(-1, 1)
        oh = OneHotEncoder(sparse=False, categories='auto')
        self.actions = oh.fit_transform(actions)
        self.q_values = np.zeros(shape=len(self.actions))
        for i in range(len(states)):
            self.q_values[i] = np.sum(rewards[i:])


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
        self.print_freq = print_freq
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.sess = sess
        self.render = render
        self.rnb = ReinforceNetworkBuilder(self.input_dim,
                                           self.output_dim,
                                           layers=(64,),
                                           activations=(tf.nn.relu, tf.nn.softmax),
                                           scope='reinforce_network')

    def act(self, observation):
        pred = self.sess.run(self.rnb.output_pred,
                             feed_dict={self.rnb.input_ph: np.reshape(observation, (1, self.input_dim))})
        return np.random.choice(range(self.output_dim), p=pred.flatten())

    def reinforce(self):
        avg_reward = 0
        for e in range(self.max_episodes):
            # Sample trajectories from our policy (run it on the robot)
            traj, reward = self.sample()
            avg_reward += reward

            # Compute and apply the gradient of the log policy multiplied by our baseline
            self.update(traj.states, traj.actions, traj.q_values)

            if e % self.print_freq == 0 and e > 0:
                new_mean_reward = avg_reward / self.print_freq
                avg_reward = 0
                print(f"-------------------------------------------------------")
                print(f"Mean {self.print_freq} Episode Reward: {new_mean_reward}")
                # print(f"Exploration fraction: {eps}")
                print(f"Total Episodes: {e}")
                # print(f"Total timesteps: {t}")
                print(f"-------------------------------------------------------")

    def update(self, states, actions, q_values):
        self.sess.run([self.rnb.opt, self.rnb.loss], feed_dict={self.rnb.input_ph: states,
                                                                self.rnb.actions_ph: actions,
                                                                self.rnb.q_values_ph: q_values})

    def sample(self):
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


def main():
    with tf.Session() as sess:
        env_name = 'CartPole-v0'
        env = gym.make(env_name)
        reinforce = TrainReinforce(env, sess, render=False, max_episodes=2000, print_freq=10)
        sess.run(tf.initialize_all_variables())
        reinforce.reinforce()


if __name__ == '__main__':
    main()
