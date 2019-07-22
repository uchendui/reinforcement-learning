import gym
import sys
import time
import logging
import numpy as np
import multiprocessing
import tensorflow as tf

sys.path.append('../')

from absl import flags
from util.ports import find_open_ports
from util.network import ActorCriticNetworkBuilder
from util.replay_buffer import ReplayBuffer
from tensorflow.core.protobuf import cluster_pb2 as cluster

MSG_STOP = 'stop'
MSG_STEP = 'step'
MSG_RESET = 'reset'


class Worker(multiprocessing.Process):
    def __init__(self, worker_id,
                 conn,
                 env_name,
                 cluster_config,
                 input_dim,
                 output_dim,
                 render,
                 max_episodes=10000,
                 gamma=0.99,
                 print_freq=10,
                 save_path=None,
                 load_path=None):
        super(Worker, self).__init__()
        self.id = worker_id
        self.conn = conn
        self.cluster = cluster_config
        self.env = gym.make(env_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.render = render
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.print_freq = print_freq
        self.save_path = save_path

    def reset(self):
        self.env.reset()
        for i in range(1000):
            if self.render:
                self.env.render()
            obs, rew, done, _ = self.env.step(self.env.action_space.sample())
            time.sleep(0.01)
            if done:
                obs = self.env.reset()

    def update(self, sess, states, reward):
        """Takes a single gradient step using a trajectory.
        Args:
            states: array of visited states
            actions: array
            q_values: array of q values corresponding to each state
        """

        # Calculate discounted predictions for the subsequent states using target network
        state_values = self.gamma * sess.run(self.ac.critic.output_pred,
                                             feed_dict={self.ac.critic.input_ph: states}, )

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

        # Update the policy and the value function
        # sess.run()
        sess.run([self.ac.actor.opt, self.ac.actor.loss, self.ac.critic.opt, self.ac.critic.loss],
                 feed_dict={self.ac.actor.input_ph: states, self.ac.critic.input_ph: states,
                            self.ac.actor.q_values_ph: advantage, self.ac.critic.target_ph: targets})

        # self.sess.run([self.rnb.opt, self.rnb.loss], feed_dict={self.rnb.input_ph: states,
        #                                                         self.rnb.actions_ph: actions,
        #                                                         self.rnb.q_values_ph: q_values})

    def act(self, sess, obs):
        pred = sess.run(self.ac.actor.output_pred, feed_dict={self.ac.actor.input_ph: obs})
        return np.random.choice(range(self.output_dim), p=pred.flatten())

    def sample(self, sess):
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

            action = self.act(sess, obs)
            obs, rew, done, _ = self.env.step(action)

            total_reward += rew
            ep_len += 1

            rewards.append(rew)
            actions.append(action)
        return np.array(states), np.array(rewards), total_reward

    def learn(self, sess):
        obs = self.env.reset()
        mean_reward = None
        total_reward = 0
        ep = 0
        ep_len = 0
        rand_actions = 0
        for e in range(self.max_episodes):
            # Sample trajectories from our policy (run it on the robot)
            states, rewards, reward = self.sample(sess)
            total_reward += reward

            self.update(sess, states, rewards)
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
                    # print(f'Location: {self.save_path}')
                    # self.save()
                    mean_reward = new_mean_reward

            pass
        pass

    def run(self):
        server = tf.train.Server(self.cluster, job_name='worker', task_index=self.id)
        with tf.Session(target=server.target) as sess:
            with tf.device("/job:global/task:0"):
                self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)
                self.global_counter = tf.get_variable('global_counter', dtype=tf.int32)

            while len(sess.run(tf.report_uninitialized_variables())) > 0:
                print("Worker %d: waiting for variable initialization..." % self.id)
                time.sleep(1.0)
            print("Worker %d: variables initialized" % self.id)

            self.learn(sess)


class TrainA2C:
    def __init__(self, num_workers, env_name, max_steps=40000, render=False):
        self.par_connections, self.child_connections = zip(*[multiprocessing.Pipe() for i in range(num_workers)])
        self.max_steps = max_steps
        self.render = render
        self.num_workers = num_workers
        env = gym.make(env_name)
        self.env_name = env_name
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        with tf.device("/job:global/task:0"):
            self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)
            self.global_counter = tf.Variable(0, dtype=tf.int32, name='global_counter')
        self.create_parameter_server()

    def create_parameter_server(self):
        # Create parameter server: http://amid.fish/distributed-tensorflow-a-gentle-introduction

        ports = find_open_ports(self.num_workers + 1)
        jobs = {'global': [f'127.0.0.1:{ports[0]}'],
                'worker': [f'127.0.0.1:{ports[i + 1]}' for i in range(self.num_workers)]}

        self.cluster = tf.train.ClusterSpec(jobs)
        self.server = tf.distribute.Server(self.cluster,
                                           job_name='global',
                                           task_index=0)
        self.workers = [Worker(
            worker_id=i,
            # server=worker_servers[i],
            cluster_config=self.cluster,
            conn=self.child_connections[i],
            env_name=self.env_name,
            render=self.render,
            input_dim=self.input_dim,
            output_dim=self.output_dim) for i in range(self.num_workers)]

    def learn(self, sess):
        # Reset the environ
        mean_reward = None
        total_reward = 0
        ep = 0

        for t in range(self.max_steps):
            pass

    def send_message(self, msg):
        for conn in self.par_connections:
            conn.send(msg)

    def start(self):
        for w in self.workers:
            w.start()
        with tf.Session(target=self.server.target) as sess:
            sess.run(tf.global_variables_initializer())
            self.learn(sess)
            self.send_message(MSG_RESET)
            self.send_message(MSG_STOP)

        for w in self.workers:
            w.join()
        # sess.close()


def main():
    a2c = TrainA2C(1, render=True, env_name='CartPole-v0')
    a2c.start()


if __name__ == '__main__':
    main()
