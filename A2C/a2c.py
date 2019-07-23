import os
import gym
import sys
import time
import json
import numpy as np
import multiprocessing as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.path.append('../')

from util.ports import find_open_ports
from util.network import ActorCriticNetworkBuilder
from util.misc import to_one_hot

MSG_STOP = 'stop'
MSG_STEP = 'step'
MSG_RESET = 'reset'


class Worker(mp.Process):
    def __init__(self,
                 worker_id,
                 conn,
                 exp_queue,
                 rew_queue,
                 env_name,
                 input_dim,
                 output_dim,
                 render):
        super(Worker, self).__init__()
        self.id = worker_id
        self.env = gym.make(env_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.render = render
        self.conn = conn
        self.exp_queue = exp_queue
        self.rew_queue = rew_queue
        self.episode_reward = 0
        self.obs = self.env.reset()

    def run(self):
        """This method is called once a 'Worker' process is started."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.create_shared_variables()
        self.listen_for_commands()

    def create_shared_variables(self):
        """Creates shared variables and a tensorflow server session."""
        config = json.loads(os.environ.get('TF_CONFIG'))
        cluster = tf.train.ClusterSpec(config)
        self.server = tf.distribute.Server(cluster, job_name='worker', task_index=self.id)
        with tf.device("/job:global/task:0"):
            self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)

        print(f'Worker {self.id}: Created variables')
        self.sess = tf.Session(target=self.server.target)
        print(f'Worker {self.id}: Session Created')

        while len(self.sess.run(tf.report_uninitialized_variables())) > 0:
            print(f'Worker {self.id}: waiting for variable initialization...')
            time.sleep(1.0)
        print(f'Worker {self.id}: Variables initialized')

    def listen_for_commands(self):
        """Listens for commands from the global worker."""
        while True:
            msg = self.conn.recv()
            if msg == MSG_STOP:
                break
            elif msg == MSG_RESET:
                pass
            elif msg == MSG_STEP:
                self.step()

    def act(self, observation):
        """Select an action according to the policy."""
        pred = self.sess.run(self.ac.actor.output_pred,
                             feed_dict={self.ac.actor.input_ph: np.reshape(observation, (1, self.input_dim))})
        return np.random.choice(range(self.output_dim), p=pred.flatten())

    def step(self):
        """Performs a single step of the environment and adds a transition to the experience queue."""
        if self.render:
            self.env.render()
        action = self.act(self.obs)
        obs, rew, done, _ = self.env.step(action)
        self.episode_reward += rew
        self.exp_queue.put((self.obs, action, rew, obs, done))
        self.obs = obs
        if done:
            self.obs = self.env.reset()
            self.rew_queue.put(self.episode_reward)
            self.episode_reward = 0


class TrainA2C:
    def __init__(self,
                 num_workers,
                 env_name,
                 gamma=0.99,
                 max_steps=40000,
                 print_freq=10,
                 render=False,
                 save_path=None,
                 load_path=None):
        self.max_steps = max_steps
        self.gamma = gamma
        self.render = render
        self.print_freq = print_freq
        self.num_workers = num_workers
        env = gym.make(env_name)
        self.env_name = env_name
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.par_connections, self.child_connections = zip(*[mp.Pipe() for i in range(num_workers)])
        self.transition_queue = mp.Queue()
        self.rew_queue = mp.Queue()
        self.workers = [Worker(
            worker_id=i,
            conn=self.child_connections[i],
            exp_queue=self.transition_queue,
            rew_queue=self.rew_queue,
            env_name=self.env_name,
            render=self.render,
            input_dim=self.input_dim,
            output_dim=self.output_dim) for i in range(self.num_workers)]
        self.create_config()
        self.save_path = save_path
        if load_path is not None:
            self.ac.saver.restore(self.sess, load_path)
            print(f'Successfully loaded model from {load_path}')

    def create_config(self):
        """Creates distributed tensorflow configuration"""

        ports = find_open_ports(self.num_workers + 1)
        jobs = {'global': [f'127.0.0.1:{ports[0]}'],
                'worker': [f'127.0.0.1:{ports[i + 1]}' for i in range(self.num_workers)]}
        os.environ['TF_CONFIG'] = json.dumps(jobs)

        cluster = tf.train.ClusterSpec(jobs)
        self.server = tf.distribute.Server(cluster, job_name='global', task_index=0)
        self.sess = tf.Session(target=self.server.target)
        with tf.device("/job:global/task:0"):
            self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)

    def get_transitions(self):
        """Waits for every worker to sample the environment.

        Returns: list of states, actions, rewards, next_states, and done for every transition.
        """
        not_rec = self.num_workers
        samples = []
        while not_rec > 0:
            while not self.transition_queue.empty():
                samples.append(self.transition_queue.get())
                not_rec -= 1

        return list(map(np.array, zip(*samples)))

    def update(self, states, action, reward, next_states, done):
        """Updates the actor and critic networks.

        Args:
            states: array of states visited
            action: array of integer ids of selected actions
            reward: array of rewards received
            next_states: array of resulting states after taking action
            done: boolean array denoting if the corresponding state was terminal
        """
        # Adjust shapes for batch input to the models
        action = to_one_hot(action, self.output_dim)
        reward = reward.reshape(len(reward), 1)

        #######################################################
        # Update value function network (critic)
        #######################################################
        next_state_values = self.gamma * self.sess.run(self.ac.critic.output_pred,
                                                       feed_dict={self.ac.critic.input_ph: next_states})
        state_values = self.sess.run(self.ac.critic.output_pred, feed_dict={self.ac.critic.input_ph: states})

        # Adjust the targets for non-terminal states
        targets = np.copy(reward)
        loc = np.argwhere(done != True).flatten()
        if len(loc) > 0:
            targets[loc] = np.add(
                targets[loc],
                next_state_values[loc].reshape(next_state_values[loc].shape[0], 1), casting='unsafe')

        _, value_loss = self.sess.run([self.ac.critic.opt, self.ac.critic.loss],
                                      feed_dict={self.ac.critic.input_ph: states,
                                                 self.ac.critic.target_ph: targets})

        #######################################################
        # Update actor network
        #######################################################
        advantage = (reward + self.gamma * next_state_values) - state_values
        _, policy_loss = self.sess.run([self.ac.opt, self.ac.loss], feed_dict={self.ac.actor.input_ph: states,
                                                                               self.ac.advantage_ph: advantage,
                                                                               self.ac.actor.actions_ph: action})

    def learn(self):
        """Trains an agent via advantage actor-critic"""
        self.sess.run(tf.global_variables_initializer())
        mean_reward = None
        ep = 0
        for t in range(self.max_steps):
            # Sample from each environment
            self.send_message(MSG_STEP)

            # Aggregate transitions
            batch = self.get_transitions()

            # Update the actor and critic models
            self.update(*batch)

            # Print rewards
            if self.rew_queue.qsize() >= self.print_freq:
                ep += self.print_freq
                new_mean_reward = 0
                for i in range(self.print_freq):
                    new_mean_reward += self.rew_queue.get()
                new_mean_reward = new_mean_reward / self.print_freq
                print(f"-------------------------------------------------------")
                print(f"Mean {self.print_freq} Episode Reward: {new_mean_reward}")
                print(f"Total Episodes: {ep}")
                print(f"Total timesteps: {(t + 1) * self.num_workers}")
                print(f"-------------------------------------------------------")
        self.send_message(MSG_STOP)

    def send_message(self, msg):
        """Sends a message to all of the worker processes via pipes.
        Args:
            msg: constant to to send to worker pipes.
        """
        for conn in self.par_connections:
            conn.send(msg)

    def start(self):
        """Starts the worker processes and waits until they are done."""
        for i, w in enumerate(self.workers):
            w.daemon = True
            w.start()
            if i < self.num_workers - 1:
                # Need to separate workers starting due to resource usage
                # this can be adjusted for faster/slower machines
                time.sleep(0.5)
        self.learn()
        for w in self.workers:
            w.join()
            w.terminate()
        sys.exit()

    def save(self):
        """Saves the actor and critic networks."""
        self.ac.actor.saver.save(self.sess, self.save_path)
        self.ac.critic.saver.save(self.sess, self.save_path)
        # self.rnb.saver.save(self.sess, self.save_path)

    def load(self):
        """Loads the actor and critic networks."""
        self.ac.actor.saver.restore(self.sess, self.save_path)
        self.ac.critic.saver.restore(self.sess, self.save_path)


def main():
    # The default method, "fork", copies over the tensorflow module from the parent process
    #   which is problematic w.r.t GPU resources
    mp.set_start_method('spawn')
    t = TrainA2C(32, 'CartPole-v0', max_steps=50000, print_freq=100, render=False)
    t.start()


if __name__ == '__main__':
    main()
