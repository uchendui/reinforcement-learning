import os
import gym
import sys
import time
import json
import numpy as np
import tensorflow as tf
import multiprocessing as mp

sys.path.append('../')

from util.ports import find_open_ports
from util.network import ActorCriticNetworkBuilder
from util.misc import to_one_hot

MSG_STOP = 'stop'
MSG_STEP = 'step'
MSG_RESET = 'reset'


class Worker(mp.Process):
    def __init__(self, worker_id,
                 conn,
                 experiences,
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
        self.experiences = experiences
        self.obs = self.env.reset()

    def act(self, observation):
        """Select an action according to the policy."""
        pred = self.sess.run(self.ac.actor.output_pred,
                             feed_dict={self.ac.actor.input_ph: np.reshape(observation, (1, self.input_dim))})
        return np.random.choice(range(self.output_dim), p=pred.flatten())

    def step(self):
        """Performs a single step of the environment and adds a transition to the experience queue."""
        # print(f'Worker {self.id} Parameters:')
        # print(self.sess.run(tf.trainable_variables())[0][0])

        if self.render:
            self.env.render()
        action = self.act(self.obs)
        obs, rew, done, _ = self.env.step(action)
        self.experiences.put((self.obs, action, rew, obs, done))
        self.obs = obs
        if done:
            self.obs = self.env.reset()

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
            # print(f'{msg} from worker {self.id}')

    def run(self):
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


class TrainA2C:
    def __init__(self, num_workers, env_name, max_steps=40000, render=False):
        self.max_steps = max_steps
        self.render = render
        self.num_workers = num_workers
        env = gym.make(env_name)
        self.env_name = env_name
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.par_connections, self.child_connections = zip(*[mp.Pipe() for i in range(num_workers)])
        self.experiences = mp.Queue()
        self.workers = [Worker(
            worker_id=i,
            conn=self.child_connections[i],
            experiences=self.experiences,
            env_name=self.env_name,
            render=self.render,
            input_dim=self.input_dim,
            output_dim=self.output_dim) for i in range(self.num_workers)]
        self.create_config()

    def create_config(self):
        """Creates distributed tensorflow configuration"""

        ports = find_open_ports(self.num_workers + 1)
        jobs = {'global': [f'127.0.0.1:{ports[0]}'],
                'worker': [f'127.0.0.1:{ports[i + 1]}' for i in range(self.num_workers)]}
        os.environ['TF_CONFIG'] = json.dumps(jobs)

        # config = json.loads(os.environ.get('TF_CONFIG'))
        cluster = tf.train.ClusterSpec(jobs)
        self.server = tf.distribute.Server(cluster, job_name='global', task_index=0)
        self.sess = tf.Session(target=self.server.target)
        with tf.device("/job:global/task:0"):
            self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)

    def get_transitions(self):
        """Waits for every environment to be sampled."""
        not_rec = self.num_workers
        samples = []
        while not_rec > 0:
            while not self.experiences.empty():
                samples.append(self.experiences.get())
                not_rec -= 1

        return list(map(np.array, zip(*samples)))

    def update(self, states, action, reward, next_states, done):
        # Adjust shapes for input to the models
        action = to_one_hot(action, self.output_dim)
        reward = reward.reshape(len(reward), 1)

        #######################################################
        # Update value function network
        #######################################################
        next_state_values = 0.99 * self.sess.run(self.ac.critic.output_pred,
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
        # Update critic network
        #######################################################
        advantage = (reward + 0.99 * next_state_values) - state_values
        _, policy_loss = self.sess.run([self.ac.opt, self.ac.loss], feed_dict={self.ac.actor.input_ph: states,
                                                                               self.ac.advantage_ph: advantage,
                                                                               self.ac.actor.actions_ph: action})
        print(f'Policy loss:{policy_loss}\nCritic loss:{value_loss}')

    def learn(self):
        """Trains an agent via advantage actor-critic"""
        self.sess.run(tf.global_variables_initializer())
        for t in range(40000):
            # Sample from each environment
            self.send_message(MSG_STEP)

            # Aggregate transitions
            batch = self.get_transitions()

            # Update the actor and critic models
            self.update(*batch)

            print('Processed Batch')
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
            w.start()
            if i < self.num_workers - 1:
                # Need to separate workers starting due to resource usage
                # this can be adjusted for faster/slower machines
                time.sleep(0.5)
        self.learn()
        for w in self.workers:
            w.join()


def main():
    # The default method, "fork", copies over the tensorflow module from the parent process
    #   which disables us from using GPU resources
    mp.set_start_method('spawn')
    t = TrainA2C(32, 'CartPole-v0', render=True)
    t.start()


if __name__ == '__main__':
    main()
