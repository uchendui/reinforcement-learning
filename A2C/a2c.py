import os
import gym
import sys
import time
import json
import logging
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from absl import flags
from util.ports import find_open_ports
from util.network import ActorCriticNetworkBuilder
from tensorflow.core.protobuf import cluster_pb2 as cluster

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
                # self.server.join()
                break
            elif msg == MSG_RESET:
                pass
            elif msg == MSG_STEP:
                self.step()
            print(f'{msg} from worker {self.id}')

    def run(self):
        config = json.loads(os.environ.get('TF_CONFIG'))
        cluster = tf.train.ClusterSpec(config)
        self.server = tf.distribute.Server(cluster, job_name='worker', task_index=self.id)
        self.create_parameters()
        self.listen_for_commands()

    def create_parameters(self):
        self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)
        print('Worker: Created variables')
        self.sess = tf.Session(target=self.server.target)
        print('Worker Session Created')

        while len(self.sess.run(tf.report_uninitialized_variables())) > 0:
            print("Worker %d: waiting for variable initialization..." % self.id)
            time.sleep(1.0)
        print("Worker %d: variables initialized" % self.id)


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
        self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)

    def get_transitions(self):
        """Waits for every environment to be sampled."""
        not_rec = self.num_workers
        while not_rec > 0:
            while not self.experiences.empty():
                print(self.experiences.get())
                not_rec -= 1

    def learn(self):
        """Trains an agent via advantage actor-critic"""
        self.sess.run(tf.global_variables_initializer())
        for t in range(1000):
            # Sample from each environment
            self.send_message(MSG_STEP)

            # Aggregate transitions
            self.get_transitions()
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
                time.sleep(5)  # Need to separate workers starting due to resource usage
        self.learn()
        for w in self.workers:
            w.join()


def main():
    mp.set_start_method('spawn')
    t = TrainA2C(2, 'CartPole-v0', render=True)
    t.start()


if __name__ == '__main__':
    main()
