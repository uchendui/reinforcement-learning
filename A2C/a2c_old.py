import gym
import sys
import time
import logging
import numpy as np
import multiprocessing
import tensorflow as tf

from absl import flags
from util.ports import find_open_ports
from util.network import ActorCriticNetworkBuilder
from tensorflow.core.protobuf import cluster_pb2 as cluster

MSG_STOP = 'stop'
MSG_STEP = 'step'
MSG_RESET = 'reset'


class Worker(multiprocessing.Process):
    def __init__(self, worker_id, conn, env_name,
                 cluster_config,
                 # server,
                 input_dim, output_dim, render):
        super(Worker, self).__init__()
        self.id = worker_id
        self.conn = conn
        self.cluster = cluster_config
        self.env = gym.make(env_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.render = render

    def reset(self):
        self.env.reset()
        for i in range(1000):
            if self.render:
                self.env.render()
            obs, rew, done, _ = self.env.step(self.env.action_space.sample())
            time.sleep(0.01)
            if done:
                obs = self.env.reset()

    def act(self, sess, observation):
        return self.env.action_space.sample()

    def step(self, observation):
        action = self.act(observation)
        obs, rew, done, _ = self.env.step(action)

    def run(self):
        server = tf.train.Server(self.cluster, job_name='worker', task_index=self.id)
        with tf.Session(target=server.target) as sess:
            with tf.device("/job:global/task:0"):
                self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)

            while len(sess.run(tf.report_uninitialized_variables())) > 0:
                print("Worker %d: waiting for variable initialization..." % self.id)
                time.sleep(1.0)
            print("Worker %d: variables initialized" % self.id)

            # Wait for commands to be passed
            while True:
                msg = self.conn.recv()
                if msg == MSG_STOP:
                    # self.server.join()
                    break
                elif msg == MSG_RESET:
                    self.reset()
                elif msg == MSG_STEP:
                    self.step(observation=)
                print(f'{msg} from worker {self.id}')


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
        self.create_parameter_server()

    def create_parameter_server(self):
        # Create parameter server: http://amid.fish/distributed-tensorflow-a-gentle-introduction

        ports = find_open_ports(self.num_workers + 1)
        jobs = {'cluster': {'global': [f'127.0.0.1:{ports[0]}'],
                            'worker': [f'127.0.0.1:{ports[i + 1]}' for i in range(self.num_workers)]},}

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
    a2c = TrainA2C(2, render=False, env_name='CartPole-v0')
    a2c.start()


if __name__ == '__main__':
    main()
