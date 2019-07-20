import gym
import sys
import time
import logging
import numpy as np
import multiprocessing
import tensorflow as tf

from absl import flags
from util.ports import find_open_ports
from util import network, ports

FLAGS = flags.FLAGS
flags.DEFINE_string('step_msg', 'step', 'String message sent to workers to step the environment')
flags.DEFINE_string('reset_msg', 'reset', 'String message sent to workers to reset the environment')
flags.DEFINE_string('stop_msg', 'stop', 'String message sent to workers to stop')
FLAGS(sys.argv)


class Global(multiprocessing.Process):
    def __init__(self, child_pipes, cluster, max_steps=100000):
        super(Global, self).__init__()
        self.cluster = cluster
        self.child_pipes = child_pipes
        self.max_steps = max_steps
        # self.ports = find_open_ports(len(child_pipes))
        x = 2


class Worker(multiprocessing.Process):
    def __init__(self, id, conn, env_name, render, cluster):
        super(Worker, self).__init__()
        self.id = id
        self.cluster = cluster
        self.conn = conn
        self.env_name = env_name
        self.render = render

    def reset(self):
        self.env = gym.make(self.env_name)
        self.env.reset()
        for i in range(100):
            if self.render:
                self.env.render()
            obs, rew, done, _ = self.env.step(self.env.action_space.sample())
            time.sleep(0.01)
            if done:
                obs = self.env.reset()

    def act(self, observation):
        return self.env.action_space.sample()

    def step(self, observation):
        action = self.act(observation)
        obs, rew, done, _ = self.env.step(action)

    def rollout(self):
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

    def run(self):
        self.server = tf.train.Server(self.cluster, job_name='workers', task_index=self.id)

        # Wait for commands to be passed
        while True:
            msg = self.conn.recv()
            if msg == FLAGS.stop_msg:
                # self.server.join()
                break
            elif msg == FLAGS.reset_msg:
                self.reset()
            elif msg == FLAGS.step_msg:
                pass
            print(f'{msg} from worker {self.id}')


class TrainA2C:
    def __init__(self, num_workers, env_name, render=False):
        self.par_connections, self.child_connections = zip(*[multiprocessing.Pipe() for i in range(num_workers)])

        # Create parameter server
        tasks = []
        for i in range(1, num_workers + 1):
            port = 1001 + i
            host = f'localhost:{port}'
            tasks.append(host)
        jobs = {'workers': tasks, 'global': [
            "localhost:1001"
        ]}

        self.cluster = tf.train.ClusterSpec(jobs)
        self.server = tf.train.Server(self.cluster, job_name='global', task_index=0)
        self.workers = [
            Worker(id=i, cluster=self.cluster, conn=self.child_connections[i], env_name=env_name, render=render)
            for i in
            range(num_workers)]
        # self.glbl = Global(par_connections, cluster=self.cluster)

    def send_message(self, msg):
        for conn in self.par_connections:
            conn.send(msg)

    def start(self):
        for worker in self.workers:
            worker.start()

        self.send_message(FLAGS.reset_msg)
        self.send_message(FLAGS.stop_msg)

        for worker in self.workers:
            worker.join()


def main():
    a2c = TrainA2C(1, env_name='CartPole-v0')
    a2c.start()


if __name__ == '__main__':
    main()
