import gym
import sys
import time
import logging
import numpy as np
import multiprocessing
import tensorflow as tf

from absl import flags
from util.network import ActorCriticNetworkBuilder
from tensorflow.core.protobuf import cluster_pb2 as cluster

FLAGS = flags.FLAGS
flags.DEFINE_string('step_msg', 'step', 'String message sent to workers to step the environment')
flags.DEFINE_string('reset_msg', 'reset', 'String message sent to workers to reset the environment')
flags.DEFINE_string('stop_msg', 'stop', 'String message sent to workers to stop')
FLAGS(sys.argv)


class Global(multiprocessing.Process):
    def __init__(self, conns, server, cluster_config):
        super(Global, self).__init__()
        self.conns = conns
        self.server = server
        self.cluster_config = cluster_config
        pass

    def run(self):
        with tf.Session(target=self.server.target, config=self.cluster_config) as sess:
            with tf.device("/job:ps/task:0"):
                x = 2
                sess.run(tf.report_uninitialized_variables())
                sess.run(tf.global_variables_initializer())
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.report_uninitialized_variables())

    pass


class Worker(multiprocessing.Process):
    def __init__(self, id, conn, env_name, cluster_config, server, input_dim, output_dim, render):
        super(Worker, self).__init__()
        self.id = id
        self.conn = conn
        # self.cluster = cluster
        self.cluster_config = cluster_config
        self.server = server
        self.env = gym.make(env_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.render = render

        # Servers created within the same cluster can share variables.
        # self.server = tf.distribute.Server(self.cluster, job_name='worker', task_index=0)

    def reset(self):
        # self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)
        print('reset')
        # self.env.reset()
        # for i in range(5000):
        #     if self.render:
        #         self.env.render()
        #     obs, rew, done, _ = self.env.step(self.env.action_space.sample())
        #     time.sleep(0.01)
        #     if done:
        #         obs = self.env.reset()

    def act(self, observation):
        return self.env.action_space.sample()

    def step(self, observation):
        action = self.act(observation)
        obs, rew, done, _ = self.env.step(action)

    def run(self):
        # ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)
        # self.server = tf.train.Server(self.server.target, job_name='worker', task_index=self.id)
        self.sess = tf.Session(target=self.server.target, config=self.cluster_config)
        print('Before session run report')
        x = self.sess.run(tf.report_uninitialized_variables())
        print('After session run report')

        print(x)
        while len(self.sess.run(tf.report_uninitialized_variables())) > 0:
            print("Worker %d: waiting for variable initialization..." % self.id)
            time.sleep(1.0)
        print("Worker %d: variables initialized" % self.id)
        # # Wait for commands to be passed
        # while True:
        #     msg = self.conn.recv()
        #     if msg == FLAGS.stop_msg:
        #         # self.server.join()
        #         break
        #     elif msg == FLAGS.reset_msg:
        #         self.reset()
        #     elif msg == FLAGS.step_msg:
        #         pass
        #     print(f'{msg} from worker {self.id}')


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

        self.create_parameter_server()

    def create_parameter_server(self):
        # Create parameter server: http://amid.fish/distributed-tensorflow-a-gentle-introduction
        # Setting the port to 0 lets the os choose an open port
        # jobs = {'global': ['127.0.0.1:0'], 'worker': ['127.0.0.1:0' for i in range(self.num_workers)]}
        jobs = {"tmp": ["localhost:0"]}
        tmp_cluster = tf.train.ClusterSpec(jobs)

        # self.cluster = tf.train.ClusterSpec(jobs)
        self.ac = ActorCriticNetworkBuilder(self.input_dim, self.output_dim)
        # self.server = tf.distribute.Server(tmp_cluster,
        #                                    job_name='tmp',
        #                                    task_index=0)
        worker_servers = [tf.distribute.Server(tmp_cluster, job_name='tmp', task_index=i) for i in
                          range(self.num_workers)]
        gbl_server = tf.distribute.Server(tmp_cluster,
                                          job_name='tmp',
                                          task_index=0)

        cluster_def = cluster.ClusterDef()
        gbl_job = cluster_def.job.add()
        gbl_job.name = 'global'
        gbl_job.tasks[0] = gbl_server.target[len('grpc://'):]  # strip off `grpc://` prefix

        for i in range(self.num_workers):
            worker_job = cluster_def.job.add()
            worker_job.name = 'worker'
            worker_job.tasks[i] = worker_servers[i].target[len('grpc://'):]

        self.cluster_config = tf.ConfigProto(cluster_def=cluster_def)
        self.gbl = Global(self.par_connections, cluster_config=self.cluster_config, server=gbl_server)
        self.workers = [Worker(
            id=i,
            server=worker_servers[i],
            cluster_config=self.cluster_config,
            conn=self.child_connections[i],
            env_name=self.env_name,
            render=self.render,
            input_dim=self.input_dim,
            output_dim=self.output_dim) for i in range(self.num_workers)]

    def learn(self):
        for t in range(self.max_steps):
            pass

    def send_message(self, msg):
        for conn in self.par_connections:
            conn.send(msg)

    def start(self):
        # tmp_cluster = tf.train.ClusterSpec({"tmp": ["localhost:0"]})
        #
        # ps = tf.train.Server(tmp_cluster, job_name="tmp", task_index=0)
        # worker = tf.train.Server(tmp_cluster, job_name="tmp", task_index=0)
        #
        # print("PS: {0}".format(ps.target))
        # print("Worker: {0}".format(worker.target))
        #
        # cluster_def = cluster.ClusterDef()
        # ps_job = cluster_def.job.add()
        # ps_job.name = 'ps'
        # ps_job.tasks[0] = ps.target[len('grpc://'):]  # strip off `grpc://` prefix
        #
        # worker_job = cluster_def.job.add()
        # worker_job.name = 'worker'
        # worker_job.tasks[0] = worker.target[len('grpc://'):]
        #
        # config = tf.ConfigProto(cluster_def=cluster_def)

        # sess = tf.Session(target=self.server.target, config=self.cluster_config)

        self.gbl.start()
        time.sleep(30)

        for w in self.workers:
            w.start()

        # sess2 = tf.Session(target=server2.target)
        # sess1.run(tf.report_uninitialized_variables())
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.report_uninitialized_variables())

        for w in self.workers:
            w.join()
        self.gbl.join()
        # sess.close()


def main():
    a2c = TrainA2C(1, render=True, env_name='CartPole-v0')
    a2c.start()


if __name__ == '__main__':
    main()
