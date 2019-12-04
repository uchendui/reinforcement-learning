import gym
import os
import sys
import time
import json
import numpy as np
import multiprocessing as mp
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.path.append('../')

from util.ports import find_open_ports
from util.network import A2CPolicyBuilder

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'CartPole-v0', 'Environment name')
flags.DEFINE_string('save_path', './checkpoints', 'Save  location for the model')
flags.DEFINE_string('log_path', '../logging/a2c', 'Location to log training data')
flags.DEFINE_float('learning_rate', 0.001, 'network learning rate')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_float('l2_coef', 1e-2, 'L2 Regularization coefficient')
flags.DEFINE_float('entropy', 1e-2, 'Entropy coefficient')
flags.DEFINE_integer('num_workers', 3, 'Number of workers')
flags.DEFINE_integer('max_steps', 50000, 'Maximum number of environment samples to train on')
flags.DEFINE_integer('n_steps', 32, 'Number of steps to bootstrap per update')
flags.DEFINE_integer('print_freq', 10, 'Episodes between displaying log info')
flags.DEFINE_string('load_path', None, 'Load location for the model')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_enum('extract', 'mlp', ['mlp', 'cnn'], 'Feature extraction methods')

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
                 learning_rate,
                 gamma,
                 entropy,
                 extract,
                 n_steps,
                 l2_coef,
                 render):
        super(Worker, self).__init__()
        self.id = worker_id
        self.env = gym.make(env_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.l2_coef = l2_coef
        self.gamma = gamma
        self.n_step = n_steps
        self.extract = extract
        self.entropy = entropy
        self.render = render
        self.conn = conn
        self.exp_queue = exp_queue
        self.rew_queue = rew_queue
        self.episode_reward = 0
        self.obs = self.env.reset()

    def run(self):
        """This method is called once a 'Worker' process is started."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self._create_shared_variables()
        self.listen_for_commands()

    def _create_shared_variables(self):
        """Creates shared variables and a tensorflow server session."""
        config = json.loads(os.environ.get('TF_CONFIG'))
        cluster = tf.train.ClusterSpec(config)
        self.server = tf.distribute.Server(cluster, job_name='worker', task_index=self.id)

        with tf.device("/job:global/task:0"):
            with tf.variable_scope('a2c'):
                self.ac = A2CPolicyBuilder(self.input_dim,
                                           self.output_dim,
                                           learning_rate=self.learning_rate,
                                           l2_coef=self.l2_coef,
                                           entropy=self.entropy,
                                           gamma=self.gamma,
                                           extract=self.extract)

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
        pred = self.sess.run(self.ac.out_pred,
                             feed_dict={self.ac.in_ph: np.reshape(observation, (1, self.input_dim))})
        return np.random.choice(range(self.output_dim), p=pred.flatten())

    def step(self):
        """Performs a n-steps of the environment and adds a transition to the experience queue."""
        exp = []

        for i in range(self.n_step):
            if self.render:
                self.env.render()
            action = self.act(self.obs)
            obs, rew, done, _ = self.env.step(action)
            self.episode_reward += rew
            exp.append((self.obs, action, rew, obs, done))
            self.obs = obs
            if done:
                self.obs = self.env.reset()
                self.rew_queue.put(self.episode_reward)
                self.episode_reward = 0
                break

        self.exp_queue.put(exp)


class TrainA2C:
    def __init__(self,
                 num_workers,
                 env_name,
                 gamma=0.99,
                 max_steps=40000,
                 print_freq=10,
                 entropy=1e-2,
                 extract='mlp',
                 l2_coef=1e-2,
                 learning_rate=1e-3,
                 render=False,
                 save_path=None,
                 load_path=None,
                 n_steps=128,
                 log_path='./logs'):
        self.max_steps = max_steps
        self.gamma = gamma
        self.render = render
        self.print_freq = print_freq
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.entropy = entropy
        self.extract = extract
        self.l2_coef = l2_coef
        self.n_steps = n_steps
        env = gym.make(env_name)
        self.env_name = env_name
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.log_dir = log_path
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
            learning_rate=self.learning_rate,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            entropy=self.entropy,
            extract=self.extract,
            l2_coef=self.l2_coef,
            gamma=self.gamma,
            n_steps=self.n_steps) for i in range(self.num_workers)]
        self._create_config()
        self.save_path = save_path
        if load_path is not None:
            self.ac.saver.restore(self.sess, load_path)
            print(f'Successfully loaded model from {load_path}')

    def _create_config(self):
        """Creates distributed tensorflow configuration"""
        ports = find_open_ports(self.num_workers + 1)
        jobs = {'global': [f'127.0.0.1:{ports[0]}'],
                'worker': [f'127.0.0.1:{ports[i + 1]}' for i in range(self.num_workers)]}
        os.environ['TF_CONFIG'] = json.dumps(jobs)

        cluster = tf.train.ClusterSpec(jobs)
        self.server = tf.distribute.Server(cluster, job_name='global', task_index=0)
        self.sess = tf.Session(target=self.server.target)

        with tf.device("/job:global/task:0"):
            with tf.variable_scope('a2c'):
                self.ac = A2CPolicyBuilder(self.input_dim,
                                           self.output_dim,
                                           learning_rate=self.learning_rate,
                                           l2_coef=self.l2_coef,
                                           entropy=self.entropy,
                                           gamma=self.gamma,
                                           extract=self.extract)
                summaries = self.ac.add_summaries()
        self.merged = tf.summary.merge(summaries)
        self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def calculate_n_step_returns(self, batch):
        obs = action = reward = next_obs = done = None
        for seq in batch:
            s, a, r, ns, d = list(map(np.array, zip(*seq)))
            val = self.sess.run(self.ac.value_pred, feed_dict={self.ac.in_ph: s[-1].reshape(1, -1)})
            r[-1] += val if not d[-1] else 0

            for i in range(d.shape[0] - 2, -1, -1):
                r[i] = r[i] + self.gamma * r[i + 1]

            obs = s if obs is None else np.concatenate((obs, s))
            next_obs = ns if next_obs is None else np.concatenate((next_obs, ns))
            action = a if action is None else np.concatenate((action, a))
            reward = r if reward is None else np.concatenate((reward, r))
            done = d if done is None else np.concatenate((done, r))
        return obs, action, reward, next_obs, done

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

        return samples

    def update(self, states, action, reward, next_states, done):
        """Updates the actor and critic networks.

        Args:
            states: array of states visited
            action: array of integer ids of selected actions
            reward: array of rewards received
            next_states: array of resulting states after taking action
            done: boolean array denoting if the corresponding state was terminal
        """

        summary = self.ac.update(self.sess, self.merged, states, action, reward, next_states, done)
        self.train_writer.add_summary(summary)

    def learn(self):
        """Trains an agent via advantage actor-critic"""
        self.sess.run(tf.global_variables_initializer())
        ep = 0
        num_steps = int(self.max_steps / (self.num_workers * self.n_steps))
        for t in range(num_steps):
            # Sample from each environment
            self.send_message(MSG_STEP)

            # Aggregate transitions
            batch = self.get_transitions()

            # Modify n step returns
            batch = self.calculate_n_step_returns(batch)

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
                print(f"Total updates: {(t + 1) * self.num_workers}")
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
        self.ac.saver.save(self.sess, self.save_path)

    def load(self):
        """Loads the actor and critic networks."""
        self.ac.saver.restore(self.sess, self.save_path)


def main():
    # The default method, "fork", copies over the tensorflow module from the parent process
    #   which is problematic w.r.t GPU resources
    mp.set_start_method('spawn')
    t = TrainA2C(num_workers=FLAGS.num_workers,
                 env_name=FLAGS.env_name,
                 gamma=FLAGS.gamma,
                 max_steps=FLAGS.max_steps,
                 print_freq=FLAGS.print_freq,
                 render=FLAGS.render,
                 save_path=FLAGS.save_path,
                 load_path=FLAGS.load_path,
                 log_path=FLAGS.log_path,
                 learning_rate=FLAGS.learning_rate,
                 n_steps=FLAGS.n_steps,
                 l2_coef=FLAGS.l2_coef,
                 entropy=FLAGS.entropy)
    t.start()


if __name__ == '__main__':
    main()
