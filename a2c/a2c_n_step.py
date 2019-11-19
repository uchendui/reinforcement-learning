import os
import gym
import sys
import time
import json
import numpy as np
import multiprocessing as mp
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.path.append('../')

from util.ports import find_open_ports
from util.network import ActorCriticNetworkBuilder
from util.misc import to_one_hot
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_episodes', 10, 'Maximum number of episodes to run')
flags.DEFINE_integer('threads', 8, 'number of processes for training')
flags.DEFINE_integer('max_steps', 10000, 'number of processes for training')
flags.DEFINE_integer('n_step', 10, 'number of steps to boostrap for reward')
flags.DEFINE_integer('print_freq', 10, 'number of episodes between printing stats')
flags.DEFINE_bool('render', False, 'render the environment')
flags.DEFINE_string('env_name', 'CartPole-v0', 'Environment name')

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
                 n_step,
                 render):
        super(Worker, self).__init__()
        self.id = worker_id
        self.env = gym.make(env_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.render = render
        self.conn = conn
        self.n_step = n_step
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
            self.ac = ActorCriticNetworkBuilder(self.input_dim,
                                                self.output_dim,
                                                actor_layers=(32, 32),
                                                critic_layers=(32, 32),
                                                actor_activations=(tf.nn.relu, tf.nn.relu, tf.nn.softmax),
                                                critic_activations=(tf.nn.relu, tf.nn.relu, None))

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
                 n_step=1,
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
            n_step=n_step,
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
            self.ac = ActorCriticNetworkBuilder(self.input_dim,
                                                self.output_dim,
                                                actor_layers=(32, 32),
                                                critic_layers=(32, 32),
                                                actor_activations=(tf.nn.relu, tf.nn.relu, tf.nn.softmax),
                                                critic_activations=(tf.nn.relu, tf.nn.relu, None))

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

    def calculate_n_step_returns(self, batch):
        obs = action = reward = next_obs = done = None
        for seq in batch:
            s, a, r, ns, d = list(map(np.array, zip(*seq)))
            val = self.sess.run(self.ac.critic.output_pred, feed_dict={self.ac.critic.input_ph: s[-1].reshape(1, -1)})
            r[-1] += val if not d[-1] else 0

            for i in range(d.shape[0] - 2, -1, -1):
                r[i] = r[i] + self.gamma * r[i + 1]

            obs = s if obs is None else np.concatenate((obs, s))
            next_obs = ns if next_obs is None else np.concatenate((next_obs, ns))
            action = a if action is None else np.concatenate((action, a))
            reward = r if reward is None else np.concatenate((reward, r))
            done = d if done is None else np.concatenate((done, r))
        return obs, action, reward, next_obs, done

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
        state_values = self.sess.run(self.ac.critic.output_pred, feed_dict={self.ac.critic.input_ph: states})
        _, value_loss = self.sess.run([self.ac.critic.opt, self.ac.critic.loss],
                                      feed_dict={self.ac.critic.input_ph: states,
                                                 self.ac.critic.target_ph: reward})

        #######################################################
        # Update actor network
        #######################################################
        advantage = reward - state_values
        _, policy_loss = self.sess.run([self.ac.actor.opt, self.ac.actor.loss],
                                       feed_dict={self.ac.actor.input_ph: states,
                                                  self.ac.actor.baseline_ph: advantage.flatten(),
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
    t = TrainA2C(FLAGS.threads,
                 FLAGS.env_name,
                 n_step=FLAGS.n_step,
                 max_steps=FLAGS.max_steps,
                 print_freq=FLAGS.print_freq,
                 render=FLAGS.render)
    t.start()


if __name__ == '__main__':
    main()
