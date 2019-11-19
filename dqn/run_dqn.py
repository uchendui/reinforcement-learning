import gym
import time
import numpy as np
import tensorflow as tf

from dqn import TrainDQN
from util.replay_buffer import PrioritizedReplayBuffer
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_episodes', 10, 'Maximum number of episodes to run')
flags.DEFINE_string('buffer_save_path', './buffers', 'Save location for the replay buffer')


def main():
    with tf.Session() as sess:
        env = gym.make(FLAGS.env_name)

        dqn = TrainDQN(env,
                       sess,
                       max_steps=0,
                       load_path=FLAGS.load_path)
        sess.run(tf.initialize_all_variables())
        dqn.load()
        num_episodes = FLAGS.max_episodes

        buffer = PrioritizedReplayBuffer(capacity=env.spec.max_episode_steps * FLAGS.max_episodes)
        reward = 0
        obs = env.reset()
        while num_episodes > 0:
            if FLAGS.render:
                env.render()
                time.sleep(0.01)

            action = dqn.act(obs)
            new_obs, rew, done, _ = env.step(action)

            # Store transition for saving
            buffer.add((obs, action, reward, new_obs, done))

            reward += rew
            obs = new_obs
            if done:
                num_episodes -= 1
                obs = env.reset()
                print('Episode Reward:', reward)
                reward = 0

        loc = f'{FLAGS.buffer_save_path}/{FLAGS.env_name}/{FLAGS.env_name}.npy'
        buffer.save(loc)


if __name__ == '__main__':
    main()
