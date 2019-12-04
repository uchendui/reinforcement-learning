import numpy as np
import tensorflow as tf

from util.misc import to_one_hot


def conv2d(in_ph, reuse=False):
    network = tf.contrib.layers.conv2d(in_ph, 32, kernel_size=8, activation_fn=tf.nn.relu, stride=4, scope='conv1',
                                       reuse=reuse)
    network = tf.contrib.layers.conv2d(network, 64, kernel_size=4, activation_fn=tf.nn.relu, stride=2, scope='conv2',
                                       reuse=reuse)
    network = tf.contrib.layers.conv2d(network, 64, kernel_size=3, activation_fn=tf.nn.relu, stride=1, scope='conv3',
                                       reuse=reuse)
    network = tf.contrib.layers.flatten(network)
    network = tf.contrib.layers.fully_connected(network, 512, activation_fn=tf.nn.relu, scope='conv_lin', reuse=reuse)
    return network


def forward(in_ph, layers=(512,), activations=(tf.nn.relu,), reuse=False, scope=''):
    assert len(layers) == len(activations)
    network = in_ph
    for i in range(len(layers)):
        network = tf.contrib.layers.fully_connected(network, layers[i], activation_fn=activations[i],
                                                    scope=f'{scope}fc_{i}',
                                                    reuse=reuse)
    return network


def cross_entropy_loss(logits, actions_ph, baseline_ph, entropy_strength=0.0):
    # Create loss function for policy gradient
    assert logits.get_shape().as_list() == actions_ph.get_shape().as_list()
    entropy = -tf.reduce_sum(logits * tf.log(logits + 1e-6))

    # Choose which actions to adjust probability for
    negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions_ph, logits=logits)
    weighted_negative_likelihoods = tf.multiply(negative_likelihoods, baseline_ph)
    loss = tf.reduce_mean(weighted_negative_likelihoods) - entropy_strength * entropy
    return loss, entropy


def l2_weight_loss(scope='', l2_coef=0.01, extras=tuple()):
    variables = tf.trainable_variables(scope=scope)
    prefixes = list(extras) + ['weight']
    weights = [v for v in variables if any(x in v.name for x in prefixes)]
    reg = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights])
    return l2_coef * reg


class PolicyBuilder:
    def _make_input_placeholders(self):
        raise NotImplementedError()

    def _make_target_placeholders(self):
        raise NotImplementedError()

    def update(self, sess, merged, s, a, r, ns, d):
        raise NotImplementedError()

    def add_summaries(self):
        raise NotImplementedError()


class A2CPolicyBuilder(PolicyBuilder):

    def __init__(self, in_dim, out_dim, learning_rate=1e-3, gamma=0.99, entropy=1e-2, l2_coef=1e-3, extract='mlp'):
        scope = tf.get_variable_scope()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gamma = gamma
        self.entropy = entropy
        self.extract = extract
        self.in_ph, self.baseline, self.action_ph = self._make_input_placeholders()
        self.value_ph = self._make_target_placeholders()

        # Forward pass
        features = self.in_ph
        if extract == 'cnn':
            features = conv2d(features)

        with tf.variable_scope('critic'):
            self.value_pred = forward(features, layers=(32, 32, 1), activations=(tf.nn.relu, tf.nn.relu, None))
        with tf.variable_scope('actor'):
            self.out_pred = forward(features, layers=(32, 32, self.out_dim),
                                    activations=(tf.nn.relu, tf.nn.relu, tf.nn.softmax))

        self.critic_loss = tf.losses.mean_squared_error(self.value_pred, self.value_ph)
        self.actor_loss, self.entropy_value = cross_entropy_loss(
            logits=self.out_pred,
            actions_ph=self.action_ph,
            baseline_ph=self.baseline,
            entropy_strength=self.entropy)
        self.l2_loss = l2_weight_loss(l2_coef=l2_coef, scope=scope.name)
        self.loss = self.critic_loss + self.actor_loss + self.l2_loss

        # Optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Save
        self.variables = scope.trainable_variables()
        self.saver = tf.train.Saver(self.variables)

    def _make_input_placeholders(self):
        in_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.in_dim], name='state_input')
        baseline_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='baseline_input')
        actions_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim], name='action_input')
        return in_ph, baseline_ph, actions_ph

    def _make_target_placeholders(self):
        value_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='critic_input')
        return value_ph

    def update(self, sess, merged, s, a, r, ns, d):
        summary, _ = sess.run([merged, self.opt], feed_dict=self.get_feed_dict(sess, s, a, r))
        return summary

    def get_feed_dict(self, sess, s, a, r):
        feed = self.get_input_feed_dict(s, a, )
        feed.update(self.get_target_feed_dict(sess, s, r))
        return feed

    def get_input_feed_dict(self, s, a):
        return {self.in_ph: s,
                self.action_ph: to_one_hot(a, self.out_dim),
                }

    def get_target_feed_dict(self, sess, s, r, ):
        r = r.reshape((-1, 1))
        state_values = sess.run(self.value_pred, feed_dict={self.in_ph: s})
        advantage = r - state_values
        return {self.value_ph: r, self.baseline: advantage}

    def add_summaries(self):
        a = tf.summary.scalar('Loss', self.loss, )
        b = tf.summary.scalar('Mean Estimated Value', tf.reduce_mean(self.value_pred))
        c = tf.summary.scalar('Critic Loss', self.critic_loss, )
        d = tf.summary.scalar('Policy Gradient Loss', self.actor_loss, )
        e = tf.summary.scalar('Entropy', self.entropy_value, )
        return a, b, c, d


class DQNPolicyBuilder(PolicyBuilder):
    def __init__(self, in_dim, out_dim, learning_rate=1e-3, gamma=0.99, extract='mlp'):
        scope = tf.get_variable_scope()
        self.target_network = None
        self.update_target_network_op = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gamma = gamma
        self.in_ph, self.is_weights = self._make_input_placeholders()
        self.action_indices_ph, self.q_target = self._make_target_placeholders()

        # Forward pass
        features = self.in_ph
        if extract == 'cnn':
            features = conv2d(features)
        self.out_pred = forward(features, layers=(512, self.out_dim), activations=(tf.nn.relu, None))

        # Create loss function
        batch_range = tf.range(start=0, limit=tf.shape(self.action_indices_ph)[0])
        indices = tf.stack((batch_range, self.action_indices_ph), axis=1)
        self.action_values = tf.gather_nd(self.out_pred, indices)
        self.loss = tf.losses.mean_squared_error(self.q_target, self.action_values, weights=self.is_weights)

        # Optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Save
        self.variables = scope.trainable_variables()
        self.saver = tf.train.Saver(self.variables)

    def update(self, sess, merged, s, a, r, ns, d, is_weights=None):
        summary, loss = sess.run([merged, self.opt], feed_dict=self.get_feed_dict(sess, s, a, r, ns, d, is_weights))
        return summary

    def add_summaries(self):
        a = tf.summary.scalar('Loss', self.loss, )
        b = tf.summary.scalar('Mean Estimated Value', tf.reduce_mean(self.out_pred))
        return a, b

    def get_feed_dict(self, sess, s, a, r, ns, d, is_weights=None):
        feed = self.get_input_feed_dict(s, a, is_weights=is_weights)
        feed.update(self.get_target_feed_dict(sess, a, ns, r, d))
        return feed

    def get_input_feed_dict(self, s, a, is_weights=None):
        return {self.in_ph: s, self.action_indices_ph: a,
                self.is_weights: np.ones(s.shape[0]) if is_weights is None else is_weights}

    def get_target_feed_dict(self, sess, a, ns, r, d):
        next_state_pred = self.gamma * sess.run(self.target_network.out_pred,
                                                feed_dict=self.target_network.get_input_feed_dict(ns, a))

        # Adjust the targets for non-terminal states
        r = r.reshape(-1, 1)
        targets = r
        loc = np.argwhere(d != True).flatten()
        if len(loc) > 0:
            max_q = np.amax(next_state_pred, axis=1)
            targets[loc] = np.add(
                targets[loc],
                max_q[loc].reshape(max_q[loc].shape[0], 1),
                casting='unsafe')

        return {self.q_target: targets.flatten()}

    def set_target_network(self, target):
        self.target_network = target
        self.update_target_network_op = [old.assign(new) for (new, old) in
                                         zip(self.variables, target.variables)]

    def _make_input_placeholders(self):
        in_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.in_dim], name='state_input')
        weights = tf.placeholder(dtype=tf.float32, shape=[None], name='importance_sampling_weights')
        return in_ph, weights

    def _make_target_placeholders(self):
        target_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='q_value_target')
        action_indices_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='action_indices')
        return action_indices_ph, target_ph


class DDPGPolicyBuilder(PolicyBuilder):
    def __init__(self, in_dim, out_dim, critic_lr=1e-3, actor_lr=1e-4, gamma=0.99, extract='mlp', tau=1e-3):
        scope = tf.get_variable_scope()
        self.tau = tau
        self.gamma = gamma
        self.target = None
        self.update_target_network_op = None
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Build actor network
        self.in_ph = self._make_input_placeholders()
        self.q_target = self._make_target_placeholders()

        # Forward pass
        if extract == 'mlp':
            with tf.variable_scope('actor'):
                self.actor_pred = forward(self.in_ph, layers=(512, self.out_dim), activations=(tf.nn.relu, None))
            q_input = tf.concat([self.in_ph, self.actor_pred], axis=1)
            with tf.variable_scope('critic'):
                self.q_pred = forward(q_input, layers=(512, 1), activations=(tf.nn.relu, None))
        elif extract == 'cnn':
            self.actor_pred = conv2d(self.in_ph)
        else:
            raise ValueError('Unknown feature extractor type')

        # Create loss functions
        self.actor_loss = tf.reduce_mean(-self.q_pred)
        self.critic_loss = tf.losses.mean_squared_error(self.q_target, self.q_pred) + l2_weight_loss(
            scope=f'{scope.name}/critic')

        # Optimizers
        self.critic_opt = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.critic_loss)
        self.actor_variables = tf.trainable_variables(
            scope=f'{scope.name}/actor')
        self.actor_opt = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(self.actor_loss,
                                                                                 var_list=self.actor_variables)

        # Save
        self.variables = scope.trainable_variables()
        self.saver = tf.train.Saver(self.variables)

    def update(self, sess, merged, s, a, r, ns, d):
        summary, _, _ = sess.run([merged, self.critic_opt, self.actor_opt],
                                 feed_dict=self.get_feed_dict(sess, s, a, r, ns, d))
        return summary

    def get_feed_dict(self, sess, s, a, r, ns, d):
        feed = self.get_input_feed_dict(s)
        feed.update(self.get_target_feed_dict(sess, a, ns, r, d))
        return feed

    def get_input_feed_dict(self, s):
        return {self.in_ph: s}

    def get_target_feed_dict(self, sess, a, ns, r, d):
        next_state_pred = self.gamma * sess.run(self.target.q_pred,
                                                feed_dict=self.target._get_input_feed_dict(ns))

        # Adjust the targets for non-terminal states
        r = r.reshape(-1, 1)
        targets = r
        loc = np.argwhere(d != True).flatten()
        if len(loc) > 0:
            max_q = np.amax(next_state_pred, axis=1)
            targets[loc] = np.add(
                targets[loc],
                max_q[loc].reshape(max_q[loc].shape[0], 1),
                casting='unsafe')

        return {self.q_target: targets}

    def add_summaries(self):
        a = tf.summary.scalar('Actor Loss', self.actor_loss, )
        b = tf.summary.scalar('Critic Loss', self.critic_loss, )
        c = tf.summary.scalar('Mean Estimated Value', tf.reduce_mean(self.q_pred))
        return a, b, c

    def set_target(self, target):
        self.target = target
        self.update_target_network_op = [old.assign(self.tau * old + (1 - self.tau) * new) for (new, old) in
                                         zip(self.variables, target.variables)]

    def _make_input_placeholders(self):
        in_ph = tf.placeholder(tf.float32, [None, self.in_dim], 'obs_input')
        return in_ph

    def _make_target_placeholders(self):
        target_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='q_value_target')
        return target_ph


def main():
    dqn = DQNPolicyBuilder(1, 1)


if __name__ == '__main__':
    main()
