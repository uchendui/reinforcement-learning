import tensorflow as tf


class NetworkBuilder:
    def __init__(self, input_dim, output_dim, layers=(512,), activations=(tf.nn.relu, None),
                 conv=False):
        """Creates a neural network. Derived classes handle specific losses and opts.
        Args:
            input_dim: Input dimensions
            output_dim: Output dimensions
            layers: list of hidden unit numbers for each layer
            activations: activations for each layer
            conv: True for convolutional neural network, False for multi-layer perceptron
        """
        assert len(layers) > 0, 'There must be at least one hidden layer'

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layers
        self.activations = activations
        self.conv = conv

    def create_network(self):
        """Creates a neural network.
        Returns:
            input_ph: Placeholder for the inputs.
            output_pred: Tensor for the output layer.
        """
        if self.conv:
            input_ph = tf.placeholder(dtype=tf.float32, shape=[None, *self.input_dim])
            network = tf.image.rgb_to_grayscale(input_ph)
            network = tf.contrib.layers.conv2d(network, 30, kernel_size=4, activation_fn=tf.nn.relu, stride=4)
            network = tf.contrib.layers.conv2d(network, 30, kernel_size=2, activation_fn=tf.nn.relu, stride=2)
            network = tf.contrib.layers.conv2d(network, 30, kernel_size=1, activation_fn=tf.nn.relu, stride=1)
            network = tf.contrib.layers.flatten(network)
            network = tf.contrib.layers.fully_connected(network, 512, activation_fn=tf.nn.relu)
            output_pred = tf.contrib.layers.fully_connected(network, self.output_dim, activation_fn=None)

        else:
            weights = [tf.get_variable(name='W0',
                                       shape=[self.input_dim, self.layer_sizes[0]],
                                       initializer=tf.contrib.layers.xavier_initializer())]
            biases = [tf.get_variable(name=f'b0',
                                      shape=[self.layer_sizes[0]],
                                      initializer=tf.constant_initializer(0.))]
            for i in range(1, len(self.layer_sizes)):
                in_dim = weights[-1].shape[1].value
                out_dim = self.layer_sizes[i]
                weights.append(tf.get_variable(name=f'W{i}',
                                               shape=[in_dim, out_dim],
                                               initializer=tf.contrib.layers.xavier_initializer()))
                biases.append(tf.get_variable(name=f'b{i}',
                                              shape=[out_dim],
                                              initializer=tf.constant_initializer(0.)))

            weights.append(tf.get_variable(name=f'W{len(self.layer_sizes)}',
                                           shape=[self.layer_sizes[-1], self.output_dim],
                                           initializer=tf.contrib.layers.xavier_initializer()))
            biases.append(tf.get_variable(name=f'b{len(self.layer_sizes)}',
                                          shape=[self.output_dim],
                                          initializer=tf.constant_initializer(0.)))

            # Create computation graph for network
            input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
            layer = input_ph
            for W, b, activation in zip(weights, biases, self.activations):
                layer = tf.nn.xw_plus_b(x=layer, weights=W, biases=b)
                if activation is not None:
                    layer = activation(layer)
            output_pred = layer

        self.saver = tf.train.Saver()
        return input_ph, output_pred


class QNetworkBuilder(NetworkBuilder):
    def __init__(self, input_dim, output_dim, layers=(512,), activations=(tf.nn.relu, None), scope='q_network',
                 conv=False):
        """Creates a Q-network and defines ops."""
        super(QNetworkBuilder, self).__init__(input_dim, output_dim, layers, activations, conv)
        with tf.variable_scope(scope):
            # Create variables
            self.input_ph, self.output_pred = self.create_network()
            self.target_ph = tf.placeholder(dtype=tf.float32, shape=[None])
            self.action_indices_ph = tf.placeholder(dtype=tf.int32, shape=[None])

            # Create loss function
            batch_range = tf.range(start=0, limit=tf.shape(self.action_indices_ph)[0])
            indices = tf.stack((batch_range, self.action_indices_ph), axis=1)
            self.action_values = tf.gather_nd(self.output_pred, indices)
            self.loss = tf.losses.mean_squared_error(self.target_ph, self.action_values)

            # Optimizer
            self.adam = tf.train.AdamOptimizer()
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)


class ValueNetworkBuilder(NetworkBuilder):
    def __init__(self, input_dim, output_dim=1, layers=(512,), activations=(tf.nn.relu, None), scope='value_network',
                 conv=False):
        """Creates a network for estimating the value function. Very similar to the QNetworkBuilder"""
        super(ValueNetworkBuilder, self).__init__(input_dim, output_dim, layers, activations, conv)
        with tf.variable_scope(scope):
            # Create variables
            self.input_ph, self.output_pred = self.create_network()
            self.target_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
            self.action_indices_ph = tf.placeholder(dtype=tf.int32, shape=[None])

            # Create loss function
            self.loss = tf.losses.mean_squared_error(self.output_pred, self.target_ph)

            # Optimizer
            self.adam = tf.train.AdamOptimizer()
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)


class PolicyNetworkBuilder(NetworkBuilder):
    def __init__(self, input_dim, output_dim, layers=(512,), activations=(tf.nn.relu, None), scope='reinforce_network',
                 conv=False):
        """Creates a network for policy gradient and defines ops."""
        super(PolicyNetworkBuilder, self).__init__(input_dim, output_dim, layers, activations, conv)
        with tf.variable_scope(scope):
            # Create additional inputs
            self.input_ph, self.output_pred = self.create_network()
            self.q_values_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='Reward-to-go')
            self.actions_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim], name='actions')

            # Create loss function for policy gradient
            logits = self.output_pred
            negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=self.actions_ph, logits=logits)
            weighted_negative_likelihoods = tf.multiply(negative_likelihoods, self.q_values_ph)
            self.loss = tf.reduce_mean(weighted_negative_likelihoods)

            # Optimizer
            self.adam = tf.train.AdamOptimizer()
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)


class ActorCriticNetworkBuilder(NetworkBuilder):
    """Creates a network for A2C and A3C and defines ops"""

    def __init__(self, input_dim, output_dim, layers=(512,), activations=(tf.nn.relu, tf.nn.softmax),
                 scope='ac_network', conv=False):
        super(ActorCriticNetworkBuilder, self).__init__(input_dim, output_dim, layers, activations, conv)

        # Create two networks: one for the policy and one for the value function
        with tf.variable_scope('actor'):
            self.actor = PolicyNetworkBuilder(input_dim, output_dim, layers, activations, scope, conv)
        with tf.variable_scope('critic'):
            self.critic = ValueNetworkBuilder(input_dim=input_dim,
                                              output_dim=1,
                                              layers=layers,
                                              activations=activations,
                                              scope=scope,
                                              conv=conv)

        self.advantage_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='Advantage')

        # Loss will be the same as the policy network!
        logits = self.actor.output_pred
        negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=self.actor.actions_ph, logits=logits)
        weighted_negative_likelihoods = tf.multiply(negative_likelihoods, self.advantage_ph)
        self.loss = tf.reduce_mean(weighted_negative_likelihoods)


def main():
    qnb = QNetworkBuilder(1, 1)


if __name__ == '__main__':
    main()
