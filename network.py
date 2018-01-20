import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def _make_network(convs, inpt, rnn_state_tuple, num_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('convnet'):
            for num_outputs, kernel_size, stride, padding in convs:
                out = layers.convolution2d(
                    out,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='VALID',
                    activation_fn=tf.nn.relu
                )

        conv_out = layers.fully_connected(
            layers.flatten(out),
            256,
            activation_fn=tf.nn.relu
        )

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            rnn_in = tf.expand_dims(conv_out, [0])
            step_size = tf.shape(inpt)[:1]
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_in,
                initial_state=rnn_state_tuple,
                sequence_length=step_size,
                time_major=False
            )
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        policy = layers.fully_connected(
            rnn_out,
            num_actions, activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(),
            biases_initializer=None
        )

        value = layers.fully_connected(
            rnn_out,
            1,
            activation_fn=None,
            weights_initializer=normalized_columns_initializer(),
            biases_initializer=None
        )

    return policy, value, lstm_state

def make_network(convs):
    return lambda *args, **kwargs: _make_network(convs, *args, **kwargs)
