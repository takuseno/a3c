import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partiion_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def _make_network(convs, inpt, num_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('convnet'):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                        num_outputs=num_outputs,
                        kernel_size=kernel_size,
                        stride=stride,
                        activation_fn=tf.nn.elu)
        conv_out = layers.flatten(out)
        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            state_init = (c_init, h_init)

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(conv_out, [0])
            step_size = tf.shape(inpt)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=state_in,
                    sequence_length=step_size, time_major=False)
            lstm_c, lstm_h = lstm_state
            state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        policy = layers.fully_connected(rnn_out,
                num_actions, activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01))

        value = layers.fully_connected(rnn_out, 1, activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0))

    return policy, value

def make_network(convs):
    return lambda *args, **kwargs: _make_network(convs, *args, **kwargs)
