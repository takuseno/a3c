import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def _make_network(convs, inpt, rnn_state_tuple, num_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('convnet'):
            for num_outputs, kernel_size, stride, padding in convs:
                out = layers.convolution2d(out,
                        num_outputs=num_outputs,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding='VALID',
                        activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            rnn_in = tf.expand_dims(conv_out, [0])
            step_size = tf.shape(inpt)[0]
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=rnn_state_tuple,
                    sequence_length=[step_size], time_major=False)
            lstm_c, lstm_h = lstm_state
            state_out = tf.reshape(tf.concat([lstm_c, lstm_h], 0), [2, 1, 256])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        policy = layers.fully_connected(rnn_out,
                num_actions, activation_fn=tf.nn.softmax)

        value = layers.fully_connected(rnn_out, 1, activation_fn=None)

    return policy, value, state_out

def make_network(convs):
    return lambda *args, **kwargs: _make_network(convs, *args, **kwargs)
