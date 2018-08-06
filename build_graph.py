import numpy as np
import tensorflow as tf


def build_train(model,
                num_actions,
                optimizer,
                lstm_unit=256,
                state_shape=[84, 84, 1],
                grad_clip=40.0,
                value_factor=0.5,
                policy_factor=1.0,
                entropy_factor=0.01,
                scope='a3c',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # placeholers
        obs_input = tf.placeholder(tf.float32, [None] + state_shape, name='obs')
        rnn_state_ph0 = tf.placeholder(
            tf.float32, [1, lstm_unit], name='rnn_state_0')
        rnn_state_ph1 = tf.placeholder(
            tf.float32, [1, lstm_unit], name='rnn_state_1')
        actions_ph = tf.placeholder(tf.uint8, [None], name='action')
        returns_ph = tf.placeholder(tf.float32, [None], name='return')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')

        # rnn state in tuple
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            rnn_state_ph0, rnn_state_ph1)

        # network outpus
        policy, value, state_out = model(
            obs_input, rnn_state_tuple, num_actions, lstm_unit, scope='model')

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        log_prob = tf.reduce_sum(log_policy * actions_one_hot, axis=1, keep_dims=True)

        # loss
        advantages  = tf.reshape(advantages_ph, [-1, 1])
        returns = tf.reshape(returns_ph, [-1, 1])
        with tf.variable_scope('value_loss'):
            value_loss = tf.reduce_sum((returns - value) ** 2)
        with tf.variable_scope('entropy_penalty'):
            entropy = -tf.reduce_sum(policy * log_policy)
        with tf.variable_scope('policy_loss'):
            policy_loss = tf.reduce_sum(log_prob * advantages)
        loss = value_factor * value_loss\
            - policy_factor * policy_loss - entropy_factor * entropy

        # local network weights
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        # global network weights
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

        # gradients
        gradients = tf.gradients(loss, local_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
        grads_and_vars = zip(clipped_gradients, global_vars)
        optimize_expr = optimizer.apply_gradients(grads_and_vars)

        update_local_expr = []
        for local_var, global_var in zip(local_vars, global_vars):
            update_local_expr.append(local_var.assign(global_var))
        update_local_expr = tf.group(*update_local_expr)

        def update_local():
            sess = tf.get_default_session()
            sess.run(update_local_expr)

        def train(obs, rnn_state0, rnn_state1, actions, returns, advantages):
            feed_dict = {
                obs_input: obs,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1,
                actions_ph: actions,
                returns_ph: returns,
                advantages_ph: advantages
            }
            sess = tf.get_default_session()
            return sess.run([loss, optimize_expr], feed_dict=feed_dict)[0]

        def act(obs, rnn_state0, rnn_state1):
            feed_dict = {
                obs_input: obs,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1
            }
            sess = tf.get_default_session()
            return sess.run([policy, value, state_out], feed_dict=feed_dict)

    return act, train, update_local
