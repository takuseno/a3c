import numpy as np
import tensorflow as tf


def build_train(model,
                num_actions,
                optimizer,
                global_step,
                lstm_unit=256,
                state_shape=[84, 84, 1],
                grad_clip=40.0,
                value_factor=0.5,
                policy_factor=1.0,
                entropy_factor=0.01,
                scope='a3c',
                shared_device='/cpu:0',
                worker_device='/cpu:0',
                reuse=None):
    with tf.device(worker_device):
        with tf.variable_scope(scope, reuse=reuse) as var_scope:
            # placeholers
            obs_input = tf.placeholder(
                tf.float32, [None] + state_shape, name='obs')
            rnn_state_ph0 = tf.placeholder(
                tf.float32, [1, lstm_unit], name='rnn_state_0')
            rnn_state_ph1 = tf.placeholder(
                tf.float32, [1, lstm_unit], name='rnn_state_1')
            actions_ph = tf.placeholder(tf.uint8, [None], name='action')
            target_values_ph = tf.placeholder(tf.float32, [None], name='value')
            advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')

            # increment global step
            inc_step = global_step.assign_add(tf.shape(obs_input)[0])

            # rnn state in tuple
            rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
                rnn_state_ph0, rnn_state_ph1)

            # network outpus
            policy,\
            value,\
            state_out = model(obs_input, rnn_state_tuple, num_actions,
                              lstm_unit, scope='model')

            # local network weights
            local_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            # global network weights
            global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

            # synchronize parameters with global network
            update_local_expr = []
            for local_var, global_var in zip(local_vars, global_vars):
                update_local_expr.append(local_var.assign(global_var))
            update_local_expr = tf.group(*update_local_expr)

            actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
            log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
            log_prob = tf.reduce_sum(log_policy * actions_one_hot, [1])

            # loss
            advantages  = tf.reshape(advantages_ph, [-1, 1])
            target_values = tf.reshape(target_values_ph, [-1, 1])
            with tf.variable_scope('value_loss'):
                value_loss = tf.reduce_sum((target_values - value) ** 2)
            with tf.variable_scope('entropy_penalty'):
                entropy = -tf.reduce_sum(policy * log_policy)
            with tf.variable_scope('policy_loss'):
                policy_loss = tf.reduce_sum(log_prob * advantages)
            loss = value_factor * value_loss\
                - policy_factor * policy_loss - entropy_factor * entropy

            # gradients
            gradients = tf.gradients(loss, local_vars)
            gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)

    # share momentum with other workers
    with tf.device(shared_device):
        with tf.variable_scope('global/'):
            optimize_expr = optimizer.apply_gradients(
                zip(gradients, global_vars))

    def update_local():
        sess = tf.get_default_session()
        sess.run(update_local_expr)

    def train(obs, rnn_state0, rnn_state1, actions, target_values, advantages):
        feed_dict = {
            obs_input: obs,
            rnn_state_ph0: rnn_state0,
            rnn_state_ph1: rnn_state1,
            actions_ph: actions,
            target_values_ph: target_values,
            advantages_ph: advantages
        }
        sess = tf.get_default_session()
        return sess.run([loss, optimize_expr, inc_step], feed_dict=feed_dict)[0]

    def act(obs, rnn_state0, rnn_state1):
        feed_dict = {
            obs_input: obs,
            rnn_state_ph0: rnn_state0,
            rnn_state_ph1: rnn_state1
        }
        sess = tf.get_default_session()
        return sess.run([policy, value, state_out], feed_dict=feed_dict)

    return act, train, update_local
