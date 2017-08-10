import numpy as np
import tensorflow as tf
import lightsaber.tensorflow.util as util


def initial_state():
    return np.zeros((2, 1, 256), np.float32)

def build_act(observations_ph, rnn_state_ph, model, num_actions, scope='a3c', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(rnn_state_ph[0], rnn_state_ph[1])
        policy, value, state_out = model(observations_ph, rnn_state_tuple, num_actions, scope='model')
        act = util.function(inputs=[observations_ph, rnn_state_ph], outputs=[policy, state_out], updates=[], givens={rnn_state_ph: initial_state()})
        return act

def build_train(model, num_actions, optimizer, scope='a3c', reuse=None):
    obs_input = tf.placeholder(tf.float32, [None, 4, 84, 84], name='obs')
    rnn_state_ph = tf.placeholder(tf.float32, [2, 1, 256])

    act_f = build_act(obs_input, rnn_state_ph, model, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        actions_ph = tf.placeholder(tf.int32, [None], name='action')
        target_values_ph = tf.placeholder(tf.float32, [None], name='value')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(rnn_state_ph[0], rnn_state_ph[1])

        policy, value, state_out = model(obs_input, rnn_state_tuple, num_actions, scope='model', reuse=True)

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        responsible_outputs = tf.reduce_sum(policy * actions_one_hot, [1])

        value_loss = 0.5 * tf.reduce_sum(tf.square(target_values_ph - tf.reshape(value, [-1])))
        entropy = -tf.reduce_sum(policy * tf.log(policy))
        policy_loss = -tf.reduce_sum(tf.log(responsible_outputs) * advantages_ph)
        loss = 0.5 * value_loss + policy_loss - entropy * 0.005

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients = tf.gradients(loss, local_vars)
        var_norms = tf.global_norm(local_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        optimize_expr = optimizer.apply_gradients(zip(grads, global_vars))

        update_local_expr = []
        for local_var, global_var in zip(sorted(local_vars, key=lambda v: v.name),
                                    sorted(global_vars, key=lambda v: v.name)):
            update_local_expr.append(local_var.assign(global_var))
        update_local_expr = tf.group(*update_local_expr)
        update_local = util.function([], [], updates=[update_local_expr])

        train = util.function(
            inputs=[
                obs_input, rnn_state_ph, actions_ph, target_values_ph, advantages_ph
            ],
            outputs=loss,
            updates=[optimize_expr],
            givens={rnn_state_ph: initial_state()}
        )

        action_dist = util.function([obs_input, rnn_state_ph],
                policy, givens={rnn_state_ph: initial_state()})
        state_value = util.function([obs_input, rnn_state_ph],
                value, givens={rnn_state_ph: initial_state()})

    return act_f, train, update_local, action_dist, state_value
