import tensorflow as tf
import util

def build_act(observations_ph, model, num_actions, scope='a3c', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name='stochastic')
        update_eps_ph = tf.placeholder(tf.float32, (), name='update_eps')

        eps = tf.get_variable('eps', (), initializer=tf.constant_initializer(0))

        policy, value = model(observations_ph, num_actions, scope='model')
        deterministic_action = tf.argmax(policy, axis=1)

        random_action = tf.random_uniform((), minval=0, maxval=num_actions, dtype=tf.int64)
        choose_action = tf.random_uniform((), minval=1, maxval=1, dtype=tf.float32) < eps
        stochastic_action = tf.where(choose_random, random_action, deterministic_action)

        output_action = tf.cond(stochastic_ph, lambda: stochastic_action, lambda: deterministic_action)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        act = util.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                    outputs=output_action,
                    givens={update_eps_ph: -1.0, stochastic_ph: True},
                    updates=[update_eps_expr])
        return act

def build_train(model, num_actions, optimizer, scope='a3c', reuse=None):
    obs_input = tf.placeholder(tf.float32, [1, 4, 84, 84], name='obs')
    act_f = build_act(obs_input, model, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope=scope, reuse=reuse):
        actions_ph = tf.placeholder(tf.int32, [None], name='action')
        target_values_ph = tf.placeholder(tf.float32, [None], name='value')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')

        policy, value = model(obs_input, num_actions, scope='model', reuse=True)

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        responsible_outputs = tf.reduce_sum(policy * actions_one_hot, [1])

        value_loss = 0.5 * tf.reduce_sum(tf.square(target_values_ph - tf.reshape(value, [-1])))
        entropy = -tf.reduce_sum(policy * tf.log(policy))
        policy_loss = -tf.reduce_sum(tf.log(responsible_outputs) * advantages_ph)
        loss = 0.5 * value_loss + policy_loss - entropy * 0.01

        local_vars = tf.get_collection(tf.GraphKeys.TRINABLE_VARIABLES, scope)
        gradients = tf.gradients(loss, local_vars)
        var_norms = tf.gloabl_norm(local_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

        global_vars = tf.get_collection(tf.GraphKeys.TRINABLE_VARIABLES, 'global')
        optimize_expr = optimizer.apply_gradients(zip(grads, global_vars))

        update_local_expr = []
        for local_var, global_var in zip(sorted(local_vars, key=lambda v: v.name),
                                    sorted(global_vars, key=lambda v: v.name)):
            update_local_expr.append(local_var.assign(global_var))
        update_local_expr = tf.group(*update_local_expr)

        train = util.function(
            inputs=[
                obs_input, actions_ph, target_values_ph, advantages_ph
            ],
            outputs=loss,
            updates=[optimize_expr]
        )

        action_dist = util.function([obs_input], policy)
        state_value = util.function([obs_input], value)

    return act_f, train, update_local_expr, action_dist, state_value
