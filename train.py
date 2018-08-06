import threading
import argparse
import cv2
import gym
import copy
import os
import time
import atari_constants
import box_constants
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from rlsaber.log import TfBoardLogger, dump_constants
from rlsaber.trainer import Trainer
from rlsaber.trainer import Evaluator, Recorder
from rlsaber.env import EnvWrapper, NoopResetEnv, EpisodicLifeEnv
from rlsaber.preprocess import atari_preprocess
from actions import get_action_space
from network import make_network
from agent import Agent
from datetime import datetime

def make_agent(model,
               actions,
               optimizer,
               global_step,
               state_shape,
               phi,
               name,
               shared_device,
               worker_device,
               constants):
    return Agent(
        model,
        actions,
        optimizer,
        global_step,
        gamma=constants.GAMMA,
        lstm_unit=constants.LSTM_UNIT,
        time_horizon=constants.TIME_HORIZON,
        policy_factor=constants.POLICY_FACTOR,
        value_factor=constants.VALUE_FACTOR,
        entropy_factor=constants.ENTROPY_FACTOR,
        grad_clip=constants.GRAD_CLIP,
        state_shape=state_shape,
        phi=phi,
        name=name,
        shared_device=shared_device,
        worker_device=worker_device
    )

def train(server, cluster, args):
    is_chief = args.index == 0

    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    env_name = args.env
    tmp_env = gym.make(env_name)
    is_atari = len(tmp_env.observation_space.shape) != 1
    # box environment
    if not is_atari:
        observation_space = tmp_env.observation_space
        constants = box_constants
        actions = range(tmp_env.action_space.n)
        state_shape = [observation_space.shape[0], constants.STATE_WINDOW]
        state_preprocess = lambda s: s
        # (window_size, dim) -> (dim, window_size)
        phi = lambda s: np.transpose(s, [1, 0])
    # atari environment
    else:
        constants = atari_constants
        actions = get_action_space(env_name)
        state_shape = constants.STATE_SHAPE + [constants.STATE_WINDOW]
        def state_preprocess(state):
            # atari specific preprocessing
            state = atari_preprocess(state, constants.STATE_SHAPE)
            state = np.array(state, dtype=np.float32)
            return state / 255.0
        # (window_size, H, W) -> (H, W, window_size)
        phi = lambda s: np.transpose(s, [1, 2, 0])

    # save settings
    if is_chief:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        dump_constants(constants, os.path.join(logdir, 'constants.json'))

    model = make_network(
        constants.CONVS, constants.FCS,
        lstm=constants.LSTM, padding=constants.PADDING)

    # share Adam optimizer with all threads!
    worker_device = '/job:worker/task:{}/cpu:0'.format(args.index)
    shared_device = tf.train.replica_device_setter(
        1, worker_device=worker_device, cluster=cluster)
    with tf.device(shared_device):
        lr = tf.Variable(constants.LR)
        decayed_lr = tf.placeholder(tf.float32)
        decay_lr_op = lr.assign(decayed_lr)
        if constants.OPTIMIZER == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=0.1)
        else:
            optimizer = tf.train.AdamOptimizer(lr)
        global_step = tf.Variable(0, dtype=tf.int32, name='step')
        add_global_step_op = global_step.assign(tf.add(global_step, 1))

    # global parameters
    master = make_agent(model, actions, optimizer, global_step, state_shape,
                        phi, 'global', shared_device, shared_device, constants)
    global_vars = tf.global_variables()
    init_op = tf.variables_initializer(global_vars)

    # local parameters
    agent = make_agent(model, actions, optimizer, global_step, state_shape,
                       phi, 'worker', shared_device, worker_device, constants)
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'worker')
    local_init_op = tf.variables_initializer(local_vars)

    env = gym.make(args.env)
    env.seed(args.index)
    if is_atari:
        env = NoopResetEnv(env, noop_max=7)
        env = EpisodicLifeEnv(env)
    wrapped_env = EnvWrapper(
        env,
        r_preprocess=lambda r: np.clip(r, -1, 1),
        s_preprocess=state_preprocess
    )

    summary_writer = tf.summary.FileWriter(logdir)
    tflogger = TfBoardLogger(summary_writer)
    tflogger.register('reward', dtype=tf.float32)
    tflogger.register('eval_reward', dtype=tf.float32)

    saver = tf.train.Saver(global_vars)
    def init_fn(sess):
        if is_chief and args.load is not None:
            saver.restore(sess, args.load)

    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=logdir,
                             init_op=init_op,
                             local_init_op=local_init_op,
                             global_step=global_step,
                             recovery_wait_secs=1,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(global_vars),
                             saver=saver)

    config = tf.ConfigProto(device_filters=["/job:ps", worker_device])

    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        if is_chief:
            pbar = tqdm(total=constants.FINAL_STEP, dynamic_ncols=True)

        def end_episode(reward, step, episode):
            if is_chief:
                step = sess.run(global_step)
                tflogger.plot('reward', reward, step)
                # hack to set the direct step
                pbar.n = step
                pbar.update(0)
                pbar.set_description('step: {}, reward: {}'.format(step, reward))

        def after_action(state, reward, step, local_step):
            step = sess.run(global_step)
            if constants.LR_DECAY == 'linear':
                decay = 1.0 - (float(step) / constants.FINAL_STEP)
                if decay < 0.0:
                    decay = 0.0
                sess.run(decay_lr_op, feed_dict={decayed_lr: constants.LR * decay})

        agent.update_local()

        trainer = Trainer(
            env=wrapped_env,
            agent=agent,
            render=args.render,
            state_shape=state_shape[:-1],
            state_window=constants.STATE_WINDOW,
            final_step=constants.FINAL_STEP,
            after_action=after_action,
            end_episode=end_episode,
            progress_bar=False,
            training=not args.demo
        )
        trainer.start()
