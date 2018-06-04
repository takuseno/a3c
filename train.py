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

from rlsaber.log import TfBoardLogger, dump_constants
from rlsaber.trainer import AsyncTrainer
from rlsaber.trainer import Evaluator, Recorder
from rlsaber.env import EnvWrapper
from rlsaber.preprocess import atari_preprocess
from actions import get_action_space
from network import make_network
from agent import Agent
from datetime import datetime

def make_agent(model, actions, optimizer, state_shape, phi, name, constants):
    return Agent(
        model,
        actions,
        optimizer,
        gamma=constants.GAMMA,
        lstm_unit=constants.LSTM_UNIT,
        time_horizon=constants.TIME_HORIZON,
        policy_factor=constants.POLICY_FACTOR,
        value_factor=constants.VALUE_FACTOR,
        entropy_factor=constants.ENTROPY_FACTOR,
        grad_clip=constants.GRAD_CLIP,
        state_shape=state_shape,
        phi=phi,
        name=name
    )

def main():
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--load', type=str)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.logdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    env_name = args.env
    tmp_env = gym.make(env_name)
    # box environment
    if len(tmp_env.observation_space.shape) == 1:
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
    dump_constants(constants, os.path.join(outdir, 'constants.json'))

    sess = tf.Session()
    sess.__enter__()

    model = make_network(
        constants.CONVS, constants.FCS,
        lstm=constants.LSTM, padding=constants.PADDING)

    # share Adam optimizer with all threads!
    lr = tf.Variable(constants.LR)
    decayed_lr = tf.placeholder(tf.float32)
    decay_lr_op = lr.assign(decayed_lr)
    if constants.OPTIMIZER == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=0.1)
    else:
        optimizer = tf.train.AdamOptimizer(lr)


    master = make_agent(
        model, actions, optimizer, state_shape, phi, 'global', constants)

    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    saver = tf.train.Saver(global_vars)
    if args.load:
        saver.restore(sess, args.load)

    agents = []
    envs = []
    for i in range(args.threads):
        name = 'worker{}'.format(i)
        agent = make_agent(
            model, actions, optimizer, state_shape, phi, name, constants)
        agents.append(agent)
        env = gym.make(args.env)
        env.seed(i)
        wrapped_env = EnvWrapper(
            env,
            r_preprocess=lambda r: np.clip(r, -1, 1),
            s_preprocess=state_preprocess
        )
        envs.append(wrapped_env)

    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    tflogger = TfBoardLogger(summary_writer)
    tflogger.register('reward', dtype=tf.float32)
    tflogger.register('eval_reward', dtype=tf.float32)
    end_episode = lambda r, gs, s, ge, e: tflogger.plot('reward', r, gs)

    def after_action(state, reward, shared_step, global_step, local_step):
        if constants.LR_DECAY == 'linear':
            decay = 1.0 - (float(shared_step) / constants.FINAL_STEP)
            if decay < 0.0:
                decay = 0.0
            sess.run(decay_lr_op, feed_dict={decayed_lr: constants.LR * decay})
        if shared_step % 10 ** 6 == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=shared_step)

    evaluator = Evaluator(
        env=copy.deepcopy(envs[0]),
        state_shape=state_shape[:-1],
        state_window=constants.STATE_WINDOW,
        eval_episodes=constants.EVAL_EPISODES,
        recorder=Recorder(outdir) if args.record else None,
        record_episodes=constants.RECORD_EPISODES
    )
    def should_eval(last_step, last_episode, global_step, global_episode, step, episode):
        duration = global_step - last_step
        return duration > 0 and duration > constants.EVAL_INTERVAL
    end_eval = lambda gs, ge, s, e, r: tflogger.plot('eval_reward', np.mean(r), gs)

    trainer = AsyncTrainer(
        envs=envs,
        agents=agents,
        render=args.render,
        state_shape=state_shape[:-1],
        state_window=constants.STATE_WINDOW,
        final_step=constants.FINAL_STEP,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo,
        n_threads=args.threads,
        evaluator=evaluator,
        should_eval=should_eval,
        end_eval=end_eval
    )
    trainer.start()

if __name__ == '__main__':
    main()
