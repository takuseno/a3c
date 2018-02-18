import threading
import multiprocessing
import argparse
import cv2
import gym
import copy
import os
import time
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.tensorflow.log import TfBoardLogger
from lightsaber.rl.trainer import AsyncTrainer
from lightsaber.rl.env_wrapper import EnvWrapper
from actions import get_action_space
from network import make_network
from agent import Agent
from datetime import datetime

def main():
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--final-step', type=int, default=10 ** 7)
    parser.add_argument('--load', type=str)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--outdir', type=str, default=date)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    sess = tf.Session()
    sess.__enter__()

    model = make_network(
        [[16, 8, 4, 0], [32, 4, 2, 0]])

    env_name = args.env
    actions = get_action_space(env_name)
    master = Agent(model, actions, name='global')

    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    saver = tf.train.Saver(global_vars)
    if args.load:
        saver.restore(sess, args.load)

    def s_preprocess(state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (84, 84))
        state = np.array(state, dtype=np.float32)
        return state / 255.0

    agents = []
    envs = []
    for i in range(args.threads):
        agent = Agent(model, actions, name='worker{}'.format(i))
        agents.append(agent)
        env = EnvWrapper(
            gym.make(args.env),
            r_preprocess=lambda r: np.clip(r, -1, 1),
            s_preprocess=s_preprocess
        )
        envs.append(env)

    initialize()

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    logger = TfBoardLogger(summary_writer)
    logger.register('reward', dtype=tf.int8)
    end_episode = lambda r, gs, s, ge, e: logger.plot('reward', r, gs)

    def after_action(state, reward, shared_step, global_step, local_step):
        if shared_step % 10 ** 6 == 0:
            path = os.path.join(outdir, '{}/model.ckpt'.format(shared_step))
            saver.save(sess, path)

    trainer = AsyncTrainer(
        envs=envs,
        agents=agents,
        render=args.render,
        state_shape=[84, 84],
        state_window=1,
        final_step=args.final_step,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo,
        n_threads=args.threads
    )
    trainer.start()

if __name__ == '__main__':
    main()
