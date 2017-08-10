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
from actions import get_action_space
from network import make_network
from agent import Agent
from worker import Worker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    sess = tf.Session()
    sess.__enter__()

    model = make_network(
        [[32, 3, 2, 1], [32, 3, 2, 1], [32, 3, 2, 1], [32, 3, 2, 1]])

    env_name = args.env
    actions = get_action_space(env_name)
    master = Agent(model, len(actions), name='global')

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

    workers = []
    for i in range(4):
        worker = Worker('worker{}'.format(i), model, global_step, env_name, render=False)
        workers.append(worker)

    initialize()

    coord = tf.train.Coordinator()
    threads = []
    for i in range(4):
        worker_thread = lambda: workers[i].run(sess)
        thread = threading.Thread(target=worker_thread)
        thread.start()
        threads.append(thread)
        time.sleep(0.1)
    coord.join(threads)

if __name__ == '__main__':
    main()
