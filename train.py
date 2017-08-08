import threading
import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf

from util import initialize, get_session
from actions import get_action_space
from network import make_network
from agent import Agent
from explorer import LinearDecayExplorer
from worker import Worker

def main():
    sess = tf.Session()
    sess.__enter__()

    model = make_network(
        [[32, 3, 2], [32, 3, 2], [32, 3, 2], [32, 3, 2]])

    env_name = 'PongDeterministic-v4'
    actions = get_action_space(env_name)
    master = Agent(model, len(actions), None, name='global')

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

    workers = []
    for i in range(8):
        worker = Worker('worker{}'.format(i), model, global_step, env_name)
        workers.append(worker)

    initialize()

    coord = tf.train.Coordinator()
    threads = []
    for i in range(8):
        worker_thread = lambda: workers[i].run()
        thread = threading.Thread(target=worker_thread)
        thread.start()
        threads.append(thread)
    coord.join(threads)

if __name__ == '__main__':
    main()
