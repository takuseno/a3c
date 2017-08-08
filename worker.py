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

class Worker:
    def __init__(self, name, model, global_step, env_name):
        env = env.make(env_name)
        actions = get_action_space(env_name)
        explorer = LinearDecayExplorer(final_exploration_step=100000)

        self.name = name
        self.agent = Agent(model, len(actions), explorer, name=name)
        self.global_step = global_step
        self.inc_global_step = global_step.assign_add(1)

    def run(self):
        while True:
            states = np.zeros((4, 84, 84), dtype=np.float32)
            reward = 0
            done = False
            clipped_reward = 0
            sum_of_rewards = 0
            step = 0
            state = env.reset()

            while True:
                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                state = cv2.resize(state, (84, 84))
                states = np.roll(states, 1, axis=0)
                states[0] = state

                if done:
                    summary, _ = sess.run([merged, reward_summary], feed_dict={reward_summary: sum_of_rewards})
                    train_writer.add_summary(summary, global_step)
                    agent.stop_episode_and_train(states, clipped_reward, done=done)
                    break

                action = actions[agent.act_and_train(states, clipped_reward)]

                state, reward, done, info = env.step(action)

                if reward > 0:
                    clipped_reward = 1
                elif reward < 0:
                    clipped_reward = -1
                else:
                    clipped_reward = 0
                sum_of_rewards += reward
                step += 1
                local_step += 1 
                get_session().run(inc_global_step)

            print('worker: {}, global: {}, local: {}, reward: {}'.format(
                    self.name, self.global_step.get_variable(), local_step, sum_of_rewards))
