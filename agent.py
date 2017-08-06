import build_graph
import numpy as np
import tensorflow as tf

class Agent:
    def __init__(self, model, num_actions, exploration, name='global', lr=2.5e-4):
        self.exoploration = exploration
        self.t = 0

        act, train, update_local, action_dist, state_value = build_graph.build_train(
            model=model,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
            scope=name
        )

        self._act = act
        self._train = train
        self._update_local = update_local
        self._action_dist = action_dist
        self._state_value = state_value

        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None

        self.states = []
        self.rewards = []
        self.actions = []
        self.values = []

    def compute_returns(self, rewards, gamma, bootstrap_value):
        returns = [bootstrap_value]
        length = len(rewards)
        for i in reversed(range(length)):
            value = rewards[i] + gamma * returns[length - 1 - i]
            returns.append(value)
        return reversed(returns)

    def train(self, bootstrap_value):
        states = np.array(self.states, dtype=np.float32) / 255.0
        rewards = np.array(self.rewards)
        actions = np.array(self.actions)
        values = np.array(self.values + [bootstrap_value], dtype=np.float32)

        target_values = self.compute_returns(rewards, gamma, bootstrap_value)
        advantages = 0.99 * values[1:] + rewards - values[:-1]
        loss = self._train(states, actions, target_values, advantages)
        return loss

    def act_and_train(self, obs, reward):
        update_eps = self.exploration.value(self.t)
        normalized_obs = np.zeros((30, 4, 84, 84), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32)/ 255.0
        action = self._act(normalized_obs, update_eps=update_eps)[0]
        value = self._state_value(normalized_obs)[0]

        if len(states) == 30:
            bootstrap_value = self._state_value(normalized_obs)
            self.train(bootstrap_value)
            self.states = []
            self.rewards = []
            self.actions = []
            self.values = []
            self.update_local()

        if self.last_obs is not None
            self.states.append(self.last_obs)
            self.rewards.append(reward)
            self.actions.append(self.action)
            self.values.append(self.last_value)

        self.t += 1
        self.last_obs = obs
        self.last_reward = reward
        self.last_action = action
        self.last_value = value
        return action

    def stop_episode_and_train(self, obs, reward, done=False):
        sefl.states.append(self.last_obs)
        self.rewards.append(self.reward)
        self.actions.append(self.last_action)
        self.values.append(self.last_value)
        self.train(0)
        self.stop_episode()

    def stop_episode(self):
        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
