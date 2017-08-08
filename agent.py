import build_graph
import numpy as np
import tensorflow as tf

class Agent:
    def __init__(self, model, num_actions, exploration, name='global', lr=2.5e-4, gamma=0.99):
        self.num_actions = num_actions
        self.exploration = exploration
        self.gamma = gamma
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

        self.rnn_state = None
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
        return np.array(list(reversed(returns)), dtype=np.float32).flatten()

    def train(self, bootstrap_value):
        states = np.zeros((30, 4, 84, 84), dtype=np.float32)
        states[:len(self.states)] = np.array(self.states, dtype=np.float32) / 255.0
        actions = np.zeros((30), dtype=np.float32)
        actions[:len(self.actions)] = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values + [bootstrap_value], dtype=np.float32).flatten()

        target_values = self.compute_returns(rewards, self.gamma, bootstrap_value)[:len(rewards)]
        advantages = 0.99 * values[1:] + rewards - values[:-1]

        tmp_target_values = np.zeros((30), dtype=np.float32)
        tmp_target_values[:len(target_values)] = target_values
        target_values = tmp_target_values

        tmp_advantages = np.zeros((30), dtype=np.float32)
        tmp_advantages[:len(advantages)] = advantages
        advantages = tmp_advantages

        loss = self._train(states, None, actions, target_values, advantages)
        return loss

    def act_and_train(self, obs, reward):
        normalized_obs = np.zeros((30, 4, 84, 84), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        prob, rnn_state = self._act(normalized_obs, self.rnn_state)
        action = np.random.choice(range(self.num_actions), p=prob[0])
        value = self._state_value(normalized_obs, self.rnn_state)[0]

        if len(self.states) == 30:
            bootstrap_value = self._state_value(normalized_obs)[0]
            self.train(bootstrap_value)
            self.states = []
            self.rewards = []
            self.actions = []
            self.values = []
            self._update_local()

        if self.last_obs is not None:
            self.states.append(self.last_obs)
            self.rewards.append(reward)
            self.actions.append(self.last_action)
            self.values.append(self.last_value)

        self.t += 1
        self.rnn_state = np.array(rnn_state, dtype=np.float32)
        self.last_obs = obs
        self.last_reward = reward
        self.last_action = action
        self.last_value = value
        return action

    def stop_episode_and_train(self, obs, reward, done=False):
        self.states.append(self.last_obs)
        self.rewards.append(reward)
        self.actions.append(self.last_action)
        self.values.append(self.last_value)
        self.train(0)
        self.stop_episode()

    def stop_episode(self):
        self.rnn_state = None
        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
