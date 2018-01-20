import build_graph
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, model, num_actions,
            final_step=10 ** 7, gamma=0.99, name='global'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.final_step = final_step
        self.t = 0
        self.name = name

        self._act,\
        self._train,\
        self._update_local,\
        self._action_dist,\
        self._state_value = build_graph.build_train(
            model=model,
            num_actions=num_actions,
            scope=name
        )

        self.initial_state = np.zeros((1, 256), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None

        self.states = []
        self.rewards = []
        self.actions = []
        self.values = []

    def train(self, bootstrap_value, summary_writer, global_step):
        states = np.array(self.states, dtype=np.float32) / 255.0
        actions = np.array(self.actions, dtype=np.uint8)
        returns = []
        R = bootstrap_value
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns = np.array(list(reversed(returns)), dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)

        advantages = returns - values

        factor = 1.0 - float(global_step) / self.final_step
        if factor < 0:
            factor = 0
        lr = 7e-4 * factor

        summary, loss = self._train(states, self.initial_state,
                self.initial_state, actions, returns, advantages, lr)
        summary_writer.add_summary(summary, self.t)
        self._update_local()
        return loss

    def act(self, obs):
        normalized_obs = np.zeros((1, 84, 84, 4), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        prob, rnn_state = self._act(
            normalized_obs,
            self.rnn_state0,
            self.rnn_state1
        )
        action = np.random.choice(range(self.num_actions), p=prob[0])
        self.rnn_state0, self.rnn_state1 = rnn_state
        return action

    def act_and_train(self, obs, reward, summary_writer, global_step):
        normalized_obs = np.zeros((1, 84, 84, 4), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        prob, rnn_state = self._act(
            normalized_obs,
            self.rnn_state0,
            self.rnn_state1
        )
        action = np.random.choice(range(self.num_actions), p=prob[0])
        value = self._state_value(
            normalized_obs,
            self.rnn_state0,
            self.rnn_state1
        )[0][0]

        if len(self.states) == 5:
            self.train(self.last_value, summary_writer, global_step)
            self.states = []
            self.rewards = []
            self.actions = []
            self.values = []

        if self.last_obs is not None:
            self.states.append(self.last_obs)
            self.rewards.append(reward)
            self.actions.append(self.last_action)
            self.values.append(self.last_value)

        self.t += 1
        self.rnn_state0, self.rnn_state1 = rnn_state
        self.last_obs = obs
        self.last_reward = reward
        self.last_action = action
        self.last_value = value
        return action

    def stop_episode_and_train(self, obs, reward,
            summary_writer, global_step, done=False):
        self.states.append(self.last_obs)
        self.rewards.append(reward)
        self.actions.append(self.last_action)
        self.values.append(self.last_value)
        self.train(0, summary_writer, global_step)
        self.stop_episode()

    def stop_episode(self):
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
