from lightsaber.rl.util import Rollout
from lightsaber.rl.trainer import AgentInterface
import build_graph
import numpy as np
import tensorflow as tf


class Agent(AgentInterface):
    def __init__(self, model, actions, gamma=0.99, name='global'):
        self.actions = actions
        self.gamma = gamma
        self.t = 0
        self.name = name

        self._act,\
        self._train,\
        self._update_local,\
        self._action_dist = build_graph.build_train(
            model=model,
            num_actions=len(actions),
            scope=name
        )

        self.initial_state = np.zeros((1, 256), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_action = None
        self.last_value = None

        self.rollout = Rollout()

    def train(self, bootstrap_value):
        states = np.array(self.rollout.states, dtype=np.float32)
        actions = np.array(self.rollout.actions, dtype=np.uint8)
        v, adv = self.rollout.compute_v_and_adv(bootstrap_value, self.gamma)

        loss = self._train(
            states,
            self.rollout.features[0][0],
            self.rollout.features[0][1],
            actions,
            v,
            adv
        )
        self._update_local()
        return loss

    def act(self, obs, reward, training=True):
        # change state shape to WHC
        obs = np.transpose(obs, [1, 2, 0])
        # clip reward
        reward = np.clip(reward, -1.0, 1.0)
        # take next action
        prob, value, rnn_state = self._act(
            obs.reshape(1, 84, 84, 1),
            self.rnn_state0,
            self.rnn_state1
        )
        action = np.random.choice(range(len(self.actions)), p=prob[0])

        if training:
            if len(self.rollout.states) == 5:
                self.train(self.last_value)
                self.rollout.flush()

            if self.last_obs is not None:
                self.rollout.add(
                    state=self.last_obs,
                    reward=reward,
                    action=self.last_action,
                    value=self.last_value,
                    terminal=False,
                    feature=[self.rnn_state0, self.rnn_state1]
                )

        self.t += 1
        self.rnn_state0, self.rnn_state1 = rnn_state
        self.last_obs = obs
        self.last_action = action
        self.last_value = value[0][0]
        return self.actions[action]

    def stop_episode(self, obs, reward, done=False, training=True):
        if training:
            reward = np.clip(reward, -1.0, 1.0)
            self.rollout.add(
                state=self.last_obs,
                action=self.last_action,
                reward=reward,
                value=self.last_value,
                terminal=True
            )
            self.train(0)
            self.rollout.flush()
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_action = None
        self.last_value = None
