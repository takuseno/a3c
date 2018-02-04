from lightsaber.rl.util import Rollout
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
        self._action_dist = build_graph.build_train(
            model=model,
            num_actions=num_actions,
            scope=name
        )

        self.initial_state = np.zeros((1, 256), np.float32)
        self.rnn_state0 = copy.deepcopy(self.initial_state)
        self.rnn_state1 = copy.deepcopy(self.initial_state)
        self.last_obs = None
        self.last_action = None
        self.last_value = None

        self.rollout = Rollout()

    def train(self, bootstrap_value, global_step):
        states = np.array(self.rollout.states, dtype=np.float32) / 255.0
        actions = np.array(self.rollout.actions, dtype=np.uint8)
        v, adv = self.rollout.compute_v_and_adv(bootstrap_value, self.gamma)

        factor = 1.0 - float(global_step) / self.final_step
        if factor < 0:
            factor = 0
        lr = 7e-4 * factor

        loss = self._train(
            states,
            self.rollout.features[0][0],
            self.rollout.features[0][1],
            actions,
            v,
            adv,
            lr
        )
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

    def act_and_train(self, obs, reward, global_step):
        normalized_obs = np.zeros((1, 84, 84, 1), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        prob, value, rnn_state = self._act(
            normalized_obs,
            self.rnn_state0,
            self.rnn_state1
        )
        action = np.random.choice(range(self.num_actions), p=prob[0])

        if len(self.rollout.states) == 5:
            self.train(self.last_value, global_step)
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
        return action

    def stop_episode_and_train(self, obs, reward, global_step, done=False):
        self.rollout.add(
            state=self.last_obs,
            action=self.last_action,
            reward=reward,
            value=self.last_value,
            terminal=True
        )
        self.train(0, global_step)
        self.stop_episode()

    def stop_episode(self):
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_action = None
        self.last_value = None
        self.rollout.flush()
