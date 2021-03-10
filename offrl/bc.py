import pickle
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from offrl_evaluate import evaluate_policy
from .base import SquahedGaussianActor # pylint: disable=relative-beyond-top-level

class BC(tf.keras.layers.Layer):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(BC, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.obs_ph = tf.keras.layers.Input(state_dim)
        self.actor = SquahedGaussianActor(action_dim, hidden_dim=hidden_dim)
        self.det_action, self.squashed_action, _, self.policy_dist = self.actor([self.obs_ph])

        def loss(y_true, _):
            return self.actor.nlogp(self.policy_dist, y_true)

        self.det_model = tf.keras.models.Model(inputs=[self.obs_ph], outputs=[self.det_action])
        self.model = tf.keras.models.Model(inputs=[self.obs_ph], outputs=[self.squashed_action])
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.model.compile(optimizer=self.optimizer, loss=loss)
        self.standardizer = None

    def train(self, replay_buffer, iterations, batch_size=64):
        # Sample replay buffer / batch
        obs, action, *_ = replay_buffer.sample(batch_size * iterations)
        self.model.fit(x=obs, y=action, batch_size=batch_size, epochs=1)

    #######################################
    # interfaces for dmain.py
    #######################################

    def batch_learn(self, replay_buffer, vec_env, total_timesteps, log_interval,
                    result_filepath=None, **_):
        self.standardizer = replay_buffer.standardizer

        # Start...
        start_time = time.time()
        timestep = 0
        result = {'timestep': [], 'evals': [], 'infos': []}
        with tqdm(total=total_timesteps, desc="BC") as pbar:
            while timestep < total_timesteps:
                evaluation = evaluate_policy(vec_env, self)
                print(f't={timestep}: {evaluation} (elapsed_time={time.time() - start_time})')

                self.train(replay_buffer, iterations=log_interval, batch_size=64)
                pbar.update(log_interval)
                timestep += log_interval

                if result_filepath:
                    result['timestep'].append(timestep)
                    result['evals'].append(evaluation)
                    np.save(result_filepath + '.tmp.npy', result)

        return result

    def predict(self, obs, deterministic=False):
        assert len(obs.shape) == 2
        obs = self.standardizer(obs)
        if deterministic:
            return self.det_model.predict(obs)
        else:
            return self.model.predict(obs)

    def get_parameters(self):
        parameters = []
        weights = self.get_weights()
        for idx, variable in enumerate(self.trainable_variables):
            weight = weights[idx]
            parameters.append((variable.name, weight))
        return parameters

    def save(self, filepath):
        parameters = self.get_parameters()
        with open(filepath, 'wb') as f:
            pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
