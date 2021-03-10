import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

SCALE_DIAG_MIN_MAX = (-20, 2)
EPS = 1e-6

def apply_squashing_func(sample, logp):
    """
    Squash the ouput of the gaussian distribution and account for that in the log probability.

    :param sample: (tf.Tensor) Action sampled from Gaussian distribution
    :param logp: (tf.Tensor) Log probability before squashing
    """
    # Squash the output
    squashed_action = tf.tanh(sample)
    squashed_action_logp = logp - tf.reduce_sum(tf.log(1 - squashed_action ** 2 + 1e-6), axis=1)
    # incurred by change of variable
    return squashed_action, squashed_action_logp

class OnlineReplayBuffer:

    def __init__(self, state_dim, action_dim, buffer_size):
        self.buffer_size = buffer_size
        self.obs = np.zeros([buffer_size, state_dim])
        self.action = np.zeros([buffer_size, action_dim])
        self.reward = np.zeros([buffer_size, 1])
        self.next_obs = np.zeros([buffer_size, state_dim])
        self.done = np.zeros([buffer_size, 1])
        self._pointer = 0
        self.size = 0
        self.buffer = [self.obs, self.action, self.reward, self.next_obs, self.done]

    def add_samples(self, *samples):
        num_samples = len(samples[0])
        index = np.arange(self._pointer, self._pointer + num_samples) % self.buffer_size
        for buf, new_samples in zip(self.buffer, samples):
            assert len(new_samples) == num_samples
            buf[index] = new_samples
        self._pointer = (self._pointer + num_samples) % self.buffer_size
        self.size = min(self.size + num_samples, self.buffer_size)

    def add_sample(self, *sample):
        none_sample = [np.array(each)[None] for each in sample]
        self.add_samples(*none_sample)

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return [each[indices] for each in self.buffer]

    def sample_obs(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.obs[indices]

    def format_for_model_training(self):
        obs, action, next_obs, reward = self.obs[:self.size], \
            self.action[:self.size], self.next_obs[:self.size], self.reward[:self.size]
        inputs = np.concatenate([obs, action], axis=-1)
        targets = np.concatenate([reward, next_obs - obs], axis=-1)
        return inputs, targets

# Simple replay buffer
class OfflineReplayBuffer:

    def __init__(self, obs, action, reward, next_obs, done):
        self.obs, self.action, self.reward, self.next_obs, self.done \
            = obs, action, reward, next_obs, done

        self.obs_mean = np.mean(self.obs, axis=0, keepdims=True)
        self.obs_std = np.std(self.obs, axis=0, keepdims=True) + 1e-3
        self.stan_obs = self.standardizer(np.array(self.obs))
        self.stan_next_obs = self.standardizer(np.array(self.next_obs))

    def standardizer(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def unstandardizer(self, obs):
        return obs * self.obs_std + self.obs_mean

    def format_for_model_training(self):
        inputs = np.concatenate([self.stan_obs, self.action], axis=-1)
        delta_obs = self.stan_next_obs - self.stan_obs
        targets = np.concatenate([np.array(self.reward)[:, None], delta_obs], axis=-1)
        terminals = np.reshape(np.array(self.done), [-1, 1])
        return inputs, targets, terminals

    def sample(self, batch_size):
        obs, action, reward, next_obs, done = [], [], [], [], []
        indices = np.random.randint(0, len(self.obs), size=batch_size)
        for idx in indices:
            obs.append(self.obs[idx])
            action.append(self.action[idx])
            reward.append(self.reward[idx])
            next_obs.append(self.next_obs[idx])
            done.append(self.done[idx])
        obs, next_obs, action = np.array(obs), np.array(next_obs), np.array(action)

        obs, next_obs = self.standardizer(obs), self.standardizer(next_obs)
        return obs, action, np.array(reward)[:, None], next_obs, np.array(done)[:, None]

    def sample_obs(self, batch_size):
        indices = np.random.randint(0, len(self.obs), size=batch_size)
        obs = [self.obs[idx] for idx in indices]
        return self.standardizer(np.array(obs))

class SquahedGaussianActor(tf.keras.layers.Layer):

    def __init__(self, action_dim, hidden_dim=256):
        super(SquahedGaussianActor, self).__init__()
        self.action_dim = action_dim

        # Actor parameters
        self.a_l0 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='a/f0')
        self.a_l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='a/f1')
        self.a_l2_mu = tf.keras.layers.Dense(action_dim, name='a/f2_mu')
        self.a_l2_log_std = tf.keras.layers.Dense(action_dim, name='a/f2_log_std')

    def feedforward(self, obs):
        h = self.a_l0(obs)
        h = self.a_l1(h)
        mean = self.a_l2_mu(h)
        log_std = self.a_l2_log_std(h)
        std = tf.exp(tf.clip_by_value(log_std, *SCALE_DIAG_MIN_MAX))
        return mean, std

    def call(self, inputs, **_):
        obs, = inputs
        mean, std = self.feedforward(obs)
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        dist.shape = mean.shape

        sampled_action = dist.sample()
        sampled_action_logp = dist.log_prob(sampled_action)
        squahsed_action, squahsed_action_logp = \
            apply_squashing_func(sampled_action, sampled_action_logp)
        deterministic_action, _ = \
            apply_squashing_func(mean, dist.log_prob(mean))

        return deterministic_action, squahsed_action, squahsed_action_logp, dist

    def nlogp(self, dist, action):
        ''' negative logp of unnormalized action '''
        before_squahed_action = tf.atanh(
            tf.clip_by_value(action, -1 + EPS, 1 - EPS))
        log_likelihood = dist.log_prob(before_squahed_action)
        log_likelihood -= tf.reduce_sum(
            tf.log(1 - action ** 2 + EPS), axis=1)
        return -tf.reduce_mean(log_likelihood)

class VNetwork(tf.keras.layers.Layer):

    def __init__(self, output_dim=1, hidden_dim=64):
        super(VNetwork, self).__init__()

        self.v_l0 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='v/f0')
        self.v_l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='v/f1')
        self.v_l2 = tf.keras.layers.Dense(output_dim, name='v/f2')

    def call(self, inputs, **_):
        obs, = inputs
        h = self.v_l0(obs)
        h = self.v_l1(h)
        return self.v_l2(h)

class QNetwork(tf.keras.layers.Layer):

    def __init__(self, output_dim=1, num_critics=2, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.num_critics = num_critics

        self.qs_l0, self.qs_l1, self.qs_l2 = [], [], []
        for i in range(self.num_critics):
            self.qs_l0.append(tf.keras.layers.Dense(hidden_dim, activation='relu', name=f'q{i}/f0'))
            self.qs_l1.append(tf.keras.layers.Dense(hidden_dim, activation='relu', name=f'q{i}/f1'))
            self.qs_l2.append(tf.keras.layers.Dense(output_dim, name=f'q{i}/f2'))

    def call(self, inputs, **_):
        obs, action = inputs
        obs_action = tf.concat([obs, action], axis=1)
        outputs = []
        for i in range(self.num_critics):
            h = self.qs_l0[i](obs_action)
            h = self.qs_l1[i](h)
            outputs.append(self.qs_l2[i](h))
        return outputs
