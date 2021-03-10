
import time
import itertools
from collections import OrderedDict
import tensorflow as tf
import numpy as np

np.set_printoptions(precision=5)

class TensorStandardScaler:
    """Helper class for automatically normalizing inputs into the network. """
    def __init__(self, x_dim, scope=''):
        """Initializes a scaler.
        Arguments: x_dim (int): The dimensionality of the inputs into the scaler. Returns: None. """
        with tf.variable_scope(scope):
            self._mean = tf.get_variable(
                name="scaler_mu", shape=[1, x_dim], initializer=tf.constant_initializer(0.0),
                trainable=False)
            self._stddev = tf.get_variable(
                name="scaler_std", shape=[1, x_dim], initializer=tf.constant_initializer(1.0),
                trainable=False)

    def fit(self, data):
        mean = np.mean(data, axis=0, keepdims=True)
        stddev = np.std(data, axis=0, keepdims=True)
        stddev[stddev < 1e-12] = 1.0
        self._mean.load(mean)
        self._stddev.load(stddev)

    def transform(self, data):
        return (data - self._mean) / self._stddev

    def inverse_transform(self, data):
        return self._stddev * data + self._mean

    def inverse_transform_gaussian(self, data_mean, data_logstd):
        return self._stddev * data_mean + self._mean, data_logstd + tf.log(self._stddev)

    def get_vars(self):
        return [self._mean, self._stddev]

class EnsembleDense(tf.keras.layers.Layer):

    def __init__(self, ensemble_size, units, activation=None, weight_decay=0.0, **kwargs):
        super(EnsembleDense, self).__init__(**kwargs)

        self.ensemble_size = int(ensemble_size)
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = tf.keras.activations.get(activation)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, max_ndim=3)
        self.weight_decay = weight_decay
        self.kernel, self.bias, self.built = None, None, False

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, max_ndim=3, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel', shape=[self.ensemble_size, last_dim, self.units],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=1 / (2 * np.sqrt(last_dim))),
            regularizer=tf.keras.regularizers.l2(self.weight_decay), trainable=True)
        self.bias = self.add_weight(
            'bias', shape=[self.ensemble_size, 1, self.units],
            initializer=tf.keras.initializers.zeros(), trainable=True)
        self.built = True

    def call(self, inputs):
        if len(inputs.shape) == 2:
            raw_output = tf.einsum("ij,ajk->aik", inputs, self.kernel) + self.bias
        elif len(inputs.shape) == 3 and inputs.shape[0].value == self.ensemble_size:
            raw_output = tf.matmul(inputs, self.kernel) + self.bias
        else:
            raise ValueError("Invalid input dimension.")
        return self.activation(raw_output)

class BNN(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, num_networks=7, num_elites=5, hidden_dim=200,
                 actor=None, ipm_coef=0.1, use_pre_mean=True):
        super(BNN, self).__init__()
        self.num_nets = num_networks
        self.num_elites = num_elites
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sess = tf.keras.backend.get_session()
        print(f'[ BNN ] Initializing model: BNN | {num_networks} networks | {num_elites} elites')

        self.input_scaler = TensorStandardScaler(input_dim, scope='input')
        self.target_scaler = TensorStandardScaler(output_dim, scope='target')
        self.sess.run(tf.variables_initializer(
            self.input_scaler.get_vars() + self.target_scaler.get_vars()))

        swish = tf.keras.layers.Activation(lambda x: x * tf.keras.backend.sigmoid(x))
        self.representation_layers = [
            EnsembleDense(num_networks, hidden_dim, swish, 0.000025),
            EnsembleDense(num_networks, hidden_dim, swish, 0.00005),
            EnsembleDense(num_networks, hidden_dim, 'tanh', 0.000075)]
        self.after_representation = [
            EnsembleDense(num_networks, hidden_dim, swish, 0.000075)]
        self.hidden_layers = self.representation_layers + self.after_representation
        self.mean_layer = EnsembleDense(num_networks, output_dim, None, 0.0001)
        self.var_layer = EnsembleDense(num_networks, output_dim, None, 0.0001)
        self.layers = self.hidden_layers + [self.mean_layer, self.var_layer]

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.max_logvar = tf.keras.backend.variable(
            np.full([1, output_dim], 0.5), dtype=tf.float32, name="max_log_var")
        self.min_logvar = tf.keras.backend.variable(
            np.full([1, output_dim], -10), dtype=tf.float32, name="min_log_var")

        self.input_3d_ph = tf.placeholder(
            shape=[self.num_nets, None, input_dim], name='inputs_3d', dtype=tf.float32)
        self.target_ph = tf.placeholder(
            shape=[self.num_nets, None, output_dim], name='train_targets', dtype=tf.float32)
        self.terminal_ph = tf.placeholder(
            shape=[self.num_nets, None, 1], name='terminals_3d', dtype=tf.float32)
        self.input_3d_ph2 = tf.placeholder(
            shape=[self.num_nets, None, input_dim], name='init_inputs_3d', dtype=tf.float32)
        mean_3d, log_var_3d, _ = self._compile_outputs(self.input_3d_ph)

        self.mse_loss = tf.reduce_mean(tf.square(mean_3d - self.target_ph), axis=[1, 2])

        weighted_mse_losses = tf.reduce_mean(
            tf.square(mean_3d - self.target_ph) * tf.exp(-log_var_3d), axis=[1, 2])
        var_losses = tf.reduce_mean(log_var_3d, axis=[1, 2])
        train_loss = tf.reduce_sum(weighted_mse_losses + var_losses) \
            + 0.01 * tf.reduce_sum(self.max_logvar) - 0.01 * tf.reduce_sum(self.min_logvar)
        train_loss += tf.add_n(self.losses)
        if actor is not None:
            train_loss += ipm_coef * self.build_ipm(actor)

        self.train_op = self.optimizer.minimize(train_loss, var_list=self.trainable_variables)

        self.sess.run(tf.variables_initializer(
            self.trainable_variables + self.optimizer.variables()))

        self.input_2d_ph = tf.placeholder(
            shape=[None, input_dim], name='inputs_2d', dtype=tf.float32)
        self.mean_2d, log_var_2d, pre_mean_2d = self._compile_outputs(self.input_2d_ph)
        self.var_2d = tf.exp(log_var_2d)
        self.uncertainty = tf.math.reduce_std(pre_mean_2d, axis=0) \
            if use_pre_mean else tf.math.reduce_std(self.mean_2d, axis=0)

        self._state, self._prev_losses, self._epochs_since_update = None, None, None
        self._model_inds = None

    def build_ipm(self, actor):
        gamma = 0.99
        rep = self.representation(self.input_3d_ph)
        obs_dim = self.input_dim - actor.action_dim
        next_obs = self.input_3d_ph[:, :, :obs_dim] + self.target_ph[:, :, 1:]
        init_obs = self.input_3d_ph2[:, :, :obs_dim]
        _, next_action, _, _ = actor([tf.reshape(next_obs, [-1, obs_dim])])
        next_action = tf.reshape(next_action, [self.num_nets, -1, actor.action_dim])
        _, init_action, _, _ = actor([tf.reshape(init_obs, [-1, obs_dim])])
        init_action = tf.reshape(init_action, [self.num_nets, -1, actor.action_dim])
        nrep = self.representation(tf.concat([next_obs, next_action], axis=-1))
        irep = self.representation(tf.concat([init_obs, init_action], axis=-1))

        def dot_k(rep1, rep2):
            # use dot kernel
            return tf.matmul(rep1, rep2, transpose_b=True)
        mask = 1 - self.terminal_ph
        ipm = dot_k(rep, rep) + (1 - gamma) ** 2 * dot_k(irep, irep) \
            + gamma ** 2 * dot_k(nrep, nrep) * mask * tf.transpose(mask, [0, 2, 1])\
            - 2 * (1 - gamma) * dot_k(irep, rep) \
            - 2 * gamma * dot_k(nrep, rep) * mask \
            + 2 * (1 - gamma) * gamma * dot_k(nrep, irep) * mask
        return tf.reduce_sum(tf.reduce_mean(ipm, axis=[1, 2]))

    def representation(self, inputs):
        cur_out = self.input_scaler.transform(inputs)
        for layer in self.representation_layers:
            cur_out = layer(cur_out)
        return cur_out

    def _compile_outputs(self, inputs):
        cur_out = self.input_scaler.transform(inputs)
        for layer in self.hidden_layers:
            cur_out = layer(cur_out)
        pre_mean = self.mean_layer(cur_out)
        logvar = self.var_layer(cur_out)

        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)

        mean, logvar = self.target_scaler.inverse_transform_gaussian(pre_mean, logvar)

        return mean, logvar, pre_mean

    def _save_state(self, idx):
        weights = self.get_weights()[2:14]
        self._state[idx] = [weight[idx] for weight in weights]

    def _set_state(self):
        new_weights = self.get_weights()[:2]
        for weight_list in zip(*self._state):
            new_weights.append(np.stack(weight_list, axis=0))
        new_weights.extend(self.get_weights()[14:])
        self.set_weights(new_weights)

    def random_inds(self, batch_size):
        inds = np.random.choice(self._model_inds, size=batch_size)
        return inds

    def set_model_inds(self, inputs, targets):
        with self.sess.as_default():
            self.input_scaler.fit(inputs)
            self.target_scaler.fit(targets)
        n_holdout = int(0.1 * inputs.shape[0])
        perm = np.random.permutation(inputs.shape[0])
        inputs = np.tile(inputs[perm[:n_holdout]][None], [self.num_nets, 1, 1])
        targets = np.tile(targets[perm[:n_holdout]][None], [self.num_nets, 1, 1])
        losses = self.sess.run(self.mse_loss, feed_dict={
            self.input_3d_ph: inputs, self.target_ph: targets})
        sorted_inds = np.argsort(losses)
        self._model_inds = sorted_inds[:self.num_elites].tolist()

    def train(self, inputs, targets, terminals, max_epochs=None, max_t=None, batch_size=256,
              max_epochs_since_update=5, max_holdout=1000, holdout_ratio=0.2):
        """Trains/Continues network training

        Arguments:
            inputs (np.ndarray): Network inputs in the training dataset in rows.
            targets (np.ndarray): Network target outputs in the training dataset in rows
                corresponding to the rows in inputs.
            batch_size (int): The minibatch size to be used for training.
            epochs (int): Number of epochs (full network passes that will be done.
        Returns: None
        """
        self._state = [[] for _ in range(self.num_nets)]
        self._prev_losses = [1e10 for _ in range(self.num_nets)]
        self._epochs_since_update = 0

        # Split into training and holdout sets
        N = inputs.shape[0]
        n_holdout = min(max_holdout, int(N * holdout_ratio))
        n_training = N - n_holdout
        perm = np.random.permutation(N)
        inputs, holdout_inputs = inputs[perm[n_holdout:]], inputs[perm[:n_holdout]]
        targets, holdout_targets = targets[perm[n_holdout:]], targets[perm[:n_holdout]]
        terminals = terminals[perm[n_holdout:]]

        print(f'[ BNN ] Training {inputs.shape} | Holdout: {holdout_inputs.shape}')
        with self.sess.as_default():
            self.input_scaler.fit(inputs)
            self.target_scaler.fit(targets)
        idxs = np.random.randint(n_training, size=[self.num_nets, n_training])
        idxs2 = idxs[:, np.random.permutation(n_training)]
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        start_time = time.time()
        epoch_iter = range(max_epochs) if max_epochs else itertools.count()
        for _ in epoch_iter:
            for batch_num in range(int(np.ceil(n_training / batch_size))):
                # update using maximum 1e5 data
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch_idxs2 = idxs2[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                self.sess.run(
                    self.train_op,
                    feed_dict={self.input_3d_ph: inputs[batch_idxs],
                               self.target_ph: targets[batch_idxs],
                               self.input_3d_ph2: inputs[batch_idxs2],
                               self.terminal_ph: terminals[batch_idxs]})

            # shuffle idxs (column-wise)
            sorted_idxs = np.argsort(np.random.uniform(size=idxs.shape), axis=-1)
            idxs = idxs[np.arange(self.num_nets)[:, None], sorted_idxs]
            idxs2 = idxs[:, np.random.permutation(n_training)]

            holdout_losses = self.sess.run(self.mse_loss, feed_dict={
                self.input_3d_ph: holdout_inputs, self.target_ph: holdout_targets})
            updated = False
            for i, (current, best) in enumerate(zip(holdout_losses, self._prev_losses)):
                if (best - current) / best > 0.01:
                    self._prev_losses[i] = current
                    self._save_state(i)
                    updated = True
            print(self._prev_losses)
            self._epochs_since_update = 0 if updated else self._epochs_since_update + 1
            if self._epochs_since_update > max_epochs_since_update:
                break

            if max_t and time.time() - start_time > max_t:
                print(f'Breaking because of timeout!, max_t: {max_t}')
                break

        self._set_state()

        holdout_losses = self.sess.run(self.mse_loss, feed_dict={
            self.input_3d_ph: holdout_inputs, self.target_ph: holdout_targets})
        sorted_inds = np.argsort(holdout_losses)
        self._model_inds = sorted_inds[:self.num_elites].tolist()
        print('Using {} / {} models: {}'.format(self.num_elites, self.num_nets, self._model_inds))

        val_loss = (np.sort(holdout_losses)[:self.num_elites]).mean()
        model_metrics = {'val_loss': val_loss}
        print('[ BNN ] Holdout', np.sort(holdout_losses), model_metrics)
        return OrderedDict(model_metrics)

    def predict(self, inputs):
        return self.sess.run([self.mean_2d, self.var_2d, self.uncertainty], feed_dict={
            self.input_2d_ph: inputs})

class FakeEnv:

    def __init__(self, model, termination_fn, mopo_penalty_coeff=0., my_penalty_coeff=0.):
        self.model = model
        self.termination_fn = termination_fn
        self.mopo_penalty_coeff = mopo_penalty_coeff
        self.my_penalty_coeff = my_penalty_coeff

    def step(self, obs, act, replay_buffer=None):
        assert len(obs.shape) == len(act.shape)

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_means, ensemble_vars, uncertainty = self.model.predict(inputs)

        batch_size = len(inputs)
        model_inds = self.model.random_inds(batch_size)
        batch_inds = np.arange(0, batch_size)
        means, stds \
            = ensemble_means[model_inds, batch_inds], np.sqrt(ensemble_vars[model_inds, batch_inds])
        means[:, 1:] += obs
        samples = means + np.random.normal(size=means.shape) * stds

        rewards, next_obs = samples[:, :1], samples[:, 1:]

        if replay_buffer is not None:
            terminals = self.termination_fn(replay_buffer.unstandardizer(obs), act,
                                            replay_buffer.unstandardizer(next_obs))
            unstan_ensemble_vars = np.copy(ensemble_vars)
            unstan_ensemble_vars[:, :, 1:] \
                = unstan_ensemble_vars[:, :, 1:] * (replay_buffer.obs_std ** 2)
        else:
            terminals = self.termination_fn(obs, act, next_obs)

        penalty = np.amax(np.linalg.norm(np.sqrt(unstan_ensemble_vars), axis=2), axis=0)[:, None]
        assert penalty.shape == rewards.shape
        rewards = rewards - self.mopo_penalty_coeff * penalty

        my_penalty = np.linalg.norm(uncertainty, axis=1, keepdims=True)
        rewards = rewards - my_penalty * self.my_penalty_coeff

        return next_obs, rewards, terminals, {}
