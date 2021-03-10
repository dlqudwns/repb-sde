import time
import numpy as np
import tensorflow as tf
from offrl_evaluate import evaluate_policy
from .mbrl_base import BNN, FakeEnv # pylint: disable=relative-beyond-top-level
from .base import OnlineReplayBuffer, SquahedGaussianActor, QNetwork # pylint: disable=relative-beyond-top-level

class REPBRL(tf.keras.layers.Layer):
    def __init__(self, state_dim, action_dim, termination_fn, rollout_steps=5,
                 max_model_t=None, ipm_coef=0, mopo_penalty_coeff=0.0, my_penalty_coeff=0.0,
                 use_pre_mean=True, **_):
        self.state_dim, self.action_dim, self.max_model_t \
            = state_dim, action_dim, max_model_t
        self.obs_ph, self.action_ph, self.reward_ph, self.terminal_ph, self.next_obs_ph \
            = [tf.keras.layers.Input(d) for d in [state_dim, action_dim, 1, 1, state_dim]]
        self.data_obs_ph, self.data_action_ph, self.data_next_obs_ph, self.data_init_obs_ph,\
            self.data_terminal_ph = [
                tf.keras.layers.Input(d) for d in [state_dim, action_dim, state_dim, state_dim, 1]]

        # Model params
        self.model_hidden_dim = 200
        self.model = BNN(input_dim=state_dim + action_dim, output_dim=state_dim + 1,
                         hidden_dim=self.model_hidden_dim, use_pre_mean=use_pre_mean)
        self.fake_env = FakeEnv(self.model, termination_fn, mopo_penalty_coeff=mopo_penalty_coeff,
                                my_penalty_coeff=my_penalty_coeff)

        # policy optimization params
        self.rollout_steps = rollout_steps # 1 or 5 in MOPO
        self.sac_hidden_dim, self.batch_size = 256, 256
        self.rollout_batch_size = int(1e4) if rollout_steps == 1000 else int(1e6 / rollout_steps)
        self.model_retain_epochs = 5
        self.model_train_freq, self.evaluate_freq = 10000, 10000
        self.real_ratio, self.gamma, self.tau, self.learning_rate = 0.05, 0.99, 5e-3, 3e-4
        self.env_batch_size = int(self.batch_size * self.real_ratio)
        self.model_batch_size = self.batch_size - self.env_batch_size
        self.model_buffer_size = int(self.model_retain_epochs * rollout_steps \
            * self.rollout_batch_size * self.evaluate_freq / self.model_train_freq)
        self.target_entropy = -self.action_dim

        # IPM params
        self.ipm_coef = ipm_coef
        self.sess = tf.keras.backend.get_session()

        # build SAC
        self.actor = actor = SquahedGaussianActor(action_dim, hidden_dim=self.sac_hidden_dim)
        critic_online = QNetwork(hidden_dim=self.sac_hidden_dim)
        critic_target = QNetwork(hidden_dim=self.sac_hidden_dim)
        log_ent_coef = tf.keras.backend.variable(1.0, dtype=tf.float32, name='log_ent_coef')
        ent_coef = tf.exp(log_ent_coef)

        # Actor training (with kl regularization)
        self.det_actions, self.cur_actions, cur_logp, dist = actor([self.obs_ph])
        _, next_actions, next_logp, _ = actor([self.next_obs_ph])
        ipm = self._build_ipm(actor)

        bc_loss = actor.nlogp(dist, self.action_ph)
        cur_qpis = critic_online([self.obs_ph, self.cur_actions])
        actor_loss = tf.reduce_mean(ent_coef * cur_logp - tf.reduce_min(cur_qpis, axis=0))
        if self.ipm_coef > 0:
            actor_loss += tf.reduce_mean(ipm_coef * ipm)
        ent_coef_loss = -tf.reduce_mean(
            log_ent_coef * tf.stop_gradient(cur_logp + self.target_entropy))

        cur_qs = critic_online([self.obs_ph, self.action_ph])
        next_qs = critic_target([self.next_obs_ph, next_actions])
        next_q = tf.reduce_min(next_qs, axis=0) - ent_coef * next_logp
        q_backup = tf.stop_gradient(self.reward_ph + (1 - self.terminal_ph) * self.gamma * next_q)
        q_loss = tf.reduce_sum([tf.losses.mean_squared_error(q_backup, cur_q) for cur_q in cur_qs])

        bc_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.behavior_clone_op = bc_optimizer.minimize(bc_loss, var_list=actor.trainable_variables)
        actor_opt = tf.train.AdamOptimizer(self.learning_rate)
        actor_train_op = actor_opt.minimize(actor_loss, var_list=actor.trainable_variables)
        critic_opt = tf.train.AdamOptimizer(self.learning_rate)
        critic_train_op = critic_opt.minimize(q_loss, var_list=critic_online.trainable_variables)
        entropy_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        entropy_train_op = entropy_opt.minimize(ent_coef_loss, var_list=log_ent_coef)

        with tf.control_dependencies([critic_train_op]):
            # Update target network
            source_params = critic_online.trainable_variables
            target_params = critic_target.trainable_variables
            target_update_op = [
                tf.assign(target, (1 - self.tau) * target + self.tau * source)
                for target, source in zip(target_params, source_params)]

        self.sess.run(tf.variables_initializer(
            actor_opt.variables() + critic_opt.variables() + entropy_opt.variables()
            + bc_optimizer.variables()))
        critic_target.set_weights(critic_online.get_weights())
        self.step_ops = [actor_train_op, critic_train_op, entropy_train_op, target_update_op]
        self.step_ops += [actor_loss, q_loss, tf.reduce_mean(cur_qs), ent_coef,
                          tf.reduce_mean(dist.entropy()), tf.reduce_mean(cur_logp), ipm]
        self.info_labels = ['actor_loss', 'q_loss', 'mean(cur_qs)', 'ent_coef', 'entropy',
                            'cur_logp', 'ipm']

        self.standardizer = None
        self.unstandardizer = None

    def _build_ipm(self, actor):
        _, initial_actions, _, _ = actor([self.data_init_obs_ph])
        _, next_actions, _, _ = actor([self.data_next_obs_ph])
        rep = self.model.representation(tf.concat([self.data_obs_ph, self.data_action_ph], axis=-1))
        nrep = self.model.representation(tf.concat([self.data_next_obs_ph, next_actions], axis=-1))
        irep = self.model.representation(
            tf.concat([self.data_init_obs_ph, initial_actions], axis=-1))

        def dot_k(rep1, rep2):
            # use dot kernel
            return tf.reduce_mean(tf.matmul(rep1, rep2, transpose_b=True), axis=0)
        mask = 1 - self.data_terminal_ph
        ipm = dot_k(rep, rep) + (1 - self.gamma) ** 2 * dot_k(irep, irep) \
            + self.gamma ** 2 * dot_k(nrep, nrep) * mask * tf.transpose(mask)\
            - 2 * (1 - self.gamma) * dot_k(irep, rep) \
            - 2 * self.gamma * dot_k(nrep, rep) * mask \
            + 2 * (1 - self.gamma) * self.gamma * dot_k(nrep, irep) * mask
        return tf.reduce_mean(ipm)

    def _pretrain_actors(self, replay_buffer, iterations):
        for _ in range(iterations):
            obs, act, *_ = replay_buffer.sample(self.batch_size)
            self.sess.run(self.behavior_clone_op, feed_dict={self.obs_ph: obs, self.action_ph: act})

    def train_step(self, replay_buffer, model_buffer):
        if self.model_batch_size == 0:
            env_batch = replay_buffer.sample(self.env_batch_size)
            obs, action, reward, next_obs, terminal = env_batch
        elif self.env_batch_size == 0:
            model_batch = model_buffer.sample(self.model_batch_size)
            obs, action, reward, next_obs, terminal = model_batch
        else:
            env_batch = replay_buffer.sample(self.env_batch_size)
            model_batch = model_buffer.sample(self.model_batch_size)
            obs, action, reward, next_obs, terminal = [np.concatenate([each1, each2], axis=0) \
                for each1, each2 in zip(env_batch, model_batch)]
        feed_dict = {self.obs_ph: obs, self.action_ph: action, self.reward_ph: reward,
                     self.next_obs_ph: next_obs, self.terminal_ph: terminal}
        # sample from obs instead of initial obs
        #feed_dict[self.initial_obs_ph] = replay_buffer.sample_obs(self.batch_size)
        obs, action, _, next_obs, terminal = replay_buffer.sample(self.batch_size)
        init_obs = replay_buffer.sample_obs(self.batch_size)
        feed_dict.update({self.data_obs_ph:obs, self.data_action_ph:action, \
            self.data_next_obs_ph: next_obs, self.data_terminal_ph: terminal, \
                self.data_init_obs_ph: init_obs})

        step_result = self.sess.run(self.step_ops, feed_dict=feed_dict)
        return step_result[-len(self.info_labels):]

    def batch_learn(self, buffer, vec_env, total_timesteps, result_filepath=None, model_path=None,
                    policy_path=None, **_):
        model_buffer = OnlineReplayBuffer(self.state_dim, self.action_dim, self.model_buffer_size)
        replay_buffer = buffer
        self.standardizer = replay_buffer.standardizer

        start_time = time.time()
        result = {'timestep': [], 'evals': [], 'infos': []}
        v_max, v_min = 10 * np.max(buffer.reward) / (1 - self.gamma), \
            10 * np.min(buffer.reward) / (1 - self.gamma)
        v_max, v_min = max(v_max, -v_min), min(-v_max, v_min)
        #self._pretrain_actors(replay_buffer, 30000)

        if self.real_ratio < 1.0: # train model
            if model_path is None:
                print('[ REPBRL ] Train Model')
                train_inputs, train_targets, terminals = replay_buffer.format_for_model_training()
                self.model.train(
                    train_inputs, train_targets, terminals, batch_size=self.batch_size,
                    max_epochs=None, max_t=self.max_model_t) #train models on standardized input outputs
            else:
                train_inputs, train_targets, terminals = replay_buffer.format_for_model_training()
                self.model.set_weights(np.load(model_path, allow_pickle=True)[()])
                self.model.set_model_inds(train_inputs, train_targets)
        infos = []
        for timestep in range(total_timesteps):
            if timestep % self.model_train_freq == 0 and self.real_ratio < 1.0:
                self._rollout_model(replay_buffer, model_buffer)

            infos.append(self.train_step(replay_buffer, model_buffer))

            if timestep % self.evaluate_freq == 0:
                current_info = np.mean(infos, axis=0)
                infos = []
                evaluation = evaluate_policy(vec_env, self)
                result['timestep'].append(timestep)
                result['evals'].append(evaluation)
                result['infos'].append(current_info)
                print(f't={timestep}: {evaluation} (elapsed_time={time.time() - start_time})')
                print('\n============================')
                for label, value in zip(self.info_labels, current_info):
                    print(f'{label:>12}: {value:>10.5f}')
                print('============================\n', flush=True)
                np.save(policy_path, self.actor.get_weights())
                #if current_info[2] > v_max or current_info[2] < v_min:
                    #print('Q diverged! terminating training')
                    #break

                if result_filepath:
                    np.save(result_filepath + '.tmp.npy', result)
        return result

    def _rollout_model(self, replay_buffer, model_buffer):
        print(f'[ Model Rollout ] Starting | '
              f'Rollout steps: {self.rollout_steps} | Batch size: {self.rollout_batch_size}')
        # get standardized obs from real samples
        obs = replay_buffer.sample_obs(self.rollout_batch_size)
        steps_added = []
        for i in range(self.rollout_steps):
            act = self.sess.run(self.cur_actions, feed_dict={self.obs_ph:obs})
            next_obs, rew, term, _ = self.fake_env.step(obs, act, replay_buffer)
            # fake env gives standardized obs
            steps_added.append(len(obs))
            model_buffer.add_samples(obs, act, rew, next_obs, term)
            # model_buffer contains standardized obs

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print(f'[ Model Rollout ] Breaking early: {i} | '
                      f'{nonterm_mask.sum()} / {nonterm_mask.shape}')
                break

            obs = next_obs[nonterm_mask]

        mean_rollout_length = sum(steps_added) / self.rollout_batch_size
        print(f'[ Model Rollout ] Added: {sum(steps_added):.1e} | Model pool: '
              f'{model_buffer.size:.1e} (max {model_buffer.buffer_size:.1e}) | '
              f'Length: {mean_rollout_length}')

    def predict(self, obs, deterministic=False):
        assert len(obs.shape) == 2
        if self.standardizer is not None:
            obs = self.standardizer(obs)
        if deterministic:
            return self.sess.run(self.det_actions, feed_dict={self.obs_ph:obs})
        else:
            return self.sess.run(self.cur_actions, feed_dict={self.obs_ph:obs})

    def _initial_exploration(self, train_env, replay_buffer):
        # initial exploration, 500 steps in case of pendulum
        obs = train_env.reset()
        for _ in range(5000):
            action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
            next_obs, reward, done, _ = train_env.step(action)
            replay_buffer.add_sample(obs, action, [reward], next_obs, [float(done)])
            obs = next_obs
            if done:
                obs = train_env.reset()

    def learn(self, train_env, vec_env, total_epochs, result_filepath=None, epoch_length=1000,
              update_per_step=20):
         # epoch length: 250 in case of pendulum
         # update_per_step: depends on domain
        replay_buffer = OnlineReplayBuffer(self.state_dim, self.action_dim, int(1e6))
        model_buffer = OnlineReplayBuffer(self.state_dim, self.action_dim, self.model_buffer_size)
        self._initial_exploration(train_env, replay_buffer)
        episode_rewards = [0.0]

        start_time = time.time()
        result = {'timestep': [], 'evals': [], 'infos': []}
        obs = train_env.reset()
        for epoch in range(total_epochs):
            infos_values = []
            for timestep in range(epoch_length):
                if timestep % self.model_train_freq == 0 and self.real_ratio < 1.0:
                    print(f'[ REPBRL ] Freq {self.model_train_freq} | '
                          f'timestep {timestep}/{epoch_length}')
                    train_inputs, train_targets = replay_buffer.format_for_model_training()
                    self.model.train(
                        train_inputs, train_targets, batch_size=self.batch_size, holdout_ratio=0.2,
                        max_t=self.max_model_t)
                    self._rollout_model(replay_buffer, model_buffer)
                # Take an action and store transition in the replay buffer.
                action = self.predict(np.array([obs]), deterministic=False)[0].flatten()
                next_obs, reward, done, _ = train_env.step(action)
                replay_buffer.add_sample(obs, action, [reward], next_obs, [float(done)])
                obs = next_obs
                episode_rewards[-1] += reward
                if done:
                    obs = train_env.reset()
                    episode_rewards.append(0.0)
                if replay_buffer.can_sample(self.env_batch_size):
                    infos_values.append(self._sac_train(
                        replay_buffer, model_buffer, update_per_step, verbose=0))
            evaluation = evaluate_policy(vec_env, self, deterministic=True)
            infos_value = np.mean(infos_values, axis=0)
            result['timestep'].append(epoch * epoch_length)
            result['evals'].append(evaluation)
            result['infos'].append(infos_value)
            print(f't={epoch * epoch_length}: {evaluation} '
                  f'(elapsed_time={time.time() - start_time})')
            print('\n============================')
            for label, value in zip(self.info_labels, infos_value):
                print(f'{label:>12}: {value:>10.3f}')
            print('============================\n')

            if result_filepath:
                np.save(result_filepath + '.tmp.npy', result)

        return episode_rewards
