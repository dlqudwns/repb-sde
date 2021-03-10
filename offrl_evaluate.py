import multiprocessing
import time
import numpy as np
from stable_baselines.common.vec_env import SubprocVecEnv
from tqdm import tqdm
from offrl_util import make_env, set_global_seeds

def make_vectorized_env(env_id, n_envs=multiprocessing.cpu_count()):
    def make_env_proc(seed=0):
        def _init():
            _env = make_env(env_id)
            _env.seed(seed)
            return _env
        set_global_seeds(seed)
        return _init
    vec_env = SubprocVecEnv([make_env_proc(i) for i in range(n_envs)])
    return vec_env

def evaluate_policy(vec_env, agent, num_episodes=30, deterministic=False, render=False):
    episode_rewards = []

    with tqdm(total=num_episodes, desc="policy_evaluation", ncols=70) as pbar:
        episode_reward = np.zeros(vec_env.num_envs)
        obs = vec_env.reset()
        while len(episode_rewards) < num_episodes:
            if hasattr(agent, 'predict'):
                action = agent.predict(obs, deterministic=deterministic)
            else:
                action = agent(obs)
            next_obs, reward, done, _ = vec_env.step(action)

            episode_reward = episode_reward + reward
            if np.count_nonzero(done) > 0:
                episode_rewards += list(episode_reward[done])
                episode_reward[done] = 0
                pbar.update(np.count_nonzero(done))

            obs = next_obs

            if render:
                vec_env.render()
                time.sleep(0.1)

        episode_rewards = np.array(episode_rewards)

    mean = np.mean(episode_rewards)
    ste = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
    print(f'\n{mean} +- {ste}')
    return np.mean(episode_rewards)
