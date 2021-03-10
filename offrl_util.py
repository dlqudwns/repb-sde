
import random
import numpy as np
import tensorflow as tf
import gym
import h5py
from offrl_envs.wrapper import NormalizeActionWrapper
import offrl_envs.static as static
from offrl.base import OfflineReplayBuffer

def make_env(env_id):
    mapping = {"halfcheetah": "HalfCheetah-v2",
               "hopper": "Hopper-v2",
               "ant": "AntTruncatedObsEnv-v2",
               "walker": "Walker2d-v2"}
    return NormalizeActionWrapper(gym.make(mapping[env_id]))

def get_termination_fn(env_id):
    return static[env_id].termination_fn # pylint: disable=unsubscriptable-object

def set_global_seeds(seed, env=None):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)

def get_dataset(env_id, dataset_type, dataset_dir='', terminate_on_end=False):
    def get_keys(h5file):
        keys = []
        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)
        h5file.visititems(visitor)
        return keys

    env = make_env(env_id)

    h5path = f'{dataset_dir}/{env_id}_{dataset_type}.hdf5'
    dataset_file = h5py.File(h5path, 'r')
    dataset = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
    dataset_file.close()

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        if env_id == 'ant':
            obs = dataset['observations'][i][:27]
            new_obs = dataset['observations'][i+1][:27]
        else:
            obs = dataset['observations'][i]
            new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1
    print(f'Loaded dataset {h5path} - Consists of {N} experiences')
    return env, OfflineReplayBuffer(obs_, action_, reward_, next_obs_, done_)
