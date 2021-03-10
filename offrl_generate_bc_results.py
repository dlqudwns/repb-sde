import os
import numpy as np
from offrl_main import ROOT_DIR
from offrl_util import get_dataset, set_global_seeds
from offrl.bc import BC
from offrl_evaluate import make_vectorized_env

np.set_printoptions(precision=3, suppress=True, linewidth=250)
TOT_STEPS = 100000

def run_bc(env_id, dataset_type, seed=0):
    print(f'running {env_id}-{dataset_type}')
    env, buffer = get_dataset(env_id, dataset_type, dataset_dir=f'{ROOT_DIR}d4rl_dataset')
    set_global_seeds(seed, env)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # Load model
    model = BC(state_dim, action_dim, hidden_dim=256)

    # Set result path
    result_dir = f'bc_results/{env_id}-{dataset_type}'
    os.makedirs(result_dir, exist_ok=True)
    result_filepath = f'{result_dir}/seed_{seed}.npy'
    if os.path.exists(result_filepath):
        print(f'Result file already exists: {result_filepath}')
        return np.load(result_filepath, allow_pickle=True)[()]

    # Run algorithm and save the result
    print('==============================================')
    print('Run: ', result_filepath)
    vec_env = make_vectorized_env(env_id)  # for policy evaluation
    result = model.batch_learn(
        buffer, vec_env, total_timesteps=TOT_STEPS, log_interval=10000,
        result_filepath=result_filepath)
    np.save(result_filepath, result)
    os.remove(result_filepath + '.tmp.npy')
    vec_env.close()
    del vec_env
    return result


if __name__ == "__main__":
    run_setups = []
    for _env_name in ['walker', 'hopper', 'halfcheetah', 'ant']:
        for _dataset_type in ['random', 'medium', 'mixed', 'medium_expert']:
            run_bc(_env_name, _dataset_type)
