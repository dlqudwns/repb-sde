import os
import numpy as np

from offrl_evaluate import make_vectorized_env
from offrl.repbrl import REPBRL
from offrl_util import set_global_seeds, get_termination_fn, get_dataset

np.set_printoptions(precision=3, suppress=True, linewidth=250)
ROOT_DIR = ''

def run(env_id, dataset_type, total_timesteps, seed, model_id, alg_params, model_coeff=0.1):
    env, buffer = get_dataset(env_id, dataset_type, dataset_dir=f'{ROOT_DIR}d4rl_dataset')
    set_global_seeds(seed, env)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # Load model
    dataset_name = f'{env_id}-{dataset_type}'
    if model_id == 0:
        alg_name = f'repbrl_{model_id}_' \
            + '_'.join([f'{k}_{v}' for k, v in alg_params.items()])
    else:
        alg_name = f'repbrl_{model_id}_{model_coeff}_' \
            + '_'.join([f'{k}_{v}' for k, v in alg_params.items()])
    model = REPBRL(state_dim, action_dim, get_termination_fn(env_id), **alg_params)

    # Set result path
    result_dir = f'{ROOT_DIR}eval_results/{dataset_name}/{alg_name}'
    os.makedirs(result_dir, exist_ok=True)
    result_filepath = f'{result_dir}/seed_{seed}.npy'
    if os.path.exists(result_filepath):
        print(f'Result file already exists: {result_filepath}')
        return np.load(result_filepath, allow_pickle=True)[()]

    # Run algorithm and save the result
    print('==============================================')
    print('Run: ', result_filepath)
    vec_env = make_vectorized_env(env_id)  # for policy evaluation
    os.makedirs(f'{ROOT_DIR}trained_policy/{model_id}/', exist_ok=True)

    model_path = f'{ROOT_DIR}trained_model/{model_id}/{env_id}'
    model_path = model_path + f'-{dataset_type}_{seed}.npy' if model_id == 0 \
        else model_path + f'_{dataset_type}_{seed}_{model_coeff}.npy'
    result = model.batch_learn(
        buffer, vec_env, total_timesteps=total_timesteps, log_interval=10000, seed=seed,
        result_filepath=result_filepath,
        model_path=model_path,
        policy_path=f'{ROOT_DIR}trained_policy/{model_id}/{dataset_name}_{seed}_' \
            + '_'.join([f'{k}_{v}' for k, v in alg_params.items()]) + '.npy')
    np.save(result_filepath, result)
    os.remove(result_filepath + '.tmp.npy')
    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="name of the env to train", default='halfcheetah')
    parser.add_argument("--dataset_type", help="name of the dataset", default='medium_expert')
    parser.add_argument("--total_timesteps", help="total timesteps", default=1500000, type=int)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--model_id", help="model id", default=0, type=int)
    args = parser.parse_args()

    full_alg_params = {'rollout_steps':5, 'ipm_coef':0.0, 'mopo_penalty_coeff':0.0,
                       'my_penalty_coeff':10.0}
    run(args.env_id, args.dataset_type, args.total_timesteps, args.seed, args.model_id,
        full_alg_params, model_coeff=0.01)
