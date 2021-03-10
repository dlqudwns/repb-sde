
import os
import argparse
import numpy as np
import tensorflow as tf
from offrl.base import SquahedGaussianActor
from offrl.mbrl_base import BNN
from offrl_util import set_global_seeds, get_dataset

ROOT_DIR = '/ext/bjlee/repbrl/' if os.path.isdir('/ext') else ''
def train_model(env_id, dataset_type, seed, model_iter, ipm_coef=0.1):
    env, buffer = get_dataset(env_id, dataset_type, dataset_dir=f'{ROOT_DIR}d4rl_dataset/')
    set_global_seeds(seed, env)

    model_path \
        = f'{ROOT_DIR}trained_model/{model_iter}/{env_id}_{dataset_type}_{seed}_{ipm_coef}.npy'
    print(model_path)
    if os.path.exists(model_path):
        print(f'Result file already exists: {model_path}')
        return np.load(model_path, allow_pickle=True)[()]

    if model_iter == 0:
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        model = BNN(input_dim=state_dim + action_dim, output_dim=state_dim + 1, hidden_dim=200)
        train_inputs, train_targets, terminals = buffer.format_for_model_training()
        model.train(train_inputs, train_targets, terminals, batch_size=256)
    else:
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        weights = np.load(
            f'{ROOT_DIR}trained_policy/{model_iter - 1}/{env_id}-{dataset_type}_{seed}.npy',
            allow_pickle=True)
        actor = SquahedGaussianActor(action_dim, hidden_dim=256)
        actor([tf.keras.Input(state_dim)])
        actor.set_weights(weights)

        model = BNN(input_dim=state_dim + action_dim, output_dim=state_dim + 1, hidden_dim=200,
                    actor=actor, ipm_coef=ipm_coef)
        train_inputs, train_targets, terminals = buffer.format_for_model_training()
        model.train(train_inputs, train_targets, terminals, batch_size=256)
    #train models on standardized input outputs
    np.save(model_path, model.get_weights())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", help="condor pid", default=0, type=int)
    args = parser.parse_args()
    run_setups = []
    for _seed in range(5, 10):
        for _env_id in ['halfcheetah']:
            for _dataset_type in ['medium_expert']:
                for _ipm_coef in [0.01, 0.1, 1.0, 10.0]:
                    run_setups.append([_env_id, _dataset_type, _seed, 1, _ipm_coef])

    print(f'Total number of setups: {len(run_setups)}')
    print(f'Current pid: {args.pid}')
    print('==================================================')
    print(run_setups[args.pid])
    print('==================================================')
    train_model(*run_setups[args.pid])
