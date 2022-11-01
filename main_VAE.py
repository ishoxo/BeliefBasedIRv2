"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings

import ipdb
import numpy as np
import torch

# get configs
from config.gridworld import \
    args_grid_belief_oracle, args_grid_rl2, args_grid_varibad
from config.pointrobot import \
    args_pointrobot_multitask, args_pointrobot_varibad, args_pointrobot_rl2, args_pointrobot_humplik
from config import args_cheetah_varibad
from environments.parallel_envs import make_vec_envs
from learner import Learner
from vae_learner import VAELearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    # parser.add_argument('--env-type', default='cheetah')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == 'cheetah':
        args = args_cheetah_varibad.get_args(rest_args)

    elif env == 'gridworld_belief_oracle':
        args = args_grid_belief_oracle.get_args(rest_args)
    elif env == 'gridworld_varibad':
        args = args_grid_varibad.get_args(rest_args)
    elif env == 'gridworld_rl2':
        args = args_grid_rl2.get_args(rest_args)


    else:
        raise Exception("Invalid Environment")

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None,
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        if args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(args)
        else:
            learner = VAELearner(args)
        learner.train()


if __name__ == '__main__':
    main()
