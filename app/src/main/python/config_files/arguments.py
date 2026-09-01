from argparse import ArgumentParser
import argparse
import json

def add_bool_arg(parser, name, default=False, help=''):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=name, action='store_false', help=help)
    parser.set_defaults(**{name: default})


def ppo_args():
    parser = argparse.ArgumentParser(description="Training Configuration CLI")

    # Experiment / General
    parser.add_argument('--exp_name', type=str, default='training', help='the name of this experiment')
    parser.add_argument('--seed', type=int, default=874, help='seed of the experiment')
    add_bool_arg(parser=parser, name='torch_deterministic', default=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    add_bool_arg(parser=parser, name='cuda', default=True, help='if toggled, cuda will be enabled by default')
    add_bool_arg(parser=parser, name='track', default=True, help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb_project_name', type=str, default="planet-wars-ppo", help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="the entity (team) of wandb's project")
    add_bool_arg(parser=parser, name='capture_video', default=False, help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--num_hours', type=float, default=24, help='number of hours before experiment is killed by HPC, decimal hours are accepted')

    # Algorithm specific arguments
    parser.add_argument('--env_id', type=str, default="PlanetWarsForwardModel", help='the id of the environment. Filled in runtime, either `PlanetWarsForwardModel` or `PlanetWarsForwardModelGNN` according to agent type')
    parser.add_argument('--total_timesteps', type=int, default=15000000, help='total timesteps of the experiments')
    parser.add_argument('--learning_rate', type=float, default=7.5e-4, help='the learning rate of the optimizer')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "muon"], help="the optimizer to use: 'adam' or 'muon'")
    parser.add_argument('--num_envs', type=int, default=32, help='the number of parallel game environments')
    parser.add_argument('--num_steps', type=int, default=2048, help='the number of steps to run in each environment per policy rollout')
    add_bool_arg(parser=parser, name='anneal_lr', default=True, help='Toggle learning rate annealing for policy and value networks')
    add_bool_arg(parser=parser, name='anneal_ent_coef', default=False, help='Toggle entropy coefficient annealing')
    parser.add_argument('--gamma', type=float, default=0.999, help='the discount factor gamma')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='the lambda for the general advantage estimation')
    parser.add_argument('--num_minibatches', type=int, default=64, help='the number of mini-batches')
    parser.add_argument('--update_epochs', type=int, default=4, help='the K epochs to update the policy')
    add_bool_arg(parser=parser, name='norm_adv', default=True, help='Toggles advantages normalization')
    parser.add_argument('--clip_coef', type=float, default=0.2, help='the surrogate clipping coefficient')
    add_bool_arg(parser=parser, name='clip_vloss', default=True, help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='coefficient of the entropy')
    parser.add_argument('--vf_coef', type=float, default=1.3, help='coefficient of the value function')
    parser.add_argument('--max_grad_norm', type=float, default=3.5, help='the maximum norm for the gradient clipping')
    parser.add_argument('--target_kl', type=float, default=None, help='the target KL divergence threshold')

    # Planet Wars specific
    parser.add_argument('--agent_type', type=str, default="gnn", choices=["mlp", "edge_mlp", "gnn", "edge_gnn"], help='the type of agent to train')
    parser.add_argument('--num_planets', type=int, default=None, help='number of planets in the game. If None, will be set to a random value between num_planets_min and num_planets_max (new_map_each_run needs to be set to true)')
    parser.add_argument('--num_planets_min', type=int, default=10, help='minimum number of planets in the game')
    parser.add_argument('--num_planets_max', type=int, default=30, help='maximum number of planets in the game')
    parser.add_argument('--node_feature_dim', type=int, default=0, help='dimension of node features (owner, ship_count, growth_rate, x, y)')
    parser.add_argument('--max_ticks', type=int, default=2000, help='maximum game ticks')
    add_bool_arg(parser=parser, name='use_adjacency_matrix', default=False, help='whether to include adjacency matrix in observations')
    add_bool_arg(parser=parser, name='flatten_observation', default=True, help='Filled on run time, mlp uses flattened observation, gnn uses graph observation')
    parser.add_argument('--discretized_ratio_bins', type=int, default=0, help='number of bins for the discretized ratio actor. Set to 0 to disable discretization')
    add_bool_arg(parser=parser, name='discretize_include_zero', default=False, help='whether to include zero in the discretized ratio bins')
    add_bool_arg(parser=parser, name='new_map_each_run', default=True, help='whether to create a new map for each run or use the same map')
    parser.add_argument('--maps_pkl', type=str, default=None, help='file path to .pkl file containing maps to be used.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension for the layers')
    parser.add_argument('--profile_path', type=str, default=None, help='Path to save profiling data, if None profiling is disabled')
    add_bool_arg(parser=parser, name='use_async', default=True, help='if toggled, AsyncVectorEnv will be used')
    add_bool_arg(parser=parser, name='use_tick', default=False, help='if toggled, the game tick will be passed as an observation')
    parser.add_argument('--model_weights', type=str, default=None, help='path to model weights to load (for continuing training)')
    add_bool_arg(parser=parser, name='resume_training', default=False, help='if toggled, training will be resumed from the provided model weights. If false, model weights will be loaded but training will start from iteration 1')
    parser.add_argument('--gnn_layer_type', type=str, default="gat", choices=["gat", "res_gated"], help='type of gnn layer to use')
    add_bool_arg(parser=parser, name='hierarchical_action', default=True, help='if toggled, hierarchical action space will be used (only for gnn agent)')
    add_bool_arg(parser=parser, name='shared_gnn', default=False, help='if toggled, the same gnn will be used for actor and value features (only for gnn agent)')
    parser.add_argument('--target_mask', type=str, default="all", choices=["all", "enemy", "not_self", "not_neutral"], help='the target mask to use for the actions')
    add_bool_arg(parser=parser, name='use_global_features_ratio', default=True, help='if toggled, global features from the GNN will be concatenated to the local features in the ratio action head')
    parser.add_argument('--noop_penalty', type=float, default=0.05, help='penalty for doing nothing action when there are possible actions')
    parser.add_argument('--reward_type', type=str, default="score_delta", choices=["score_delta", "ship_delta", "adversarial"], help='type of reward to use')
    add_bool_arg(parser=parser, name='normalize_features', default=False, help='if toggled, features will be normalized using running mean and std normalization (only for gnn agent)')

    # Opponent configuration
    parser.add_argument('--opponent_type', type=str, default="passive", help='type of opponent to train against')
    parser.add_argument('--curriculum_opponents', nargs='*', type=str, default=['passive', 'random', 'careful_random', 'greedy', 'better_greedy', 'galactic'], help='list of (opponent_type, win_rate_threshold) tuples for curriculum learning')
    parser.add_argument('--opponent_baselines', nargs='*', type=str, default=['better_greedy', 'galactic'], help='list of baseline opponents to use for self-play.')
    parser.add_argument('--self_play', type=str, default=None, choices=["naive", "buffer", "baseline_buffer"], help='self-play strategy to use, if applicable')
    parser.add_argument('--fixed_weight_opponent_type', type=str, default= 'gnn', help='agetn type of fixed weight opponent')
    parser.add_argument('--fixed_weight_opponent_weights', type=str, default='models/adv_cont_2__1783426827_final.pt', help='file of model weights for fixed weight opponent')
    parser.add_argument('--buffer_opponents', nargs='*', default=[], help='list of opponents to use for buffer')
    parser.add_argument('--opponent_device', type=str, default='cuda', help='device to load opponent models onto')

    # To be filled in runtime
    parser.add_argument('--batch_size', type=int, default=0, help='the batch size (computed in runtime)')
    parser.add_argument('--minibatch_size', type=int, default=0, help='the mini-batch size (computed in runtime)')
    parser.add_argument('--num_iterations', type=int, default=0, help='the number of iterations (computed in runtime)')
    parser.add_argument('--run_name', type=str, default='temp', help='the name of the run (computed in runtime)')

    args = parser.parse_args()
    return args

def adversarial_classifier_args():
    parser = argparse.ArgumentParser(description="Adversarial Policy Classifier Training Configuration CLI")

    #Experiment / General
    parser.add_argument('--exp_name', type=str, default='classifier', help='name used for naming models and file')
    parser.add_argument('--save_dir', type=str, default='models/classifier', help='dir models are saved to')
    parser.add_argument('--seed', type = int, default = 874, help='seed used random during experiment')
    parser.add_argument('--device', type=str, default='cuda', help='the device passed to torch')
    add_bool_arg(parser=parser, name='track', default=True, help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='number of training epochs beofre checkpoint')
    
    #Training
    parser.add_argument('--num_epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=127, help='batch size used during training')
    parser.add_argument('--learning_rate', type = float, default= 0.001, help='learning rate')
    parser.add_argument('--dataset_size', type=int, default = 1000000, help='the number of datapoitns used for train and test')
    parser.add_argument('--train_split', type=float, default=0.8, help='size of train split in train/test split')
    parser.add_argument('--train_dir', type=str, default='saved_games/train', help='dir of train set data')
    parser.add_argument('--test_dir', type=str, default='saved_games/test', help='dir of test set data')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=100, help='number of training epochs before eval epoch')

    #Model
    parser.add_argument('--node_feature_dim', type=int, default=0, help='dimension of node features (owner, ship_count, growth_rate, x, y)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden dimenison in the classifier model')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "muon"], help="the optimizer to use: 'adam' or 'muon'")
    parser.add_argument('--model_weights', type=str, default=None, help='fiel apth of model weights when using pretrained model')
    args = parser.parse_args()
    return args


def local_battle_args():
    parser = argparse.ArgumentParser(description="Loacl Battle ConfigCLI")

    # Experiment / General
    parser.add_argument('--exp_name', type=str, default='render_game', help='the name of this experiment')
    parser.add_argument('--seed', type=int, default=874, help='seed of the experiment')
    add_bool_arg(parser=parser, name='torch_deterministic', default=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    add_bool_arg(parser=parser, name='cuda', default=True, help='if toggled, cuda will be enabled by default')
    add_bool_arg(parser=parser, name='render', default=True, help='Whether to render the game using pygame.')
    add_bool_arg(parser=parser, name='save_obs', default=False, help='Whether to store agents observaiton')
    parser.add_argument('--save_dir', default='saved_games', help='Where obs are stored.')
    parser.add_argument('--label', type = int, default = None, choices=[0,1], help= 'label used when creating dataset.')
    parser.add_argument('--samples_per_game', type=int, default=None, help='number of samples saved per game.')
    parser.add_argument('--device', type=str, default='cuda', help='the device passed to torch')

    # Planet Wars Env
    parser.add_argument('--num_games', type=int, default=1, help="Number of games played.")
    parser.add_argument('--num_steps', type=int, default=2000, help='max number of stpes a game can last for')
    parser.add_argument('--num_planets', type=int, default=None, help='number of planets in the game. If None, will be set to a random value between num_planets_min and num_planets_max (new_map_each_run needs to be set to true)')
    parser.add_argument('--num_planets_min', type=int, default=10, help='minimum number of planets in the game')
    parser.add_argument('--num_planets_max', type=int, default=30, help='maximum number of planets in the game')
    parser.add_argument('--node_feature_dim', type=int, default=0, help='dimension of node features (owner, ship_count, growth_rate, x, y)')
    parser.add_argument('--max_ticks', type=int, default=2000, help='maximum game ticks')
    add_bool_arg(parser=parser, name='use_adjacency_matrix', default=False, help='whether to include adjacency matrix in observations')
    add_bool_arg(parser=parser, name='flatten_observation', default=True, help='Filled on run time, mlp uses flattened observation, gnn uses graph observation')
    parser.add_argument('--discretized_ratio_bins', type=int, default=0, help='number of bins for the discretized ratio actor. Set to 0 to disable discretization')
    add_bool_arg(parser=parser, name='discretize_include_zero', default=False, help='whether to include zero in the discretized ratio bins')
    add_bool_arg(parser=parser, name='new_map_each_run', default=True, help='whether to create a new map for each run or use the same map')
    parser.add_argument('--maps_pkl', type=str, default=None, help='file path to .pkl file containing maps to be used.')
    parser.add_argument('--profile_path', type=str, default=None, help='Path to save profiling data, if None profiling is disabled')
    add_bool_arg(parser=parser, name='use_async', default=True, help='if toggled, AsyncVectorEnv will be used')
    add_bool_arg(parser=parser, name='use_tick', default=False, help='if toggled, the game tick will be passed as an observation')
    parser.add_argument('--reward_type', type=str, default="score_delta", choices=["score_delta", "ship_delta"], help='type of reward to use')
    parser.add_argument('--noop_penalty', type=float, default=0.0, help='penalty for doing nothing action when there are possible actions')

    #Agent1
    parser.add_argument('--agent_name', type=str, default='torchGNN', help='Name used for print statements and file naming.')
    parser.add_argument('--agent_type', type=str, default="gnn", choices=["mlp", "edge_mlp", "gnn", "edge_gnn"], help='the type of agent to train')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension for the layers')
    parser.add_argument('--model_weights', type=str, default=None, help='path to model weights to load (for continuing training)')
    parser.add_argument('--gnn_layer_type', type=str, default="gat", choices=["gat", "res_gated"], help='type of gnn layer to use')
    add_bool_arg(parser=parser, name='hierarchical_action', default=True, help='if toggled, hierarchical action space will be used (only for gnn agent)')
    add_bool_arg(parser=parser, name='shared_gnn', default=False, help='if toggled, the same gnn will be used for actor and value features (only for gnn agent)')
    parser.add_argument('--target_mask', type=str, default="all", choices=["all", "enemy", "not_self", "not_neutral"], help='the target mask to use for the actions')
    add_bool_arg(parser=parser, name='use_global_features_ratio', default=True, help='if toggled, global features from the GNN will be concatenated to the local features in the ratio action head')
    add_bool_arg(parser=parser, name='normalize_features', default=False, help='if toggled, features will be normalized using running mean and std normalization (only for gnn agent)')

    #Agent2
    parser.add_argument('--agent_name2', type=str, default='fixed_weight', help='Name used for print statements and file naming.')
    parser.add_argument('--agent_type2', type=str, default="gnn", help='the type of agent to train')
    parser.add_argument('--hidden_dim2', type=int, default=128, help='hidden dimension for the layers')
    parser.add_argument('--model_weights2', type=str, default=None, help='path to model weights to load (for continuing training)')
    parser.add_argument('--gnn_layer_type2', type=str, default="gat", choices=["gat", "res_gated"], help='type of gnn layer to use')
    add_bool_arg(parser=parser, name='hierarchical_action2', default=True, help='if toggled, hierarchical action space will be used (only for gnn agent)')
    add_bool_arg(parser=parser, name='shared_gnn2', default=False, help='if toggled, the same gnn will be used for actor and value features (only for gnn agent)')
    parser.add_argument('--target_mask2', type=str, default="all", choices=["all", "enemy", "not_self", "not_neutral"], help='the target mask to use for the actions')
    add_bool_arg(parser=parser, name='use_global_features_ratio2', default=True, help='if toggled, global features from the GNN will be concatenated to the local features in the ratio action head')
    add_bool_arg(parser=parser, name='normalize_features2', default=False, help='if toggled, features will be normalized using running mean and std normalization (only for gnn agent)')

    args = parser.parse_args()
    return args