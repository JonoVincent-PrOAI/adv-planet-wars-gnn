import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import shutil
import time
from dataclasses import dataclass
import sys
from itertools import chain
import glob
if not "../" in sys.path:
    sys.path.append('../')
if not "./" in sys.path:
    sys.path.append('./')
import gymnasium as gym
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.utils import to_dense_batch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from typing import List
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from gym_utils.gym_wrapper import PlanetWarsForwardModelEnv, PlanetWarsForwardModelGNNEnv
from core.game_state import Player, GameParams
from agents.mlp import PlanetWarsAgentMLP
from agents.edge_mlp import PlanetWarsAgentEdgeMLP
from agents.edge_gnn import PlanetWarsAgentEdgeGNN
from agents.gnn import PlanetWarsAgentGNN, GraphInstanceToPyG
from agents.passive_agent import PassiveAgent
from agents.baseline_policies import GreedyPolicy,RandomPolicy, FocusPolicy, DefensivePolicy
from agents.GalacticArmada import GalacticArmada
from agents.random_agents import CarefulRandomAgent, PureRandomAgent
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.better_greedy_heuristic_agent import BetterGreedyHeuristicAgent
from agents.aggressive_greedy_heuristic_agent import AggressiveGreedyHeuristicAgent
from agents.torch_agent_gnn import TorchAgentGNN
from gym_utils.self_play import get_self_play_class
from gym_utils.gnn_utils import preprocess_graph_data, owner_one_hot_encoding, collate_source_mask, preprocess_graph_data_unbatched
from config_files.ppo_config import Args
from gym_utils.graph_normalize_wrapper import NormalizeGraphObservation, NormalizeMLPObservation

import math
import pygame
import json
from config_files.arguments import local_battle_args

def get_opponent_from_string(opponent_tye):
    if opponent_tye == "better_greedy":
        return BetterGreedyHeuristicAgent()
    elif opponent_tye == "galactic":
        return GalacticArmada()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_tye}")

def make_env(env_id, idx, capture_video, run_name, device, args, self_play=None):

    if args.num_planets is None:
        num_planets = np.random.randint(args.num_planets_min, args.num_planets_max + 1)
    else:
        num_planets = args.num_planets
    game_params = {
                'numPlanets': num_planets,
                'maxTicks': args.max_ticks,
                'transporterSpeed': np.random.uniform(2.0, 5.0),
                'width': 640,
                'height': 480,
                'newMapEachRun': args.new_map_each_run,
                'minGrowthRate': 0.05,
                'maxGrowthRate': 0.2,
                'initialNeutralRatio': np.random.uniform(0.25, 0.35)
            }

    if env_id == "PlanetWarsForwardModel":
        env = PlanetWarsForwardModelEnv(
            args,
            controlled_player=Player.Player1,
            opponent_player=Player.Player2,
            max_ticks=args.max_ticks,
            game_params=game_params,
            self_play= None,
        )
    elif env_id == "PlanetWarsForwardModelGNN":

        env = PlanetWarsForwardModelGNNEnv(
            args,
            controlled_player=Player.Player1,
            opponent_player=Player.Player2,
            max_ticks=args.max_ticks,
            game_params=game_params,
            self_play= None
        )
    
    if args.agent_type2 == "greedy":
        env.set_opponent_policy(GreedyPolicy(game_params=env.game_params, player=Player.Player2))
    elif args.agent_type2 == "passive":
        opponent = PassiveAgent()
        opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
        env.set_opponent_policy(opponent)
    elif args.agent_type2 == "random":
        env.set_opponent_policy(RandomPolicy(game_params=env.game_params, player=Player.Player2))
    elif args.agent_type2 == "focus":
        env.set_opponent_policy(FocusPolicy(game_params=env.game_params, player=Player.Player2))
    elif args.agent_type2 == "defensive":
        env.set_opponent_policy(DefensivePolicy(game_params=env.game_params, player=Player.Player2))
    elif args.agent_type2 == "careful_random":
        opponent = CarefulRandomAgent()
        opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
        env.set_opponent_policy(opponent)
    elif args.agent_type2 == "better_greedy":
        opponent = BetterGreedyHeuristicAgent()
        opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
        env.set_opponent_policy(opponent)
    elif args.agent_type2 == "aggressive_greedy":
        opponent = AggressiveGreedyHeuristicAgent()
        opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
        env.set_opponent_policy(opponent)
    elif args.agent_type2 == "galactic":
        opponent = GalacticArmada()
        opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
        env.set_opponent_policy(opponent)
    elif args.agent_type2 == "fixed_weight" :
        
        if args.fixed_weight_opponent['agent_type'] == "gnn":
            opponent_agent = PlanetWarsAgentGNN(args).to(args.opponent_device)
            opponent_agent = torch.compile(opponent_agent, dynamic=True)

        elif args.fixed_weight_opponent['agent_type'] == "edge_mlp":
            opponent_agent = PlanetWarsAgentEdgeMLP(args).to(args.opponent_device)
            opponent_agent = torch.compile(opponent_agent)

        elif args.fixed_weight_opponent['agent_type'] == "mlp":
            opponent_agent = PlanetWarsAgentMLP(args).to(args.opponent_device)
            opponent_agent = torch.compile(opponent_agent)

        elif args.fixed_weight_opponent['agent_type'] == "edge_gnn":
            opponent_agent = PlanetWarsAgentEdgeGNN(args).to(args.opponent_device)
            opponent_agent = torch.compile(opponent_agent, dynamic=True)
        
        opponent_model_weights = args.fixed_weight_opponent['model_weights']
        state_dict = torch.load(opponent_model_weights, map_location=torch.device(args.opponent_device), weights_only=False)
        opponent_agent.load_state_dict(state_dict['model_state_dict'])
        opponent = TorchAgentGNN(model=opponent_agent.copy_as_opponent(), device=args.opponent_device)

    env = PlanetWarsActionWrapper(env, num_planets, args.use_adjacency_matrix, args.flatten_observation, device, node_feature_dim=args.node_feature_dim)
    # env = NormalizeGraphObservation(env, node_feature_dim=args.node_feature_dim, edge_feature_dim=5)
    return env



def make_vector_env(env_id, capture_video, run_name, device, args, self_play=None):
    if args.use_async and "gnn" in args.agent_type:
        envs = gym.vector.AsyncVectorEnv([make_env(env_id, i, capture_video, run_name, device, args, self_play=self_play) for i in range(args.num_envs)], shared_memory=False)
        if args.normalize_features:
            envs = NormalizeGraphObservation(envs, clip=10.0, training=True)
        return envs
    elif (not args.use_async) and "gnn" in args.agent_type:
        envs = gym.vector.SyncVectorEnv([make_env(env_id, i, capture_video, run_name, device, args, self_play=self_play) for i in range(args.num_envs)])
        if args.normalize_features:
            envs = NormalizeGraphObservation(envs, clip=10.0, training=True)
        return envs
    elif "gnn" not in args.agent_type:
        envs = gym.vector.AsyncVectorEnv([make_env(env_id, i, capture_video, run_name, device, args, self_play=self_play) for i in range(args.num_envs)])
        if args.normalize_features:
            envs = NormalizeMLPObservation(envs, clip=10.0, training=True)
        return envs
    
class PlanetWarsActionWrapper(gym.Wrapper):
    """Wrapper to flatten the tuple action space for Planet Wars"""

    def __init__(self, env, num_planets, use_adjacency_matrix=False, flatten_observation=False, device='cpu', node_feature_dim=0):
        super().__init__(env)
        self.num_planets = num_planets
        self.use_adjacency_matrix = use_adjacency_matrix
        self.flatten_observation = flatten_observation
        self.device = device
        self.node_feature_dim = node_feature_dim

        # Action space: source_planet (discrete) + target_planet (discrete) + ship_ratio (continuous)
        self.action_space = gym.spaces.Box(
                low=np.array([0, 0, 0.0]),
                high=np.array([args.num_planets_max-1, args.num_planets_max-1, 1.0]),
                dtype=np.float32
            )
        if flatten_observation:
            # Calculate observation space size based on whether adjacency matrix is included
            # obs_size = num_planets * self.node_feature_dim  # node_features
            # if use_adjacency_matrix:
            #     obs_size += num_planets * num_planets  # adjacency_matrix
        
            # Flatten observation space from Dict to Box
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(30,self.node_feature_dim),
                dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Graph(node_space=gym.spaces.Box(low=0, high=1000, shape=(14,), dtype=np.float32),
                                                  edge_space=gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32))
    
    def step(self, action):
        # Convert flattened action back to tuple format
        source_planet = int(action[0])
        target_planet = int(action[1])
        ship_ratio = np.clip(action[2], 0.0, 1.0)
        
        tuple_action = (source_planet, target_planet, np.array([ship_ratio]))
        obs, reward, done, truncated, info = self.env.step(tuple_action)
        
        # Flatten observation
        if self.flatten_observation:
            obs = obs.x
            #Pad to max number of planets
            if obs.shape[0] < 30:
                padding = torch.zeros((30 - obs.shape[0], obs.shape[1]), dtype=torch.float32)
                obs = torch.cat((obs, padding), dim=0)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.flatten_observation:
            obs = obs.x
            #Pad to max number of planets
            if obs.shape[0] < 30:
                padding = torch.zeros((30 - obs.shape[0], obs.shape[1]), dtype=torch.float32)
                obs = torch.cat((obs, padding), dim=0)
        return obs, info

def render_game_scene(surface, game_state):
    """
    Renders the current game state.
    """
    if not hasattr(render_game_scene, "initialized"):
        pygame.font.init()

        # to guarantee font compilation across all Pygame versions.
        font_choices = "arial,helvetica,dejavusans,liberationsans,freesans"
        render_game_scene.font_planet = pygame.font.SysFont(font_choices, 18, bold=True)
        render_game_scene.font_transport = pygame.font.SysFont(font_choices, 13, bold=True)
        render_game_scene.font_game_over = pygame.font.SysFont(font_choices, 60, bold=True)
        render_game_scene.font_status = pygame.font.SysFont(font_choices, 20)
        
        render_game_scene.text_cache = {}

        # Initialize Star Background
        screen_size = surface.get_size()
        render_game_scene.star_background = pygame.Surface(screen_size)
        render_game_scene.star_background.fill((15, 15, 20))

        for _ in range(200):
            x = random.randint(0, screen_size[0] - 1)
            y = random.randint(0, screen_size[1] - 1)
            radius = random.choice([1, 1, 1, 2])
            brightness = random.randint(180, 255)
            color_variation = random.choice([(0,0,0), (0,0,20), (20,20,0), (20,20,20)])
            star_color = tuple(min(255, max(0, brightness + c)) for c in color_variation)
            pygame.draw.circle(render_game_scene.star_background, star_color, (x, y), radius)

        render_game_scene.initialized = True

    # Draw Background
    surface.blit(render_game_scene.star_background, (0, 0))

    COLORS = {0: (128, 128, 128), 1: (0, 100, 255), 2: (255, 50, 50)}
    TEXT_COLOR = (255, 255, 255)
    
    # Draw Planets & Ship Counts
    for planet in game_state.get('planets', []):
        owner = planet.get('owner', 0)
        color = COLORS.get(owner, (128, 128, 128))
        pos = (int(planet['x']), int(planet['y']))
        radius = int(planet['radius'])
        
        pygame.draw.circle(surface, color, pos, radius)
        pygame.draw.circle(surface, (255, 255, 255), pos, radius, 1)
        
        ship_count = int(planet.get('numShips', planet.get('num_ships', planet.get('ships', planet.get('ship_count', 0)))))
        
        if ship_count not in render_game_scene.text_cache:
            text_surf = render_game_scene.font_planet.render(str(ship_count), True, TEXT_COLOR)
            render_game_scene.text_cache[ship_count] = text_surf
        else:
            text_surf = render_game_scene.text_cache[ship_count]
            
        text_rect = text_surf.get_rect(center=pos)
        surface.blit(text_surf, text_rect)

    # Draw Transports & Ship Counts
    for transport in game_state.get('transporters', []):
        owner = transport.get('owner', 0)
        color = COLORS.get(owner, (128, 128, 128))
        x, y = transport['x'], transport['y']
        vx, vy = transport.get('vx', 0), transport.get('vy', 0)
        
        size = 8  
        speed = math.hypot(vx, vy)
        if speed == 0:
            p1 = (x, y - size)
            p2 = (x - size, y + size)
            p3 = (x + size, y + size)
        else:
            dx, dy = vx / speed, vy / speed
            perpx, perpy = -dy, dx
            p1 = (x + dx * size, y + dy * size)
            p2 = (x - dx * size + perpx * (size * 0.7), y - dy * size + perpy * (size * 0.7))
            p3 = (x - dx * size - perpx * (size * 0.7), y - dy * size - perpy * (size * 0.7))
            
        pygame.draw.polygon(surface, color, [p1, p2, p3])
        pygame.draw.polygon(surface, (255, 255, 255), [p1, p2, p3], 1)

        ship_count = int(transport.get('numShips', transport.get('num_ships', transport.get('ships', transport.get('ship_count', 0)))))
        transport_key = f"t_{ship_count}"
        
        if transport_key not in render_game_scene.text_cache:
            text_surf = render_game_scene.font_transport.render(str(ship_count), True, TEXT_COLOR)
            render_game_scene.text_cache[transport_key] = text_surf
        else:
            text_surf = render_game_scene.text_cache[transport_key]
        
        text_rect = text_surf.get_rect(center=(int(x), int(y) - 14))
        surface.blit(text_surf, text_rect)

    # Game Over Overlay
    if game_state.get('isTerminal', False) or game_state.get('game_over', False):
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200)) 
        surface.blit(overlay, (0, 0))
        
        go_surf = render_game_scene.font_game_over.render("GAME OVER", True, (255, 75, 75))
        go_rect = go_surf.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2 - 25))
        surface.blit(go_surf, go_rect)
        if game_state.get('leader') == env.unwrapped.player_int:
            if ("gnn" in args.agent_type or "mlp" in args.agent_type) and args.model_weights != None:
                status_text = (args.model_weights + " Won")
            else:
                status_text = (args.agent_type + " Won")
        else:
            if args.agent_type2 == "fixed_weight":
                status_text = (args.fixed_weight_opponent['model_weights'] + " Won")
            else:
                status_text = (args.agent_type2 + " Won")
        status_surf = render_game_scene.font_status.render(status_text, True, (220, 220, 220))
        status_rect = status_surf.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2 + 25))
        surface.blit(status_surf, status_rect)
    
if __name__ == "__main__":

    args = local_battle_args()
    args.flatten_observation = "gnn" not in args.agent_type
    args.env_id = "PlanetWarsForwardModelGNN" if "gnn" in args.agent_type else "PlanetWarsForwardModel"
    args.node_feature_dim = 5 if "gnn" in args.agent_type else 9
    if args.use_tick:
        args.node_feature_dim += 1  # Add tick feature if use_tick is enabled

    if args.use_async:
        import multiprocessing as mp
        mp.set_start_method("spawn")


    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.agent_type == "gnn":
        agent = PlanetWarsAgentGNN(args).to(device)
        agent = torch.compile(agent, dynamic=True)
    elif args.agent_type == "edge_mlp":
        agent = PlanetWarsAgentEdgeMLP(args).to(device)
        agent = torch.compile(agent)
    elif args.agent_type == "mlp":
        agent = PlanetWarsAgentMLP(args).to(device)
        agent = torch.compile(agent)
    elif args.agent_type == "edge_gnn":
        agent = PlanetWarsAgentEdgeGNN(args).to(device)
        agent = torch.compile(agent, dynamic=True)

    if args.model_weights is not None:
        state_dict = torch.load(args.model_weights, map_location=torch.device(device), weights_only=False)
        agent.load_state_dict(state_dict['model_state_dict'])

    env = make_env(
        env_id=args.env_id,
        idx=0,
        capture_video=False,
        run_name='local_game',
        device=device,
        args=args,
        self_play=None,
    )

    if args.render:
        pygame.init()
        
        # Define window dimensions based on your coordinates
        WINDOW_WIDTH = 700
        WINDOW_HEIGHT = 500
        
        # Create the main display surface
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Planet Wars")
        clock = pygame.time.Clock()

    next_obs, _ = env.reset(seed=args.seed)
    if args.flatten_observation:
        next_obs = torch.Tensor(next_obs).to(device)

    running = True
    game_over = False
    games_played = 0

    print("Starting visual environment rollout...")
    game_records = {'meta_data': {'agent_1': args.agent_name, 
                                  'agent_2':args.agent_name2,
                                  'seed': [], 
                                  'num_planets': [],
                                  'winner':[]},

                    'game_data': {'obs':[], 
                                  'labels': []}
                    }
    while running :

        if args.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

        if games_played < args.num_games:

            next_obs, infos = env.reset()
            num_planets = env.unwrapped.num_planets
            step = 0
            game_over = False
            all_obs = []
            all_labels = []
            while (step < args.num_steps) and not game_over and running:
                #TODO doesn't work for PO env
                # obs_dict = {
                #     'x': next_obs.x.detach().cpu().numpy().tolist(),
                #     'edge_index': next_obs.edge_index.detach().cpu().numpy().tolist(),
                #     'edge_attr': next_obs.edge_attr.detach().cpu().numpy().tolist(),
                #     'source_mask': next_obs.source_mask.detach().cpu().numpy().tolist(),
                #     'tick': float(next_obs.tick) if isinstance(next_obs.tick, torch.Tensor) else next_obs.tick
                # }
                all_obs.append(next_obs)
                #FIX labels not implemented
                all_labels.append('test')

                if args.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                
                step += 1
                with torch.no_grad():
                    if args.flatten_observation:
                        action, logprob, _, value = agent.get_action_and_value(next_obs)
                    else:
                        input = next_obs
                        source_mask = to_dense_batch(input.source_mask, input.batch, fill_value=False)[0]
                        source_mask = torch.cat((torch.ones(source_mask.size(0), 1, dtype=torch.bool, device=source_mask.device), source_mask), dim=1)
                        action, logprob, _, value = agent.get_action_and_value(input.to(device), source_mask=source_mask.to(device))
                    
                next_obs, reward, terminations, truncations, infos = env.step(action[0].cpu().numpy())

                game_state = env.unwrapped.current_game_state

                if args.render:
                    render_game_scene(screen, game_state)
                    pygame.display.flip()
                    clock.tick(60)
                
                if terminations or truncations:
                    game_over = True
                    games_played +=1

                    game_records['game_data']['obs'].append(all_obs)
                    game_records['game_data']['labels'].append(all_labels)

                    game_records['meta_data']['winner'].append(game_state.get('leader'))
                    game_records['meta_data']['seed'].append(args.seed)
                    game_records['meta_data']['num_planets'].append(num_planets)

                    if args.render:
                        for i in range(100):
                            game_state['isTerminal'] = True
                            game_state['game_over'] = True
                            render_game_scene(screen, game_state)
                            pygame.display.flip()
                            clock.tick(60)
        else:
            running = False  
            

    if args.save_obs:
        i = 0
        game_path = args.save_dir + '/' + args.agent_name + '_vs_' + args.agent_name2 + '/games.pt'
        meta_path = args.save_dir + '/' + args.agent_name + '_vs_' + args.agent_name2 + '/meta_data.json'

        if os.path.isfile(game_path):

            game_file = torch.load(game_path, weights_only=False)
            game_file['obs'] = game_file['obs'] + game_records['game_data']['obs']
            game_file['labels'] = game_file['labels'] + game_records['game_data']['labels']
            torch.save(game_file, game_path)
        else:
            os.makedirs(os.path.dirname(game_path), exist_ok=True)
            torch.save(game_records['game_data'], game_path)

        if os.path.isfile(meta_path):
            
            with open(meta_path, mode = 'r') as f:
                meta_file = json.load(f)
                meta_file['winner'] = meta_file['winner'] + game_records['meta_data']['winner']
                meta_file['seed'] = meta_file['seed'] + game_records['meta_data']['seed']
                meta_file['num_planets'] = meta_file['num_planets'] + game_records['meta_data']['num_planets']
            
            with open(meta_path, mode='w') as f:
                f.write(json.dumps(meta_file))
        else:
            with open(meta_path, mode='w') as f:
                f.write(json.dumps(game_records['meta_data']))

    print("Match finished! Close the window to exit the script.")
    pygame.quit()