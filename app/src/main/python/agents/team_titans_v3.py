import random
from typing import Optional
import numpy as np
from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, GameParams
from core.game_state_factory import GameStateFactory
from core.forward_model import ForwardModel
import math
import copy

class TeamTitan(PlanetWarsPlayer):

    def __init__(self):
        self.SIMULATED_TICKS = 50 #number of ticks simulated before state eval
        super().__init__()

    def get_agent_type(self):
        return 'team_titans'

    def get_action(self, game_state: GameState) -> Action:
        best_action = Action.do_nothing()
        best_value = -9999999999
        src_planets = [p for p in game_state.planets
                if p.owner == self.player and p.transporter is None and p.n_ships > 0]
        all_planets = game_state.planets

        for src_planet in src_planets:
            if src_planet.n_ships >= 1:
                for dest_planet in all_planets:
                    num_ships_options = [src_planet.n_ships, int(src_planet.n_ships*0.5), int(src_planet.n_ships*0.25)]
                    for n_ships in num_ships_options:
                        value = self.eval_action(game_state, src_planet, dest_planet, n_ships)
                        if value > best_value:
                            best_action = Action(            
                                player_id = self.player, 
                                source_planet_id = src_planet.id, 
                                destination_planet_id = dest_planet.id, 
                                num_ships = n_ships
                            )
                            best_value = value
        return(best_action)
    
    def eval_action(self, game_state, src_planet, dest_planet, n_ships):

        action = Action(
            player_id = self.player, 
            source_planet_id = src_planet.id, 
            destination_planet_id = dest_planet.id, 
            num_ships = n_ships
        )
        opp_action = Action.do_nothing()
        action_dict = {self.player : action, self.player.opponent: opp_action}
        game_state_copy = copy.deepcopy(game_state)
        forward_model = ForwardModel(game_state_copy, self.params)
        forward_model.step(action_dict)

        action_dict = {self.player : Action.do_nothing(), self.player.opponent: Action.do_nothing()}
        for _ in range(self.SIMULATED_TICKS):
            forward_model.step(action_dict)

        value = self.calculate_value(forward_model.state)
        return(value)

    def calculate_value(self, game_state):
        friendly_planets = [p for p in game_state.planets if p.owner == self.player]
        opp_planets = [p for p in game_state.planets if p.owner != self.player]

        nearest_enemy = []
        for p in friendly_planets:
            min_dist = 1000 #acts as cap on max penalty for distance
            for op in opp_planets:
                dx = p.position.x - op.position.x
                dy = p.position.y - op.position.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
            nearest_enemy.append(min_dist)

        total_ships = sum([ (p.n_ships + (0.0005*(p.n_ships/d))) for p,d in zip(friendly_planets, nearest_enemy)]) #weights ships based on closeness to enemy
        total_growth = sum([p.growth_rate for p in friendly_planets])

        total_opp_ships = sum([p.n_ships for p in opp_planets])
        total_opp_growth = sum([p.growth_rate for p in opp_planets])

        value = (total_ships  - total_opp_ships) + (30*(total_growth - total_opp_growth))

        #ensure win/loss states are represented in value
        if opp_planets == []:
            value += 10000000000
        if friendly_planets == []:
            value += -10000000000

        return(value)
