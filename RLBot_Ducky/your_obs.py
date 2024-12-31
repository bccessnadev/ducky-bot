import imp
import math
import numpy as np
from typing import Any, List

from util.game_state import GameState
from util.physics_object import PhysicsObject
from util.player_data import PlayerData

from util import common_values

# IMPORTANT NOTICE:
# In some cases, the right row of player rotmats may actually be the left
# If your bot seems broken for no reason, try flipping right/left of rotmat

class YourOBS:
    def __init__(self, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi):
        """
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        """
        super().__init__()
        
        # self.POS_COEF = pos_coef
        # self.ANG_COEF = ang_coef
        # self.LIN_VEL_COEF = lin_vel_coef
        # self.ANG_VEL_COEF = ang_vel_coef
        
        # Use same calcualted values as training
        self.POS_COEF = np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z])
        self.ANG_COEF = 1 / np.pi
        self.LIN_VEL_COEF = 1 / common_values.CAR_MAX_SPEED
        self.ANG_VEL_COEF = 1 / common_values.CAR_MAX_ANG_VEL

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action,
               pads]

        self._add_player_to_obs(obs, player, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, inverted)

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car