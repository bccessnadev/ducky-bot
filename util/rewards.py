import numpy as np # Import numpy, the python math library

from rlgym_sim.utils import RewardFunction # Import the base RewardFunction class
from rlgym_sim.utils.gamestates import GameState, PlayerData # Import game state stuff

from rlgym_sim.utils.common_values import CAR_MAX_SPEED, BALL_MAX_SPEED, CEILING_Z, SIDE_WALL_X, BACK_WALL_Y, \
    ORANGE_TEAM, GOAL_HEIGHT, ORANGE_GOAL_BACK, BLUE_GOAL_BACK, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER # Import needed common values

# Source: https://github.com/ZealanL/RLGym-PPO-Guide/blob/cef4469986d8147cc9071af2b561917790ac00ae/rewards.md

class SpeedTowardBallReward(RewardFunction):   
    
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0
        
class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:

        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0
        
class StrongTouchReward(RewardFunction):
    
    def __init__(self, min_speed_change=0, max_speed_change=BALL_MAX_SPEED):
        super().__init__()
        """
        :param min_speed_change: Minimum change in ball speed after a touch needed to get reward
        :param max_speed_change: Change in ball speed needed to get maximum reward
        """
        # Ensure min_value and max_value are not equal to avoid division by zero
        if min_speed_change == max_speed_change:
            raise ValueError("min_value and max_value must be different")
        
        self.min_speed_change = min_speed_change
        self.max_speed_change = max_speed_change

        self.prev_ball_velocity = np.array([0,0,0])
        
    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        self.prev_ball_velocity = np.array([0,0,0])
    
    # Get the reward for a player strongly influcing ball velocity with a touch
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:

        # If the ball has been touched since last update
        if player.ball_touched:
            # Calculate the speed change of the ball
            ball_velocity_delta = state.ball.linear_velocity - self.prev_ball_velocity
            ball_speed_delta = np.linalg.norm(ball_velocity_delta)
            
            # Normalize the speed change based on given min an max speeds
            reward = (ball_speed_delta - self.min_speed_change) / (self.max_speed_change - self.min_speed_change)
        else:
            reward = 0
            
        # Cache the current ball velocity for the next call
        self.prev_ball_velocity = state.ball.linear_velocity
        
        return reward
    
class AirTouchReward(RewardFunction):
    
    def __init__(self, max_time_in_air=1.75, step_time=8/120):
        """
        :param max_time_in_air: Time in air needed to get maximum reward
        """
        super().__init__()
        
        self.max_time_in_air = max_time_in_air
        self.step_time = step_time
        self.air_time = 0
        
    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        self.air_time = 0
    
    # Get the reward for the player touching a ball in the air
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # Track how long the player has been in the air
        if player.on_ground:
            self.air_time = 0
        else:
            self.air_time += self.step_time
            
        # If the ball has been touched since last update
        if player.ball_touched:
            # Reward the player based on how high the touch was or how long they have been in the air
            air_time_frac = min(self.air_time, self.max_time_in_air) / self.max_time_in_air
            height_frac = state.ball.position[2] / CEILING_Z
            reward = min(air_time_frac, height_frac)
        else:
            reward = 0
            
        return reward
    
class BallInCornerReward(RewardFunction):
    
    # Empty default constructor (required)
    def __init__(self, corner_radius=2500):
        """
        :param corner_radius: Max distance from corner until concidered no longer within corner
        """
        super().__init__()
        
        self.corner_radius=corner_radius
        
    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets
    
    # Get reward based on how close ball is to a corner. Positive reward for defence, negative for offence
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        defence_corners = np.array([[SIDE_WALL_X, BACK_WALL_Y], [-SIDE_WALL_X, BACK_WALL_Y]])
        offence_corners = np.array([[SIDE_WALL_X, -BACK_WALL_Y], [-SIDE_WALL_X, -BACK_WALL_Y]])
        
        if player.team_num == ORANGE_TEAM:
            defence_corners, offence_corners = offence_corners, defence_corners
            
        ball_pos = state.ball.position[:2]  # Use only the x, y coordinates of the ball
        
        reward = 0
            
        for corner in defence_corners:
            ball_corner_dist = np.linalg.norm(corner - ball_pos)            
            if ball_corner_dist < self.corner_radius:
                reward = 1 - (ball_corner_dist / self.corner_radius) # Reward for ball being closer to the defence corner
        
        for corner in offence_corners:
            ball_corner_dist = np.linalg.norm(corner - ball_pos)            
            if ball_corner_dist < self.corner_radius:
                reward = (ball_corner_dist / self.corner_radius) - 1 # Punish for ball being closer to the defence corner
        
        return reward

class AerialVelocityReward(RewardFunction):
    
    def __init__(self, aerial_min_height=GOAL_HEIGHT, velocity_dir_threshold=0.9):
        """
        :param aerial_min_height: Height the ball needs to be above to be concidered aerial
        :param velocity_dir_threshold: Threshold of car velocity to ball dot product that must be exceeded to get the reward
        """
        super().__init__()
        
        self.aerial_min_height = aerial_min_height
        self.velocity_dir_threshold = velocity_dir_threshold
        
    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets
    
    # Get reward for when player has upwards velocity when the ball is in the air
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        reward = 0

        if state.ball.position[2] > self.aerial_min_height and not player.on_ground:
            # Check if player is facing ball
            car_to_ball = state.ball.position - player.car_data.position
            car_to_ball_dir = car_to_ball / np.linalg.norm(car_to_ball)
            car_speed = np.linalg.norm(player.car_data.linear_velocity)
            if car_speed != 0: # Prevent divide by 0
                car_velocity_dir = player.car_data.linear_velocity / car_speed
                car_velocity_to_ball_dot = np.dot(car_to_ball_dir, car_velocity_dir)
                if car_velocity_to_ball_dot > self.velocity_dir_threshold:
                    reward = car_velocity_to_ball_dot
            
        return reward
    
class AdvancedVelocityBallToGoalReward(RewardFunction):
    
    # Empty default constructor (required)
    def __init__(self, aim_at_front_corner_size=(2000,1000)):
        """
        :param aim_at_front_dist: While within this distance from the goal, aim at the front instead of the back to avoid hitting side wall
        """
        super().__init__()
        
        self.aim_at_front_corner_size = aim_at_front_corner_size
        
    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets
    
    # Get reward for when player has upwards velocity when the ball is in the air
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:         
        near_side_wall = np.fabs(player.car_data.position[0]) > self.aim_at_front_corner_size[0]
        
        # Determine the objective location. If in the corner near the objective, aim for the front instead of the back
        if player.team_num == ORANGE_TEAM:
            if near_side_wall and player.car_data.position[1] < (-BACK_WALL_Y + self.aim_at_front_corner_size[1]):
                objective = np.array(BLUE_GOAL_CENTER)
            else:
                objective = np.array(BLUE_GOAL_BACK)             
        else:
             if near_side_wall and player.car_data.position[1] > (BACK_WALL_Y - self.aim_at_front_corner_size[1]):
                objective = np.array(ORANGE_GOAL_CENTER)
             else:
                objective = np.array(ORANGE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED
        
        # Only reward for ball going towarrds goal, don't pushing for going away
        reward = max(float(np.dot(norm_pos_diff, norm_vel)), 0)

        return reward
    
class BallTowardsOwnGoalPunish(RewardFunction):
    
    # Empty default constructor (required)
    def __init__(self, direction_threshold=0.8, own_touch_multiplier=2):
        """
        :param direction_threshold: Punish is given if the dot product of ball velocity direction to goal is greater than this
        :param own_touch_multiplier: Punish is multiplied by this if player touched ball last  
        """
        super().__init__()
        
        self.direction_threshold = direction_threshold
        self.own_touch_multiplier = own_touch_multiplier
        
    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets
    
    # Get reward for when player has upwards velocity when the ball is in the air
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float: 
        punish = 0
        
        if player.team_num == ORANGE_TEAM:
            own_goal_pos = ORANGE_GOAL_BACK
        else:
            own_goal_pos = BLUE_GOAL_BACK
        
        ball_speed = np.linalg.norm(state.ball.linear_velocity)
        if ball_speed != 0: # Prevent divide by 0
            ball_velocity_dir = state.ball.linear_velocity / ball_speed
            ball_to_goal = own_goal_pos - state.ball.position
            ball_to_goal_dir = ball_to_goal / np.linalg.norm(ball_to_goal)
        
            velocity_goal_dot = np.dot(ball_to_goal_dir, ball_velocity_dir)
            if velocity_goal_dot > self.direction_threshold:
                punish = -velocity_goal_dot
                if state.last_touch == player.car_id:           
                    punish *= self.own_touch_multiplier

        return punish
            
        

