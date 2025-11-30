"""
OpenAI Gym wrapper for Pacman Capture-the-Flag environment
Allows training with Stable-Baselines3 and other RL libraries
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

from contest.game import Directions
import contest.util as util
import random


class PacmanCaptureEnv(gym.Env):
    """
    OpenAI Gym environment wrapper for Pacman Capture-the-Flag
    
    This wrapper allows you to train agents using standard RL libraries
    like Stable-Baselines3, RLlib, etc.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, agent_index: int, game_state=None, capture_agent=None, reset_func=None):
        """
        Args:
            agent_index: Index of the agent (0 or 2 for red team)
            game_state: Initial game state (will be set during reset)
            capture_agent: Reference to CaptureAgent for helper methods
            reset_func: Function to call to get a new game state on reset
        """
        super(PacmanCaptureEnv, self).__init__()
        
        self.agent_index = agent_index
        self.game_state = game_state
        self.capture_agent = capture_agent
        self.reset_func = reset_func
        self.last_score = 0
        self.steps = 0
        self.max_steps = 1200
        
        # Action space: 5 discrete actions (North, South, East, West, Stop)
        self.action_space = spaces.Discrete(5)
        
        # Observation space: Feature vector (continuous)
        # Features: [bias, food_dist, food_count, ghost_dist, carrying, home_dist, capsule_dist, score]
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(8,), 
            dtype=np.float32
        )
        
        # Action mapping
        self.action_to_direction = {
            0: Directions.NORTH,
            1: Directions.SOUTH,
            2: Directions.EAST,
            3: Directions.WEST,
            4: Directions.STOP
        }
    
    def _get_observation(self, game_state) -> np.ndarray:
        """
        Extract observation features from game state
        Returns normalized feature vector
        """
        if game_state is None or self.capture_agent is None:
            return np.zeros(8, dtype=np.float32)
        
        agent_state = game_state.get_agent_state(self.agent_index)
        agent_pos = agent_state.get_position()
        
        # Get map dimensions for normalization
        walls = game_state.get_walls()
        map_size = walls.width * walls.height
        
        features = np.zeros(8, dtype=np.float32)
        
        # Feature 0: Bias
        features[0] = 1.0
        
        # Feature 1: Distance to nearest food (normalized)
        food_list = self.capture_agent.get_food(game_state).as_list()
        if food_list:
            min_food_dist = min([self.capture_agent.get_maze_distance(agent_pos, food) for food in food_list])
            features[1] = float(min_food_dist) / map_size
        
        # Feature 2: Food remaining (normalized)
        features[2] = len(food_list) / 20.0
        
        # Feature 3: Ghost distance (normalized)
        enemies = [game_state.get_agent_state(i) for i in self.capture_agent.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        if ghosts:
            min_ghost_dist = min([self.capture_agent.get_maze_distance(agent_pos, g.get_position()) for g in ghosts])
            features[3] = float(min_ghost_dist) / map_size
        else:
            features[3] = 1.0  # No ghosts nearby
        
        # Feature 4: Carrying food (normalized)
        features[4] = agent_state.num_carrying / 10.0
        
        # Feature 5: Distance to home (normalized)
        if agent_state.num_carrying > 0:
            home_dist = self.capture_agent.get_maze_distance(agent_pos, self.capture_agent.start)
            features[5] = float(home_dist) / map_size
        
        # Feature 6: Capsule proximity (normalized)
        capsules = self.capture_agent.get_capsules(game_state)
        if capsules:
            min_cap_dist = min([self.capture_agent.get_maze_distance(agent_pos, cap) for cap in capsules])
            features[6] = float(min_cap_dist) / map_size
        
        # Feature 7: Score (normalized)
        features[7] = self.capture_agent.get_score(game_state) / 100.0
        
        return features
    
    def _get_reward(self, prev_state, curr_state) -> float:
        """
        Calculate reward for transition from prev_state to curr_state
        """
        if prev_state is None or curr_state is None or self.capture_agent is None:
            return 0.0
        
        reward = 0.0
        
        prev_agent = prev_state.get_agent_state(self.agent_index)
        curr_agent = curr_state.get_agent_state(self.agent_index)
        curr_pos = curr_agent.get_position()
        
        # Reward 1: Eating food (BIG reward!)
        prev_food_count = len(self.capture_agent.get_food(prev_state).as_list())
        curr_food_count = len(self.capture_agent.get_food(curr_state).as_list())
        food_eaten = prev_food_count - curr_food_count
        if food_eaten > 0:
            reward += 100.0 * food_eaten
        
        # Reward 2: Returning food (MASSIVE reward!)
        food_returned = curr_agent.num_returned - prev_agent.num_returned
        if food_returned > 0:
            reward += 200.0 * food_returned
        
        # Penalty 3: Getting caught
        if curr_agent.get_position() == self.capture_agent.start and prev_agent.is_pacman:
            if prev_agent.num_carrying > 0:
                reward -= 150.0
            else:
                reward -= 50.0
        
        # Reward 4: Moving toward food (distance shaping)
        if curr_agent.is_pacman and curr_agent.num_carrying < 5:
            prev_pos = prev_agent.get_position()
            food_list = self.capture_agent.get_food(curr_state).as_list()
            if food_list:
                prev_min_food_dist = min([self.capture_agent.get_maze_distance(prev_pos, food) for food in food_list])
                curr_min_food_dist = min([self.capture_agent.get_maze_distance(curr_pos, food) for food in food_list])
                distance_improvement = prev_min_food_dist - curr_min_food_dist
                reward += distance_improvement * 2.0
        
        # Penalty 5: Being near ghosts
        enemies = [curr_state.get_agent_state(i) for i in self.capture_agent.get_opponents(curr_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        if ghosts and curr_agent.is_pacman:
            min_ghost_dist = min([self.capture_agent.get_maze_distance(curr_pos, g.get_position()) for g in ghosts])
            if min_ghost_dist <= 5:
                reward -= (6 - min_ghost_dist) ** 2
        
        # Reward 6: Crossing to opponent side
        if curr_agent.is_pacman and not prev_agent.is_pacman:
            reward += 5.0
        
        # Reward 7: Returning home when loaded
        if curr_agent.num_carrying >= 3:
            home_dist = self.capture_agent.get_maze_distance(curr_pos, self.capture_agent.start)
            prev_home_dist = self.capture_agent.get_maze_distance(prev_agent.get_position(), self.capture_agent.start)
            if home_dist < prev_home_dist:
                reward += 3.0 * curr_agent.num_carrying
        
        # Small time penalty to encourage efficiency
        reward -= 0.1
        
        return reward
    
    def _get_terminal_reward(self, game_state) -> float:
        """
        Calculate terminal reward based on game outcome (win/loss/tie)
        Called when episode ends
        """
        if game_state is None or self.capture_agent is None:
            return 0.0
        
        final_score = self.capture_agent.get_score(game_state)
        
        if final_score > 0:
            # WIN: Huge positive reward!
            return 500.0
        elif final_score < 0:
            # LOSS: Huge negative penalty
            return -500.0
        else:
            # TIE: Small penalty (winning is better than tying)
            return -50.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Integer action (0-4)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.game_state is None or self.capture_agent is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Convert action to direction
        direction = self.action_to_direction[action]
        
        # Get legal actions
        legal_actions = self.game_state.get_legal_actions(self.agent_index)
        
        # Store previous state
        prev_state = self.game_state
        
        # Penalty for illegal actions
        illegal_action_penalty = 0.0
        if direction not in legal_actions:
            # Penalize illegal moves and use a legal alternative
            illegal_action_penalty = -10.0
            # Choose STOP if legal, otherwise first legal action
            if Directions.STOP in legal_actions:
                direction = Directions.STOP
            else:
                direction = legal_actions[0]
        
        # Execute action (this would normally be done by the game engine)
        # For Gym wrapper, we'll need to get the successor
        successor = self.game_state.generate_successor(self.agent_index, direction)
        
        # Update game state
        self.game_state = successor
        self.steps += 1
        
        # Calculate reward
        reward = self._get_reward(prev_state, self.game_state)
        
        # Add illegal action penalty
        reward += illegal_action_penalty
        
        # Get observation
        observation = self._get_observation(self.game_state)
        
        # Check if episode is done
        terminated = self._is_terminal(self.game_state)
        truncated = self.steps >= self.max_steps
        
        # Add terminal reward for win/loss/tie
        if terminated or truncated:
            terminal_reward = self._get_terminal_reward(self.game_state)
            reward += terminal_reward
        
        # Additional info
        final_score = self.capture_agent.get_score(self.game_state)
        game_result = 'ongoing'
        if terminated or truncated:
            if final_score > 0:
                game_result = 'win'
            elif final_score < 0:
                game_result = 'loss'
            else:
                game_result = 'tie'
        
        info = {
            'score': final_score,
            'food_eaten': prev_state.get_agent_state(self.agent_index).num_carrying - 
                          self.game_state.get_agent_state(self.agent_index).num_carrying,
            'steps': self.steps,
            'game_result': game_result
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Reset will be called by the game engine with a new game_state
        # If reset_func is provided, use it to get fresh state
        if self.reset_func is not None:
            self.game_state = self.reset_func()
        
        self.steps = 0
        
        if self.game_state is not None:
            self.last_score = self.capture_agent.get_score(self.game_state) if self.capture_agent else 0
        
        observation = self._get_observation(self.game_state)
        info = {'score': self.last_score}
        
        return observation, info
    
    def _is_terminal(self, game_state) -> bool:
        """Check if game state is terminal"""
        if game_state is None:
            return False
        
        # Game ends when all food is eaten or time runs out
        food_remaining = len(self.capture_agent.get_food(game_state).as_list())
        return food_remaining <= 2
    
    def render(self):
        """Render the environment (handled by Pacman's graphics)"""
        pass
    
    def close(self):
        """Clean up resources"""
        pass


def make_pacman_env(agent_index: int, capture_agent=None):
    """
    Factory function to create Pacman Gym environment
    
    Args:
        agent_index: Index of the agent
        capture_agent: Reference to CaptureAgent
        
    Returns:
        PacmanCaptureEnv instance
    """
    return PacmanCaptureEnv(agent_index=agent_index, capture_agent=capture_agent)
