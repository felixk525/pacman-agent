"""
Pacman agent using Stable-Baselines3 with Gym wrapper
Train using PPO, A2C, or DQN algorithms
"""

import random
import numpy as np
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from contest.pacman_gym_env import PacmanCaptureEnv

# Import Stable-Baselines3 (install: pip install stable-baselines3)
try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("WARNING: stable-baselines3 not installed. Install with: pip install stable-baselines3")


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='SB3Agent', second='DefensiveReflexAgent', num_training=0):
    """
    Create team with SB3 agent for offense and reflex agent for defense
    """
    first_agent = eval(first)(first_index, num_training=num_training)
    second_agent = eval(second)(second_index)
    return [first_agent, second_agent]


##########
# Agents #
##########

class SB3Agent(CaptureAgent):
    """
    Agent that uses Stable-Baselines3 (PPO, A2C, or DQN) for decision making
    """
    
    def __init__(self, index, algorithm='PPO', num_training=0, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        
        self.algorithm_name = algorithm
        self.num_training = num_training
        self.episodes_so_far = 0
        self.episode_rewards = 0.0
        
        # Gym environment wrapper
        self.env = None
        self.model = None
        
        # Training tracking
        self.training_scores = []
        self.last_action = None
        
        # Check if SB3 is available
        if not SB3_AVAILABLE:
            print(f"Agent {index}: Falling back to random policy (SB3 not installed)")
    
    def register_initial_state(self, game_state):
        """Initialize agent at game start"""
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # Create Gym environment only once
        if self.env is None:
            self.env = PacmanCaptureEnv(
                agent_index=self.index,
                game_state=game_state,
                capture_agent=self
            )
        
        # Load or create model only once
        if SB3_AVAILABLE and self.model is None:
            self._load_or_create_model()
        
        # Update environment with new game state
        self.env.game_state = game_state
        self.env.reset()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        import os
        
        model_path = f'sb3_model_{self.algorithm_name}_agent_{self.index}.zip'
        
        # Select algorithm
        algorithm_class = {
            'PPO': PPO,
            'A2C': A2C,
            'DQN': DQN
        }.get(self.algorithm_name, PPO)
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self.model = algorithm_class.load(model_path, env=self.env)
                print(f"Agent {self.index}: Loaded {self.algorithm_name} model from {model_path}")
            except Exception as e:
                print(f"Agent {self.index}: Error loading model: {e}")
                self._create_new_model(algorithm_class)
        else:
            self._create_new_model(algorithm_class)
    
    def _create_new_model(self, algorithm_class):
        """Create a new SB3 model"""
        print(f"Agent {self.index}: Creating new {self.algorithm_name} model")
        
        # Hyperparameters for each algorithm
        if self.algorithm_name == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1
            )
        elif self.algorithm_name == 'A2C':
            self.model = A2C(
                'MlpPolicy',
                self.env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                verbose=1
            )
        elif self.algorithm_name == 'DQN':
            self.model = DQN(
                'MlpPolicy',
                self.env,
                learning_rate=1e-4,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                verbose=1
            )
        else:
            # Default to PPO
            self.model = PPO('MlpPolicy', self.env, verbose=1)
    
    def choose_action(self, game_state):
        """
        Choose action using SB3 model
        """
        # Update environment's game state
        self.env.game_state = game_state
        
        legal_actions = game_state.get_legal_actions(self.index)
        
        # If SB3 not available, use random
        if not SB3_AVAILABLE or self.model is None:
            return random.choice(legal_actions)
        
        # Get observation
        obs = self.env._get_observation(game_state)
        
        # Predict action using model
        action, _states = self.model.predict(obs, deterministic=not self.is_in_training())
        
        # Convert numpy array to int if necessary
        if hasattr(action, 'item'):
            action = action.item()
        elif isinstance(action, (list, np.ndarray)):
            action = int(action[0])
        else:
            action = int(action)
        
        # Convert action to direction
        direction = self.env.action_to_direction[action]
        
        # Ensure action is legal
        if direction not in legal_actions:
            direction = random.choice(legal_actions)
        
        return direction
    
    def final(self, game_state):
        """Called at end of each game"""
        self.episodes_so_far += 1
        
        score = self.get_score(game_state)
        self.training_scores.append(score)
        
        if self.is_in_training():
            # Train the model after each episode
            if SB3_AVAILABLE and self.model is not None:
                print(f"\nAgent {self.index} - Episode {self.episodes_so_far}/{self.num_training}")
                print(f"Score: {score}")
                
                # Save model periodically
                if self.episodes_so_far % 10 == 0:
                    self.save_model()
                
                # Final save
                if self.episodes_so_far >= self.num_training:
                    self.save_model()
                    print(f"\n=== Training Complete ===")
                    print(f"Average Score: {np.mean(self.training_scores):.2f}")
        
        # Call parent
        super().final(game_state)
    
    def is_in_training(self):
        """Check if still in training mode"""
        return self.episodes_so_far < self.num_training
    
    def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            model_path = f'sb3_model_{self.algorithm_name}_agent_{self.index}.zip'
            try:
                self.model.save(model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")


class DefensiveReflexAgent(CaptureAgent):
    """
    Simple defensive reflex agent (reused from baseline)
    """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
    
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
    
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)
    
    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor
    
    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        
        if action == Directions.STOP:
            features['stop'] = 1
        
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        
        return features
    
    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2
        }
