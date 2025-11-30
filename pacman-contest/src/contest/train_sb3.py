"""
Standalone training script for Pacman agent using Stable-Baselines3

This script allows you to train agents outside of the game loop
for better control and monitoring.

Usage:
    python train_sb3.py --algorithm PPO --timesteps 100000
    python train_sb3.py --algorithm DQN --timesteps 50000
"""

import argparse
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os


class GameScoreCallback(BaseCallback):
    """
    Callback for logging game scores and metrics during training
    """
    def __init__(self, verbose=0):
        super(GameScoreCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            # Get episode info
            info = self.locals.get('infos', [{}])[0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                # Log every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-10:])
                    mean_length = np.mean(self.episode_lengths[-10:])
                    print(f"\nEpisode {len(self.episode_rewards)}")
                    print(f"  Mean Reward (last 10): {mean_reward:.2f}")
                    print(f"  Mean Length (last 10): {mean_length:.0f}")
        
        return True


def train_sb3_agent(algorithm='PPO', total_timesteps=100000, save_freq=10000):
    """
    Train a Stable-Baselines3 agent
    
    Args:
        algorithm: Algorithm to use (PPO, A2C, DQN)
        total_timesteps: Total training timesteps
        save_freq: Save model every N timesteps
    """
    print(f"\n{'='*60}")
    print(f"Training {algorithm} agent for Pacman Capture-the-Flag")
    print(f"{'='*60}\n")
    
    try:
        # Import the gym environment wrapper
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from contest.pacman_gym_env import PacmanCaptureEnv
        
        print("✓ Gym environment imported successfully")
        
        # For standalone training, we need a mock CaptureAgent
        # In real scenario, this would be integrated with actual game
        print("\nNOTE: Standalone training requires integration with game engine.")
        print("This script provides a training template but needs:")
        print("1. A way to initialize game states without running full games")
        print("2. A mock or simplified CaptureAgent for environment")
        print("3. Proper state transitions outside game loop")
        print("\nFor now, use the integrated training approach:")
        print("  python capture.py -r my_team_sb3 -b baseline_team -n 100 -x 100 -q")
        print("\nAttempting to create a training pipeline...")
        
        # Create a simple dummy environment for demonstration
        # In production, you'd integrate with actual game initialization
        class DummyCaptureAgent:
            """Mock agent for standalone training environment"""
            def __init__(self):
                self.index = 0
                self.start = (1, 1)
                
            def get_food(self, state):
                # Return actual food grid from state
                class MockGrid:
                    def __init__(self, food_list):
                        self.food_list = food_list
                    def as_list(self):
                        return self.food_list
                return MockGrid(state.food_list)
            
            def get_opponents(self, state):
                return [1, 3]
            
            def get_maze_distance(self, pos1, pos2):
                return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            
            def get_capsules(self, state):
                return []
            
            def get_score(self, state):
                return 0
        
        # Create a mock game state with actual state transitions
        class DummyGameState:
            """Mock game state for standalone training with state transitions"""
            def __init__(self, position=(2, 2), food_list=None, carrying=0, returned=0, is_pacman=True, step=0):
                self.position = position
                self.carrying = carrying
                self.returned = returned
                self.is_pacman = is_pacman
                self.step = step
                
                # Initialize food positions if not provided
                if food_list is None:
                    self.food_list = [
                        (6, 2), (7, 2), (8, 2),  # Row of food
                        (6, 3), (7, 3), (8, 3),
                        (6, 4), (7, 4), (8, 4)
                    ]
                else:
                    self.food_list = food_list
                
                self.data = type('obj', (object,), {'score': returned})()
                
            def get_agent_state(self, index):
                # Return mock agent state
                parent = self
                class MockAgentState:
                    def __init__(self):
                        self.num_carrying = parent.carrying
                        self.num_returned = parent.returned
                        self.is_pacman = parent.is_pacman
                        self.scared_timer = 0
                        
                    def get_position(self):
                        return parent.position
                
                return MockAgentState()
            
            def get_walls(self):
                # Return mock walls
                class MockWalls:
                    width = 20
                    height = 10
                return MockWalls()
            
            def get_legal_actions(self, index):
                from contest.game import Directions
                return [Directions.NORTH, Directions.SOUTH, 
                       Directions.EAST, Directions.WEST, Directions.STOP]
            
            def generate_successor(self, index, action):
                from contest.game import Directions
                
                # Calculate new position based on action
                x, y = self.position
                if action == Directions.NORTH:
                    new_pos = (x, y + 1)
                elif action == Directions.SOUTH:
                    new_pos = (x, y - 1)
                elif action == Directions.EAST:
                    new_pos = (x + 1, y)
                elif action == Directions.WEST:
                    new_pos = (x - 1, y)
                else:  # STOP
                    new_pos = (x, y)
                
                # Boundary check (simple grid)
                new_x, new_y = new_pos
                if new_x < 0 or new_x >= 20 or new_y < 0 or new_y >= 10:
                    new_pos = self.position  # Stay in place if hit wall
                
                # Check if food is eaten
                new_food_list = list(self.food_list)
                new_carrying = self.carrying
                new_returned = self.returned
                new_is_pacman = self.is_pacman
                
                if new_pos in new_food_list:
                    new_food_list.remove(new_pos)
                    new_carrying += 1
                    new_is_pacman = True
                
                # Check if returning home (crossing to left side)
                home_boundary = 2  # X coordinate of home
                if new_x <= home_boundary and new_carrying > 0:
                    new_returned += new_carrying
                    new_carrying = 0
                    new_is_pacman = False
                
                # Create new state with updated values
                return DummyGameState(
                    position=new_pos,
                    food_list=new_food_list,
                    carrying=new_carrying,
                    returned=new_returned,
                    is_pacman=new_is_pacman,
                    step=self.step + 1
                )
        
        # Create environment with dummy agent and state
        print("\nCreating training environment...")
        dummy_agent = DummyCaptureAgent()
        
        # Function to create new random game states
        def create_new_state():
            import random
            # Randomize starting position slightly
            start_x = random.randint(1, 3)
            start_y = random.randint(1, 5)
            
            # Randomize food positions
            food_list = []
            for i in range(6, 10):
                for j in range(2, 6):
                    if random.random() < 0.6:  # 60% chance for food at each spot
                        food_list.append((i, j))
            
            # Ensure at least some food exists
            if len(food_list) < 3:
                food_list = [(6, 2), (7, 3), (8, 4)]
            
            return DummyGameState(position=(start_x, start_y), food_list=food_list)
        
        initial_state = create_new_state()
        env = PacmanCaptureEnv(
            agent_index=0, 
            game_state=initial_state, 
            capture_agent=dummy_agent,
            reset_func=create_new_state
        )
        
        # Wrap environment
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        print("✓ Environment created")
        
        # Select and create algorithm
        print(f"\nInitializing {algorithm} algorithm...")
        if algorithm == 'PPO':
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,  # Encourage exploration
                verbose=1
            )
        elif algorithm == 'A2C':
            model = A2C(
                'MlpPolicy',
                env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                verbose=1
            )
        elif algorithm == 'DQN':
            model = DQN(
                'MlpPolicy',
                env,
                learning_rate=1e-4,
                buffer_size=10000,
                batch_size=32,
                gamma=0.99,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"✓ {algorithm} model initialized")
        
        # Create callback for monitoring
        callback = GameScoreCallback()
        
        # Train the model
        print(f"\nStarting training for {total_timesteps} timesteps...")
        print("Press Ctrl+C to stop training early\n")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save the trained model
        model_path = f'sb3_model_{algorithm}_standalone'
        model.save(model_path)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Model saved to: {model_path}.zip")
        print(f"Total episodes: {len(callback.episode_rewards)}")
        if callback.episode_rewards:
            print(f"Mean reward: {np.mean(callback.episode_rewards):.2f}")
            print(f"Best reward: {max(callback.episode_rewards):.2f}")
        print(f"\nTo use this model in the game:")
        print(f"1. Rename/copy {model_path}.zip to sb3_model_{algorithm}_agent_0.zip")
        print(f"2. Run: python capture.py -r my_team_sb3 -b baseline_team")
        print(f"{'='*60}\n")
        
    except ImportError as e:
        print(f"\n✗ Error: Required packages not installed")
        print(f"Details: {e}")
        print("\nPlease install required packages:")
        print("  pip install stable-baselines3 gymnasium numpy torch")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        print("\nFor production training, use the integrated approach:")
        print("  python capture.py -r my_team_sb3 -b baseline_team -n 100 -x 100 -q")


def main():
    parser = argparse.ArgumentParser(description='Train Pacman agent with Stable-Baselines3')
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'A2C', 'DQN'],
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='Save model every N timesteps')
    
    args = parser.parse_args()
    
    train_sb3_agent(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq
    )


if __name__ == '__main__':
    main()
