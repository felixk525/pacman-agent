"""
Test the trained SB3 agent to see what it learned
"""
import sys
import numpy as np
from stable_baselines3 import PPO
from contest.pacman_gym_env import PacmanCaptureEnv

# Load the trained model
print("Loading trained model...")
try:
    model = PPO.load("sb3_model_PPO_standalone.zip")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Create test environment
class TestCaptureAgent:
    def __init__(self):
        self.index = 0
        self.start = (1, 1)
    
    def get_food(self, state):
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
        return state.data.score

class TestGameState:
    def __init__(self, position=(2, 2), food_list=None, carrying=0, returned=0, is_pacman=True, step=0):
        self.position = position
        self.carrying = carrying
        self.returned = returned
        self.is_pacman = is_pacman
        self.step = step
        
        if food_list is None:
            self.food_list = [
                (6, 2), (7, 2), (8, 2),
                (6, 3), (7, 3), (8, 3),
                (6, 4), (7, 4), (8, 4)
            ]
        else:
            self.food_list = food_list
        
        self.data = type('obj', (object,), {'score': returned})()
    
    def get_agent_state(self, index):
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
        
        x, y = self.position
        if action == Directions.NORTH:
            new_pos = (x, y + 1)
        elif action == Directions.SOUTH:
            new_pos = (x, y - 1)
        elif action == Directions.EAST:
            new_pos = (x + 1, y)
        elif action == Directions.WEST:
            new_pos = (x - 1, y)
        else:
            new_pos = (x, y)
        
        new_x, new_y = new_pos
        if new_x < 0 or new_x >= 20 or new_y < 0 or new_y >= 10:
            new_pos = self.position
        
        new_food_list = list(self.food_list)
        new_carrying = self.carrying
        new_returned = self.returned
        new_is_pacman = self.is_pacman
        
        if new_pos in new_food_list:
            new_food_list.remove(new_pos)
            new_carrying += 1
            new_is_pacman = True
        
        home_boundary = 2
        if new_x <= home_boundary and new_carrying > 0:
            new_returned += new_carrying
            new_carrying = 0
            new_is_pacman = False
        
        return TestGameState(
            position=new_pos,
            food_list=new_food_list,
            carrying=new_carrying,
            returned=new_returned,
            is_pacman=new_is_pacman,
            step=self.step + 1
        )

# Run test episode
print("\n" + "="*60)
print("Testing Trained Agent")
print("="*60)

test_agent = TestCaptureAgent()
test_state = TestGameState()
env = PacmanCaptureEnv(agent_index=0, game_state=test_state, capture_agent=test_agent)

obs, info = env.reset()
total_reward = 0
done = False
step = 0
max_steps = 100

print(f"\nStarting position: {test_state.position}")
print(f"Food locations: {test_state.food_list[:3]}... ({len(test_state.food_list)} total)")
print(f"\nSimulating {max_steps} steps...\n")

action_names = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'STOP']

while not done and step < max_steps:
    # Predict action
    action, _ = model.predict(obs, deterministic=True)
    
    # Convert numpy array to int
    if hasattr(action, 'item'):
        action = action.item()
    elif isinstance(action, (list, np.ndarray)):
        action = int(action[0])
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    
    # Print interesting events
    current_state = env.game_state
    if reward > 50:  # Significant positive reward
        print(f"Step {step}: {action_names[action]} -> Pos={current_state.position}, "
              f"Carrying={current_state.carrying}, Returned={current_state.returned}, "
              f"Reward={reward:.1f}")
    
    if step % 20 == 0:
        print(f"Step {step}: Pos={current_state.position}, Food_left={len(current_state.food_list)}, "
              f"Carrying={current_state.carrying}, Returned={current_state.returned}")
    
    step += 1

print(f"\n{'='*60}")
print(f"Episode finished after {step} steps")
print(f"Total reward: {total_reward:.2f}")
print(f"Final position: {env.game_state.position}")
print(f"Food returned: {env.game_state.returned}")
print(f"Food remaining: {len(env.game_state.food_list)}")
print(f"{'='*60}")
