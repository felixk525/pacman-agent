# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='QLearningAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # Pass num_training to the Q-learning agent
    first_agent = eval(first)(first_index, num_training=num_training)
    second_agent = eval(second)(second_index)
    return [first_agent, second_agent]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


class QLearningAgent(ReflexCaptureAgent):
    """
    An agent that learns to play the game using Q-learning with function approximation.
    """

    def __init__(self, index, epsilon=0.1, alpha=0.2, gamma=0.9, num_training=100, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # Q-learning parameters
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.weights = util.Counter()  # feature weights for function approximation
        
        # Training tracking
        self.num_training = num_training
        self.episodes_so_far = 0
        self.episode_rewards = 0.0
        self.accumulated_training_rewards = 0.0
        
        # Game outcome tracking
        self.game_results = {'wins': 0, 'losses': 0, 'ties': 0}
        
        # State tracking for learning
        self.last_state = None
        self.last_action = None
        
        # Try to load pre-trained weights
        self.load_weights()
    
    def register_initial_state(self, game_state):
        """Called at the beginning of each game"""
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # Reset episode tracking
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0
        
        # Initialize position history for loop detection
        self.position_history = []
    
    def get_q_value(self, state, action):
        """
        Returns Q(state, action) using linear function approximation
        Q(s,a) = w1*f1(s,a) + w2*f2(s,a) + ... = weights · features
        """
        features = self.get_features(state, action)
        return features * self.weights

    def choose_action(self, game_state):
        """
        Choose action using epsilon-greedy policy during training,
        or greedy policy during competition.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        
        # Remove STOP from actions if possible
        legal_actions = [a for a in legal_actions if a != Directions.STOP] or legal_actions
        
        action = None
        
        # Epsilon-greedy: explore during training
        if self.is_in_training() and util.flip_coin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            # Exploit: choose best action based on Q-values
            q_values = [self.get_q_value(game_state, a) for a in legal_actions]
            max_q_value = max(q_values)
            best_actions = [a for a, q in zip(legal_actions, q_values) if q == max_q_value]
            action = random.choice(best_actions)
        
        # Perform learning update from previous transition
        if self.last_state is not None and self.is_in_training():
            reward = self.get_reward(self.last_state, game_state)
            self.update(self.last_state, self.last_action, game_state, reward)
            self.episode_rewards += reward
        
        # Track position for loop detection
        current_pos = game_state.get_agent_state(self.index).get_position()
        self.position_history.append(current_pos)
        # Keep only last 10 positions
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Store for next update
        self.last_state = game_state
        self.last_action = action
        
        return action
    
    def update(self, state, action, next_state, reward):
        """
        Q-learning update rule with function approximation:
        
        TD error = reward + gamma * max_a' Q(s', a') - Q(s, a)
        
        For each feature i:
        w_i = w_i + alpha * TD_error * f_i(s, a)
        """
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Best Q-value from next state
        next_actions = next_state.get_legal_actions(self.index)
        if next_actions:
            next_q_values = [self.get_q_value(next_state, a) for a in next_actions]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0.0  # Terminal state
        
        # Temporal difference error
        td_error = (reward + self.gamma * max_next_q) - current_q
        
        # Update weights using gradient descent
        features = self.get_features(state, action)
        for feature, value in features.items():
            self.weights[feature] += self.alpha * td_error * value
    
    def get_features(self, game_state, action):
        """
        Extract normalized features from state-action pair.
        Features are designed to capture important game dynamics.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        agent_state = successor.get_agent_state(self.index)
        agent_pos = agent_state.get_position()
        
        # Get map dimensions for normalization
        walls = game_state.get_walls()
        map_size = walls.width * walls.height

        # Feature 1: Bias (always 1)
        features['bias'] = 1.0
        
        # Feature 2: Distance to nearest food
        food_list = self.get_food(successor).as_list()
        if food_list:
            min_food_dist = min([self.get_maze_distance(agent_pos, food) for food in food_list])
            features['closest-food'] = float(min_food_dist) / map_size
        
        # Feature 3: Number of food remaining (normalized)
        features['food-remaining'] = len(food_list) / 20.0
        
        # Feature 4: Ghost danger
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        
        if ghosts:
            min_ghost_dist = min([self.get_maze_distance(agent_pos, g.get_position()) for g in ghosts])
            features['ghost-distance'] = float(min_ghost_dist) / map_size
            # Inverse danger: closer = more dangerous
            if min_ghost_dist <= 5:
                features['ghost-danger'] = 1.0 / (min_ghost_dist + 1)
        
        # Feature 5: Carrying food (encourage returning)
        features['carrying-food'] = agent_state.num_carrying / 10.0
        
        # Feature 6: Distance to home when carrying food
        if agent_state.num_carrying > 0:
            home_dist = self.get_maze_distance(agent_pos, self.start)
            features['distance-to-home'] = float(home_dist) / map_size
        
        # Feature 7: Capsule proximity
        capsules = self.get_capsules(successor)
        if capsules:
            min_cap_dist = min([self.get_maze_distance(agent_pos, cap) for cap in capsules])
            features['closest-capsule'] = float(min_cap_dist) / map_size
        
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        
        # Detect loops - check if we're revisiting same position
        if len(self.position_history) > 4:
            if self.position_history[-4:].count(agent_pos) >= 2:
                features['looping'] = 1
        
        # Feature 8: Score change
        features['score-delta'] = (self.get_score(successor) - self.get_score(game_state)) / 100.0
        
        return features
    
    def get_reward(self, state, next_state):
        """
        Calculate reward for transitioning from state to next_state.
        Shaped rewards to guide learning towards good behavior.
        """
        reward = 0.0
        
        prev_agent = state.get_agent_state(self.index)
        curr_agent = next_state.get_agent_state(self.index)
        curr_pos = curr_agent.get_position()
        
        # Reward 1: Eating food (BIG reward to encourage eating!)
        prev_food_count = len(self.get_food(state).as_list())
        curr_food_count = len(self.get_food(next_state).as_list())
        food_eaten = prev_food_count - curr_food_count
        if food_eaten > 0:
            reward += 100.0 * food_eaten  # Increased from 10 to 100!
        
        # Reward 2: Returning food (massive reward!)
        food_returned = curr_agent.num_returned - prev_agent.num_returned
        if food_returned > 0:
            reward += 200.0 * food_returned  # Increased from 50 to 200!
        
        # Penalty 3: Getting caught (lose carried food)
        if curr_agent.get_position() == self.start and prev_agent.is_pacman:
            if prev_agent.num_carrying > 0:
                reward -= 200.0  # Heavy penalty for dying with food
            else:
                reward -= 50.0  # Still penalize dying
        
        # Penalty 4: Stopping (encourage movement)
        if self.last_action == Directions.STOP:
            reward -= 10.0
        
        # Reward 5: Moving toward food (distance-based shaping)
        # This encourages the agent to get closer to food
        if curr_agent.is_pacman and curr_agent.num_carrying < 5:
            prev_pos = prev_agent.get_position()
            food_list = self.get_food(next_state).as_list()
            if food_list:
                prev_min_food_dist = min([self.get_maze_distance(prev_pos, food) for food in food_list])
                curr_min_food_dist = min([self.get_maze_distance(curr_pos, food) for food in food_list])
                # Reward for getting closer, penalty for getting farther
                distance_improvement = prev_min_food_dist - curr_min_food_dist
                reward += distance_improvement * 2.0  # Stronger shaping
        
        # Penalty 6: Being near ghosts (but don't let it dominate food rewards)
        enemies = [next_state.get_agent_state(i) for i in self.get_opponents(next_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        
        if ghosts and curr_agent.is_pacman:
            min_ghost_dist = min([self.get_maze_distance(curr_pos, g.get_position()) for g in ghosts])
            if min_ghost_dist <= 5:
                # Exponential penalty - very dangerous when close
                reward -= (12 - min_ghost_dist) ** 2  # e.g., dist=1: -25, dist=3: -9, dist=5: -1
        
        # Reward 7: Encourage crossing to opponent side
        if curr_agent.is_pacman and not prev_agent.is_pacman:
            reward += 5.0  # Small reward for entering enemy territory
        
        # Reward 8: Encourage returning home when carrying food
        if curr_agent.num_carrying >= 3:
            # Get distance to home
            home_dist = self.get_maze_distance(curr_pos, self.start)
            prev_home_dist = self.get_maze_distance(prev_agent.get_position(), self.start)
            # Reward for moving toward home when loaded
            if home_dist < prev_home_dist:
                reward += 3.0 * curr_agent.num_carrying
        
        return reward
    
    def final(self, game_state):
        """
        Called at the end of each game. Handle learning cleanup and tracking.
        """
        # Final update if we have a last state
        if self.last_state is not None and self.is_in_training():
            reward = self.get_reward(self.last_state, game_state)
            
            # Add game outcome reward (win/loss/tie)
            final_score = self.get_score(game_state)
            game_outcome_reward = 0.0
            
            if final_score > 0:
                # WIN: Big positive reward
                game_outcome_reward = 500.0
                game_result = "WIN"
            elif final_score < 0:
                # LOSS: Big negative penalty
                game_outcome_reward = -500.0
                game_result = "LOSS"
            else:
                # TIE: Small penalty (winning is better than tie)
                game_outcome_reward = -500.0
                game_result = "TIE"
            
            # Terminal state: no next actions
            # Include score delta + game outcome
            final_reward = reward + final_score * 10 + game_outcome_reward
            self.update(self.last_state, self.last_action, game_state, final_reward)
            self.episode_rewards += final_reward
            
            # Track game result
            if not hasattr(self, 'game_results'):
                self.game_results = {'wins': 0, 'losses': 0, 'ties': 0}
            
            if final_score > 0:
                self.game_results['wins'] += 1
            elif final_score < 0:
                self.game_results['losses'] += 1
            else:
                self.game_results['ties'] += 1
        
        # Track episode completion
        self.episodes_so_far += 1
        self.accumulated_training_rewards += self.episode_rewards
        
        # Decay exploration rate over training
        if self.is_in_training():
            # Gradually reduce epsilon from initial value to 0.01
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            # Print training progress with win/loss stats
            if self.episodes_so_far % 10 == 0:
                avg_reward = self.accumulated_training_rewards / 10.0
                
                # Calculate win rate
                if hasattr(self, 'game_results'):
                    total_games = sum(self.game_results.values())
                    win_rate = (self.game_results['wins'] / total_games * 100) if total_games > 0 else 0
                    print(f"Episode {self.episodes_so_far}/{self.num_training} - "
                          f"Avg Reward: {avg_reward:.2f} - Epsilon: {self.epsilon:.3f}")
                    print(f"  Record: {self.game_results['wins']}W-{self.game_results['losses']}L-{self.game_results['ties']}T "
                          f"(Win Rate: {win_rate:.1f}%)")
                else:
                    print(f"Episode {self.episodes_so_far}/{self.num_training} - "
                          f"Avg Reward: {avg_reward:.2f} - Epsilon: {self.epsilon:.3f}")
                
                self.accumulated_training_rewards = 0.0
            
            # Save weights every episode during training (important for recovery)
            self.save_weights()
            
            # If training just finished, save final weights
            if self.episodes_so_far == self.num_training:
                print(f"\n{'='*60}")
                print(f"=== Training Complete! ===")
                print(f"{'='*60}")
                print(f"Total episodes: {self.num_training}")
                print(f"Final weights saved to q_weights_agent_{self.index}.pkl")
                
                # Print final statistics
                if hasattr(self, 'game_results'):
                    total_games = sum(self.game_results.values())
                    win_rate = (self.game_results['wins'] / total_games * 100) if total_games > 0 else 0
                    print(f"\nFinal Record:")
                    print(f"  Wins:   {self.game_results['wins']} ({self.game_results['wins']/total_games*100:.1f}%)")
                    print(f"  Losses: {self.game_results['losses']} ({self.game_results['losses']/total_games*100:.1f}%)")
                    print(f"  Ties:   {self.game_results['ties']} ({self.game_results['ties']/total_games*100:.1f}%)")
                    print(f"  Win Rate: {win_rate:.1f}%")
                print(f"{'='*60}\n")
        else:
            # Competition mode: report performance
            final_score = self.get_score(game_state)
            result = "WIN" if final_score > 0 else "LOSS" if final_score < 0 else "TIE"
            print(f"Game complete - Result: {result} (Score: {final_score}) - Total Reward: {self.episode_rewards:.2f}")
        
        # Reset for next episode
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0
        
        # Call parent's final method
        super().final(game_state)
    
    def is_in_training(self):
        """Check if agent is still in training mode"""
        return self.episodes_so_far < self.num_training
    
    def save_weights(self):
        """Save learned weights to file"""
        import pickle
        import os
        
        weights_file = f'q_weights_agent_{self.index}.pkl'
        try:
            with open(weights_file, 'wb') as f:
                pickle.dump(dict(self.weights), f)
            # Only print every 10 episodes to avoid spam
            if self.episodes_so_far % 10 == 0 or self.episodes_so_far == self.num_training:
                print(f"Weights saved to {weights_file} (Episode {self.episodes_so_far})")
        except Exception as e:
            print(f"Error saving weights: {e}")
    
    def load_weights(self):
        """Load pre-trained weights from file"""
        import pickle
        import os
        
        weights_file = f'q_weights_agent_{self.index}.pkl'
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'rb') as f:
                    loaded_weights = pickle.load(f)
                    self.weights = util.Counter(loaded_weights)
                print(f"Loaded weights from {weights_file}")
                print(f"Weights: {dict(self.weights)}")
            except Exception as e:
                print(f"Error loading weights: {e}")
                self.weights = util.Counter()
        else:
            print(f"No pre-trained weights found at {weights_file}, starting fresh")