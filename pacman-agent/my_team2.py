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
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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
    return [eval(first)(first_index), eval(second)(second_index)]


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

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.position_history = []

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        agent_state = successor.get_agent_state(self.index)
        agent_pos = agent_state.get_position()
        features['successor_score'] = -len(food_list)

        # Distance to nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(agent_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        # Distance to capsules
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            min_capsule_distance = min([self.get_maze_distance(agent_pos, cap) for cap in capsules])
            features['distance_to_cap'] = min_capsule_distance
        
        # Ghost danger assessment
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        active_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]
        
        min_g_dist = float('inf')
        if len(active_ghosts) > 0:
            g_dists = [self.get_maze_distance(agent_pos, enemy.get_position()) for enemy in active_ghosts]
            min_g_dist = min(g_dists)
            # Only apply danger if ghost is actually close
            if min_g_dist <= 5:
                features['danger_from_ghost'] = 6 - min_g_dist  # Linear: 5,4,3,2,1 when dist is 1,2,3,4,5
        
        # Carrying and home features
        features['carrying_food'] = agent_state.num_carrying
        home_dist = self.get_maze_distance(agent_pos, self.start)
        
        # When carrying food, track distance to home
        if agent_state.num_carrying > 0:
            features['distance_to_home'] = home_dist
        
        # Emergency return home: use NEGATIVE distance so closer to home = better score
        if agent_state.num_carrying >= 3 and min_g_dist <= 5:
            features['must_return_home'] = -home_dist  # Closer to home = less negative = better
        elif agent_state.num_carrying >= 5:
            # Always return when carrying lots of food
            features['must_return_home'] = -home_dist

        # Anti-stuck mechanisms
        if action == Directions.STOP:
            features['stop'] = 1
        
        # Penalize reversing direction
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        
        # Detect loops - check if we're revisiting same position
        if len(self.position_history) > 4:
            if self.position_history[-4:].count(agent_pos) >= 2:
                features['looping'] = 1
        
        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,        # Primary goal: eat food
            'distance_to_food': -1,        # Get closer to food
            'distance_to_cap': -3,         # Capsules are valuable
            'danger_from_ghost': -100,     # Strong avoidance of ghosts
            'carrying_food': 5,            # Bonus for carrying (but not too much)
            'distance_to_home': -2,        # When carrying, get home
            'must_return_home': 500,       # OVERRIDE: return when triggered
            'stop': -200,                  # Never stop
            'reverse': -5,                 # Avoid going backwards
            'looping': -500,               # Heavy penalty for looping
        }
    
    def choose_action(self, game_state):
        """Track positions to detect loops and call parent's choose_action"""
        my_pos = game_state.get_agent_state(self.index).get_position()
        self.position_history.append(my_pos)
        
        # Keep only recent history
        if len(self.position_history) > 8:
            self.position_history.pop(0)
        
        return super().choose_action(game_state)


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    An improved defensive agent with patrol behavior, capsule defense,
    and smart positioning based on game state.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_point = None

    def get_patrol_point(self, game_state):
        """Calculate optimal patrol position - center of food cluster"""
        if self.patrol_point is None:
            food_defending = self.get_food_you_are_defending(game_state).as_list()
            if len(food_defending) > 0:
                # Calculate center of food cluster
                x_center = sum([f[0] for f in food_defending]) / len(food_defending)
                y_center = sum([f[1] for f in food_defending]) / len(food_defending)
                center_point = (int(x_center), int(y_center))
                
                # Find the nearest valid (non-wall) position to the center
                walls = game_state.get_walls()
                if not walls[center_point[0]][center_point[1]]:
                    self.patrol_point = center_point
                else:
                    # If center is a wall, find closest valid position
                    min_dist = float('inf')
                    for food_pos in food_defending:
                        dist = abs(food_pos[0] - x_center) + abs(food_pos[1] - y_center)
                        if dist < min_dist:
                            min_dist = dist
                            self.patrol_point = food_pos
            else:
                self.patrol_point = self.start
        return self.patrol_point

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            
            # Block escape route - get between invader and their border
            invader_pos = invaders[0].get_position()
            border_x = game_state.data.layout.width // 2
            walls = game_state.get_walls()
            
            # Calculate blocking position (on the border between invader and their home)
            if self.red:
                # Red defends left side, want to be between invader and right border
                blocking_x = border_x - 1
            else:
                # Blue defends right side, want to be between invader and left border
                blocking_x = border_x + 1
            
            # Find valid blocking position at same Y as invader
            blocking_pos = (int(blocking_x), int(invader_pos[1]))
            
            # If blocking position is a wall, find nearest valid position
            if walls[blocking_pos[0]][blocking_pos[1]]:
                # Search for valid positions near the blocking line
                min_dist = float('inf')
                best_pos = invader_pos
                for y in range(max(0, blocking_pos[1] - 3), min(walls.height, blocking_pos[1] + 4)):
                    test_pos = (blocking_x, y)
                    if not walls[test_pos[0]][test_pos[1]]:
                        dist = abs(y - blocking_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_pos = test_pos
                blocking_pos = best_pos
            
            features['blocking_distance'] = self.get_maze_distance(my_pos, blocking_pos)
        
        else:
            # No invaders visible - patrol strategically
            patrol_point = self.get_patrol_point(game_state)
            features['patrol_distance'] = self.get_maze_distance(my_pos, patrol_point)
        
        # Defend capsules
        capsules_defending = self.get_capsules_you_are_defending(successor)
        if len(capsules_defending) > 0:
            cap_dists = [self.get_maze_distance(my_pos, cap) for cap in capsules_defending]
            min_cap_dist = min(cap_dists)
            
            # If invaders are near capsules, prioritize defending them
            if len(invaders) > 0:
                invader_to_cap_dists = [self.get_maze_distance(inv.get_position(), cap) 
                                       for inv in invaders for cap in capsules_defending]
                if len(invader_to_cap_dists) > 0 and min(invader_to_cap_dists) < 5:
                    features['defend_capsule'] = min_cap_dist
        
        # Behavior when scared (enemy ate our capsule)
        if my_state.scared_timer > 0:
            features['scared'] = 1
            # Flee from invaders when scared
            if len(invaders) > 0:
                invader_dists = [self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders]
                # Negative distance means we want to maximize it
                features['flee_when_scared'] = -min(invader_dists)

        # Anti-stuck mechanisms
        if action == Directions.STOP: 
            features['stop'] = 1
        
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """Dynamic weights based on whether agent is scared or not"""
        my_state = game_state.get_agent_state(self.index)
        
        if my_state.scared_timer > 0:
            # Defensive weights when scared - avoid invaders
            return {
                'on_defense': 200,
                'flee_when_scared': 100,
                'num_invaders': -500,
                'scared': -1000,
                'stop': -100,
                'reverse': -2
            }
        else:
            # Aggressive weights when not scared
            return {
                'num_invaders': -1000,      # Eliminate invaders (top priority)
                'invader_distance': -10,    # Chase them down
                'on_defense': 100,          # Stay on defensive side
                'blocking_distance': -5,    # Get to blocking position
                'defend_capsule': -20,      # Guard capsules when threatened
                'patrol_distance': -1,      # Patrol when no threats
                'stop': -100,               # Never stop
                'reverse': -2               # Avoid reversing
            }
