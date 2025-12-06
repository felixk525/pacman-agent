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
                first='OmniReflexCaptureAgent', second='OmniReflexCaptureAgent', num_training=0):
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
    return OmniReflexCaptureAgent(first_index, True), OmniReflexCaptureAgent(second_index, False)
    #return [eval(first)(first_index), eval(second)(second_index)]


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

class OmniReflexCaptureAgent(CaptureAgent):
    #TODO:
    # Potential improvements in offense function comments 

    def __init__(self, index, attacker, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.offense = attacker
        self.start = None
        self.flee_timer = 0 # How long should we flee
        self.chase_timer = 0 # How long have we been chased
        self.switch = 0 # How long should we switch attacker defender role
        self.recent_positions = []
        self.carrying = 0
        self.two_ghost = 0 # How long should the offense be able to consider special moves
        self.patrol_point = None

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.distancer.get_maze_distances()
        self.start = game_state.get_agent_position(self.index)

    def local_mobility(self, game_state, old_pos, start_pos, max_depth=5):
        # Deadend detector. Does a sort of BFS into a path and if its a deadend it returns 
        # the amount of fields on that path. (x9)
        walls = game_state.get_walls()
        max_reached_depth = 0
        visited = {start_pos, old_pos}
        queue = [(start_pos, 1)]
        while queue:
            pos, depth = queue.pop(0)
            max_reached_depth = max(max_reached_depth, depth)
            if depth == max_depth:
                continue
            x, y = pos
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                if walls[int(nx)][int(ny)]:
                    continue
                nxt = (nx, ny)
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth+1))
        if max_reached_depth < max_depth:
            depth_penalty = (len(visited) - 1) * 9
            return depth_penalty
        return 0

    def ghost_distance(self, game_state, multiple=False):
        # Distance to closest ghost. Semi redundant due to ghost_positions
        # Includes threat weighting for the border
        enemies = self.get_opponents(game_state)
        my_pos = game_state.get_agent_state(self.index).get_position()
        noisy = game_state.get_agent_distances()
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        border_x = width // 2
        
        # for multiple=True collect all (dist, idx)
        results = []
        best_estimate = 9999
        best_index = 0
        for index in enemies:
            s = game_state.get_agent_state(index)
            real_pos = game_state.get_agent_position(index)

            # compute threat factor
            if s.is_pacman:
                # threat increases the closer they are to their border
                if real_pos is not None:
                    dx = abs(real_pos[0] - border_x)
                    threat_factor = 0.7 - 0.6 * (min(dx, 5) / 5.0)
                else:
                    threat_factor = 0.1
            else:
                threat_factor = 1.0  # enemy ghost

            if real_pos is not None:
                dist = self.get_maze_distance(my_pos, real_pos)
                scaled = dist / threat_factor

            else:
                orig_n_dist = noisy[index]
                if orig_n_dist is None:
                    continue
                noisy_dist = orig_n_dist
                if noisy_dist <= 5:
                    noisy_dist = 7
                # collect weighted distances
                candidates = []
                for x in range(width):
                    for y in range(height):
                        if walls[x][y]:
                            continue
                        manh = abs(x - my_pos[0]) + abs(y - my_pos[1])
                        if noisy_dist - 1 <= manh <= noisy_dist and manh > 5:
                            prob = game_state.get_distance_prob(manh, orig_n_dist)
                            real_d = self.get_maze_distance(my_pos, (x, y))
                            if real_d > 5:
                                candidates.append(real_d * prob)

                if not candidates:
                    continue
                estimate = sum(candidates) * 0.9
                scaled = estimate / threat_factor

            # multiple mode
            results.append((round(scaled,3), index))
            # single mode
            if scaled < best_estimate:
                best_estimate = scaled
                best_index = index

        if multiple:
            # sort by distance (closest first)
            results.sort(key=lambda t: t[0])
            return results, best_index
        #if best_estimate == 0:
            #print(f"{best_estimate}, {scaled}, {estimate}, {threat_factor}")
        return best_estimate, best_index

    def home_border_x(self, game_state):
        # Brief helper function to get the border x coordinate
        width = game_state.get_walls().width
        if self.red:
            return width // 2 - 1
        else:
            return width // 2

    def food_risk(self, pos, game_state):
        # This function checks the ratio of enemy and team food. If the agent is carrying a 
        # lot of food or the situaiton looks bad he returns home.
        foods = self.get_food(game_state).as_list()
        team_foods = self.get_food_you_are_defending(game_state).as_list()
        if not foods:
            return 999 # we want to go home
        if not team_foods:
            return 999
        food_enemies = len(foods)
        food_team = len(team_foods)
        if food_team == 0:
            food_team = 1
        factor_1 = game_state.get_agent_state(self.index).num_carrying # ~0-20
        imbalance = max(0,(food_team - food_enemies)) + 30/food_team
        imbalance = factor_1 * 18/food_team 
        flee_val = 1
        if imbalance > 10:
            imbalance = -min([self.get_maze_distance(pos, food) for food in team_foods])
            # Use maze distance and add timer
            imbalance = imbalance
            flee_val = -1
        return imbalance, flee_val # balanced - imbalanced | ~0 - 2
    
    def on_home_ground(self, game_state):
        # Brief helper function to detect if the agent is on their side
        x,y  = game_state.get_agent_state(self.index).get_position()
        if x < self.home_border_x(game_state):
            if self.red:
                return True
            else:
                return False
        else:
            if self.red:
                return False
            else:
                return True

    def border_bias(self, game_state):
        # should return bias to stay on border -> enable easier trapping
        # Redundant - unused
        x, y = game_state.get_agent_state(self.index).get_position()
        borderx = self.home_border_x(game_state)
        value = 0
        if self.red:
            if x >= borderx -2:
                value = abs(x-(borderx-2)) * 4
        else:
            if x <= borderx +2:
                value = abs(x-(borderx+2)) * 4
        #consider multiple invaders - take min
        return value
    
    def closest_ghost_position(self, game_state, invasion=False):
        # for offense - estimates the closes ghost position. Semi redundant due to ghost positions
        enemies = self.get_opponents(game_state)
        my_pos = game_state.get_agent_state(self.index).get_position()
        my_pos = (int(my_pos[0]), int(my_pos[1]))
        noisy = game_state.get_agent_distances()
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        best_dist = 9999

        # fallback position
        best_pos = None
        if self.red:
            # start from right side (width-1) and move left
            for x in range(width - 1, -1, -1):
                for y in range(height):
                    if not walls[x][y]:
                        best_pos = (x, y)
                        break
                if best_pos:
                    break
        else:
            # start from left side (0) and move right
            for x in range(width):
                for y in range(height):
                    if not walls[x][y]:
                        best_pos = (x, y)
                        break
                if best_pos:
                    break
        for index in enemies:
            s = game_state.get_agent_state(index)
            # invasion=True  → only track invaders (enemy Pacman)
            # invasion=False → only track ghosts
            if invasion:
                if not s.is_pacman:
                    continue
            else:
                if s.is_pacman:
                    continue

            real_pos = game_state.get_agent_position(index)
            # position known
            if real_pos is not None:
                dist = self.get_maze_distance(my_pos, real_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = real_pos
                continue

            # noisy position estimate
            orig_n_dist = noisy[index]
            if orig_n_dist is None:
                continue
            noisy_dist = orig_n_dist
            if noisy_dist <= 5:
                noisy_dist = 7
            candidates = []
            for x in range(width):
                for y in range(height):
                    if walls[x][y]:
                        continue

                    manh = abs(x - my_pos[0]) + abs(y - my_pos[1])
                    prob = game_state.get_distance_prob(manh, orig_n_dist)
                    # ring check
                    if noisy_dist - 1 <= manh <= noisy_dist:
                        real_d = self.get_maze_distance(my_pos, (x, y))

                        # ignore trivial positions
                        if real_d > 5:
                            candidates.append(((x, y), real_d, prob))
            if not candidates:
                continue
            # score = real distance weighted by probability
            best_candidate = min(
                candidates,
                key=lambda c: c[1] / max(c[2], 1e-6)
            )
            pos, dist, prob = best_candidate

            if dist < best_dist:
                best_dist = dist
                best_pos = pos
        return best_pos

    def ghost_positions(self,game_state):
        # For the special movement consideration this estimates the positions of both enemy agents
        enemies = self.get_opponents(game_state)
        my_pos = game_state.get_agent_state(self.index).get_position()
        my_pos = (int(my_pos[0]), int(my_pos[1]))
        noisy = game_state.get_agent_distances()
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        positions = []

        for index in enemies:
            real_pos = game_state.get_agent_position(index)
            if real_pos is not None:
                positions.append(real_pos)
                continue

            orig_n_dist = noisy[index]
            if orig_n_dist is None:
                continue
            noisy_dist = orig_n_dist
            if noisy_dist <= 5:
                noisy_dist = 7

            candidates = []
            for x in range(width):
                for y in range(height):
                    if walls[x][y]:
                        continue
                    manh = abs(x - my_pos[0]) + abs(y - my_pos[1])
                    prob = game_state.get_distance_prob(manh, orig_n_dist)
                    if noisy_dist - 1 <= manh <= noisy_dist:
                        real_d = self.get_maze_distance(my_pos, (x, y))
                        if real_d > 5:
                            candidates.append(((x, y), real_d, prob))

            if not candidates:
                continue
            # choose (distance/prob) minimal
            best_candidate = min(candidates, key=lambda c: c[1] / max(c[2], 1e-6))
            pos, _, _ = best_candidate
            positions.append(pos)
        return positions

    def offensive(self, game_state, action):
        # potential improvements
        # add y reset entry after defense recovery switch (-)
        # add better a-evaluation to target food outliers (-)
        # when eating the pill go deep in enemy territory (-)
        old_pos = game_state.get_agent_state(self.index).get_position()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        side_reset = False # False - nothing / True -> 20 defensive move afterwards
        scared_time = game_state.get_agent_state(self.index).scared_timer
        foods = self.get_food(game_state).as_list()
        stop_penalty = 0
        
        # If on own side and invader is close - temporarily switch to defense via side reset
        if self.on_home_ground(game_state) and invaders and scared_time < 2:
            invad_pos = self.closest_ghost_position(game_state, True)
            if self.get_maze_distance(my_pos, invad_pos) <= 3:
                side_reset = True
        
        x, y = my_pos
        if action == "Stop": # Stop penalty
            stop_penalty = 100
        # Closest food distance
        if foods:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in foods])
        else:
            min_distance = 0
        # Detect dead ends up to 6 deep
        dead_end_penalty = self.local_mobility(successor, old_pos, my_pos, max_depth=7) # max depth detected - 1 x 9 -> 54 limit in this case
        # Current closest ghost distance and the ghosts index & future state ghost distance
        ghost_dist, e_index = self.ghost_distance(game_state, False)
        ghost_dist_f, _ = self.ghost_distance(successor, False)
        multi_dist, _ = self.ghost_distance(successor, True)
        distances = [d for d, idx in multi_dist]

        # Check whether the enemy is scared
        e_scared_time = game_state.get_agent_state(e_index).scared_timer
        if e_scared_time > 1:
            scared_enemy = 0
        else:
            scared_enemy = 1
        
        # Chase protocol - if chased return home -> Cummulative chase bool of multiple turns > 3
        if ghost_dist > 0 and ghost_dist <= 2:
            chase_bool = 1/ghost_dist
        elif self.chase_timer > 0:
            chase_bool = -0.5
        else:
            chase_bool = 0
        
        # Softlock penalty to avoid standoff situations
        cycle_penalty = 0
        k = self.recent_positions.count(my_pos)
        if k > 1:
            cycle_penalty = -(2 ** (k * 2))

        # A = Incentive to go for food | B = Incentive to bring food home | 
        # C = Incentive against dead ends and stopping | D = Path home via food of own team | 
        # E = Penalty for getting eaten & softlock penalty
        a_evaluation = -min_distance 
        b_evaluation, safety_bool = self.food_risk(my_pos, game_state)
        c_evaluation = -stop_penalty
        if dead_end_penalty > 0:
            if ghost_dist != 0:
                # C-evaluation (1) - Delicate balance for deadlocks - if it exceeds -2 -> we dont commit
                # -> if it stays below -2 -> commit
                c_evaluation -= ((2*(dead_end_penalty/9) + 1)/ (ghost_dist + 0.5)) * 2

        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if not invaders:
            two_ghoster = True
        else:
            two_ghoster = False

        # Special movement calculation | This happens if the agent is chased and the enemy not 
        # scared, and capsules exist and both enemies are in the enemy area. We consider the 
        # border and capsules and find the place where we have the best odds to escape to. 
        # Fluctuations are expected.
        special_movement = False
        best_target = None
        if (self.chase_timer > 3 and e_scared_time < 2):
            capsules = self.get_capsules(game_state)
            if capsules:
                if not invaders:
                    ghost_positions = self.ghost_positions(game_state)

                    # target candidates
                    border_x = self.home_border_x(game_state)
                    walls = game_state.get_walls()
                    height = walls.height
                    border_cells = [(border_x, y) for y in range(height) if not walls[border_x][y]]
                    targets = border_cells + capsules
                    best_margin = -9999

                    # Evaluate each target
                    for t in targets:
                        my_d = self.get_maze_distance(my_pos, t)
                        # compute smallest ghost distance to this target
                        ghost_d = min(self.get_maze_distance(gpos, t) for gpos in ghost_positions)
                        margin = ghost_d - my_d
                        if margin > best_margin:
                            best_margin = margin
                            best_target = t

                    # commit to the escape target
                    if best_target is not None:
                        #print(best_target)
                        special_movement = True
                        c_evaluation -= self.get_maze_distance(my_pos, best_target)
        # if not in special escape mode -> normal pincer avoidance
        if not special_movement:
            c_evaluation -= sum(10 / ((d + 1) ** 2) for d in distances) * 1#((dead_end_penalty + 1) ** 0.5)

        # D-evaluation -> x coordinate incentive
        if self.red:
            d_evaluation = -x#-min_team_distance
        else:
            walls = game_state.get_walls()
            width = walls.width
            d_evaluation = -(width - x)
        
        # E-evaluation - jitter movement penalty, death penalty via respawn movement, next turn death area penalty
        e_evaluation = -150 if (abs(old_pos[0] - my_pos[0]) + abs(old_pos[1] - my_pos[1])) > 2 else 0
        if ghost_dist > 2:  
            e_evaluation += cycle_penalty
        if ghost_dist_f < 2 and scared_enemy == 1: 
            e_evaluation -= 75
        evaluation = a_evaluation + c_evaluation * scared_enemy + e_evaluation

        # Debugging
        # if e_evaluation < 0:
        #     print(f"{scared_enemy} {c_evaluation} {e_evaluation} {old_pos} {my_pos} {action}")
        # if e_scared_time > 3:
        #     safety_bool = 1
        #if ghost_dist == 1 and action != "Stop" or (abs(old_pos[0] - my_pos[0]) + abs(old_pos[1] - my_pos[1])) > 2:
        #if ghost_dist <= 3:
        #if special_movement:
        #print(f"{str(scared_enemy):>6} "f"{a_evaluation:>7.2f} "f"{c_evaluation:>7.2f} "f"{e_evaluation:>7.2f} "f"{dead_end_penalty:>6.2f} "f"{ghost_dist:>6.2f} "f"{ghost_dist_f:>6.2f} "f"{str(action):>10} "f"{evaluation:>7.2f} "f"{str(my_pos):>12} "f"{str(old_pos):>12} "f"{cycle_penalty:>6.2f} "f"{str(special_movement):>6} " f"{distances}")
        
        flee = 0
        if self.flee_timer > 0: # We have already decided to flee - execute
            # This is decided by the food risk tradeoff - roughly - if we carry a lot we flee.
            flee =  - 1
            if self.on_home_ground(game_state):
                flee = -self.flee_timer
            evaluation = b_evaluation + c_evaluation * scared_enemy + e_evaluation

        # Value settings for the chase scenario
        carrying = game_state.get_agent_state(self.index).num_carrying
        if (self.chase_timer > 3 and scared_time < 1 and e_scared_time < 2) or (self.chase_timer > 3 and e_scared_time < 2 and carrying > 2): 
            evaluation = d_evaluation + c_evaluation * scared_enemy + e_evaluation * (carrying + 1)
            if self.on_home_ground(game_state):
                chase_bool = -self.chase_timer
                side_reset = True
        # Debug 2
        #print(f"{a_evaluation:>7.2f} "f"{b_evaluation:>5.1f} "f"{d_evaluation:>7.2f} "f"{c_evaluation:>7.2f} "f"{e_evaluation:>7.2f} "f"{ghost_dist:>6.2f} "f"{ghost_dist_f:>6.2f} "f"{str(action):>10} "f"{evaluation:>7.2f} "f"{str(my_pos):>12} "f"{str(old_pos):>12} "f"{cycle_penalty:>6.2f} "f"{str(special_movement):>6} " f"{distances}"f"{best_target}")
        #print(f"{a_evaluation:>7.2f} "f"{d_evaluation:>7.2f} "f"{c_evaluation:>7.2f} "f"{e_evaluation:>7.2f} "f"{dead_end_penalty:>6.2f} "f"{ghost_dist:>6.2f} "f"{str(action):>7} "f"{evaluation:>7.2f} "f"{str(my_pos):>10} "f"{str(old_pos):>10} "f"{cycle_penalty:>6.2f} "f"{str(special_movement):>6} " f"{distances}")
        return evaluation, safety_bool, chase_bool, flee, side_reset, two_ghoster
    
    def get_defensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        # Use game_state instead of successor to get current invader positions
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        # if len(invaders) > 0 and not self.offense:
        #     inv_positions = [inv.get_position() for inv in invaders]
        #     border_x = self.home_border_x(game_state)
            # print(f"Agent {self.index}: Detected {len(invaders)} invader(s) at {inv_positions}, my_pos={my_pos}, border_x={border_x}, action={action}")
        
        if len(invaders) > 0:
            # Prioritize invaders based on depth in our territory
            border_x = self.home_border_x(game_state)
            invader_priorities = []
            
            for invader in invaders:
                inv_pos = invader.get_position()
                
                # Calculate how deep the invader is in our territory
                if self.red:
                    depth_in_territory = border_x - inv_pos[0]  # Higher = deeper
                else:
                    depth_in_territory = inv_pos[0] - border_x  # Higher = deeper
                
                # if not self.offense:
                    # print(f"  Invader at {inv_pos}: depth={depth_in_territory}, red={self.red}")
                
                # Track all invaders in our territory (depth >= 0)
                # Even invaders at the border (depth=0) should be tracked
                if depth_in_territory < 0:
                    if not self.offense:
                        print(f"  SKIPPED invader at {inv_pos} - not in our territory (depth={depth_in_territory})")
                    continue  # Skip invaders not yet in our territory
                
                dist_to_me = self.get_maze_distance(my_pos, inv_pos)
                
                # Prioritize: deeper invaders get higher priority (lower score)
                # Also consider invaders carrying food
                food_carrying = invader.num_carrying
                
                # Priority score: lower is more urgent
                # Depth is most important, then food carrying, then distance
                priority = -depth_in_territory * 100 - food_carrying * 30 + dist_to_me * 2
                
                invader_priorities.append((priority, invader, dist_to_me))
            
            if invader_priorities:
                # Sort by priority (most urgent first)
                invader_priorities.sort(key=lambda x: x[0])
                
                # Use the most urgent invader
                most_urgent_invader = invader_priorities[0][1]
                features['invader_distance'] = invader_priorities[0][2]
            
                # Block escape route - get between invader and their border
                invader_pos = most_urgent_invader.get_position()
                border_x = game_state.data.layout.width // 2
                walls = game_state.get_walls()
                
                # Calculate blocking position (on the border between invader and their home)
                if self.red:
                    # Red defends left side, want to be between invader and right border
                    blocking_x = border_x - 4
                else:
                    # Blue defends right side, want to be between invader and left border
                    blocking_x = border_x + 5
                
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
                # Invaders detected but all filtered out (shouldn't happen often)
                # Chase the closest visible invader anyway
                if len(invaders) > 0:
                    closest_invader = min(invaders, key=lambda inv: self.get_maze_distance(my_pos, inv.get_position()))
                    features['invader_distance'] = self.get_maze_distance(my_pos, closest_invader.get_position())
        
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

        # Penalize leaving our territory (becoming Pacman when we should defend)
        if my_state.is_pacman:
            features['left_territory'] = 1
        
        # Detect looping behavior - penalize repeated positions
        prev_pos = game_state.get_agent_state(self.index).get_position()
        if hasattr(self, 'position_history'):
            self.position_history.append(prev_pos)
            # Keep only last 15 positions
            if len(self.position_history) > 15:
                self.position_history.pop(0)
            
            # Check for loops - if current position appears multiple times recently
            if len(self.position_history) >= 4:
                position_counts = {}
                for pos in self.position_history[-4:]:
                    position_counts[pos] = position_counts.get(pos, 0) + 1
                
                # If we've been at my_pos multiple times in recent history
                if my_pos in position_counts and position_counts[my_pos] >= 2:
                    features['looping'] = position_counts[my_pos]
        else:
            self.position_history = [prev_pos]

        return features

    def get_defensive_weights(self, game_state, action):
        """Dynamic weights based on whether agent is scared or not"""
    
        return {
            'num_invaders': -1000,      # Eliminate invaders (top priority)
            'invader_distance': -950,    # Chase them down aggressively
            'on_defense': 100,          # Stay on defensive side
            'blocking_distance': -300,   # Get to blocking position (increased importance)
            'defend_capsule': -500,      # Guard capsules when threatened
            'patrol_distance': -1,       # Patrol when no threats (minimal weight)
            'stop': 0,                   # No penalty - holding position can be strategic
            'reverse': 0,                # No penalty - Y-axis movement needs flexibility
            'looping': -5,              # Still avoid stuck patterns
            'left_territory': -500       # Strong penalty for leaving territory
        }
    
    def get_patrol_point(self, game_state):
        """Calculate optimal patrol position - balancing food density, mobility, and tactical advantage"""
        if self.patrol_point is None:
            food_defending = self.get_food_you_are_defending(game_state).as_list()
            capsules_defending = self.get_capsules_you_are_defending(game_state)
            walls = game_state.get_walls()
            border_x = self.home_border_x(game_state)
            height = walls.height
            width = walls.width
            
            if len(food_defending) > 0:
                # Generate candidate patrol positions in a ZONE near the border (not just the line)
                # Search 1-4 positions into our territory for better mobility
                candidates = []
                
                # Define patrol zone depth based on team
                if self.red:
                    # Red defends left side, search positions to the left of border
                    x_range = range(max(0, border_x - 4), border_x + 1)
                else:
                    # Blue defends right side, search positions to the right of border
                    x_range = range(border_x, min(width, border_x + 5))
                
                # Sample positions in the patrol zone
                for x in x_range:
                    for y in range(1, height - 1, 2):  # Sample every 2 rows for efficiency
                        pos = (x, y)
                        if walls[pos[0]][pos[1]]:
                            continue
                        
                        # Calculate multiple factors for this position
                        
                        # 1. Food density score - how much food is nearby
                        food_density = 0
                        for food_pos in food_defending:
                            dist = abs(food_pos[0] - pos[0]) + abs(food_pos[1] - pos[1])
                            if dist <= 6:  # Within 6 Manhattan distance
                                food_density += 1.0 / (dist + 1)  # Closer food = higher weight
                        
                        # 2. Mobility score - how open is the area (MOST IMPORTANT)
                        # Use local_mobility to measure openness (lower penalty = better mobility)
                        mobility_penalty = self.local_mobility(game_state, pos, pos, max_depth=6)
                        mobility_score = (6 * 9 - mobility_penalty) / (6 * 9)  # Normalize to 0-1
                        
                        # 3. Neighbor accessibility - count non-wall neighbors
                        neighbor_count = 0
                        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nx, ny = pos[0] + dx, pos[1] + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if not walls[nx][ny]:
                                    neighbor_count += 1
                        accessibility = neighbor_count / 4.0
                        
                        # 4. Proximity to border - closer to border is better for interception
                        # But not a hard requirement (allows flexibility into territory)
                        border_distance = abs(x - border_x)
                        border_proximity = 1.0 / (border_distance + 1)  # Inverse: closer = higher score
                        
                        # 5. Centrality - prefer positions in the middle vertical area
                        centrality = 1.0 - abs(y - height / 2) / (height / 2)
                        
                        # 6. Capsule coverage - closer to capsules if they exist
                        capsule_coverage = 0
                        if capsules_defending:
                            min_cap_dist = min([self.get_maze_distance(pos, cap) for cap in capsules_defending])
                            capsule_coverage = 1.0 / (min_cap_dist + 1)
                        
                        # 7. Interception potential - can we reach the border quickly from here?
                        # Check if there's a clear path to border
                        border_reach_score = 0
                        if border_distance <= 3:
                            # Good interception position
                            border_reach_score = 1.0 - (border_distance / 3.0)
                        
                        # Combined score with weights
                        # Heavily prioritize mobility and accessibility
                        combined_score = (
                            mobility_score * 5.0 +          # HIGHEST: open space is critical
                            accessibility * 5.0 +           # Multiple exit routes important
                            food_density * 2.0 +            # Still care about food
                            border_proximity * 1.2 +        # Prefer being reasonably close to border
                            border_reach_score * 2.0 +      # Can we intercept quickly?
                            centrality * 1.0 +              # Central positions help response time
                            capsule_coverage * 2.5          # Protect capsules if present
                        )
                        
                        candidates.append((combined_score, pos, mobility_score, x))
                
                if candidates:
                    # Select the position with the highest combined score
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    self.patrol_point = candidates[0][1]
                    
                    # Debug: uncomment to see what was chosen
                    # best = candidates[0]
                    # print(f"Patrol point: {best[1]}, Score: {best[0]:.2f}, Mobility: {best[2]:.2f}, X-offset from border: {abs(best[3] - border_x)}")
                else:
                    # Fallback: center of border
                    self.patrol_point = (border_x, height // 2)
            else:
                # No food left, patrol at border center
                self.patrol_point = (border_x, height // 2)
        
        return self.patrol_point

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        
        # If scared - defense switch to offense
        my_scared_timer = game_state.get_agent_state(self.index).scared_timer
        if not self.offense and my_scared_timer > 5:
            print(f"Agent {self.index}: Scared! Switching to attack mode for {my_scared_timer} moves")
            self.switch = max(self.switch, my_scared_timer)
        
        # Determine which strategy to use based on role and switch state
        # Offensive agent: use offense unless switch > 0 (then defend)
        # Defensive agent: use defense unless switch > 0 (then attack)
        use_offensive = (self.offense and not self.switch > 0) or (not self.offense and self.switch > 0)      
        if use_offensive:
            results = {a: self.offensive(game_state, a) for a in actions}
            values = {a: results[a][0] for a in actions}
        else:
            features_dict = {a: self.get_defensive_features(game_state, a) for a in actions}
            weights_dict = {a: self.get_defensive_weights(game_state, a) for a in actions}
            # Use weights for feature weight evaluation
            values = {a: features_dict[a] * weights_dict[a] for a in actions}
            results = {a: (values[a], 0) for a in actions}

        max_value = max(values.values())
        best_actions = [a for a in actions if values[a] == max_value]
        chosen_action = random.choice(best_actions)    
        
        
        if use_offensive:
            # Unpack the offense values and various potential timers
            # 0 = evaluated value | 1 = should we flee ? (Negative if yes) | 2 = Were we chased ? (1 = yes)
            # 3 = Flee timer | 4 = Should we defend for a while ? (1 = yes) | 5 = Are both enemies ghosts?
            chosen_safety_bool = results[chosen_action][1]
            chase_bool = results[chosen_action][2]
            self.flee_timer += results[chosen_action][3]
            side_reset = results[chosen_action][4]
            two_ghoster = results[chosen_action][5]
            if two_ghoster:
                # Timer for niche cases in which we consider special moves such as going for the pellet
                self.two_ghost = 5
            else:
                if self.two_ghost > 0:
                    self.two_ghost -= 1 
            if side_reset:
                self.switch += 20
                print("decided to defend")
            self.chase_timer += chase_bool
            if self.chase_timer == 3 and chase_bool == 1:
                print("chased -> flee")
                if self.chase_timer == 3 and chase_bool == 1 and game_state.get_agent_state(self.index).scared_timer > 0:
                    print("not fleeing yet - scared")
            if chosen_safety_bool < 0 and self.flee_timer < 2:
                self.flee_timer = 10
                print("decided to flee because food")
        
        # Control switch timer for both offensive and defensive agents
        if self.switch > 0:
            self.switch -= 1
            if self.switch == 0:
                if self.offense:
                    print(f"[SWITCH] Agent {self.index}: Back to attacking (offense={self.offense})")
                else:
                    print(f"[SWITCH] Agent {self.index}: Back to defending (offense={self.offense})")
            elif self.switch <= 3:
                print(f"[SWITCH] Agent {self.index}: Switch expires in {self.switch} turns (offense={self.offense})")
            elif self.switch <= 3:
                print(f"[SWITCH] Agent {self.index}: Switch expiring soon ({self.switch} turns left)")
        
        # Recent position queue handling - used to avoid jitter movement
        cur_pos = game_state.get_agent_state(self.index).get_position()
        food_carry = game_state.get_agent_state(self.index).num_carrying
        if food_carry != self.carrying:
            # Reset recent positions if we collected food -> tolerate repeat movement briefly
            self.carrying = food_carry
            self.recent_positions = []
        if len(self.recent_positions) > 6:
            self.recent_positions.pop(0)
        self.recent_positions.append(cur_pos)

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
