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
    # Add pillseeker and according attackphase for both
    # Defensive and attacker crossover for according points
    # Defensive
    # No pillseeker behavious on either side yet.

    # Add safety for no food (both sides) and for no enemy on each field

    def __init__(self, index, attacker, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.offense = attacker
        self.start = None
        self.flee_timer = 0 # How long should we flee
        self.chase_timer = 0 # How long have we been chased
        self.switch = 0 # How long should we switch attacker defender role
        self.recent_positions = []
        self.carrying = 0
        self.two_ghost = 0

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.distancer.get_maze_distances()
        self.start = game_state.get_agent_position(self.index)

    def local_mobility(self, game_state, old_pos, start_pos, max_depth=5):
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
            depth_penalty = (len(visited) - 1) * 9#(max_reached_depth)*9
            return depth_penalty
        return 0

    def ghost_distance(self, game_state, multiple=False):
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
        width = game_state.get_walls().width
        if self.red:
            return width // 2 - 1
        else:
            return width // 2

    def food_risk(self, pos, game_state):
        foods = self.get_food(game_state).as_list()
        team_foods = self.get_food_you_are_defending(game_state).as_list()
        # x,y = pos
        # mid_x = game_state.data.layout.width // 2
        if not foods:
            return 999 # we want to go home
        food_enemies = len(foods)
        food_team = len(team_foods)
        if food_team == 0:
            food_team = 1
        factor_1 = game_state.get_agent_state(self.index).num_carrying # ~0-20
        imbalance = max(0,(food_team - food_enemies)) + 30/food_team
        imbalance = factor_1 * 18/food_team 
        flee_val = 1
        if imbalance > 10:
            # if self.red:
            #     factor_2 = x
            # else:
            #     factor_2 = game_state.data.layout.width - x
            imbalance = -min([self.get_maze_distance(pos, food) for food in team_foods])
            # Use maze distance and add timer
            imbalance = imbalance #* factor_2
            flee_val = -1
        return imbalance, flee_val # balanced - imbalanced | ~0 - 2
    
    def on_home_ground(self, game_state):
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

    def get_enemy_scared_times(self, game_state):
        times = {}
        for enemy in self.get_opponents(game_state):
            st = game_state.get_agent_state(enemy)
            times[enemy] = getattr(st, "scaredTimer", getattr(st, "scared_timer", 0))
        return times

    def border_bias(self, game_state):
        # should return bias to stay on border -> enable easier trapping
        # penalty for very close to border
        x, y = game_state.get_agent_state(self.index).get_position()
        borderx = self.home_border_x(game_state)
        value = 0
        if self.red:
            if x >= borderx -2: #14 16
                value = abs(x-(borderx-2)) * 4
        else:
            if x <= borderx +2:
                value = abs(x-(borderx+2)) * 4
        #consider multiple invaders - take min
        return value
    
    def invader_distance(self, game_state):
        return 0
    
    def trap_factor(self, game_state, e_index):
        # should return how trapped the enemy is
        return 0
    
    def closest_ghost_position(self, game_state, invasion=False):
        enemies = self.get_opponents(game_state)
        my_pos = game_state.get_agent_state(self.index).get_position()
        my_pos = (int(my_pos[0]), int(my_pos[1]))
        noisy = game_state.get_agent_distances()
        walls = game_state.get_walls()
        #print(walls)
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
        #print(f"print 1 {best_pos} {my_pos}")
        for index in enemies:
            s = game_state.get_agent_state(index)
            # invasion=True  → only track invaders (enemy Pacman)
            # invasion=False → only track ghosts
            if invasion:
                if not s.is_pacman:
                    continue  # skip ghosts
            else:
                if s.is_pacman:
                    continue  # skip invaders

            real_pos = game_state.get_agent_position(index)
            # case 1: exact known position
            if real_pos is not None:
                dist = self.get_maze_distance(my_pos, real_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = real_pos
                continue

            # case 2: noisy sensor reading
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
        #print(f"print 2{best_pos} {my_pos}")
        # x, y = best_pos
        # x = min(max(int(x), 0), width - 1)
        # y = min(max(int(y), 0), height - 1)
        # if not walls[x][y]:
        #     best_pos = (x, y)

        return best_pos

    def ghost_positions(self,game_state):
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
        # add y reset entry after defense recovery switch (urgent)
        # add pill search consideration in chase scenario (urgent)
        # urgent fix d_evaluation - replace with closest field dist?
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
        min_distance = min([self.get_maze_distance(my_pos, food) for food in foods])
        old_min_distance = min([self.get_maze_distance(old_pos, food) for food in foods])
        team_foods = self.get_food_you_are_defending(game_state).as_list()
        # Closest food distance for food of own team
        min_team_distance = min([self.get_maze_distance(my_pos, food) for food in team_foods])
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
            #print("ignore enemies")
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
            #if old_min_distance > min_distance:
                if ghost_dist != 0: # change -> catch 0 case
                    #min_distance += 1
                    # changed to dead end penalty from ghost dist
                    c_evaluation -= ((2*(dead_end_penalty/9) + 1)/ (ghost_dist + 0.5)) * 2 # scales between 0 and smaller values than -2

        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if not invaders:
            two_ghoster = True
        else:
            two_ghoster = False

        special_movement = False
        best_target = None
        if self.flee_timer > 0 or (self.chase_timer > 3 and e_scared_time < 2):
            if (self.chase_timer > 3 and e_scared_time < 2):
                #print("considering special move 1")
                capsules = self.get_capsules(game_state)
                if capsules:
                    #print("considering special move")
                    if not invaders:
                        #print("no invaders - proceed")
                        ghost_positions = self.ghost_positions(game_state)

                        # gather border cells
                        border_x = self.home_border_x(game_state)
                        walls = game_state.get_walls()
                        height = walls.height
                        border_cells = [(border_x, y) for y in range(height) if not walls[border_x][y]]

                        # target candidates
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
        # if not in special escape mode → normal pincer avoidance
        if not special_movement:
            c_evaluation -= sum(10 / ((d + 1) ** 2) for d in distances) * ((dead_end_penalty + 1) ** 0.5)

        if self.red:
            d_evaluation = -x#-min_team_distance
        else:
            walls = game_state.get_walls()
            width = walls.width
            d_evaluation = -(width - x)
        e_evaluation = -150 if (abs(old_pos[0] - my_pos[0]) + abs(old_pos[1] - my_pos[1])) > 2 else 0
        if ghost_dist > 2:  
            e_evaluation += cycle_penalty
        if ghost_dist_f < 2 and scared_enemy == 1: 
            e_evaluation -= 75
        evaluation = a_evaluation + c_evaluation * scared_enemy + e_evaluation
        flee = 0

        # Debugging
        # if e_evaluation < 0:
        #     print(f"{scared_enemy} {c_evaluation} {e_evaluation} {old_pos} {my_pos} {action}")
        # if e_scared_time > 3:
        #     safety_bool = 1
        #if ghost_dist == 1 and action != "Stop" or (abs(old_pos[0] - my_pos[0]) + abs(old_pos[1] - my_pos[1])) > 2:
        #if ghost_dist <= 3:
        #if special_movement:
        #print(f"{str(scared_enemy):>6} "f"{a_evaluation:>7.2f} "f"{c_evaluation:>7.2f} "f"{e_evaluation:>7.2f} "f"{dead_end_penalty:>6.2f} "f"{ghost_dist:>6.2f} "f"{ghost_dist_f:>6.2f} "f"{str(action):>10} "f"{evaluation:>7.2f} "f"{str(my_pos):>12} "f"{str(old_pos):>12} "f"{cycle_penalty:>6.2f} "f"{str(special_movement):>6} " f"{distances}")

        if self.flee_timer > 0: # We have already decided to flee - execute
            # This is decided by the food risk tradeoff - roughly - if we carry a lot we flee.
            flee =  - 1
            if self.on_home_ground(game_state):
                flee = -self.flee_timer # reset flee behaviour
            #safety_bool = 1
            evaluation = b_evaluation + c_evaluation * scared_enemy + e_evaluation

        # Value settings for the chase scenario
        carrying = game_state.get_agent_state(self.index).num_carrying
        if (self.chase_timer > 3 and scared_time < 1 and e_scared_time < 2) or (self.chase_timer > 3 and e_scared_time < 2 and carrying > 2): 
            #print("homebound")
            evaluation = d_evaluation + c_evaluation * scared_enemy + e_evaluation * (carrying + 1)
            if self.on_home_ground(game_state):
                chase_bool = -self.chase_timer
                side_reset = True
        #print(f"{a_evaluation:>7.2f} "f"{b_evaluation:>5.1f} "f"{d_evaluation:>7.2f} "f"{c_evaluation:>7.2f} "f"{e_evaluation:>7.2f} "f"{ghost_dist:>6.2f} "f"{ghost_dist_f:>6.2f} "f"{str(action):>10} "f"{evaluation:>7.2f} "f"{str(my_pos):>12} "f"{str(old_pos):>12} "f"{cycle_penalty:>6.2f} "f"{str(special_movement):>6} " f"{distances}"f"{best_target}")
        #print(f"{a_evaluation:>7.2f} "f"{d_evaluation:>7.2f} "f"{c_evaluation:>7.2f} "f"{e_evaluation:>7.2f} "f"{dead_end_penalty:>6.2f} "f"{ghost_dist:>6.2f} "f"{str(action):>7} "f"{evaluation:>7.2f} "f"{str(my_pos):>10} "f"{str(old_pos):>10} "f"{cycle_penalty:>6.2f} "f"{str(special_movement):>6} " f"{distances}")
        # Add fix for ghost on heels - pill search or reset to defense for about 10+ turns (how to ensure this? - use min)
        return evaluation, safety_bool, chase_bool, flee, side_reset, two_ghoster  # call defensive
    
    def defensive(self, game_state, action):
        # integrate switch for scared logic
        # consider attacker - when close do not decrease self.offense
        # dont add stop prevention!
        # add eat reward
        # reward if the enemy is chased into depth
        # make border stop nasty - stop direct standoff
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        enemies_old = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders_old = [e for e in enemies_old if e.is_pacman and e.get_position() is not None]
        my_pos = successor.get_agent_state(self.index).get_position()
        eat_reward = 0
        if len(invaders_old) > 0 and len(invaders) == 0:
            # your agent must have eaten the invader
            eat_reward = 200
        if not invaders: # list empty
            team_foods = self.get_food_you_are_defending(game_state).as_list()
            min_team_distance = [self.get_maze_distance(my_pos, food) for food in team_foods]
            min_team_distance = sum(min_team_distance) / max(len(min_team_distance),1)
            if self.on_home_ground(successor):
                invad_pos = self.closest_ghost_position(game_state, False)
                #print(invad_pos)
                #print(f"print 3 {my_pos} {invad_pos}")
                evaluation_a = -self.get_maze_distance(my_pos, invad_pos) - min_team_distance * 0.8
                evaluation_c = -self.border_bias(successor)
            # elif self.switch > 0:
            #     #??? 
            else:
                evaluation_a = -100 # change to potential attack?
                evaluation_c = 0
            evaluation = evaluation_a + eat_reward + evaluation_c
        else:
            #print("b_eval")
            invad_pos = self.closest_ghost_position(game_state, True)
            evaluation_b = -self.get_maze_distance(my_pos, invad_pos)
            evaluation = evaluation_b + eat_reward # chase & cutoff
        return evaluation, 0 # evaluation, various values



    def choose_action(self, game_state):
        #print(game_state) % walls, . = food, G Ghost, o pill, 
        #integrate defense logic
        actions = game_state.get_legal_actions(self.index)
        #print(actions) ['North', 'South', 'Stop']
        
        # Check if we're scared (opponent ate capsule) - defensive agent should attack
        my_scared_timer = game_state.get_agent_state(self.index).scared_timer
        if not self.offense and my_scared_timer > 5:
            # Temporarily switch to offense while scared
            print(f"Agent {self.index}: Scared! Switching to attack mode for {my_scared_timer} moves")
            self.switch = max(self.switch, my_scared_timer)
        
        # Determine which strategy to use based on role and switch state
        # Offensive agent: use offense unless switch > 0 (then defend)
        # Defensive agent: use defense unless switch > 0 (then attack)
        use_offensive = (self.offense and not self.switch > 0) or (not self.offense and self.switch > 0)
        
        if use_offensive:
            # Use offensive logic
            results = {a: self.offensive(game_state, a) for a in actions}
            values = {a: results[a][0] for a in actions}
        else:
            # Use defensive logic
            features_dict = {a: self.get_defensive_features(game_state, a) for a in actions}
            weights_dict = {a: self.get_defensive_weights(game_state, a) for a in actions}
            # Recalculate values using feature-weight evaluation
            values = {a: features_dict[a] * weights_dict[a] for a in actions}
            results = {a: (values[a], 0) for a in actions}

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        max_value = max(values.values())
        best_actions = [a for a in actions if values[a] == max_value]
        chosen_action = random.choice(best_actions)    
        # Handle offensive agent timers and state updates
        if use_offensive:
            chosen_safety_bool = results[chosen_action][1] # Is continuing offense risky? negative if yes
            chase_bool = results[chosen_action][2] # chase_bool
            self.flee_timer += results[chosen_action][3] # How long should we flee
            side_reset = results[chosen_action][4] # Should we stay on our side for some time?
            two_ghoster = results[chosen_action][5]
            if two_ghoster:
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
        
        # Decrement switch timer for both offensive and defensive agents
        if self.switch > 0:
            self.switch -= 1
            if self.switch == 0:
                if self.offense:
                    print("Back to attacking")
                else:
                    print("Back to defending")
        food_left = len(self.get_food(game_state).as_list())
        # if self.chase_timer > 0:
        #     print(best_actions)
        if food_left <= 2:
            best_dist = 9999 # improve logic or switch to defense
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        cur_pos = game_state.get_agent_state(self.index).get_position()
        food_carry = game_state.get_agent_state(self.index).num_carrying
        if food_carry != self.carrying:
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
