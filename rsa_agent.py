import numpy as np
import itertools
import time
import os
import collections

try:
    from env import GridEnvironment
except ImportError:
    print("Error: Could not import GridEnvironment.")
    print("Please ensure the updated GridEnvironment class is in 'env.py'.")
    exit()


# --- ANSI Color Constants for Rendering ---
COLOR_RESET = "\x1b[0m"

# Define the number of states to sample when the total number of possibilities is too large.
NUM_STATE_SAMPLES = 1000
VIEW_SIZE = 5

def _chebyshev_distance(pos1, pos2):
    """Calculates Chebyshev distance (for grid with diagonal moves)."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

def _get_char_for_prob(prob):
    """
    Maps a wall probability (0.0 to 1.0) to a continuous color gradient string.
    Low probability (empty space) is black, high probability (wall) is white.
    """
    # For cells that are almost certainly empty, display a black dot.
    if prob < 0.1:
        # The ANSI code for black is 232.
        return f"\x1b[38;5;232m . {COLOR_RESET}"

    # Map the probability to the 24-shade grayscale ramp (232-255).
    # A low 'prob' maps to a darker color, a high 'prob' maps to a lighter one.
    gray_index = 232 + round(prob * 23)

    # Dynamically create the ANSI color code for the calculated gray shade.
    color_code = f"\x1b[38;5;{gray_index}m"
    block = "███"

    return f"{color_code}{block}{COLOR_RESET}"

def _render_belief_map_with_chars(belief_map, grid_size, agent_pos, target_pos):
    """Renders a belief map using 3-character strings to match the environment style."""
    grid_str = ""
    for r in range(grid_size):
        for c in range(grid_size):
            pos = (r, c)
            if pos == tuple(agent_pos):
                grid_str += ' A '  # 3 characters for alignment
            elif pos == tuple(target_pos):
                grid_str += ' T '  # 3 characters for alignment
            else:
                prob = belief_map[r, c]
                grid_str += _get_char_for_prob(prob)
        grid_str += "\n"
    print(grid_str)

def _find_shortest_path_bfs(start_pos, end_pos, state_grid):
    """
    Finds the shortest path length between two points on the 5x5 grid using BFS.
    Returns path length or np.inf if no path exists.
    """
    if start_pos == end_pos:
        return 0

    q = collections.deque([(start_pos, 0)])  # (position, cost)
    visited = {start_pos}
    
    # All 8 possible moves
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while q:
        (r, c), cost = q.popleft()

        for dr, dc in moves:
            nr, nc = r + dr, c + dc

            if (nr, nc) == end_pos:
                return cost + 1

            if (0 <= nr < VIEW_SIZE and 0 <= nc < VIEW_SIZE and
                    (nr, nc) not in visited and state_grid[nr, nc] != 1): # 1 is _wall_cell
                visited.add((nr, nc))
                q.append(((nr, nc), cost + 1))
    
    return np.inf

def _calculate_heuristic_utilities(agent_pos, target_pos, states, env, k_candidates=3):
    """
    Calculates action utilities using a planning heuristic. It identifies the top k
    cells on the edge of the 5x5 view, finds the shortest path to them, and
    calculates a utility based on the endpoint value and path cost.
    """
    num_states = len(states)
    num_actions = env.action_space.n
    all_utilities = np.zeros((num_states, num_actions))
    view_radius = VIEW_SIZE // 2
    agent_local_pos = (view_radius, view_radius)

    action_moves = {
        0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
        4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)
    }

    for i, state in enumerate(states):
        # 1. Identify all non-wall cells on the edge of the 5x5 view
        edge_cells = []
        for r_loc in range(VIEW_SIZE):
            for c_loc in range(VIEW_SIZE):
                if r_loc == 0 or r_loc == VIEW_SIZE - 1 or c_loc == 0 or c_loc == VIEW_SIZE - 1:
                    if state[r_loc, c_loc] != env._wall_cell:
                        edge_cells.append((r_loc, c_loc))

        # 2. Select top k candidate cells based on their distance to the global target
        if not edge_cells:
            all_utilities[i, :] = -np.inf  # No valid edges to plan towards
            continue

        candidate_endpoints = sorted(edge_cells, key=lambda pos: _chebyshev_distance(
            (agent_pos[0] + pos[0] - view_radius, agent_pos[1] + pos[1] - view_radius),
            target_pos
        ))[:k_candidates]

        action_utilities = []
        for action_idx in range(num_actions):
            move = action_moves[action_idx]
            start_pos = (agent_local_pos[0] + move[0], agent_local_pos[1] + move[1])

            # Immediate move into a wall is invalid
            if state[start_pos] == env._wall_cell:
                action_utilities.append(-np.inf)
                continue
            
            # 3. For each candidate, compute path and combine values
            path_values = []
            for end_pos in candidate_endpoints:
                path_cost = _find_shortest_path_bfs(start_pos, end_pos, state)

                if path_cost != np.inf:
                    # Convert local endpoint to global position
                    global_end_pos = (
                        agent_pos[0] + end_pos[0] - view_radius,
                        agent_pos[1] + end_pos[1] - view_radius
                    )
                    
                    # Value of the endpoint is higher if it's closer to the global target
                    endpoint_value = (env.grid_size * 2) - _chebyshev_distance(global_end_pos, target_pos)
                    
                    # Combine endpoint value with the cost to get there
                    combined_value = endpoint_value - path_cost
                    path_values.append(combined_value)

            # The utility of the action is the best value it can achieve among all candidates
            if not path_values:
                action_utilities.append(-np.inf) # Can't reach any candidate from this move
            else:
                action_utilities.append(max(path_values))
        
        all_utilities[i, :] = np.array(action_utilities, dtype=float)

    # Note: Normalization to "probabilities" is handled by the RSA reasoning functions
    # that call this utility function. We return the raw utilities here.
    return all_utilities
        

def _generate_possible_states(agent_pos, target_pos, belief_map, env, true_state_view=None, use_intelligent_sampling=False):
    """
    Generates possible 5x5 local states based on a belief map.
    If the number of possibilities is too high, it samples them.
    Sampling can be uniform or weighted (intelligent).
    """
    base_state = np.full((VIEW_SIZE, VIEW_SIZE), env._empty_cell)
    uncertain_cells = []
    view_radius = VIEW_SIZE // 2

    for r_local in range(VIEW_SIZE):
        for c_local in range(VIEW_SIZE):
            r_global = agent_pos[0] + r_local - view_radius
            c_global = agent_pos[1] + c_local - view_radius
            global_pos = (r_global, c_global)

            if global_pos == agent_pos:
                base_state[r_local, c_local] = env._agent_cell if agent_pos != target_pos else env._target_cell
                continue

            if global_pos == target_pos:
                base_state[r_local, c_local] = env._target_cell
                continue

            if not (0 <= r_global < env.grid_size and 0 <= c_global < env.grid_size):
                base_state[r_local, c_local] = env._wall_cell
            else:
                prob_wall = belief_map[r_global, c_global]
                if prob_wall == 1.0:
                    base_state[r_local, c_local] = env._wall_cell
                elif prob_wall == 0.0:
                    base_state[r_local, c_local] = env._empty_cell
                else:
                    uncertain_cells.append(((r_local, c_local), prob_wall))

    possible_states = []
    num_uncertain = len(uncertain_cells)

    if 0 < num_uncertain <= 10:
        coords = [cell[0] for cell in uncertain_cells]
        for combo in itertools.product([env._empty_cell, env._wall_cell], repeat=num_uncertain):
            new_state = base_state.copy()
            for i in range(num_uncertain):
                new_state[coords[i]] = combo[i]
            possible_states.append(new_state)

    elif num_uncertain > 10:
        for _ in range(NUM_STATE_SAMPLES):
            new_state = base_state.copy()
            for (r_loc, c_loc), prob_wall in uncertain_cells:
                if use_intelligent_sampling:
                    cell_type = np.random.choice([env._wall_cell, env._empty_cell], p=[prob_wall, 1 - prob_wall])
                else:
                    cell_type = np.random.choice([env._empty_cell, env._wall_cell])
                new_state[r_loc, c_loc] = cell_type
            possible_states.append(new_state)
    else:
        possible_states.append(base_state)

    if true_state_view is not None:
        if not any(np.array_equal(s, true_state_view) for s in possible_states):
            possible_states.append(true_state_view)

    return possible_states

def render_side_by_side_views(wall_probabilities, ground_truth_view, env):
    """
    Renders the observer's inferred wall probabilities next to the
    agent's actual 5x5 ground truth view for comparison.
    """
    header_left = "Observer's Inference"
    header_right = "Agent's Ground Truth"
    print(f"\n{header_left:^24}   |   {header_right:^24}")
    print(f"{'-'*24:^24}   |   {'-'*24:^24}")

    # Map for rendering the ground truth characters
    truth_char_map = {
        env._empty_cell: '.',
        env._wall_cell: '#',
        env._agent_cell: 'A',
        env._target_cell: 'T'
    }

    for r in range(VIEW_SIZE):
        # Build the left side (probabilities)
        left_row_str = ""
        for c in range(VIEW_SIZE):
            prob = wall_probabilities[r, c]
            left_row_str += f" {int(prob * 100):>3d}% "

        # Build the right side (ground truth)
        right_row_str = ""
        for c in range(VIEW_SIZE):
            cell_val = ground_truth_view[r, c]
            char = truth_char_map.get(cell_val, '?')
            right_row_str += f"  {char}   "
        
        print(f"{left_row_str:^24}   |   {right_row_str:^24}")
    print()


class RSAAgent:
    """
    An agent that uses its private 'internal_belief_map' to decide on an action.
    """
    def __init__(self, env: GridEnvironment, rsa_iterations: int = 10, rationality: float = 10, utility_beta: float = 10, initial_prob: float = 0.3):
        self.env = env
        self.rsa_iterations = rsa_iterations
        self.alpha = rationality
        self.beta = utility_beta
        self.num_actions = env.action_space.n
        self.internal_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        

        if env.agent_pos is not None:
            self.internal_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.internal_belief_map[env.target_pos] = 0.0

    def choose_action(self, observation):
        agent_pos = tuple(observation['agent_pos'])
        target_pos = tuple(observation['target_pos'])
        s_true = observation['local_view']

        possible_states = _generate_possible_states(
            agent_pos, target_pos, self.internal_belief_map, self.env, true_state_view=s_true
        )

        s_true_idx = -1
        for i, state in enumerate(possible_states):
            if np.array_equal(state, s_true):
                s_true_idx = i
                break
        
        assert s_true_idx != -1, "True state not found in generated states."

        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env)

        # This is the agent's reasoning based on its private information
        P_S_k = self._run_rsa_reasoning(world_utilities, len(possible_states))

        final_action_probs = P_S_k[s_true_idx, :]
        max_prob = np.max(final_action_probs)
        best_action_indices = np.where(final_action_probs == max_prob)[0]
        action = np.random.choice(best_action_indices)

        return int(action), final_action_probs

    def _run_rsa_reasoning(self, world_utilities, num_states):
        """Performs the RSA iterative calculation."""
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.beta * world_utilities)
            P_S0 = np.nan_to_num(exp_utilities / exp_utilities.sum(axis=1, keepdims=True))

        P_S_k = P_S0
        for _ in range(self.rsa_iterations):
            with np.errstate(divide='ignore', invalid='ignore'):
                P_S_k = np.nan_to_num(np.exp(world_utilities) / np.exp(world_utilities).sum(axis=1, keepdims=True))
        return P_S_k

    def update_internal_belief(self, observation):
        """Updates the agent's private belief map with ground truth from its 5x5 view."""
        local_view = observation['local_view']
        agent_pos = observation['agent_pos']
        view_radius = VIEW_SIZE // 2

        for r_local in range(VIEW_SIZE):
            for c_local in range(VIEW_SIZE):
                r_global = agent_pos[0] + r_local - view_radius
                c_global = agent_pos[1] + c_local - view_radius
                if 0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size:
                    is_wall = (local_view[r_local, c_local] == self.env._wall_cell)
                    self.internal_belief_map[r_global, c_global] = 1.0 if is_wall else 0.0

    def render_internal_belief(self, agent_pos, target_pos):
        """Renders the agent's internal belief map using a color gradient."""
        _render_belief_map_with_chars(self.internal_belief_map, self.env.grid_size, agent_pos, target_pos)


class Observer:
    """
    An observer that only sees the agent's position and actions.
    It maintains its own belief map and updates it by inverting the agent's RSA model.
    """
    def __init__(self, env: GridEnvironment, agent_params: dict, initial_prob: float = 0.25, learning_rate: float = 0.25, intelligent_sampling: bool = False):
        self.env = env
        self.beta = agent_params.get('beta', 10)
        self.rsa_iterations = agent_params.get('rsa_iterations', 3)
        self.learning_rate = learning_rate
        self.intelligent_sampling = intelligent_sampling
        self.observer_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.view_radius = VIEW_SIZE // 2

        if env.agent_pos is not None:
            self.observer_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.observer_belief_map[env.target_pos] = 0.0

    def _run_rsa_reasoning_for_observer(self, world_utilities, num_states):
        """Performs the RSA iterative calculation."""
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.beta * world_utilities)
            P_S0 = np.nan_to_num(exp_utilities / exp_utilities.sum(axis=1, keepdims=True))

        P_S_k = P_S0
        for _ in range(self.rsa_iterations):
            with np.errstate(divide='ignore', invalid='ignore'):
                P_S_k = np.nan_to_num(np.exp(world_utilities) / np.exp(world_utilities).sum(axis=1, keepdims=True))
        return P_S_k
    
    def _compute_local_belief(self, agent_pos, target_pos, action) -> np.ndarray:
        """Computes the inferred 5x5 local probability map based on the agent's action."""
        possible_states = _generate_possible_states(
            agent_pos, target_pos, self.observer_belief_map, self.env,
            use_intelligent_sampling=self.intelligent_sampling
        )
        local_wall_probs = np.zeros((VIEW_SIZE, VIEW_SIZE))

        if not possible_states:
            return local_wall_probs

        num_local_states = len(possible_states)
        prior = np.full(num_local_states, 1.0 / num_local_states)

        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env)
        P_S_k = self._run_rsa_reasoning_for_observer(world_utilities, num_local_states)

        with np.errstate(divide='ignore', invalid='ignore'):
            L_numerator = P_S_k.T * prior
            P_L_k = np.nan_to_num(L_numerator / L_numerator.sum(axis=1, keepdims=True))

        state_posterior = P_L_k[action, :]

        for r_local in range(VIEW_SIZE):
            for c_local in range(VIEW_SIZE):
                prob_wall = sum(state_posterior[i] for i, s in enumerate(possible_states) if s[r_local, c_local] == self.env._wall_cell)
                local_wall_probs[r_local, c_local] = prob_wall
        
        return local_wall_probs
    
    def _update_global_belief(self, local_wall_probs, agent_pos):
        """Updates the global belief map using the newly inferred local beliefs."""
        self.observer_belief_map[agent_pos] = 0.0
        

        for r_local in range(VIEW_SIZE):
            for c_local in range(VIEW_SIZE):
                if r_local == self.view_radius and c_local == self.view_radius:
                    continue

                r_global = agent_pos[0] + r_local - self.view_radius
                c_global = agent_pos[1] + c_local - self.view_radius

                if 0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size:
                    local_prob = local_wall_probs[r_local, c_local]
                    global_prob = self.observer_belief_map[r_global, c_global]
                    
                    new_global_prob = global_prob + self.learning_rate * (local_prob - global_prob)
                    self.observer_belief_map[r_global, c_global] = np.clip(new_global_prob, 0.0, 1.0)

    def update_belief(self, agent_pos, target_pos, action) -> np.ndarray:
        """Orchestrates the two-step belief update process."""
        local_probs = self._compute_local_belief(agent_pos, target_pos, action)
        self._update_global_belief(local_probs, agent_pos)
        return local_probs

    def render_belief(self, agent_pos, target_pos):
        """Renders the observer's belief map using a color gradient."""
        _render_belief_map_with_chars(self.observer_belief_map, self.env.grid_size, agent_pos, target_pos)


# --- Main execution block ---
if __name__ == '__main__':
    try:
        custom_map = [
            "############",
            "#          #",
            "#  # T     #",
            "#  ####    #",
            "#  #       #",
            "#      ### #",
            "#  #       #",
            "#  #   #   #",
            "# #### ##  #",
            "#   A      #",
            "#          #",
            "############",
        ]

        env = GridEnvironment(grid_map=custom_map, render_mode='human')
        obs, info = env.reset()

        # Calculate the true probability of a wall in the given map
        num_walls = sum(row.count('#') for row in custom_map)
        total_cells = env.grid_size * env.grid_size
        #true_wall_prob = num_walls / total_cells
        true_wall_prob = 0.1  # For testing, we set a fixed probability

        # By default, intelligent_sampling is False.
        # To enable, set intelligent_sampling=True in the constructors below.

        agent = RSAAgent(env, rsa_iterations=3, initial_prob=true_wall_prob)
        observer = Observer(env, agent_params={
            'beta': agent.beta, 'rsa_iterations': agent.rsa_iterations
        }, initial_prob=true_wall_prob, intelligent_sampling=False)

        max_steps = 100
        for step in range(max_steps):
            agent.update_internal_belief(obs)

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Step {step + 1} ---")

            agent_pos, target_pos = tuple(obs['agent_pos']), tuple(obs['target_pos'])

            action, action_probs = agent.choose_action(obs)
            local_probs = observer.update_belief(agent_pos, target_pos, action)
            render_side_by_side_views(local_probs, obs['local_view'], env)
            
            print("Observer's Inferred Belief Map (Shading indicates wall probability):")
            observer.render_belief(agent_pos, target_pos)
            
            action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Up-Left", 5: "Up-Right", 6: "Down-Left", 7: "Down-Right"}
            print(f"\nAction Probabilities: {[f'{p:.2f}' for p in action_probs]}")
            print(f"Chosen Action: {action_map[action]}")

            new_obs, reward, terminated, truncated, info = env.step(action)

            print("\nActual Environment State:")
            env.render()
            obs = new_obs
            time.sleep(0.1)

            if terminated or truncated:
                end_reason = "Target reached" if terminated else "Max steps reached"
                print(f"\nEpisode finished: {end_reason} in {step + 1} steps.")
                break
        else:
             print("\nEpisode finished without reaching target.")

        print("\n" + "="*40 + "\n---           EPISODE END          ---\n" + "="*40 + "\n")
        final_agent_pos, final_target_pos = obs['agent_pos'], obs['target_pos']
        print("Observer's Final Inferred Belief Map:")
        observer.render_belief(final_agent_pos, final_target_pos)
        print("\nAgent's Final Internal Belief Map (Ground Truth):")
        agent.render_internal_belief(final_agent_pos, final_target_pos)
        print("\nFinal Actual Environment State:")
        env.render()
        env.close()

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()