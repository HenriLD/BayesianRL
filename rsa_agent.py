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


def _calculate_heuristic_utilities(agent_pos, target_pos, states, env):
    """
    Calculates the 'world utility' of each action in each state.
    A "state" is a full 3x3 hypothetical grid.
    """
    num_states = len(states)
    num_actions = env.action_space.n
    all_utilities = np.zeros((num_states, num_actions))

    action_moves = {
        0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
        4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)
    }

    for i, state in enumerate(states):
        hypothetical_walls = set()
        for r_local in range(3):
            for c_local in range(3):
                if state[r_local, c_local] == env._wall_cell:
                    r_global = agent_pos[0] + r_local - 1
                    c_global = agent_pos[1] + c_local - 1
                    hypothetical_walls.add((r_global, c_global))

        action_utilities = []
        current_dist_to_target = _chebyshev_distance(agent_pos, target_pos)

        for action_idx in range(num_actions):
            move = action_moves[action_idx]
            next_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])

            if (next_pos in hypothetical_walls or
                not (0 <= next_pos[0] < env.grid_size and 0 <= next_pos[1] < env.grid_size)):
                action_utilities.append(-np.inf)
                continue

            next_dist_to_target = _chebyshev_distance(next_pos, target_pos)
            utility = current_dist_to_target - next_dist_to_target
            action_utilities.append(utility)

        all_utilities[i, :] = np.array(action_utilities, dtype=float)

    return all_utilities

def _generate_possible_states(agent_pos, target_pos, belief_map, env):
    """
    Generic function to generate all possible 3x3 local states based on a given belief map.
    Used by both the agent (with its internal map) and the observer (with its own map).
    """
    base_state = np.full((3, 3), env._empty_cell)
    uncertain_cells = []

    for r_local in range(3):
        for c_local in range(3):
            r_global = agent_pos[0] + r_local - 1
            c_global = agent_pos[1] + c_local - 1
            global_pos = (r_global, c_global)

            if global_pos == agent_pos:
                if agent_pos == target_pos:
                    base_state[r_local, c_local] = env._target_cell
                else:
                    base_state[r_local, c_local] = env._agent_cell
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
                    uncertain_cells.append((r_local, c_local))

    if not uncertain_cells:
        return [base_state]

    possible_states = []
    num_uncertain = len(uncertain_cells)
    for combo in itertools.product([env._empty_cell, env._wall_cell], repeat=num_uncertain):
        new_state = base_state.copy()
        for i in range(num_uncertain):
            new_state[uncertain_cells[i]] = combo[i]
        possible_states.append(new_state)

    return possible_states

def render_side_by_side_views(wall_probabilities, ground_truth_view, env):
    """
    Renders the observer's inferred wall probabilities next to the
    agent's actual 3x3 ground truth view for comparison.
    """
    header_left = "Observer's Inference"
    header_right = "Agent's Ground Truth"
    print(f"\n{header_left:^20}   |   {header_right:^20}")
    print(f"{'-'*20:^20}   |   {'-'*20:^20}")

    # Map for rendering the ground truth characters
    truth_char_map = {
        env._empty_cell: '.',
        env._wall_cell: '#',
        env._agent_cell: 'A',
        env._target_cell: 'T'
    }

    for r in range(3):
        # Build the left side (probabilities)
        left_row_str = ""
        for c in range(3):
            prob = wall_probabilities[r, c]
            left_row_str += f" {int(prob * 100):>2d}% "

        # Build the right side (ground truth)
        right_row_str = ""
        for c in range(3):
            cell_val = ground_truth_view[r, c]
            char = truth_char_map.get(cell_val, '?')
            right_row_str += f"  {char}  "
        
        print(f"{left_row_str:^20}   |   {right_row_str:^20}")
    print()


class RSAAgent:
    """
    An agent that uses its private 'internal_belief_map' to decide on an action.
    """
    def __init__(self, env: GridEnvironment, rsa_iterations: int = 10, rationality: float = 10, utility_beta: float = 10, initial_prob: float = 0.5):
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

        possible_states = _generate_possible_states(agent_pos, target_pos, self.internal_belief_map, self.env)

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
        prior = np.full(num_states, 1.0 / num_states)

        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.beta * world_utilities)
            sum_exp_utilities = exp_utilities.sum(axis=1, keepdims=True)
            P_S0 = np.nan_to_num(exp_utilities / sum_exp_utilities)

        P_S_k = P_S0
        for _ in range(self.rsa_iterations):
            with np.errstate(divide='ignore', invalid='ignore'):
                exp_utility = np.exp(world_utilities)
                S_denominator = exp_utility.sum(axis=1, keepdims=True)
                P_S_k = np.nan_to_num(exp_utility / S_denominator)
        return P_S_k

    def update_internal_belief(self, observation):
        """Updates the agent's private belief map with ground truth from its 3x3 view."""
        local_view = observation['local_view']
        agent_pos = observation['agent_pos']

        for r_local in range(3):
            for c_local in range(3):
                r_global = agent_pos[0] + r_local - 1
                c_global = agent_pos[1] + c_local - 1
                if 0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size:
                    cell_value = local_view[r_local, c_local]
                    self.internal_belief_map[r_global, c_global] = 1.0 if cell_value == self.env._wall_cell else 0.0

    def render_internal_belief(self, agent_pos, target_pos):
        """Renders the agent's internal belief map using a color gradient."""
        _render_belief_map_with_chars(self.internal_belief_map, self.env.grid_size, agent_pos, target_pos)


class Observer:
    """
    An observer that only sees the agent's position and actions.
    It maintains its own belief map and updates it by inverting the agent's RSA model.
    """
    def __init__(self, env: GridEnvironment, agent_params: dict, initial_prob: float = 0.5, learning_rate: float = 0.3):
        self.env = env
        self.agent_params = agent_params
        self.num_actions = env.action_space.n
        self.observer_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.beta = agent_params.get('beta', 10)
        self.rsa_iterations = agent_params.get('rsa_iterations', 3)
        self.learning_rate = learning_rate

        if env.agent_pos is not None:
            self.observer_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.observer_belief_map[env.target_pos] = 0.0

    def _run_rsa_reasoning_for_observer(self, world_utilities, num_states):
        """Performs the RSA iterative calculation."""
        prior = np.full(num_states, 1.0 / num_states)

        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.beta * world_utilities)
            sum_exp_utilities = exp_utilities.sum(axis=1, keepdims=True)
            P_S0 = np.nan_to_num(exp_utilities / sum_exp_utilities)

        P_S_k = P_S0
        for _ in range(self.rsa_iterations):
            with np.errstate(divide='ignore', invalid='ignore'):
                exp_utility = np.exp(world_utilities)
                S_denominator = exp_utility.sum(axis=1, keepdims=True)
                P_S_k = np.nan_to_num(exp_utility / S_denominator)
        return P_S_k
    
    def _compute_local_belief(self, agent_pos, target_pos, action) -> np.ndarray:
        """
        Computes the inferred 3x3 local probability map based on the agent's action.
        This is the core RSA reversal step.
        """
        possible_states = _generate_possible_states(agent_pos, target_pos, self.observer_belief_map, self.env)
        local_wall_probs = np.zeros((3, 3))

        if not possible_states:
            return local_wall_probs

        num_local_states = len(possible_states)
        prior = np.full(num_local_states, 1.0 / num_local_states)

        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env)
        P_S_k = self._run_rsa_reasoning_for_observer(world_utilities, num_local_states)

        with np.errstate(divide='ignore', invalid='ignore'):
            L_numerator = P_S_k.T * prior
            L_denominator = L_numerator.sum(axis=1, keepdims=True)
            P_L_k = np.nan_to_num(L_numerator / L_denominator)

        state_posterior = P_L_k[action, :]

        # Calculate the probability of a wall for each cell in the 3x3 view
        for r_local in range(3):
            for c_local in range(3):
                prob_wall = sum(state_posterior[i] for i, s in enumerate(possible_states) if s[r_local, c_local] == self.env._wall_cell)
                local_wall_probs[r_local, c_local] = prob_wall
        
        return local_wall_probs
    
    def _update_global_belief(self, local_wall_probs, agent_pos):
        """
        Updates the global belief map by moving its values toward the
        newly inferred local beliefs, controlled by a learning rate.
        """
        self.observer_belief_map[agent_pos] = 0.0  # Agent's location is not a wall

        for r_local in range(3):
            for c_local in range(3):
                if r_local == 1 and c_local == 1: continue

                r_global = agent_pos[0] + r_local - 1
                c_global = agent_pos[1] + c_local - 1

                if 0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size:
                    local_prob = local_wall_probs[r_local, c_local]
                    global_prob = self.observer_belief_map[r_global, c_global]
                    
                    # Move the global belief towards the local inference by the learning rate
                    new_global_prob = global_prob + self.learning_rate * (local_prob - global_prob)
                    self.observer_belief_map[r_global, c_global] = new_global_prob

    def update_belief(self, agent_pos, target_pos, action) -> np.ndarray:
        """
        Orchestrates the two-step belief update process.
        1. Infers local beliefs from the agent's action.
        2. Updates the global belief map with the new local information.
        Returns the inferred local probability map for visualization.
        """
        # Step 1: Compute the local belief based on the action
        local_probs = self._compute_local_belief(agent_pos, target_pos, action)
        
        # Step 2: Update the global belief map using the local inference
        self._update_global_belief(local_probs, agent_pos)

        # Return the local probabilities for visualization purposes
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
        true_wall_prob = 0.3  # For testing, we set a fixed probability

        agent = RSAAgent(env, initial_prob=true_wall_prob)
        observer = Observer(env, agent_params={
            'alpha': agent.alpha,
            'beta': agent.beta,
            'rsa_iterations': agent.rsa_iterations
        }, initial_prob=true_wall_prob)

        max_steps = 100
        for step in range(max_steps):
            agent.update_internal_belief(obs)

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Step {step + 1} ---")

            agent_pos = tuple(obs['agent_pos'])
            target_pos = tuple(obs['target_pos'])

            if agent_pos == target_pos:
                print(f"\nTarget reached in {step} steps!")
                break

            action, action_probs = agent.choose_action(obs)

            # Get local probabilities from the observer and render the new visualization
            local_probs = observer.update_belief(agent_pos, target_pos, action)
            render_side_by_side_views(local_probs, obs['local_view'], env)
            
            # Display observer belief map only every 5 steps (includes step 0).
            if step % 5 == 0:
                print("Observer's Inferred Belief Map (Shading indicates wall probability):")
                observer.render_belief(agent_pos, target_pos)
            
            print()

            action_map = {
                0: "Up", 1: "Down", 2: "Left", 3: "Right",
                4: "Up-Left", 5: "Up-Right", 6: "Down-Left", 7: "Down-Right"
            }
            print(f"Chosen Action: {action_map[action]}")

            new_obs, reward, terminated, truncated, info = env.step(action)

            print()
            print("\nActual Environment State:")
            env.render()
            obs = new_obs

            time.sleep(0.5)

            if truncated:
                print("\nEpisode truncated. Max steps reached.")
                break
        else:
             print("\nEpisode finished.")

        # --- Final Rendering ---
        print("\n" + "="*40)
        print("---           EPISODE END          ---")
        print("="*40 + "\n")

        final_agent_pos = obs['agent_pos']
        final_target_pos = obs['target_pos']

        print("Observer's Final Inferred Belief Map:")
        observer.render_belief(final_agent_pos, final_target_pos)

        print("\nAgent's Final Internal Belief Map (Agent's Perspective):")
        agent.render_internal_belief(final_agent_pos, final_target_pos)

        print("\nFinal Actual Environment State:")
        env.render()

        env.close()

        print(f"True wall probability in the environment: {true_wall_prob:.2f}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()