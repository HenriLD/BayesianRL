import numpy as np
import itertools
import time
import os

try:
    from env import GridEnvironment
except ImportError:
    print("Error: Could not import GridEnvironment.")
    print("Please ensure the updated GridEnvironment class is in 'env.py'.")
    exit()

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Error: Could not import Stable-Baselines3.")
    print("Please install it with: pip install stable-baselines3[extra]")
    exit()


# --- ANSI Color Constants for Rendering ---
COLOR_RESET = "\x1b[0m"

# Define the number of states to sample when the total number of possibilities is too large.
NUM_STATE_SAMPLES = 20000
VIEW_SIZE = 5

def _get_char_for_prob(prob, default_prob, max_dev, min_dev):
    """
    Maps a wall probability to a color, scaled relative to the highest and
    lowest deviations from the default probability in the current belief map.
    """
    # Epsilon to prevent division by zero
    epsilon = 1e-9
    current_dev = prob - default_prob

    # If the cell's deviation is negligible, render as a neutral mid-gray
    if abs(current_dev) < 0.01:
        gray_index = 232 + round(0.5 * 23)
    
    # If deviation is positive, scale from mid-gray to white
    elif current_dev > 0:
        # Normalize this cell's deviation relative to the max positive deviation
        normalized_val = current_dev / (max_dev + epsilon)
        gray_index = 232 + round((0.5 + 0.5 * normalized_val) * 23)
    
    # If deviation is negative, scale from mid-gray to black
    else: # current_dev < 0
        # Normalize this cell's deviation relative to the max negative deviation
        # Note: neg / neg -> positive value
        normalized_val = current_dev / (min_dev - epsilon)
        gray_index = 232 + round((0.5 - 0.5 * normalized_val) * 23)

    # Ensure the final index is within the valid ANSI grayscale range
    gray_index = int(np.clip(gray_index, 232, 255))

    color_code = f"\x1b[38;5;{gray_index}m"
    block = "███"

    return f"{color_code}{block}{COLOR_RESET}"


def _render_belief_map_with_chars(belief_map, grid_size, agent_pos, target_pos, default_prob):
    """
    Renders a belief map, normalizing colors based on the maximum and minimum
    deviations from the default probability in the current map.
    """
    # First, find the max/min deviations across the entire map for normalization
    deviations = belief_map - default_prob
    max_dev = deviations.max()
    min_dev = deviations.min()

    grid_str = ""
    for r in range(grid_size):
        for c in range(grid_size):
            pos = (r, c)
            if pos == tuple(agent_pos):
                grid_str += ' A '
            elif pos == tuple(target_pos):
                grid_str += ' T '
            else:
                prob = belief_map[r, c]
                # Pass the full context, including max/min deviations, for rendering
                grid_str += _get_char_for_prob(prob, default_prob, max_dev, min_dev)
        grid_str += "\n"
    print(grid_str)

def _calculate_heuristic_utilities(agent_pos, target_pos, states, env, heuristic_model, sharpening_factor=1.0):
    """
    Calculates action utilities using the value function of a trained RL agent.
    This version uses BATCH PROCESSING to significantly speed up inference.
    """
    num_states = len(states)
    num_actions = env.action_space.n
    all_utilities = np.zeros((num_states, num_actions))
    view_radius = VIEW_SIZE // 2

    action_moves = {
        0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
        4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)
    }

    for i, state_view in enumerate(states):
        # --- Batch Preparation ---
        # Instead of predicting one by one, we create a batch of all valid
        # observations for the current hypothetical state.
        batch_obs_np = []
        valid_action_indices = []

        for action_idx in range(num_actions):
            move = action_moves[action_idx]
            
            # Check if the move is into a wall within the hypothetical state
            local_next_pos = (view_radius + move[0], view_radius + move[1])
            if state_view[local_next_pos] == env._wall_cell:
                continue

            # If the action is valid, prepare its observation and add it to the batch
            next_agent_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])
            obs = {
                'local_view': state_view.astype(np.int32),
                'agent_pos': np.array(next_agent_pos, dtype=np.int32),
                'target_pos': np.array(target_pos, dtype=np.int32)
            }
            batch_obs_np.append(obs)
            valid_action_indices.append(action_idx)

        # --- Batch Prediction ---
        # If there are any valid actions, predict their values all at once.
        action_utilities = np.full(num_actions, -np.inf, dtype=float)
        if batch_obs_np:
            # Collate the list of dictionaries into a single dictionary of batches
            obs_tensor = {
                key: torch.as_tensor(np.stack([obs[key] for obs in batch_obs_np]))
                for key in batch_obs_np[0]
            }
            
            # Use the trained model's value function to predict all utilities in one go
            with torch.no_grad():
                values = heuristic_model.policy.predict_values(obs_tensor)
            
            # Distribute the results back to the corresponding action indices
            for j, action_idx in enumerate(valid_action_indices):
                action_utilities[action_idx] = values[j].item() * sharpening_factor
            
        all_utilities[i, :] = action_utilities

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
                    cell_type = np.random.choice(
                    [env._wall_cell, env._empty_cell],
                    p=[true_wall_prob, 1 - true_wall_prob]
                )
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
    def __init__(self, env: GridEnvironment, rsa_iterations: int = 10, rationality: float = 1, utility_beta: float = 1, initial_prob: float = 0.3):
        self.env = env
        self.rsa_iterations = rsa_iterations
        self.alpha = rationality
        self.beta = utility_beta
        self.num_actions = env.action_space.n
        self.internal_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.default_prob = initial_prob
        

        if env.agent_pos is not None:
            self.internal_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.internal_belief_map[env.target_pos] = 0.0

        # Load the trained heuristic model
        try:
            self.heuristic_model = PPO.load(model_path, env=env)
            print("Heuristic model loaded successfully.")
        except Exception as e:
            print(f"Error loading heuristic model: {e}")
            self.heuristic_model = None

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

        # Calculate the world utilities for each possible state using the heuristic model
        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env, heuristic_model=self.heuristic_model)

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
        _render_belief_map_with_chars(self.internal_belief_map, self.env.grid_size, agent_pos, target_pos, self.default_prob)


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
        self.default_prob = initial_prob

        if env.agent_pos is not None:
            self.observer_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.observer_belief_map[env.target_pos] = 0.0

        # Load the trained heuristic model
        try:
            self.heuristic_model = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"Error loading heuristic model for observer: {e}")
            self.heuristic_model = None

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

        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env, heuristic_model=self.heuristic_model)
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
        _render_belief_map_with_chars(self.observer_belief_map, self.env.grid_size, agent_pos, target_pos, self.default_prob)


# --- Main execution block ---
if __name__ == '__main__':
    try:
        import torch
        """custom_map = [
            "#############",
            "#      #    #",
            "#  # T      #",
            "#  ####  #  #",
            "#  #     #  #",
            "#      ### ##",
            "#  #       ##",
            "#  #   #   ##",
            "# #### ##  ##",
            "#   #   #   #",
            "#   #       #",
            "#  ##A#  #  #",
            "#############",
        ]"""
        custom_map = [
            "##############",
            "#      #     #",
            "#  ##T   #   #",
            "#   ## # #   #",
            "#      #     #",
            "# # #        #",
            "#   #   ### ##",
            "## ## ##    ##",
            "#           ##",
            "# ####  ##  ##",
            "#    #  # #  #",
            "# ##    #    #",
            "#A#  ####    #",
            "##############",
        ]

        env = GridEnvironment(grid_map=custom_map, render_mode='human')
        obs, info = env.reset()

        # Calculate the true probability of a wall in the given map
        num_walls = sum(row.count('#') for row in custom_map)
        total_cells = env.grid_size * env.grid_size
        true_wall_prob = num_walls / total_cells

        model_path = "heuristic_agent.zip"
        
        # By default, intelligent_sampling is False.
        # To enable, set intelligent_sampling=True in the constructors below.

        agent = RSAAgent(env, rsa_iterations=10, initial_prob=true_wall_prob)
        observer = Observer(env, agent_params={
            'beta': agent.beta, 'rsa_iterations': agent.rsa_iterations
        }, initial_prob=true_wall_prob, intelligent_sampling=False, learning_rate=0.25)

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
            time.sleep(0.3)

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