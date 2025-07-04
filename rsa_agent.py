import numpy as np
import itertools
import time
import os
import torch

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
NUM_STATE_SAMPLES = 50000
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

def _calculate_heuristic_utilities(agent_pos, target_pos, states, env, heuristic_model, sharpening_factor=5.0):
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

        

def _generate_possible_states(agent_pos, target_pos, belief_map, env, true_state_view=None, use_intelligent_sampling=False, unintelligent_prob=0.5, num_samples=NUM_STATE_SAMPLES):
    """
    Generates possible 5x5 local states based on a belief map.
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
        # --- Vectorized State Generation ---
        uncertain_coords, uncertain_probs = zip(*uncertain_cells)
        uncertain_probs = np.array(uncertain_probs)
        
        random_matrix = np.random.rand(num_samples, num_uncertain)
        
        if use_intelligent_sampling:
            is_wall_matrix = random_matrix < uncertain_probs
        else:
            is_wall_matrix = random_matrix < unintelligent_prob
            
        cell_values = np.where(is_wall_matrix, env._wall_cell, env._empty_cell)
        
        possible_states_np = np.tile(base_state, (num_samples, 1, 1))
        
        rows, cols = zip(*uncertain_coords)
        possible_states_np[:, rows, cols] = cell_values
        
        possible_states = [s for s in possible_states_np]
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
    def __init__(self, env: GridEnvironment, rsa_iterations: int = 5, rationality: float = 10, utility_beta: float = 1, initial_prob: float = 0.25, model_path: str = "heuristic_agent.zip", sharpening_factor: float = 5.0, num_samples: int = NUM_STATE_SAMPLES):
        self.env = env
        self.rsa_iterations = rsa_iterations
        self.alpha = rationality
        self.beta = utility_beta
        self.num_actions = env.action_space.n
        self.internal_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.default_prob = initial_prob
        self.model_path = model_path
        self.sharpening_factor = sharpening_factor
        self.num_samples = num_samples
        

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
            agent_pos, target_pos, self.internal_belief_map, self.env, true_state_view=s_true, unintelligent_prob=self.default_prob, num_samples=self.num_samples,
        )

        s_true_idx = -1
        for i, state in enumerate(possible_states):
            if np.array_equal(state, s_true):
                s_true_idx = i
                break
        
        assert s_true_idx != -1, "True state not found in generated states."

        # Calculate the world utilities for each possible state using the heuristic model
        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env, heuristic_model=self.heuristic_model, sharpening_factor=self.sharpening_factor)

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
    def __init__(self, env: GridEnvironment, agent_params: dict, initial_prob: float = 0.25, learning_rate: float = 0.5, intelligent_sampling: bool = False, model_path: str = "heuristic_agent.zip", sharpening_factor: float = 5.0, num_samples: int = NUM_STATE_SAMPLES):
        self.env = env
        self.beta = agent_params.get('beta', 10)
        self.rsa_iterations = agent_params.get('rsa_iterations', 3)
        self.learning_rate = learning_rate
        self.intelligent_sampling = intelligent_sampling
        self.observer_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.view_radius = VIEW_SIZE // 2
        self.default_prob = initial_prob
        self.model_path = model_path
        self.sharpening_factor = sharpening_factor
        self.num_samples = num_samples

        if env.agent_pos is not None:
            self.observer_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.observer_belief_map[env.target_pos] = 0.0

        # Load the trained heuristic model
        try:
            self.heuristic_model = PPO.load(self.model_path, env=env)
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
            use_intelligent_sampling=self.intelligent_sampling, unintelligent_prob=self.default_prob, num_samples=self.num_samples
        )
        local_wall_probs = np.zeros((VIEW_SIZE, VIEW_SIZE))

        if not possible_states:
            return local_wall_probs

        num_local_states = len(possible_states)
        prior = np.full(num_local_states, 1.0 / num_local_states)

        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env, heuristic_model=self.heuristic_model, sharpening_factor=self.sharpening_factor)
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

def run_simulation(params: dict):
    """
    Runs a single simulation episode with a given set of parameters.

    Args:
        params (dict): A dictionary containing all hyperparameters.
    
    Returns:
        dict: A dictionary containing key metrics from the simulation.
    """
    # Extract hyperparameters from the params dictionary
    custom_map = params["custom_map"]
    model_path = params["model_path"]
    rsa_iterations = params.get("rsa_iterations", 10)
    agent_rationality = params.get("agent_rationality", 10)
    agent_utility_beta = params.get("agent_utility_beta", 1)
    sharpening_factor = params.get("sharpening_factor", 5.0)
    observer_learning_rate = params.get("observer_learning_rate", 0.33)
    observer_intelligent_sampling = params.get("observer_intelligent_sampling", False)
    max_steps = params.get("max_steps", 100)
    render = params.get("render", True)
    time_delay = params.get("time_delay", 0.3)
    num_samples = params.get("num_samples", NUM_STATE_SAMPLES)
    
    render_mode = 'human' if render else None
    env = GridEnvironment(grid_map=custom_map, render_mode=render_mode)
    obs, info = env.reset()

    num_walls = sum(row.count('#') for row in custom_map)
    total_cells = env.grid_size * env.grid_size
    true_wall_prob = num_walls / total_cells

    agent = RSAAgent(
        env,
        model_path=model_path,
        rsa_iterations=rsa_iterations,
        initial_prob=true_wall_prob,
        rationality=agent_rationality,
        utility_beta=agent_utility_beta,
        sharpening_factor=sharpening_factor,
        num_samples=num_samples
    )
    
    observer = Observer(
        env,
        model_path=model_path,
        agent_params={'beta': agent.beta, 'rsa_iterations': agent.rsa_iterations},
        initial_prob=true_wall_prob,
        intelligent_sampling=observer_intelligent_sampling,
        learning_rate=observer_learning_rate,
        sharpening_factor=sharpening_factor
        num_samples=num_samples
    )

    for step in range(max_steps):
        agent.update_internal_belief(obs)
        agent_pos, target_pos = tuple(obs['agent_pos']), tuple(obs['target_pos'])

        if render:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Step {step + 1} ---")

        action, action_probs = agent.choose_action(obs)
        local_probs = observer.update_belief(agent_pos, target_pos, action)
        
        if render:
            render_side_by_side_views(local_probs, obs['local_view'], env)
            print("Observer's Inferred Belief Map (Shading indicates wall probability):")
            observer.render_belief(agent_pos, target_pos)
            
            action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Up-Left", 5: "Up-Right", 6: "Down-Left", 7: "Down-Right"}
            print(f"\nAction Probabilities: {[f'{p:.2f}' for p in action_probs]}")
            print(f"Chosen Action: {action_map[action]}")

        new_obs, reward, terminated, truncated, info = env.step(action)
        
        if render:
            print("\nActual Environment State:")
            env.render()
        
        obs = new_obs
        
        if render:
            time.sleep(time_delay)

        if terminated or truncated:
            if render:
                end_reason = "Target reached" if terminated else "Max steps reached"
                print(f"\nEpisode finished: {end_reason} in {step + 1} steps.")
            break
    else:
         if render:
             print("\nEpisode finished without reaching target.")

    final_agent_pos, final_target_pos = obs['agent_pos'], obs['target_pos']
    if render:
        print("\n" + "="*40 + "\n---           EPISODE END          ---\n" + "="*40 + "\n")
        print("Observer's Final Inferred Belief Map:")
        observer.render_belief(final_agent_pos, final_target_pos)
        print("\nAgent's Final Internal Belief Map (Ground Truth):")
        agent.render_internal_belief(final_agent_pos, final_target_pos)
        print("\nFinal Actual Environment State:")
        env.render()
    
    env.close()

    # Calculate final mean squared error between belief maps
    final_mse = np.mean((agent.internal_belief_map - observer.observer_belief_map)**2)
    
    metrics = {
        "steps_taken": step + 1,
        "target_reached": terminated,
        "final_mse": final_mse
    }
    return metrics

if __name__ == '__main__':
    # Define default hyperparameters in a dictionary for a single run.
    # To perform a hyperparameter search, you can create a loop
    # that calls run_simulation with different parameter dicts.
    default_params = {
        "custom_map": [
            "##############",
            "#     T#     #",
            "#  ##    #   #",
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
        ],
        "model_path": "heuristic_agent.zip",
        "rsa_iterations": 10,
        "agent_rationality": 10,
        "agent_utility_beta": 1,
        "sharpening_factor": 5.0,
        "observer_learning_rate": 0.33,
        "observer_intelligent_sampling": False,
        "max_steps": 100,
        "render": True,
        "time_delay": 0.3,
        "num_samples": 50000
    }
    
    try:
        results = run_simulation(default_params)
        print("\n" + "="*40)
        print("---       SIMULATION COMPLETE      ---")
        print("="*40)
        print(f"  Target Reached: {results['target_reached']}")
        print(f"  Steps Taken:    {results['steps_taken']}")
        print(f"  Final Belief MSE: {results['final_mse']:.4f}")
        print("="*40)
        
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()