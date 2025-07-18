import numpy as np
import torch
from itertools import product

NUM_STATE_SAMPLES = 1000
VIEW_SIZE = 5
COLOR_RESET = "\x1b[0m"

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

        

def _get_base_state_and_uncertain_cells(agent_pos, target_pos, belief_map, env):
    """
    Creates a base 5x5 state and identifies cells with uncertain wall probabilities.
    """
    base_state = np.full((VIEW_SIZE, VIEW_SIZE), env._empty_cell)
    uncertain_cells = []
    view_radius = VIEW_SIZE // 2

    for r_local in range(VIEW_SIZE):
        for c_local in range(VIEW_SIZE):
            r_global = agent_pos[0] + r_local - view_radius
            c_global = agent_pos[1] + c_local - view_radius
            global_pos = (r_global, c_global)

            # Handle agent and target positions
            if global_pos == tuple(agent_pos):
                base_state[r_local, c_local] = env._agent_cell if global_pos != tuple(target_pos) else env._target_cell
                continue
            if global_pos == tuple(target_pos):
                base_state[r_local, c_local] = env._target_cell
                continue
            
            # Handle out-of-bounds as walls
            if not (0 <= r_global < env.grid_size and 0 <= c_global < env.grid_size):
                base_state[r_local, c_local] = env._wall_cell
                continue

            # Handle cells based on belief map
            prob_wall = belief_map[r_global, c_global]
            if prob_wall == 1.0:
                base_state[r_local, c_local] = env._wall_cell
            elif prob_wall == 0.0:
                base_state[r_local, c_local] = env._empty_cell
            else:
                # This cell is uncertain
                uncertain_cells.append(((r_local, c_local), prob_wall))
    
    return base_state, uncertain_cells

def _sample_states_from_uncertainty(base_state, uncertain_cells, env, sampling_mode, uniform_prob, num_samples):
    """
    Generates states for 'uniform' or 'belief_based' sampling modes.
    """
    if not uncertain_cells:
        return [base_state]

    uncertain_coords, uncertain_probs = zip(*uncertain_cells)
    uncertain_probs = np.array(uncertain_probs)
    random_matrix = np.random.rand(num_samples, len(uncertain_cells))

    if sampling_mode == 'belief_based':
        is_wall_matrix = random_matrix < uncertain_probs
    elif sampling_mode == 'uniform':
        is_wall_matrix = random_matrix < uniform_prob
    else:
        # This function should not be called for other modes
        return []

    cell_values = np.where(is_wall_matrix, env._wall_cell, env._empty_cell)
    
    # Create a base set of states and fill in the uncertain cells
    possible_states_np = np.tile(base_state, (num_samples, 1, 1))
    rows, cols = zip(*uncertain_coords)
    possible_states_np[:, rows, cols] = cell_values
            
    return [s for s in possible_states_np]

def _generate_simple_states(base_state, env, num_samples):
    """
    Generates states for the 'simple' sampling mode using cardinal regions. 
    """
    if num_samples != 4 * (2**10):
        raise ValueError(f"For 'simple' sampling mode, num_samples must be {4 * (2**10)}")

    possible_states = []
    regions = {
        'up': [(r, c) for r in range(2) for c in range(5)],
        'down': [(r, c) for r in range(3, 5) for c in range(5)],
        'left': [(r, c) for r in range(5) for c in range(2)],
        'right': [(r, c) for r in range(5) for c in range(3, 5)]
    }
    wall_values = [env._wall_cell, env._empty_cell]

    for region_cells in regions.values():
        # Generate all 2^10 combinations for the current region
        for combination in product(wall_values, repeat=10):
            new_state = base_state.copy()
            for i, cell_coord in enumerate(region_cells):
                if combination[i] == env._wall_cell:
                    new_state[cell_coord] = env._wall_cell
            possible_states.append(new_state)
    
    return possible_states

def _generate_deterministic_states(env):
    """
    Generates all possible 5x5 views from the ground-truth environment grid.
    """
    possible_states = []
    view_radius = VIEW_SIZE // 2
    
    # Create a grid copy without the agent for clean state generation
    pristine_grid = np.copy(env.grid)
    agent_pos_in_grid = np.where(pristine_grid == env._agent_cell)
    if agent_pos_in_grid[0].size > 0:
        pristine_grid[agent_pos_in_grid] = env._empty_cell

    padded_grid = np.pad(pristine_grid, pad_width=view_radius, mode='constant', constant_values=env._wall_cell)

    # Iterate through every non-wall cell to generate a potential view
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if pristine_grid[r, c] != env._wall_cell:
                padded_r, padded_c = r + view_radius, c + view_radius
                
                # Extract the 5x5 local view
                local_view = padded_grid[
                    padded_r - view_radius : padded_r + view_radius + 1,
                    padded_c - view_radius : padded_c + view_radius + 1
                ].copy()

                # Place the agent in the center, unless the target is already there
                if local_view[view_radius, view_radius] != env._target_cell:
                    local_view[view_radius, view_radius] = env._agent_cell
                
                possible_states.append(local_view)

    # Remove duplicates to get a unique set of states
    unique_states_tuples = {tuple(map(tuple, state)) for state in possible_states}
    unique_states = [np.array(state) for state in unique_states_tuples]
    
    return unique_states

def _generate_possible_states(agent_pos, target_pos, belief_map, env, true_state_view=None, sampling_mode='uniform', uniform_prob=0.5, num_samples=NUM_STATE_SAMPLES):
    """
    Generates possible 5x5 local states based on a belief map and a sampling mode.
    """
    possible_states = []

    if sampling_mode == 'deterministic':
        possible_states = _generate_deterministic_states(env)
    else:
        # Most modes start by identifying a base state and uncertain cells
        base_state, uncertain_cells = _get_base_state_and_uncertain_cells(agent_pos, target_pos, belief_map, env)

        if not uncertain_cells and sampling_mode in ['uniform', 'belief_based']:
            possible_states = [base_state]
        elif sampling_mode in ['uniform', 'belief_based']:
            possible_states = _sample_states_from_uncertainty(base_state, uncertain_cells, env, sampling_mode, uniform_prob, num_samples)
        elif sampling_mode == 'simple':
            possible_states = _generate_simple_states(base_state, env, num_samples)
        elif sampling_mode == 'deterministic':
             raise NotImplementedError("Deterministic sampling mode is not implemented yet.")
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    # Ensure the agent's true view is always included in the set of possibilities
    if true_state_view is not None:
        is_present = any(np.array_equal(s, true_state_view) for s in possible_states)
        if not is_present:
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