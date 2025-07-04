import numpy as np
import itertools
import time
import os
import torch

try:
    from env import GridEnvironment
except ImportError:
    print("Error: Could not import GridEnvironment.")
    print("Please ensure the GridEnvironment class is in 'env.py'.")
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
                grid_str += _get_char_for_prob(prob, default_prob, max_dev, min_dev)
        grid_str += "\n"
    print(grid_str)

def _calculate_heuristic_utilities(agent_pos, target_pos, states, env, heuristic_model, sharpening_factor=1.0):
    """
    Calculates action utilities using the value function of a trained RL agent.
    This is used by the Observer to model the agent's decision-making process.
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
        batch_obs_np = []
        valid_action_indices = []

        for action_idx in range(num_actions):
            move = action_moves[action_idx]
            
            local_next_pos = (view_radius + move[0], view_radius + move[1])
            if state_view[local_next_pos] == env._wall_cell:
                continue

            next_agent_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])
            obs = {
                'local_view': state_view.astype(np.int32),
                'agent_pos': np.array(next_agent_pos, dtype=np.int32),
                'target_pos': np.array(target_pos, dtype=np.int32)
            }
            batch_obs_np.append(obs)
            valid_action_indices.append(action_idx)

        action_utilities = np.full(num_actions, -np.inf, dtype=float)
        if batch_obs_np:
            # Ensure tensors are created on the same device as the model
            device = heuristic_model.device
            obs_tensor = {
                key: torch.as_tensor(np.stack([obs[key] for obs in batch_obs_np]), device=device)
                for key in batch_obs_np[0]
            }
            
            with torch.no_grad():
                values = heuristic_model.policy.predict_values(obs_tensor)
            
            for j, action_idx in enumerate(valid_action_indices):
                action_utilities[action_idx] = values[j].item() * sharpening_factor
            
        all_utilities[i, :] = action_utilities

    return all_utilities

def _generate_possible_states(agent_pos, target_pos, belief_map, env, use_intelligent_sampling=False, unintelligent_prob=0.5):
    """
    Generates possible 5x5 local states based on a belief map.
    Used by the Observer to create hypotheses about the true state.
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
                base_state[r_local, c_local] = env._agent_cell
            elif global_pos == target_pos:
                base_state[r_local, c_local] = env._target_cell
            elif not (0 <= r_global < env.grid_size and 0 <= c_global < env.grid_size):
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
        uncertain_coords, uncertain_probs = zip(*uncertain_cells)
        uncertain_probs = np.array(uncertain_probs)
        random_matrix = np.random.rand(NUM_STATE_SAMPLES, num_uncertain)
        
        is_wall_matrix = random_matrix < (uncertain_probs if use_intelligent_sampling else unintelligent_prob)
        cell_values = np.where(is_wall_matrix, env._wall_cell, env._empty_cell)
        
        possible_states_np = np.tile(base_state, (NUM_STATE_SAMPLES, 1, 1))
        rows, cols = zip(*uncertain_coords)
        possible_states_np[:, rows, cols] = cell_values
        possible_states = [s for s in possible_states_np]
    else:
        possible_states.append(base_state)

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

    truth_char_map = {
        env._empty_cell: '.', env._wall_cell: '#',
        env._agent_cell: 'A', env._target_cell: 'T'
    }

    for r in range(VIEW_SIZE):
        left_row_str = "".join([f" {int(prob * 100):>3d}% " for prob in wall_probabilities[r, :]])
        right_row_str = "".join([f"  {truth_char_map.get(cell_val, '?')}   " for cell_val in ground_truth_view[r, :]])
        print(f"{left_row_str:^24}   |   {right_row_str:^24}")
    print()


class BaseAgent:
    """
    An agent that uses a pre-trained policy to decide on an action.
    It acts based on its ground-truth local view, without RSA reasoning.
    """
    def __init__(self, env: GridEnvironment, model_path: str, initial_prob: float = 0.3):
        self.env = env
        self.num_actions = env.action_space.n
        self.internal_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.default_prob = initial_prob
        
        if env.agent_pos is not None: self.internal_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None: self.internal_belief_map[env.target_pos] = 0.0

        try:
            self.heuristic_model = PPO.load(model_path, env=env)
            print("Heuristic model loaded successfully for BaseAgent.")
        except Exception as e:
            print(f"Error loading heuristic model for BaseAgent: {e}")
            self.heuristic_model = None

    def choose_action(self, observation):
        """Chooses an action directly from the policy network given the observation."""
        if self.heuristic_model is None:
            return self.env.action_space.sample(), np.full(self.num_actions, 1.0 / self.num_actions)

        # Manually create tensor and send to device, adding a batch dimension.
        device = self.heuristic_model.device
        obs_tensor = {key: torch.as_tensor([value], device=device) for key, value in observation.items()}
        
        with torch.no_grad():
            dist = self.heuristic_model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy().squeeze()
            action = dist.sample().cpu().item()

        return int(action), probs.flatten()

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
        """Renders the agent's internal belief map."""
        _render_belief_map_with_chars(self.internal_belief_map, self.env.grid_size, agent_pos, target_pos, self.default_prob)


class Observer:
    """
    An observer that only sees the agent's actions and position.
    It maintains a belief map and updates it by inverting the agent's policy model.
    """
    def __init__(self, env: GridEnvironment, model_path: str, policy_beta: float = 1.0, initial_prob: float = 0.25, learning_rate: float = 0.25, intelligent_sampling: bool = False):
        self.env = env
        self.beta = policy_beta
        self.learning_rate = learning_rate
        self.intelligent_sampling = intelligent_sampling
        self.observer_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.view_radius = VIEW_SIZE // 2
        self.default_prob = initial_prob

        if env.agent_pos is not None: self.observer_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None: self.observer_belief_map[env.target_pos] = 0.0

        try:
            self.heuristic_model = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"Error loading heuristic model for observer: {e}")
            self.heuristic_model = None

    def _get_action_probs_from_utilities(self, world_utilities):
        """
        Calculates action probabilities based on utilities. P(a|s) = softmax(beta * U(s,a)).
        This is the observer's model of the agent's policy.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.beta * world_utilities)
            action_probs = np.nan_to_num(exp_utilities / exp_utilities.sum(axis=1, keepdims=True))
        return action_probs
    
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
        action_probs_given_state = self._get_action_probs_from_utilities(world_utilities)

        likelihood = action_probs_given_state[:, action]
        posterior = likelihood * prior
        
        posterior_sum = posterior.sum()
        state_posterior = posterior / posterior_sum if posterior_sum > 0 else prior

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
        """Renders the observer's belief map."""
        _render_belief_map_with_chars(self.observer_belief_map, self.env.grid_size, agent_pos, target_pos, self.default_prob)


# --- Main execution block ---
if __name__ == '__main__':
    try:
        custom_map = [
            "##############",
            "#      #    T#",
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
        ]

        env = GridEnvironment(grid_map=custom_map, render_mode='human')
        obs, info = env.reset()

        num_walls = sum(row.count('#') for row in custom_map)
        total_cells = env.grid_size * env.grid_size
        true_wall_prob = num_walls / total_cells

        model_path = "heuristic_agent.zip"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at '{model_path}'")
            print("Please train a model first using a separate training script.")
            exit()

        agent = BaseAgent(env, model_path=model_path, initial_prob=true_wall_prob)
        observer = Observer(env, model_path=model_path, policy_beta=10.0, 
                            initial_prob=true_wall_prob, intelligent_sampling=False, 
                            learning_rate=0.25)

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
            
            action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Up-L", 5: "Up-R", 6: "Down-L", 7: "Down-R"}
            print(f"\nAction Probabilities: {[f'{p:.2f}' for p in action_probs]}")
            print(f"Chosen Action: {action_map.get(action, 'Unknown')}")

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