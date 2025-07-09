import numpy as np
import itertools
import time
import os
import torch
import random
from collections import deque
from scipy.spatial.distance import jensenshannon

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

from utils import (
    _render_belief_map_with_chars,
    _calculate_heuristic_utilities,
    _generate_possible_states,
    render_side_by_side_views,
)

random.seed(42)  # For reproducibility
VIEW_SIZE = 5

class RSAAgent:
    """
    An agent that uses its private 'internal_belief_map' to decide on an action.
    """
    def __init__(self, env: GridEnvironment, rsa_iterations: int = 5, rationality: float = 1, utility_beta: float = 1, 
                 initial_prob: float = 0.25, model_path: str = "heuristic_agent.zip", sharpening_factor: float = 1.0, 
                 num_samples: int = 1000, convergence_threshold: float = 0.001, max_cycle: int = 0,
                 sampling_mode: str = 'uniform'):   
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
        self.convergence_threshold = convergence_threshold
        self.position_history = deque(maxlen=max_cycle)
        self.sampling_mode = sampling_mode  # Default sampling mode
        

        if env.agent_pos is not None:
            self.internal_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.internal_belief_map[env.target_pos] = 0.0

        # Load the trained heuristic model
        try:
            self.heuristic_model = PPO.load(model_path, env=env)
            # print("Heuristic model loaded successfully.")
        except Exception as e:
            print(f"Error loading heuristic model: {e}")
            self.heuristic_model = None

    def choose_action(self, observation):
        agent_pos = tuple(observation['agent_pos'])
        target_pos = tuple(observation['target_pos'])
        s_true = observation['local_view']

        if self.sampling_mode == 'deterministic':
            # TODO
            # possible_states = _generate_possible_states(agent_pos, target_pos, self.internal_belief_map, self.env, 
            # true_state_view=s_true, uniform_prob=self.default_prob, num_samples=self.num_samples, sampling_mode=self.sampling_mode)
            raise NotImplementedError("Deterministic sampling mode is not implemented yet.")

        else:
            possible_states = _generate_possible_states(
                agent_pos, target_pos, self.internal_belief_map, self.env, true_state_view=s_true, 
                uniform_prob=self.default_prob, num_samples=self.num_samples, sampling_mode=self.sampling_mode
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
        P_S_k = self._run_rsa_reasoning(world_utilities)

        final_action_probs = P_S_k[s_true_idx, :]

        # Cycle-breaking logic
        sorted_actions = np.argsort(final_action_probs)[::-1]
        
        action_moves = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
            4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)
        }

        chosen_action = None
        for action in sorted_actions:
            move = action_moves.get(action)
            if move:
                next_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])
                if next_pos not in self.position_history:
                    chosen_action = action
                    self.position_history.append(next_pos)
                    break # Found the best non-cyclic move

        return int(chosen_action), final_action_probs

    def _run_rsa_reasoning(self, world_utilities):
        """
        Performs the alternating speaker-listener RSA calculation.
        """
        # S_0: Initialize with a "literal speaker" that acts based on raw utilities.
        # This speaker model computes P(action | state).
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.beta * world_utilities)
            speaker_probs = np.nan_to_num(exp_utilities / exp_utilities.sum(axis=1, keepdims=True), posinf=0)

        # Iteratively refine the speaker and listener models.
        for _ in range(self.rsa_iterations):
            prev_speaker_probs = speaker_probs.copy()
            # L_k: Listener model infers P(state | action).
            # It assumes a uniform prior over states, so P(s|a) is proportional to P(a|s).
            # We normalize over states for each action.
            with np.errstate(divide='ignore', invalid='ignore'):
                # The sum is over states (axis 0).
                listener_probs = speaker_probs / speaker_probs.sum(axis=0, keepdims=True)
                listener_probs = np.nan_to_num(listener_probs)

            # S_k: Speaker model updates based on the listener's (in)ability to understand.
            # The speaker's utility for an action is how likely the listener is to infer the correct state.
            with np.errstate(divide='ignore'): # Ignore log(0) warnings.
                log_listener_probs = np.log(listener_probs)

            # The new utility combines the pragmatic utility (from the listener) and the original RL utility.
            # The 'alpha' parameter controls the weight of the pragmatic reasoning.
            combined_utility = (self.alpha * log_listener_probs) + (self.beta * world_utilities)

            # The new speaker model is a softmax over the combined utilities.
            with np.errstate(divide='ignore', invalid='ignore'):
                exp_combined = np.exp(combined_utility)
                speaker_probs = np.nan_to_num(exp_combined / exp_combined.sum(axis=1, keepdims=True), posinf=0)

            # Convergence check
            if np.allclose(speaker_probs, prev_speaker_probs, atol=self.convergence_threshold):
                # print(f"Agent converged at iteration {i+1}") # Optional: for debugging
                break
        
        return speaker_probs

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
    def __init__(self, env: GridEnvironment, agent_params: dict, initial_prob: float = 0.25, learning_rate: float = 0.5, 
                 sampling_mode: str = 'uniform', model_path: str = "heuristic_agent.zip", sharpening_factor: float = 5.0, num_samples: int = 1000, confidence=False):
        self.env = env
        self.alpha = agent_params.get('rationality', 1)
        self.beta = agent_params.get('beta', 1)
        self.rsa_iterations = agent_params.get('rsa_iterations', 3)
        self.learning_rate = learning_rate
        self.sampling_mode = sampling_mode
        self.observer_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.view_radius = VIEW_SIZE // 2
        self.default_prob = initial_prob
        self.model_path = model_path
        self.sharpening_factor = sharpening_factor
        self.num_samples = num_samples
        self.convergence_threshold = agent_params.get('convergence_threshold', 0.001)
        self.confidence = confidence
        self.confidence_scores = []

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

    def _run_rsa_reasoning_for_observer(self, world_utilities):
        """
        Performs the alternating speaker-listener RSA calculation to model the agent, with a convergence check.
        """
        # S_0: Initialize with a "literal speaker" that acts based on raw utilities.
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.beta * world_utilities)
            speaker_probs = np.nan_to_num(exp_utilities / exp_utilities.sum(axis=1, keepdims=True), posinf=0)

        # Iteratively refine the speaker and listener models.
        for i in range(self.rsa_iterations):
            prev_speaker_probs = speaker_probs.copy()

            # L_k: Listener model infers P(state | action).
            with np.errstate(divide='ignore', invalid='ignore'):
                listener_probs = speaker_probs / speaker_probs.sum(axis=0, keepdims=True)
                listener_probs = np.nan_to_num(listener_probs)

            # S_k: Speaker model updates based on the listener's understanding.
            with np.errstate(divide='ignore'):
                log_listener_probs = np.log(listener_probs)
            
            combined_utility = (self.alpha * log_listener_probs) + (self.beta * world_utilities)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                exp_combined = np.exp(combined_utility)
                speaker_probs = np.nan_to_num(exp_combined / exp_combined.sum(axis=1, keepdims=True), posinf=0)

            # Convergence check
            if np.allclose(speaker_probs, prev_speaker_probs, atol=self.convergence_threshold):
                # print(f"Observer converged at iteration {i+1}")
                break
        
        return speaker_probs
    
    def _compute_local_belief(self, agent_pos, target_pos, action) -> np.ndarray:
        """Computes the inferred 5x5 local probability map based on the agent's action."""
        possible_states = _generate_possible_states(
            agent_pos, target_pos, self.observer_belief_map, self.env,
            sampling_mode=self.sampling_mode, uniform_prob=self.default_prob, num_samples=self.num_samples
        )
        
        # If there are no uncertain states, no inference is needed.
        if not possible_states:
            return np.zeros((VIEW_SIZE, VIEW_SIZE)), 1.0

        num_local_states = len(possible_states)
        prior_state_prob = np.full(num_local_states, 1.0 / num_local_states)

        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env, heuristic_model=self.heuristic_model, sharpening_factor=self.sharpening_factor)
        final_speaker_probs = self._run_rsa_reasoning_for_observer(world_utilities)

        likelihood = final_speaker_probs[:, action]
        posterior = likelihood * prior_state_prob
        
        posterior_sum = posterior.sum()
        if posterior_sum > 1e-9:
            state_posterior = posterior / posterior_sum
        else:
            state_posterior = prior_state_prob

        # Entropy and Confidence Calculation
        log_posterior = np.log(state_posterior + 1e-9) # Add epsilon to avoid log(0)
        posterior_entropy = -np.sum(state_posterior * log_posterior)
        
        max_entropy = np.log(num_local_states) if num_local_states > 1 else 1.0
        confidence = 1.0 - (posterior_entropy / max_entropy)
        confidence = np.clip(confidence, 0.0, 1.0)
        self.confidence_scores.append(confidence)
        # print(f"Observer confidence: {confidence:.4f}") # for debugging
        
        # Wall Probability Calculation
        # Convert the list of states to a NumPy array for vectorized operations.
        possible_states_np = np.array(possible_states)
        
        # Create a boolean mask where a cell is a wall.
        is_wall_mask = (possible_states_np == self.env._wall_cell)
        
        # Use the posterior probabilities to weight the wall mask and sum over all states.
        local_wall_probs = np.einsum('i,ijk->jk', state_posterior, is_wall_mask)
        
        return local_wall_probs, confidence
    
    def _update_global_belief(self, local_wall_probs, agent_pos, confidence):
        """Updates the global belief map using the newly inferred local beliefs."""
        self.observer_belief_map[agent_pos] = 0.0

        self.learning_rate *= 5 * confidence  # Scale learning rate by confidence

        # Define the global and local boundaries
        r_global_start = agent_pos[0] - self.view_radius
        r_global_end = agent_pos[0] + self.view_radius + 1
        c_global_start = agent_pos[1] - self.view_radius
        c_global_end = agent_pos[1] + self.view_radius + 1
        
        # Ensure boundaries are within the grid
        r_clip_start = max(0, r_global_start)
        r_clip_end = min(self.env.grid_size, r_global_end)
        c_clip_start = max(0, c_global_start)
        c_clip_end = min(self.env.grid_size, c_global_end)

        # Get the corresponding slices
        global_slice = self.observer_belief_map[r_clip_start:r_clip_end, c_clip_start:c_clip_end]
        local_slice = local_wall_probs[r_clip_start - r_global_start : r_clip_end - r_global_start,
                                    c_clip_start - c_global_start : c_clip_end - c_global_start]

        # Update the belief map
        new_global_prob = global_slice + self.learning_rate * (local_slice - global_slice)
        self.observer_belief_map[r_clip_start:r_clip_end, c_clip_start:c_clip_end] = np.clip(new_global_prob, 0.0, 1.0)

    def update_belief(self, agent_pos, target_pos, action) -> np.ndarray:
        """Orchestrates the two-step belief update process."""
        local_probs, conf = self._compute_local_belief(agent_pos, target_pos, action)
        if not self.confidence:
            conf = 1
        self._update_global_belief(local_probs, agent_pos, conf)
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
    observer_learning_rate = params.get("observer_learning_rate", 1)
    max_steps = params.get("max_steps", 100)
    render = params.get("render", True)
    time_delay = params.get("time_delay", 0.3)
    num_samples = params.get("num_samples", NUM_STATE_SAMPLES)
    num_iterations = params.get("num_iterations", 1)
    randomize_agent_after_goal = params.get("randomize_agent_after_goal", True)
    randomize_target_after_goal = params.get("randomize_target_after_goal", False)
    randomize_initial_placement = params.get("randomize_initial_placement", False)
    convergence_threshold = params.get("convergence_threshold", 0.001)
    confidence = params.get("confidence", False)
    max_cycle = params.get("max_cycle", 0)
    agent_sampling_mode = params.get("agent_sampling_mode", 'uniform')
    observer_sampling_mode = params.get("observer_sampling_mode", 'uniform')

    # Randomize Initial Agent/Target Placement
    
    map_template = [list(row) for row in custom_map]
    valid_spawn_points = []

    if randomize_initial_placement:
    # Find all empty spaces (' '), agent start ('A'), and target ('T')
        for r, row in enumerate(map_template):
            for c, char in enumerate(row):
                if char in [' ', 'A', 'T']:
                    valid_spawn_points.append((r, c))
                    # Clear original A and T to make them valid spawn points
                    if char in ['A', 'T']:
                        map_template[r][c] = ' '

        if len(valid_spawn_points) < 2:
            raise ValueError("Map must have at least two empty spaces for agent and target.")

        # Choose two distinct random positions for the first agent and target
        agent_start_pos, target_start_pos = random.sample(valid_spawn_points, 2)
        
        # Place them in the map
        map_template[agent_start_pos[0]][agent_start_pos[1]] = 'A'
        map_template[target_start_pos[0]][target_start_pos[1]] = 'T'
        
    # Convert back to list of strings
    final_map = ["".join(row) for row in map_template]
    
    render_mode = 'human' if render else None
    env = GridEnvironment(grid_map=final_map, render_mode=render_mode, max_steps=max_steps)
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
        num_samples=num_samples,
        convergence_threshold=convergence_threshold,
        max_cycle=max_cycle,
        sampling_mode=agent_sampling_mode
    )
    
    observer = Observer(
        env,
        model_path=model_path,
        agent_params={'beta': agent.beta, 'rsa_iterations': agent.rsa_iterations},
        initial_prob=true_wall_prob,
        sampling_mode=observer_sampling_mode,
        learning_rate=observer_learning_rate,
        sharpening_factor=sharpening_factor,
        num_samples=num_samples,
        confidence=confidence
    )

    total_steps_taken = 0
    goals_reached = 0
    observer_belief_history = []

    for iteration in range(num_iterations):
        done = False
        if render:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Iteration {iteration + 1} / {num_iterations} ---")
            print(f"Goals reached so far: {goals_reached}")
            time.sleep(1.0)

        for step in range(max_steps):
            agent.update_internal_belief(obs)
            agent_pos, target_pos = tuple(obs['agent_pos']), tuple(obs['target_pos'])

            if render:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Iteration {iteration + 1} / {num_iterations} | Step {step + 1} ---")

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
            total_steps_taken += 1
            
            if render:
                print("\nActual Environment State:")
                env.render()
            
            obs = new_obs
            
            if render:
                time.sleep(time_delay)

            if terminated or truncated:

                if terminated:
                    goals_reached += 1

                if render:
                    print(f"\nEpisode finished: Target reached in {step + 1} steps.")
                
                # If not the last iteration, handle respawning based on toggles
                if iteration < num_iterations - 1:
                    respawned = False
                    if randomize_target_after_goal:
                        obs = env.respawn_target()
                        if render: print("Target has respawned.")
                        respawned = True
                    
                    if randomize_agent_after_goal:
                        new_obs, done = env.respawn_agent()
                        if done:
                            if render: print("No space left to respawn. Ending simulation.")
                            break 
                        obs = new_obs
                        if render: print("Agent has respawned.")
                        respawned = True

                    if render and respawned:
                        time.sleep(1)
                
                # Break from the inner step loop to start the next iteration
                break

            if truncated:
                if render:
                    end_reason = "Max steps reached"
                    print(f"\nEpisode finished: {end_reason} in {step + 1} steps.")
                # Break from inner loop to start next iteration
                break
        
        # Store the observer's belief map and the current agent/target positions at the end of the iteration
        observer_belief_history.append((observer.observer_belief_map.copy(), obs['agent_pos'], obs['target_pos']))

        if done:
            break

    final_agent_pos, final_target_pos = obs['agent_pos'], obs['target_pos']
    
    # --- Final Printouts ---
    if render:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "="*40 + "\n---      Observer Belief History     ---\n" + "="*40 + "\n")
        for i, (belief_map, agent_p, target_p) in enumerate(observer_belief_history):
            print(f"--- Belief Map After Iteration {i + 1} ---")
            _render_belief_map_with_chars(belief_map, env.grid_size, agent_p, target_p, observer.default_prob)
            print("\n")
            time.sleep(1)

        print("\n" + "="*40 + "\n---           FINAL STATE          ---\n" + "="*40 + "\n")
        print("Observer's Final Inferred Belief Map:")
        observer.render_belief(final_agent_pos, final_target_pos)
        print("\nAgent's Final Internal Belief Map (Ground Truth):")
        agent.render_internal_belief(final_agent_pos, final_target_pos)
        print("\nFinal Actual Environment State:")
        env.render()
    
    env.close()

    # Create a mask to exclude cells the agent has never seen.
    # The agent's map is 1.0 for wall, 0.0 for empty, and initial_prob for unseen.
    observed_mask = agent.internal_belief_map != true_wall_prob
    
    # Flatten the maps and apply the mask
    agent_map_flat = agent.internal_belief_map[observed_mask]
    observer_map_flat = observer.observer_belief_map[observed_mask]

    # 1. Mean Squared Error (on observed cells only)
    final_mse = np.mean((agent_map_flat - observer_map_flat)**2) if agent_map_flat.size > 0 else 0

    # 2. Jensen-Shannon Divergence
    # Add epsilon for numerical stability if we treat them as distributions
    agent_map_prob = agent_map_flat + 1e-9
    observer_map_prob = observer_map_flat + 1e-9
    final_js_divergence = jensenshannon(agent_map_prob, observer_map_prob, base=2) if agent_map_flat.size > 0 else 0

    # 3. Binary Cross-Entropy (Log Loss)
    epsilon = 1e-9
    observer_map_clipped = np.clip(observer_map_flat, epsilon, 1 - epsilon)
    log_loss = - (agent_map_flat * np.log(observer_map_clipped) + (1 - agent_map_flat) * np.log(1 - observer_map_clipped))
    final_log_loss = np.mean(log_loss) if agent_map_flat.size > 0 else 0

    metrics = {
        "total_steps_taken": total_steps_taken,
        "goals_reached": goals_reached,
        "final_mse": final_mse,
        "final_js_divergence": final_js_divergence,
        "final_log_loss": final_log_loss
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
        "agent_rationality": 1.0,
        "agent_utility_beta": 1.0,
        "sharpening_factor": 5.0, # Factor to sharpen the heuristic model's predictions (make it more confident)
        "observer_learning_rate": 0.5,
        "max_steps": 20,
        "render": True,
        "time_delay": 0.1,
        "num_samples": 4000, # Number of samples for generating possible states
        "num_iterations": 5, # Number of agent respawns
        "randomize_agent_after_goal": True, # Respawn the agent in a random cell
        "randomize_target_after_goal": True, # Respawn the target in a random cell
        "randomize_initial_placement": False, # Will use fixed positions in the custom map if set to False
        "convergence_threshold": 0.001, # Convergence threshold for RSA iterations
        "confidence": True, # Whether to use confidence in observer updates
        "max_cycle": 0,  # Length of the anti-cycling memory, set to 0 for no cycle-breaking
        "agent_sampling_mode": 'uniform', # Sampling mode for agent's belief map
        "observer_sampling_mode": 'uniform' # Sampling mode for observer's belief map
    }
    
    try:
        results = run_simulation(default_params)
        print("\n" + "="*40)
        print("---       SIMULATION COMPLETE      ---")
        print("="*40)
        print(f"  Goals Reached:      {results['goals_reached']} / {default_params['num_iterations']}")
        print(f"  Final MSE:          {results['final_mse']:.4f}")
        print(f"  Final JS Divergence: {results['final_js_divergence']:.4f}")
        print(f"  Final Log Loss:     {results['final_log_loss']:.4f}")
        print("="*40)
        
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()