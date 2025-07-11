import numpy as np
import random
from scipy.stats import entropy
from env import GridEnvironment
from stable_baselines3 import PPO
from utils.utils import (
    _render_belief_map_with_chars,
    _calculate_heuristic_utilities,
    _generate_possible_states,
)

random.seed(42)  # For reproducibility
VIEW_SIZE = 5

class BaseObserver:
    """
    A base observer class that infers an agent's belief state by observing its actions.
    It maintains a belief map and updates it by inverting a model of the agent's policy.
    
    This base class uses a simple "literal" policy model, which assumes the agent
    chooses actions via a softmax function over the learned utilities.
    """
    def __init__(self, env, model_path, policy_beta=1.0, initial_prob=0.25, 
                 learning_rate=0.5, sharpening_factor=1.0, num_samples=50000, 
                 use_confidence=False, sampling_mode='uniform'):
        """
        Initializes the BaseObserver.

        Args:
            env (GridEnvironment): The environment instance.
            model_path (str): Path to the pre-trained PPO model.
            policy_beta (float): Rationality parameter for the literal agent model.
            initial_prob (float): The initial probability of a cell being a wall.
            learning_rate (float): Rate at which the observer updates its beliefs.
            sharpening_factor (float): Multiplier for the heuristic model's value predictions.
            num_samples (int): Number of world states to sample for inference.
            use_confidence (bool): Whether to scale the learning rate by confidence.
            sampling_mode (str): Method for sampling possible world states ('uniform' or 'belief_based').
        """
        self.env = env
        self.beta = policy_beta  # For the literal listener model
        self.learning_rate = learning_rate
        self.observer_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.view_radius = VIEW_SIZE // 2
        self.default_prob = initial_prob
        self.model_path = model_path
        self.sharpening_factor = sharpening_factor
        self.num_samples = num_samples
        self.use_confidence = use_confidence
        self.sampling_mode = sampling_mode
        self.confidence_scores = []

        # Initialize known empty cells
        if env.agent_pos is not None:
            self.observer_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.observer_belief_map[env.target_pos] = 0.0

        try:
            self.heuristic_model = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"Error loading heuristic model for observer: {e}")
            self.heuristic_model = None

    def _get_action_likelihoods(self, world_utilities):
        """
        Calculates action probabilities based on utilities: P(a|s) = softmax(beta * U(s,a)).
        This is the observer's "literal" model of the agent's policy. This method
        is intended to be overridden by more complex observer models (like RSA).
        
        Args:
            world_utilities (np.ndarray): A matrix of utilities of shape (num_states, num_actions).

        Returns:
            np.ndarray: A matrix of action probabilities of shape (num_states, num_actions).
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.beta * world_utilities)
            action_probs = np.nan_to_num(exp_utilities / exp_utilities.sum(axis=1, keepdims=True))
        return action_probs

    def _compute_local_belief(self, agent_pos, target_pos, action):
        """
        Computes the inferred 5x5 local probability map based on the agent's action.
        This involves generating hypotheses (possible states), evaluating them using the
        agent policy model, and calculating a posterior distribution over those states.

        Args:
            agent_pos (tuple): The agent's current position.
            target_pos (tuple): The target's current position.
            action (int): The action the agent took.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the inferred local wall
                                      probability map and the confidence of the inference.
        """
        possible_states = _generate_possible_states(
            agent_pos, target_pos, self.observer_belief_map, self.env,
            sampling_mode=self.sampling_mode, uniform_prob=self.default_prob, num_samples=self.num_samples
        )
        
        if not possible_states:
            return np.zeros((VIEW_SIZE, VIEW_SIZE)), 1.0

        num_local_states = len(possible_states)
        prior = np.full(num_local_states, 1.0 / num_local_states)

        world_utilities = _calculate_heuristic_utilities(
            agent_pos, target_pos, possible_states, self.env, 
            heuristic_model=self.heuristic_model, sharpening_factor=self.sharpening_factor
        )
        
        # Get P(a|s) from the agent policy model (literal or RSA)
        action_likelihoods = self._get_action_likelihoods(world_utilities)

        # Bayes' Rule: P(s|a) ∝ P(a|s) * P(s)
        likelihood = action_likelihoods[:, action]
        posterior = likelihood * prior
        
        posterior_sum = posterior.sum()
        state_posterior = posterior / posterior_sum if posterior_sum > 0 else prior
        
        # --- Confidence Calculation (based on entropy of the posterior) ---
        posterior_entropy = entropy(state_posterior)
        max_entropy = np.log(num_local_states) if num_local_states > 1 else 1.0
        confidence = np.clip(1.0 - (posterior_entropy / max_entropy), 0.0, 1.0)
        self.confidence_scores.append(confidence)
        
        # --- Wall Probability Calculation ---
        possible_states_np = np.array(possible_states)
        is_wall_mask = (possible_states_np == self.env._wall_cell)
        local_wall_probs = np.einsum('i,ijk->jk', state_posterior, is_wall_mask)
        
        return local_wall_probs, confidence

    def _update_global_belief(self, local_wall_probs, agent_pos, confidence):
        """
        Updates the global belief map using the newly inferred local beliefs.
        The update is a weighted average between the old belief and the new inference.
        
        Args:
            local_wall_probs (np.ndarray): The 5x5 inferred wall probability map.
            agent_pos (tuple): The agent's current position.
            confidence (float): The confidence score of the local inference.
        """
        self.observer_belief_map[agent_pos] = 0.0

        effective_learning_rate = self.learning_rate
        if self.use_confidence:
            # Scale learning rate: 1x at 0 confidence, up to 5x at 1 confidence
            effective_learning_rate *= (1 + 4 * confidence) 

        # Define global and local boundaries for the update
        r_global_start = agent_pos[0] - self.view_radius
        r_global_end = agent_pos[0] + self.view_radius + 1
        c_global_start = agent_pos[1] - self.view_radius
        c_global_end = agent_pos[1] + self.view_radius + 1
        
        # Clip boundaries to be within the grid
        r_clip_start = max(0, r_global_start)
        r_clip_end = min(self.env.grid_size, r_global_end)
        c_clip_start = max(0, c_global_start)
        c_clip_end = min(self.env.grid_size, c_global_end)

        # Get the corresponding slices from the global map and local inference
        global_slice = self.observer_belief_map[r_clip_start:r_clip_end, c_clip_start:c_clip_end]
        local_slice = local_wall_probs[
            r_clip_start - r_global_start : r_clip_end - r_global_start,
            c_clip_start - c_global_start : c_clip_end - c_global_start
        ]

        # Perform the belief update and clip probabilities to [0, 1]
        new_global_prob = global_slice + effective_learning_rate * (local_slice - global_slice)
        self.observer_belief_map[r_clip_start:r_clip_end, c_clip_start:c_clip_end] = np.clip(new_global_prob, 0.0, 1.0)

    def update_belief(self, agent_pos, target_pos, action):
        """
        Orchestrates the full, two-step belief update process.
        
        Args:
            agent_pos (tuple): The agent's current position.
            target_pos (tuple): The target's current position.
            action (int): The action the agent took.

        Returns:
            np.ndarray: The inferred local wall probability map.
        """
        local_probs, conf = self._compute_local_belief(agent_pos, target_pos, action)
        self._update_global_belief(local_probs, agent_pos, conf)
        return local_probs

    def render_belief(self, agent_pos, target_pos):
        """Renders the observer's global belief map to the console."""
        _render_belief_map_with_chars(
            self.observer_belief_map, self.env.grid_size, 
            agent_pos, target_pos, self.default_prob
        )
