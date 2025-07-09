import numpy as np
import torch
from collections import deque
from stable_baselines3 import PPO
from env import GridEnvironment
from utils import _render_belief_map_with_chars

class BaseAgent:
    """
    A base agent class that navigates the grid world. It uses a pre-trained
    policy to decide on actions based on its ground-truth local view.
    
    This class contains the common functionalities shared by all agents, such as
    updating beliefs from observations and rendering belief maps.
    """
    def __init__(self, env: GridEnvironment, model_path: str, initial_prob: float = 0.3, max_cycle: int = 0, sampling_mode: str = 'uniform'):
        """
        Initializes the BaseAgent.

        Args:
            env (GridEnvironment): The environment instance.
            model_path (str): Path to the pre-trained PPO model.
            initial_prob (float): The initial probability of a cell being a wall.
            max_cycle (int): The number of recent positions to remember to avoid cycles.
        """
        self.env = env
        self.num_actions = env.action_space.n
        self.internal_belief_map = np.full((env.grid_size, env.grid_size), initial_prob)
        self.default_prob = initial_prob
        self.position_history = deque(maxlen=max_cycle)
        self.sampling_mode = sampling_mode # Unused in this base class, but can be useful for subclasses
        
        # Action moves mapping, useful for subclasses and cycle-breaking
        self.action_moves = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
            4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)
        }
        
        # Initialize known empty cells
        if env.agent_pos is not None:
            self.internal_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.internal_belief_map[env.target_pos] = 0.0

        try:
            self.heuristic_model = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"Error loading heuristic model for BaseAgent: {e}")
            self.heuristic_model = None

    def _get_action_with_anti_cycle(self, sorted_actions, agent_pos):
        """
        Selects the best action that avoids recently visited positions.

        Args:
            sorted_actions (np.ndarray): An array of action indices, sorted from best to worst.
            agent_pos (tuple): The current position of the agent.

        Returns:
            int: The chosen action.
        """
        # If anti-cycling is disabled (maxlen is 0), just return the best action
        if self.position_history.maxlen == 0:
            return sorted_actions[0]

        # Try to choose the best action that doesn't lead to a recently visited state
        for action in sorted_actions:
            move = self.action_moves.get(action)
            if move:
                next_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])
                if next_pos not in self.position_history:
                    self.position_history.append(next_pos)
                    return action  # Found the best non-cyclic move
        
        # If all possible moves are cyclic, fallback to the best action
        return sorted_actions[0]

    def choose_action(self, observation):
        """
        Chooses an action directly from the policy network given the observation.
        
        Args:
            observation (dict): The agent's current observation from the environment.

        Returns:
            tuple[int, np.ndarray]: The chosen action and the probability distribution over actions.
        """
        if self.heuristic_model is None:
            return self.env.action_space.sample(), np.full(self.num_actions, 1.0 / self.num_actions)

        device = self.heuristic_model.device
        obs_tensor = {key: torch.as_tensor([value], device=device) for key, value in observation.items()}
        
        with torch.no_grad():
            dist = self.heuristic_model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy().squeeze()
            
        sorted_actions = np.argsort(probs)[::-1]
        agent_pos = tuple(observation['agent_pos'])
        
        chosen_action = self._get_action_with_anti_cycle(sorted_actions, agent_pos)

        return int(chosen_action), probs.flatten()

    def update_internal_belief(self, observation):
        """Updates the agent's private belief map with ground truth from its 5x5 view."""
        local_view = observation['local_view']
        agent_pos = observation['agent_pos']
        view_radius = local_view.shape[0] // 2

        for r_local in range(local_view.shape[0]):
            for c_local in range(local_view.shape[1]):
                r_global = agent_pos[0] + r_local - view_radius
                c_global = agent_pos[1] + c_local - view_radius
                if 0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size:
                    is_wall = (local_view[r_local, c_local] == self.env._wall_cell)
                    self.internal_belief_map[r_global, c_global] = 1.0 if is_wall else 0.0

    def render_internal_belief(self, agent_pos, target_pos):
        """Renders the agent's internal belief map."""
        print("Agent's Internal Belief Map (Ground Truth):")
        _render_belief_map_with_chars(
            self.internal_belief_map, self.env.grid_size, 
            agent_pos, target_pos, self.default_prob
        )

    def update_beliefs_after_action(self, observation, action):
        """
        Updates the agent's internal belief based on the action taken.
        
        Args:
            observation (dict): The agent's current observation from the environment.
            action (int): The action taken by the agent.
        """
        # Update the internal belief map with the new observation
        self.update_internal_belief(observation)