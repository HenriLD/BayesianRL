import numpy as np
from rsa_observer import RSAObserver
from base_agent import BaseAgent
from utils import _generate_possible_states, _calculate_heuristic_utilities, _render_belief_map_with_chars

class RSAAgent(BaseAgent):
    """
    An agent that uses Rational Speech Act (RSA) reasoning to decide on an action.
    It considers not only what action is best for it, but also how an observer
    would interpret that action, aiming to be pragmatically informative.
    """
    def __init__(self, env, model_path, rsa_params, **kwargs):
        """
        Initializes the RSAAgent.

        Args:
            env (GridEnvironment): The environment instance.
            model_path (str): Path to the pre-trained PPO model.
            rsa_params (dict): Dictionary of RSA-specific parameters like rationality,
                               iterations, beta, etc.
            **kwargs: Additional arguments for the BaseAgent constructor.
        """
        super().__init__(env=env, model_path=model_path, **kwargs)
        
        # RSA-specific parameters
        self.alpha = rsa_params.get('rationality', 1.0)
        self.beta = rsa_params.get('utility_beta', 1.0)
        self.rsa_iterations = rsa_params.get('rsa_iterations', 5)
        self.sharpening_factor = rsa_params.get('sharpening_factor', 1.0)
        self.num_samples = rsa_params.get('num_samples', 5000)
        self.convergence_threshold = rsa_params.get('convergence_threshold', 0.001)
        self.sampling_mode = rsa_params.get('sampling_mode', 'uniform')

        if self. sampling_mode == "belief_based":
            observer_kwargs = {k: v for k, v in kwargs.items() if k != 'max_cycle'}
            self.internal_observer = RSAObserver(env, model_path, agent_params=rsa_params, **observer_kwargs)

    def _run_rsa_reasoning(self, world_utilities):
        """Performs the alternating speaker-listener RSA calculation."""
        # S_0: Literal speaker model, P(a|s) = softmax(beta * U(s,a))
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(world_utilities)
            speaker_probs = np.nan_to_num(exp_utilities / exp_utilities.sum(axis=1, keepdims=True), posinf=0)

        for _ in range(self.rsa_iterations):
            prev_speaker_probs = speaker_probs.copy()
            # L_k: Listener model, P(s|a) ∝ P(a|s)
            with np.errstate(divide='ignore', invalid='ignore'):
                listener_probs = speaker_probs / speaker_probs.sum(axis=0, keepdims=True)
                listener_probs = np.nan_to_num(listener_probs)

            # S_k+1: Speaker updates utility based on listener's understanding
            with np.errstate(divide='ignore'):
                log_listener_probs = np.log(listener_probs)
            
            combined_utility = (self.alpha * log_listener_probs) + (self.beta * world_utilities)

            with np.errstate(divide='ignore', invalid='ignore'):
                exp_combined = np.exp(combined_utility)
                speaker_probs = np.nan_to_num(exp_combined / exp_combined.sum(axis=1, keepdims=True), posinf=0)

            if np.allclose(speaker_probs, prev_speaker_probs, atol=self.convergence_threshold):
                break
        
        return speaker_probs

    def choose_action(self, observation):
        """
        Overrides the base method to choose an action using RSA reasoning.
        
        Args:
            observation (dict): The agent's current observation from the environment.

        Returns:
            tuple[int, np.ndarray]: The chosen action and the final probability distribution.
        """
        agent_pos = tuple(observation['agent_pos'])
        target_pos = tuple(observation['target_pos'])
        s_true = observation['local_view']

        # Use the internal observer's belief for sampling if in belief-based mode
        belief_map_for_sampling = (
            self.internal_observer.observer_belief_map 
            if self.sampling_mode == 'belief_based' 
            else self.internal_belief_map
        )

        possible_states = _generate_possible_states(
            agent_pos, target_pos, belief_map_for_sampling, self.env, 
            true_state_view=s_true, sampling_mode=self.sampling_mode,
            uniform_prob=self.default_prob, num_samples=self.num_samples
        )

        try:
            s_true_idx = next(i for i, state in enumerate(possible_states) if np.array_equal(state, s_true))
        except StopIteration:
            raise RuntimeError("True state not found in generated possible states.")

        world_utilities = _calculate_heuristic_utilities(
            agent_pos, target_pos, possible_states, self.env, 
            heuristic_model=self.heuristic_model, sharpening_factor=self.sharpening_factor
        )

        # Run RSA reasoning to get the pragmatic speaker probabilities
        P_S_k = self._run_rsa_reasoning(world_utilities)
        final_action_probs = P_S_k[s_true_idx, :]

        # Use the cycle-breaking logic from the parent class, but with RSA probabilities
        sorted_actions = np.argsort(final_action_probs)[::-1]
        
        chosen_action = self._get_action_with_anti_cycle(sorted_actions, agent_pos)

        return int(chosen_action), final_action_probs
    
    def update_beliefs_after_action(self, observation, action):
        """
        Updates both the agent's internal ground-truth belief and the belief
        of its internal observer model based on the action taken.
        """
        # 1. Update the agent's own ground-truth map from the new observation
        self.update_internal_belief(observation)
        
        # 2. Update the internal observer's belief based on the action taken
        if self.sampling_mode == 'belief_based':
            agent_pos = tuple(observation['agent_pos'])
            target_pos = tuple(observation['target_pos'])
            self.internal_observer.update_belief(agent_pos, target_pos, action)