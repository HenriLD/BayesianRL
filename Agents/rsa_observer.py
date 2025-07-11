import numpy as np
from Agents.base_observer import BaseObserver

class RSAObserver(BaseObserver):
    """
    An observer that models the agent using Rational Speech Act (RSA) reasoning.
    It inherits from the BaseObserver and overrides the policy model to include
    iterative pragmatic reasoning, assuming the agent acts to be maximally
    informative within the constraints of its goal.
    """
    def __init__(self, env, model_path, agent_params, **kwargs):
        """
        Initializes the RSAObserver.

        Args:
            env (GridEnvironment): The environment instance.
            model_path (str): Path to the pre-trained PPO model.
            agent_params (dict): A dictionary containing RSA-specific parameters for
                                 modeling the agent, such as 'rationality', 
                                 'rsa_iterations', and 'utility_beta'.
            **kwargs: Additional arguments to be passed to the BaseObserver constructor.
        """
        # Pass common parameters to the parent constructor
        super().__init__(env=env, model_path=model_path, **kwargs)

        # RSA-specific parameters for modeling the agent
        self.alpha = agent_params.get('rationality', 1.0)
        self.rsa_iterations = agent_params.get('rsa_iterations', 5)
        self.convergence_threshold = agent_params.get('convergence_threshold', 0.001)
        
        # The 'beta' for utility calculation is also needed here for the RSA formula
        self.beta = agent_params.get('utility_beta', 1.0)

    def _get_action_likelihoods(self, world_utilities):
        """
        Overrides the base method to calculate action likelihoods P(a|s)
        using the full RSA reasoning process. This models a "pragmatic" agent.

        Args:
            world_utilities (np.ndarray): A matrix of utilities of shape (num_states, num_actions).

        Returns:
            np.ndarray: A matrix of action probabilities of shape (num_states, num_actions),
                        refined by RSA reasoning.
        """
        # S_0: Initialize a "literal speaker" that acts based on raw utilities.
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(world_utilities)
            speaker_probs = np.nan_to_num(exp_utilities / exp_utilities.sum(axis=1, keepdims=True), posinf=0)

        # Iteratively refine the speaker (agent model) and listener (observer model).
        for i in range(self.rsa_iterations):
            prev_speaker_probs = speaker_probs.copy()

            # L_k: The listener model infers P(state | action) using the current speaker model.
            # P(s|a) ∝ P(a|s) * P(s). Assuming uniform P(s), this simplifies.
            with np.errstate(divide='ignore', invalid='ignore'):
                listener_probs = speaker_probs / speaker_probs.sum(axis=0, keepdims=True)
                listener_probs = np.nan_to_num(listener_probs)

            # S_k+1: The speaker model updates. The utility of an action now includes
            # its "pragmatic" value: how well it helps the listener infer the correct state.
            with np.errstate(divide='ignore'): # Ignore log(0) warnings
                log_listener_probs = np.log(listener_probs)
            
            # The new utility combines pragmatic reasoning (log_listener) and original utility.
            # `alpha` controls the agent's rationality about being informative.
            combined_utility = (self.alpha * log_listener_probs) + (self.beta * world_utilities)
            
            # The new speaker model is a softmax over these combined utilities.
            with np.errstate(divide='ignore', invalid='ignore'):
                exp_combined = np.exp(combined_utility)
                speaker_probs = np.nan_to_num(exp_combined / exp_combined.sum(axis=1, keepdims=True), posinf=0)

            # Check for convergence to stop iterating early if the model is stable.
            if np.allclose(speaker_probs, prev_speaker_probs, atol=self.convergence_threshold):
                break
        
        return speaker_probs
