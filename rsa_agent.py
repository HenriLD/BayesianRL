import numpy as np
import itertools
import time
import os
import collections

# We import the GridEnvironment from the file created previously.
try:
    from env import GridEnvironment
except ImportError:
    print("Error: Could not import GridEnvironment.")
    print("Please ensure the updated GridEnvironment class is in 'env.py'.")
    exit()

def _chebyshev_distance(pos1, pos2):
    """Calculates Chebyshev distance (for grid with diagonal moves)."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

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


class RSAAgent:
    """
    An agent that uses its private 'internal_belief_map' to decide on an action.
    """
    def __init__(self, env: GridEnvironment, rsa_iterations: int = 3, rationality: float = 1.0, utility_beta: float = 1.5):
        self.env = env
        self.rsa_iterations = rsa_iterations
        self.alpha = rationality 
        self.beta = utility_beta
        self.num_actions = env.action_space.n
        self.internal_belief_map = np.full((env.grid_size, env.grid_size), 0.5)
        
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
                L_numerator = P_S_k.T * prior
                L_denominator = L_numerator.sum(axis=1, keepdims=True)
                P_L_k = np.nan_to_num(L_numerator / L_denominator)

            with np.errstate(divide='ignore', invalid='ignore'):
                log_P_L_k = np.log(P_L_k)
                log_P_L_k[np.isneginf(log_P_L_k)] = -1e9
                informational_utility = self.alpha * log_P_L_k.T
                combined_utility = world_utilities + informational_utility
                exp_combined_utility = np.exp(combined_utility)
                S_denominator = exp_combined_utility.sum(axis=1, keepdims=True)
                P_S_k = np.nan_to_num(exp_combined_utility / S_denominator)
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

class Observer:
    """
    An observer that only sees the agent's position and actions.
    It maintains its own belief map and updates it by inverting the agent's RSA model.
    """
    def __init__(self, env: GridEnvironment, agent_params: dict):
        self.env = env
        self.agent_params = agent_params
        self.num_actions = env.action_space.n
        self.observer_belief_map = np.full((env.grid_size, env.grid_size), 0.5)

        if env.agent_pos is not None:
            self.observer_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.observer_belief_map[env.target_pos] = 0.0

    def update_belief(self, agent_pos, target_pos, action):
        """Infers the state of the world based on the agent's action."""
        # 1. Generate states based on the *observer's* current belief.
        possible_states = _generate_possible_states(agent_pos, target_pos, self.observer_belief_map, self.env)
        if not possible_states:
            return
        
        num_local_states = len(possible_states)
        prior = np.full(num_local_states, 1.0 / num_local_states)

        # 2. Simulate the agent's reasoning process to get the policy it would use.
        world_utilities = _calculate_heuristic_utilities(agent_pos, target_pos, possible_states, self.env)
        
        # Re-run the same RSA reasoning the agent would have
        P_S_k = self._run_rsa_reasoning_for_observer(world_utilities, num_local_states)
        
        # 3. Invert the model: Calculate the listener's belief (posterior over states).
        with np.errstate(divide='ignore', invalid='ignore'):
            L_numerator = P_S_k.T * prior
            L_denominator = L_numerator.sum(axis=1, keepdims=True)
            P_L_k = np.nan_to_num(L_numerator / L_denominator)

        # The probability distribution over states, given the action the agent took.
        state_posterior = P_L_k[action, :]

        # 4. Update the observer's belief map using this posterior.
        self.observer_belief_map[agent_pos] = 0.0
        for r_local in range(3):
            for c_local in range(3):
                if r_local == 1 and c_local == 1: continue

                r_global = agent_pos[0] + r_local - 1
                c_global = agent_pos[1] + c_local - 1

                if 0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size:
                    prob_wall = sum(state_posterior[i] for i, s in enumerate(possible_states) if s[r_local, c_local] == self.env._wall_cell)
                    self.observer_belief_map[r_global, c_global] = prob_wall

    def _run_rsa_reasoning_for_observer(self, world_utilities, num_states):
        """A copy of the agent's reasoning process for the observer to use in its simulation."""
        prior = np.full(num_states, 1.0 / num_states)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_utilities = np.exp(self.agent_params['beta'] * world_utilities)
            sum_exp_utilities = exp_utilities.sum(axis=1, keepdims=True)
            P_S0 = np.nan_to_num(exp_utilities / sum_exp_utilities)
        
        P_S_k = P_S0
        for _ in range(self.agent_params['rsa_iterations']):
            with np.errstate(divide='ignore', invalid='ignore'):
                L_numerator = P_S_k.T * prior
                L_denominator = L_numerator.sum(axis=1, keepdims=True)
                P_L_k = np.nan_to_num(L_numerator / L_denominator)

            with np.errstate(divide='ignore', invalid='ignore'):
                log_P_L_k = np.log(P_L_k)
                log_P_L_k[np.isneginf(log_P_L_k)] = -1e9
                informational_utility = self.agent_params['alpha'] * log_P_L_k.T
                combined_utility = world_utilities + informational_utility
                exp_combined_utility = np.exp(combined_utility)
                S_denominator = exp_combined_utility.sum(axis=1, keepdims=True)
                P_S_k = np.nan_to_num(exp_combined_utility / S_denominator)
        return P_S_k

    def render_belief(self, agent_pos, target_pos):
        """Renders the observer's belief map."""
        display_map = self.observer_belief_map.copy()
        grid_str = ""
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                pos = (r, c)
                if pos == tuple(agent_pos):
                    grid_str += ' A   '
                elif pos == tuple(target_pos):
                    grid_str += ' T   '
                else:
                    prob = display_map[r, c]
                    grid_str += f'{prob:<4.2f} '
            grid_str += "\n"
        print(grid_str)

# --- Main execution block ---
if __name__ == '__main__':
    try:
        custom_map = [
            "##########",
            "#        #",
            "#A###### #",
            "#   ##   #",
            "# # ## T #",
            "# ## #   #",
            "# ## ### #",
            "#        #",
            "#        #",
            "##########",
        ]
        
        env = GridEnvironment(grid_map=custom_map, render_mode='human')
        obs, info = env.reset()
        
        agent = RSAAgent(env)
        observer = Observer(env, agent_params={
            'alpha': agent.alpha,
            'beta': agent.beta,
            'rsa_iterations': agent.rsa_iterations
        })

        max_steps = 100
        for step in range(max_steps):
            agent.update_internal_belief(obs)

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Step {step + 1} ---")
            
            agent_pos = tuple(obs['agent_pos'])
            target_pos = tuple(obs['target_pos'])

            if agent_pos == target_pos:
                print("Observer's Inferred Belief Map:")
                observer.render_belief(agent_pos, target_pos)
                print("\nActual Environment State:")
                env.render()
                print(f"\nTarget reached in {step} steps!")
                break

            action, action_probs = agent.choose_action(obs)
            
            observer.update_belief(agent_pos, target_pos, action)
            print("Observer's Inferred Belief Map:")
            observer.render_belief(agent_pos, target_pos)
            
            action_map = {
                0: "Up", 1: "Down", 2: "Left", 3: "Right",
                4: "Up-Left", 5: "Up-Right", 6: "Down-Left", 7: "Down-Right"
            }
            print("\nAction Probabilities (Agent's Perspective):")
            for i, prob in enumerate(action_probs): print(f"  - {action_map[i]:<10}: {prob:.4f}")
            print(f"Chosen Action: {action_map[action]}")

            new_obs, reward, terminated, truncated, info = env.step(action)
            
            print("\nActual Environment State:")
            env.render()
            obs = new_obs
            
            time.sleep(1.0)
            
            if truncated:
                print("\nEpisode truncated. Max steps reached.")
                break
        
        else:
             print("\nEpisode truncated. Max steps reached.")

        env.close()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
