import numpy as np
import itertools
import time
import os
import collections

# We import the GridEnvironment from the file created previously.
# This assumes the class is saved in a file named `env.py`.
try:
    from env import GridEnvironment
except ImportError:
    print("Error: Could not import GridEnvironment.")
    print("Please ensure the GridEnvironment class from the previous step is in a file named 'env.py' in the same directory.")
    exit()

class RSAAgent:
    """
    An agent that uses Rational Speech Act (RSA) inspired reasoning to navigate.
    
    The agent reasons about which action to take by considering how a "listener"
    would interpret that action. It maintains two belief maps:
    1. internal_belief_map: The agent's private knowledge, updated with its 5x5 view.
    2. observer_belief_map: An external view of what a third party could infer about
       the world knowing only the agent's actions and policy. This is used for rendering.
    """

    def __init__(self, env: GridEnvironment, rsa_iterations: int = 15, rationality: float = 1.0, num_state_samples: int = 100):
        """
        Initializes the RSA Agent.
        """
        self.env = env
        self.rsa_iterations = rsa_iterations
        self.alpha = rationality # Speaker rationality
        self.beta = 1.0          # Literal speaker's confidence in its heuristic
        self.num_state_samples = num_state_samples
        self.num_actions = env.action_space.n

        # Agent's internal belief, updated with ground truth from its senses. Used for sampling states.
        self.internal_belief_map = np.full((env.grid_size, env.grid_size), 0.5)
        
        # Observer's belief, updated with inferred probabilities. This is what gets rendered.
        self.observer_belief_map = np.full((env.grid_size, env.grid_size), 0.5)
        
        if env.agent_pos is not None:
            self.internal_belief_map[env.agent_pos] = 0.0
            self.observer_belief_map[env.agent_pos] = 0.0
        if env.target_pos is not None:
            self.internal_belief_map[env.target_pos] = 0.0
            self.observer_belief_map[env.target_pos] = 0.0

    def _manhattan_distance(self, pos1, pos2):
        """Calculates Manhattan distance between two points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _sample_states(self, agent_pos, true_state):
        """
        Samples a set of possible 5x5 states from the agent's *internal* belief map.
        The true observed state is always included to ensure the agent can reason about it.
        """
        sampled_states = [true_state]
        for _ in range(self.num_state_samples - 1):
            sample = np.full((5, 5), self.env._empty_cell)
            for r_local in range(5):
                for c_local in range(5):
                    if r_local == 2 and c_local == 2:
                        sample[2, 2] = self.env._agent_cell
                        continue
                    
                    r_global = agent_pos[0] + r_local - 2
                    c_global = agent_pos[1] + c_local - 2

                    if not (0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size):
                        sample[r_local, c_local] = self.env._wall_cell
                        continue
                    
                    prob_wall = self.internal_belief_map[r_global, c_global]
                    if np.random.rand() < prob_wall:
                        sample[r_local, c_local] = self.env._wall_cell
                    else:
                        sample[r_local, c_local] = self.env._empty_cell
            sampled_states.append(sample)
        return sampled_states

    def _literal_speaker(self, agent_pos, target_pos, states):
        """
        Calculates P_S0(action | state) using a path-based heuristic.
        A "state" is a full 5x5 hypothetical grid.
        """
        num_states = len(states)
        probs = np.zeros((num_states, self.num_actions))
        current_dist_to_target = self._manhattan_distance(agent_pos, target_pos)

        for i, state in enumerate(states):
            hypothetical_walls = set()
            for r_local in range(5):
                for c_local in range(5):
                    if state[r_local, c_local] == self.env._wall_cell:
                        r_global = agent_pos[0] + r_local - 2
                        c_global = agent_pos[1] + c_local - 2
                        hypothetical_walls.add((r_global, c_global))

            action_utilities = []
            for action_idx in range(self.num_actions):
                move = [(-1, 0), (1, 0), (0, -1), (0, 1)][action_idx]
                start_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])

                if (start_pos in hypothetical_walls or 
                    not (0 <= start_pos[0] < self.env.grid_size and 0 <= start_pos[1] < self.env.grid_size)):
                    action_utilities.append(-np.inf)
                    continue

                q = collections.deque([(start_pos, 0)])
                visited = {start_pos}
                path_endpoints = {start_pos}
                max_depth = 4

                while q:
                    curr_pos, depth = q.popleft()
                    if depth >= max_depth: continue
                    for next_move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        next_pos = (curr_pos[0] + next_move[0], curr_pos[1] + next_move[1])
                        if (next_pos not in visited and next_pos not in hypothetical_walls and
                            0 <= next_pos[0] < self.env.grid_size and 0 <= next_pos[1] < self.env.grid_size):
                            visited.add(next_pos)
                            q.append((next_pos, depth + 1))
                            path_endpoints.add(next_pos)
                
                min_dist_on_path = min([self._manhattan_distance(p, target_pos) for p in path_endpoints]) if path_endpoints else current_dist_to_target
                utility = current_dist_to_target - min_dist_on_path
                action_utilities.append(utility)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                utilities = np.array(action_utilities)
                if not np.any(np.isfinite(utilities)):
                    probs[i, :] = 1.0 / self.num_actions
                else:
                    exp_utilities = np.exp(self.beta * utilities)
                    probs[i, :] = exp_utilities / np.sum(exp_utilities)
        return np.nan_to_num(probs)

    def choose_action(self, observation):
        """
        Chooses an action by running RSA reasoning over a set of sampled states.
        """
        agent_pos = tuple(observation['agent_pos'])
        target_pos = tuple(observation['target_pos'])
        s_true = observation['local_view']
        
        possible_states = self._sample_states(agent_pos, s_true)
        num_local_states = len(possible_states)
        s_true_idx = 0

        prior = np.full(num_local_states, 1.0 / num_local_states)
        P_S0 = self._literal_speaker(agent_pos, target_pos, possible_states)
        P_S_k = P_S0
        P_L_k = np.zeros((self.num_actions, num_local_states))

        for _ in range(self.rsa_iterations):
            with np.errstate(divide='ignore', invalid='ignore'):
                L_numerator = P_S_k.T * prior
                L_denominator = L_numerator.sum(axis=1, keepdims=True)
                P_L_k = np.nan_to_num(L_numerator / L_denominator)

            with np.errstate(divide='ignore', invalid='ignore'):
                log_P_L_k = np.log(P_L_k)
                log_P_L_k[np.isneginf(log_P_L_k)] = -1e9
                S_numerator = np.exp(self.alpha * log_P_L_k.T)
                S_denominator = S_numerator.sum(axis=1, keepdims=True)
                P_S_k = np.nan_to_num(S_numerator / S_denominator)
        
        print("naive_action_probs = ", P_S0[s_true_idx, :])
        final_action_probs = P_S_k[s_true_idx, :]

        # Choose the best action, breaking ties randomly.
        max_prob = np.max(final_action_probs)
        best_action_indices = np.where(final_action_probs == max_prob)[0]
        action = np.random.choice(best_action_indices)
        
        final_listener_belief = P_L_k[action, :]

        return int(action), final_listener_belief, possible_states, final_action_probs

    def update_internal_belief(self, observation):
        """Updates the agent's private belief map with ground truth from its 5x5 view."""
        local_view = observation['local_view']
        agent_pos = observation['agent_pos']
        
        for r_local in range(5):
            for c_local in range(5):
                r_global = agent_pos[0] + r_local - 2
                c_global = agent_pos[1] + c_local - 2
                if 0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size:
                    cell_value = local_view[r_local, c_local]
                    self.internal_belief_map[r_global, c_global] = 1.0 if cell_value == self.env._wall_cell else 0.0

    def update_observer_belief(self, agent_pos, listener_belief, states):
        """
        Updates the observer's belief map based on the agent's action (via the listener belief).
        This represents what a third party could infer.
        """
        # An observer knows the agent isn't on a wall.
        self.observer_belief_map[agent_pos] = 0.0

        # Update probabilities of surrounding cells based on the listener's inference.
        for r_local in range(5):
            for c_local in range(5):
                if r_local == 2 and c_local == 2: continue

                r_global = agent_pos[0] + r_local - 2
                c_global = agent_pos[1] + c_local - 2

                if 0 <= r_global < self.env.grid_size and 0 <= c_global < self.env.grid_size:
                    # Calculate the inferred probability of this cell being a wall
                    prob_wall = sum(listener_belief[i] for i, s in enumerate(states) if s[r_local, c_local] == self.env._wall_cell)
                    self.observer_belief_map[r_global, c_global] = prob_wall

    def render_observer_belief(self, agent_pos, target_pos):
        """
        Renders the observer's belief map.
        """
        display_map = self.observer_belief_map
        prob_ranges = {(0.8, 1.0): '███', (0.2, 0.8): '▓▓▓', (0.0, 0.2): '▒▒▒'}
        grid_str = ""
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                pos = (r, c)
                if pos == tuple(agent_pos): grid_str += ' A '
                elif pos == tuple(target_pos): grid_str += ' T '
                else:
                    prob = display_map[r, c]
                    char_to_render = None
                    if prob == 1.0: char_to_render = '███'
                    elif prob == 0.5: char_to_render = ' ? '
                    elif prob == 0.0: char_to_render = ' . '
                    else:
                        for (low, high), char in prob_ranges.items():
                            if low < prob <= high:
                                char_to_render = char
                                break
                    grid_str += char_to_render if char_to_render else ' ? '
            grid_str += "\n"
        print(grid_str)

# --- Main execution block ---
if __name__ == '__main__':
    try:
        custom_map = [
            "##########",
            "#        #",
            "#A# #### #",
            "#    #   #",
            "# #   T  #",
            "# #      #",
            "# ###### #",
            "#        #",
            "#        #",
            "##########",
        ]
        
        env = GridEnvironment(grid_map=custom_map, render_mode='human')
        obs, info = env.reset()
        
        agent = RSAAgent(env, num_state_samples=100)
        # Initial update for both belief maps
        agent.update_internal_belief(obs)
        agent.update_observer_belief(obs['agent_pos'], np.full(agent.num_state_samples, 1./agent.num_state_samples), agent._sample_states(obs['agent_pos'], obs['local_view']))


        max_steps = 100
        for step in range(max_steps):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Step {step + 1} ---")
            
            action, listener_belief, states, action_probs = agent.choose_action(obs)
            
            # Update and render the observer's map BEFORE the move
            agent.update_observer_belief(obs['agent_pos'], listener_belief, states)
            print("Observer's Inferred Belief Map:")
            agent.render_observer_belief(obs['agent_pos'], obs['target_pos'])
            
            action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
            print("\nAction Probabilities:")
            for i, prob in enumerate(action_probs): print(f"  - {action_map[i]}: {prob:.4f}")
            print(f"Chosen Action: {action_map[action]}")

            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update agent's internal knowledge with new ground truth
            agent.update_internal_belief(new_obs)
            
            print("\nActual Environment State:")
            env.render()

            obs = new_obs
            
            time.sleep(0.8)
            
            if terminated:
                print(f"\nTarget reached in {step + 1} steps!")
                break
            if truncated:
                print("\nEpisode truncated. Max steps reached.")
                break

        env.close()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This might be due to the terminal window being too small to display the output.")

