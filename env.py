import gymnasium as gym
import numpy as np
import typing

class GridEnvironment(gym.Env):
    """
    A 2D Grid World environment for a reinforcement learning agent.

    The agent's goal is to navigate from a starting position to a target position
    while avoiding walls. The environment can be initialized from a string-based map
    or as a randomly generated grid.

    Attributes:
        grid_size (int): The size of the square grid.
        agent_pos (np.ndarray): The current (row, col) coordinates of the agent.
        target_pos (np.ndarray): The (row, col) coordinates of the target.
        walls (list[tuple[int, int]]): A list of coordinates for the walls.
        grid (np.ndarray): The numerical representation of the grid state.
        action_space (gym.spaces.Discrete): The space of possible actions.
        observation_space (gym.spaces.Dict): The observation space for the agent.
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, grid_map: list[str] = None, grid_size: int = 30, wall_percentage: float = 0.2, render_mode: str = 'human'):
        """
        Initializes the Grid Environment.

        Args:
            grid_map (list[str], optional): A list of strings representing the grid layout.
                ' ' for empty, '#' for wall, 'A' for agent, 'T' for target.
                If provided, `grid_size` and `wall_percentage` are ignored.
            grid_size (int): The width and height of the grid for random generation.
            wall_percentage (float): The percentage of the grid to be filled with walls.
            render_mode (str): The mode for rendering the environment ('human' or 'ansi').
        """
        super().__init__()

        self.grid_map = grid_map
        if self.grid_map:
            # Ensure the map is square
            assert all(len(row) == len(self.grid_map) for row in self.grid_map), "Grid map must be square."
            self.grid_size = len(self.grid_map)
        else:
            self.grid_size = grid_size
        
        self.wall_percentage = wall_percentage
        self.render_mode = render_mode

        # Define spaces
        # 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = gym.spaces.Discrete(4)

        # The observation space is a dictionary containing:
        # 1. 'local_view': A 5x5 area around the agent.
        # 2. 'agent_pos': The agent's own coordinates.
        # 3. 'target_pos': The target's coordinates.
        self.observation_space = gym.spaces.Dict({
            'local_view': gym.spaces.Box(low=0, high=4, shape=(5, 5), dtype=np.int32),
            'agent_pos': gym.spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
            'target_pos': gym.spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
        })

        # Internal state
        self.agent_pos = None
        self.target_pos = None
        self.walls = []
        self._max_steps = self.grid_size * self.grid_size # Set a limit for steps per episode
        self._current_step = 0

        # Grid cell values for representation
        self._empty_cell = 0
        self._wall_cell = 1
        self._agent_cell = 2
        self._target_cell = 3

    def _generate_grid_from_map(self):
        """Generates the grid from the predefined string map."""
        self.grid = np.full((self.grid_size, self.grid_size), self._empty_cell)
        self.walls = []
        agent_found = False
        target_found = False

        for r, row_str in enumerate(self.grid_map):
            for c, char in enumerate(row_str):
                pos = (r, c)
                if char == '#':  # Wall
                    self.walls.append(pos)
                    self.grid[pos] = self._wall_cell
                elif char == 'A':  # Agent
                    self.agent_pos = pos
                    self.grid[pos] = self._agent_cell
                    agent_found = True
                elif char == 'T':  # Target
                    self.target_pos = pos
                    self.grid[pos] = self._target_cell
                    target_found = True
        
        if not agent_found:
            raise ValueError("Grid map must contain an 'A' to specify the agent's start position.")
        if not target_found:
            raise ValueError("Grid map must contain a 'T' to specify the target's position.")

    def _generate_random_grid(self):
        """Generates a new grid with random walls, agent, and target positions."""
        self.grid = np.full((self.grid_size, self.grid_size), self._empty_cell)
        
        self.walls = []
        num_walls = int(self.grid_size * self.grid_size * self.wall_percentage)
        for _ in range(num_walls):
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            self.walls.append(pos)
            self.grid[pos] = self._wall_cell

        self.agent_pos = self._place_entity(exclude=self.walls)
        self.target_pos = self._place_entity(exclude=self.walls + [self.agent_pos])

        self.grid[self.agent_pos] = self._agent_cell
        self.grid[self.target_pos] = self._target_cell

    def _place_entity(self, exclude: list[tuple[int, int]]) -> tuple[int, int]:
        """Places an entity on a random valid (not in the exclude list) cell."""
        while True:
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos not in exclude:
                return pos

    def _get_observation(self) -> dict:
        """Gets the current observation for the agent."""
        padded_grid = np.pad(self.grid, pad_width=2, mode='constant', constant_values=self._wall_cell)
        padded_agent_pos = np.array(self.agent_pos) + 2
        r, c = padded_agent_pos
        local_view = padded_grid[r-2:r+3, c-2:c+3]

        return {
            'local_view': local_view.astype(np.int32),
            'agent_pos': np.array(self.agent_pos, dtype=np.int32),
            'target_pos': np.array(self.target_pos, dtype=np.int32)
        }

    def reset(self, seed: typing.Optional[int] = None, options: typing.Optional[dict] = None) -> tuple[dict, dict]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        if self.grid_map:
            self._generate_grid_from_map()
        else:
            self._generate_random_grid()
        
        self._current_step = 0
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Executes one time step within the environment."""
        old_pos = self.agent_pos
        new_pos = list(self.agent_pos)

        if action == 0: new_pos[0] -= 1  # Up
        elif action == 1: new_pos[0] += 1 # Down
        elif action == 2: new_pos[1] -= 1 # Left
        elif action == 3: new_pos[1] += 1 # Right

        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size) or tuple(new_pos) in self.walls:
            new_pos = old_pos

        self.grid[old_pos] = self._empty_cell
        self.agent_pos = tuple(new_pos)
        self.grid[self.agent_pos] = self._agent_cell
        self.grid[self.target_pos] = self._target_cell

        terminated = (self.agent_pos == self.target_pos)
        reward = 100.0 if terminated else -0.1

        self._current_step += 1
        truncated = (self._current_step >= self._max_steps)

        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the environment."""
        if self.render_mode == 'ansi':
            return self._render_to_string()
        elif self.render_mode == 'human':
            print(self._render_to_string())

    def _render_to_string(self) -> str:
        """Helper method to create a string representation of the grid."""
        if self.agent_pos is None:
            return "Grid not initialized. Call reset() first."

        char_map = {
            self._empty_cell: ' . ',
            self._wall_cell:  '███',
            self._agent_cell: ' A ',
            self._target_cell:' T '
        }
        
        grid_str = ""
        # Create a temporary grid for rendering to correctly show agent and target
        render_grid = np.copy(self.grid)
        render_grid[self.agent_pos] = self._agent_cell
        render_grid[self.target_pos] = self._target_cell

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                grid_str += char_map[render_grid[r, c]]
            grid_str += "\n"
        
        grid_str += f"Step: {self._current_step}\n"
        grid_str += f"Agent Position: {self.agent_pos}\n"
        grid_str += f"Target Position: {self.target_pos}\n"
        return grid_str

    def close(self):
        """Performs any necessary cleanup."""
        pass

# Example of how to use the environment
if __name__ == '__main__':
    # --- Example 1: Using a predefined map ---
    print("--- Running example with a custom map ---")
    custom_map = [
        "##########",
        "#A       #",
        "# ###### #",
        "#        #",
        "#   #    #",
        "#   #    #",
        "#   #    #",
        "#        #",
        "#       T#",
        "##########",
    ]
    env_from_map = GridEnvironment(grid_map=custom_map, render_mode='human')
    obs, info = env_from_map.reset()
    env_from_map.render()
    
    for _ in range(5):
        action = env_from_map.action_space.sample()
        obs, reward, terminated, truncated, info = env_from_map.step(action)
        print(f"Action: {action}, Reward: {reward}")
        env_from_map.render()
        if terminated or truncated:
            break
    env_from_map.close()

    print("\n\n--- Running example with a randomly generated grid ---")
    # --- Example 2: Using random generation ---
    env_random = GridEnvironment(grid_size=10, wall_percentage=0.3, render_mode='human')
    obs, info = env_random.reset()
    env_random.render()
    env_random.close()
