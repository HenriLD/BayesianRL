import os
import random
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Attempt to import the custom environment
try:
    from env import GridEnvironment
except ImportError:
    print("Error: Could not import GridEnvironment.")
    print("Please ensure 'env.py' is in the same directory.")
    exit()

# --- Predefined Maps for Training ---
# Define a set of fixed map layouts. The agent 'A' and target 'T' will be
# placed randomly in empty spaces at the start of each training episode.
PREDEFINED_MAPS = [
    [
        "##############",
        "#      #     #",
        "#   #     # ##",
        "#  ######  # #",
        "#      #    ##",
        "######## #   #",
        "#  #  ##     #",
        "# #   #      #",
        "#   ###     ##",
        "###   # #    #",
        "#       #    #",
        "#  #  # #    #",
        "#  #  #      #",
        "##############",
    ],
    [
        "##############",
        "#  #         #",
        "#  #         #",
        "#  # ### ### #",
        "#    #   #   #",
        "#    #   #   #",
        "#    #   #   #",
        "#            #",
        "#  #         #",
        "#  #    ### ##",
        "#  #     #   #",
        "#      ###   #",
        "#  #         #",
        "##############",
    ],
    [
        "##############",
        "#      #     #",
        "#  ##    #   #",
        "#   ## # #   #",
        "# #    ###   #",
        "# # #        #",
        "#   #   ### ##",
        "## ## ##     #",
        "#           ##",
        "# ####  ## ###",
        "#    #  #  # #",
        "# ##    #  # #",
        "# #  ####    #",
        "##############",
    ],
    [
        "##############",
        "#      #     #",
        "#  #         #",
        "#  ##### #   #",
        "#  #     #   #",
        "#     ####  ##",
        "#  #        ##",
        "#  #   #   # #",
        "# #### ##  # #",
        "#   #   #    #",
        "#   #      # #",
        "#  ## #  # # #",
        "#     #    # #",
        "##############",
    ]
]

# --- Custom Training Environment ---
class TrainingEnv(GridEnvironment):
    """
    A wrapper around the GridEnvironment that, on each reset, selects a random
    map layout, and then randomly places both the agent 'A' and the target 'T'
    in any of the available empty spaces.
    """
    def __init__(self, maps):
        self.maps = maps
        # Initialize with the first map structure to set up observation/action spaces
        super().__init__(grid_map=self.maps[0])

    def reset(self, seed=None, options=None):
        # This must be called first to seed the parent's RNG
        super(GridEnvironment, self).reset(seed=seed)

        # 1. Randomly select a map template from the predefined list
        selected_map_template = random.choice(self.maps)
        
        # Create a mutable copy (list of lists of characters) to modify
        new_map = [list(row) for row in selected_map_template]

        # 2. Find all valid spawn locations (empty spaces ' ' and the original 'T' position)
        valid_spawn_points = []
        for r, row in enumerate(new_map):
            for c, char in enumerate(row):
                if char == ' ' or char == 'T':
                    valid_spawn_points.append((r, c))
                    # Ensure the original 'T' position is treated as an empty space for placement
                    if char == 'T':
                        new_map[r][c] = ' '

        # 3. Ensure there are at least two empty spots for the agent and target
        if len(valid_spawn_points) < 2:
            raise ValueError("A predefined map has fewer than two empty spaces for the agent and target.")
        
        # 4. Choose two distinct random positions for the agent and target
        agent_pos, target_pos = random.sample(valid_spawn_points, 2)
        
        # 5. Place the agent 'A' and target 'T' in the chosen positions
        new_map[agent_pos[0]][agent_pos[1]] = 'A'
        new_map[target_pos[0]][target_pos[1]] = 'T'

        # 6. Set this newly configured map as the one to be used by the environment
        self.grid_map = ["".join(row) for row in new_map]

        # 7. Now, call the parent's logic to build the grid state from self.grid_map.
        # This populates self.grid, self.walls, self.agent_pos, self.target_pos etc.
        self._generate_grid_from_map()
        
        # 8. Finalize the reset process
        self._current_step = 0
        observation = self._get_observation()
        info = {}
        
        return observation, info

def train_agent(training_steps: int = 100_000, save_path: str = "heuristic_agent.zip"):
    """
    Initializes the environment, trains a PPO agent, and saves the model.

    Args:
        training_steps (int): The total number of steps to train the agent for.
        save_path (str): The file path to save the trained model to.
    """
    print("Initializing environment for training...")
    # The TrainingEnv now randomizes both agent and target positions on predefined layouts.
    env = TrainingEnv(maps=PREDEFINED_MAPS)

    # It's good practice to check that the custom environment follows the gym API
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return

    print("Initializing PPO model...")
    # 'MultiInputPolicy' is used because our environment's observation space is a dictionary.
    model = PPO(
        'MultiInputPolicy',
        env,
        verbose=1,  # Set to 1 to print training progress
        tensorboard_log="./ppo_grid_tensorboard/"
    )

    print(f"Starting training for {training_steps} steps...")
    model.learn(total_timesteps=training_steps)

    print("Training finished.")

    # Save the trained model
    try:
        model.save(save_path)
        print(f"Model successfully saved to {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"Error saving model: {e}")

    env.close()

if __name__ == '__main__':

    # Set the total number of training steps.
    TOTAL_TRAINING_STEPS = 1_500_000
    
    # Define the name for the saved model file.
    MODEL_SAVE_PATH = "heuristic_agent.zip"

    train_agent(training_steps=TOTAL_TRAINING_STEPS, save_path=MODEL_SAVE_PATH)

    print("\n--- How to use the trained model ---")
    print("1. Keep the saved file ('heuristic_agent.zip') in your project directory.")
    print("2. In 'rsa_agent.py', you will load this model to get heuristic values.")
    print("   The value function of this trained agent can predict the expected future reward")
    print("   from any given state, making it a powerful heuristic for your RSA agent.")