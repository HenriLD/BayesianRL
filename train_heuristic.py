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
# Define a set of fixed maps. The agent will be trained on these environments.
# 'T' is the target. ' ' is empty space. '#' is a wall.
# The agent 'A' will be placed randomly in an empty space at the start of each episode.
PREDEFINED_MAPS = [
    [
        "##############",
        "#      #     #",
        "#   #     # ##",
        "#  ######  # #",
        "#     T#    ##",
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
        "#    T       #",
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
        "#  ##T   #   #",
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
        "#  # T       #",
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
    map from a predefined list and places the agent at a random valid starting position.
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

        # 2. Find all valid starting locations (empty spaces)
        valid_starts = []
        for r, row in enumerate(new_map):
            for c, char in enumerate(row):
                if char == ' ':
                    valid_starts.append((r, c))

        if not valid_starts:
            raise ValueError("A predefined map has no empty spaces for the agent to start.")
        
        # 3. Choose a random start position and place the agent 'A'
        start_pos = random.choice(valid_starts)
        new_map[start_pos[0]][start_pos[1]] = 'A'

        # 4. Set this newly configured map as the one to be used by the environment
        self.grid_map = ["".join(row) for row in new_map]

        # 5. Now, call the parent's logic to build the grid state from self.grid_map.
        # This populates self.grid, self.walls, self.agent_pos, self.target_pos etc.
        self._generate_grid_from_map()
        
        # 6. Finalize the reset process
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
    # Instead of a single random grid, we use the custom TrainingEnv
    # which cycles through a set of predefined maps with random start points.
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
    TOTAL_TRAINING_STEPS = 150_000
    
    # Define the name for the saved model file.
    MODEL_SAVE_PATH = "heuristic_agent.zip"

    train_agent(training_steps=TOTAL_TRAINING_STEPS, save_path=MODEL_SAVE_PATH)

    print("\n--- How to use the trained model ---")
    print("1. Keep the saved file ('heuristic_agent.zip') in your project directory.")
    print("2. In 'rsa_agent.py', you will load this model to get heuristic values.")
    print("   The value function of this trained agent can predict the expected future reward")
    print("   from any given state, making it a powerful heuristic for your RSA agent.")
