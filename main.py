import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

# Import environment, agents, observers, and utilities
# This assumes a directory structure where main.py is in the root,
# and the other modules are in their respective subdirectories.
from env import GridEnvironment
from base_agent import BaseAgent
from rsa_agent import RSAAgent
from base_observer import BaseObserver
from rsa_observer import RSAObserver
from utils import render_side_by_side_views, _render_belief_map_with_chars

def setup_environment(params):
    """Creates and configures the grid environment."""
    map_template = [list(row) for row in params["custom_map"]]
    if params.get("randomize_initial_placement", False):
        valid_spawn_points = []
        for r, row in enumerate(map_template):
            for c, char in enumerate(row):
                if char in [' ', 'A', 'T']:
                    valid_spawn_points.append((r, c))
                    if char in ['A', 'T']:
                        map_template[r][c] = ' '
        
        if len(valid_spawn_points) < 2:
            raise ValueError("Map must have at least two empty spaces for agent and target.")
        
        agent_start_pos, target_start_pos = random.sample(valid_spawn_points, 2)
        map_template[agent_start_pos[0]][agent_start_pos[1]] = 'A'
        map_template[target_start_pos[0]][target_start_pos[1]] = 'T'
        
    final_map = ["".join(row) for row in map_template]
    # Disable rendering during multi-trial runs unless explicitly requested
    render_mode = 'human' if params.get("render", True) else None
    
    return GridEnvironment(grid_map=final_map, render_mode=render_mode, max_steps=params.get("max_steps", 100))

def run_simulation(params: dict):
    """
    Runs a single simulation episode with a given set of parameters.
    This function is now generalized to handle different agent and observer types.
    """
    env = setup_environment(params)
    obs, _ = env.reset()

    num_walls = sum(row.count('#') for row in env.grid_map)
    total_cells = env.grid_size * env.grid_size
    true_wall_prob = num_walls / total_cells

    # --- Agent and Observer Initialization ---
    agent_type = params.get("agent_type", "RSA")
    observer_type = params.get("observer_type", "RSA")
    
    agent_kwargs = {
        'initial_prob': true_wall_prob,
        'max_cycle': params.get("max_cycle", 0),
        'sampling_mode': params.get("agent_sampling_mode", 'uniform'),
    }

    if agent_type == "RSA":
        agent = RSAAgent(env, params["model_path"], rsa_params=params, **agent_kwargs)
    else: # "Base"
        agent = BaseAgent(env, params["model_path"], **agent_kwargs)

    observer_kwargs = {
        'initial_prob': true_wall_prob,
        'learning_rate': params.get("observer_learning_rate", 0.5),
        'sharpening_factor': params.get("sharpening_factor", 5.0),
        'num_samples': params.get("num_samples", 5000),
        'use_confidence': params.get("use_confidence", False),
        'sampling_mode': params.get("observer_sampling_mode", 'uniform')
    }
    
    if observer_type == "RSA":
        observer = RSAObserver(env, params["model_path"], agent_params=params, **observer_kwargs)
    else: # "Base"
        observer = BaseObserver(env, params["model_path"], policy_beta=params.get("agent_utility_beta", 1.0), **observer_kwargs)

    # --- Simulation Loop ---
    total_steps_taken = 0
    goals_reached = 0
    render = params.get("render", True)
    time_delay = params.get("time_delay", 0.1)

    for iteration in range(params.get("num_iterations", 1)):
        done = False
        if render:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Iteration {iteration + 1} / {params.get('num_iterations', 1)} ---")
            time.sleep(1.0)

        for step in range(params.get("max_steps", 100)):
            agent.update_internal_belief(obs)
            agent_pos, target_pos = tuple(obs['agent_pos']), tuple(obs['target_pos'])

            if render:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Iteration {iteration + 1} | Step {step + 1} ---")

            action, action_probs = agent.choose_action(obs)
            local_probs = observer.update_belief(agent_pos, target_pos, action)
            
            if render:
                render_side_by_side_views(local_probs, obs['local_view'], env)
                print("Observer's Inferred Belief Map:")
                observer.render_belief(agent_pos, target_pos)
                action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Up-L", 5: "Up-R", 6: "Down-L", 7: "Down-R"}
                print(f"\nAction Probs: {[f'{p:.2f}' for p in action_probs]}")
                print(f"Chosen Action: {action_map.get(action, 'Unknown')}")

            new_obs, _, terminated, truncated, _ = env.step(action)
            total_steps_taken += 1

            agent.update_beliefs_after_action(new_obs, action)
            
            if render:
                print("\nActual Environment State:")
                env.render()
            
            obs = new_obs
            if render:
                time.sleep(time_delay)

            if terminated or truncated:
                if terminated:
                    goals_reached += 1
                
                if iteration < params.get("num_iterations", 1) - 1:
                    if params.get("randomize_target_after_goal", False):
                        obs = env.respawn_target()
                    if params.get("randomize_agent_after_goal", False):
                        new_obs, done = env.respawn_agent()
                        if not done:
                            obs = new_obs
                break
        if done:
            break

    # --- Metrics Calculation ---
    observed_mask = agent.internal_belief_map != true_wall_prob
    agent_map_flat = agent.internal_belief_map[observed_mask]
    observer_map_flat = observer.observer_belief_map[observed_mask]

    final_mse = np.mean((agent_map_flat - observer_map_flat)**2) if agent_map_flat.size > 0 else 0
    final_js_divergence = jensenshannon(agent_map_flat + 1e-9, observer_map_flat + 1e-9, base=2) if agent_map_flat.size > 0 else 0
    
    metrics = {
        "total_steps_taken": total_steps_taken,
        "goals_reached": goals_reached,
        "final_mse": final_mse,
        "final_js_divergence": final_js_divergence,
    }
    return metrics

if __name__ == '__main__':
    # Centralized parameter configuration
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
        "agent_type": "RSA",  # Can be "RSA" or "Base"
        "observer_type": "RSA", # Can be "RSA" or "Base"
        "agent_sampling_mode": 'belief_based',  # Sampling mode for agent's belief updates
        
        # RSA-specific params (used by RSAAgent and RSAObserver)
        "rsa_iterations": 10,
        "rationality": 1.0,         # alpha
        "agent_utility_beta": 1.0,  # beta
        "sharpening_factor": 5.0,
        "convergence_threshold": 0.01,
        
        # Observer-specific params
        "observer_learning_rate": 0.5,
        "use_confidence": True,
        "observer_sampling_mode": 'belief_based',
        
        # Simulation control
        "max_steps": 25,
        "num_iterations": 3, # Iterations within a single trial (e.g., agent respawns)
        "num_trials": 32,    # Number of times to run the full simulation
        "render": False,     # Disable rendering for multi-trial runs to speed it up
        "time_delay": 0.0,
        "num_samples": 4000,
        "max_cycle": 0,      # Anti-cycle memory
        "randomize_agent_after_goal": True,
        "randomize_target_after_goal": True,
        "randomize_initial_placement": True,
    }
    
    try:
        num_trials = default_params.get("num_trials", 1)
        
        # If only one trial, render by default. Otherwise, don't.
        if num_trials == 1:
            default_params["render"] = True

        all_results = []
        print(f"Running {num_trials} trials with the same parameters...")
        
        for _ in tqdm(range(num_trials), desc="Running Trials"):
            # Each trial is a full run of the simulation
            results = run_simulation(default_params)
            all_results.append(results)

        # Aggregate and display results
        results_df = pd.DataFrame(all_results)
        
        print("\n" + "="*50)
        print("---   AGGREGATED SIMULATION RESULTS  ---")
        print("="*50)
        print(f"  Agent Type:         {default_params['agent_type']}")
        print(f"  Observer Type:      {default_params['observer_type']}")
        print(f"  Number of Trials:   {num_trials}")
        print("-" * 50)
        
        # Print mean and std for each collected metric
        for metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric.replace('_', ' ').title():<22}: {mean_val:.4f} ± {std_val:.4f}")

        print("="*50)
        
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()