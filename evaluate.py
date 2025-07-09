import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools
import pandas as pd
import seaborn as sns
import numpy as np
from rsa_agent import run_simulation as run_rsa_simulation
from base_agent import run_simulation as run_base_simulation
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
import tqdm

# This function takes a single 'params' dict and is used by the multiprocessing Pool.
def run_simulation_wrapper(params):
    """
    Wrapper to call run_simulation with a single argument for use with multiprocessing.
    """
    try:
        agent_type = params.get("agent_type")
        
        # Select the correct simulation function
        if agent_type == "RSA":
            results = run_rsa_simulation(params)
        elif agent_type == "Base":
            results = run_base_simulation(params)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        # Combine the initial parameters with the simulation results
        current_results = params.copy()
        current_results.update(results)
        return current_results
    except Exception as e:
        print(f"\nAn error occurred in a simulation process with params: {params}")
        import traceback
        traceback.print_exc()
        return None

def perform_grid_search():
    """
    Performs a hyperparameter grid search for the RSA simulation and visualizes the results.
    """
    # Define the grid of hyperparameters to search.
    param_grid = {
        "agent_type": ["Base"],  # Pick Agent type
        "rsa_iterations": [10],
        "agent_rationality": [0, 0.5, 1.0, 3.0], # How rational the agent is in its decision-making
        "agent_utility_beta": [1], # tradeoff between exploration and exploitation, 0 is bad
        "sharpening_factor": [3.0], # How much the agent sharpens its beliefs, 3 is a good default value
        "observer_learning_rate": [0.5], # 0.5 is a good default value for the observer's learning rate
        "num_samples": [10000],
        "convergence_threshold": [0.01],
        "confidence": [True], # Improves performance a lot
        "max_cycle": [0],
        "model_path": ["heuristic_agent.zip"],
        "max_steps": [20],
        "render": [False],  # Disable rendering for speed
        "time_delay": [0.0],
        "num_iterations": [3],
        "randomize_agent_after_goal": [True],
        "randomize_target_after_goal": [True],
        "randomize_initial_placement": [True],
        "custom_map": [
            [
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
            ]
        ]
    }

    # Create a list of all parameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results_history = []

    num_runs = 128
    # Create a list of all runs to be executed.
    all_runs_params = [params for params in param_combinations for _ in range(num_runs)]
    total_simulations = len(all_runs_params)

    print(f"Starting hyperparameter grid search with {len(param_combinations)} combinations...")

    # Use a multiprocessing Pool to run simulations in parallel
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        # 'map' will distribute the 'all_runs_params' list to the 'run_simulation_wrapper' function
        # across the available CPU cores. It blocks until all results are ready.
        pbar = tqdm.tqdm(pool.imap_unordered(run_simulation_wrapper, all_runs_params), total=total_simulations)
        for result in pbar:
            if result:
                results_history.append(result)

    results_history = [r for r in results_history if r is not None] # Filter out any None results due to exceptions

    if results_history:
        df = pd.DataFrame(results_history)
        
        # Create a unique filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"results\grid_search_results_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(results_filename, index=False)
        print(f"Results saved to '{os.path.abspath(results_filename)}'")

        # --- Visualization ---
        
        # Identify which parameters were actually varied
        varying_params = [key for key, value in param_grid.items() if len(value) > 1]
        
        if not varying_params:
            print("No hyperparameters were varied. Skipping plot generation.")
            return

        metrics_to_plot = ['final_mse', 'final_js_divergence', 'final_log_loss', 'goals_reached']
        
        num_metrics = len(metrics_to_plot)
        num_params = len(varying_params)
        
        # Create a figure with a row for each metric
        fig, axes = plt.subplots(num_params, num_metrics, figsize=(5 * num_metrics, 4 * num_params), squeeze=False)
        
        fig.suptitle('Hyperparameter Grid Search Results', fontsize=20, y=1.03)

        for i, param in enumerate(varying_params):
            for j, metric in enumerate(metrics_to_plot):
                ax = axes[i, j]
                sns.violinplot(x=param, y=metric, data=df, ax=ax, inner='quartile', cut=0)
                ax.set_title(f'{metric.replace("_", " ").title()} by {param.replace("_", " ").title()}')
                ax.set_xlabel(param.replace("_", " ").title())
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save the plot
        plot_filename = f"results\grid_search_plots_{timestamp}.png"
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Plots saved to '{os.path.abspath(plot_filename)}'")
        
        plt.show()

if __name__ == '__main__':
    perform_grid_search()