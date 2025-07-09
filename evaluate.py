import os
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now we can import the main simulation runner
from main import run_simulation

def run_simulation_wrapper(params):
    """
    A simple wrapper to call the main run_simulation function.
    This is used by the multiprocessing Pool to run simulations in parallel.
    It catches exceptions within the process to prevent the entire grid search from crashing.
    """
    try:
        return run_simulation(params)
    except Exception as e:
        print(f"\nAn error occurred in a simulation process with params: {params}")
        import traceback
        traceback.print_exc()
        # Return None so we can filter out failed runs
        return None

def perform_grid_search():
    """
    Performs a hyperparameter grid search for the simulation and visualizes the results.
    This function is highly customizable for experimentation.
    """
    # Grid Search Configuration
    param_grid = {
        "agent_type": ["RSA", "Base"],
        "observer_type": ["RSA", "Base"],
        "num_trials": [1], # Set to 1 for grid search, as we aggregate across param combinations
        "render": [False], # Must be False for multiprocessing
        
        # Key RSA parameter to vary
        "rationality": [1.0],
        
        # Constant parameters for this experiment
        "agent_utility_beta": [1.0],
        "sharpening_factor": [3.0],
        "observer_learning_rate": [0.5],
        "num_samples": [3000],
        "convergence_threshold": [0.01],
        "use_confidence": [True],
        "max_cycle": [4],
        "model_path": ["heuristic_agent.zip"],
        "max_steps": [25],
        "num_iterations": [5],
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

    # --- Execution ---
    # Create a list of all parameter combinations from the grid
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Define how many times to run each unique parameter combination
    runs_per_combination = 32 
    all_runs_params = [params for params in param_combinations for _ in range(runs_per_combination)]
    total_simulations = len(all_runs_params)

    print(f"Starting hyperparameter grid search...")
    print(f"Total unique parameter combinations: {len(param_combinations)}")
    print(f"Runs per combination: {runs_per_combination}")
    print(f"Total simulations to run: {total_simulations}")

    results_history = []
    # Use a multiprocessing Pool to run simulations in parallel
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=total_simulations, desc="Grid Search Progress") as pbar:
            for result in pool.imap_unordered(run_simulation_wrapper, all_runs_params):
                if result:
                    results_history.append(result)
                pbar.update()

    if not results_history:
        print("No simulation results were generated. Exiting.")
        return

    # --- Data Processing and Saving ---
    df = pd.DataFrame(results_history)
    
    # Create a unique filename with a timestamp for the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, f"grid_search_results_{timestamp}.csv")
    
    df.to_csv(results_filename, index=False)
    print(f"\nResults saved to '{os.path.abspath(results_filename)}'")

    # --- Visualization ---
    # Identify which parameters were actually varied in the grid search
    varying_params = [key for key, value in param_grid.items() if len(value) > 1]
    
    if not varying_params:
        print("No hyperparameters were varied. Skipping plot generation.")
        return

    metrics_to_plot = ['final_mse', 'final_js_divergence', 'goals_reached']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(8 * len(varying_params), 6))
        g = sns.catplot(
            data=df, 
            x="rationality", 
            y=metric, 
            col="observer_type", 
            row="agent_type",
            kind="violin", 
            inner="quartile",
            height=5, 
            aspect=1.2
        )
        g.fig.suptitle(f'{metric.replace("_", " ").title()} Analysis', y=1.03)
        g.set_axis_labels(x_var="Agent Rationality (alpha)", y_var=metric.replace("_", " ").title())
        
        # Save the plot
        plot_filename = os.path.join(results_dir, f"plot_{metric}_{timestamp}.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Plot saved to '{os.path.abspath(plot_filename)}'")
        plt.close() # Close the figure to avoid displaying it if not needed

    print("\nGrid search and visualization complete.")
    # To display plots, uncomment the line below
    # plt.show()

if __name__ == '__main__':
    # This check is important for multiprocessing on Windows
    multiprocessing.freeze_support()
    perform_grid_search()
