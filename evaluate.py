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
        # It's better to create a new dictionary to avoid modifying the original params
        run_results = params.copy()
        metrics = run_simulation(params)
        run_results.update(metrics)
        return run_results
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
        "agent_type": ["RSA"],
        "observer_type": ["RSA"],
        "num_trials": [1], # Set to 1 for grid search, as we aggregate across param combinations
        "render": [False], # Must be False for multiprocessing
        "rationality": [1.0],
        "agent_utility_beta": [1.0],
        "sharpening_factor": [0.5],
        "observer_learning_rate": [0.5],
        "num_samples": [100],
        "convergence_threshold": [0.00001],
        "rsa_iterations": [10000],
        "use_confidence": [True],
        "max_cycle": [4],
        "model_path": ["heuristic_agent.zip"],
        "max_steps": [20],
        "num_iterations": [3],
        "agent_sampling_mode": ['belief_based'],
        "observer_sampling_mode": ['belief_based'],
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
    runs_per_combination = 5
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
    
    # Generate a plot for each combination of a varied parameter and a metric
    for x_param in varying_params:
        for y_metric in metrics_to_plot:
            # Determine which parameters to use for faceting, avoiding the x_param
            facet_params = ["agent_type", "observer_type"]
            col_param = None
            row_param = None

            available_facets = [p for p in facet_params if p != x_param]
            if available_facets:
                col_param = available_facets[0]
            if len(available_facets) > 1:
                row_param = available_facets[1]


            plt.figure(figsize=(12, 6)) # Adjusted figure size for better layout
            g = sns.catplot(
                data=df, 
                x=x_param, 
                y=y_metric, 
                col=col_param, 
                row=row_param,
                kind="violin", 
                inner="quartile",
                height=5, 
                aspect=1.2
            )
            
            title = f'{y_metric.replace("_", " ").title()} vs. {x_param.replace("_", " ").title()}'
            g.fig.suptitle(title, y=1.03)
            g.set_axis_labels(x_var=x_param.replace("_", " ").title(), y_var=y_metric.replace("_", " ").title())
            
            # Save the plot with a descriptive name
            plot_filename = os.path.join(results_dir, f"plot_{y_metric}_vs_{x_param}_{timestamp}.png")
            plt.savefig(plot_filename, bbox_inches='tight')
            print(f"Plot saved to '{os.path.abspath(plot_filename)}'")
            plt.close('all') # Close all figures to free up memory

    print("\nGrid search and visualization complete.")
    # To display plots, uncomment the line below
    # plt.show()

if __name__ == '__main__':
    # This check is important for multiprocessing on Windows
    multiprocessing.freeze_support()
    perform_grid_search()
