import itertools
import pandas as pd
import seaborn as sns
import numpy as np
from rsa_agent import run_simulation
import matplotlib.pyplot as plt
import os
from datetime import datetime

KMP_DUPLICATE_LIB_OK= True

def perform_grid_search():
    """
    Performs a hyperparameter grid search for the RSA simulation and visualizes the results.
    """
    # Define the grid of hyperparameters to search.
    param_grid = {
        "rsa_iterations": [50],
        "agent_rationality": [5.0], # How rational the agent is in its decision-making (useless for now)
        "agent_utility_beta": [2.0], #tradeoff between exploration and exploitation
        "sharpening_factor": [5.0],
        "observer_learning_rate": [0.25], #maybe doesnt matter
        "num_samples": [1000],
        "convergence_threshold": [0.01],
        "confidence": [True, False],
        "max_cycle": [0],
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
        ],
        "model_path": ["heuristic_agent.zip"],
        "max_steps": [20],
        "render": [False],  # Disable rendering for speed
        "time_delay": [0.0],
        "num_iterations": [3],
        "randomize_agent_after_goal": [True],
        "randomize_target_after_goal": [True],
        "randomize_initial_placement": [True]
    }

    # Create a list of all parameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results_history = []
    best_mse = float('inf')
    best_params = None

    num_runs = 20

    print(f"Starting hyperparameter grid search with {len(param_combinations)} combinations...")

    for i, params in enumerate(param_combinations):
        print(f"\n--- Running Combination {i+1}/{len(param_combinations)} ---")
        
        try:

            for j in range(num_runs):
                print(f"\n--- Combination {i+1}, Run {j+1}/{num_runs} ---")

                results = run_simulation(params)
                current_results = params.copy()
                current_results.update(results)
                results_history.append(current_results)
            
            
            print(f"  Goals Reached:  {results['goals_reached']} / {params['num_iterations']}")
            print(f"  Final Belief MSE: {results['final_mse']:.4f}")

            # Check for best performing parameters
            if results['final_mse'] < best_mse:
                best_mse = results['final_mse']
                best_params = params

        except Exception as e:
            print(f"\nAn error occurred during simulation with params: {params}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*40)
    print("---  HYPERPARAMETER SEARCH COMPLETE  ---")
    print("="*40)
    print(f"Best Mean Squared Error: {best_mse:.4f}")
    print("Best Parameters:")
    print(best_params)
    print("="*40)


    if results_history:
        df = pd.DataFrame(results_history)
        
        # Create a unique filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"grid_search_results_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(results_filename, index=False)
        print(f"Results saved to '{os.path.abspath(results_filename)}'")

        # Visualization ---
        params_to_plot = [
            'confidence', 'convergence_threshold', 'rsa_iterations',
            'num_samples', 'observer_learning_rate', 'sharpening_factor',
            'agent_utility_beta', 'agent_rationality'
        ]

        num_params = len(params_to_plot)
        cols = 4
        rows = int(np.ceil(num_params / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten()

        for i, param in enumerate(params_to_plot):
            ax = axes[i]
            # Use Seaborn's violinplot
            sns.violinplot(x=param, y='final_mse', data=df, ax=ax, inner='quartile', cut=0)
            ax.set_title(f'MSE Distribution by {param}')
            ax.set_xlabel(param)
            ax.set_ylabel('Final MSE')

        for j in range(num_params, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('MSE Distribution vs. Hyperparameters', fontsize=16)
        plt.show()

if __name__ == '__main__':
    perform_grid_search()