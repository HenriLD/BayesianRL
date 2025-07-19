import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import os

def visualize_results(csv_filepath):
    """
    Loads grid search results from a CSV file and generates visualizations.

    Args:
        csv_filepath (str): The path to the input CSV file.
    """
    # --- 1. Load the data from the CSV file ---
    try:
        df = pd.read_csv(csv_filepath)
        print(f"✅ Successfully loaded '{os.path.basename(csv_filepath)}'")
    except FileNotFoundError:
        print(f"❌ Error: The file '{csv_filepath}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        sys.exit(1)

    # --- 2. Identify varying parameters and metrics to plot ---

    # Define the potential hyperparameters that might have been varied in the search.
    # This helps distinguish parameters from result metrics.
    potential_params = [
        "agent_type", "observer_type", "rationality", "agent_utility_beta",
        "sharpening_factor", "observer_learning_rate", "num_samples",
        "convergence_threshold", "rsa_iterations", "use_confidence", "max_cycle",
        "model_path", "max_steps", "num_iterations", "agent_sampling_mode",
        "observer_sampling_mode"
    ]
    
    # Find which of these parameters actually have more than one unique value in the data.
    varying_params = [
        p for p in potential_params
        if p in df.columns and df[p].nunique() > 1
    ]

    # The metrics we want to see on the y-axis.
    metrics_to_plot = ['final_mse', 'final_js_divergence', 'goals_reached']

    # --- 3. Validate that necessary columns exist ---
    if not varying_params:
        print("\n⚠️ No varying hyperparameters found in the data. Cannot generate comparison plots.")
        return

    for metric in metrics_to_plot:
        if metric not in df.columns:
            print(f"❌ Error: Metric column '{metric}' not found in the CSV file. Cannot generate plots.")
            return

    print(f"📊 Found varying parameter(s): {varying_params}")
    print(f"📈 Plotting metrics: {metrics_to_plot}")

    # --- 4. Generate and display plots ---
    # This loop replicates the plotting logic from your original script.
    for x_param in varying_params:
        for y_metric in metrics_to_plot:
            
            # Determine faceting parameters (for creating subplots).
            # This logic is kept to match the original script's capability.
            facet_params = ["agent_type", "observer_type"]
            col_param = None
            row_param = None

            available_facets = [
                p for p in facet_params 
                if p != x_param and p in df.columns and df[p].nunique() > 1
            ]
            if available_facets:
                col_param = available_facets[0]
            if len(available_facets) > 1:
                row_param = available_facets[1]
            
            print(f"\nGenerating plot for: {y_metric} vs. {x_param}")

            # Create a violin plot using seaborn's catplot.
            g = sns.catplot(
                data=df,
                x=x_param,
                y=y_metric,
                col=col_param,
                row=row_param,
                kind="violin",
                inner="quartile", # Shows the quartiles inside the violin.
                height=5,
                aspect=1.2
            )

            # Set plot titles and axis labels for clarity.
            title = f'{y_metric.replace("_", " ").title()} vs. {x_param.replace("_", " ").title()}'
            g.fig.suptitle(title, y=1.03) # y=1.03 raises title to prevent overlap.
            g.set_axis_labels(
                x_var=x_param.replace("_", " ").title(), 
                y_var=y_metric.replace("_", " ").title()
            )
            g.set_xticklabels(rotation=10, ha='right') # Rotate long labels.

    # After creating all plot figures, display them on the screen.
    print("\n🚀 Displaying plots. Close the plot windows to exit the script.")
    plt.show()

if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Load results from a grid search CSV file and display visualizations.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file containing the simulation results.\n"
             "Example: python visualize_results.py results/grid_search_results_20240101_120000.csv"
    )

    args = parser.parse_args()
    visualize_results(args.csv_file)