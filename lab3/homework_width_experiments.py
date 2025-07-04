from utils.experiment_utils import run_width_experiments
from utils.visualization_utils import plot_experiment_results, plot_heatmap_grid_search
from utils.model_utils import save_experiment_summary

if __name__ == "__main__":
    experiment_name = "width_experiments"
    results = run_width_experiments(output_dir=f"results/{experiment_name}/")

    plot_experiment_results(results, save_dir=f"plots/{experiment_name}/")
    plot_heatmap_grid_search(results, save_path=f"plots/{experiment_name}/heatmap.png")
    save_experiment_summary(results, f"results/{experiment_name}/summary.json")
