from utils.experiment_utils import run_depth_experiments
from utils.visualization_utils import plot_experiment_results
from utils.model_utils import save_experiment_summary

if __name__ == "__main__":
    experiment_name = "depth_experiments"
    results = run_depth_experiments(output_dir=f"results/{experiment_name}/")
    
    plot_experiment_results(results, save_dir="plots/depth_experiments/")
    
    save_experiment_summary(results, f"results/{experiment_name}/summary.json")
