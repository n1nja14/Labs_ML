from utils.experiment_utils import run_regularization_experiments
from utils.visualization_utils import plot_experiment_results, plot_weight_histograms
from utils.model_utils import save_experiment_summary

if __name__ == "__main__":
    experiment_name = "regularization_experiments"
    results = run_regularization_experiments(output_dir=f"results/{experiment_name}/")

    plot_experiment_results(results, save_dir=f"plots/{experiment_name}/")
    plot_weight_histograms(results, save_dir=f"plots/{experiment_name}/weights/")
    save_experiment_summary(results, f"results/{experiment_name}/summary.json")
