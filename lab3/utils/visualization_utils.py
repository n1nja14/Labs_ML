import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import os

def plot_experiment_results(results, save_dir=None):
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, result in results.items():
        history = result['history']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['test_losses'], label='Test Loss')
        ax1.set_title(f'{model_name} - Loss')
        ax1.legend()

        ax2.plot(history['train_accs'], label='Train Acc')
        ax2.plot(history['test_accs'], label='Test Acc')
        ax2.set_title(f'{model_name} - Accuracy')
        ax2.legend()

        plt.suptitle(model_name)
        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/{model_name}.png")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_heatmap_grid_search(results, save_path=None):
    models = list(results.keys())
    accs = [results[m]['history']['test_accs'][-1] for m in models]

    fig, ax = plt.subplots(figsize=(10, 2))
    data = np.array(accs).reshape(1, -1)

    im = ax.imshow(data, cmap="YlGnBu")

    # Подписи по x
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")

    # Подписи по y
    ax.set_yticks([0])
    ax.set_yticklabels(["Test Accuracy"])

    # Аннотации на клетках
    for i in range(1):
        for j in range(len(models)):
            text = ax.text(j, i, f"{data[i, j]:.4f}",
                           ha="center", va="center", color="black")

    plt.title("Test Accuracy Heatmap (Width Variants)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_weight_histograms(results, save_dir):
    import os
    os.makedirs(save_dir, exist_ok=True)

    for name, result in results.items():
        weights = result.get("weights", [])
        if not weights:
            continue

        for i, w in enumerate(weights):
            plt.figure()
            plt.hist(w.flatten().detach().cpu().numpy(), bins=50)
            plt.title(f"{name} - Layer {i} Weights")
            plt.xlabel("Weight")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{name}_layer{i}.png")
            plt.close()
