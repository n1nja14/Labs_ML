import json
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_experiment_summary(results, path):
    summary = {}
    for name, result in results.items():
        summary[name] = {
            "final_train_acc": result["history"]["train_accs"][-1],
            "final_test_acc": result["history"]["test_accs"][-1],
            "params": result["params"],
            "training_time_sec": result["time"]
        }
    with open(path, "w") as f:
        json.dump(summary, f, indent=4)

def extract_weights(model):
    """Извлекает веса линейных слоев для анализа"""
    weights = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            weights.append(m.weight)
    return weights