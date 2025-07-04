import time
import torch
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.datasets import get_mnist_loaders
from fully_connected_basics.trainer import train_model
from utils.model_utils import count_parameters

def get_depth_configs():
    return {
        "1_layer": [{"type": "linear", "size": 128}, {"type": "relu"}],
        "2_layers": [{"type": "linear", "size": 256}, {"type": "relu"}, {"type": "linear", "size": 128}, {"type": "relu"}],
        "3_layers": [{"type": "linear", "size": 512}, {"type": "relu"}, {"type": "linear", "size": 256}, {"type": "relu"}, {"type": "linear", "size": 128}, {"type": "relu"}],
        "5_layers": [
            {"type": "linear", "size": 512}, {"type": "relu"},
            {"type": "linear", "size": 256}, {"type": "relu"},
            {"type": "linear", "size": 128}, {"type": "relu"},
            {"type": "linear", "size": 64},  {"type": "relu"}
        ],
        "7_layers": [
            {"type": "linear", "size": 512}, {"type": "batch_norm"}, {"type": "relu"},
            {"type": "linear", "size": 256}, {"type": "layer_norm"}, {"type": "relu"},
            {"type": "linear", "size": 128}, {"type": "relu"},
            {"type": "linear", "size": 64},  {"type": "relu"}
        ]
    }

def run_depth_experiments(output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    configs = get_depth_configs()
    results = {}
    
    for name, layers in configs.items():
        print(f"\nTraining model: {name}")
        model = FullyConnectedModel(input_size=784, num_classes=10, layers=layers).to(device)
        params = count_parameters(model)
        
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))
        duration = time.time() - start_time
        
        results[name] = {
            "history": history,
            "params": params,
            "time": duration
        }

    return results

def get_width_configs():
    return {
        "narrow": [
            {"type": "linear", "size": 64}, {"type": "relu"},
            {"type": "linear", "size": 32}, {"type": "relu"},
            {"type": "linear", "size": 16}, {"type": "relu"},
        ],
        "medium": [
            {"type": "linear", "size": 256}, {"type": "relu"},
            {"type": "linear", "size": 128}, {"type": "relu"},
            {"type": "linear", "size": 64}, {"type": "relu"},
        ],
        "wide": [
            {"type": "linear", "size": 1024}, {"type": "relu"},
            {"type": "linear", "size": 512}, {"type": "relu"},
            {"type": "linear", "size": 256}, {"type": "relu"},
        ],
        "very_wide": [
            {"type": "linear", "size": 2048}, {"type": "relu"},
            {"type": "linear", "size": 1024}, {"type": "relu"},
            {"type": "linear", "size": 512}, {"type": "relu"},
        ],
    }

def run_width_experiments(output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    configs = get_width_configs()
    results = {}

    for name, layers in configs.items():
        print(f"\nTraining width variant: {name}")
        model = FullyConnectedModel(input_size=784, num_classes=10, layers=layers).to(device)
        params = count_parameters(model)

        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))
        duration = time.time() - start

        results[name] = {
            "history": history,
            "params": params,
            "time": duration
        }

    return results

def get_regularization_configs():
    base = [
        {"type": "linear", "size": 256}, {"type": "relu"},
        {"type": "linear", "size": 128}, {"type": "relu"},
        {"type": "linear", "size": 64},  {"type": "relu"},
    ]
    
    return {
        "no_regularization": base,
        "dropout_0.1": insert_dropout(base, 0.1),
        "dropout_0.3": insert_dropout(base, 0.3),
        "dropout_0.5": insert_dropout(base, 0.5),
        "batchnorm": insert_batchnorm(base),
        "dropout+batchnorm": insert_batchnorm(insert_dropout(base, 0.3)),
        "l2_regularization": base  # L2 реализуется через weight_decay
    }

def insert_dropout(layers, rate):
    result = []
    for layer in layers:
        result.append(layer)
        if layer.get("type") == "relu":
            result.append({"type": "dropout", "rate": rate})
    return result

def insert_batchnorm(layers):
    result = []
    prev_size = 784
    for layer in layers:
        result.append(layer)
        if layer.get("type") == "linear":
            result.append({"type": "batch_norm"})
    return result

def run_regularization_experiments(output_dir):
    import time
    from utils.model_utils import count_parameters, extract_weights

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    configs = get_regularization_configs()
    results = {}

    for name, layers in configs.items():
        print(f"\nTraining: {name}")
        model = FullyConnectedModel(input_size=784, num_classes=10, layers=layers).to(device)
        params = count_parameters(model)

        weight_decay = 1e-4 if "l2" in name else 0.0

        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, device=str(device), weight_decay=weight_decay)
        duration = time.time() - start

        weights = extract_weights(model)

        results[name] = {
            "history": history,
            "params": params,
            "time": duration,
            "weights": weights
        }

    return results

def get_adaptive_configs():
    return {
        "adaptive_dropout": [
            {"type": "linear", "size": 512}, {"type": "relu"}, {"type": "dropout", "rate": 0.5},
            {"type": "linear", "size": 256}, {"type": "relu"}, {"type": "dropout", "rate": 0.3},
            {"type": "linear", "size": 128}, {"type": "relu"}, {"type": "dropout", "rate": 0.1},
        ],
        "bn_momentum_low": [
            {"type": "linear", "size": 256}, {"type": "batch_norm"}, {"type": "relu"},
            {"type": "linear", "size": 128}, {"type": "batch_norm"}, {"type": "relu"},
            {"type": "linear", "size": 64},  {"type": "relu"},
        ],
        "combo_all": [
            {"type": "linear", "size": 256}, {"type": "batch_norm"}, {"type": "relu"}, {"type": "dropout", "rate": 0.3},
            {"type": "linear", "size": 128}, {"type": "batch_norm"}, {"type": "relu"}, {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 64},  {"type": "relu"},
        ]
    }