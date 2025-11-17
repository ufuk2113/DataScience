# main.py
#import os
import json
import csv
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from PyTorchNeuralNetwork import PyTorchNeuralNetwork

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # macOS MPS support
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def prepare_data(batch_size):
    # standard MNIST transforms: ToTensor and normalize (mean/std for MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def run_experiment(activation, activation_param, device, epochs, batch_size, lr, optimizer_name):
    print(f"\n--- Experiment: {activation} (param={activation_param}) | optimizer={optimizer_name} | lr={lr} | epochs={epochs} ---")
    train_loader, test_loader = prepare_data(batch_size)

    model = PyTorchNeuralNetwork(
        activation=activation,
        activation_param=activation_param,
        learning_rate=lr,
        optimizer_name=optimizer_name
    )

    model.to_device(device)

    model.train_model(train_loader, epochs=epochs, log_every=200)
    acc = model.evaluate(test_loader)
    print(f"Result: {activation} param={activation_param}  -> Test accuracy: {acc:.2f}%")
    return acc

def save_results_csv(results, out_path="results.csv"):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["activation", "param", "accuracy"])
        for r in results:
            writer.writerow([r[0], r[1], r[2]])
    print(f"Saved CSV to {out_path}")

def save_results_json(results, out_path="results.json"):
    out = [{"activation": r[0], "param": r[1], "accuracy": r[2]} for r in results]
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved JSON to {out_path}")

def plot_results(results, out_path="results.png"):
    # simple bar plot of accuracies
    labels = [f"{r[0]}({r[1]})" for r in results]
    accs = [r[2] for r in results]

    plt.figure(figsize=(10,5))
    plt.bar(range(len(accs)), accs)
    plt.xticks(range(len(accs)), labels, rotation=45, ha='right')
    plt.ylabel("Test Accuracy (%)")
    plt.title("Activation Function Experiment Results")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()

def main_cli():
    parser = argparse.ArgumentParser(description="Run activation experiments (MNIST).")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam", help="Optimizer")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if GPU available")
    parser.add_argument("--out_prefix", type=str, default="", help="Optional output file prefix")
    args = parser.parse_args()

    device = get_device(force_cpu=args.force_cpu)
    print("Using device:", device)

    # experiment list
    experiments = [
        ("sigmoid", None),
        ("leaky_relu", 0.01),
        ("leaky_relu", 0.05),
        ("leaky_relu", 0.1),
        ("leaky_relu", 0.5),
        ("prelu", None),
        ("elu", 0.1),
        ("elu", 0.2),
        ("elu", 0.3),
    ]

    results = []
    for activation, param in experiments:
        acc = run_experiment(
            activation=activation,
            activation_param=param,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer
        )
        results.append((activation, param if param is not None else "-", acc))

    # save files with optional prefix
    prefix = args.out_prefix
    csv_path = f"{prefix}results.csv" if prefix else "results.csv"
    json_path = f"{prefix}results.json" if prefix else "results.json"
    plot_path = f"{prefix}results.png" if prefix else "results.png"

    save_results_csv(results, csv_path)
    save_results_json(results, json_path)
    plot_results(results, plot_path)

    print("\nFinal results:")
    print(f"{'Activation':<15} {'Param':<8} {'Accuracy (%)':<12}")
    print("-" * 40)
    for r in results:
        print(f"{r[0]:<15} {str(r[1]):<8} {r[2]:<12.2f}")

if __name__ == "__main__":
    main_cli()
