import argparse
import csv
from pathlib import Path

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Write MECG-E loss CSV and plots from a checkpoint directory.")
    parser.add_argument("--checkpoint-dir", required=True)
    return parser.parse_args()


def load_histories(checkpoint_dir):
    train_loss_history = []
    val_metric_history = []

    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        train_loss_history = list(state.get("train_loss_history", []))
        val_metric_history = list(state.get("val_metric_history", []))

    metrics_path = checkpoint_dir / "validation_metrics.yaml"
    if not val_metric_history and metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as handle:
            val_metric_history = yaml.safe_load(handle) or []

    return train_loss_history, val_metric_history


def write_csv(path, train_loss_history, val_metric_history):
    rows_by_epoch = {}
    for item in train_loss_history:
        epoch = int(item["epoch"])
        rows_by_epoch.setdefault(epoch, {"epoch": epoch, "step": item.get("step", ""), "train_loss": "", "val_loss": ""})
        rows_by_epoch[epoch]["step"] = item.get("step", rows_by_epoch[epoch]["step"])
        rows_by_epoch[epoch]["train_loss"] = item.get("train_loss", "")
    for item in val_metric_history:
        epoch = int(item["epoch"])
        rows_by_epoch.setdefault(epoch, {"epoch": epoch, "step": item.get("step", ""), "train_loss": "", "val_loss": ""})
        rows_by_epoch[epoch]["step"] = item.get("step", rows_by_epoch[epoch]["step"])
        rows_by_epoch[epoch]["val_loss"] = item.get("val_loss", "")

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch))


def plot_curves(checkpoint_dir, train_loss_history, val_metric_history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping loss plots because matplotlib is unavailable: {exc}")
        return

    def points(items, key):
        return [(int(item["epoch"]), float(item[key])) for item in items if item.get(key) not in {"", None}]

    def save_single(curve, title, filename):
        if not curve:
            return
        epochs, values = zip(*curve)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, values, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(checkpoint_dir / filename, dpi=200)
        plt.close(fig)

    train_points = points(train_loss_history, "train_loss")
    val_points = points(val_metric_history, "val_loss")
    save_single(train_points, "MECG-E Train Loss", "train_loss.png")
    save_single(val_points, "MECG-E Validation Loss", "val_loss.png")

    if train_points and val_points:
        fig, ax = plt.subplots(figsize=(10, 6))
        train_epochs, train_values = zip(*train_points)
        val_epochs, val_values = zip(*val_points)
        ax.plot(train_epochs, train_values, label="Train Loss", linewidth=2)
        ax.plot(val_epochs, val_values, label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("MECG-E Train / Validation Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(checkpoint_dir / "train_val_loss.png", dpi=200)
        plt.close(fig)


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    train_loss_history, val_metric_history = load_histories(checkpoint_dir)
    write_csv(checkpoint_dir / "loss_history.csv", train_loss_history, val_metric_history)
    plot_curves(checkpoint_dir, train_loss_history, val_metric_history)
    print(f"saved {checkpoint_dir / 'loss_history.csv'}")


if __name__ == "__main__":
    main()
