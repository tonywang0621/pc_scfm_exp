import argparse
import csv
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import get_model


def atomic_pickle_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(obj, handle)
    os.replace(tmp_path, path)


def atomic_torch_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def write_loss_history(checkpoint_dir, train_loss_history, val_loss_history):
    rows = {}
    for item in train_loss_history:
        rows.setdefault(item["epoch"], {"epoch": item["epoch"], "train_loss": "", "val_loss": ""})
        rows[item["epoch"]]["train_loss"] = item["train_loss"]
    for item in val_loss_history:
        rows.setdefault(item["epoch"], {"epoch": item["epoch"], "train_loss": "", "val_loss": ""})
        rows[item["epoch"]]["val_loss"] = item["val_loss"]

    path = checkpoint_dir / "loss_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a local ECG denoiser with the MECG-E Table 1 official training/evaluation flow."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-pkl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def as_n1t(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, None, :]
    elif array.ndim == 3 and array.shape[1] != 1 and array.shape[2] == 1:
        array = np.transpose(array, (0, 2, 1))
    if array.ndim != 3 or array.shape[1] != 1:
        raise ValueError(f"Expected [N, T], [N, 1, T], or [N, T, 1], got {array.shape}.")
    return array


def load_mecge_dataset(path):
    with open(path, "rb") as handle:
        dataset = pickle.load(handle)
    if isinstance(dataset, dict):
        return (
            dataset["X_train"],
            dataset["y_train"],
            dataset["X_test"],
            dataset["y_test"],
        )
    if len(dataset) < 4:
        raise ValueError(f"{path} must contain [X_train, y_train, X_test, y_test].")
    return dataset[0], dataset[1], dataset[2], dataset[3]


def save_training_state(
    path,
    model,
    optimizer,
    scheduler,
    epoch_no,
    best_valid_loss,
    patience_counter,
    train_loss_history,
    val_loss_history,
):
    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch_no,
            "best_valid_loss": best_valid_loss,
            "patience_counter": patience_counter,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
        },
        path,
    )


def build_scheduler(optimizer, args):
    scheduler_name = str(args.scheduler).lower()
    if scheduler_name in {"none", "null", "constantlr"}:
        return None
    if scheduler_name == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1)
    if scheduler_name in {"reducelronplateau", "plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.factor,
            patience=args.lr_scheduler_patience_epochs,
            threshold=args.lr_scheduler_min_delta,
            min_lr=args.min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def build_optimizer(model, args):
    optimizer_name = str(args.optimizer).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def step_scheduler(scheduler, val_loss):
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def build_loaders(dataset_pkl, batch_size):
    x_train, y_train, x_test, y_test = load_mecge_dataset(dataset_pkl)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.3,
        shuffle=True,
        random_state=1,
    )

    train_noisy = torch.from_numpy(as_n1t(x_train))
    train_clean = torch.from_numpy(as_n1t(y_train))
    val_noisy = torch.from_numpy(as_n1t(x_val))
    val_clean = torch.from_numpy(as_n1t(y_val))
    test_noisy = torch.from_numpy(as_n1t(x_test))
    test_clean = torch.from_numpy(as_n1t(y_test))

    train_loader = DataLoader(
        TensorDataset(train_noisy, train_clean),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_noisy, val_clean),
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_noisy, test_clean),
        batch_size=50,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader, np.asarray(x_test, dtype=np.float32), np.asarray(y_test, dtype=np.float32)


def train_model(model, train_loader, val_loader, args, checkpoint_dir, writer):
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    model_filepath = checkpoint_dir / "best_model.pt"
    model_last_filepath = checkpoint_dir / "model_last.pt"
    training_state_path = checkpoint_dir / "training_state.pt"

    best_valid_loss = 1e10
    train_loss_history = []
    val_loss_history = []
    start_epoch = 0
    patience_counter = 0

    if args.resume:
        resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else training_state_path
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"--resume was set but resume checkpoint was not found: {resume_checkpoint}")
        state = torch.load(resume_checkpoint, map_location=args.device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        best_valid_loss = float(state.get("best_valid_loss", best_valid_loss))
        train_loss_history = list(state.get("train_loss_history", []))
        val_loss_history = list(state.get("val_loss_history", []))
        patience_counter = int(state.get("patience_counter", 0))
        start_epoch = int(state["epoch"]) + 1
        print(f"Resumed local official-flow training from {resume_checkpoint} at epoch {start_epoch}.")
        if patience_counter >= args.patience:
            print(
                "Resume checkpoint already reached early stopping patience "
            f"({patience_counter}/{args.patience}); skipping training and running test."
            )
            write_loss_history(checkpoint_dir, train_loss_history, val_loss_history)
            return

    for epoch_no in range(start_epoch, args.epochs):
        avg_loss = 0.0
        model.train()
        with tqdm(train_loader) as it:
            for batch_no, batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model.compute_loss(batch, args.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch_no + 1}: {loss.item()}.")
                loss.backward()
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()
                avg_loss += loss.item()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )

        current_train_loss = avg_loss / batch_no
        train_loss_history.append({"epoch": int(epoch_no + 1), "train_loss": float(current_train_loss)})

        model.eval()
        avg_loss_valid = 0.0
        with torch.no_grad():
            with tqdm(val_loader) as it:
                for batch_no, batch in enumerate(it, start=1):
                    loss = model.compute_loss(batch, args.device)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite validation loss at epoch {epoch_no + 1}: {loss.item()}.")
                    avg_loss_valid += loss.item()
                    it.set_postfix(
                        ordered_dict={
                            "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                            "epoch": epoch_no,
                        },
                        refresh=True,
                    )

        if writer is not None:
            writer.add_scalar("val_loss", avg_loss_valid / batch_no, epoch_no)
        current_valid_loss = avg_loss_valid / batch_no
        val_loss_history.append({"epoch": int(epoch_no + 1), "val_loss": float(current_valid_loss)})
        step_scheduler(scheduler, current_valid_loss)

        if best_valid_loss > current_valid_loss + args.early_stopping_min_delta:
            best_valid_loss = current_valid_loss
            print("\n best loss is updated to ", current_valid_loss, "at", epoch_no)
            patience_counter = 0
            atomic_torch_save(model.state_dict(), model_filepath)
        else:
            patience_counter += 1
            print(f"No validation loss improvement. Patience: {patience_counter}/{args.patience}")

        atomic_torch_save(model.state_dict(), model_last_filepath)
        save_training_state(
            training_state_path,
            model,
            optimizer,
            scheduler,
            epoch_no,
            best_valid_loss,
            patience_counter,
            train_loss_history,
            val_loss_history,
        )
        write_loss_history(checkpoint_dir, train_loss_history, val_loss_history)
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch_no + 1}.")
            break


def test_model(model, test_loader, checkpoint_dir, x_test_original, y_test_original, output_pkl, device):
    model_filepath = checkpoint_dir / "best_model.pt"
    model.load_state_dict(torch.load(model_filepath, map_location="cpu"))
    model.to(device)
    model.eval()

    restored_sig = []
    with torch.no_grad():
        with tqdm(test_loader) as it:
            for noisy_batch, _clean_batch in it:
                noisy_batch = noisy_batch.to(device)
                if hasattr(model, "denoising"):
                    output = model.denoising(noisy_batch)
                else:
                    output = model(noisy_batch)
                restored_sig.append(output.permute(0, 2, 1).cpu().numpy())

    y_pred = np.concatenate(restored_sig)
    atomic_pickle_dump([x_test_original, y_test_original, y_pred.astype(np.float32)], output_pkl)
    print(f"saved {output_pkl}")


def main():
    args = parse_args()
    args.device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    output_pkl = Path(args.output_pkl).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    training_cfg = cfg.get("training", {})
    args.epochs = int(args.max_epochs if args.max_epochs is not None else training_cfg.get("train_epochs", 50000))
    args.patience = int(
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else training_cfg.get("early_stopping_patience_epochs", 30)
    )
    args.batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 96))
    args.lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 1.0e-4))
    args.optimizer = training_cfg.get("optimizer", "AdamW")
    args.betas = list(training_cfg.get("betas", [0.8, 0.99]))
    args.weight_decay = float(training_cfg.get("weight_decay", 0.0))
    args.scheduler = training_cfg.get("scheduler", "ExponentialLR")
    args.gamma = float(training_cfg.get("gamma", 0.99))
    args.factor = float(training_cfg.get("factor", 0.5))
    args.lr_scheduler_patience_epochs = int(training_cfg.get("lr_scheduler_patience_epochs", 2))
    args.lr_scheduler_min_delta = float(training_cfg.get("lr_scheduler_min_delta", 1.0e-4))
    args.min_lr = float(training_cfg.get("min_lr", 0.0))
    args.early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
    grad_clip_norm = training_cfg.get("grad_clip_norm", None)
    args.grad_clip_norm = None if grad_clip_norm in {None, False, "none", "None", "null", "Null", 0, 0.0} else float(grad_clip_norm)
    model = get_model(cfg.model_name, **OmegaConf.to_container(cfg.model, resolve=True)).to(args.device)
    train_loader, val_loader, test_loader, x_test_original, y_test_original = build_loaders(
        args.dataset_pkl,
        args.batch_size,
    )

    if not args.skip_train:
        writer = SummaryWriter(str(log_dir))
        try:
            train_model(model, train_loader, val_loader, args, checkpoint_dir, writer)
        finally:
            writer.close()

    test_model(model, test_loader, checkpoint_dir, x_test_original, y_test_original, output_pkl, args.device)


if __name__ == "__main__":
    main()
