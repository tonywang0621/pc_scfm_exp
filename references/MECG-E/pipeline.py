#============================================================
#
#  Deep Learning BLW Filtering
#  Deep Learning pipelines
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import argparse
import csv
import torch
import datetime
import json
import os
import yaml
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import numpy as np
from models.MECGE import MECGE


def _model_weight_path(experiment, n_type, nv):
    return os.environ.get(
        "MECGE_MODEL_WEIGHT_PATH",
        'model_weight/' + experiment + f'_{n_type}_nv{nv}_weights.pth',
    )


def _artifact_path(filename):
    artifact_dir = os.environ.get("MECGE_ARTIFACT_DIR")
    if not artifact_dir:
        return None
    os.makedirs(artifact_dir, exist_ok=True)
    return os.path.join(artifact_dir, filename)


def _atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_replace_writer(path, write_fn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    write_fn(tmp_path)
    os.replace(tmp_path, path)


def _atomic_savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.png"
    fig.savefig(tmp_path, dpi=200)
    os.replace(tmp_path, path)


def _save_training_state(model, optimizer, scheduler, epoch_no, best_valid_loss, patience_counter, train_loss_history, val_loss_history):
    path = _artifact_path("training_state.pt")
    if not path:
        return
    _atomic_torch_save(
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


def _write_validation_metrics(val_loss_history):
    path = _artifact_path("validation_metrics.yaml")
    if not path:
        return
    def write(tmp_path):
        with open(tmp_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(val_loss_history, handle, sort_keys=False)
    _atomic_replace_writer(path, write)


def _write_loss_artifacts(train_loss_history, val_loss_history):
    csv_path = _artifact_path("loss_history.csv")
    if not csv_path:
        return
    rows = {}
    for item in train_loss_history:
        rows.setdefault(item["epoch"], {"epoch": item["epoch"], "train_loss": "", "val_loss": ""})
        rows[item["epoch"]]["train_loss"] = item["train_loss"]
    for item in val_loss_history:
        rows.setdefault(item["epoch"], {"epoch": item["epoch"], "train_loss": "", "val_loss": ""})
        rows[item["epoch"]]["val_loss"] = item["val_loss"]
    def write_csv(tmp_path):
        with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(rows[key] for key in sorted(rows))
    _atomic_replace_writer(csv_path, write_csv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping loss plots because matplotlib is unavailable: {exc}")
        return

    def plot(items, key, title, filename):
        if not items:
            return
        path = _artifact_path(filename)
        epochs = [item["epoch"] for item in items]
        values = [item[key] for item in items]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, values, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _atomic_savefig(fig, path)
        plt.close(fig)

    plot(train_loss_history, "train_loss", "MECG-E Train Loss", "train_loss.png")
    plot(val_loss_history, "val_loss", "MECG-E Validation Loss", "val_loss.png")
    if train_loss_history and val_loss_history:
        path = _artifact_path("train_val_loss.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([item["epoch"] for item in train_loss_history], [item["train_loss"] for item in train_loss_history], label="Train Loss", linewidth=2)
        ax.plot([item["epoch"] for item in val_loss_history], [item["val_loss"] for item in val_loss_history], label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("MECG-E Train / Validation Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _atomic_savefig(fig, path)
        plt.close(fig)


def train_dl(Dataset, experiment, n_type, config, nv, tb_writer, valid_epoch_interval=1, signal_size=512):

    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))
    model_filepath = _model_weight_path(experiment, n_type, nv)
    model_last_filepath = _artifact_path("model_last.pt")
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    [X_train, y_train, X_test, y_test] = Dataset

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)

    X_train = torch.FloatTensor(X_train)
    X_train = X_train.permute(0,2,1)
    
    y_train = torch.FloatTensor(y_train)
    y_train = y_train.permute(0,2,1)
    
    X_val = torch.FloatTensor(X_val)
    X_val = X_val.permute(0,2,1)
    
    y_val = torch.FloatTensor(y_val)
    y_val = y_val.permute(0,2,1)

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0,2,1)
    
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0,2,1)
    

    train_set = TensorDataset(y_train, X_train)
    val_set = TensorDataset(y_val, X_val)
    test_set = TensorDataset(y_test, X_test)
    
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=0)
    
    # ==================
    # LOAD THE DL MODEL
    # ==================

    device = os.environ.get("MECGE_DEVICE", "cuda:0")
    model = MECGE(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']["lr"], betas=[0.8, 0.99])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
    
    best_valid_loss = 1e10
    train_loss_history = []
    val_loss_history = []
    start_epoch = 0
    patience = config['train'].get('early_stopping_patience_epochs', 30)
    patience = int(patience)
    patience_counter = 0
    resume = os.environ.get("MECGE_RESUME", "0") == "1"
    if resume:
        resume_checkpoint = os.environ.get("MECGE_RESUME_CHECKPOINT") or _artifact_path("training_state.pt")
        if not resume_checkpoint or not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"MECGE_RESUME=1 but resume checkpoint was not found: {resume_checkpoint}")
        state = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict"):
            lr_scheduler.load_state_dict(state["scheduler_state_dict"])
        best_valid_loss = float(state.get("best_valid_loss", best_valid_loss))
        train_loss_history = list(state.get("train_loss_history", []))
        val_loss_history = list(state.get("val_loss_history", []))
        patience_counter = int(state.get("patience_counter", 0))
        start_epoch = int(state["epoch"]) + 1
        print(f"Resumed MECG-E training from {resume_checkpoint} at epoch {start_epoch}.")
        if patience_counter >= patience:
            print(
                "Resume checkpoint already reached early stopping patience "
                f"({patience_counter}/{patience}); skipping training and running test."
            )
            _write_validation_metrics(val_loss_history)
            _write_loss_artifacts(train_loss_history, val_loss_history)
            return
    
    for epoch_no in range(start_epoch, config['train']["epochs"]):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                
                loss = model(clean_batch, noisy_batch)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                avg_loss += loss.item()
                
                #ema.update(model)
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
            
            lr_scheduler.step()
        current_train_loss = avg_loss / batch_no
        train_loss_history.append({"epoch": int(epoch_no + 1), "train_loss": float(current_train_loss)})
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        loss = model(clean_batch, noisy_batch)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            if tb_writer is not None:
                tb_writer.add_scalar('val_loss', avg_loss_valid / batch_no, epoch_no)
            current_valid_loss = avg_loss_valid / batch_no
            val_loss_history.append({"epoch": int(epoch_no + 1), "val_loss": float(current_valid_loss)})
            
            if best_valid_loss > current_valid_loss:
                best_valid_loss = current_valid_loss
                print("\n best loss is updated to ", current_valid_loss, "at", epoch_no,)
                patience_counter = 0
                _atomic_torch_save(model.state_dict(), model_filepath)
            else:
                patience_counter += 1
                print(f"No validation loss improvement. Patience: {patience_counter}/{patience}")
        if model_last_filepath:
            _atomic_torch_save(model.state_dict(), model_last_filepath)
        _save_training_state(
            model,
            optimizer,
            lr_scheduler,
            epoch_no,
            best_valid_loss,
            patience_counter,
            train_loss_history,
            val_loss_history,
        )
        _write_validation_metrics(val_loss_history)
        _write_loss_artifacts(train_loss_history, val_loss_history)
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch_no + 1}.")
            break



def test_dl(Dataset, experiment, n_type, config, nv, device, signal_size=512):
    
    model = MECGE(config).to(device)

    model_filepath = _model_weight_path(experiment, n_type, nv)
    model.load_state_dict(torch.load(model_filepath,map_location='cpu'))
    model.eval()
    print('Deep Learning pipeline: Testing the model')

    [train_set, train_set_GT, X_test, y_test] = Dataset

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0,2,1)
    
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0,2,1)

    test_set = TensorDataset(y_test, X_test)
    test_loader = DataLoader(test_set, batch_size=50, num_workers=0)
    
    
    # ==================
    # LOAD THE DL MODEL
    # ==================

    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)

            output = model.denoising(noisy_batch) #B,1,L
            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1) #B,L,1
            out_numpy = output.cpu().detach().numpy()
            
            restored_sig.append(out_numpy)
    
    y_pred = np.concatenate(restored_sig)
    X_test = X_test.permute(0, 2, 1).cpu().detach().numpy()
    y_test = y_test.permute(0, 2, 1).cpu().detach().numpy()
    #np.save(foldername + '/denoised.npy', restored_sig)

    return [X_test, y_test, y_pred]
