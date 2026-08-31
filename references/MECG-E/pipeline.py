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
import torch
import datetime
import importlib.util
import json
import os
import sys
import yaml
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import numpy as np
from models.MECGE import MECGE


_PROJECT_MODELS_PACKAGE = "_mecge_table1_project_models"


def _load_project_models_package():
    app_dir = os.environ["MECGE_APP_DIR"]
    models_init = os.path.join(app_dir, "models", "__init__.py")
    if _PROJECT_MODELS_PACKAGE in sys.modules:
        return sys.modules[_PROJECT_MODELS_PACKAGE]
    spec = importlib.util.spec_from_file_location(
        _PROJECT_MODELS_PACKAGE,
        models_init,
        submodule_search_locations=[os.path.join(app_dir, "models")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PROJECT_MODELS_PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def build_model(config):
    model_name = os.environ.get("MECGE_PROJECT_MODEL_NAME")
    if not model_name:
        return MECGE(config), False
    project_models = _load_project_models_package()
    return project_models.get_model(model_name, **config.get("model", {})), True


def compute_training_loss(model, clean_batch, noisy_batch, device, is_project_model):
    if is_project_model and hasattr(model, "compute_loss"):
        return model.compute_loss((noisy_batch, clean_batch), device)
    return model(clean_batch, noisy_batch)


def as_nt(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.ndim != 2:
        raise ValueError(f"Expected [N, T], [N, 1, T], or [N, T, 1], got {array.shape}.")
    return array


def validation_metric_summary(clean, restored):
    clean = as_nt(clean)
    restored = as_nt(restored)
    diff = restored - clean
    ssd = np.sum(diff ** 2, axis=-1)
    mad = np.max(np.abs(diff), axis=-1)
    prd_den = np.sum((restored - np.mean(clean)) ** 2, axis=-1)
    prd = np.sqrt(ssd / (prd_den + 1.0e-10)) * 100.0
    cossim = np.sum(clean * restored, axis=-1) / (
        np.linalg.norm(clean, axis=-1) * np.linalg.norm(restored, axis=-1) + 1.0e-10
    )
    return {
        "SSD": float(np.nanmean(ssd)),
        "MAD": float(np.nanmean(mad)),
        "PRD": float(np.nanmean(prd)),
        "CosSim": float(np.nanmean(cossim)),
    }


def build_optimizer(model, config):
    train_config = config["train"]
    name = str(train_config.get("optimizer", "AdamW")).lower()
    lr = float(train_config["lr"])
    betas = tuple(train_config.get("betas", [0.8, 0.99]))
    weight_decay = float(train_config.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "radam":
        return torch.optim.RAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {train_config.get('optimizer')}")


def build_scheduler(optimizer, config):
    train_config = config["train"]
    name = str(train_config.get("scheduler", "ExponentialLR")).lower()
    if name == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(train_config.get("gamma", 0.99)))
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_config.get("factor", 0.5)),
            patience=int(train_config.get("lr_scheduler_patience_epochs", train_config.get("patience", 2))),
            threshold=float(train_config.get("lr_scheduler_min_delta", train_config.get("min_delta", 1.0e-4))),
            min_lr=float(train_config.get("min_lr", 0.0)),
        )
    if name in {"none", "null", "constantlr"}:
        return None
    raise ValueError(f"Unsupported scheduler: {train_config.get('scheduler')}")


def step_scheduler(scheduler, val_loss):
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def model_weight_path(experiment, n_type, nv):
    template = os.environ.get("MECGE_MODEL_WEIGHT_TEMPLATE")
    if template:
        return template.format(experiment=experiment, n_type=n_type, nv=nv)
    return 'model_weight/' + experiment + f'_{n_type}_nv{nv}_weights.pth'


def _checkpoint_sidecar_path(model_filepath, filename):
    return os.path.join(os.path.dirname(model_filepath), filename)


def _resume_checkpoint_path(experiment, n_type, nv, default_path):
    template = os.environ.get("MECGE_RESUME_CHECKPOINT")
    if template:
        return template.format(experiment=experiment, n_type=n_type, nv=nv)
    return default_path


def _atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _save_training_state(path, model, optimizer, scheduler, epoch_no, best_valid_loss, config, val_metric_history):
    _atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
            "epoch": epoch_no,
            "best_valid_loss": best_valid_loss,
            "patience_counter": config["train"].get("patience_counter", 0),
            "val_metric_history": val_metric_history,
            "config": config,
        },
        path,
    )


def train_dl(Dataset, experiment, n_type, config, nv, tb_writer, valid_epoch_interval=1, signal_size=512):

    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))
    model_filepath = model_weight_path(experiment, n_type, nv)
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    model_last_filepath = _checkpoint_sidecar_path(model_filepath, 'model_last.pt')
    training_state_filepath = _checkpoint_sidecar_path(model_filepath, 'training_state.pt')
    validation_metrics_filepath = _checkpoint_sidecar_path(model_filepath, 'validation_metrics.yaml')
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
    model, is_project_model = build_model(config)
    model = model.to(device)

    optimizer = build_optimizer(model, config)
    lr_scheduler = build_scheduler(optimizer, config)
    
    best_valid_loss = 1e10
    start_epoch = 0
    patience = config['train'].get('early_stopping_patience_epochs', config['train'].get('early_stopping_patience', 30))
    early_stopping_enabled = patience not in {None, False, "none", "None", "null", "Null", 0}
    patience = int(patience) if early_stopping_enabled else 0
    min_delta = float(config['train'].get('early_stopping_min_delta', 0.0))
    patience_counter = 0
    val_metric_history = []
    resume = os.environ.get("MECGE_RESUME", "0") == "1"
    if resume:
        resume_checkpoint = _resume_checkpoint_path(experiment, n_type, nv, training_state_filepath)
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"MECGE_RESUME=1 but resume checkpoint was not found: {resume_checkpoint}")
        state = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state["scheduler_state_dict"])
        best_valid_loss = state["best_valid_loss"]
        patience_counter = int(state.get("patience_counter", 0))
        val_metric_history = list(state.get("val_metric_history", []))
        start_epoch = int(state["epoch"]) + 1
        print(f"Resumed MECG-E training from {resume_checkpoint} at epoch {start_epoch}.")
        if early_stopping_enabled and patience_counter >= patience:
            print(
                "Resume checkpoint already reached early stopping patience "
                f"({patience_counter}/{patience}); skipping training and running test/robustness."
            )
            return
    else:
        for stale_path in [model_filepath, model_last_filepath, training_state_filepath, validation_metrics_filepath]:
            if os.path.exists(stale_path):
                os.remove(stale_path)
    
    for epoch_no in range(start_epoch, config['train']["epochs"]):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                
                loss = compute_training_loss(model, clean_batch, noisy_batch, device, is_project_model)
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
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            val_clean_batches = []
            val_pred_batches = []
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        loss = compute_training_loss(model, clean_batch, noisy_batch, device, is_project_model)
                        avg_loss_valid += loss.item()
                        pred_batch = model.denoising(noisy_batch) if hasattr(model, "denoising") else model(noisy_batch)
                        val_clean_batches.append(clean_batch.detach().cpu().numpy())
                        val_pred_batches.append(pred_batch.detach().cpu().numpy())
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
            current_step = (epoch_no + 1) * len(train_loader)
            val_metrics = validation_metric_summary(
                np.concatenate(val_clean_batches, axis=0),
                np.concatenate(val_pred_batches, axis=0),
            )
            improved = False
            if best_valid_loss > current_valid_loss + min_delta:
                best_valid_loss = current_valid_loss
                patience_counter = 0
                improved = True
                print("\n best loss is updated to ",current_valid_loss,"at", epoch_no + 1,)
                _atomic_torch_save(model.state_dict(), model_filepath)
            elif early_stopping_enabled:
                patience_counter += 1
                print(f"No validation loss improvement. Patience: {patience_counter}/{patience}")
            config["train"]["patience_counter"] = patience_counter
            val_record = {
                "epoch": int(epoch_no + 1),
                "step": int(current_step),
                "val_loss": float(current_valid_loss),
                **val_metrics,
            }
            val_metric_history.append(val_record)
            with open(validation_metrics_filepath, "w", encoding="utf-8") as handle:
                yaml.safe_dump(val_metric_history, handle, sort_keys=False)
            step_scheduler(lr_scheduler, current_valid_loss)
            if early_stopping_enabled and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch_no + 1}.")
                _atomic_torch_save(model.state_dict(), model_last_filepath)
                _save_training_state(
                    training_state_filepath,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch_no,
                    best_valid_loss,
                    config,
                    val_metric_history,
                )
                break
        _atomic_torch_save(model.state_dict(), model_last_filepath)
        _save_training_state(
            training_state_filepath,
            model,
            optimizer,
            lr_scheduler,
            epoch_no,
            best_valid_loss,
            config,
            val_metric_history,
        )



def test_dl(Dataset, experiment, n_type, config, nv, device, signal_size=512):
    
    model, _ = build_model(config)
    model = model.to(device)

    model_filepath = model_weight_path(experiment, n_type, nv)
    model.load_state_dict(torch.load(model_filepath,map_location='cpu'))
    model.eval()
    print('Deep Learning pipeline: Testing the model')

    [train_set, train_set_GT, X_test, y_test] = Dataset

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0,2,1)
    
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0,2,1)

    test_set = TensorDataset(y_test, X_test)
    test_loader = DataLoader(test_set, batch_size=config['train']['batch_size'], num_workers=0)
    
    
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
