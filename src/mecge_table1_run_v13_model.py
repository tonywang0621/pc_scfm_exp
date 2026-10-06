"""Isolated v13 runner; the original official runner remains unchanged.

The primary selector is the composite validation loss, including terminal v13
weights. A second checkpoint uses deterministic denoising SSD over every
validation window. All four diagnostics use the unchanged official metrics.
Neither its loader nor its forward pass advances training's random streams.
Training-state pickle files are trusted local artifacts, never external input.
"""

import argparse
import csv
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import get_model
from utils_ecg import cosine_similarity, maximum_absolute_distance, prd_mecge_official, ssd
from mecge_table1_run_official_local_model import (
    atomic_pickle_dump,
    atomic_torch_save,
    build_loaders,
    build_optimizer,
    build_scheduler,
    step_scheduler,
)


HISTORY_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_ssd", "val_mad", "val_prd", "val_cossim", "val_count",
    "topk_weight", "val_topk_weight",
    "extra_mse_weight", "val_extra_mse_weight", "raw_cos_weight", "val_raw_cos_weight",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-pkl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--ssd-output-pkl", default=None,
                        help="Also export test predictions from best_val_ssd.pt to this path.")
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
    parser.add_argument("--eval-checkpoint", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Exact resume requires the same visible CUDA devices as the saved run.")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


@contextmanager
def isolated_evaluation_rng(seed=0):
    state = capture_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        restore_rng_state(state)


def full_validation_loader(val_loader):
    """Reuse the exact official split, including its final partial batch."""
    return DataLoader(val_loader.dataset, batch_size=val_loader.batch_size,
                      shuffle=False, drop_last=False, num_workers=0)


def denoise(model, noisy):
    return model.denoising(noisy) if hasattr(model, "denoising") else model(noisy)


def deterministic_validation_metrics(model, val_loader, device):
    if len(val_loader.dataset) == 0:
        raise ValueError("The validation dataset is empty.")
    was_training = model.training
    clean_windows, predictions = [], []
    count = 0
    try:
        # Loader iteration itself draws a Torch base seed, even with zero workers.
        with isolated_evaluation_rng(), torch.no_grad():
            model.eval()
            for noisy, clean in val_loader:
                output = denoise(model, noisy.to(device))
                if output.shape != clean.shape:
                    raise ValueError("Deterministic validation prediction shape must match clean targets.")
                if not torch.isfinite(output).all() or not torch.isfinite(clean).all():
                    raise FloatingPointError("Non-finite deterministic validation prediction.")
                clean_windows.append(clean.flatten(start_dim=1).cpu().numpy().astype(np.float32))
                predictions.append(output.flatten(start_dim=1).cpu().numpy().astype(np.float32))
                count += len(noisy)
    finally:
        model.train(was_training)
    if count != len(val_loader.dataset):
        raise ValueError("Deterministic validation must include every window; use drop_last=False.")
    # Official PRD uses one global clean mean over the complete split, including
    # the partial final batch; computing PRD separately per batch changes it.
    clean, prediction = np.concatenate(clean_windows), np.concatenate(predictions)
    values = {
        "val_ssd": ssd(clean, prediction),
        "val_mad": maximum_absolute_distance(clean, prediction),
        "val_prd": prd_mecge_official(clean, prediction),
        "val_cossim": cosine_similarity(clean, prediction),
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Non-finite deterministic validation metric.")
    return {**{key: float(np.asarray(value, dtype=np.float64).mean())
               for key, value in values.items()}, "val_count": count}


def loss_weights(model):
    """Read scheduled weights without coupling the runner to core internals."""
    result = {}
    for name, method in (("topk_weight", "tail_weight"),
                         ("extra_mse_weight", "extra_mse_weight"),
                         ("raw_cos_weight", "raw_cos_weight")):
        getter = getattr(model, method, None)
        result[name] = float(getter()) if getter is not None else 0.0
    return result


def write_loss_history(checkpoint_dir, history):
    path = checkpoint_dir / "loss_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def save_training_state(path, model, optimizer, scheduler, epoch_no,
                        best_valid_loss, best_val_ssd, patience_counter,
                        history, total_epochs):
    atomic_torch_save({
        "format_version": 1,
        "runner_version": "v13",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch_no,
        "total_epochs": total_epochs,
        "best_valid_loss": best_valid_loss,
        "best_val_ssd": best_val_ssd,
        "patience_counter": patience_counter,
        "history": history,
        "rng_state": capture_rng_state(),
    }, path)


def train_model(model, train_loader, val_loader, args, checkpoint_dir, writer=None):
    for label, loader in (("training", train_loader), ("validation loss", val_loader)):
        if len(loader) == 0:
            raise ValueError(f"The {label} loader has no complete batches; reduce training.batch_size.")
    if args.epochs < 1:
        raise ValueError("The number of epochs must be positive.")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    training_state_path = checkpoint_dir / "training_state.pt"
    metric_loader = full_validation_loader(val_loader)
    best_valid_loss, best_val_ssd = float("inf"), float("inf")
    history, start_epoch, patience_counter = [], 0, 0

    if args.resume:
        resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else training_state_path
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint was not found: {resume_checkpoint}")
        # This checkpoint contains Python/NumPy RNG state in addition to tensors.
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if (state.get("format_version") != 1 or state.get("runner_version") != "v13"
                or "rng_state" not in state):
            raise ValueError("Exact v13 resume requires a v13 training_state.pt with RNG state.")
        if state["total_epochs"] != args.epochs:
            raise ValueError("Resume must keep the original epoch budget so the top-k schedule is unchanged.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        best_valid_loss = float(state["best_valid_loss"])
        best_val_ssd = float(state["best_val_ssd"])
        history = list(state["history"])
        patience_counter = int(state["patience_counter"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng_state(state["rng_state"])
        print(f"Resumed v13 training at epoch {start_epoch + 1} from {resume_checkpoint}.")
        write_loss_history(checkpoint_dir, history)
        if args.patience > 0 and patience_counter >= args.patience:
            print("Resume checkpoint already reached early stopping patience; exporting saved checkpoints.")
            return history

    for epoch_no in range(start_epoch, args.epochs):
        model.set_epoch(epoch_no + 1, args.epochs)
        model.train()
        training_weights = loss_weights(model)
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch_no + 1}: train") as batches:
            for batch in batches:
                optimizer.zero_grad()
                loss = model.compute_loss(batch, args.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch_no + 1}.")
                loss.backward()
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()
                train_loss += loss.item()
        current_train_loss = train_loss / len(train_loader)

        # Preserve the original validation loss's random draws and drop_last flow.
        model.eval()
        valid_loss = 0.0
        with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch_no + 1}: validation") as batches:
            for batch in batches:
                loss = model.compute_loss(batch, args.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch_no + 1}.")
                valid_loss += loss.item()
        current_valid_loss = valid_loss / len(val_loader)
        metrics = deterministic_validation_metrics(model, metric_loader, args.device)
        row = {"epoch": epoch_no + 1, "train_loss": current_train_loss,
               "val_loss": current_valid_loss, **metrics,
               **training_weights, **{"val_" + key: value for key, value in loss_weights(model).items()}}
        history.append(row)
        if writer is not None:
            for key in HISTORY_FIELDS:
                if key in {"epoch", "val_count"}:
                    continue
                writer.add_scalar(key, row[key], epoch_no)
        step_scheduler(scheduler, current_valid_loss)

        if best_valid_loss > current_valid_loss + args.early_stopping_min_delta:
            best_valid_loss, patience_counter = current_valid_loss, 0
            atomic_torch_save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
        if metrics["val_ssd"] < best_val_ssd:
            best_val_ssd = metrics["val_ssd"]
            atomic_torch_save(model.state_dict(), checkpoint_dir / "best_val_ssd.pt")
        atomic_torch_save(model.state_dict(), checkpoint_dir / "model_last.pt")
        save_training_state(training_state_path, model, optimizer, scheduler, epoch_no,
                            best_valid_loss, best_val_ssd, patience_counter, history, args.epochs)
        write_loss_history(checkpoint_dir, history)
        print(f"Epoch {epoch_no + 1}: val_loss={current_valid_loss:.7g}, "
              f"SSD={metrics['val_ssd']:.7g}, MAD={metrics['val_mad']:.7g}, "
              f"PRD={metrics['val_prd']:.7g}, CosSim={metrics['val_cossim']:.7g}, "
              f"training weights={training_weights}")
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch_no + 1}.")
            break
    return history


def test_model(model, test_loader, checkpoint_dir, x_test_original, y_test_original,
               output_pkl, device, eval_checkpoint=None):
    checkpoint = Path(eval_checkpoint) if eval_checkpoint else Path(checkpoint_dir) / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    restored = []
    with isolated_evaluation_rng(), torch.no_grad():
        for noisy, _clean in tqdm(test_loader, desc=f"Test: {checkpoint.name}"):
            output = denoise(model, noisy.to(device))
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"Non-finite test predictions from {checkpoint}.")
            restored.append(output.permute(0, 2, 1).cpu().numpy())
    if not restored:
        raise ValueError("The test dataset is empty.")
    prediction = np.concatenate(restored).astype(np.float32)
    atomic_pickle_dump([x_test_original, y_test_original, prediction], Path(output_pkl))
    print(f"Saved {output_pkl} from {checkpoint.name}")


def main(argv=None):
    import models.lstm_v13  # noqa: F401 -- opt-in registration, no legacy changes
    from v13_experiments import check_output_root

    args = parse_args(argv)
    args.device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_dir, log_dir = Path(args.checkpoint_dir).resolve(), Path(args.log_dir).resolve()
    output_pkl = Path(args.output_pkl).resolve()
    if args.ssd_output_pkl and Path(args.ssd_output_pkl).resolve() == output_pkl:
        raise ValueError("The two checkpoint selectors need distinct output-pkl paths.")
    for path in (checkpoint_dir, log_dir, output_pkl, args.ssd_output_pkl):
        if path is not None:
            check_output_root(Path(path))
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    training = cfg.get("training", {})
    if cfg.model_name != "lstm_metric_balanced_v13_ecg":
        raise ValueError("The isolated v13 runner requires model_name=lstm_metric_balanced_v13_ecg.")
    if training.get("selection_metric", "val_loss") != "val_loss":
        raise ValueError("Primary selection remains val_loss; best_val_ssd is reported separately.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    args.epochs = int(args.max_epochs if args.max_epochs is not None else training.get("train_epochs", 30))
    args.patience = int(args.early_stopping_patience if args.early_stopping_patience is not None
                        else training.get("early_stopping_patience_epochs", 0))
    args.batch_size = int(args.batch_size if args.batch_size is not None else training.get("batch_size", 96))
    args.lr = float(args.lr if args.lr is not None else training.get("lr", 1.0e-4))
    args.optimizer = training.get("optimizer", "AdamW")
    args.betas = list(training.get("betas", [0.8, 0.99]))
    args.weight_decay = float(training.get("weight_decay", 0.0))
    args.scheduler = training.get("scheduler", "ExponentialLR")
    args.gamma = float(training.get("gamma", 0.99))
    args.factor = float(training.get("factor", 0.5))
    args.lr_scheduler_patience_epochs = int(training.get("lr_scheduler_patience_epochs", 2))
    args.lr_scheduler_min_delta = float(training.get("lr_scheduler_min_delta", 1e-4))
    args.min_lr = float(training.get("min_lr", 0.0))
    args.early_stopping_min_delta = float(training.get("early_stopping_min_delta", 0.0))
    clipping = training.get("grad_clip_norm", None)
    args.grad_clip_norm = None if clipping in {None, False, "none", "None", "null", "Null", 0, 0.0} else float(clipping)
    model = get_model(cfg.model_name, **OmegaConf.to_container(cfg.model, resolve=True)).to(args.device)
    train_loader, val_loader, test_loader, original_noisy, original_clean = build_loaders(args.dataset_pkl, args.batch_size)
    if not args.skip_train:
        writer = SummaryWriter(str(log_dir))
        try:
            train_model(model, train_loader, val_loader, args, checkpoint_dir, writer)
        finally:
            writer.close()
    test_model(model, test_loader, checkpoint_dir, original_noisy, original_clean,
               output_pkl, args.device, eval_checkpoint=args.eval_checkpoint)
    if args.ssd_output_pkl:
        test_model(model, test_loader, checkpoint_dir, original_noisy, original_clean,
                   args.ssd_output_pkl, args.device, eval_checkpoint=checkpoint_dir / "best_val_ssd.pt")


if __name__ == "__main__":
    main()
