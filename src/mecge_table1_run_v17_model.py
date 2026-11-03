"""v17 runner: matched shuffle, full validation, exact epoch resume and diagnostics."""
import argparse
import csv
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import get_model
import models.lstm_v12  # opt-in controls
import models.lstm_v17  # opt-in registration
from models.lstm_v17.audit import audit_parameters
from mecge_table1_run_official_local_model import atomic_torch_save, build_loaders, build_optimizer, build_scheduler, step_scheduler
from mecge_table1_run_v12_model import capture_rng_state, restore_rng_state, isolated_evaluation_rng, full_validation_loader, test_model
from mecge_table1_collect_official_protocol import metric_values
from v11_experiments import atomic_json


class LegacyControl(torch.nn.Module):
    """Expose epoch/top-k APIs without changing the legacy model's parameters."""
    def __init__(self, model):
        super().__init__()
        self.core = model.core

    def forward(self, noisy):
        return self.core.restore(noisy)

    def denoising(self, noisy):
        return self(noisy)

    def compute_loss(self, batch, device):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        return self.core(clean, noisy)

    def set_epoch(self, epoch, total_epochs):
        if hasattr(self.core, "set_epoch"):
            self.core.set_epoch(epoch, total_epochs)

    def tail_weight(self):
        return float(self.core.h.get("lambda_max", 0.05))


def make_model(config):
    model = get_model(config.model_name, **OmegaConf.to_container(config.model, resolve=True))
    if not hasattr(model, "set_epoch"):
        model = LegacyControl(model)
    return model


def train_settings(config, device, seed, resume=False):
    h = config.training
    return SimpleNamespace(device=torch.device(device), seed=seed, resume=resume,
                           epochs=int(h.train_epochs), lr=float(h.lr), optimizer=h.optimizer,
                           betas=list(h.betas), weight_decay=float(h.weight_decay), scheduler=h.scheduler,
                           gamma=float(h.get("gamma", 0.99)), factor=float(h.factor),
                           lr_scheduler_patience_epochs=int(h.lr_scheduler_patience_epochs),
                           lr_scheduler_min_delta=float(h.lr_scheduler_min_delta), min_lr=float(h.min_lr),
                           grad_clip_norm=float(h.grad_clip_norm))


def profile_model(model, noisy, repeats=10):
    was_training = model.training
    device = noisy.device
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    try:
        with isolated_evaluation_rng(), torch.no_grad():
            model.eval()
            for _ in range(3):
                model.denoising(noisy)
            sync()
            started = time.perf_counter()
            for _ in range(repeats):
                model.denoising(noisy)
            sync()
            elapsed = time.perf_counter()-started
    finally:
        model.train(was_training)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "batch_size": len(noisy), "length": noisy.shape[-1], "device": str(device),
            "torch_version": torch.__version__, "repeats": repeats,
            "milliseconds_per_batch": 1000*elapsed/repeats,
            "windows_per_second": repeats*len(noisy)/elapsed,
            "torch_num_threads": torch.get_num_threads(),
            "flow_nfe": getattr(model.core, "steps", getattr(model.core, "cfm_inference_steps", None))
                        if getattr(model.core, "residual_flow", None) is not None and getattr(model.core, "use_fm", True) else 0,
            "direct_refiner": bool(getattr(model.core, "use_flow", False) and not getattr(model.core, "use_fm", True))}


def validate(model, loader, device):
    old_training = model.training
    clean_arrays, prediction_arrays = [], []
    sums, waveforms, count, loss_sum = {}, {}, 0, 0.0
    try:
        with isolated_evaluation_rng(), torch.no_grad():
            model.eval()
            for noisy, clean in full_validation_loader(loader):
                noisy, clean = noisy.to(device), clean.to(device)
                batch = len(noisy)
                loss = model.compute_loss((noisy, clean), device)
                loss_sum += float(loss) * batch
                if hasattr(model, "stages"):
                    stages = model.stages(noisy)
                    final = stages["final"]
                    for key in ("direct", "pre_flow", "baseline_candidate", "clean_path", "coarse_fusion"):
                        sums[key + "_ssd"] = sums.get(key + "_ssd", 0.0) + float((stages[key]-clean).square().flatten(1).sum())
                    improved = (final-clean).square().flatten(1).sum(1) < (stages["pre_flow"]-clean).square().flatten(1).sum(1)
                    sums["fraction_improved_ssd"] = sums.get("fraction_improved_ssd", 0.0) + float(improved.sum())
                    names = (["flow"] if model.core.use_flow else []) + (["baseline_blend"] if model.core.independent else [])
                    for index, name in enumerate(names):
                        g = stages["gates"][:, index]
                        for suffix, value in (("mean", g.mean(-1)), ("temporal_std", g.std(-1, unbiased=False)),
                                              ("saturated_fraction", ((g < .01) | (g > .99)).float().mean(-1))):
                            key = f"gate_{name}_{suffix}"
                            sums[key] = sums.get(key, 0.0) + float(value.sum())
                else:
                    final, stages = model.denoising(noisy), {}
                if not torch.isfinite(loss) or not torch.isfinite(final).all():
                    raise FloatingPointError("Non-finite validation loss/prediction")
                if count == 0:
                    waveforms = {key: value[:16].detach().cpu().numpy() for key, value in
                                 dict(noisy=noisy, clean=clean, final=final,
                                      **{k: v for k, v in stages.items() if k not in {"final", "spectrum", "condition"}}).items()}
                clean_arrays.append(clean.flatten(1).cpu().numpy())
                prediction_arrays.append(final.flatten(1).cpu().numpy())
                count += batch
    finally:
        model.train(old_training)
    if count != len(loader.dataset) or not count:
        raise ValueError("Validation must include every official validation window")
    metrics = metric_values(np.concatenate(clean_arrays), np.concatenate(prediction_arrays))
    if not all(np.isfinite(value).all() for value in metrics.values()):
        raise FloatingPointError("Non-finite official validation metrics")
    return {"val_loss": loss_sum/count, "val_count": count,
            **{f"val_{name.lower()}": float(value.mean()) for name, value in metrics.items()},
            **{key: value/count for key, value in sums.items()}}, waveforms


def write_history(path, history):
    fields = list(dict.fromkeys(key for row in history for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def train_model(model, train_loader, val_loader, settings, checkpoint, writer=None, stop_after=None):
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    optimizer, scheduler = build_optimizer(model, settings), None
    scheduler = build_scheduler(optimizer, settings)
    history, start, best_ssd, best_loss = [], 0, float("inf"), float("inf")
    best_epochs = {}
    state_path = checkpoint / "training_state.pt"
    if settings.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format_version") != 17 or state["total_epochs"] != settings.epochs:
            raise ValueError("Resume requires a v17 state with the unchanged epoch budget")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        history, start = state["history"], state["epoch"]
        best_ssd, best_loss, best_epochs = state["best_ssd"], state["best_loss"], state["best_epochs"]
        restore_rng_state(state["rng"])
    else:
        # Compaction/new module construction consumes different random counts.
        # Give all fresh jobs the same training stream after initialization.
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
    if len(train_loader.dataset) < train_loader.batch_size:
        raise ValueError("Training data must contain at least one complete batch")
    for epoch in range(start + 1, settings.epochs + 1):
        model.set_epoch(epoch, settings.epochs)
        model.train()
        generator = torch.Generator().manual_seed((settings.seed + epoch) % (2**63-1))
        loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True,
                            drop_last=True, num_workers=0, generator=generator)
        total, samples, components = 0.0, 0, {}
        started = time.perf_counter()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch, settings.device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            n = len(batch[0])
            total, samples = total + float(loss.detach()) * n, samples + n
            for key, value in getattr(model.core, "loss_components", {}).items():
                components[key] = components.get(key, 0.0) + value*n
        metrics, waveforms = validate(model, val_loader, settings.device)
        step_scheduler(scheduler, metrics["val_ssd"])
        row = {"epoch": epoch, "train_loss": total/samples, **metrics,
               "topk_weight": (float(model.core.h.get("lambda_max", 0.05))
                               if "max" in model.core.loss_fn else model.tail_weight()),
               "lr": optimizer.param_groups[0]["lr"],
               "last_grad_norm": float(norm), "seconds": time.perf_counter()-started,
               **{"train_" + key: value/samples for key, value in components.items()}}
        history.append(row)
        if metrics["val_ssd"] < best_ssd:
            best_ssd, best_epochs["val_ssd"] = metrics["val_ssd"], epoch
            atomic_torch_save(model.state_dict(), checkpoint / "best_model.pt")
            np.savez_compressed(checkpoint / "validation_selected_waveforms.npz", **waveforms)
        if metrics["val_loss"] < best_loss:
            best_loss, best_epochs["val_loss"] = metrics["val_loss"], epoch
            atomic_torch_save(model.state_dict(), checkpoint / "best_loss.pt")
        atomic_torch_save(model.state_dict(), checkpoint / "model_last.pt")
        atomic_torch_save(dict(format_version=17, epoch=epoch, total_epochs=settings.epochs,
                               model=model.state_dict(), optimizer=optimizer.state_dict(),
                               scheduler=scheduler.state_dict() if scheduler is not None else None,
                               history=history, best_ssd=best_ssd, best_loss=best_loss,
                               best_epochs=best_epochs, rng=capture_rng_state()), state_path)
        write_history(checkpoint / "loss_history.csv", history)
        atomic_json(checkpoint / "selection.json", {"primary": "val_ssd", "secondary": "val_loss", "best_epochs": best_epochs})
        if writer is not None:
            for key, value in row.items():
                writer.add_scalar(key, value, epoch)
        print(f"Epoch {epoch}/{settings.epochs}: train={total/samples:.6g}, "
              f"SSD={metrics['val_ssd']:.6g}, MAD={metrics['val_mad']:.6g}, CosSim={metrics['val_cossim']:.6g}", flush=True)
        if stop_after is not None and epoch >= stop_after:
            break
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "dataset-pkl", "checkpoint-dir", "log-dir", "output-pkl", "loss-output-pkl"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--skip-train", action="store_true")
    args = parser.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config)
    settings = train_settings(cfg, args.device, args.seed, args.resume)
    checkpoint = Path(args.checkpoint_dir)
    checkpoint.mkdir(parents=True, exist_ok=True)
    if Path(args.output_pkl).resolve() == Path(args.loss_output_pkl).resolve():
        raise ValueError("Checkpoint selectors require distinct result paths")
    model = make_model(cfg).to(settings.device)
    loaders = build_loaders(args.dataset_pkl, int(cfg.training.batch_size))
    train, val, test, original_noisy, original_clean = loaders
    audit_path = checkpoint / "parameter_audit.json"
    if not args.skip_train and not args.resume:
        tensors = train.dataset.tensors
        strict = cfg.model_name in {"lstm_cooperative_v17_ecg", "lstm_v17_compact_control_ecg"}
        audit = audit_parameters(model, tensors[0][:2].to(settings.device), tensors[1][:2].to(settings.device), strict=strict)
        audit["legacy_control"] = not strict
        atomic_json(audit_path, audit)
        with SummaryWriter(str(args.log_dir)) as writer:
            train_model(model, train, val, settings, checkpoint, writer)
    elif not args.skip_train:
        if not audit_path.is_file():
            raise FileNotFoundError("Resume requires the original parameter audit")
        with SummaryWriter(str(args.log_dir)) as writer:
            train_model(model, train, val, settings, checkpoint, writer)
    test_model(model, test, checkpoint, original_noisy, original_clean, args.output_pkl,
               settings.device, eval_checkpoint=checkpoint / "best_model.pt")
    atomic_json(checkpoint / "profile.json", profile_model(model, val.dataset.tensors[0][:2].to(settings.device)))
    test_model(model, test, checkpoint, original_noisy, original_clean, args.loss_output_pkl,
               settings.device, eval_checkpoint=checkpoint / "best_loss.pt")


if __name__ == "__main__":
    main()
