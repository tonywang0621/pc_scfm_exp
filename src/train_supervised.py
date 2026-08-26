import torch 
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter 

import numpy as np 
from omegaconf import OmegaConf 
from pathlib import Path 
from utils import (
    seed_everything,
    setup_logger,
    plot_loss_curves,
    plot_prediction_results,
    get_reconstruction_metric_summary,
    reconstruction_metrics_from_arrays,
    profile_model_complexity,
)
import yaml 
import time 
import shutil 
import argparse
from tqdm.auto import tqdm

from models import get_model 
from datasets import get_dataset 


class _NullScheduler:
    def step(self, *args, **kwargs):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


def _build_optimizer(model, training_args):
    optimizer_name = str(training_args.get("optimizer", "AdamW")).lower()
    lr = float(training_args.lr)
    betas = tuple(training_args.get("betas", [0.8, 0.99]))
    weight_decay = float(training_args.get("weight_decay", 0.0))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    if optimizer_name == "radam":
        # EDDM paper-faithful optimizer (Li et al. 2025, Section IV-C2).
        return torch.optim.RAdam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {training_args.get('optimizer')}")


def _build_scheduler(optimizer, training_args):
    scheduler_name = str(training_args.get("scheduler", "ExponentialLR"))
    scheduler_key = scheduler_name.lower()

    if scheduler_key in {"none", "null", "constantlr"}:
        return _NullScheduler()
    if scheduler_key == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(training_args.get("gamma", 0.99))
        )
    if scheduler_key == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training_args.get("step_size", 200)),
            gamma=float(training_args.get("gamma", 0.5)),
        )
    if scheduler_key in {"reducelronplateau", "plateau"}:
        # DeepFilter paper-faithful schedule (Romero et al. 2021 / official
        # fperdigon/DeepFilter train_dl()): halve lr after `patience` epochs
        # without validation-loss improvement, down to a floor of min_lr.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_args.get("factor", 0.5)),
            patience=int(
                training_args.get(
                    "lr_scheduler_patience_epochs", training_args.get("patience", 2)
                )
            ),
            min_lr=float(training_args.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {training_args.get('scheduler')}")


def _scheduler_step(scheduler, metric=None):
    """Step `scheduler`, routing the validation metric to ReduceLROnPlateau
    schedulers (which require it) and ignoring it for epoch-based schedulers.
    No-ops for ReduceLROnPlateau when no fresh metric is available yet.
    """
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is not None:
            scheduler.step(metric)
        return
    scheduler.step()


def _save_training_state(
    path,
    model,
    optimizer,
    scheduler,
    step,
    best_val_loss,
    best_val_pcc,
    patience_counter,
    train_losses,
    val_losses,
    val_pccs,
    val_metric_history,
    args,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "best_val_pcc": best_val_pcc,
            "patience_counter": patience_counter,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_pccs": val_pccs,
            "val_metric_history": val_metric_history,
            "config": OmegaConf.to_container(args, resolve=True),
        },
        path,
    )


def train():     
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default="configs/ecg_baseline_wander_mecg_e.yaml")
    known_args, remaining = parser.parse_known_args()

    base_args = OmegaConf.load(known_args.config)
    cli_args = OmegaConf.from_dotlist(remaining) # load command line overrides
    args = OmegaConf.merge(base_args, cli_args)
    START_TIME = time.time() 
    separator = "-" * 100
    
    seed_everything(args.seed) 
    exp_name = args.exp_name 
    model_name = args.model_name 
    writer = None 
    metric_display_map = {
        'SSD': 'SSD (↓)',
        'MAD': 'MAD (↓)',
        'PRD': 'PRD % (↓)',
        'CosSim': 'CosSim (↑)',
        'Output_SNR_dB': 'Output SNR dB (↑)',
        'SNR_Improvement_dB': 'SNR Improvement dB (↑)',
        'LF_Reduction_dB': 'LF Reduction dB (↑)',
        'R_Peak_Timing_Error_ms': 'R-Peak Timing Error ms (↓)',
        'RR_Interval_MAE_ms': 'RR Interval MAE ms (↓)',
        'RMSE': 'RMSE (↓)',
        'Centered_CosSim': 'Centered CosSim (↑)',
        'QRS_Amplitude_Error': 'QRS Amplitude Error (↓)',
        'Parameters': 'Parameters',
        'Trainable_Parameters': 'Trainable Parameters',
        'FLOPs': 'FLOPs',
        'Inference_Time_ms': 'Inference Time ms',
        'Peak_Memory_MB': 'Peak Memory MB',
    }
    
    # Build a run tag from loss weights so each parameter combination gets its own dir
    _W_KEYS   = ['mse_w', 'pcc_w', 'spec_mag_w', 'spec_phase_w', 'grad_w', 'amp_w',
                 'stft_w', 'inst_phase_w', 'xcorr_lag_w', 'stft_mag_w', 'env_w']
    _W_ABBREV = {'mse_w': 'mse', 'pcc_w': 'pcc', 'spec_mag_w': 'smag',
                 'spec_phase_w': 'sph', 'grad_w': 'grad', 'amp_w': 'amp',
                 'stft_w': 'stft', 'inst_phase_w': 'iphase', 'xcorr_lag_w': 'xclag',
                 'stft_mag_w': 'stftmag', 'env_w': 'env'}
    model_cfg_flat = OmegaConf.to_container(args.model, resolve=True)
    weight_parts = [f"{_W_ABBREV[k]}{model_cfg_flat[k]}" for k in _W_KEYS if k in model_cfg_flat and model_cfg_flat[k] != 0]
    run_tag = "_".join(weight_parts)  # e.g. "mse0.1_pcc2.0_smag0.3_sph0.5_grad1.5_amp0.5"

    # setup directories — each unique weight combination gets its own subdirectory
    sub = Path(model_name) / run_tag if run_tag else Path(model_name)
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name / sub
    results_dir    = Path(args.results_dir)    / exp_name / sub
    log_dir        = Path(args.log_dir)        / exp_name / sub

    resume = bool(args.training.get("resume", False))
    resume_checkpoint_value = args.training.get("resume_checkpoint", None)
    resume_checkpoint = (
        Path(resume_checkpoint_value)
        if resume_checkpoint_value
        else checkpoint_dir / "training_state.pt"
    )

    # delete only the current run's directory (not siblings from other weight combos)
    if not resume:
        for path in [checkpoint_dir, results_dir, log_dir]:
            if path.exists():
                shutil.rmtree(path)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)    
    
    # print config 
    logger = setup_logger(args, log_dir) 
    if args.get("log_config", False):
        logger.info("\n"+"-"*30+"Running with configs"+"-"*30+"\n")
        logger.info(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))
        logger.info(separator+"\n")
    else:
        logger.info(
            f"Run: exp_name={exp_name} | model_name={model_name} | "
            f"train_epochs={args.training.get('train_epochs', 'step_override')} | "
            f"batch_size={args.training.batch_size}"
        )
    
    # setup tensorboard 
    if args.get('log_tensorboard', True):
        tb_dir = checkpoint_dir / "tensorboard_logs"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard logging enabled. Logs will be saved to {tb_dir}")
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    # print(f'Using device {device}')
    
    # model_configs = OmegaConf.load(args.model_config_path) 
    # model_kwargs = OmegaConf.to_container(model_configs, resolve=True)  
    # logger.info("\n"+"-"*30+"Model configs"+"-"*30+"\n")
    # logger.info(yaml.dump(model_kwargs, indent=4)) 
    # logger.info(separator+"\n")
    model_kwargs = OmegaConf.to_container(args.model, resolve=True)
    model = get_model(args['model_name'], **model_kwargs).to(device) 

    init_checkpoint_value = args.training.get("init_checkpoint", None)
    if init_checkpoint_value and not resume:
        init_checkpoint = Path(init_checkpoint_value)
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"training.init_checkpoint was not found: {init_checkpoint}")
        init_state = torch.load(init_checkpoint, map_location=device)
        if isinstance(init_state, dict) and "model_state_dict" in init_state:
            init_state = init_state["model_state_dict"]
        load_result = model.load_state_dict(init_state)
        logger.info(
            f"Initialized model weights from {init_checkpoint} | "
            f"missing_keys={len(load_result.missing_keys)} | "
            f"unexpected_keys={len(load_result.unexpected_keys)}"
        )

    optimizer = _build_optimizer(model, args.training)
    scheduler = _build_scheduler(optimizer, args.training)
    logger.info(
        f"Optimizer: {args.training.get('optimizer', 'AdamW')} | "
        f"lr={args.training.lr} | betas={list(args.training.get('betas', [0.8, 0.99]))} | "
        f"weight_decay={args.training.get('weight_decay', 0.0)}"
    )
    logger.info(
        f"Scheduler: {args.training.get('scheduler', 'ExponentialLR')} | "
        f"gamma={args.training.get('gamma', 'n/a')} | "
        f"step_size={args.training.get('step_size', 'n/a')} | "
        f"factor={args.training.get('factor', 'n/a')} | "
        f"lr_scheduler_patience_epochs={args.training.get('lr_scheduler_patience_epochs', 'n/a')} | "
        f"min_lr={args.training.get('min_lr', 'n/a')}"
    )
    if args.get("log_model_architecture", False):
        logger.info("\n"+"-"*30+"Model Architecture"+"-"*30+"\n")
        logger.info(f'{model}\n')
    logger.info(f'model size: {sum(param.numel() for param in model.parameters())} parameters')

    dataset_kwargs = OmegaConf.to_container(args.dataset, resolve=True)
    train_dataset = get_dataset(data_mode='train', **dataset_kwargs) 
    val_dataset = get_dataset(data_mode='val', **dataset_kwargs) 
    test_dataset = get_dataset(data_mode='test', **dataset_kwargs) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.training.batch_size, 
        shuffle=True, 
        num_workers=4, 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.training.batch_size, 
        shuffle=False, 
        num_workers=4, 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.training.batch_size, 
        shuffle=False, 
        num_workers=4, 
    )
    # print(f'data loaders sucessfully loaded')
    logger.info(f'data loaders sucessfully loaded')
    logger.info("\n"+"-"*30+"Begin Training"+"-"*30+"\n")

    # training
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Train loader is empty; cannot start training.")
    train_epochs = int(args.training.get("train_epochs", 30))
    if args.training.get("train_iterations", None) is not None:
        total_steps = int(args.training.train_iterations)
        train_epochs = int(np.ceil(total_steps / steps_per_epoch))
    else:
        total_steps = train_epochs * steps_per_epoch
    eval_every_epochs = int(args.training.get("eval_every_epochs", 1))
    validation_metrics_every_epochs = int(
        args.training.get("validation_metrics_every_epochs", eval_every_epochs)
    )
    save_every_epochs = int(args.training.get("save_every_epochs", 0))
    step_based_schedule = (
        args.training.get("train_iterations", None) is not None
        and args.training.get("eval_every", None) is not None
    )
    if step_based_schedule:
        eval_every_steps = max(int(args.training.eval_every), 1)
        validation_metrics_every_steps = max(
            int(args.training.get("validation_metrics_every", eval_every_steps)),
            eval_every_steps,
        )
        save_every_steps = int(args.training.get("save_every", 0))
    else:
        eval_every_steps = max(eval_every_epochs * steps_per_epoch, 1)
        validation_metrics_every_steps = max(
            validation_metrics_every_epochs * steps_per_epoch,
            eval_every_steps,
        )
        save_every_steps = save_every_epochs * steps_per_epoch if save_every_epochs else 0
    logger.info(
        "Training schedule: "
        f"train_epochs={train_epochs} | total_steps={total_steps} | "
        f"steps_per_epoch={steps_per_epoch} | eval_every_epochs={eval_every_epochs}"
    )

    train_losses, val_losses = [], []
    val_pccs = []
    val_metric_history = []
    best_val_loss = np.inf
    best_val_pcc = -np.inf
    best_model_ckpt = checkpoint_dir / 'best_model.pt'
    best_pcc_model_ckpt = checkpoint_dir / 'best_pcc_model.pt'
    training_state_ckpt = checkpoint_dir / 'training_state.pt'
    skip_training = not bool(getattr(model, "requires_training", True))

    patience = args.training.get(
        'early_stopping_patience_epochs',
        args.training.get('early_stopping_patience', 8),
    )
    min_delta = float(args.training.get('early_stopping_min_delta', 0.0))
    patience_counter = 0
    stop_training = False

    step = 0
    running_train_loss = 0
    running_train_steps = 0

    if skip_training:
        logger.info("Model is a fixed baseline; skipping optimization and evaluating directly.")
        torch.save(model.state_dict(), best_pcc_model_ckpt)
        torch.save(model.state_dict(), best_model_ckpt)
        torch.save(model.state_dict(), checkpoint_dir / 'model_last.pt')
        _save_training_state(
            training_state_ckpt,
            model,
            optimizer,
            scheduler,
            0,
            best_val_loss,
            best_val_pcc,
            patience_counter,
            train_losses,
            val_losses,
            val_pccs,
            val_metric_history,
            args,
        )
    elif resume:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(
                f"training.resume=True but resume checkpoint was not found: {resume_checkpoint}"
            )
        state = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        step = int(state.get("step", 0))
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        best_val_pcc = float(state.get("best_val_pcc", best_val_pcc))
        patience_counter = int(state.get("patience_counter", patience_counter))
        train_losses = list(state.get("train_losses", train_losses))
        val_losses = list(state.get("val_losses", val_losses))
        val_pccs = list(state.get("val_pccs", val_pccs))
        val_metric_history = list(state.get("val_metric_history", val_metric_history))
        logger.info(f"Resumed training from {resume_checkpoint} at step {step}.")
        if patience_counter >= patience:
            logger.info(
                f"Resume checkpoint already reached early stopping patience "
                f"({patience_counter}/{patience}); skipping training and running final evaluation."
            )
            stop_training = True

    progress_bar = tqdm(
        total=total_steps,
        initial=step,
        desc=f"Training {model_name}",
        unit="step",
        dynamic_ncols=True,
        disable=skip_training or not args.training.get("progress_bar", True),
    )

    try:
        while not skip_training and step < total_steps and not stop_training:
            model.train()

            for train_batch in train_loader:
                optimizer.zero_grad()
                train_loss = model.compute_loss(train_batch, device)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_train_loss += train_loss.item()
                running_train_steps += 1
                current_step = step + 1
                current_epoch = int(np.ceil(current_step / steps_per_epoch))
                epoch_finished = current_step % steps_per_epoch == 0
                eval_due = current_step % eval_every_steps == 0 if step_based_schedule else (
                    epoch_finished and current_step % eval_every_steps == 0
                )

                # evaluation
                if eval_due:
                    model.eval()
                    val_pcc_batches = []
                    last_metrics_step = int(val_metric_history[-1]["step"]) if val_metric_history else None
                    metrics_due = (
                        last_metrics_step is None
                        or current_step - last_metrics_step >= validation_metrics_every_steps
                    )
                    avg_val_loss = None
                    val_noisy_batches, val_clean_batches, val_pred_batches = [], [], []

                    with torch.no_grad():
                        for val_batch in val_loader:
                            noisy = val_batch[0].to(device)
                            clean = val_batch[1].to(device)
                            pred = model(noisy)
                            pred_c = pred - pred.mean(dim=-1, keepdim=True)
                            clean_c = clean - clean.mean(dim=-1, keepdim=True)
                            pcc = (pred_c * clean_c).sum(dim=-1) / (
                                pred_c.norm(dim=-1) * clean_c.norm(dim=-1) + 1e-8
                            )
                            val_pcc_batches.append(pcc)
                            if metrics_due:
                                val_noisy_batches.append(noisy.detach().cpu().numpy())
                                val_clean_batches.append(clean.detach().cpu().numpy())
                                val_pred_batches.append(pred.detach().cpu().numpy())

                    avg_val_pcc = torch.cat([batch.flatten() for batch in val_pcc_batches]).mean().item()
                    val_metrics = None
                    if metrics_due:
                        avg_val_loss = 0
                        for val_batch in val_loader:
                            avg_val_loss += model.compute_loss(val_batch, device).item()
                        avg_val_loss /= len(val_loader)
                        fs = args.dataset.get("resample_hz", args.model.get("sampling_rate", 250))
                        low_freq_hz = args.get("evaluation", {}).get("low_frequency_high_hz", 0.5)
                        val_metrics = reconstruction_metrics_from_arrays(
                            np.concatenate(val_noisy_batches, axis=0),
                            np.concatenate(val_clean_batches, axis=0),
                            np.concatenate(val_pred_batches, axis=0),
                            fs=fs,
                            low_freq_hz=low_freq_hz,
                            eps=args.dataset.eps,
                        )
                        val_metric_history.append(
                            {
                                "epoch": current_epoch,
                                "step": current_step,
                                "val_loss": avg_val_loss,
                                **val_metrics,
                            }
                        )
                        with open(checkpoint_dir / "validation_metrics.yaml", "w", encoding="utf-8") as f:
                            yaml.safe_dump(val_metric_history, f, sort_keys=False)

                    current_train_loss = running_train_loss / running_train_steps
                    log_message = (
                        f'Training epoch {current_epoch}/{train_epochs} '
                        f'(step {current_step}/{total_steps}) | train loss: {current_train_loss:.4f} | '
                        f'val PCC: {avg_val_pcc:.4f}'
                    )
                    if val_metrics is not None:
                        log_message += (
                            f' | val loss: {avg_val_loss:.4f} | '
                            f'val PRD: {val_metrics["PRD"]:.4f} | '
                            f'val SNR imp: {val_metrics["SNR_Improvement_dB"]:.4f} | '
                            f'val LF red: {val_metrics["LF_Reduction_dB"]:.4f} | '
                            f'val R-peak err: {val_metrics["R_Peak_Timing_Error_ms"]:.4f} ms | '
                            f'val RR MAE: {val_metrics["RR_Interval_MAE_ms"]:.4f} ms'
                        )
                    logger.info(log_message)

                    train_losses.append(current_train_loss)
                    val_losses.append(avg_val_loss if avg_val_loss is not None else float("nan"))
                    val_pccs.append(avg_val_pcc)

                    if writer is not None:
                        writer.add_scalar('Train/Loss', current_train_loss, current_step)
                        writer.add_scalar('Val/PCC', avg_val_pcc, current_step)
                        if avg_val_loss is not None:
                            writer.add_scalar('Val/Loss', avg_val_loss, current_step)
                        if val_metrics is not None:
                            for key, value in val_metrics.items():
                                writer.add_scalar(f'Val/{key}', value, current_step)

                    # Primary selection and early stopping use validation PCC only.
                    if avg_val_pcc > best_val_pcc + min_delta:
                        logger.info(f'New best val PCC {avg_val_pcc:.4f} at epoch {current_epoch}! Saving model...')
                        best_val_pcc = avg_val_pcc
                        torch.save(model.state_dict(), best_pcc_model_ckpt)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        logger.info(f'No PCC improvement. Patience: {patience_counter}/{patience}')

                    # Save best validation-loss checkpoint for auxiliary analysis only.
                    if avg_val_loss is not None and avg_val_loss < best_val_loss:
                        logger.info(f'New best val loss {avg_val_loss:.4f} at epoch {current_epoch}! Saving model...')
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), best_model_ckpt)

                    if patience_counter >= patience:
                        logger.info(f'Early stopping triggered at epoch {current_epoch}.')
                        torch.save(model.state_dict(), checkpoint_dir / 'model_last.pt')
                        step = current_step
                        progress_bar.update(1)
                        _save_training_state(
                            training_state_ckpt,
                            model,
                            optimizer,
                            scheduler,
                            step,
                            best_val_loss,
                            best_val_pcc,
                            patience_counter,
                            train_losses,
                            val_losses,
                            val_pccs,
                            val_metric_history,
                            args,
                        )
                        stop_training = True
                        break

                    _save_training_state(
                        training_state_ckpt,
                        model,
                        optimizer,
                        scheduler,
                        current_step,
                        best_val_loss,
                        best_val_pcc,
                        patience_counter,
                        train_losses,
                        val_losses,
                        val_pccs,
                        val_metric_history,
                        args,
                    )

                    # Reset running training metrics
                    running_train_loss = 0
                    running_train_steps = 0
                    model.train()
                    _scheduler_step(scheduler, avg_val_loss)
                elif epoch_finished:
                    _scheduler_step(scheduler)

                if save_every_steps and current_step % save_every_steps == 0:
                    torch.save(model.state_dict(), checkpoint_dir / f'model_step_{current_step}.pt')
                    _save_training_state(
                        training_state_ckpt,
                        model,
                        optimizer,
                        scheduler,
                        current_step,
                        best_val_loss,
                        best_val_pcc,
                        patience_counter,
                        train_losses,
                        val_losses,
                        val_pccs,
                        val_metric_history,
                        args,
                    )

                if current_step >= total_steps:
                    logger.info(f'Saving last model')
                    torch.save(model.state_dict(), checkpoint_dir / 'model_last.pt')
                    step = current_step
                    progress_bar.update(1)
                    _save_training_state(
                        training_state_ckpt,
                        model,
                        optimizer,
                        scheduler,
                        step,
                        best_val_loss,
                        best_val_pcc,
                        patience_counter,
                        train_losses,
                        val_losses,
                        val_pccs,
                        val_metric_history,
                        args,
                    )
                    break

                step = current_step
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=f"{current_epoch}/{train_epochs}",
                    train_loss=f"{train_loss.item():.4f}",
                    patience=f"{patience_counter}/{patience}",
                )
    finally:
        progress_bar.close()
            
    # print(f"Training complete after {(time.time()-START_TIME)/60:.2f} minutes. Best val loss: {best_val_loss:.4f}")
    logger.info(
        f"Training complete after {(time.time()-START_TIME)/60:.2f} minutes. "
        f"Best val loss: {best_val_loss:.4f} | Best val PCC: {best_val_pcc:.4f}"
    )
    # final evaluation — test best checkpoints independently
    eval_checkpoints = [
        ('best_pcc',  best_pcc_model_ckpt),
    ]

    for ckpt_name, ckpt_path in eval_checkpoints:
        if not ckpt_path.exists():
            logger.info(f"\n{'='*30} Skipping {ckpt_name} model ({ckpt_path} not found) {'='*30}")
            continue
        logger.info(f"\n{'='*30} Evaluating {ckpt_name} model {'='*30}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        eval_dir = results_dir / ckpt_name
        if eval_dir.exists():
            logger.info(f"Overwriting existing evaluation directory: {eval_dir}")
            shutil.rmtree(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        loss_curve_path = plot_loss_curves(
            train_losses, val_losses,
            eval_every_epochs, eval_dir, val_pccs=val_pccs, x_axis_label="Training epoch",
        )
        logger.info(f"Saved loss curves to {loss_curve_path}")

        fs = args.dataset.get("resample_hz", args.model.get("sampling_rate", 250))
        low_freq_hz = args.get("evaluation", {}).get("low_frequency_high_hz", 0.5)
        eval_items = [("ptbxl_fold10_test", test_dataset, test_loader)]
        for external_name in args.dataset.get("external_test_datasets", []):
            try:
                external_dataset = get_dataset(data_mode=external_name, **dataset_kwargs)
            except FileNotFoundError as exc:
                logger.info(f"Skipping external test dataset {external_name}: {exc}")
                continue
            external_loader = DataLoader(
                external_dataset,
                batch_size=args.training.batch_size,
                shuffle=False,
                num_workers=4,
            )
            eval_items.append((external_name, external_dataset, external_loader))

        for eval_name, eval_dataset, _ in eval_items:
            prediction_paths = plot_prediction_results(
                model, eval_dataset, device, eval_dir,
                num_samples=3, seed=args.seed, eps=args.dataset.eps, fs=fs,
                filename_prefix=eval_name,
            )
            for prediction_path in prediction_paths:
                logger.info(f"Saved prediction plot to {prediction_path}")

        for eval_name, _, eval_loader in eval_items:
            metrics_summary = get_reconstruction_metric_summary(
                model, eval_loader, device, fs=fs, low_freq_hz=low_freq_hz, eps=args.dataset.eps
            )
            logger.info('\n')
            logger.info(separator)
            logger.info(f"{ckpt_name} | {eval_name}")
            logger.info(f"{'Metric':>24} | {'Mean':>12} | {'Std':>12} | {'Count':>8}")
            logger.info(separator)
            for key, stats in metrics_summary.items():
                display_name = metric_display_map.get(key, key)
                logger.info(
                    f"{display_name:>24} | {stats['mean']:>12.4f} | "
                    f"{stats['std']:>12.4f} | {stats['count']:>8}"
                )
            logger.info(separator + "\n")
            with open(eval_dir / f"metrics_{eval_name}.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(metrics_summary, f, sort_keys=False)

        complexity_path = eval_dir / "complexity_summary.yaml"
        complexity = profile_model_complexity(
            model,
            device,
            input_length=int(args.dataset.get("window_size", 512)),
            batch_size=1,
        )
        logger.info('\n')
        logger.info(separator)
        logger.info(f"{ckpt_name} | complexity")
        logger.info(f"{'Metric':>24} | {'Value':>12}")
        logger.info(separator)
        for key, value in complexity.items():
            display_name = metric_display_map.get(key, key)
            logger.info(f"{display_name:>24} | {value:>12.4f}")
        logger.info(separator + "\n")
        with open(complexity_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(complexity, f, sort_keys=False)

    
            

    
    


if __name__ == "__main__": 
    train() 
