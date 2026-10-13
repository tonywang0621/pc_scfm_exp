"""Warm-start v16 runner with validation-only, four-metric checkpoint protection.

Trusted local datasets and training states use pickle. Test windows are exported
only after selection; epoch zero is never reported as an improved v16 model.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from models import get_model
from utils_ecg import cosine_similarity, maximum_absolute_distance, prd_mecge_official, ssd
from mecge_table1_run_official_local_model import atomic_torch_save, build_loaders
from mecge_table1_run_v14_model import (
    capture_rng_state, restore_rng_state, isolated_evaluation_rng,
    full_validation_loader, test_model,
)

METRICS = ("ssd", "mad", "prd", "cossim")
SELECTION_POLICY = {
    "metric_tolerance": 0.0,
    "constraints": "SSD/MAD/PRD <= frozen v7; CosSim >= frozen v7 on full validation",
    "strict_improvement_required": True,
    "ranking": ["highest CosSim", "lowest MAD", "earliest epoch"],
    "evaluation_seed": 0,
    "test_used_for_selection": False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset-pkl", "init-checkpoint", "checkpoint-dir", "log-dir", "output-pkl"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--baseline-output-pkl")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--skip-train", action="store_true")
    return parser.parse_args(argv)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def metric_arrays(clean, prediction):
    """Official PRD receives the entire split, preserving its global clean mean."""
    if clean.shape != prediction.shape or clean.ndim != 2 or not len(clean):
        raise ValueError("Metrics require equally shaped, nonempty [N,T] arrays.")
    values = {"ssd": ssd(clean, prediction), "mad": maximum_absolute_distance(clean, prediction),
              "prd": prd_mecge_official(clean, prediction), "cossim": cosine_similarity(clean, prediction)}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Non-finite validation metric.")
    return values


def feasible(metrics, baseline):
    if not all(math.isfinite(float(mapping[name])) for mapping in (metrics, baseline) for name in METRICS):
        return False
    return (all(metrics[name] <= baseline[name] for name in METRICS[:3])
            and metrics["cossim"] >= baseline["cossim"]
            and any(metrics[name] != baseline[name] for name in METRICS))


def prefer_candidate(metrics, epoch, selected):
    if selected is None:
        return True
    def rank(item, number):
        return (-item["cossim"], item["mad"], number)
    return rank(metrics, epoch) < rank(selected["metrics"], selected["epoch"])


WAVEFORM_LIMIT = 1024
GATE_GRID_POINTS = 5
PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-6
TRAIN_SAMPLER_POLICY = "independent epoch generator: (run_seed + epoch) modulo (2**63 - 1)"


def correction_geometry(clean, pre, final):
    """Exact Euclidean error identity; the two correction channels need not be orthogonal."""
    target = np.asarray(clean, np.float64) - np.asarray(pre, np.float64)
    correction = np.asarray(final, np.float64) - np.asarray(pre, np.float64)
    target_sq = np.sum(target * target, axis=1)
    correction_sq = np.sum(correction * correction, axis=1)
    dot = np.sum(target * correction, axis=1)
    scale = np.sqrt(target_sq * correction_sq)
    alignment = np.divide(dot, scale, out=np.zeros_like(dot), where=scale > 0)
    alpha = np.clip(np.divide(dot, correction_sq, out=np.zeros_like(dot), where=correction_sq > 0), 0, 1)
    delta = correction_sq - 2 * dot
    measured = np.sum((target - correction) ** 2, axis=1) - target_sq
    oracle_ssd = np.sum((target - alpha[:, None] * correction) ** 2, axis=1)
    return {"target_residual_l2": np.sqrt(target_sq), "correction_l2": np.sqrt(correction_sq),
            "alignment_cosine": alignment, "target_correction_dot": dot,
            "ssd_delta_identity": delta, "ssd_identity_residual": measured - delta,
            "shrink_oracle_alpha": alpha, "shrink_oracle_ssd": oracle_ssd,
            "shrink_oracle_ssd_delta": oracle_ssd - np.sum((target-correction)**2, axis=1)}


def bounded_gate_grid(stage, clean, bound):
    """Validation-label diagnostic: best sampled bounded offsets, never an inference rule.

    Evaluate a 5^3 grid using the full two-channel Gram matrix, including its
    cross term. The current prediction is an additional feasible candidate.
    """
    array = lambda value: torch.as_tensor(value).detach().double().cpu().numpy()
    pre = array(stage["pre_flow"]).reshape(len(clean), -1)
    target = np.asarray(clean, np.float64).reshape(len(clean), -1) - pre
    limited = array(stage["limited_flow"])
    clean_delta, baseline_delta = limited[:, 0], limited[:, 1]
    logits = array(stage["base_logits"])
    maxima = array(stage["gate_maxima"]).reshape(1, 1, 3)
    axes = np.linspace(-bound, bound, GATE_GRID_POINTS)
    grid = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), -1).reshape(-1, 3)
    gates = maxima / (1 + np.exp(-np.clip(logits[:, None, :] + grid[None], -700, 700)))
    a = (1 - gates[:, :, 2]) * gates[:, :, 0]
    b = -gates[:, :, 2] * gates[:, :, 1]
    inner = lambda x, y: np.sum(x*y, axis=1)[:, None]
    ssd_values = (inner(target, target) + a*a*inner(clean_delta, clean_delta)
                  + b*b*inner(baseline_delta, baseline_delta)
                  + 2*a*b*inner(clean_delta, baseline_delta)
                  - 2*a*inner(target, clean_delta) - 2*b*inner(target, baseline_delta))
    choice = np.argmin(ssd_values, axis=1)
    grid_best = np.maximum(ssd_values[np.arange(len(clean)), choice], 0)
    current = array(stage["final"]).reshape(len(clean), -1)
    current_ssd = np.sum((np.asarray(clean, np.float64).reshape(len(clean), -1) - current)**2, axis=1)
    use_grid = grid_best < current_ssd
    selected_offsets = np.where(use_grid[:, None], grid[choice], array(stage["adapter_offsets"]))
    best = np.where(use_grid, grid_best, current_ssd)
    return {"gate_grid_best_ssd": best, "gate_grid_ssd_delta": best-current_ssd,
            **{f"gate_grid_offset_{name}": selected_offsets[:, i]
               for i, name in enumerate(("clean", "baseline", "blend"))}}


def validate(model, loader, device, reference=None):
    """Full deterministic validation. Oracles are diagnostics and cannot select weights."""
    loader = full_validation_loader(loader)
    if not len(loader.dataset):
        raise ValueError("The validation dataset is empty.")
    old_training = model.training
    windows, diagnostics, waveform_chunks = {}, {}, {}
    parity_max, seen, retained = 0.0, 0, 0
    ref_training = reference.training if reference is not None else None
    to_array = lambda value: torch.as_tensor(value).detach().cpu().numpy()
    prediction_keys = ("pre_flow", "final", "direct", "budget_bypass_clean",
                       "budget_bypass_baseline", "budget_bypass_both")
    waveform_keys = ("pre_flow", "final", "direct", "raw_flow", "projected_flow",
                     "limited_flow", "gated_flow", "effective_flow", "gates", "adapter_offsets",
                     "budget_bypass_clean", "budget_bypass_baseline", "budget_bypass_both")
    try:
        with isolated_evaluation_rng(), torch.no_grad():
            model.eval()
            if reference is not None:
                reference.eval()
            for noisy, clean in loader:
                noisy = noisy.to(device)
                stage = model.stages(noisy)
                for key in prediction_keys:
                    if key not in stage:
                        continue
                    value = stage[key]
                    if value.shape != noisy.shape or not torch.isfinite(value).all():
                        raise ValueError("Every stage must be finite and have input shape.")
                    windows.setdefault(key, []).append(to_array(value.flatten(1)).astype(np.float32))
                if clean.shape != noisy.shape or not torch.isfinite(clean).all():
                    raise ValueError("Validation targets must be finite and have input shape.")
                final, pre = stage["final"], stage["pre_flow"]
                torch.testing.assert_close(final, model.denoising(noisy), rtol=PARITY_RTOL, atol=PARITY_ATOL,
                                           msg="Stage output differs from normal inference.")
                if reference is not None:
                    original = reference.denoising(noisy)
                    torch.testing.assert_close(final, original, rtol=PARITY_RTOL, atol=PARITY_ATOL,
                                               msg="Warm-start output differs from the frozen v7.")
                    parity_max = max(parity_max, float((final-original).abs().max()))
                target = to_array(clean.flatten(1)).astype(np.float32)
                windows.setdefault("clean", []).append(target)
                current = dict(stage.get("diagnostics", {}))
                current.update(correction_geometry(target, to_array(pre.flatten(1)), to_array(final.flatten(1))))
                correction = final-pre
                current["flow_correction_mean_abs"] = correction.abs().flatten(1).mean(1)
                current["flow_correction_max_abs"] = correction.abs().flatten(1).amax(1)
                if all(key in stage for key in ("limited_flow", "base_logits", "gate_maxima", "adapter_offsets")):
                    current.update(bounded_gate_grid(stage, target, model.core.gate_adapter.bound))
                for name, values in current.items():
                    values = to_array(values).reshape(-1).astype(np.float64)
                    if len(values) != len(clean) or not np.isfinite(values).all():
                        raise FloatingPointError(f"Invalid per-window stage diagnostic: {name}")
                    diagnostics.setdefault(name, []).append(values)
                take = min(len(clean), WAVEFORM_LIMIT-retained)
                if take:
                    for key, value in {"clean": clean, "noisy": noisy,
                                       **{key: stage[key] for key in waveform_keys if key in stage}}.items():
                        waveform_chunks.setdefault(key, []).append(to_array(value[:take]).astype(np.float32))
                    retained += take
                seen += len(clean)
    finally:
        model.train(old_training)
        if reference is not None:
            reference.train(ref_training)
    if seen != len(loader.dataset):
        raise ValueError("Validation must include every window.")
    arrays = {key: np.concatenate(value) for key, value in windows.items()}
    clean = arrays.pop("clean")
    metrics = {key: metric_arrays(clean, value) for key, value in arrays.items()}
    diagnostics = {key: np.concatenate(value) for key, value in diagnostics.items()}
    mean = lambda mapping: {key: float(np.asarray(value, np.float64).mean()) for key, value in mapping.items()}
    original_indices = getattr(loader.dataset, "original_train_indices", None)
    if original_indices is not None and len(original_indices) != seen:
        raise ValueError("Validation original-index mapping has the wrong size.")
    rows = []
    for index in range(seen):
        row = {"validation_index": index}
        if original_indices is not None:
            row["original_train_index"] = int(original_indices[index])
        for stage_name, stage_metrics in metrics.items():
            for name in METRICS:
                row[f"{name}_{stage_name}"] = float(stage_metrics[name][index])
        for name in METRICS:
            delta = float(metrics["final"][name][index] - metrics["pre_flow"][name][index])
            row[f"{name}_delta"] = delta
            row[f"{name}_benefit"] = int(delta < 0 if name != "cossim" else delta > 0)
        row.update({key: float(values[index]) for key, values in diagnostics.items()})
        rows.append(row)
    summary = {key: {"mean": float(value.mean()), "min": float(value.min()),
                     "max": float(value.max()), "count": int(value.size)} for key, value in diagnostics.items()}
    waveforms = {key: np.concatenate(value) for key, value in waveform_chunks.items()}
    waveforms["validation_index"] = np.arange(retained, dtype=np.int64)
    if original_indices is not None:
        waveforms["original_train_index"] = np.asarray(original_indices[:retained], dtype=np.int64)
    return {"metrics": mean(metrics["final"]), "stage1_metrics": mean(metrics["pre_flow"]),
            "all_stage_metrics": {key: mean(value) for key, value in metrics.items()},
            "benefit_rates": {name: float(np.mean([row[name+"_benefit"] for row in rows])) for name in METRICS},
            "count": seen, "clean_global_mean": float(np.mean(clean)), "diagnostics": summary,
            "parity_max_abs": parity_max, "windows": rows, "waveforms": waveforms,
            "oracle_policy": {"validation_labels_only": True, "used_for_selection": False,
                              "shrink_range": [0, 1], "gate_grid_points_per_axis": GATE_GRID_POINTS,
                              "gate_grid_includes_current_prediction": True,
                              "gate_grid_is_sampled_diagnostic_not_global_optimum": True,
                              "budget_bypass_holds_actual_gates_fixed": True}}


def frozen_reference_metrics(reference, loader, device):
    """Use the actual v7 predictions for the protection baseline, never v16's proxy."""
    clean_windows, outputs = [], []
    old_training = reference.training
    try:
        with isolated_evaluation_rng(), torch.no_grad():
            reference.eval()
            for noisy, clean in full_validation_loader(loader):
                output = reference.denoising(noisy.to(device))
                if output.shape != clean.shape or not torch.isfinite(output).all():
                    raise ValueError("Invalid frozen-v7 validation output.")
                clean_windows.append(clean.flatten(1).cpu().numpy().astype(np.float32))
                outputs.append(output.flatten(1).cpu().numpy().astype(np.float32))
    finally:
        reference.train(old_training)
    values = metric_arrays(np.concatenate(clean_windows), np.concatenate(outputs))
    return {key: float(np.asarray(value, dtype=np.float64).mean()) for key, value in values.items()}


def save_validation(checkpoint_dir, epoch, result):
    path = Path(checkpoint_dir) / "validation"
    write_csv(path / f"epoch_{epoch:03d}_windows.csv", result["windows"])
    atomic_json(path / f"epoch_{epoch:03d}_diagnostics.json",
                {key: value for key, value in result.items() if key not in {"windows", "waveforms"}})


def frozen_parameters_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_waveforms(checkpoint_dir, label, epoch, result, identity, checkpoint_path):
    path = Path(checkpoint_dir) / "validation" / f"{label}_waveforms.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **result["waveforms"])
    os.replace(temporary, path)
    atomic_json(path.with_suffix(".json"), {
        "epoch": epoch, "waveform_sha256": sha256(path),
        "checkpoint_sha256": sha256(checkpoint_path), "identity": identity,
        "full_validation_count": result["count"],
        "retained_count": len(result["waveforms"]["validation_index"]),
        "sampling": f"first up to {WAVEFORM_LIMIT} windows in fixed validation order",
        "original_train_indices_available": "original_train_index" in result["waveforms"],
        "purpose": "validation-only diagnostics; never used for training or test selection",
    })


def gradient_norm(parameters):
    squares = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0


def component_gradient_norms(model, groups):
    """One batch per epoch diagnoses objective influence without another forward/RNG draw."""
    rows = {}
    parameters = [parameter for group in groups for parameter in group["params"]]
    for name, component in getattr(model, "loss_component_tensors", {}).items():
        gradients = (torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True)
                     if component.requires_grad else [None] * len(parameters))
        offset = 0
        for group in groups:
            subset = gradients[offset:offset+len(group["params"])]
            offset += len(group["params"])
            squares = [value.detach().double().square().sum() for value in subset if value is not None]
            rows[f"first_batch_grad_{name}_{group['name']}"] = (
                float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0)
    return rows


def set_epoch_sampler(loader, run_seed, epoch):
    """Keep batch order equal across frozen-flow and trainable-flow dropout draws."""
    generator = torch.Generator().manual_seed((int(run_seed) + epoch) % (2**63 - 1))
    loader.generator = generator
    if hasattr(loader.sampler, "generator"):
        loader.sampler.generator = generator


def selection_payload(baseline, selected, history, complete, identity):
    count = history[0]["val_count"] if history else 0
    return {"runner_version": "v16", "policy": SELECTION_POLICY, "baseline": baseline,
            "baseline_metrics": {**{"val_" + key: value for key, value in baseline.items()}, "val_count": count},
            "selected_checkpoint": ({"epoch": selected["epoch"], "metrics": {
                **{"val_" + key: value for key, value in selected["metrics"].items()}, "val_count": count}}
                if selected else None),
            "selected": selected, "checkpoints": {"best_model.pt": selected} if selected else {},
            "epochs_completed": len(history), "complete": complete,
            "status": ("selected" if selected else "no_feasible_checkpoint") if complete else "training",
            "identity": identity}


def save_training_state(path, model, optimizer, epoch, history, baseline, selected, identity, baseline_hash):
    best_path = Path(path).parent / "best_model.pt"
    atomic_torch_save({
        "format_version": 1, "runner_version": "v16", "epoch": epoch,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "history": history, "baseline": baseline, "selected": selected,
        "identity": identity, "policy": SELECTION_POLICY,
        "baseline_sha256": baseline_hash,
        "best_sha256": sha256(best_path) if selected else None,
        "rng_state": capture_rng_state(),
    }, Path(path))


def load_training_state(checkpoint_dir, identity, baseline):
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / "training_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or state.get("runner_version") != "v16" or "rng_state" not in state:
        raise ValueError("Exact v16 resume requires its own training_state.pt with RNG state.")
    if state.get("identity") != identity or state.get("policy") != SELECTION_POLICY:
        raise ValueError("Resume/evaluation must preserve config, dataset, v7 checkpoint, seed, code, and epoch budget.")
    if state.get("baseline") != baseline:
        raise ValueError("Frozen-v7 validation baseline changed.")
    if sha256(checkpoint_dir / "baseline.pt") != state.get("baseline_sha256"):
        raise ValueError("Saved v7 baseline checkpoint changed.")
    if state.get("selected"):
        if sha256(checkpoint_dir / "best_model.pt") != state.get("best_sha256"):
            raise ValueError("Selected checkpoint changed after training.")
        if not feasible(state["selected"]["metrics"], baseline):
            raise ValueError("Saved selected checkpoint violates the four-metric policy.")
    elif (checkpoint_dir / "best_model.pt").exists():
        raise ValueError("Unexpected best_model.pt for a run without a qualified candidate.")
    if len(state["history"]) != state["epoch"] or state["epoch"] > identity["epochs"]:
        raise ValueError("Invalid completed epoch/history in training state.")
    return state


def train_model(model, reference, train_loader, val_loader, settings, checkpoint_dir, identity,
                *, resume=False, writer=None):
    checkpoint_dir = Path(checkpoint_dir)
    if not len(train_loader):
        raise ValueError("The training loader has no complete batches; reduce batch_size.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline = frozen_reference_metrics(reference, val_loader, settings["device"])
    groups = model.configure_finetuning(scope=settings["finetune_scope"],
        lr_adapter=settings["lr_adapter"], lr_flow=settings["lr_flow"])
    frozen_hash = frozen_parameters_digest(model)
    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "radam": torch.optim.RAdam}
    optimizer = optimizers[settings["optimizer"].lower()](groups, betas=tuple(settings["betas"]),
                                                         weight_decay=settings["weight_decay"])
    history, selected, start_epoch = [], None, 0
    if resume:
        state = load_training_state(checkpoint_dir, identity, baseline)
        torch.nn.Module.load_state_dict(model, state["model_state_dict"], strict=True)
        if frozen_parameters_digest(model) != frozen_hash:
            raise ValueError("Resumed model changed frozen v7 parameters.")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history, selected, start_epoch = state["history"], state["selected"], state["epoch"]
        restore_rng_state(state["rng_state"])
    else:
        occupied = [name for name in ("baseline.pt", "training_state.pt", "model_last.pt", "best_model.pt")
                    if (checkpoint_dir / name).exists()]
        if occupied:
            raise FileExistsError("Existing checkpoints require --resume or a new directory: " + ", ".join(occupied))
        initial = validate(model, val_loader, settings["device"], reference)
        save_validation(checkpoint_dir, 0, initial)
        atomic_torch_save(reference.state_dict(), checkpoint_dir / "baseline.pt")
        save_waveforms(checkpoint_dir, "epoch_000", 0, initial, identity, checkpoint_dir / "baseline.pt")
        atomic_json(checkpoint_dir / "initialization_parity.json", {
            "max_abs": initial["parity_max_abs"], "count": initial["count"],
            "rtol": PARITY_RTOL, "atol": PARITY_ATOL, "frozen_v7_metrics": baseline,
            "initialized_v16_metrics": initial["metrics"], "identity": identity})
        del initial
    atomic_json(checkpoint_dir / "trainable_parameters.json", {
        "frozen_parameters_sha256": frozen_hash,
        "parameters": model.finetuning_summary() if hasattr(model, "finetuning_summary") else {},
        "gradient_logging": "total norms before clipping averaged over batches; component norms first batch only",
    })
    baseline_hash = sha256(checkpoint_dir / "baseline.pt")
    for epoch in range(start_epoch + 1, settings["epochs"] + 1):
        set_epoch_sampler(train_loader, identity.get("seed", 3407), epoch)
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch, settings["epochs"])
        model.train()
        totals, count, first_batch_gradients = {}, 0, {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch, settings["device"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            if not count:
                first_batch_gradients = component_gradient_norms(model, optimizer.param_groups)
            loss.backward()
            group_gradients = {"grad_norm_" + group["name"]: gradient_norm(group["params"])
                               for group in optimizer.param_groups}
            if settings["grad_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            batch_count = len(batch[0])
            components = dict(model.loss_components)
            components.update(group_gradients)
            components["train_loss"] = float(loss.detach())
            for name, value in components.items():
                value = float(value)
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite loss component {name}.")
                totals[name] = totals.get(name, 0.0) + value * batch_count
            count += batch_count
        if frozen_parameters_digest(model) != frozen_hash:
            raise RuntimeError("Frozen v7 parameters changed during fine-tuning.")
        result = validate(model, val_loader, settings["device"])
        metrics = result["metrics"]
        row = {"epoch": epoch, "train_count": count, "val_count": result["count"],
               **{key: value / count for key, value in totals.items()},
               **{"val_" + key: value for key, value in metrics.items()},
               **{"val_stage1_" + key: value for key, value in result["stage1_metrics"].items()},
               **{"lr_" + group["name"]: group["lr"] for group in optimizer.param_groups},
               **first_batch_gradients,
               **{"val_benefit_rate_" + key: value for key, value in result["benefit_rates"].items()},
               **{"val_diag_" + key: value["mean"] for key, value in result["diagnostics"].items()},
               "frozen_parameters_sha256": frozen_hash,
               "feasible": feasible(metrics, baseline)}
        history.append(row)
        if row["feasible"] and prefer_candidate(metrics, epoch, selected):
            selected = {"epoch": epoch, "metrics": metrics, "baseline_deltas": {
                key: metrics[key] - baseline[key] for key in METRICS}}
            atomic_torch_save(model.state_dict(), checkpoint_dir / "best_model.pt")
            save_waveforms(checkpoint_dir, "selected", epoch, result, identity, checkpoint_dir / "best_model.pt")
        save_validation(checkpoint_dir, epoch, result)
        atomic_torch_save(model.state_dict(), checkpoint_dir / "model_last.pt")
        if epoch == settings["epochs"]:
            save_waveforms(checkpoint_dir, "final", epoch, result, identity, checkpoint_dir / "model_last.pt")
        save_training_state(checkpoint_dir / "training_state.pt", model, optimizer, epoch, history,
                            baseline, selected, identity, baseline_hash)
        write_csv(checkpoint_dir / "loss_history.csv", history)
        atomic_json(checkpoint_dir / "validation_selection.json",
                    selection_payload(baseline, selected, history, epoch == settings["epochs"], identity))
        if writer is not None:
            for key, value in row.items():
                if key != "epoch" and isinstance(value, (int, float, bool)):
                    writer.add_scalar(key, value, epoch)
        print(f"Epoch {epoch}/{settings['epochs']}: {metrics}; feasible={row['feasible']}", flush=True)
    # Also restore reports after a completed training state whose report write was interrupted.
    write_csv(checkpoint_dir / "loss_history.csv", history)
    payload = selection_payload(baseline, selected, history, True, identity)
    atomic_json(checkpoint_dir / "validation_selection.json", payload)
    return payload


def validated_settings(cfg, device):
    training = OmegaConf.to_container(cfg.training, resolve=True)
    if cfg.model_name != "lstm_v16_ecg":
        raise ValueError("The v16 runner requires model_name=lstm_v16_ecg.")
    if training.get("scheduler", "none") not in (None, "none", "None"):
        raise ValueError("v16 uses fixed learning rates without a scheduler.")
    if training.get("early_stopping_patience_epochs", 0) != 0:
        raise ValueError("v16 must complete its full fine-tuning budget without early stopping.")
    if training.get("selection_metric", "v7_pareto") != "v7_pareto":
        raise ValueError("v16 selection_metric must be v7_pareto for every ablation.")
    settings = {"device": device, "epochs": training.get("train_epochs", 10),
                "batch_size": training.get("batch_size", 64),
                "finetune_scope": training.get("finetune_scope", "adapter"),
                "optimizer": training.get("optimizer", "AdamW"),
                "betas": training.get("betas", [0.8, 0.99]),
                "weight_decay": training.get("weight_decay", 0.0),
                "grad_clip_norm": training.get("grad_clip_norm", 0.5),
                "lr_adapter": training.get("lr_adapter", 1e-4),
                "lr_flow": training.get("lr_flow", 1e-5)}
    for key in ("epochs", "batch_size"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(key + " must be a positive integer.")
    for key in ("lr_adapter", "lr_flow", "weight_decay"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative.")
        if key != "weight_decay" and value == 0:
            raise ValueError(key + " must be positive; choose a finetune_scope to freeze components.")
    if settings["finetune_scope"] not in ("adapter", "adapter_flow"):
        raise ValueError("finetune_scope must be adapter or adapter_flow.")
    if str(settings["optimizer"]).lower() not in ("adamw", "adam", "radam"):
        raise ValueError("Unsupported optimizer.")
    clip = settings["grad_clip_norm"]
    if clip is not None and (isinstance(clip, bool) or not math.isfinite(float(clip)) or clip <= 0):
        raise ValueError("grad_clip_norm must be null or finite and positive.")
    betas = settings["betas"]
    if len(betas) != 2 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x < 1 for x in betas):
        raise ValueError("betas must contain two numbers in [0,1).")
    return settings


def run_identity(cfg, args, settings):
    app = Path(__file__).resolve().parent
    # The original loss, normalization, transforms and decoders live in multiple
    # model modules. Hash their full dependency tree, including registration, so
    # direct-runner resume cannot silently mix implementations.
    source_files = [Path(__file__), app / "utils_ecg.py",
                    app / "mecge_table1_run_official_local_model.py", app / "mecge_table1_run_v14_model.py",
                    *sorted((app / "models").rglob("*.py"))]
    return {"config": OmegaConf.to_container(cfg, resolve=True),
            "dataset_sha256": sha256(args.dataset_pkl), "init_checkpoint_sha256": sha256(args.init_checkpoint),
            "epochs": settings["epochs"], "seed": args.seed, "device": str(args.device),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "sources": {str(path.relative_to(app)): sha256(path) for path in source_files},
            "split": {"validation_ratio": 0.3, "random_state": 1, "full_validation": True},
            "training_sampler_policy": TRAIN_SAMPLER_POLICY,
            "training_rng_isolated_from_model_construction": True}


def main(argv=None):
    from models.lstm_v16 import initialize_from_v7
    args = parse_args(argv)
    cfg = OmegaConf.load(args.config)
    args.device = torch.device(args.device)
    settings = validated_settings(cfg, args.device)
    checkpoint_dir, output = Path(args.checkpoint_dir).resolve(), Path(args.output_pkl).resolve()
    if args.baseline_output_pkl and Path(args.baseline_output_pkl).resolve() == output:
        raise ValueError("Baseline predictions need a separate output path.")
    protected = {checkpoint_dir / name for name in (
        "best_model.pt", "baseline.pt", "model_last.pt", "training_state.pt",
        "validation_selection.json", "loss_history.csv", "initialization_parity.json",
        "warm_start_provenance.json", "trainable_parameters.json")}
    for path in (output, Path(args.baseline_output_pkl).resolve() if args.baseline_output_pkl else None):
        if path is not None and path in protected | {
                Path(args.dataset_pkl).resolve(), Path(args.init_checkpoint).resolve(), Path(args.config).resolve()}:
            raise ValueError("Output must not overwrite inputs or checkpoint/selection artifacts.")
    if not args.resume and not args.skip_train and output.exists():
        raise FileExistsError("Primary result already exists; use a new output directory.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Different model constructors (e.g. adding a local gate) must not shift the
    # subsequent shuffle/dropout streams between ablation conditions.
    with isolated_evaluation_rng(args.seed):
        parameters = OmegaConf.to_container(cfg.model, resolve=True)
        model = get_model(cfg.model_name, **parameters).to(args.device)
        provenance = initialize_from_v7(model, args.init_checkpoint)
        reference = get_model("lstm_dualpath_dapp_cfm_unet_bd_ecg", **parameters).to(args.device)
        raw = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        raw = raw.get("model_state_dict", raw.get("state_dict", raw))
        torch.nn.Module.load_state_dict(reference, raw, strict=True)
        reference.requires_grad_(False).eval()
    identity = run_identity(cfg, args, settings)
    train_loader, val_loader, test_loader, noisy, clean = build_loaders(args.dataset_pkl, settings["batch_size"])
    # The official loader applies this exact split to all original training rows.
    # Save indices from that same algorithm, rather than inventing record IDs.
    from sklearn.model_selection import train_test_split
    _, validation_indices = train_test_split(
        np.arange(len(train_loader.dataset) + len(val_loader.dataset)),
        test_size=0.3, shuffle=True, random_state=1)
    val_loader.dataset.original_train_indices = validation_indices
    if args.skip_train:
        baseline = frozen_reference_metrics(reference, val_loader, args.device)
        state = load_training_state(checkpoint_dir, identity, baseline)
        if state["epoch"] != settings["epochs"]:
            raise ValueError("Evaluation requires the completed original fine-tuning budget; use --resume.")
        payload = selection_payload(baseline, state["selected"], state["history"], True, identity)
    else:
        writer = SummaryWriter(args.log_dir)
        try:
            payload = train_model(model, reference, train_loader, val_loader, settings, checkpoint_dir,
                                  identity, resume=args.resume, writer=writer)
        finally:
            writer.close()
        atomic_json(checkpoint_dir / "warm_start_provenance.json", provenance)
    if payload["selected"] is not None:
        selected_weights = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=True)
        torch.nn.Module.load_state_dict(model, selected_weights, strict=True)
        metrics = validate(model, val_loader, args.device)["metrics"]
        if metrics != payload["selected"]["metrics"] or not feasible(metrics, payload["baseline"]):
            raise ValueError("Selected checkpoint fails reproducibility/validation guard; test export stopped.")
        test_model(model, test_loader, checkpoint_dir, noisy, clean, output, args.device)
    elif output.exists():
        raise ValueError("No feasible checkpoint, but a stale primary prediction PKL exists.")
    else:
        print("No updated checkpoint passes all four validation constraints; no v16 test PKL exported.")
    if args.baseline_output_pkl:
        test_model(reference, test_loader, checkpoint_dir, noisy, clean,
                   args.baseline_output_pkl, args.device, eval_checkpoint=checkpoint_dir / "baseline.pt")
    return payload


if __name__ == "__main__":
    main()
