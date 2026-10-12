"""Plot the specified v7 model immediately before and after its CFM refinement.

This is checkpoint inference, not training. Dataset PKLs follow the existing
official local protocol. Only trusted local experiment PKLs should be loaded.
"""

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from models import get_model
from utils_ecg import cosine_similarity, maximum_absolute_distance, ssd


PROJECT = Path(__file__).resolve().parents[1]
V7 = "lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience"
MODEL_NAME = "lstm_dualpath_dapp_cfm_unet_bd_ecg"
DEFAULT_CONFIG = PROJECT / "src/configs" / f"mecge_table1_repro_{V7}.yaml"
METRICS = ("SSD", "MAD", "PRD", "CosSim")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt / tensor state_dict checkpoint")
    parser.add_argument("--dataset-pkl", type=Path, required=True, help="Official [X_train,y_train,X_test,y_test] or dict with X_test/y_test")
    parser.add_argument("--indices", nargs="+", type=int, default=[0], help="Zero-based test-window indices (default: 0)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda:0")
    parser.add_argument("--nv", choices=("1", "2"), help="Noise-version label; otherwise infer only from the dataset filename")
    parser.add_argument("--seed", type=int, help="Training-seed label supplied by the user; not an inference setting")
    parser.add_argument("--sampling-rate", type=float, help="Actual data Hz; default plots sample indices (no resampling)")
    parser.add_argument("--output-dir", type=Path, help="New directory; default runs/stage_waveform_analysis/v7_<NV>_<seed>_<checkpoint>_<UTC>")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or any(index < 0 for index in args.indices):
        parser.error("batch-size must be positive and indices must be nonnegative")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("seed label must be in [0, 2**32)")
    if args.sampling_rate is not None and (not math.isfinite(args.sampling_rate) or args.sampling_rate <= 0):
        parser.error("sampling-rate must be finite and positive")
    args.indices = list(dict.fromkeys(args.indices))
    return args


def as_nt(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError("Test arrays must be nonempty finite single-lead [N,T], [N,1,T], or [N,T,1].")
    return np.ascontiguousarray(array)


def load_test_data(path):
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        if "X_test" not in data:
            raise ValueError("Dataset dictionary must contain X_test.")
        noisy, clean = data["X_test"], data.get("y_test")
    elif isinstance(data, (list, tuple)) and len(data) >= 4:
        noisy, clean = data[2], data[3]
    else:
        raise ValueError("Use an official dataset PKL, not a [noisy,clean,prediction] result PKL.")
    noisy = as_nt(noisy)
    clean = None if clean is None else as_nt(clean)
    if clean is not None and clean.shape != noisy.shape:
        raise ValueError("X_test and y_test must have identical sample order and shape.")
    return noisy, clean


def load_v7(config_path, checkpoint_path, device):
    config = OmegaConf.load(config_path)
    exp_name = str(config.get("exp_name", ""))
    if config.get("model_name") != MODEL_NAME or not (exp_name == V7 or exp_name.startswith(V7 + "__")):
        raise ValueError(f"This script supports only {V7}; use its config or resolved_config.yaml.")
    settings = OmegaConf.to_container(config.model, resolve=True)
    if (settings.get("attention_replacement") != "residual_dilated_conv"
            or settings.get("cfm_unet_use_attention") is not False):
        raise ValueError("Expected the v7 residual-convolution, attention-free U-Net configuration.")
    if (settings.get("cfm_inference_steps", 1) != 1
            or settings.get("cfm_inference_shots", 1) != 1
            or settings.get("cfm_inference_start_noise_scale", 0.0) != 0.0):
        raise ValueError("This v7 analysis requires deterministic one-step, single-shot, zero-start inference.")
    model = get_model(MODEL_NAME, **settings)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint must contain a complete tensor state_dict.")
    # Use strict base-class loading; never leave a missing flow branch random.
    torch.nn.Module.load_state_dict(model, checkpoint, strict=True)
    return model.to(device).eval(), config


@torch.inference_mode()
def extract_stages(model, noisy):
    """Return original-domain stage outputs and verify the public inference path."""
    was_training = model.training
    try:
        model.eval()
        core = model.core
        factor = core._norm_factor(noisy)
        if not torch.isfinite(factor).all() or (factor <= 0).any():
            raise ValueError("Invalid normalization factor for the selected window(s).")
        normalized = noisy * factor
        coarse, _, baseline, _, direct = core._restore_components(normalized)
        final, _, _, _, _ = core._refine_from_base(normalized, coarse, baseline_hat=baseline)
        stages = {"lstm_pre_flow": coarse / factor, "flow_final": final / factor,
                  "decoder_direct": direct / factor}
        for name, output in stages.items():
            if output.shape != noisy.shape or not torch.isfinite(output).all():
                raise ValueError(f"Invalid {name} output shape or nonfinite samples.")
        reference = model.denoising(noisy)
        torch.testing.assert_close(stages["flow_final"], reference, rtol=1e-5, atol=1e-6)
        max_difference = float((stages["flow_final"] - reference).abs().max().cpu())
        stages["flow_correction"] = stages["flow_final"] - stages["lstm_pre_flow"]
        return {name: value[:, 0].cpu().numpy().copy() for name, value in stages.items()}, max_difference
    finally:
        model.train(was_training)


def stage_metrics(clean, prediction, full_test_clean_mean):
    """Official per-window metrics using one clean mean from the complete test set.

    This is a single dataset's test scope, not the combined NV1+NV2 protocol.
    Using the mean of selected windows would make PRD depend on --indices.
    """
    numerator = ssd(clean, prediction)
    denominator = np.sum((prediction - np.float32(full_test_clean_mean)) ** 2, axis=-1)
    values = {"SSD": numerator, "MAD": maximum_absolute_distance(clean, prediction),
              "PRD": np.sqrt(numerator / (denominator + 1e-10)) * 100.0,
              "CosSim": cosine_similarity(clean, prediction)}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Nonfinite stage metrics; inspect the input and checkpoint.")
    return values


def plot_sample(folder, index, waves, sampling_rate, row):
    # Direct Agg canvases work on servers without a display or pyplot state.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    x = np.arange(len(waves["noisy"]), dtype=np.float64)
    xlabel = "Sample index within window"
    if sampling_rate is not None:
        x /= sampling_rate
        xlabel = "Time within window (s)"
    clean = waves.get("clean")
    colors = {"noisy": "#7b8794", "lstm_pre_flow": "#d97706", "flow_final": "#007f83"}
    labels = {"noisy": "Input: noisy ECG", "lstm_pre_flow": "Stage 1: LSTM + decoder + dual head (pre-flow)",
              "flow_final": "Stage 2: final ECG after flow refinement"}
    fig = Figure(figsize=(12, 8), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True, sharey=True)
    for ax, key in zip(axes, colors):
        if clean is not None:
            ax.plot(x, clean, color="#172b4d", linewidth=1.2, alpha=0.8, label="Clean reference")
        ax.plot(x, waves[key], color=colors[key], linewidth=1.1, alpha=0.9, label=labels[key])
        title = labels[key]
        if clean is not None:
            title += f"   |   CosSim={row['CosSim_' + key]:.5f}, MAD={row['MAD_' + key]:.5f}"
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Amplitude (input units)")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"v7 stage comparison | test window {index} | shared amplitude scale", fontsize=13)
    fig.savefig(folder / "waveform_comparison.png", dpi=160)
    fig.clear()

    fig = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    axes[0].plot(x, waves["flow_correction"], color="#6d4c9a", linewidth=1.1)
    axes[0].axhline(0, color="black", linewidth=0.6, alpha=0.5)
    axes[0].set_title("Actual flow correction = final ECG - pre-flow ECG", loc="left", fontsize=11)
    axes[0].set_ylabel("Correction (input units)")
    for key in ("lstm_pre_flow", "flow_final"):
        y = waves[key] if clean is None else np.abs(waves[key] - clean)
        axes[1].plot(x, y, color=colors[key], linewidth=1.1, label="Pre-flow" if key == "lstm_pre_flow" else "After flow")
    axes[1].set_ylabel("ECG amplitude" if clean is None else "Absolute error vs clean")
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel(xlabel)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle(f"v7 flow effect | test window {index}", fontsize=13)
    fig.savefig(folder / "flow_correction.png", dpi=160)
    fig.clear()


def main(argv=None):
    args = parse_args(argv)
    args.config, args.checkpoint, args.dataset_pkl = (path.resolve() for path in (args.config, args.checkpoint, args.dataset_pkl))
    for path in (args.config, args.checkpoint, args.dataset_pkl):
        if not path.is_file():
            raise FileNotFoundError(f"Input is missing: {path}")
    inferred_nv = re.search(r"(?:^|_)nv([12])(?:_|$)", args.dataset_pkl.stem)
    if args.nv and inferred_nv and args.nv != inferred_nv.group(1):
        raise ValueError("--nv conflicts with the dataset filename; check the dataset/checkpoint pairing.")
    nv = args.nv or (inferred_nv.group(1) if inferred_nv else "unknown")
    created = datetime.now(timezone.utc)
    label = re.sub(r"[^\w.-]+", "_", args.checkpoint.stem)
    output = (args.output_dir if args.output_dir else PROJECT / "runs/stage_waveform_analysis" /
              f"v7_nv{nv}_seed{args.seed if args.seed is not None else 'unknown'}_{label}_{created.strftime('%Y%m%dT%H%M%S%fZ')}").resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; existing results are preserved: {output}")
    noisy, clean = load_test_data(args.dataset_pkl)
    if max(args.indices) >= len(noisy):
        raise ValueError(f"Test indices must be between 0 and {len(noisy) - 1}.")
    full_mean = float(np.mean(clean)) if clean is not None else None
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or your training environment.")
    model, config = load_v7(args.config, args.checkpoint, device)
    import matplotlib

    manifest = {
        "schema_version": 1, "complete": False, "created_utc": created.isoformat(), "model": V7,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_digest(args.checkpoint),
        "config": str(args.config), "config_sha256": file_digest(args.config),
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "dataset": str(args.dataset_pkl), "dataset_sha256": file_digest(args.dataset_pkl),
        "noise_version_label": nv, "noise_version_label_source": "user" if args.nv else "dataset_filename_or_unknown",
        "training_seed_label": args.seed, "training_seed_label_source": "user_or_unknown_not_inferred_from_template",
        "test_window_count": len(noisy), "window_length": noisy.shape[1], "sample_indices": args.indices,
        "clean_available": clean is not None, "sampling_rate_hz": args.sampling_rate,
        "sampling_rate_source": "user" if args.sampling_rate is not None else "unspecified_sample_axis",
        "metric_scope": "selected windows from one complete dataset test split; NOT combined NV1+NV2",
        "prd_clean_mean_scope": "full_dataset_y_test" if clean is not None else None,
        "full_test_clean_mean": full_mean,
        "metric_definitions": {"MAD": "maximum absolute error per window", "CosSim": "uncentered cosine",
                               "PRD": "100*sqrt(SSD/(sum((prediction-full_test_clean_mean)^2)+1e-10))",
                               "delta": "flow_final minus lstm_pre_flow; negative is better except CosSim"},
        "stages": {"lstm_pre_flow": "decoder_direct + residual_refine_scale * dual-head residual (before CFM)",
                   "flow_final": "original _refine_from_base output after projection, budgets and gate blending",
                   "flow_correction": "flow_final - lstm_pre_flow, not raw two-channel flow velocity"},
        "inference_parity": {"reference": "model.denoising", "rtol": 1e-5, "atol": 1e-6, "max_abs_difference": None},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "device": str(device)},
        "source_sha256": {str(p.relative_to(PROJECT)): file_digest(p) for p in
                          [Path(__file__).resolve(), PROJECT / "src/utils_ecg.py", *sorted((PROJECT / "src/models").glob("*.py"))]},
        "command": list(sys.argv) if argv is None else ["analyze_v7_stages.py", *map(str, argv)],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    rows, parity_max = [], 0.0
    try:
        for start in range(0, len(args.indices), args.batch_size):
            indices = args.indices[start:start + args.batch_size]
            batch = torch.from_numpy(noisy[indices, None, :]).to(device)
            stages, difference = extract_stages(model, batch)
            parity_max = max(parity_max, difference)
            predictions = {"noisy": noisy[indices], "lstm_pre_flow": stages["lstm_pre_flow"], "flow_final": stages["flow_final"]}
            metrics = {key: stage_metrics(clean[indices], value, full_mean) for key, value in predictions.items()} if clean is not None else {}
            for offset, index in enumerate(indices):
                waves = {"noisy": noisy[index], **{key: value[offset] for key, value in stages.items()}}
                if clean is not None:
                    waves["clean"] = clean[index]
                row = {"sample_index": index, "flow_correction_mean_abs": float(np.abs(waves["flow_correction"]).mean()),
                       "flow_correction_max_abs": float(np.abs(waves["flow_correction"]).max())}
                for metric in METRICS:
                    for key in predictions:
                        row[f"{metric}_{key}"] = float(metrics[key][metric][offset]) if clean is not None else ""
                    delta = row[f"{metric}_flow_final"] - row[f"{metric}_lstm_pre_flow"] if clean is not None else ""
                    row[f"{metric}_delta"] = delta
                    row[f"{metric}_improved"] = (delta > 0 if metric == "CosSim" else delta < 0) if clean is not None else ""
                folder = output / f"sample_{index:06d}"
                folder.mkdir()
                arrays = {**waves, "sample_index": np.int64(index), "sample_position": np.arange(noisy.shape[1])}
                if args.sampling_rate is not None:
                    arrays["time_seconds"] = np.arange(noisy.shape[1], dtype=np.float64) / args.sampling_rate
                np.savez_compressed(folder / "waveforms.npz", **arrays)
                plot_sample(folder, index, waves, args.sampling_rate, row)
                rows.append(row)
                print(f"Saved test window {index}: {folder}", flush=True)
        with (output / "stage_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["inference_parity"]["max_abs_difference"] = parity_max
        manifest["artifacts_sha256"] = {str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob("*"))
                                      if p.is_file() and p.name != "manifest.json"}
        manifest["complete"] = True
        write_json(output / "manifest.json", manifest)
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "manifest.json", manifest)
        raise
    print(f"Analysis complete: {output}\nFinal-stage inference parity: max absolute difference {parity_max:.6g}")
    return output


if __name__ == "__main__":
    main()
