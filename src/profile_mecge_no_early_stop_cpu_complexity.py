import argparse
import math
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from models import get_model
from utils import profile_model_complexity


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile MECG-E no-early-stop complexity and use the same MECG-E "
            "architecture with CPU reference kernels for the CPU latency section."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--model-key", default="mecge_no_early_stop")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def normalize_yaml_values(values):
    normalized = {}
    for key, value in values.items():
        if isinstance(value, float) and math.isnan(value):
            normalized[key] = ".nan"
        else:
            normalized[key] = value
    return normalized


def device_label(device):
    return "gpu" if device.type == "cuda" else "cpu"


def enable_mamba_cpu_reference_kernels():
    """Switch Mamba CPU execution from fused CUDA kernels to reference PyTorch.

    This does not replace MECG-E blocks or change the model graph. It keeps
    MECG-E -> TSMambaBlock -> MambaBlock -> Mamba, but routes Mamba's scan and
    norm calls through CPU-capable reference functions for latency profiling.
    """
    from mamba_ssm.modules import mamba_simple
    from mamba_ssm.ops import selective_scan_interface
    from mamba_ssm.ops.triton import layer_norm

    mamba_simple.selective_scan_fn = selective_scan_interface.selective_scan_ref
    mamba_simple.causal_conv1d_fn = None
    mamba_simple.causal_conv1d_update = None
    mamba_simple.mamba_inner_fn = None

    def rms_norm_cpu_ref(
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        return_dropout_mask=False,
        **kwargs,
    ):
        if residual is not None and residual_in_fp32:
            residual = residual.float()
        return layer_norm.rms_norm_ref(
            x,
            weight,
            bias,
            residual=residual,
            x1=x1,
            weight1=weight1,
            bias1=bias1,
            eps=eps,
            dropout_p=dropout_p,
            rowscale=rowscale,
            prenorm=prenorm,
        )

    def layer_norm_cpu_ref(
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
        return_dropout_mask=False,
        **kwargs,
    ):
        if is_rms_norm:
            return rms_norm_cpu_ref(
                x,
                weight,
                bias,
                residual=residual,
                x1=x1,
                weight1=weight1,
                bias1=bias1,
                eps=eps,
                dropout_p=dropout_p,
                rowscale=rowscale,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
                return_dropout_mask=return_dropout_mask,
            )
        if residual is not None and residual_in_fp32:
            residual = residual.float()
        return layer_norm.layer_norm_ref(
            x,
            weight,
            bias,
            residual=residual,
            x1=x1,
            weight1=weight1,
            bias1=bias1,
            eps=eps,
            dropout_p=dropout_p,
            rowscale=rowscale,
            prenorm=prenorm,
        )

    layer_norm.rms_norm_fn = rms_norm_cpu_ref
    layer_norm.layer_norm_fn = layer_norm_cpu_ref


def force_mamba_reference_path(model):
    for module in model.modules():
        if module.__class__.__name__ == "Mamba" and hasattr(module, "use_fast_path"):
            module.use_fast_path = False


def build_original_mecge(cfg, device):
    model = get_model(cfg.model_name, **OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    return model


def profile_model(cfg, args, device, input_length, *, cpu_reference_kernels=False):
    try:
        if cpu_reference_kernels:
            enable_mamba_cpu_reference_kernels()
        model = build_original_mecge(cfg, device)
        if cpu_reference_kernels:
            force_mamba_reference_path(model)
        complexity = profile_model_complexity(
            model,
            device,
            input_length=input_length,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        return {
            "status": "ok",
            "device": str(device),
            "timing_variant": (
                "original_mecg_e_cpu_reference_kernels"
                if cpu_reference_kernels
                else "original_mecg_e"
            ),
            "architecture": "unchanged_mecg_e_tsmamba_mambablock_mamba",
            **normalize_yaml_values(complexity),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "device": str(device),
            "timing_variant": (
                "original_mecg_e_cpu_reference_kernels"
                if cpu_reference_kernels
                else "original_mecg_e"
            ),
            "architecture": "unchanged_mecg_e_tsmamba_mambablock_mamba",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main():
    args = parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    primary_device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    input_length = args.input_length or int(cfg.dataset.get("window_size", 512))

    output = {
        "model_key": args.model_key,
        "model_name": str(cfg.model_name),
        "config": str(args.config),
        "overrides": list(args.overrides),
        "input_length": int(input_length),
        "batch_size": int(args.batch_size),
        "mecge_no_early_stop_cpu_time_variant": "same_architecture_reference_kernels",
    }

    if primary_device.type == "cpu":
        output["cpu"] = profile_model(
            cfg,
            args,
            torch.device("cpu"),
            input_length,
            cpu_reference_kernels=True,
        )
    else:
        output[device_label(primary_device)] = profile_model(
            cfg,
            args,
            primary_device,
            input_length,
            cpu_reference_kernels=False,
        )
        if args.include_cpu:
            output["cpu"] = profile_model(
                cfg,
                args,
                torch.device("cpu"),
                input_length,
                cpu_reference_kernels=True,
            )

    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
