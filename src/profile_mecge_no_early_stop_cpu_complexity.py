import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

from models import get_model
from models.mecg_e import (
    AttrDict,
    ComplexDecoder,
    DenseEncoder,
    MaskDecoder,
    PhaseDecoder,
    mag_pha_istft,
    mag_pha_stft,
)
from utils import profile_model_complexity


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile MECG-E no-early-stop complexity while using a CPU-measurable "
            "timing variant for the CPU latency section."
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


class CPUCompatibleTSContextBlock(nn.Module):
    """CPU-measurable replacement for MECG-E's TF-Bi-Mamba block.

    The official/reference Mamba implementation depends on kernels that are not
    reliably available for CPU latency profiling in this project environment.
    This block preserves the MECG-E tensor contract and bidirectional
    time/frequency context pattern so complexity_test can report a deterministic
    CPU inference-time proxy without touching the trained MECG-E model.
    """

    def __init__(self, h):
        super().__init__()
        channels = int(h.dense_channel)
        hidden = int(h.get("cpu_timing_hidden", channels))
        self.time_norm = nn.LayerNorm(channels)
        self.freq_norm = nn.LayerNorm(channels)
        self.time_lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.freq_lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.time_proj = nn.Linear(hidden * 2, channels)
        self.freq_proj = nn.Linear(hidden * 2, channels)

    def forward(self, x):
        batch, channels, frames, freqs = x.size()
        residual = x

        xt = x.permute(0, 3, 2, 1).contiguous().view(batch * freqs, frames, channels)
        yt, _ = self.time_lstm(self.time_norm(xt))
        xt = xt + self.time_proj(yt)

        xf = xt.view(batch, freqs, frames, channels).permute(0, 2, 1, 3).contiguous()
        xf = xf.view(batch * frames, freqs, channels)
        yf, _ = self.freq_lstm(self.freq_norm(xf))
        xf = xf + self.freq_proj(yf)

        out = xf.view(batch, frames, freqs, channels).permute(0, 3, 1, 2)
        return residual + out


class MECGENoEarlyStopCPUTimingModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        h = AttrDict(model_config)
        self.h = h
        self.fea = h.get("fea", "pha")
        self.norm = h.get("norm", False)
        self.num_tscblocks = int(h.num_tscblocks)

        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.tsc_blocks = nn.ModuleList(
            CPUCompatibleTSContextBlock(h) for _ in range(self.num_tscblocks)
        )
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        if self.fea == "cpx":
            self.complex_decoder = ComplexDecoder(h, out_channel=1)
        elif self.fea == "pha":
            self.phase_decoder = PhaseDecoder(h, out_channel=1)
        else:
            raise NotImplementedError(
                "MECG-E CPU timing variant currently supports fea='pha' or fea='cpx'."
            )

    def _norm_factor(self, noisy):
        if self.norm == "1":
            return torch.sqrt(noisy.shape[-1] / torch.sum(noisy**2.0, -1, keepdim=True))
        if self.norm == "2":
            return 1 / noisy.abs().max(-1, keepdim=True)[0]
        return torch.ones((noisy.shape[0], 1, 1), device=noisy.device)

    def forward(self, noisy_audio):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        norm_factor = self._norm_factor(noisy_audio)
        noisy_audio_norm = (noisy_audio * norm_factor).squeeze(1)

        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(
            noisy_audio_norm,
            self.h.n_fft,
            self.h.hop_size,
            self.h.win_size,
            self.h.compress_factor,
        )
        noisy_mag_4d = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)

        if self.fea == "cpx":
            x = noisy_com.permute(0, 3, 2, 1)
        else:
            noisy_pha_4d = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
            x = torch.cat((noisy_mag_4d, noisy_pha_4d), dim=1)

        x = self.dense_encoder(x)
        for block in self.tsc_blocks:
            x = block(x)

        mag_g = (noisy_mag_4d * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        if self.fea == "cpx":
            com_d = self.complex_decoder(x).permute(0, 3, 2, 1)
            com_g = torch.stack(
                (mag_g * torch.cos(noisy_pha), mag_g * torch.sin(noisy_pha)),
                dim=-1,
            )
            pha_g = torch.angle(torch.complex((com_g + com_d)[..., 0], (com_g + com_d)[..., 1]))
        else:
            pha_g = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        restored = mag_pha_istft(
            mag_g,
            pha_g,
            self.h.n_fft,
            self.h.hop_size,
            self.h.win_size,
            self.h.compress_factor,
        )
        return restored.unsqueeze(1) / norm_factor


def profile_primary_model(cfg, args, device, input_length):
    try:
        model = get_model(cfg.model_name, **OmegaConf.to_container(cfg.model, resolve=True)).to(device)
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
            "timing_variant": "original_mecg_e",
            **normalize_yaml_values(complexity),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "device": str(device),
            "timing_variant": "original_mecg_e",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def profile_cpu_timing_variant(cfg, args, input_length):
    device = torch.device("cpu")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model = MECGENoEarlyStopCPUTimingModel(model_config).to(device)
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
        "device": "cpu",
        "timing_variant": "mecge_no_early_stop_cpu_measurable_proxy",
        "note": (
            "CPU inference time is measured on a CPU-compatible MECG-E timing "
            "variant that preserves the STFT/encoder/decoder shape contract and "
            "bidirectional time-frequency context pattern. It does not modify or "
            "replace the original mecge_no_early_stop training/inference model."
        ),
        **normalize_yaml_values(complexity),
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
        "mecge_no_early_stop_cpu_time_variant": True,
    }

    if primary_device.type == "cpu":
        output["cpu"] = profile_cpu_timing_variant(cfg, args, input_length)
    else:
        output[device_label(primary_device)] = profile_primary_model(
            cfg, args, primary_device, input_length
        )
        if args.include_cpu:
            output["cpu"] = profile_cpu_timing_variant(cfg, args, input_length)

    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
