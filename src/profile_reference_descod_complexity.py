import argparse
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from utils import profile_model_complexity


def parse_args():
    parser = argparse.ArgumentParser(description="Profile reference DeScoD complexity without loading a checkpoint.")
    parser.add_argument("--descod-dir", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--num-shots", type=int, required=True)
    return parser.parse_args()


def normalize_yaml_values(values):
    normalized = {}
    for key, value in values.items():
        if isinstance(value, float) and math.isnan(value):
            normalized[key] = ".nan"
        else:
            normalized[key] = value
    return normalized


def ensure_optional_reference_imports():
    if "torchsummary" not in sys.modules:
        module = types.ModuleType("torchsummary")
        module.summary = lambda *args, **kwargs: None
        sys.modules["torchsummary"] = module
    if "leaf_audio_pytorch" not in sys.modules:
        leaf_module = types.ModuleType("leaf_audio_pytorch")
        leaf_module.frontend = types.ModuleType("leaf_audio_pytorch.frontend")
        sys.modules["leaf_audio_pytorch"] = leaf_module
        sys.modules["leaf_audio_pytorch.frontend"] = leaf_module.frontend


def load_reference_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ReferenceDeScoDWrapper(nn.Module):
    def __init__(self, diffusion, num_shots):
        super().__init__()
        self.diffusion = diffusion
        self.num_shots = max(int(num_shots), 1)

    def forward(self, x):
        samples = [self.diffusion.denoising(x) for _ in range(self.num_shots)]
        return torch.stack(samples, dim=0).mean(dim=0)

    @torch.no_grad()
    def denoising(self, x):
        return self.forward(x)


def main():
    args = parse_args()
    descod_dir = Path(args.descod_dir).resolve()
    config_path = descod_dir / "config" / "base.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    ensure_optional_reference_imports()
    denoising_model = load_reference_module(
        "reference_descod_denoising_model_small",
        descod_dir / "denoising_model_small.py",
    )
    main_model = load_reference_module(
        "reference_descod_main_model",
        descod_dir / "main_model.py",
    )

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    feats = int(config["train"].get("feats", 80))
    base_model = denoising_model.ConditionalModel(feats=feats).to(device)
    diffusion = main_model.DDPM(base_model, config, device, conditional=True).to(device)
    model = ReferenceDeScoDWrapper(diffusion, args.num_shots).to(device)

    complexity = profile_model_complexity(
        model,
        device,
        input_length=args.input_length,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    output = {
        "model_key": args.model_key,
        "model_name": "reference_descod_ecg",
        "reference_dir": str(descod_dir),
        "config": str(config_path),
        "num_shots": int(args.num_shots),
        "input_length": int(args.input_length),
        "batch_size": int(args.batch_size),
        "device": str(device),
        **normalize_yaml_values(complexity),
    }

    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
