import argparse
import math
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from models import get_model
from utils import profile_model_complexity


def parse_args():
    parser = argparse.ArgumentParser(description="Profile model complexity from a config without loading a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--model-key", default=None)
    return parser.parse_args()


def normalize_yaml_values(values):
    normalized = {}
    for key, value in values.items():
        if isinstance(value, float) and math.isnan(value):
            normalized[key] = ".nan"
        else:
            normalized[key] = value
    return normalized


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = get_model(cfg.model_name, **OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    input_length = args.input_length or int(cfg.dataset.get("window_size", 512))

    complexity = profile_model_complexity(
        model,
        device,
        input_length=input_length,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    output = {
        "model_key": args.model_key,
        "model_name": str(cfg.model_name),
        "config": str(args.config),
        "input_length": int(input_length),
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
