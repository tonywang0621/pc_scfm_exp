import argparse
import csv
import math
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from inference import load_model_checkpoint
from models import get_model
from utils import profile_model_complexity


def parse_args():
    parser = argparse.ArgumentParser(description="Profile model complexity for a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def write_csv(path, values):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in values.items():
            writer.writerow({"metric": metric, "value": value})


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
    load_model_checkpoint(model, args.checkpoint, device)

    input_length = args.input_length or int(cfg.dataset.get("window_size", 512))
    complexity = profile_model_complexity(
        model,
        device,
        input_length=input_length,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "complexity_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(normalize_yaml_values(complexity), f, sort_keys=False)
    write_csv(output_dir / "complexity_summary.csv", complexity)
    print(f"saved complexity summary to: {output_dir}")


if __name__ == "__main__":
    main()
