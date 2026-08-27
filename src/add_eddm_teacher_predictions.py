import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from models import get_model
from inference import as_single_lead, load_model_checkpoint, run_inference


DEFAULT_SPLIT_FILES = (
    "train.npz",
    "val.npz",
    "test.npz",
    "mit_bih.npz",
    "chapman.npz",
    "cpsc.npz",
    "qtdb.npz",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append EDDM teacher predictions to ECG baseline-wander NPZ files."
    )
    parser.add_argument("--config", default="configs/ecg_baseline_wander_eddm.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", default="../data/ecg_baseline_wander/processed")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--teacher-key", default="eddm_restored_ecg")
    parser.add_argument("--split-files", default=",".join(DEFAULT_SPLIT_FILES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_noisy(npz_path):
    arrays = np.load(npz_path)
    noisy_key = "noisy_ecg" if "noisy_ecg" in arrays else "input"
    if noisy_key not in arrays:
        raise KeyError(f"{npz_path} must contain `noisy_ecg` or `input`.")
    return arrays, as_single_lead(arrays[noisy_key])


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = input_dir if args.in_place else Path(args.output_dir or f"{input_dir}_eddm_teacher")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = get_model(cfg.model_name, **OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    load_model_checkpoint(model, args.checkpoint, device)

    split_files = [item.strip() for item in args.split_files.split(",") if item.strip()]
    for split_file in split_files:
        src = input_dir / split_file
        if not src.exists():
            print(f"skip missing: {src}")
            continue

        arrays, noisy = load_noisy(src)
        if args.teacher_key in arrays and not args.overwrite:
            print(f"skip existing {args.teacher_key}: {src}")
            continue

        restored, _ = run_inference(model, noisy, device=device, batch_size=args.batch_size)
        payload = {key: arrays[key] for key in arrays.files}
        payload[args.teacher_key] = restored.astype(np.float32)

        dst = output_dir / split_file
        np.savez(dst, **payload)
        print(f"wrote {args.teacher_key}: {dst} shape={restored.shape}")


if __name__ == "__main__":
    main()
