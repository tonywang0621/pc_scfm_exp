import argparse
import os
import pickle
import sys
from pathlib import Path


def atomic_pickle_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(obj, handle)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the unmodified reference/MECG-E pipeline for one MECG-E noise version."
    )
    parser.add_argument("--mecge-dir", required=True)
    parser.add_argument("--config", default="config/MECGE_phase.yaml")
    parser.add_argument("--n-type", default="bw")
    parser.add_argument("--nv", type=int, choices=[1, 2], required=True)
    parser.add_argument("--dataset-pkl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--max-epochs", type=int, default=50000)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    import random

    import numpy as np
    import torch
    import yaml
    from torch.utils.tensorboard import SummaryWriter

    mecge_dir = Path(args.mecge_dir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = mecge_dir / config_path
    output_pkl = Path(args.output_pkl).resolve()
    dataset_path = Path(args.dataset_pkl).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    resume_checkpoint = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None

    os.chdir(mecge_dir)
    wrapper_dir = Path(__file__).resolve().parent
    sys.path = [
        str(mecge_dir),
        *[
            path
            for path in sys.path
            if path and Path(path).resolve() != wrapper_dir
        ],
    ]
    sys.modules.pop("models", None)

    from pipeline import train_dl, test_dl

    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MECGE_DEVICE"] = args.device
    os.environ["MECGE_MODEL_WEIGHT_PATH"] = str(checkpoint_dir / "best_model.pt")
    os.environ["MECGE_ARTIFACT_DIR"] = str(checkpoint_dir)
    os.environ["MECGE_RESUME"] = "1" if args.resume else "0"
    if resume_checkpoint:
        os.environ["MECGE_RESUME_CHECKPOINT"] = str(resume_checkpoint)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("train", {})
    config["train"]["epochs"] = args.max_epochs
    config["train"]["early_stopping_patience_epochs"] = args.early_stopping_patience
    config_name = config_path.stem

    with open(dataset_path, "rb") as handle:
        dataset = pickle.load(handle)

    if not args.skip_train:
        log_path = str(log_dir)
        writer = SummaryWriter(log_path)
        try:
            train_dl(dataset, config_name, args.n_type, config, args.nv, writer)
        finally:
            writer.close()

    result = test_dl(dataset, config_name, args.n_type, config, args.nv, args.device)
    atomic_pickle_dump(result, output_pkl)
    print(f"saved {output_pkl}")


if __name__ == "__main__":
    main()
