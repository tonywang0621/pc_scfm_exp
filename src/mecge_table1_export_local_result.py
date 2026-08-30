import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export a local model prediction as an official MECG-E result pkl.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pkl-file", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def as_n1t(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, None, :]
    elif array.ndim == 3 and array.shape[1] != 1 and array.shape[2] == 1:
        array = np.transpose(array, (0, 2, 1))
    if array.ndim != 3 or array.shape[1] != 1:
        raise ValueError(f"Expected [N, T], [N, 1, T], or [N, T, 1], got {array.shape}.")
    return array


def load_test_split(path):
    with open(path, "rb") as handle:
        dataset = pickle.load(handle)
    if isinstance(dataset, dict):
        noisy = dataset.get("X_test")
        clean = dataset.get("y_test")
        if noisy is None or clean is None:
            raise KeyError(f"{path} must contain X_test/y_test.")
    else:
        if len(dataset) < 4:
            raise ValueError(f"{path} must contain [X_train, y_train, X_test, y_test].")
        noisy, clean = dataset[2], dataset[3]
    return np.asarray(noisy, dtype=np.float32), np.asarray(clean, dtype=np.float32)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    model = get_model(cfg.model_name, **OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    noisy_original, clean_original = load_test_split(args.pkl_file)
    noisy_n1t = as_n1t(noisy_original)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(noisy_n1t), args.batch_size):
            batch = torch.from_numpy(noisy_n1t[start : start + args.batch_size]).to(device)
            pred = model(batch).detach().cpu().numpy()
            predictions.append(pred)

    pred_n1t = np.concatenate(predictions, axis=0)
    pred_nt1 = np.transpose(pred_n1t, (0, 2, 1))
    output = Path(args.output_pkl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump([noisy_original, clean_original, pred_nt1.astype(np.float32)], handle)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
