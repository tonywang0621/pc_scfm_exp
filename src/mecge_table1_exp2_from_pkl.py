import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MECG-E Table 1 Exp2 robustness NPZ files from an official dataset_bw_nv*.pkl."
    )
    parser.add_argument("--pkl-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--alpha-values", nargs="+", type=float, default=[0.2, 0.6, 1.0, 1.5, 2.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1.0e-10)
    parser.add_argument("--overwrite", action="store_true")
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


def load_mecge_pkl(path):
    with open(path, "rb") as handle:
        dataset = pickle.load(handle)
    if isinstance(dataset, dict):
        noisy = dataset.get("X_test")
        clean = dataset.get("y_test")
        if noisy is None or clean is None:
            raise KeyError(f"{path} must contain X_test/y_test for robustness generation.")
    else:
        if len(dataset) < 4:
            raise ValueError(f"{path} must contain [X_train, y_train, X_test, y_test].")
        noisy, clean = dataset[2], dataset[3]
    noisy = as_n1t(noisy)
    clean = as_n1t(clean)
    if noisy.shape != clean.shape:
        raise ValueError(f"X_test and y_test shape mismatch: {noisy.shape} vs {clean.shape}.")
    return noisy, clean


def alpha_label(alpha):
    return str(alpha).replace(".", "p").replace("-", "m")


def generate_condition(clean, baseline_pool, alpha, rng, eps):
    indices = rng.integers(0, len(baseline_pool), size=len(clean))
    baseline = baseline_pool[indices]
    baseline_ptp = np.ptp(baseline, axis=-1, keepdims=True)
    clean_ptp = np.ptp(clean, axis=-1, keepdims=True)
    normalized = baseline / np.maximum(baseline_ptp, eps)
    noisy = clean + float(alpha) * clean_ptp * normalized
    return noisy.astype(np.float32), clean.astype(np.float32), indices.astype(np.int64)


def main():
    args = parse_args()
    pkl_path = Path(args.pkl_file)
    output_root = Path(args.output_root)
    noisy_test, clean_test = load_mecge_pkl(pkl_path)
    baseline_pool = noisy_test - clean_test

    if not np.all(np.isfinite(clean_test)):
        raise ValueError("clean test set contains NaN/Inf.")
    if not np.all(np.isfinite(baseline_pool)):
        raise ValueError("baseline residual pool contains NaN/Inf.")

    manifest = {
        "source_pkl": str(pkl_path),
        "source_split": "X_test/y_test",
        "baseline_pool": "X_test - y_test, normalized per window before alpha scaling",
        "alpha_values": [float(value) for value in args.alpha_values],
        "seed": int(args.seed),
        "shape": list(clean_test.shape),
    }

    rng = np.random.default_rng(args.seed)
    for alpha in args.alpha_values:
        condition = f"alpha_{alpha_label(alpha)}"
        out_dir = output_root / "exp2_strength" / "qtdb_pkl_test" / condition / "processed"
        out_path = out_dir / "test.npz"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it.")
        out_dir.mkdir(parents=True, exist_ok=True)
        noisy, clean, baseline_indices = generate_condition(clean_test, baseline_pool, alpha, rng, args.eps)
        np.savez(
            out_path,
            noisy_ecg=noisy,
            clean_reference=clean,
            baseline_source_index=baseline_indices,
            alpha=np.full((len(clean),), float(alpha), dtype=np.float32),
        )
        with open(out_dir / "manifest.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump({**manifest, "alpha": float(alpha), "output_npz": str(out_path)}, handle, sort_keys=False)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
