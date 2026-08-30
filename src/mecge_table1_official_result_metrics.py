import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import yaml

from utils_ecg import cosine_similarity, maximum_absolute_distance, nanmean_or_nan, prd_mecge_official, ssd


CORE_METRICS = ("SSD", "MAD", "PRD", "CosSim")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize an official MECG-E result pkl.")
    parser.add_argument("--result-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def as_nt(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    elif array.ndim == 3 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    if array.ndim != 2:
        raise ValueError(f"Expected [N, T], [N, T, 1], or [N, 1, T], got {array.shape}.")
    return array


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    return {
        "mean": nanmean_or_nan(values),
        "std": float(np.std(valid)) if valid.size else float("nan"),
        "count": int(valid.size),
    }


def main():
    args = parse_args()
    with open(args.result_pkl, "rb") as handle:
        result = pickle.load(handle)
    if len(result) < 3:
        raise ValueError(f"{args.result_pkl} must contain [X_test, y_test, y_pred].")

    clean = as_nt(result[1])
    pred = as_nt(result[2])
    if clean.shape != pred.shape:
        raise ValueError(f"y_test and y_pred shape mismatch: {clean.shape} vs {pred.shape}.")

    metric_values = {
        "SSD": ssd(clean, pred),
        "MAD": maximum_absolute_distance(clean, pred),
        "PRD": prd_mecge_official(clean, pred),
        "CosSim": cosine_similarity(clean, pred),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {metric: summarize(values) for metric, values in metric_values.items()}
    with open(output_dir / "metrics_qtdb_pkl_test.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    with open(output_dir / "metrics_per_window.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["window_index", *CORE_METRICS])
        writer.writeheader()
        for idx in range(clean.shape[0]):
            writer.writerow(
                {
                    "window_index": idx,
                    **{metric: float(metric_values[metric][idx]) for metric in CORE_METRICS},
                }
            )

    with open(output_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "display_name", "direction", "mean", "std", "count"])
        writer.writeheader()
        for metric in CORE_METRICS:
            stats = summary[metric]
            writer.writerow(
                {
                    "metric": metric,
                    "display_name": metric,
                    "direction": "higher" if metric == "CosSim" else "lower",
                    **stats,
                }
            )

    print(f"saved official metric summaries to {output_dir}")


if __name__ == "__main__":
    main()
