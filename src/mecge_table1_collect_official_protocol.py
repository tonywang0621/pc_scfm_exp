import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from utils_ecg import cosine_similarity, maximum_absolute_distance, nanmean_or_nan, prd_mecge_official, ssd


CORE_METRICS = ("SSD", "MAD", "PRD", "CosSim")
BIN_EDGES = (0.2, 0.6, 1.0, 1.5, 2.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect MECG-E official nv1+nv2 protocol metrics.")
    parser.add_argument("--official-results-dir", required=True)
    parser.add_argument("--rnd-test", required=True)
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


def parse_name(path):
    stem = path.stem
    parts = stem.split("__")
    if len(parts) < 4:
        return None
    parsed = {
        "model": parts[0],
        "dataset_protocol": parts[1],
        "noise_version": parts[2],
        "seed": parts[3],
    }
    if parsed["dataset_protocol"] != "qtdb_train_qtdb_test":
        return None
    return parsed


def load_result(path):
    with open(path, "rb") as handle:
        result = pickle.load(handle)
    if len(result) < 3:
        raise ValueError(f"{path} must contain [X_test, y_test, y_pred].")
    return as_nt(result[1]), as_nt(result[2])


def metric_values(clean, pred):
    return {
        "SSD": ssd(clean, pred),
        "MAD": maximum_absolute_distance(clean, pred),
        "PRD": prd_mecge_official(clean, pred),
        "CosSim": cosine_similarity(clean, pred),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    return {
        "mean": nanmean_or_nan(values),
        "std": float(np.std(valid)) if valid.size else float("nan"),
        "count": int(valid.size),
    }


def metric_fields(row, metrics):
    for metric in CORE_METRICS:
        stats = summarize(metrics[metric])
        row[f"{metric}_mean"] = stats["mean"]
        row[f"{metric}_std"] = stats["std"]
        row[f"{metric}_count"] = stats["count"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "model", "dataset_protocol", "noise_version", "seed", "alpha", "result_name"]
    for metric in CORE_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_count"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def main():
    args = parse_args()
    results_dir = Path(args.official_results_dir)
    output_dir = Path(args.output_dir)
    rnd_test_single = np.asarray(np.load(args.rnd_test), dtype=np.float64)

    grouped = {}
    for path in results_dir.glob("*__qtdb_train_qtdb_test__nv*__seed*.pkl"):
        parsed = parse_name(path)
        if parsed is None:
            continue
        key = (parsed["model"], parsed["seed"])
        grouped.setdefault(key, {})[parsed["noise_version"]] = path

    table_rows = []
    robustness_rows = []
    for (model, seed), by_nv in sorted(grouped.items()):
        if "nv1" not in by_nv or "nv2" not in by_nv:
            continue
        clean_nv1, pred_nv1 = load_result(by_nv["nv1"])
        clean_nv2, pred_nv2 = load_result(by_nv["nv2"])
        clean = np.concatenate([clean_nv1, clean_nv2], axis=0)
        pred = np.concatenate([pred_nv1, pred_nv2], axis=0)
        metrics = metric_values(clean, pred)

        result_name = f"{model}__qtdb_train_qtdb_test__nv1_nv2__{seed}"
        row = {
            "experiment": "table1_official_nv1_nv2",
            "model": model,
            "dataset_protocol": "qtdb_train_qtdb_test",
            "noise_version": "nv1_nv2",
            "seed": seed,
            "alpha": "",
            "result_name": result_name,
        }
        metric_fields(row, metrics)
        table_rows.append(row)

        rnd_test = np.concatenate([rnd_test_single, rnd_test_single], axis=0)
        if len(rnd_test) != len(clean):
            raise ValueError(f"duplicated rnd_test length {len(rnd_test)} != combined result length {len(clean)} for {model}.")
        for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            if high == BIN_EDGES[-1]:
                mask = rnd_test > low
            else:
                mask = (rnd_test > low) & (rnd_test < high)
            alpha = f"{low}-{high}"
            bin_metrics = {metric: values[mask] for metric, values in metrics.items()}
            row = {
                "experiment": "robustness_official_nv1_nv2",
                "model": model,
                "dataset_protocol": f"qtdb_robustness_alpha_{alpha}",
                "noise_version": "nv1_nv2",
                "seed": seed,
                "alpha": alpha,
                "result_name": f"{model}__qtdb_robustness_alpha_{alpha}__nv1_nv2__{seed}",
            }
            metric_fields(row, bin_metrics)
            robustness_rows.append(row)

    write_csv(output_dir / "table1_comparison__official_nv1_nv2.csv", table_rows)
    write_csv(output_dir / "robustness_comparison__official_nv1_nv2.csv", robustness_rows)


if __name__ == "__main__":
    main()
