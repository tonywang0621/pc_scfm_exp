import argparse
import csv
from pathlib import Path

import yaml


CORE_METRICS = ("SSD", "MAD", "PRD", "CosSim")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect MECG-E Table 1 reproduction metrics.")
    parser.add_argument("--run-root", default="../runs/mecge_table1_repro")
    parser.add_argument("--noise-version", default="nv1")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def read_yaml_metrics(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def read_csv_metrics(path):
    metrics = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metric = row.get("metric")
            if metric:
                metrics[metric] = row
    return metrics


def add_metric_fields(row, metrics):
    for metric in CORE_METRICS:
        stats = metrics.get(metric, {})
        row[f"{metric}_mean"] = stats.get("mean", "")
        row[f"{metric}_std"] = stats.get("std", "")
        row[f"{metric}_count"] = stats.get("count", "")


def parse_result_name(name):
    parts = name.split("__")
    parsed = {"model": "", "dataset_protocol": "", "noise_version": "", "seed": ""}
    if len(parts) >= 4:
        parsed["model"] = parts[0]
        parsed["dataset_protocol"] = parts[1]
        parsed["noise_version"] = parts[2]
        parsed["seed"] = parts[3]
    return parsed


def collect_table1(run_root, noise_version):
    rows = []
    nv_pattern = "nv*"
    if noise_version != "all":
        nv_pattern = noise_version
    for metrics_path in run_root.glob(f"*/results/*__qtdb_train_qtdb_test__{nv_pattern}__seed*/**/metrics_qtdb_pkl_test.yaml"):
        result_name = None
        for parent in metrics_path.parents:
            if "__qtdb_train_qtdb_test__" in parent.name:
                result_name = parent.name
                break
        if result_name is None:
            continue
        row = {
            "experiment": "table1",
            "result_name": result_name,
            "metrics_file": str(metrics_path),
            **parse_result_name(result_name),
        }
        add_metric_fields(row, read_yaml_metrics(metrics_path))
        rows.append(row)
    return sorted(rows, key=lambda row: row["result_name"])


def collect_robustness(run_root, noise_version):
    rows = []
    nv_pattern = "nv*"
    if noise_version != "all":
        nv_pattern = noise_version
    for metrics_path in run_root.glob(f"*/controlled_tests/*__qtdb_robustness_alpha_*__{nv_pattern}__seed*/metrics_summary.csv"):
        result_name = metrics_path.parent.name
        parsed = parse_result_name(result_name)
        alpha = parsed["dataset_protocol"].replace("qtdb_robustness_alpha_", "")
        row = {
            "experiment": "exp2_strength",
            "alpha": alpha,
            "result_name": result_name,
            "metrics_file": str(metrics_path),
            **parsed,
        }
        add_metric_fields(row, read_csv_metrics(metrics_path))
        rows.append(row)
    return sorted(rows, key=lambda row: (row["model"], row["seed"], row["alpha"]))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model",
        "dataset_protocol",
        "noise_version",
        "seed",
        "alpha",
        "result_name",
        "metrics_file",
    ]
    for metric in CORE_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_count"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "analysis"
    write_rows(
        output_dir / f"table1_comparison__qtdb_train_qtdb_test__{args.noise_version}.csv",
        collect_table1(run_root, args.noise_version),
    )
    write_rows(
        output_dir / f"robustness_comparison__qtdb__{args.noise_version}.csv",
        collect_robustness(run_root, args.noise_version),
    )


if __name__ == "__main__":
    main()
