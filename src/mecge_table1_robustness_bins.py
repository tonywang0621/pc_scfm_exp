import argparse
import csv
from pathlib import Path

import numpy as np


CORE_METRICS = ("SSD", "MAD", "PRD", "CosSim")
BIN_EDGES = (0.2, 0.6, 1.0, 1.5, 2.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize MECG-E/DeepFilter robustness bins using rnd_test.npy and per-window metrics."
    )
    parser.add_argument("--metrics-per-window", required=True)
    parser.add_argument("--rnd-test", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--result-model", required=True)
    parser.add_argument("--noise-version", required=True)
    parser.add_argument("--seed", required=True)
    return parser.parse_args()


def label(value):
    return str(value).replace(".", "p").replace("-", "m")


def read_metric_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "count": int(arr.size),
    }


def write_summary(path, metric_stats, condition_value):
    rows = []
    for metric in CORE_METRICS:
        stats = metric_stats[metric]
        rows.append(
            {
                "metric": metric,
                "display_name": metric,
                "direction": "higher" if metric == "CosSim" else "lower",
                "mean": stats["mean"],
                "std": stats["std"],
                "count": stats["count"],
                "condition_name": "alpha_bin",
                "condition_value": condition_value,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "display_name", "direction", "mean", "std", "count", "condition_name", "condition_value"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    metric_rows = read_metric_rows(args.metrics_per_window)
    rnd_test = np.asarray(np.load(args.rnd_test), dtype=np.float64)
    if len(metric_rows) != len(rnd_test):
        raise ValueError(f"metrics rows ({len(metric_rows)}) != rnd_test length ({len(rnd_test)}).")

    output_root = Path(args.output_root)
    all_rows = []
    for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        if high == BIN_EDGES[-1]:
            mask = (rnd_test >= low) & (rnd_test <= high)
        else:
            mask = (rnd_test >= low) & (rnd_test < high)
        indices = np.where(mask)[0]
        condition_value = f"{low}-{high}"
        metric_stats = {}
        for metric in CORE_METRICS:
            metric_stats[metric] = summarize(float(metric_rows[idx][metric]) for idx in indices if metric_rows[idx].get(metric) not in {"", None})
        result_name = (
            f"{args.result_model}__qtdb_robustness_alpha_{label(low)}_{label(high)}"
            f"__{args.noise_version}__seed{args.seed}"
        )
        out_dir = output_root / result_name
        write_summary(out_dir / "metrics_summary.csv", metric_stats, condition_value)
        for metric, stats in metric_stats.items():
            all_rows.append(
                {
                    "result_name": result_name,
                    "alpha_bin": condition_value,
                    "metric": metric,
                    **stats,
                }
            )
        print(f"saved {out_dir / 'metrics_summary.csv'}")

    summary_path = output_root / "robustness_bins_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["result_name", "alpha_bin", "metric", "mean", "std", "count"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
