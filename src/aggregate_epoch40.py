import argparse
import csv
from pathlib import Path


METRIC_DIRECTIONS = {
    "SSD": "lower",
    "MAD": "lower",
    "PRD": "lower",
    "CosSim": "higher",
    "Output_SNR_dB": "higher",
    "SNR_Improvement_dB": "higher",
    "LF_Reduction_dB": "higher",
    "R_Peak_Timing_Error_ms": "lower",
    "RR_Interval_MAE_ms": "lower",
    "RMSE": "lower",
    "Centered_CosSim": "higher",
    "QRS_Amplitude_Error": "lower",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate epoch40 inference, robustness, and complexity results.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-epoch", default="40")
    parser.add_argument("--checkpoint-step", default="58080")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(raw):
    if raw in (None, ""):
        return ""
    try:
        return float(raw)
    except ValueError:
        return raw


def base_row(args, evaluation_type, dataset, metric, source):
    return {
        "run_group": "epoch40",
        "model_key": args.model_key,
        "checkpoint_epoch": args.checkpoint_epoch,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_path": args.checkpoint_path,
        "evaluation_type": evaluation_type,
        "dataset": dataset,
        "condition_name": "",
        "condition_value": "",
        "metric": metric,
        "mean": "",
        "std": "",
        "count": "",
        "direction": METRIC_DIRECTIONS.get(metric, ""),
        "source": str(source),
    }


def collect_inference(args, run_root):
    rows = []
    prefix = f"{args.model_key}_"
    inference_root = run_root / "inference"
    for summary_path in sorted(inference_root.glob(f"{prefix}*/metrics_summary.csv")):
        dataset = summary_path.parent.name.removeprefix(prefix)
        for summary in read_csv_rows(summary_path):
            metric = summary["metric"]
            row = base_row(args, "inference", dataset, metric, summary_path)
            row.update(
                {
                    "mean": parse_float(summary.get("mean")),
                    "std": parse_float(summary.get("std")),
                    "count": summary.get("count", ""),
                    "direction": summary.get("direction") or METRIC_DIRECTIONS.get(metric, ""),
                }
            )
            rows.append(row)
    return rows


def collect_robustness_strength(args, run_root):
    rows = []
    robustness_root = run_root / "controlled_tests" / args.model_key / "exp2_strength"
    for summary_path in sorted(robustness_root.glob("*/summary.csv")):
        for summary in read_csv_rows(summary_path):
            dataset = summary.get("dataset") or summary_path.parent.name
            condition_name = summary.get("condition_name", "")
            condition_value = summary.get("condition_value", "")
            metrics = sorted(
                key[: -len("_mean")]
                for key in summary
                if key.endswith("_mean") and summary.get(key, "") != ""
            )
            for metric in metrics:
                row = base_row(args, "robustness_strength", dataset, metric, summary_path)
                row.update(
                    {
                        "condition_name": condition_name,
                        "condition_value": condition_value,
                        "mean": parse_float(summary.get(f"{metric}_mean")),
                        "std": parse_float(summary.get(f"{metric}_std")),
                        "count": summary.get(f"{metric}_count", ""),
                    }
                )
                rows.append(row)
    return rows


def collect_complexity(args, run_root):
    rows = []
    summary_path = run_root / "complexity" / args.model_key / "complexity_summary.csv"
    if not summary_path.exists():
        return rows
    for summary in read_csv_rows(summary_path):
        metric = summary["metric"]
        row = base_row(args, "complexity", "complexity", metric, summary_path)
        row["mean"] = parse_float(summary.get("value"))
        rows.append(row)
    return rows


def write_rows(path, rows):
    fieldnames = [
        "run_group",
        "model_key",
        "checkpoint_epoch",
        "checkpoint_step",
        "checkpoint_path",
        "evaluation_type",
        "dataset",
        "condition_name",
        "condition_value",
        "metric",
        "mean",
        "std",
        "count",
        "direction",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    rows = []
    rows.extend(collect_inference(args, run_root))
    rows.extend(collect_robustness_strength(args, run_root))
    rows.extend(collect_complexity(args, run_root))
    write_rows(Path(args.output), rows)
    print(f"saved epoch40 aggregate table -> {args.output}")


if __name__ == "__main__":
    main()
