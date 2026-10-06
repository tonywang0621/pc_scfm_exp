"""Descriptive, fail-closed acceptance of explicitly named official ECG results.

This tool evaluates supplied results. It never chooses a model, checkpoint,
seed, or training setting using test metrics. Exit codes: 0 passed, 1 failed,
2 incomplete (including pending experiments), 3 invalid input/comparison.
Only the Python standard library is required.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


METRICS = ("SSD", "MAD", "PRD", "CosSim")
SCOPES = {"combined": ("nv1_nv2",), "per-nv": ("nv1", "nv2"),
          "all": ("nv1", "nv2", "nv1_nv2")}
BINS = ("0.2-0.6", "0.6-1.0", "1.0-1.5", "1.5-2.0")
EXIT_CODES = {"passed": 0, "failed": 1, "incomplete": 2, "invalid": 3}


def canonical_alpha(value):
    value = str(value).strip()
    if "_" in value:
        value = value.replace("p", ".").replace("_", "-")
    if not re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", value):
        raise ValueError("invalid alpha bin")
    low, high = map(float, value.split("-"))
    for official in BINS:
        if (low, high) == tuple(map(float, official.split("-"))):
            return official
    raise ValueError("alpha bin is not one of the four official bins")


def assess(candidate_csvs, baseline_csvs, *, candidate_model, baseline_model,
           scope="combined", candidate_robustness_csvs=(), baseline_robustness_csvs=(),
           require_robustness=False):
    """Return a JSON-serializable report; compare every supplied matched seed.

    By default, assess only the combined NV1+NV2 results. A pass applies only
    to the requested scopes, official means, and supplied
    experiments. It is not a statistical significance or generalization claim.
    Robustness uses the existing official open bins, including their omissions.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope: {scope}")
    if not candidate_model.strip() or not baseline_model.strip():
        raise ValueError("Both model names must be explicit and nonempty")
    issues, comparisons, sources = [], [], []
    required_scopes = SCOPES[scope]

    def issue(severity, code, message, **context):
        issues.append({"severity": severity, "code": code, "message": message, **context})

    def read(paths, model, side, kind):
        indexed, seen_paths = {}, set()
        if not paths:
            issue("incomplete", "missing_files", f"No {side} {kind} CSVs supplied.")
        for filename in paths:
            path = Path(filename).resolve()
            if path in seen_paths:
                issue("invalid", "duplicate_file", f"CSV supplied more than once: {path}")
                continue
            seen_paths.add(path)
            try:
                contents = path.read_bytes()
                source = {"path": str(path), "side": side, "kind": kind,
                          "sha256": hashlib.sha256(contents).hexdigest()}
                sources.append(source)
                reader = csv.DictReader(contents.decode("utf-8-sig").splitlines())
                required = {"model", "dataset_protocol", "noise_version", "seed"}
                required.update(f"{metric}_{field}" for metric in METRICS for field in ("mean", "count"))
                if not required.issubset(reader.fieldnames or ()):
                    issue("invalid", "missing_columns", f"Missing columns in {path}.",
                          columns=sorted(required - set(reader.fieldnames or ())))
                    continue
                for line, raw in enumerate(reader, 2):
                    if raw.get("model") != model:
                        continue
                    context = {"side": side, "kind": kind, "path": str(path), "line": line}
                    noise = (raw.get("noise_version") or "").strip()
                    if noise not in SCOPES["all"]:
                        issue("invalid", "invalid_noise_version", f"Unknown noise version: {noise}", **context)
                        continue
                    if noise not in required_scopes:
                        continue
                    try:
                        seed_text = (raw.get("seed") or "").strip()
                        if not re.fullmatch(r"(?:seed)?\d+", seed_text):
                            raise ValueError("seed must be an integer or seed<integer>")
                        seed = int(seed_text.removeprefix("seed"))
                        if seed >= 2**32:
                            raise ValueError("seed must be in [0, 2**32)")
                        protocol = (raw.get("dataset_protocol") or "").strip()
                        if kind == "table1":
                            alpha = ""
                            if protocol != "qtdb_train_qtdb_test" or (raw.get("alpha") or "").strip():
                                raise ValueError("table1 requires qtdb_train_qtdb_test and no alpha bin")
                        else:
                            prefix = "qtdb_robustness_alpha_"
                            if not protocol.startswith(prefix):
                                raise ValueError("robustness protocol must identify its alpha bin")
                            alpha = canonical_alpha(raw.get("alpha") or protocol[len(prefix):])
                            if alpha not in BINS or canonical_alpha(protocol[len(prefix):]) != alpha:
                                raise ValueError("alpha and protocol must identify the same official bin")
                            protocol = prefix + alpha
                        values, counts = {}, {}
                        for metric in METRICS:
                            values[metric] = float(raw[f"{metric}_mean"])
                            if not math.isfinite(values[metric]):
                                raise ValueError(f"{metric} mean is not finite")
                            if metric == "CosSim":
                                if not -1.000001 <= values[metric] <= 1.000001:
                                    raise ValueError("CosSim mean is outside [-1, 1]")
                            elif values[metric] < 0:
                                raise ValueError(f"{metric} mean is negative")
                            count = raw[f"{metric}_count"]
                            if not re.fullmatch(r"\d+", count or "") or int(count) <= 0:
                                raise ValueError(f"{metric} count must be a positive integer")
                            counts[metric] = int(count)
                            std = raw.get(f"{metric}_std")
                            if std not in (None, "") and (not math.isfinite(float(std)) or float(std) < 0):
                                raise ValueError(f"{metric} std is invalid")
                        if len(set(counts.values())) != 1:
                            raise ValueError("metric counts differ within the row")
                        key = (noise, seed, alpha)
                        if key in indexed:
                            issue("invalid", "duplicate_row", "Duplicate model/noise/seed/bin row.",
                                  noise_version=noise, seed=seed, alpha=alpha, **context)
                            continue
                        indexed[key] = {"means": values, "counts": counts, "protocol": protocol,
                                        "source": {"path": str(path), "line": line}}
                    except (ValueError, TypeError, KeyError) as error:
                        issue("invalid", "invalid_row", str(error), **context)
            except FileNotFoundError:
                issue("incomplete", "missing_file", f"Experiment CSV does not exist: {path}")
            except (OSError, UnicodeError, csv.Error) as error:
                issue("invalid", "unreadable_file", f"Cannot read {path}: {error}")
        if not indexed:
            issue("incomplete", "no_model_results", f"No usable {side} {kind} results for {model} in requested scopes.")
        return indexed

    candidate = read(candidate_csvs, candidate_model, "candidate", "table1")
    baseline = read(baseline_csvs, baseline_model, "baseline", "table1")
    candidate_seeds = {key[1] for key in candidate}
    baseline_seeds = {key[1] for key in baseline}
    seeds = sorted(candidate_seeds | baseline_seeds)
    if candidate_seeds and baseline_seeds and candidate_seeds != baseline_seeds:
        issue("invalid", "seed_mismatch", "Every supplied seed must have a matched baseline and candidate.",
              candidate_seeds=sorted(candidate_seeds), baseline_seeds=sorted(baseline_seeds))

    def compare(left, right, kind, alphas):
        for noise in required_scopes:
            for seed in seeds:
                for alpha in alphas:
                    key = (noise, seed, alpha)
                    context = {"kind": kind, "noise_version": noise, "seed": seed, "alpha": alpha}
                    if key not in left or key not in right:
                        issue("incomplete", "missing_pair", "Required matched result is missing.",
                              missing=[name for name, rows in (("candidate", left), ("baseline", right)) if key not in rows],
                              **context)
                        continue
                    c, b = left[key], right[key]
                    if c["protocol"] != b["protocol"] or c["counts"] != b["counts"]:
                        issue("invalid", "comparison_mismatch", "Matched rows must share protocol and all counts.",
                              candidate_counts=c["counts"], baseline_counts=b["counts"], **context)
                        continue
                    outcomes = {}
                    for metric in METRICS:
                        c_value, b_value = c["means"][metric], b["means"][metric]
                        passed = c_value > b_value if metric == "CosSim" else (
                            c_value < b_value if metric == "MAD" else c_value <= b_value)
                        outcomes[metric] = {"candidate": c_value, "baseline": b_value,
                                            "delta": c_value - b_value, "passed": passed,
                                            "rule": ">" if metric == "CosSim" else "<" if metric == "MAD" else "<="}
                    comparisons.append({**context, "protocol": c["protocol"], "counts": c["counts"],
                                        "candidate_source": c["source"], "baseline_source": b["source"],
                                        "metrics": outcomes, "passed": all(m["passed"] for m in outcomes.values())})

    compare(candidate, baseline, "table1", ("",))
    if scope == "all":
        for side, rows in (("candidate", candidate), ("baseline", baseline)):
            for seed in seeds:
                keys = [(nv, seed, "") for nv in SCOPES["all"]]
                if all(key in rows for key in keys):
                    if rows[keys[2]]["counts"]["SSD"] != sum(rows[key]["counts"]["SSD"] for key in keys[:2]):
                        issue("invalid", "combined_count_mismatch", "Combined count must equal NV1 plus NV2 counts.", side=side, seed=seed)

    if require_robustness:
        c_rob = read(candidate_robustness_csvs, candidate_model, "candidate", "robustness")
        b_rob = read(baseline_robustness_csvs, baseline_model, "baseline", "robustness")
        for side, rows, table in (("candidate", c_rob, candidate), ("baseline", b_rob, baseline)):
            rob_seeds = {key[1] for key in rows}
            if rob_seeds - set(seeds):
                issue("invalid", "robustness_seed_mismatch", "Robustness includes seeds absent from table1.",
                      side=side, seeds=sorted(rob_seeds - set(seeds)))
            for noise in required_scopes:
                for seed in seeds:
                    key = (noise, seed, "")
                    if key in table:
                        count = sum(row["counts"]["SSD"] for (nv, sd, _), row in rows.items() if nv == noise and sd == seed)
                        if count > table[key]["counts"]["SSD"]:
                            issue("invalid", "robustness_count_mismatch", "Robustness counts exceed table1 count.",
                                  side=side, noise_version=noise, seed=seed)
        compare(c_rob, b_rob, "robustness", BINS)
    elif candidate_robustness_csvs or baseline_robustness_csvs:
        issue("invalid", "unused_robustness_files", "Pass --require-robustness to evaluate supplied robustness CSVs.")

    if any(item["severity"] == "invalid" for item in issues):
        status = "invalid"
    elif any(item["severity"] == "incomplete" for item in issues) or not comparisons:
        status = "incomplete"
    elif any(not pair["passed"] for pair in comparisons):
        status = "failed"
    else:
        status = "passed"
    return {
        "schema_version": 1, "status": status, "exit_code": EXIT_CODES[status],
        "candidate_model": candidate_model, "baseline_model": baseline_model,
        "scope": scope, "required_noise_versions": list(required_scopes), "seeds": seeds,
        "require_robustness": require_robustness,
        "rules": {"SSD": "candidate <= baseline", "MAD": "candidate < baseline",
                  "PRD": "candidate <= baseline", "CosSim": "candidate > baseline"},
        "comparison_count": len(comparisons), "passed_comparison_count": sum(p["passed"] for p in comparisons),
        "comparisons": comparisons, "issues": issues, "sources": sources,
        "interpretation": "Descriptive comparison of supplied means for every matched seed; not a statistical significance claim.",
        "limits": ["CSV labels/counts cannot establish identical input windows or checkpoint provenance; verify run manifests and paired predictions.",
                   "Official robustness uses open bins and may exclude boundary windows; a pass does not cover those omitted windows.",
                   "No model, checkpoint, seed, or hyperparameter is selected by this checker.",
                   "A pass concerns reported test means only, not a guarantee for unseen data or individual windows."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--baseline-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--scope", choices=tuple(SCOPES), default="combined",
                        help="Results to assess: combined NV1+NV2 (default), each NV separately, or all three.")
    parser.add_argument("--candidate-robustness-csv", nargs="+", default=[], type=Path)
    parser.add_argument("--baseline-robustness-csv", nargs="+", default=[], type=Path)
    parser.add_argument("--require-robustness", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("v13_acceptance.json"))
    args = parser.parse_args(argv)
    inputs = args.candidate_csv + args.baseline_csv + args.candidate_robustness_csv + args.baseline_robustness_csv
    if args.output_json.resolve() in {path.resolve() for path in inputs}:
        parser.error("--output-json must not overwrite an input CSV")
    report = assess(args.candidate_csv, args.baseline_csv,
                    candidate_model=args.candidate_model, baseline_model=args.baseline_model,
                    scope=args.scope, candidate_robustness_csvs=args.candidate_robustness_csv,
                    baseline_robustness_csvs=args.baseline_robustness_csv,
                    require_robustness=args.require_robustness)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output_json.parent,
                                     prefix=args.output_json.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, args.output_json)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{report['status'].upper()}: {report['passed_comparison_count']}/{report['comparison_count']} "
          f"matched comparisons passed; scope={args.scope}; seeds={report['seeds']}.")
    for pair in report["comparisons"]:
        failures = [metric for metric, outcome in pair["metrics"].items() if not outcome["passed"]]
        if failures:
            print(f"  FAILED {pair['kind']} {pair['noise_version']} seed{pair['seed']} {pair['alpha']}: {', '.join(failures)}")
    for item in report["issues"]:
        print(f"  {item['severity'].upper()} {item['code']}: {item['message']}")
    print(f"Descriptive comparison only. JSON: {args.output_json.resolve()}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
