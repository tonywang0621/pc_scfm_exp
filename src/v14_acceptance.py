"""Compare v14 with v7, reporting the combined cosine stretch target separately.

Basic acceptance and exit codes retain v13's matched-seed, protocol, count, and
metric rules. The stretch target is descriptive and never selects checkpoints.
It applies only to combined NV1+NV2 table rows, never individual noise versions
or robustness bins. With scope='per-nv', stretch coverage is incomplete.

The bundled baseline snapshot covers seed 3407, combined NV1+NV2 only. Other
seeds or scopes require explicitly supplied matching baseline experiments.
"""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from v13_acceptance import BINS, EXIT_CODES, METRICS, SCOPES, assess as assess_baseline


BASELINE_MODEL = "lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience"
DEFAULT_BASELINE = Path(__file__).resolve().parent / "configs/v14_baseline_v7.csv"


def assess(candidate_csvs, baseline_csvs, *, candidate_model="lstm_v14", baseline_model=BASELINE_MODEL,
           scope="combined", candidate_robustness_csvs=(), baseline_robustness_csvs=(),
           require_robustness=False, cosine_target=0.945):
    """Return unchanged basic acceptance plus independent stretch-target details.

``status`` and ``exit_code`` always describe basic no-regression acceptance.
``cosine_target.status`` describes validated combined-table cosine coverage.
``target_achieved`` requires both checks to pass for every requested seed.
An invalid target does not erase the valid baseline assessment; its threshold
is serialized as null and its own status is invalid.
    """
    report = assess_baseline(
        candidate_csvs, baseline_csvs, candidate_model=candidate_model, baseline_model=baseline_model,
        scope=scope, candidate_robustness_csvs=candidate_robustness_csvs,
        baseline_robustness_csvs=baseline_robustness_csvs, require_robustness=require_robustness,
    )
    issues, per_pair = [], []
    try:
        if isinstance(cosine_target, bool):
            raise ValueError("boolean target")
        threshold = float(cosine_target)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("target outside cosine range")
    except (TypeError, ValueError, OverflowError):
        threshold = None
        issues.append({"severity": "invalid", "code": "invalid_cosine_target",
                       "message": "Cosine target must be a finite number in [-1, 1]."})

    if threshold is not None:
        for pair in report["comparisons"]:
            if pair["kind"] != "table1" or pair["noise_version"] != "nv1_nv2":
                continue
            value = pair["metrics"]["CosSim"]["candidate"]
            per_pair.append({
                "kind": "table1", "noise_version": "nv1_nv2", "seed": pair["seed"],
                "candidate": value, "threshold": threshold, "delta": value - threshold,
                "passed": value >= threshold, "rule": ">=",
                "candidate_source": pair["candidate_source"],
            })

    if threshold is None or report["status"] == "invalid":
        status = "invalid"
    elif "nv1_nv2" not in report["required_noise_versions"]:
        status = "incomplete"
        issues.append({"severity": "incomplete", "code": "combined_scope_required",
                       "message": "Stretch target needs scope=combined or scope=all; per-nv assesses basic metrics only."})
    elif (report["status"] == "incomplete" or not per_pair
          or {pair["seed"] for pair in per_pair} != set(report["seeds"])):
        status = "incomplete"
        issues.append({"severity": "incomplete", "code": "missing_validated_combined_pairs",
                       "message": "Complete valid combined candidate/baseline pairs are required for every supplied seed."})
    else:
        status = "passed" if all(pair["passed"] for pair in per_pair) else "failed"

    report["cosine_target"] = {
        "threshold": threshold, "status": status, "scope": "combined_table1_only",
        "comparison_count": len(per_pair), "passed_comparison_count": sum(pair["passed"] for pair in per_pair),
        "per_pair": per_pair, "issues": issues,
        "interpretation": "Each matched combined NV1+NV2 seed must reach the threshold; no threshold is applied to individual NVs or robustness bins.",
    }
    report["target_achieved"] = report["status"] == "passed" and status == "passed"
    report["limits"].append(
        "The bundled v7 baseline snapshot covers only combined NV1+NV2 seed 3407; other seeds/scopes need matching baseline CSVs."
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--baseline-csv", nargs="+", type=Path, default=[DEFAULT_BASELINE])
    parser.add_argument("--candidate-model", default="lstm_v14")
    parser.add_argument("--baseline-model", default=BASELINE_MODEL)
    parser.add_argument("--scope", choices=tuple(SCOPES), default="combined",
                        help="Basic acceptance scope; the stretch target is assessed only for combined rows.")
    parser.add_argument("--candidate-robustness-csv", nargs="+", default=[], type=Path)
    parser.add_argument("--baseline-robustness-csv", nargs="+", default=[], type=Path)
    parser.add_argument("--require-robustness", action="store_true")
    parser.add_argument("--cosine-target", type=float, default=0.945,
                        help="Combined NV1+NV2 stretch target; reported separately from basic acceptance (default: 0.945).")
    parser.add_argument("--output-json", type=Path, default=Path("v14_acceptance.json"))
    args = parser.parse_args(argv)
    if not math.isfinite(args.cosine_target) or not -1.0 <= args.cosine_target <= 1.0:
        parser.error("--cosine-target must be finite and in [-1, 1]")
    inputs = args.candidate_csv + args.baseline_csv + args.candidate_robustness_csv + args.baseline_robustness_csv
    output = args.output_json.resolve()
    for path in inputs:
        if output == path.resolve() or (output.exists() and path.exists() and output.samefile(path)):
            parser.error("--output-json must not overwrite an input CSV")
    report = assess(
        args.candidate_csv, args.baseline_csv, candidate_model=args.candidate_model, baseline_model=args.baseline_model,
        scope=args.scope, candidate_robustness_csvs=args.candidate_robustness_csv,
        baseline_robustness_csvs=args.baseline_robustness_csv, require_robustness=args.require_robustness,
        cosine_target=args.cosine_target,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Basic acceptance: {report['status'].upper()} "
          f"({report['passed_comparison_count']}/{report['comparison_count']} matched comparisons).")
    target = report["cosine_target"]
    print(f"Combined CosSim >= {target['threshold']}: {target['status'].upper()}; "
          f"all requirements including target achieved: {report['target_achieved']}.")
    for item in report["issues"] + target["issues"]:
        print(f"  {item['severity'].upper()} {item['code']}: {item['message']}")
    print(f"Descriptive comparison only; exit code describes basic acceptance. JSON: {output}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
