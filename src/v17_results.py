"""Official metrics, matched contrasts, stage diagnostics and seed variability."""
import csv
import json
from pathlib import Path

import numpy as np

from v12_results import collect_selector, result_files
from mecge_table1_collect_official_protocol import load_result, metric_values
from v11_experiments import atomic_json, file_digest

CONTRASTS = (
    ("v12_topk_control", "v7_max_control", "local gate under old cooperation"),
    ("v12_topk_compact", "v12_topk_control", "algebraic compaction and retraining"),
    ("v17", "v12_topk_compact", "complete cooperation redesign (multiple changed factors)"),
    ("v17", "v17_no_baseline_fusion", "baseline head participates directly in final fusion"),
    ("v17", "v17_condition_gate", "actual flow corrections supplied to gate"),
    ("v17", "v17_no_local_gate", "local gate"),
    ("v17", "v17_no_topk", "top-k objective"),
    ("v17", "v17_no_flow", "flow stage and its objectives (joint removal)"),
    ("v17", "v17_no_dual_head", "dual head and supervision (joint removal)"),
    ("v17", "v17_fixed_gate", "learned gates"),
    ("v17_tcn", "v17", "TCN versus U-Net (capacity also differs)"),
    ("v17_half_flow", "v17", "half-width flow capacity"),
    ("v17_four_steps", "v17", "four versus one Euler evaluations"),
    ("v17_bridge", "v17", "correctly supervised stochastic bridge"),
    ("v17", "v17_direct", "FM versus direct regression; direct model folds fixed time and removes zero state weights"),
)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mark_extended_reports(report_root, filenames):
    path = report_root / "analysis/report_manifest.json"
    manifest = json.loads(path.read_text())
    names = {"analysis/"+name for name in filenames}
    manifest["regenerated"] = sorted(set(manifest["regenerated"]) | names)
    manifest["retained_not_regenerated"] = sorted(set(manifest["retained_not_regenerated"]) - names)
    atomic_json(path, manifest)


def metric_rows(results):
    grouped, rows = {}, []
    for path, parsed in results:
        model, seed, nv = parsed["model"].removeprefix("lstm_"), parsed["seed"], parsed["noise_version"]
        clean, prediction = load_result(path)
        values = metric_values(clean, prediction)
        rows.append(dict(model=model, seed=seed, nv=nv, **{key: float(value.mean()) for key, value in values.items()}))
        grouped.setdefault((model, seed), {})[nv] = (clean, prediction)
    for (model, seed), pair in grouped.items():
        if set(pair) == {"nv1", "nv2"}:
            clean = np.concatenate([pair[nv][0] for nv in ("nv1", "nv2")])
            prediction = np.concatenate([pair[nv][1] for nv in ("nv1", "nv2")])
            rows.append(dict(model=model, seed=seed, nv="nv1_nv2",
                             **{key: float(value.mean()) for key, value in metric_values(clean, prediction).items()}))
    return rows


def contrasts(rows):
    lookup = {(row["model"], row["seed"], row["nv"]): row for row in rows}
    contrasts = []
    groups = sorted({(row["seed"], row["nv"]) for row in rows})
    for candidate, reference, factor in CONTRASTS:
        for seed, nv in groups:
            a, b = lookup.get((candidate, seed, nv)), lookup.get((reference, seed, nv))
            row = dict(candidate=candidate, reference=reference, factor=factor, seed=seed, nv=nv,
                       status="available" if a and b else "missing_pair")
            if a and b:
                row.update({key+"_delta": a[key]-b[key] for key in ("SSD", "MAD", "PRD", "CosSim")})
            contrasts.append(row)
    return contrasts


def collect_results(root, amplitudes=None):
    root, amplitudes = Path(root), amplitudes or {}
    any_results = False
    for report, selector in ((root, "best_val_ssd"), (root / "validation_loss", "best_loss")):
        results = result_files(report)
        if not results:
            continue
        any_results = True
        # Collection never accepts changed predictions as a new valid experiment.
        for path, parsed in results:
            job = root / parsed["model"].removeprefix("lstm_") / parsed["noise_version"] / parsed["seed"]
            manifest = json.loads((job / "manifest.json").read_text())
            if not manifest.get("complete") or manifest["artifacts_sha256"].get(str(path.relative_to(root))) != file_digest(path):
                raise RuntimeError(f"Incomplete or modified experiment result: {path}")
        collect_selector(root, report, selector, results, amplitudes)
        rows = metric_rows(results)
        write_rows(report / "analysis/paired_ablation_contrasts.csv", contrasts(rows))
        groups = {}
        for row in rows:
            groups.setdefault((row["model"], row["nv"]), []).append(row)
        seed_rows = []
        for (model, nv), items in sorted(groups.items()):
            row = dict(model=model, nv=nv, seeds=len(items),
                       note="descriptive_only" if len(items) == 1 else "between_training_seeds")
            for key in ("SSD", "MAD", "PRD", "CosSim"):
                values = np.asarray([item[key] for item in items])
                row[key+"_mean"] = float(values.mean())
                row[key+"_std"] = float(values.std(ddof=1)) if len(values) > 1 else ""
            seed_rows.append(row)
        write_rows(report / "analysis/across_seed_summary.csv", seed_rows)
        mark_extended_reports(report, ("paired_ablation_contrasts.csv", "across_seed_summary.csv"))
    if not any_results:
        raise FileNotFoundError("No completed v17 result PKLs to collect")
    diagnostics, profiles = [], []
    for path in sorted(root.glob("*/nv*/seed*/manifest.json")):
        manifest = json.loads(path.read_text())
        if not manifest.get("complete"):
            continue
        identity = manifest["identity"]
        prefix = {"model": identity["variant"], "nv": identity["nv"], "seed": identity["seed"]}
        checkpoint = path.parent / "checkpoint"
        for filename in ("loss_history.csv", "parameter_audit.json", "selection.json", "profile.json"):
            artifact = checkpoint / filename
            if file_digest(artifact) != manifest["artifacts_sha256"].get(str(artifact.relative_to(root))):
                raise RuntimeError(f"Modified diagnostic artifact: {artifact}")
        selection = json.loads((checkpoint / "selection.json").read_text())["best_epochs"]
        with (checkpoint / "loss_history.csv").open(newline="") as handle:
            history = list(csv.DictReader(handle))
        for selector, epoch in selection.items():
            row = next(row for row in history if int(row["epoch"]) == epoch)
            diagnostics.append(dict(prefix, selector=selector, **row))
        audit = json.loads((checkpoint / "parameter_audit.json").read_text())
        profile = json.loads((checkpoint / "profile.json").read_text())
        profiles.append(dict(prefix, **profile, audit_passed=audit["passed"],
                             dormant_tensors=len(audit["dormant_parameter_tensors"])))
    write_rows(root / "analysis/validation_stage_diagnostics.csv", diagnostics)
    write_rows(root / "analysis/complexity_and_parameter_audit.csv", profiles)
    if (root / "analysis/report_manifest.json").is_file():
        mark_extended_reports(root, ("validation_stage_diagnostics.csv", "complexity_and_parameter_audit.csv"))
    atomic_json(root / "analysis/v17_report_notes.json", {
        "primary_selector": "full-validation SSD", "secondary_selector": "full-validation composite loss",
        "test_used_for_selection": False, "delta_direction": "candidate minus reference",
        "amplitude_files": {str(nv): {"path": str(path), "sha256": file_digest(path)}
                            for nv, path in amplitudes.items() if path is not None},
        "limitations": ["Window-random validation split retained for comparability; not patient-wise.",
                        "Single-seed comparisons are descriptive; no independence assumption across ECG windows.",
                        "Finite gradient probes and exact rewrites cannot prove all statistical redundancy absent.",
                        "Legacy controls intentionally retain historical inactive parameters."]})
