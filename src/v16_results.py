"""Official v16 reports, including completed runs with no feasible checkpoint.

Test scores describe performance only; no training or checkpoint selection reads
this module. Historical v7 acceptance and the re-evaluated initialization are
reported separately. An absent candidate is never replaced with the baseline.
"""

import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np

from mecge_table1_collect_official_protocol import (
    BIN_EDGES, CORE_METRICS, as_nt, metric_fields, metric_values, write_csv,
)
from v11_experiments import atomic_json, file_digest
from v14_acceptance import BASELINE_MODEL, DEFAULT_BASELINE, assess
from v16_experiments import VARIANTS


APP = Path(__file__).resolve().parent
PROTOCOL = "qtdb_train_qtdb_test"
DEFAULT_ROBUSTNESS = DEFAULT_BASELINE.with_name("v14_baseline_v7_robustness.csv")


def result_path(root, model, seed, nv):
    return root / "official_results" / f"{model}__{PROTOCOL}__nv{nv}__seed{seed}.pkl"


def load_prediction(path):
    # As elsewhere in this repository, input PKLs are trusted local artifacts.
    with Path(path).open("rb") as handle:
        content = pickle.load(handle)
    if not isinstance(content, (list, tuple)) or len(content) != 3:
        raise ValueError(f"Expected [noisy, clean, prediction]: {path}")
    arrays = tuple(as_nt(value) for value in content)
    if (not arrays[0].size or any(value.shape != arrays[0].shape for value in arrays)
            or any(not np.isfinite(value).all() for value in arrays)):
        raise ValueError(f"Invalid/non-finite prediction arrays: {path}")
    return arrays


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify_job(root, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variant, noise, seed_text = (manifest_path.parents[2].name,
                               manifest_path.parents[1].name, manifest_path.parent.name)
    nv, seed = int(noise.removeprefix("nv")), int(seed_text.removeprefix("seed"))
    if variant not in VARIANTS:
        raise ValueError(f"Unknown v16 variant: {variant}")
    identity = manifest.get("identity", {})
    if (identity.get("variant"), identity.get("nv"), identity.get("seed")) != (variant, nv, seed):
        raise ValueError(f"Manifest identity mismatch: {manifest_path}")
    info = {"model": "lstm_" + variant, "nv": nv, "seed": seed,
            "manifest": str(manifest_path), "manifest_sha256": file_digest(manifest_path)}
    if manifest.get("complete") is not True:
        return {**info, "status": "incomplete", "selection": None}
    artifacts = manifest.get("artifacts_sha256", {})
    if not artifacts:
        raise ValueError(f"Missing artifact inventory: {manifest_path}")
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Artifact missing, changed, or outside run root: {path}")
    checkpoint = manifest_path.parent / "checkpoint"
    selection_path = checkpoint / "validation_selection.json"
    required = [selection_path, manifest_path.parent / "resolved_config.yaml",
                root / "baseline_v7/provenance" / f"nv{nv}__seed{seed}.json",
                checkpoint / "baseline.pt", checkpoint / "model_last.pt",
                checkpoint / "training_state.pt", checkpoint / "loss_history.csv",
                checkpoint / "baseline_test.pkl",
                result_path(root / "baseline_v7", BASELINE_MODEL, seed, nv)]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("complete") is not True:
        raise ValueError(f"Validation selection is incomplete: {selection_path}")
    status = selection.get("status")
    if manifest.get("status", status) != status:
        raise ValueError(f"Manifest and validation selection disagree: {manifest_path}")
    candidate = result_path(root, info["model"], seed, nv)
    if status == "selected":
        chosen = selection.get("selected_checkpoint")
        baseline = selection.get("baseline_metrics", {})
        if not isinstance(chosen, dict) or type(chosen.get("epoch")) is not int or chosen["epoch"] < 1:
            raise ValueError(f"Candidate must be from an updated training epoch: {selection_path}")
        candidate_metrics = chosen.get("metrics", {})
        keys = ("val_ssd", "val_mad", "val_prd", "val_cossim")
        if (any(not isinstance(mapping.get(key), (int, float)) or not np.isfinite(mapping[key])
                for mapping in (candidate_metrics, baseline) for key in keys)
                or candidate_metrics.get("val_count", 0) != baseline.get("val_count")
                or candidate_metrics.get("val_count", 0) <= 0
                or any(candidate_metrics[key] > baseline[key] for key in keys[:3])
                or candidate_metrics["val_cossim"] < baseline["val_cossim"]
                or all(candidate_metrics[key] == baseline[key] for key in keys)):
            raise ValueError(f"Selected checkpoint violates validation protection: {selection_path}")
        required += [checkpoint / "best_model.pt", candidate]
    elif status == "no_feasible_checkpoint":
        if selection.get("selected_checkpoint") is not None or candidate.exists() or (checkpoint / "best_model.pt").exists():
            raise ValueError(f"No-feasible run contains a stale candidate: {manifest_path}")
    else:
        raise ValueError(f"Unknown completed selection status: {status}")
    for path in required:
        if str(path.relative_to(root)) not in artifacts:
            raise ValueError(f"Required artifact is not covered by manifest: {path}")
    shared = result_path(root / "baseline_v7", BASELINE_MODEL, seed, nv)
    provenance = json.loads((root / "baseline_v7/provenance" / f"nv{nv}__seed{seed}.json").read_text())
    baseline_identity = {key: identity.get(key) for key in
                         ("seed", "nv", "dataset", "dataset_sha256", "init_checkpoint", "init_checkpoint_sha256")}
    if (provenance.get("identity") != baseline_identity or provenance.get("sha256") != file_digest(shared)
            or file_digest(checkpoint / "baseline_test.pkl") != file_digest(shared)):
        raise ValueError(f"Frozen-v7 reference provenance does not match the job: {manifest_path}")
    return {**info, "status": status, "selection": selection,
            "artifacts_sha256": artifacts}


def robustness_rows(model, seed, scope, values, alpha):
    """Slice already computed scope metrics; keep legacy open bin boundaries."""
    rows = []
    for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = alpha > low if high == BIN_EDGES[-1] else (alpha > low) & (alpha < high)
        label = f"{low}-{high}"
        row = {"experiment": f"robustness_official_{scope}", "model": model,
               "dataset_protocol": f"qtdb_robustness_alpha_{label}", "noise_version": scope,
               "seed": f"seed{seed}", "alpha": label,
               "result_name": f"{model}__qtdb_robustness_alpha_{label}__{scope}__seed{seed}"}
        metric_fields(row, {key: value[mask] for key, value in values.items()})
        rows.append(row)
    return rows


def load_amplitudes(path, count, nv):
    alpha = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
    if len(alpha) != count or not np.isfinite(alpha).all():
        raise ValueError(f"Invalid amplitude array for NV{nv}: {path}")
    return alpha


def evaluate_results(report_root, inputs, amplitude_files):
    """Compute both NV summaries and concatenated NV1+NV2 official scores.

    Robustness slices the complete-scope metric arrays, retaining the official
    global clean mean for PRD. It does not recalculate PRD inside each bin.
    """
    analysis = report_root / "analysis"
    per_nv, combined_rows, robust_rows, per_nv_robust = [], [], [], []
    grouped, generated = {}, set()
    for (model, seed, nv), path in sorted(inputs.items()):
        noisy, clean, pred = load_prediction(path)
        grouped.setdefault((model, seed), {})[nv] = (clean, pred)
        values = metric_values(clean, pred)
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError(f"Non-finite official metric: {path}")
        row = {"experiment": "table1_official_per_nv", "model": model,
               "dataset_protocol": PROTOCOL, "noise_version": f"nv{nv}",
               "seed": f"seed{seed}", "alpha": "", "result_name": path.stem}
        metric_fields(row, values)
        per_nv.append(row)
        if amplitude_files.get(nv) is not None:
            alpha = load_amplitudes(amplitude_files[nv], len(clean), nv)
            per_nv_robust.extend(robustness_rows(model, seed, f"nv{nv}", values, alpha))
        destination = analysis / "per_window" / f"{path.stem}.csv"
        fields = ["window_index", *CORE_METRICS]
        write_rows(destination, ({"window_index": i, **{key: float(value[i]) for key, value in values.items()}}
                                 for i in range(len(clean))), fields)
        generated.add(destination)
    for (model, seed), by_nv in sorted(grouped.items()):
        if set(by_nv) != {1, 2}:
            continue
        clean = np.concatenate([by_nv[nv][0] for nv in (1, 2)])
        pred = np.concatenate([by_nv[nv][1] for nv in (1, 2)])
        values = metric_values(clean, pred)
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError(f"Non-finite combined official metric: {model}/seed{seed}")
        row = {"experiment": "table1_official_nv1_nv2", "model": model,
               "dataset_protocol": PROTOCOL, "noise_version": "nv1_nv2", "seed": f"seed{seed}",
               "alpha": "", "result_name": f"{model}__{PROTOCOL}__nv1_nv2__seed{seed}"}
        metric_fields(row, values)
        combined_rows.append(row)
        if not all(amplitude_files.get(nv) is not None for nv in (1, 2)):
            continue
        parts = []
        for nv in (1, 2):
            parts.append(load_amplitudes(amplitude_files[nv], len(by_nv[nv][0]), nv))
        alpha = np.concatenate(parts)
        robust_rows.extend(robustness_rows(model, seed, "nv1_nv2", values, alpha))
    # Always overwrite scope tables, including empty ones; no stale pass survives.
    tables = {
        "table1_comparison__official_nv1_nv2.csv": combined_rows,
        "robustness_comparison__official_nv1_nv2.csv": robust_rows,
        "table1_comparison__qtdb_train_qtdb_test__all.csv": per_nv,
        "robustness_comparison__qtdb__all.csv": per_nv_robust,
    }
    for nv in (1, 2):
        tables[f"table1_comparison__qtdb_train_qtdb_test__nv{nv}.csv"] = [
            row for row in per_nv if row["noise_version"] == f"nv{nv}"]
        tables[f"robustness_comparison__qtdb__nv{nv}.csv"] = [
            row for row in per_nv_robust if row["noise_version"] == f"nv{nv}"]
    for name, rows in tables.items():
        path = analysis / name
        write_csv(path, rows)
        generated.add(path)
    return generated


def unavailable(status, reason):
    return {"status": status, "target_achieved": False, "reason": reason,
            "comparisons": [], "cosine_target": {"threshold": 0.945, "status": status}}


def validation_summary(root, jobs):
    """Retain interpretable validation outcomes even when no candidate qualifies."""
    fields = ["model", "nv", "seed", "status", "epochs_completed", "selected_epoch"]
    keys = ("ssd", "mad", "prd", "cossim")
    fields += [prefix + key for prefix in ("baseline_", "last_", "last_stage1_", "selected_", "last_delta_") for key in keys]
    fields += ["train_loss", "original_v7_loss", "lr_adapter", "lr_flow"]
    records = []
    history_fields = set()
    for job in jobs:
        row = {key: job[key] for key in ("model", "nv", "seed", "status")}
        selection = job.get("selection")
        if selection:
            history = read_rows(Path(job["manifest"]).parent / "checkpoint/loss_history.csv")
            last = history[-1] if history else {}
            selected = selection.get("selected_checkpoint") or {}
            selected_history = next((item for item in history
                                     if item.get("epoch") == str(selected.get("epoch"))), {})
            row.update(epochs_completed=selection.get("epochs_completed", len(history)), selected_epoch=selected.get("epoch", ""))
            for key in keys:
                reference = selection["baseline_metrics"].get("val_" + key)
                value = last.get("val_" + key)
                row.update({"baseline_" + key: reference, "last_" + key: value,
                            "last_stage1_" + key: last.get("val_stage1_" + key, ""),
                            "selected_" + key: selected.get("metrics", {}).get("val_" + key, ""),
                            "last_delta_" + key: float(value) - reference if value not in (None, "") else ""})
            row.update({key: last.get(key, "") for key in fields if key not in row})
            # Preserve gradient, adapter, frozen-hash and future runner columns
            # without requiring collector changes for every new diagnostic.
            for prefix, values in (("last_history_", last), ("selected_history_", selected_history)):
                for key, value in values.items():
                    if key is not None:
                        row[prefix + key] = value
                        history_fields.add(prefix + key)
        records.append(row)
    path = root / "analysis/validation_ablation_summary.csv"
    write_rows(path, records, fields + sorted(history_fields))
    return path


def numeric_leaves(value, prefix=""):
    """Flatten diagnostic summaries, excluding waveform arrays and metadata."""
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            yield from numeric_leaves(item, prefix + ("." if prefix else "") + key)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not np.isfinite(value):
            raise ValueError(f"Non-finite validation diagnostic: {prefix}")
        yield prefix, value


def validation_diagnostics_summary(root, jobs):
    """Keep managed epoch diagnostics visible even without any test candidate."""
    rows = []
    for job in jobs:
        selection = job.get("selection") or {}
        selected_epoch = (selection.get("selected_checkpoint") or {}).get("epoch")
        checkpoint = Path(job["manifest"]).parent / "checkpoint"
        for relative, digest in sorted(job.get("artifacts_sha256", {}).items()):
            path = root / relative
            match = re.fullmatch(r"epoch_(\d+)_diagnostics\.json", path.name)
            if not match or path.parent != checkpoint / "validation":
                continue
            epoch = int(match[1])
            for name, value in numeric_leaves(json.loads(path.read_text(encoding="utf-8"))):
                rows.append({**{key: job[key] for key in ("model", "nv", "seed", "status")},
                             "epoch": epoch, "is_selected": epoch == selected_epoch,
                             "diagnostic": name, "value": value,
                             "source": relative, "source_sha256": digest})
    path = root / "analysis/validation_diagnostics_summary.csv"
    write_rows(path, rows, ["model", "nv", "seed", "status", "epoch", "is_selected",
                            "diagnostic", "value", "source", "source_sha256"])
    return path


CONTRASTS = {
    "correction_features_frozen_flow_C_minus_B": {"lstm_v16": 1, "lstm_v16_condition_gate": -1},
    "correction_features_trainable_flow_E_minus_D": {"lstm_v16_flow": 1, "lstm_v16_condition_gate_flow": -1},
    "flow_update_condition_D_minus_B": {"lstm_v16_condition_gate_flow": 1, "lstm_v16_condition_gate": -1},
    "flow_update_correction_E_minus_C": {"lstm_v16_flow": 1, "lstm_v16": -1},
    "interaction_E_minus_D_minus_C_plus_B": {"lstm_v16_flow": 1, "lstm_v16_condition_gate_flow": -1,
                                             "lstm_v16": -1, "lstm_v16_condition_gate": 1},
}


def ablation_contrasts(root, jobs, seeds):
    """Matched descriptive differences; missing models never become v7 values."""
    observations = {}
    for job in jobs:
        selection = job.get("selection")
        if not selection:
            continue
        history = read_rows(Path(job["manifest"]).parent / "checkpoint/loss_history.csv")
        if history:
            observations[("last_validation", job["model"], job["seed"], f"nv{job['nv']}")] = {
                metric: history[-1].get("val_" + metric.lower(), "") for metric in CORE_METRICS}
        selected = selection.get("selected_checkpoint")
        if selected:
            observations[("selected_validation", job["model"], job["seed"], f"nv{job['nv']}")] = {
                metric: selected["metrics"]["val_" + metric.lower()] for metric in CORE_METRICS}
    for table in ("table1_comparison__qtdb_train_qtdb_test__all.csv", "table1_comparison__official_nv1_nv2.csv"):
        for row in read_rows(root / "analysis" / table):
            observations[("test", row["model"], int(row["seed"].removeprefix("seed")), row["noise_version"])] = {
                metric: row[metric + "_mean"] for metric in CORE_METRICS}
    rows = []
    for phase, scopes in (("last_validation", ("nv1", "nv2")),
                          ("selected_validation", ("nv1", "nv2")), ("test", ("nv1", "nv2", "nv1_nv2"))):
        for seed in seeds:
            for nv in scopes:
                for contrast, terms in CONTRASTS.items():
                    missing = [model for model in terms if (phase, model, seed, nv) not in observations]
                    row = {"phase": phase, "noise_version": nv, "seed": seed, "contrast": contrast,
                           "status": "unavailable" if missing else "available", "missing_models": ";".join(missing)}
                    for metric in CORE_METRICS:
                        values = [observations.get((phase, model, seed, nv), {}).get(metric, "") for model in terms]
                        if any(value in (None, "") for value in values):
                            row[metric + "_delta"] = ""
                            row["status"] = "unavailable"
                        else:
                            row[metric + "_delta"] = sum(coefficient * float(value)
                                                         for coefficient, value in zip(terms.values(), values))
                    rows.append(row)
    path = root / "analysis/ablation_contrasts.csv"
    write_rows(path, rows, ["phase", "noise_version", "seed", "contrast", "status", "missing_models",
                           *[metric + "_delta" for metric in CORE_METRICS]])
    return path


def compare_baseline_reproduction(actual_table, historical_tables, seeds):
    actual = {int(r["seed"].removeprefix("seed")): r for r in read_rows(actual_table)}
    history = {}
    for path in historical_tables:
        if not Path(path).is_file():
            continue
        for row in read_rows(path):
            if row.get("model") == BASELINE_MODEL and row.get("noise_version") == "nv1_nv2":
                history.setdefault(int(row["seed"].removeprefix("seed")), []).append(row)
    records = []
    for seed in seeds:
        if seed not in actual or len(history.get(seed, [])) != 1:
            records.append({"seed": seed, "status": "incomplete", "deltas": {}})
            continue
        left, right = actual[seed], history[seed][0]
        deltas, matches = {}, True
        for metric in CORE_METRICS:
            a, b = float(left[metric + "_mean"]), float(right[metric + "_mean"])
            deltas[metric] = a - b
            matches &= (int(left[metric + "_count"]) == int(right[metric + "_count"])
                        and bool(np.isclose(a, b, rtol=1e-5, atol=1e-6)))
        records.append({"seed": seed, "status": "matched" if matches else "mismatch", "deltas": deltas})
    return {"rtol": 1e-5, "atol": 1e-6, "per_seed": records,
            "note": "Reproduction tolerance is diagnostic only; acceptance uses unrounded official metrics."}


def collect_results(root, amplitude_files=None, *, expected_models=None, expected_seeds=None,
                    baseline_csvs=None, baseline_robustness_csvs=None):
    root = Path(root).resolve()
    amplitudes = amplitude_files or {}
    historical = [DEFAULT_BASELINE] if baseline_csvs is None else list(map(Path, baseline_csvs))
    historical_robust = [DEFAULT_ROBUSTNESS] if baseline_robustness_csvs is None else list(map(Path, baseline_robustness_csvs))
    analysis = root / "analysis"
    for path in [*historical, *historical_robust, *[p for p in amplitudes.values() if p is not None]]:
        if any(Path(path).resolve().is_relative_to(directory) for directory in
               (analysis, root / "baseline_v7/analysis")):
            raise ValueError(f"Reporting input is inside a regenerated output directory: {path}")
    models = set(expected_models or ())
    seeds = {int(str(s).removeprefix("seed")) for s in (expected_seeds or ())}
    report_path = analysis / "v16_acceptance.json"
    summary_path = analysis / "v16_acceptance_summary.csv"
    atomic_json(report_path, {"complete": False, "status": "collecting", "reports": {}})
    summary_fields = ["model", "seed", "nv1_status", "nv2_status", "basic_status", "loaded_v7_status",
                      "cosine_target_status", "target_achieved", *CORE_METRICS,
                      *[m + "_delta_vs_v7" for m in CORE_METRICS]]
    write_rows(summary_path, [], summary_fields)
    try:
        jobs = [verify_job(root, path) for path in sorted(root.glob("v16*/nv[12]/seed*/manifest.json"))]
        models.update(job["model"] for job in jobs)
        seeds.update(job["seed"] for job in jobs)
        lookup = {(job["model"], job["seed"], job["nv"]): job for job in jobs}
        candidate_inputs, baseline_inputs = {}, {}
        for key, job in lookup.items():
            model, seed, nv = key
            if job["status"] == "incomplete":
                continue
            baseline_path = result_path(root / "baseline_v7", BASELINE_MODEL, seed, nv)
            baseline_inputs[(BASELINE_MODEL, seed, nv)] = baseline_path
            if job["status"] == "selected":
                candidate_path = result_path(root, model, seed, nv)
                baseline, candidate = load_prediction(baseline_path), load_prediction(candidate_path)
                if any(not np.array_equal(baseline[i], candidate[i]) for i in (0, 1)):
                    raise ValueError(f"Candidate and v7 must use identical noisy/clean windows: {candidate_path}")
                candidate_inputs[key] = candidate_path
        for path in (root / "official_results").glob("*.pkl"):
            if path not in candidate_inputs.values():
                raise ValueError(f"Unmanaged or stale primary result: {path}")
        generated = evaluate_results(root, candidate_inputs, amplitudes)
        generated |= evaluate_results(root / "baseline_v7", baseline_inputs, amplitudes)
        generated.add(validation_summary(root, jobs))
        generated.add(validation_diagnostics_summary(root, jobs))
        generated.add(ablation_contrasts(root, jobs, sorted(seeds)))
        actual_table = root / "baseline_v7/analysis/table1_comparison__official_nv1_nv2.csv"
        actual_robust = root / "baseline_v7/analysis/robustness_comparison__official_nv1_nv2.csv"
        combined = analysis / "table1_comparison__official_nv1_nv2.csv"
        robust = analysis / "robustness_comparison__official_nv1_nv2.csv"
        per_nv_table = analysis / "table1_comparison__qtdb_train_qtdb_test__all.csv"
        per_nv_baseline = root / "baseline_v7/analysis/table1_comparison__qtdb_train_qtdb_test__all.csv"
        per_nv_robust = analysis / "robustness_comparison__qtdb__all.csv"
        per_nv_baseline_robust = root / "baseline_v7/analysis/robustness_comparison__qtdb__all.csv"
        reports, summary = {}, []
        nv_summary = []
        for model in sorted(models):
            historical_report = assess([combined], historical, candidate_model=model)
            loaded_report = assess([combined], [actual_table], candidate_model=model)
            robust_report = assess([combined], [actual_table], candidate_model=model,
                                   candidate_robustness_csvs=[robust], baseline_robustness_csvs=[actual_robust],
                                   require_robustness=True)
            historical_robust_report = assess([combined], historical, candidate_model=model,
                                              candidate_robustness_csvs=[robust],
                                              baseline_robustness_csvs=historical_robust,
                                              require_robustness=True)
            nv_report = assess([per_nv_table], [per_nv_baseline], candidate_model=model, scope="per-nv")
            nv_robust_report = assess([per_nv_table], [per_nv_baseline], candidate_model=model, scope="per-nv",
                                      candidate_robustness_csvs=[per_nv_robust],
                                      baseline_robustness_csvs=[per_nv_baseline_robust], require_robustness=True)
            all_reports = (historical_report, loaded_report, robust_report, historical_robust_report,
                           nv_report, nv_robust_report)
            states = {seed: {nv: lookup.get((model, seed, nv), {}).get("status", "missing")
                             for nv in (1, 2)} for seed in sorted(seeds)}
            if any(state == "no_feasible_checkpoint" for pair in states.values() for state in pair.values()):
                for report in all_reports:
                    if report["status"] != "invalid":
                        report.update(status="no_feasible_checkpoint", exit_code=1, target_achieved=False)
                        report["cosine_target"]["status"] = "no_feasible_checkpoint"
            elif any(state != "selected" for pair in states.values() for state in pair.values()):
                for report in all_reports:
                    if report["status"] != "invalid":
                        report.update(status="incomplete", exit_code=2, target_achieved=False)
                        report["cosine_target"]["status"] = "incomplete"
            historical_report["target_achieved"] = (historical_report["target_achieved"]
                                                       and loaded_report["target_achieved"])
            reports[model] = {**historical_report, "loaded_v7_comparison": loaded_report,
                              "robustness_diagnostic": robust_report,
                              "historical_robustness_diagnostic": historical_robust_report,
                              "per_nv_diagnostic": nv_report, "per_nv_robustness_diagnostic": nv_robust_report,
                              "job_states": states}
            nv_pairs = {(int(pair["seed"]), pair["noise_version"]): pair for pair in nv_report["comparisons"]
                        if pair["kind"] == "table1"}
            pairs = {int(pair["seed"]): pair for pair in historical_report["comparisons"]
                     if pair["kind"] == "table1"}
            for seed, state in states.items():
                pair = pairs.get(seed, {})
                row = {"model": model, "seed": seed, "nv1_status": state[1], "nv2_status": state[2],
                       "basic_status": historical_report["status"], "loaded_v7_status": loaded_report["status"],
                       "cosine_target_status": historical_report["cosine_target"]["status"],
                       "target_achieved": historical_report["target_achieved"]}
                for metric in CORE_METRICS:
                    item = pair.get("metrics", {}).get(metric, {})
                    row[metric] = item.get("candidate", "")
                    row[metric + "_delta_vs_v7"] = item.get("delta", "")
                summary.append(row)
                for nv in (1, 2):
                    pair = nv_pairs.get((seed, f"nv{nv}"), {})
                    item = {"model": model, "seed": seed, "noise_version": f"nv{nv}",
                            "job_status": state[nv],
                            "comparison_status": ("passed" if pair["passed"] else "failed") if pair else "unavailable"}
                    for metric in CORE_METRICS:
                        value = pair.get("metrics", {}).get(metric, {})
                        item[metric] = value.get("candidate", "")
                        item[metric + "_baseline"] = value.get("baseline", "")
                        item[metric + "_delta"] = value.get("delta", "")
                    nv_summary.append(item)
        reproduction = compare_baseline_reproduction(actual_table, historical, sorted(seeds))
        provenance = {"jobs": jobs, "historical_inputs": [
            {"path": str(path), "sha256": file_digest(path)} for path in historical + historical_robust if path.is_file()],
            "amplitude_inputs": [{"nv": nv, "path": str(path), "sha256": file_digest(path)}
                                 for nv, path in sorted(amplitudes.items()) if path is not None],
            "sources": {name: file_digest(APP / name) for name in (
                "v16_results.py", "v16_experiments.py", "v11_experiments.py",
                "v14_acceptance.py", "v13_acceptance.py", "utils_ecg.py",
                "mecge_table1_collect_official_protocol.py")}}
        write_rows(summary_path, summary, summary_fields)
        nv_summary_path = analysis / "per_nv_comparison_summary.csv"
        write_rows(nv_summary_path, nv_summary,
                   ["model", "seed", "noise_version", "job_status", "comparison_status", *CORE_METRICS,
                    *[metric + suffix for suffix in ("_baseline", "_delta") for metric in CORE_METRICS]])
        generated.add(nv_summary_path)
        provenance_path = analysis / "collection_provenance.json"
        atomic_json(provenance_path, provenance)
        generated.add(provenance_path)
        payload = {"complete": True, "checkpoint_selector": "validation_four_metric_guard",
                   "primary_baseline": "historical_v7", "reports": reports,
                   "frozen_reference": BASELINE_MODEL,
                   "baseline_reproduction": reproduction, "provenance": provenance,
                   "note": "Test reports never choose checkpoints or hyperparameters. No feasible model is not replaced with v7. All requested seeds/NVs must be present. Per-NV and interaction tables are descriptive diagnostics, not selection criteria. Last-validation contrasts include infeasible completed runs. Single-seed differences are descriptive."}
        atomic_json(report_path, payload)
        generated.update([report_path, summary_path])
        atomic_json(analysis / "report_manifest.json", {
            "complete": True, "regenerated": sorted(str(p.relative_to(root)) for p in generated),
            "artifacts_sha256": {str(p.relative_to(root)): file_digest(p) for p in sorted(generated)},
            "note": "Only listed artifacts are current; other retained files must not be treated as this collection's evidence."})
        for model, report in reports.items():
            print(f"{model}: historical v7={report['status']}; loaded v7={report['loaded_v7_comparison']['status']}; CosSim target={report['cosine_target']['status']}")
        return payload
    except Exception as error:
        atomic_json(report_path, {"complete": False, "status": "invalid", "error": str(error),
                                 "reports": {model: unavailable("invalid", str(error)) for model in sorted(models)}})
        atomic_json(analysis / "report_manifest.json", {"complete": False, "status": "invalid", "error": str(error)})
        write_rows(summary_path, [], summary_fields)
        raise
