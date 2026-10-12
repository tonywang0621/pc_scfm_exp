"""Collect official v14 scores and descriptive acceptance, separately by selector.

Training never reads these baseline CSVs or any test scores. Historical v7
snapshots supply only seed3407 combined comparisons; missing evidence remains
incomplete. Stale reports are never used to claim acceptance.
"""

import csv
import json
from pathlib import Path

from v11_experiments import atomic_json, file_digest
from v12_results import collect_selector, result_files, write_report_manifest
from v14_acceptance import BASELINE_MODEL, DEFAULT_BASELINE, assess


DEFAULT_ROBUSTNESS_BASELINE = DEFAULT_BASELINE.with_name("v14_baseline_v7_robustness.csv")
APP = Path(__file__).resolve().parent
REPORTING_SOURCES = (
    "v14_results.py", "v12_results.py", "v14_acceptance.py", "v13_acceptance.py",
    "v11_experiments.py", "utils_ecg.py", "mecge_table1_official_result_metrics.py",
    "mecge_table1_collect_official_protocol.py", "mecge_table1_robustness_bins.py",
    "mecge_table1_collect_results.py",
)
ACCEPTANCE_FILES = ("analysis/v14_acceptance.json", "analysis/v14_acceptance_summary.csv",
                    "analysis/collection_provenance.json")


def reporting_provenance(amplitude_files):
    """Record the exact reporting code and amplitude inputs, without loading PKLs."""
    amplitudes = []
    for nv, filename in sorted(amplitude_files.items()):
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Robustness amplitude input is missing: {path}")
        amplitudes.append({"noise_version": f"nv{nv}", "path": str(path), "sha256": file_digest(path)})
    return {"amplitude_inputs": amplitudes,
            "reporting_sources": {name: file_digest(APP / name) for name in REPORTING_SOURCES}}


def verify_managed_manifest(root, manifest_path):
    """Verify a completed managed job and all declared artifacts before scoring.

    Hash verification establishes agreement with this local manifest, not the
    scientific validity or authenticity of externally supplied files.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read managed run manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise RuntimeError(f"Managed run is incomplete: {manifest_path}")
    identity, artifacts = manifest.get("identity"), manifest.get("artifacts_sha256")
    variant = manifest_path.parents[2].name
    noise, seed_label = manifest_path.parents[1].name, manifest_path.parent.name
    try:
        identity_matches = (isinstance(identity, dict)
                            and identity.get("variant") == variant
                            and int(identity.get("seed")) == int(seed_label.removeprefix("seed"))
                            and int(identity.get("nv")) == int(noise.removeprefix("nv")))
    except (TypeError, ValueError):
        identity_matches = False
    if not identity_matches or not isinstance(artifacts, dict):
        raise RuntimeError(f"Managed manifest identity or artifact inventory is invalid: {manifest_path}")
    stem = f"lstm_{variant}__qtdb_train_qtdb_test__{noise}__{seed_label}.pkl"
    checkpoint = manifest_path.parent / "checkpoint"
    required = (root / "official_results" / stem, root / "validation_ssd/official_results" / stem,
                checkpoint / "best_model.pt", checkpoint / "best_val_ssd.pt")
    required_keys = {str(path.relative_to(root)) for path in required}
    if not required_keys.issubset(artifacts):
        raise RuntimeError(f"Managed manifest is missing required artifact hashes: {manifest_path}")
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Managed artifact path escapes the run root: {relative}")
        if not path.is_file():
            raise RuntimeError(f"Managed artifact is missing: {path}")
        if not isinstance(expected, str) or file_digest(path) != expected:
            raise RuntimeError(f"Managed artifact hash mismatch: {path}")
    return {"manifest": str(manifest_path), "manifest_sha256": file_digest(manifest_path),
            "artifacts_sha256": artifacts}


def collection_provenance(root, selectors, amplitude_files, managed_paths):
    """Preflight every managed job, then classify selector inputs explicitly."""
    common = reporting_provenance(amplitude_files)
    verified = {path: verify_managed_manifest(root, path) for path in managed_paths}
    provenance = {}
    for report_root, selector, results in selectors:
        inputs = []
        seen = set()
        for path, parsed in results:
            identity = (parsed["model"], parsed["noise_version"], parsed["seed"])
            if identity in seen:
                raise RuntimeError(f"Duplicate result identity for {selector}: {identity}")
            seen.add(identity)
            job = root / parsed["model"].removeprefix("lstm_") / parsed["noise_version"] / parsed["seed"]
            manifest_path = job / "manifest.json"
            checkpoint = job / "checkpoint" / ("best_model.pt" if selector == "best_val_score" else "best_val_ssd.pt")
            managed = verified.get(manifest_path)
            # An unexpected naming shape cannot silently bypass an existing
            # manifest merely because it was outside the conventional glob.
            if managed is None and manifest_path.is_file():
                managed = verify_managed_manifest(root, manifest_path)
            if managed is not None and managed["artifacts_sha256"].get(str(path.relative_to(root))) != file_digest(path):
                raise RuntimeError(f"Result is not the artifact recorded by its managed manifest: {path}")
            inputs.append({
                "result": str(path), "result_sha256": file_digest(path),
                "model": parsed["model"], "noise_version": parsed["noise_version"], "seed": parsed["seed"],
                "status": "verified" if managed is not None else "unverified",
                "manifest": str(manifest_path) if managed is not None else None,
                "manifest_sha256": managed["manifest_sha256"] if managed is not None else None,
                "selected_checkpoint": str(checkpoint) if checkpoint.is_file() else None,
                "selected_checkpoint_sha256": file_digest(checkpoint) if checkpoint.is_file() else None,
            })
        status = "incomplete" if not inputs else (
            "verified" if all(item["status"] == "verified" for item in inputs) else "unverified")
        provenance[selector] = {
            "schema_version": 1, "checkpoint_selector": selector,
            "status": status, "verified": status == "verified", "inputs": inputs, **common,
            "note": ("Verified means the current artifacts match completed local run manifests. "
                     "Missing manifests leave descriptive scores unverified; no checkpoint provenance is certified. "
                     "Hashes do not establish scientific validity or authenticity."),
        }
    return provenance


def write_acceptance(report_root, selector, models, seeds, *, regenerated,
                     baseline_csvs, baseline_robustness_csvs, cosine_target,
                     provenance=None, collection_issue=None):
    analysis = report_root / "analysis"
    combined = analysis / "table1_comparison__official_nv1_nv2.csv"
    robust = analysis / "robustness_comparison__official_nv1_nv2.csv"
    # Only consume this collection's freshly regenerated input tables. Existing
    # missing/stale paths can be represented by an empty input list in assess.
    candidates = [combined] if combined in regenerated else []
    robust_candidates = [robust] if robust in regenerated else []
    reports = {}
    summary = []
    provenance = provenance or {"status": "unverified", "verified": False,
                                "note": "This direct summary did not verify input run manifests."}
    for model in models:
        report = assess(candidates, baseline_csvs, candidate_model=model,
                        cosine_target=cosine_target)
        missing_seeds = sorted(set(seeds) - set(report["seeds"]))
        if missing_seeds:
            report["issues"].append({"severity": "incomplete", "code": "requested_seed_missing",
                                     "message": f"Requested seed(s) have no combined evidence: {missing_seeds}"})
            if report["status"] != "invalid":
                report["status"], report["exit_code"] = "incomplete", 2
                report["cosine_target"]["status"] = "incomplete"
            report["target_achieved"] = False
        # Robustness uses all four strict legacy bins. It is diagnostic, not a
        # substitute for the user's primary combined-table acceptance rule.
        report["robustness_diagnostic"] = assess(
            candidates, baseline_csvs, candidate_model=model,
            candidate_robustness_csvs=robust_candidates,
            baseline_robustness_csvs=baseline_robustness_csvs,
            require_robustness=True, cosine_target=cosine_target,
        )
        if collection_issue is not None:
            for assessment in (report, report["robustness_diagnostic"]):
                assessment["issues"].append(dict(collection_issue))
                assessment["status"] = collection_issue["severity"]
                assessment["exit_code"] = 3 if assessment["status"] == "invalid" else 2
                assessment["cosine_target"]["status"] = assessment["status"]
                assessment["target_achieved"] = False
        report["provenance_status"] = provenance["status"]
        report["provenance_verified"] = provenance["verified"]
        reports[model] = report
        comparisons = {r["seed"]: r for r in report["comparisons"] if r["kind"] == "table1"}
        row_seeds = sorted(set(seeds) | set(report["seeds"])) or [""]
        for seed in row_seeds:
            pair = comparisons.get(seed)
            row = {"model": model, "seed": seed, "checkpoint_selector": selector,
                   "basic_status": report["status"], "cosine_target": cosine_target,
                   "cosine_target_status": report["cosine_target"]["status"],
                   "all_requirements_with_target": report["target_achieved"],
                   "robustness_diagnostic_status": report["robustness_diagnostic"]["status"],
                   "provenance_status": provenance["status"], "provenance_verified": provenance["verified"]}
            for metric in ("SSD", "MAD", "PRD", "CosSim"):
                item = pair["metrics"][metric] if pair else {}
                row[metric] = item.get("candidate", "")
                row[metric + "_delta_vs_v7"] = item.get("delta", "")
            summary.append(row)
    analysis.mkdir(parents=True, exist_ok=True)
    atomic_json(analysis / "v14_acceptance.json", {
        "checkpoint_selector": selector, "baseline_model": BASELINE_MODEL,
        "primary_scope": "combined", "expected_seeds": list(seeds), "reports": reports,
        "provenance": provenance,
        "note": "Descriptive test comparison only; does not select models, checkpoints, weights, or seeds. Robustness is diagnostic. All supplied seeds must pass.",
    })
    if summary:
        path = analysis / "v14_acceptance_summary.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        temporary.replace(path)
    for model, report in reports.items():
        print(f"{selector}/{model}: vs v7={report['status']}, "
              f"CosSim >= {cosine_target}={report['cosine_target']['status']}, "
              f"provenance={provenance['status']}", flush=True)
    return reports


def finalize_manifest(report_root, selector, results, amplitudes, regenerated, provenance):
    """Mark acceptance written after the legacy collector as fresh, too."""
    atomic_json(report_root / "analysis/collection_provenance.json", provenance)
    refreshed = set(regenerated)
    refreshed.update(report_root / name for name in ACCEPTANCE_FILES
                     if (report_root / name).is_file())
    robust_versions = {parsed["noise_version"] for _, parsed in results
                       if amplitudes.get(int(parsed["noise_version"][2:])) is not None}
    write_report_manifest(report_root, selector, results, robust_versions, refreshed)


def invalidate_acceptance(selectors, models, seeds, baseline_csvs, baseline_robustness_csvs,
                          cosine_target, provenance, issue):
    """Replace any old pass before failure or before potentially failing scoring."""
    for report_root, selector, results in selectors:
        names = models
        if not names:
            previous = report_root / "analysis/v14_acceptance.json"
            if previous.is_file():
                try:
                    names = sorted(json.loads(previous.read_text(encoding="utf-8")).get("reports", {}))
                except (OSError, ValueError, TypeError):
                    names = []
        current = {**provenance.get(selector, {}), "collection_status": issue["severity"]}
        current.setdefault("status", "invalid" if issue["severity"] == "invalid" else "incomplete")
        current.setdefault("verified", False)
        if issue["severity"] == "invalid":
            current["status"], current["verified"] = "invalid", False
        current["issues"] = [issue]
        write_acceptance(report_root, selector, names, seeds, regenerated=set(),
                         baseline_csvs=baseline_csvs, baseline_robustness_csvs=baseline_robustness_csvs,
                         cosine_target=cosine_target, provenance=current, collection_issue=issue)
        finalize_manifest(report_root, selector, results, {}, set(), current)


def collect_results(root, amplitude_files=None, *, baseline_csvs=None,
                    baseline_robustness_csvs=None, expected_models=None,
                    expected_seeds=None, cosine_target=0.945):
    root = Path(root).resolve()
    amplitude_files = amplitude_files or {}
    baseline_csvs = [DEFAULT_BASELINE] if baseline_csvs is None else list(baseline_csvs)
    baseline_robustness_csvs = ([DEFAULT_ROBUSTNESS_BASELINE] if baseline_robustness_csvs is None
                              else list(baseline_robustness_csvs))
    selectors = [(root, "best_val_score", result_files(root)),
                 (root / "validation_ssd", "best_val_ssd", result_files(root / "validation_ssd"))]
    discovered = [parsed for _, _, results in selectors for _, parsed in results]
    managed_paths = sorted(root.glob("*/nv[12]/seed*/manifest.json"))
    models = sorted(set(expected_models or ()) | {r["model"] for r in discovered}
                    | {"lstm_" + path.parents[2].name for path in managed_paths})
    seeds = sorted({int(str(s).removeprefix("seed")) for s in (expected_seeds or ())})
    try:
        provenance = collection_provenance(root, selectors, amplitude_files, managed_paths)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        issue = {"severity": "invalid", "code": "collection_provenance_failed", "message": str(error)}
        invalidate_acceptance(selectors, models, seeds, baseline_csvs, baseline_robustness_csvs,
                              cosine_target, {}, issue)
        raise RuntimeError(f"Collection provenance verification failed: {error}") from error
    if not discovered and not expected_models:
        message = f"No nv1/nv2 result PKLs found under {root}"
        issue = {"severity": "incomplete", "code": "no_current_results", "message": message}
        invalidate_acceptance(selectors, models, seeds, baseline_csvs, baseline_robustness_csvs,
                              cosine_target, provenance, issue)
        raise FileNotFoundError(message)
    # A failed or interrupted report subprocess must not leave old PASS files.
    invalidate_acceptance(
        selectors, models, seeds, baseline_csvs, baseline_robustness_csvs, cosine_target, provenance,
        {"severity": "incomplete", "code": "collection_in_progress", "message": "This collection has not completed."},
    )
    for report_root, selector, results in selectors:
        regenerated = set()
        if results:
            collect_selector(root, report_root, selector, results, amplitude_files)
            # Read the report manifest after a successful fresh collection.
            manifest = json.loads((report_root / "analysis/report_manifest.json").read_text(encoding="utf-8"))
            regenerated = {report_root / name for name in manifest["regenerated"]}
        else:
            print(f"Skipping {selector}: no current result PKLs; acceptance is incomplete.", flush=True)
            write_report_manifest(report_root, selector, [], set(), set())
        write_acceptance(report_root, selector, models, seeds, regenerated=regenerated,
                         baseline_csvs=baseline_csvs, baseline_robustness_csvs=baseline_robustness_csvs,
                         cosine_target=cosine_target, provenance=provenance[selector])
        finalize_manifest(report_root, selector, results, amplitude_files, regenerated, provenance[selector])
