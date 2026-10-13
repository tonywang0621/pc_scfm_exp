"""Meaningful report checks: same data, exact official metrics, missing candidates.

Small prediction fixtures validate the reporting pipeline, not learned quality.
"""

import contextlib
import csv
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from v16_results import BASELINE_MODEL, collect_results, evaluate_results, result_path
from v11_experiments import file_digest
from mecge_table1_collect_official_protocol import metric_fields, metric_values


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class V16ResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.root = self.work / "runs"
        self.payloads = {}

    def job(self, nv, status="selected", model="v16", seed=3407):
        job = self.root / model / f"nv{nv}" / f"seed{seed}"
        cp = job / "checkpoint"
        cp.mkdir(parents=True)
        rng = np.random.default_rng(nv + seed)
        clean = (rng.normal(size=(6, 16, 1)) + nv).astype(np.float32)
        noisy = clean + np.float32(.4)
        baseline = clean + rng.normal(0, .3, clean.shape).astype(np.float32)
        prediction = clean + rng.normal(0, .01, clean.shape).astype(np.float32)
        self.payloads[nv] = (clean, prediction)
        base = result_path(self.root / "baseline_v7", BASELINE_MODEL, seed, nv)
        base.parent.mkdir(parents=True, exist_ok=True)
        content = pickle.dumps([noisy, clean, baseline])
        base.write_bytes(content)
        (cp / "baseline_test.pkl").write_bytes(content)
        artifacts = [base, cp / "baseline_test.pkl"]
        config = job / "resolved_config.yaml"
        config.write_text("fixture: true\n")
        base_manifest = self.root / "baseline_v7/provenance" / f"nv{nv}__seed{seed}.json"
        base_manifest.parent.mkdir(parents=True, exist_ok=True)
        base_manifest.write_text(json.dumps({"sha256": file_digest(base), "identity": {
            "seed": seed, "nv": nv, "dataset": None, "dataset_sha256": None,
            "init_checkpoint": None, "init_checkpoint_sha256": None}}))
        artifacts.extend([config, base_manifest])
        for name in ("baseline.pt", "model_last.pt", "training_state.pt", "loss_history.csv"):
            path = cp / name
            path.write_bytes(f"fixture/{nv}/{name}".encode())
            artifacts.append(path)
        selection = {"complete": True, "status": status,
                     "baseline_metrics": {"val_ssd": 2., "val_mad": .5, "val_prd": 25., "val_cossim": .9, "val_count": 6},
                     "selected_checkpoint": None}
        if status == "selected":
            selection["selected_checkpoint"] = {"epoch": 1, "metrics": {
                "val_ssd": 1., "val_mad": .4, "val_prd": 20., "val_cossim": .95, "val_count": 6}}
            predpath = result_path(self.root, "lstm_" + model, seed, nv)
            predpath.parent.mkdir(parents=True, exist_ok=True)
            predpath.write_bytes(pickle.dumps([noisy, clean, prediction]))
            best = cp / "best_model.pt"
            best.write_bytes(b"selected")
            artifacts.extend([predpath, best])
        selection_path = cp / "validation_selection.json"
        selection_path.write_text(json.dumps(selection))
        artifacts.append(selection_path)
        manifest = {"complete": True, "identity": {"variant": model, "nv": nv, "seed": seed},
                    "artifacts_sha256": {str(p.relative_to(self.root)): file_digest(p) for p in artifacts}}
        target = job / "manifest.json"
        target.write_text(json.dumps(manifest))
        return target

    def historical(self):
        inputs = {(BASELINE_MODEL, 3407, nv): result_path(self.root / "baseline_v7", BASELINE_MODEL, 3407, nv)
                  for nv in (1, 2) if result_path(self.root / "baseline_v7", BASELINE_MODEL, 3407, nv).exists()}
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate_results(self.work / "historical", inputs, {})
        return self.work / "historical/analysis/table1_comparison__official_nv1_nv2.csv"

    def collect(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return collect_results(self.root, baseline_csvs=[self.historical()],
                                   baseline_robustness_csvs=[], expected_models=kwargs.pop("expected_models", ["lstm_v16"]),
                                   expected_seeds=kwargs.pop("expected_seeds", [3407]), **kwargs)

    def managed_write(self, manifest_path, relative, content):
        path = manifest_path.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts_sha256"][str(path.relative_to(self.root))] = file_digest(path)
        manifest_path.write_text(json.dumps(manifest))
        return path

    def test_combined_exact_official_metrics_and_reproduced_baseline(self):
        for nv in (1, 2):
            self.job(nv)
        result = self.collect()
        self.assertTrue(result["complete"])
        self.assertEqual(result["reports"]["lstm_v16"]["status"], "passed")
        self.assertTrue(result["reports"]["lstm_v16"]["target_achieved"])
        self.assertEqual(result["baseline_reproduction"]["per_seed"][0]["status"], "matched")
        actual = rows(self.root / "analysis/table1_comparison__official_nv1_nv2.csv")[0]
        expected = {}
        clean = np.concatenate([self.payloads[n][0].squeeze(-1) for n in (1, 2)])
        pred = np.concatenate([self.payloads[n][1].squeeze(-1) for n in (1, 2)])
        metric_fields(expected, metric_values(clean, pred))
        for key, value in expected.items():
            self.assertEqual(float(actual[key]), value)
        manifest = json.loads((self.root / "analysis/report_manifest.json").read_text())
        self.assertIn("analysis/v16_acceptance.json", manifest["artifacts_sha256"])

    def test_no_feasible_is_completed_failure_without_fallback(self):
        self.job(1)
        self.job(2, "no_feasible_checkpoint")
        result = self.collect()
        report = result["reports"]["lstm_v16"]
        self.assertEqual(report["status"], "no_feasible_checkpoint")
        self.assertFalse(report["target_achieved"])
        self.assertEqual(rows(self.root / "analysis/table1_comparison__official_nv1_nv2.csv"), [])
        self.assertEqual(len(rows(self.root / "baseline_v7/analysis/table1_comparison__official_nv1_nv2.csv")), 1)

    def test_missing_seed_or_noise_cannot_pass(self):
        self.job(1)
        result = self.collect(expected_seeds=[3407, 42])
        self.assertEqual(result["reports"]["lstm_v16"]["status"], "incomplete")
        self.assertFalse(result["reports"]["lstm_v16"]["target_achieved"])
        self.assertNotEqual(result["reports"]["lstm_v16"]["exit_code"], 0)
        self.assertNotEqual(result["reports"]["lstm_v16"]["robustness_diagnostic"]["status"], "passed")

    def test_tampered_artifact_invalidates_previous_pass(self):
        for nv in (1, 2):
            self.job(nv)
        self.collect()
        result_path(self.root, "lstm_v16", 3407, 1).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Artifact missing, changed"):
            self.collect()
        report = json.loads((self.root / "analysis/v16_acceptance.json").read_text())
        self.assertEqual(report["status"], "invalid")
        self.assertFalse(report["complete"])
        self.assertEqual(rows(self.root / "analysis/v16_acceptance_summary.csv"), [])

    def test_changed_clean_data_rejected_even_with_matching_manifest_hash(self):
        manifests = [self.job(nv) for nv in (1, 2)]
        path = result_path(self.root, "lstm_v16", 3407, 1)
        content = pickle.loads(path.read_bytes())
        content[1] = content[1] + .1
        path.write_bytes(pickle.dumps(content))
        manifest = json.loads(manifests[0].read_text())
        manifest["artifacts_sha256"][str(path.relative_to(self.root))] = file_digest(path)
        manifests[0].write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identical noisy/clean"):
            self.collect()

    def test_robustness_uses_full_scope_prd_and_strict_boundaries(self):
        alpha = np.array([.3, .6, .8, 1.0, 1.2, 1.8])
        amplitudes = {}
        for nv in (1, 2):
            self.job(nv)
            path = self.work / f"alpha{nv}.npy"
            np.save(path, alpha)
            amplitudes[nv] = path
        self.collect(amplitude_files=amplitudes)
        bins = rows(self.root / "analysis/robustness_comparison__official_nv1_nv2.csv")
        self.assertEqual([int(row["PRD_count"]) for row in bins], [2, 2, 2, 2])
        clean = np.concatenate([self.payloads[n][0].squeeze(-1) for n in (1, 2)])
        pred = np.concatenate([self.payloads[n][1].squeeze(-1) for n in (1, 2)])
        official = metric_values(clean, pred)["PRD"]
        self.assertEqual(float(bins[0]["PRD_mean"]), float(official[[0, 6]].astype(np.float64).mean()))

    def test_unmanaged_result_and_missing_artifact_inventory_rejected(self):
        manifests = [self.job(nv) for nv in (1, 2)]
        manifest = json.loads(manifests[0].read_text())
        manifest["artifacts_sha256"].pop(str(result_path(self.root / "baseline_v7", BASELINE_MODEL, 3407, 1).relative_to(self.root)))
        manifests[0].write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "not covered"):
            self.collect()

    def test_selected_baseline_or_one_bad_metric_is_rejected(self):
        manifests = [self.job(nv) for nv in (1, 2)]
        path = manifests[0].parent / "checkpoint/validation_selection.json"
        original = json.loads(path.read_text())
        for kind in ("epoch_zero", "bad_mad", "incomplete"):
            selection = json.loads(json.dumps(original))
            if kind == "epoch_zero":
                selection["selected_checkpoint"]["epoch"] = 0
            elif kind == "bad_mad":
                selection["selected_checkpoint"]["metrics"]["val_mad"] = .6
            else:
                selection["complete"] = False
            path.write_text(json.dumps(selection))
            manifest = json.loads(manifests[0].read_text())
            manifest["artifacts_sha256"][str(path.relative_to(self.root))] = file_digest(path)
            manifests[0].write_text(json.dumps(manifest))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.collect()

    def test_validation_summary_retains_last_metrics_when_no_candidate_exists(self):
        manifest_path = self.job(1, "no_feasible_checkpoint")
        self.job(2, "no_feasible_checkpoint")
        history = manifest_path.parent / "checkpoint/loss_history.csv"
        history.write_text("epoch,val_ssd,val_mad,val_prd,val_cossim,val_stage1_ssd,train_loss\n"
                           "1,1.5,0.6,24,0.91,2.5,0.01\n")
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts_sha256"][str(history.relative_to(self.root))] = file_digest(history)
        manifest_path.write_text(json.dumps(manifest))
        self.collect()
        summary = rows(self.root / "analysis/validation_ablation_summary.csv")
        row = next(r for r in summary if r["nv"] == "1")
        self.assertEqual(row["status"], "no_feasible_checkpoint")
        self.assertEqual(float(row["last_mad"]), .6)
        self.assertAlmostEqual(float(row["last_delta_mad"]), .1)
        self.assertEqual(row["selected_epoch"], "")

    def test_per_nv_robustness_available_without_other_nv_and_keeps_scope_prd(self):
        self.job(1)
        alpha = self.work / "alpha1.npy"
        np.save(alpha, [.3, .6, .8, 1.0, 1.2, 1.8])
        self.collect(amplitude_files={1: alpha})
        table = rows(self.root / "analysis/robustness_comparison__qtdb__nv1.csv")
        self.assertEqual([int(r["PRD_count"]) for r in table], [1, 1, 1, 1])
        clean, pred = (item.squeeze(-1) for item in self.payloads[1])
        self.assertEqual(float(table[0]["PRD_mean"]), float(metric_values(clean, pred)["PRD"][0]))
        self.assertEqual(rows(self.root / "analysis/robustness_comparison__qtdb__nv2.csv"), [])
        self.assertEqual(rows(self.root / "analysis/robustness_comparison__official_nv1_nv2.csv"), [])
        # Removing amplitude input must clear old bins, not retain stale scores.
        self.collect()
        self.assertEqual(rows(self.root / "analysis/robustness_comparison__qtdb__nv1.csv"), [])

    def test_per_nv_comparison_does_not_hide_missing_candidate(self):
        self.job(1)
        self.job(2, "no_feasible_checkpoint")
        report = self.collect()["reports"]["lstm_v16"]
        self.assertEqual(report["per_nv_diagnostic"]["status"], "no_feasible_checkpoint")
        table = rows(self.root / "analysis/per_nv_comparison_summary.csv")
        self.assertEqual(table[0]["comparison_status"], "passed")
        self.assertEqual(table[1]["job_status"], "no_feasible_checkpoint")
        self.assertEqual(table[1]["comparison_status"], "unavailable")
        self.assertEqual(table[1]["SSD"], "")

    def test_managed_epoch_diagnostics_and_gradients_survive_infeasible_outcome(self):
        manifest = self.job(1, "no_feasible_checkpoint")
        self.job(2, "no_feasible_checkpoint")
        self.managed_write(manifest, "checkpoint/loss_history.csv",
                           "epoch,val_ssd,val_mad,val_prd,val_cossim,lr_adapter,grad_norm_adapter,frozen_parameters_sha256\n"
                           "1,2.1,.6,26,.89,0.0001,0.75,abc123\n")
        self.managed_write(manifest, "checkpoint/validation/epoch_001_diagnostics.json", json.dumps({
            "metrics": {"ssd": 2.1}, "diagnostics": {"alignment_cosine": {"mean": .4, "count": 6}},
            "waveform_indices": [0, 1, 2], "count": 6}))
        self.collect()
        summary = rows(self.root / "analysis/validation_ablation_summary.csv")
        first = next(row for row in summary if row["nv"] == "1")
        self.assertEqual(first["last_history_frozen_parameters_sha256"], "abc123")
        self.assertEqual(float(first["last_history_grad_norm_adapter"]), .75)
        diagnostics = rows(self.root / "analysis/validation_diagnostics_summary.csv")
        alignment = next(row for row in diagnostics if row["diagnostic"] == "diagnostics.alignment_cosine.mean")
        self.assertEqual(float(alignment["value"]), .4)
        self.assertEqual(alignment["status"], "no_feasible_checkpoint")
        self.assertEqual(alignment["is_selected"], "False")
        self.assertFalse(any("waveform_indices" in row["diagnostic"] for row in diagnostics))
        output_manifest = json.loads((self.root / "analysis/report_manifest.json").read_text())
        for name in ("validation_diagnostics_summary.csv", "collection_provenance.json"):
            self.assertEqual(output_manifest["artifacts_sha256"]["analysis/" + name],
                             file_digest(self.root / "analysis" / name))

    def test_last_validation_contrast_includes_infeasible_without_test_fallback(self):
        for model, offset in (("v16_condition_gate", 0), ("v16", 1)):
            for nv in (1, 2):
                manifest = self.job(nv, "no_feasible_checkpoint", model=model)
                self.managed_write(manifest, "checkpoint/loss_history.csv",
                                   "epoch,val_ssd,val_mad,val_prd,val_cossim\n"
                                   f"1,{2 + offset},{.5 + offset / 10},{25 + offset},{.9 - offset / 100}\n")
        self.collect(expected_models=["lstm_v16", "lstm_v16_condition_gate"])
        table = rows(self.root / "analysis/ablation_contrasts.csv")
        matching = [row for row in table if row["contrast"] == "correction_features_frozen_flow_C_minus_B"
                    and row["noise_version"] == "nv1"]
        last = next(row for row in matching if row["phase"] == "last_validation")
        test = next(row for row in matching if row["phase"] == "test")
        self.assertEqual(last["status"], "available")
        self.assertEqual(float(last["SSD_delta"]), 1.)
        self.assertAlmostEqual(float(last["CosSim_delta"]), -.01)
        self.assertEqual(test["status"], "unavailable")
        self.assertEqual(test["SSD_delta"], "")
        self.assertIn("lstm_v16", test["missing_models"])

    def test_all_five_variants_and_matched_interaction(self):
        models = ["v16_constant_gate", "v16_condition_gate", "v16", "v16_condition_gate_flow", "v16_flow"]
        errors = dict(zip(models, [.02, .03, .01, .04, .02]))
        for model in models:
            for nv in (1, 2):
                manifest_path = self.job(nv, model=model)
                path = result_path(self.root, "lstm_" + model, 3407, nv)
                payload = pickle.loads(path.read_bytes())
                payload[2] = payload[1] + np.float32(errors[model])
                path.write_bytes(pickle.dumps(payload))
                manifest = json.loads(manifest_path.read_text())
                manifest["artifacts_sha256"][str(path.relative_to(self.root))] = file_digest(path)
                manifest_path.write_text(json.dumps(manifest))
        result = self.collect(expected_models=["lstm_" + model for model in models])
        self.assertEqual(len(result["reports"]), 5)
        table = {row["model"]: row for row in rows(self.root / "analysis/table1_comparison__official_nv1_nv2.csv")}
        expected = (float(table["lstm_v16_flow"]["SSD_mean"]) - float(table["lstm_v16_condition_gate_flow"]["SSD_mean"])
                    - float(table["lstm_v16"]["SSD_mean"]) + float(table["lstm_v16_condition_gate"]["SSD_mean"]))
        contrast = next(row for row in rows(self.root / "analysis/ablation_contrasts.csv")
                        if row["contrast"] == "interaction_E_minus_D_minus_C_plus_B"
                        and row["phase"] == "test" and row["noise_version"] == "nv1_nv2")
        self.assertEqual(contrast["status"], "available")
        self.assertAlmostEqual(float(contrast["SSD_delta"]), expected)
        self.assertEqual(len(rows(self.root / "baseline_v7/analysis/table1_comparison__official_nv1_nv2.csv")), 1)

    def test_unmanaged_diagnostic_is_not_reported_as_current_evidence(self):
        manifest = self.job(1, "no_feasible_checkpoint")
        self.job(2, "no_feasible_checkpoint")
        path = manifest.parent / "checkpoint/validation/epoch_999_diagnostics.json"
        path.parent.mkdir()
        path.write_text('{"invented": 100}')
        self.collect()
        self.assertEqual(rows(self.root / "analysis/validation_diagnostics_summary.csv"), [])

    def test_reporting_inputs_cannot_be_overwritten_by_collection(self):
        for nv in (1, 2):
            self.job(nv)
        self.collect()
        path = self.root / "analysis/table1_comparison__official_nv1_nv2.csv"
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "regenerated output"):
            collect_results(self.root, baseline_csvs=[path])
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
