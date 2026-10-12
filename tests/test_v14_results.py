"""Official report integration, acceptance isolation, and stale-evidence guards."""

import contextlib
import csv
import hashlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from mecge_table1_collect_official_protocol import load_result, metric_fields, metric_values
from v14_acceptance import BASELINE_MODEL, DEFAULT_BASELINE
from v14_results import collect_results, write_acceptance


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class V14ResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.root = self.work / "runs"

    def result(self, nv, secondary=False, scale=1.0):
        root = self.root / "validation_ssd" if secondary else self.root
        path = root / "official_results" / f"lstm_v14__qtdb_train_qtdb_test__nv{nv}__seed3407.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(nv)
        clean = (rng.normal(size=(5 + nv, 12, 1)) + nv).astype(np.float32)
        prediction = clean + (rng.normal(size=clean.shape) * 0.01 * scale).astype(np.float32)
        with path.open("wb") as handle:
            pickle.dump([clean + 0.5, clean, prediction], handle)
        return path

    def baseline(self, count=13):
        row = rows(DEFAULT_BASELINE)[0]
        row.update(SSD_mean="1.0", MAD_mean="0.5", PRD_mean="50.0", CosSim_mean="0.9")
        for metric in ("SSD", "MAD", "PRD", "CosSim"):
            row[metric + "_count"] = str(count)
        path = self.work / "baseline.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def collect(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            collect_results(self.root, baseline_csvs=[self.baseline()], expected_models=["lstm_v14"],
                            expected_seeds=[3407], **kwargs)

    def test_both_selectors_official_combined_metrics_and_primary_acceptance(self):
        paths = [self.result(nv) for nv in (1, 2)]
        for nv in (1, 2):
            self.result(nv, secondary=True, scale=2.0)
        self.collect()
        for root, label in ((self.root, "best_val_score"), (self.root / "validation_ssd", "best_val_ssd")):
            report = json.loads((root / "analysis/v14_acceptance.json").read_text())
            self.assertEqual(report["checkpoint_selector"], label)
            candidate = report["reports"]["lstm_v14"]
            self.assertEqual(candidate["status"], "passed")
            self.assertTrue(candidate["target_achieved"])
            self.assertEqual(candidate["cosine_target"]["status"], "passed")
            self.assertFalse(candidate["robustness_diagnostic"]["status"] == "passed")
            manifest = json.loads((root / "analysis/report_manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_selector"], label)
            self.assertEqual(report["provenance"]["status"], "unverified")
            self.assertFalse(candidate["provenance_verified"])
            for artifact in ("analysis/v14_acceptance.json", "analysis/v14_acceptance_summary.csv",
                             "analysis/collection_provenance.json"):
                self.assertIn(artifact, manifest["regenerated"])
                self.assertNotIn(artifact, manifest["retained_not_regenerated"])
            summary = rows(root / "analysis/v14_acceptance_summary.csv")
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["seed"], "3407")
            self.assertNotEqual(summary[0]["CosSim_delta_vs_v7"], "")
        loaded = [load_result(path) for path in paths]
        expected = {}
        metric_fields(expected, metric_values(np.concatenate([x[0] for x in loaded]),
                                             np.concatenate([x[1] for x in loaded])))
        actual = rows(self.root / "analysis/table1_comparison__official_nv1_nv2.csv")[0]
        for field, value in expected.items():
            self.assertEqual(float(actual[field]), value)

    def test_missing_requested_variant_seed_and_selector_cannot_pass(self):
        for nv in (1, 2):
            self.result(nv)
        with contextlib.redirect_stdout(io.StringIO()):
            collect_results(self.root, baseline_csvs=[self.baseline()],
                            expected_models=["lstm_v14", "lstm_v14_no_relative"],
                            expected_seeds=[3407, "seed42"])
        report = json.loads((self.root / "analysis/v14_acceptance.json").read_text())
        candidate = report["reports"]["lstm_v14"]
        self.assertEqual(candidate["status"], "incomplete")
        self.assertFalse(candidate["target_achieved"])
        self.assertTrue(any(issue["code"] == "requested_seed_missing" for issue in candidate["issues"]))
        self.assertFalse(report["reports"]["lstm_v14_no_relative"]["status"] == "passed")
        secondary = json.loads((self.root / "validation_ssd/analysis/v14_acceptance.json").read_text())
        self.assertFalse(secondary["reports"]["lstm_v14"]["status"] == "passed")

    def test_stale_table_not_consumed_for_acceptance(self):
        for nv in (1, 2):
            self.result(nv)
        self.collect()
        table = self.root / "analysis/table1_comparison__official_nv1_nv2.csv"
        original = table.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            reports = write_acceptance(self.root, "best_val_score", ["lstm_v14"], [3407],
                                       regenerated=set(), baseline_csvs=[self.baseline()],
                                       baseline_robustness_csvs=[], cosine_target=0.945)
        self.assertEqual(reports["lstm_v14"]["status"], "incomplete")
        self.assertFalse(reports["lstm_v14"]["target_achieved"])
        self.assertEqual(table.read_bytes(), original)

    def test_snapshot_hashes_and_source_identity(self):
        manifest = json.loads(DEFAULT_BASELINE.with_name("v14_baseline_provenance.json").read_text())
        self.assertEqual(manifest["baseline_model"], BASELINE_MODEL)
        for source in manifest["sources"]:
            path = DEFAULT_BASELINE.with_name(source["snapshot"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["snapshot_sha256"])
            self.assertEqual(len(rows(path)), source["rows"])
            self.assertTrue(all(row["model"] == BASELINE_MODEL and row["seed"] == "seed3407" for row in rows(path)))

    def managed_results(self):
        manifests = []
        for nv in (1, 2):
            job = self.root / "v14" / f"nv{nv}" / "seed3407"
            checkpoint = job / "checkpoint"
            checkpoint.mkdir(parents=True)
            for name in ("best_model.pt", "best_val_ssd.pt"):
                (checkpoint / name).write_bytes(f"{nv}/{name}".encode())
            artifacts = [self.result(nv), self.result(nv, secondary=True),
                         checkpoint / "best_model.pt", checkpoint / "best_val_ssd.pt"]
            manifest = {"complete": True, "identity": {"variant": "v14", "seed": 3407, "nv": nv},
                        "artifacts_sha256": {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in artifacts}}
            target = job / "manifest.json"
            target.write_text(json.dumps(manifest))
            manifests.append(target)
        return manifests

    def old_pass(self):
        for root in (self.root, self.root / "validation_ssd"):
            analysis = root / "analysis"
            analysis.mkdir(parents=True, exist_ok=True)
            (analysis / "v14_acceptance.json").write_text(json.dumps({
                "reports": {"lstm_v14": {"status": "passed", "target_achieved": True}}}))
            (analysis / "v14_acceptance_summary.csv").write_text("basic_status\npassed\n")
            (analysis / "table1_comparison__official_nv1_nv2.csv").write_text("old table must not be scored\n")

    def assert_unavailable(self, status):
        for root in (self.root, self.root / "validation_ssd"):
            report = json.loads((root / "analysis/v14_acceptance.json").read_text())
            candidate = report["reports"]["lstm_v14"]
            self.assertEqual(candidate["status"], status)
            self.assertFalse(candidate["target_achieved"])
            self.assertEqual(rows(root / "analysis/v14_acceptance_summary.csv")[0]["basic_status"], status)
            manifest = json.loads((root / "analysis/report_manifest.json").read_text())
            self.assertNotIn("analysis/table1_comparison__official_nv1_nv2.csv", manifest["regenerated"])

    def test_managed_inputs_checkpoint_amplitude_and_reporting_sources_have_hashes(self):
        self.managed_results()
        amplitudes = {}
        for nv in (1, 2):
            path = self.work / f"alpha_nv{nv}.npy"
            np.save(path, np.linspace(.25, 1.9, 5 + nv))
            amplitudes[nv] = path
        self.collect(amplitude_files=amplitudes)
        for root in (self.root, self.root / "validation_ssd"):
            report = json.loads((root / "analysis/v14_acceptance.json").read_text())
            provenance = report["provenance"]
            self.assertEqual(provenance["status"], "verified")
            self.assertTrue(provenance["verified"])
            self.assertTrue(report["reports"]["lstm_v14"]["provenance_verified"])
            self.assertEqual(len(provenance["inputs"]), 2)
            for item in provenance["inputs"]:
                self.assertEqual(item["status"], "verified")
                for field in ("result", "selected_checkpoint", "manifest"):
                    self.assertEqual(item[field + "_sha256"], hashlib.sha256(Path(item[field]).read_bytes()).hexdigest())
            self.assertEqual(len(provenance["amplitude_inputs"]), 2)
            for item in provenance["amplitude_inputs"]:
                self.assertEqual(item["sha256"], hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest())
            self.assertIn("utils_ecg.py", provenance["reporting_sources"])
            self.assertIn("v14_results.py", provenance["reporting_sources"])

    def test_managed_tamper_incomplete_or_missing_artifacts_invalidates_both_old_passes(self):
        for problem in ("result", "checkpoint", "incomplete", "missing", "bad_json"):
            with self.subTest(problem=problem):
                self.root = self.work / problem
                manifests = self.managed_results()
                self.old_pass()
                if problem == "result":
                    path = next((self.root / "official_results").glob("*.pkl"))
                    path.write_bytes(path.read_bytes() + b"tampered")
                elif problem == "checkpoint":
                    (manifests[0].parent / "checkpoint/best_val_ssd.pt").write_bytes(b"changed checkpoint")
                elif problem == "incomplete":
                    content = json.loads(manifests[0].read_text())
                    content["complete"] = False
                    manifests[0].write_text(json.dumps(content))
                elif problem == "missing":
                    (manifests[0].parent / "checkpoint/best_model.pt").unlink()
                else:
                    manifests[0].write_text("not JSON")
                with patch("v14_results.collect_selector") as collector:
                    with self.assertRaisesRegex(RuntimeError, "provenance verification failed"):
                        self.collect()
                    collector.assert_not_called()
                self.assert_unavailable("invalid")
                report = json.loads((self.root / "analysis/v14_acceptance.json").read_text())
                self.assertEqual(report["provenance"]["status"], "invalid")
                self.assertFalse(report["provenance"]["verified"])
                self.assertEqual((self.root / "analysis/table1_comparison__official_nv1_nv2.csv").read_text(),
                                 "old table must not be scored\n")

    def test_empty_current_results_with_expected_models_replaces_stale_pass(self):
        self.old_pass()
        self.collect()
        self.assert_unavailable("incomplete")
        report = json.loads((self.root / "analysis/v14_acceptance.json").read_text())
        self.assertEqual(report["provenance"]["status"], "incomplete")
        self.assertEqual(report["provenance"]["inputs"], [])

    def test_empty_results_without_expectations_still_raise_and_clear_previous_pass(self):
        self.old_pass()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
            collect_results(self.root, baseline_csvs=[self.baseline()])
        self.assert_unavailable("incomplete")

    def test_complete_manifest_with_all_result_pkls_missing_is_invalid(self):
        self.managed_results()
        for folder in (self.root / "official_results", self.root / "validation_ssd/official_results"):
            for path in folder.glob("*.pkl"):
                path.unlink()
        self.old_pass()
        with self.assertRaisesRegex(RuntimeError, "Managed artifact is missing"):
            self.collect()
        self.assert_unavailable("invalid")

    def test_reporting_failure_cannot_leave_previous_pass_active(self):
        self.result(1)
        self.result(2)
        self.old_pass()
        with patch("v14_results.collect_selector", side_effect=RuntimeError("report subprocess failed")):
            with self.assertRaisesRegex(RuntimeError, "report subprocess failed"):
                self.collect()
        self.assert_unavailable("incomplete")

    def test_extra_file_cannot_impersonate_a_managed_or_duplicate_result(self):
        self.managed_results()
        path = next((self.root / "official_results").glob("*.pkl"))
        path.with_name(path.stem + "__copy.pkl").write_bytes(path.read_bytes())
        self.old_pass()
        with self.assertRaisesRegex(RuntimeError, "Duplicate result identity|not the artifact recorded"):
            self.collect()
        self.assert_unavailable("invalid")


if __name__ == "__main__":
    unittest.main()
