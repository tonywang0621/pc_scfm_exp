"""Matched baseline acceptance and separate v14 combined-cosine stretch target."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import v14_acceptance as acceptance


class V14AcceptanceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def row(self, candidate=False, *, seed=3407, nv="nv1_nv2", cosine=None, alpha=""):
        row = {"model": "lstm_v14" if candidate else acceptance.BASELINE_MODEL,
               "noise_version": nv, "seed": f"seed{seed}", "alpha": alpha,
               "dataset_protocol": "qtdb_robustness_alpha_" + alpha if alpha else "qtdb_train_qtdb_test"}
        values = (2.9, .29, 34.9, .945) if candidate else (3.0, .3, 35.0, .94)
        for metric, value in zip(acceptance.METRICS, values):
            row[metric + "_mean"] = cosine if metric == "CosSim" and cosine is not None else value
            row[metric + "_std"] = .01
            row[metric + "_count"] = 2 if alpha else (20 if nv == "nv1_nv2" else 10)
        return row

    def write(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.row()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def assess(self, candidates=None, baselines=None, **kwargs):
        candidates = [self.row(True)] if candidates is None else candidates
        baselines = [self.row()] if baselines is None else baselines
        return acceptance.assess([self.write("c.csv", candidates)], [self.write("b.csv", baselines)], **kwargs)

    def test_default_combined_exact_threshold_passes_both(self):
        report = self.assess()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["cosine_target"]["status"], "passed")
        self.assertEqual(report["cosine_target"]["threshold"], .945)
        self.assertEqual(len(report["cosine_target"]["per_pair"]), 1)
        self.assertTrue(report["target_achieved"])

    def test_basic_pass_does_not_claim_stretch(self):
        report = self.assess([self.row(True, cosine=.944)])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["cosine_target"]["status"], "failed")
        self.assertFalse(report["target_achieved"])

    def test_stretch_alone_does_not_hide_ssd_regression(self):
        candidate = self.row(True, cosine=.95)
        candidate["SSD_mean"] = 3.1
        report = self.assess([candidate])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(report["cosine_target"]["status"], "passed")
        self.assertFalse(report["target_achieved"])

    def test_original_metric_inequalities_remain_strict_where_required(self):
        candidate = self.row(True)
        candidate.update(SSD_mean=3.0, PRD_mean=35.0)
        self.assertEqual(self.assess([candidate])["status"], "passed")
        for metric, value in (("MAD", .3), ("CosSim", .94)):
            with self.subTest(metric=metric):
                row = dict(candidate, **{metric + "_mean": value})
                self.assertEqual(self.assess([row])["status"], "failed")

    def test_missing_candidate_file_or_named_model_is_incomplete(self):
        baseline = self.write("b.csv", [self.row()])
        for paths in ([self.root / "pending.csv"], [self.write("other.csv", [self.row()])]):
            report = acceptance.assess(paths, [baseline])
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["cosine_target"]["status"], "incomplete")
            self.assertFalse(report["target_achieved"])
            json.dumps(report, allow_nan=False)

    def test_invalid_threshold_is_serializable_without_rewriting_basic_status(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1.01, None, True):
            with self.subTest(value=value):
                report = self.assess(cosine_target=value)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["cosine_target"]["status"], "invalid")
                self.assertIsNone(report["cosine_target"]["threshold"])
                self.assertFalse(report["target_achieved"])
                json.dumps(report, allow_nan=False)

    def test_seed_mismatch_invalidates_stretch(self):
        report = self.assess([self.row(True, seed=1)], [self.row(seed=3407)])
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["cosine_target"]["status"], "invalid")
        self.assertFalse(report["target_achieved"])

    def test_each_seed_must_reach_target_instead_of_averaging(self):
        candidates = [self.row(True, seed=1, cosine=.96), self.row(True, seed=2, cosine=.944)]
        baselines = [self.row(seed=1), self.row(seed=2)]
        report = self.assess(candidates, baselines)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["cosine_target"]["status"], "failed")
        self.assertEqual(report["cosine_target"]["passed_comparison_count"], 1)
        self.assertFalse(report["target_achieved"])
        candidates[1]["CosSim_mean"] = .945
        self.assertTrue(self.assess(candidates, baselines)["target_achieved"])

    def test_validation_count_protocol_and_nonfinite_rules_are_reused(self):
        for field, value in (("SSD_count", 21), ("dataset_protocol", "other"),
                             ("CosSim_mean", "nan"), ("MAD_std", "inf")):
            with self.subTest(field=field):
                candidate = dict(self.row(True), **{field: value})
                report = self.assess([candidate])
                self.assertEqual(report["status"], "invalid")
                self.assertEqual(report["cosine_target"]["status"], "invalid")
                self.assertFalse(report["target_achieved"])

    def test_all_scope_target_does_not_apply_to_individual_noise_versions(self):
        candidates = [self.row(True, nv=nv, cosine=.945 if nv == "nv1_nv2" else .941)
                      for nv in acceptance.SCOPES["all"]]
        baselines = [self.row(nv=nv) for nv in acceptance.SCOPES["all"]]
        report = self.assess(candidates, baselines, scope="all")
        self.assertEqual(report["comparison_count"], 3)
        self.assertEqual(report["cosine_target"]["comparison_count"], 1)
        self.assertTrue(report["target_achieved"])

    def test_per_nv_basic_scope_keeps_combined_target_incomplete(self):
        candidates = [self.row(True, nv=nv) for nv in acceptance.SCOPES["per-nv"]]
        baselines = [self.row(nv=nv) for nv in acceptance.SCOPES["per-nv"]]
        report = self.assess(candidates, baselines, scope="per-nv")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["cosine_target"]["status"], "incomplete")
        self.assertFalse(report["target_achieved"])

    def test_robustness_bins_have_no_stretch_threshold_but_must_pass_basic_rules(self):
        c_bins = [self.row(True, alpha=alpha, cosine=.941) for alpha in acceptance.BINS]
        b_bins = [self.row(alpha=alpha) for alpha in acceptance.BINS]
        kwargs = dict(require_robustness=True, candidate_robustness_csvs=[self.write("cr.csv", c_bins)],
                      baseline_robustness_csvs=[self.write("br.csv", b_bins)])
        report = self.assess(**kwargs)
        self.assertTrue(report["target_achieved"])
        self.assertEqual(report["cosine_target"]["comparison_count"], 1)
        c_bins[0]["MAD_mean"] = .31
        self.write("cr.csv", c_bins)
        report = self.assess(**kwargs)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cosine_target"]["status"], "passed")
        self.assertFalse(report["target_achieved"])

    def test_cli_defaults_models_preserves_inputs_and_rejects_output_alias(self):
        candidate, baseline = self.write("c.csv", [self.row(True)]), self.write("b.csv", [self.row()])
        snapshots = candidate.read_bytes(), baseline.read_bytes()
        command = [sys.executable, str(PROJECT / "src/v14_acceptance.py"),
                   "--candidate-csv", str(candidate), "--baseline-csv", str(baseline)]
        output = self.root / "report.json"
        process = subprocess.run(command + ["--output-json", str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(json.loads(output.read_text())["target_achieved"])
        alias = self.root / "alias.csv"
        os.link(candidate, alias)
        for path in (candidate, baseline, alias):
            process = subprocess.run(command + ["--output-json", str(path)], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
        process = subprocess.run(command + ["--cosine-target", "nan", "--output-json", str(output)],
                                 capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual((candidate.read_bytes(), baseline.read_bytes()), snapshots)


if __name__ == "__main__":
    unittest.main()
