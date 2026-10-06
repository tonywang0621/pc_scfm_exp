import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "src/v13_acceptance.py"
SPEC = importlib.util.spec_from_file_location("v13_acceptance", SCRIPT)
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTANCE)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def row(self, model, noise="nv1_nv2", seed=3407, alpha="", passed=False):
        row = {"model": model, "noise_version": noise, "seed": f"seed{seed}", "alpha": alpha,
               "dataset_protocol": "qtdb_robustness_alpha_" + alpha if alpha else "qtdb_train_qtdb_test"}
        for metric, value in zip(ACCEPTANCE.METRICS, (3.0, .3, 35.0, .94)):
            if passed:
                value += .001 if metric == "CosSim" else -.001
            row[metric + "_mean"] = value
            row[metric + "_std"] = .01
            row[metric + "_count"] = (20 if noise == "nv1_nv2" else 10) if not alpha else 2
        return row

    def write(self, filename, rows):
        path = self.root / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def evaluate(self, candidates=None, baselines=None, **kwargs):
        candidates = [self.row("v13", passed=True)] if candidates is None else candidates
        baselines = [self.row("v7")] if baselines is None else baselines
        return ACCEPTANCE.assess([self.write("candidate.csv", candidates)], [self.write("baseline.csv", baselines)],
                                 candidate_model="v13", baseline_model="v7", **kwargs)

    def test_default_requires_only_combined(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["scope"], "combined")
        self.assertEqual(report["required_noise_versions"], ["nv1_nv2"])
        self.assertEqual(report["comparison_count"], 1)
        self.assertEqual(report["issues"], [])

    def test_default_ignores_per_noise_regressions(self):
        candidates = [self.row("v13", nv, passed=nv == "nv1_nv2") for nv in ACCEPTANCE.SCOPES["all"]]
        baselines = [self.row("v7", nv) for nv in ACCEPTANCE.SCOPES["all"]]
        candidates[0]["SSD_mean"] = 4.0
        report = self.evaluate(candidates, baselines)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["comparison_count"], 1)
        self.assertEqual(self.evaluate(candidates, baselines, scope="all")["status"], "failed")

    def test_explicit_all_requires_both_noise_versions_and_combined(self):
        report = self.evaluate(scope="all")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(sum(i["code"] == "missing_pair" for i in report["issues"]), 2)

    def test_full_scope_matches_all_means_and_counts(self):
        candidates = [self.row("v13", nv, passed=True) for nv in ACCEPTANCE.SCOPES["all"]]
        baselines = [self.row("v7", nv) for nv in ACCEPTANCE.SCOPES["all"]]
        report = self.evaluate(candidates, baselines, scope="all")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["comparison_count"], 3)

    def test_explicit_per_noise_scope_does_not_require_combined(self):
        candidates = [self.row("v13", nv, passed=True) for nv in ACCEPTANCE.SCOPES["per-nv"]]
        baselines = [self.row("v7", nv) for nv in ACCEPTANCE.SCOPES["per-nv"]]
        report = self.evaluate(candidates, baselines, scope="per-nv")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["comparison_count"], 2)

    def test_equal_ssd_prd_allowed_but_mad_and_cosine_strict(self):
        c = self.row("v13", passed=True)
        c["SSD_mean"], c["PRD_mean"] = 3.0, 35.0
        self.assertEqual(self.evaluate([c], scope="combined")["status"], "passed")
        for metric, equal in (("MAD", .3), ("CosSim", .94)):
            with self.subTest(metric=metric):
                changed = dict(c, **{metric + "_mean": equal})
                report = self.evaluate([changed], scope="combined")
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["comparisons"][0]["metrics"][metric]["passed"])

    def test_ssd_regression_not_hidden_by_other_gains(self):
        row = self.row("v13", passed=True)
        row["SSD_mean"] = 3.01
        self.assertEqual(self.evaluate([row], scope="combined")["status"], "failed")

    def test_duplicate_rows_fail_closed_even_if_identical(self):
        row = self.row("v13", passed=True)
        self.assertEqual(self.evaluate([row, row], scope="combined")["status"], "invalid")

    def test_nonfinite_means_invalid_counts_and_protocol_rejected(self):
        for field, value in (("MAD_mean", "nan"), ("CosSim_mean", "inf"), ("SSD_count", 0),
                             ("MAD_count", 19), ("dataset_protocol", "different_test"), ("MAD_std", "nan")):
            with self.subTest(field=field, value=value):
                row = self.row("v13", passed=True)
                row[field] = value
                self.assertEqual(self.evaluate([row], scope="combined")["status"], "invalid")

    def test_matched_count_difference_invalid(self):
        row = self.row("v13", passed=True)
        for metric in ACCEPTANCE.METRICS:
            row[metric + "_count"] = 19
        report = self.evaluate([row], scope="combined")
        self.assertEqual(report["status"], "invalid")
        self.assertIn("comparison_mismatch", [i["code"] for i in report["issues"]])

    def test_seeds_cannot_be_dropped_or_averaged_to_hide_regression(self):
        candidates = [self.row("v13", seed=1, passed=True), self.row("v13", seed=2)]
        baselines = [self.row("v7", seed=1), self.row("v7", seed=2)]
        self.assertEqual(self.evaluate(candidates, baselines, scope="combined")["status"], "failed")
        self.assertEqual(self.evaluate(candidates[:1], baselines, scope="combined")["status"], "invalid")

    def test_robustness_requires_every_bin_and_accepts_legacy_bin_names(self):
        candidate_bins = [self.row("v13", alpha=alpha, passed=True) for alpha in ACCEPTANCE.BINS]
        baseline_bins = [self.row("v7", alpha=alpha) for alpha in ACCEPTANCE.BINS]
        for row in candidate_bins:
            row["alpha"] = row["alpha"].replace(".", "p").replace("-", "_")
            row["dataset_protocol"] = "qtdb_robustness_alpha_" + row["alpha"]
        c_path, b_path = self.write("cr.csv", candidate_bins), self.write("br.csv", baseline_bins)
        kwargs = {"scope": "combined", "require_robustness": True,
                  "candidate_robustness_csvs": [c_path], "baseline_robustness_csvs": [b_path]}
        self.assertEqual(self.evaluate(**kwargs)["status"], "passed")
        self.write("cr.csv", candidate_bins[:-1])
        self.assertEqual(self.evaluate(**kwargs)["status"], "incomplete")

    def test_robustness_failure_cannot_be_hidden_by_pooled_pass(self):
        c_rows = [self.row("v13", alpha=a, passed=True) for a in ACCEPTANCE.BINS]
        b_rows = [self.row("v7", alpha=a) for a in ACCEPTANCE.BINS]
        c_rows[0]["CosSim_mean"] = .939
        report = self.evaluate(scope="combined", require_robustness=True,
                               candidate_robustness_csvs=[self.write("cr.csv", c_rows)],
                               baseline_robustness_csvs=[self.write("br.csv", b_rows)])
        self.assertEqual(report["status"], "failed")

    def test_combined_count_must_equal_per_noise_counts(self):
        c_rows = [self.row("v13", nv, passed=True) for nv in ACCEPTANCE.SCOPES["all"]]
        b_rows = [self.row("v7", nv) for nv in ACCEPTANCE.SCOPES["all"]]
        for rows in (c_rows, b_rows):
            for metric in ACCEPTANCE.METRICS:
                rows[2][metric + "_count"] = 21
        self.assertEqual(self.evaluate(c_rows, b_rows, scope="all")["status"], "invalid")

    def test_missing_experiment_is_incomplete_and_serializable(self):
        report = ACCEPTANCE.assess([self.root / "not_yet_trained.csv"], [self.write("b.csv", [self.row("v7")])],
                                   candidate_model="v13", baseline_model="v7")
        self.assertEqual(report["status"], "incomplete")
        json.dumps(report, allow_nan=False)

    def test_cli_defaults_to_combined_writes_report_and_preserves_input_bytes(self):
        c = self.write("c.csv", [self.row("v13", passed=True)])
        b = self.write("b.csv", [self.row("v7")])
        original = c.read_bytes(), b.read_bytes()
        command = [sys.executable, str(SCRIPT), "--candidate-csv", str(c), "--baseline-csv", str(b),
                   "--candidate-model", "v13", "--baseline-model", "v7"]
        report_path = self.root / "report.json"
        process = subprocess.run(command + ["--output-json", str(report_path)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["scope"], "combined")
        self.assertEqual(report["required_noise_versions"], ["nv1_nv2"])
        self.assertEqual(report["comparison_count"], 1)
        self.assertEqual(report["issues"], [])
        bad = subprocess.run(command + ["--output-json", str(c)], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual((c.read_bytes(), b.read_bytes()), original)


if __name__ == "__main__":
    unittest.main()
