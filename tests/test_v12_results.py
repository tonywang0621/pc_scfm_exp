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
import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from mecge_table1_collect_official_protocol import load_result, metric_fields, metric_values
from v11_results import collect_results as collect_legacy_reports
from v12_results import collect_results


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ResultReportsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.root = self.work / "runs"

    def result(self, nv, *, selector="best_loss", variant="v12", error_scale=1.0, root=None):
        root = root or self.root
        report_root = root if selector == "best_loss" else root / "validation_ssd"
        path = report_root / "official_results" / f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed3407.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unequal nv sizes and means detect both unweighted averaging and the
        # incorrect per-nv PRD denominator in a putative combined report.
        rng = np.random.default_rng(nv)
        shape = (8 if nv == 1 else 5, 9, 1)
        clean = (rng.normal(size=shape) + 4.0 * nv).astype(np.float32)
        prediction = clean + (rng.normal(size=shape) * (0.1 * nv * error_scale)).astype(np.float32)
        with path.open("wb") as handle:
            pickle.dump([clean + 0.4, clean, prediction], handle)
        config = root / variant / f"nv{nv}" / "seed3407" / "resolved_config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump({"model_name": f"configured_{variant}"}), encoding="utf-8")
        return path

    def amplitudes(self):
        amplitudes = {}
        values = {1: [0.2, 0.3, 0.6, 0.7, 1.0, 1.2, 1.5, 2.2], 2: [0.25, 0.65, 1.25, 1.6, 2.3]}
        for nv, array in values.items():
            amplitudes[nv] = self.work / f"rnd_test_nv{nv}.npy"
            np.save(amplitudes[nv], np.asarray(array, dtype=np.float64))
        return amplitudes

    def collect(self, amplitudes=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            collect_results(self.root, amplitudes)
        return output.getvalue()

    def assert_same_metrics(self, actual, reference):
        actual = [{key: value for key, value in row.items() if key != "metrics_file"} for row in actual]
        reference = [{key: value for key, value in row.items() if key != "metrics_file"} for row in reference]
        self.assertEqual(actual, reference)

    def test_full_reports_match_unchanged_legacy_collectors(self):
        reference = self.work / "legacy_reference"
        for nv in (1, 2):
            self.result(nv)
            self.result(nv, root=reference)
        amplitudes = self.amplitudes()
        self.collect(amplitudes)
        collect_legacy_reports(reference, amplitudes)
        # Includes official bins (strict edge exclusions and >1.5 final bin),
        # per-noise tables, per-noise robustness tables, and combined tables.
        actual_tables = sorted(path.name for path in (self.root / "analysis").glob("*.csv"))
        expected_tables = sorted(path.name for path in (reference / "analysis").glob("*.csv"))
        self.assertEqual(actual_tables, expected_tables)
        self.assertEqual(len(actual_tables), 8)
        for name in actual_tables:
            with self.subTest(table=name):
                self.assert_same_metrics(read_rows(self.root / "analysis" / name), read_rows(reference / "analysis" / name))
        for name in ("metrics_per_window.csv", "metrics_summary.csv", "metrics_qtdb_pkl_test.yaml"):
            for actual in self.root.glob(f"lstm_v12/results/**/{name}"):
                expected = reference / actual.relative_to(self.root)
                self.assertEqual(actual.read_bytes(), expected.read_bytes())

    def test_without_amplitudes_combines_predictions_and_skips_missing_ssd(self):
        paths = [self.result(nv) for nv in (1, 2)]
        output = self.collect()
        self.assertIn("Skipping best_val_ssd reports", output)
        self.assertIn("Skipping combined robustness bins", output)
        table = self.root / "analysis" / "table1_comparison__official_nv1_nv2.csv"
        rows = read_rows(table)
        self.assertEqual(len(rows), 1)
        loaded = [load_result(path) for path in paths]
        expected = {}
        metric_fields(expected, metric_values(np.concatenate([pair[0] for pair in loaded]),
                                             np.concatenate([pair[1] for pair in loaded])))
        for field, value in expected.items():
            self.assertEqual(float(rows[0][field]), value)
        per_nv_prd = [float(np.mean(metric_values(*pair)["PRD"])) for pair in loaded]
        self.assertGreater(abs(float(rows[0]["PRD_mean"]) - np.mean(per_nv_prd)), 0.1)
        self.assertFalse((self.root / "analysis" / "robustness_comparison__official_nv1_nv2.csv").exists())
        self.assertFalse((self.root / "validation_ssd").exists())

    def test_partial_noise_pairs_produce_no_combined_row(self):
        self.result(1)
        # A different variant with nv2 is not the matching nv2 run.
        self.result(2, variant="v12_topk")
        self.result(1, selector="best_val_ssd")
        output = self.collect({1: self.amplitudes()[1]})
        self.assertIn("both result PKLs are required", output)
        for report_root in (self.root, self.root / "validation_ssd"):
            with self.subTest(selector_root=report_root):
                self.assertEqual(read_rows(report_root / "analysis" / "table1_comparison__official_nv1_nv2.csv"), [])
                self.assertEqual(len(read_rows(report_root / "analysis" / "table1_comparison__qtdb_train_qtdb_test__nv1.csv")), 1)
                bins = read_rows(report_root / "analysis" / "robustness_comparison__qtdb__all.csv")
                self.assertEqual(len(bins), 4)
                self.assertTrue(all(row["noise_version"] == "nv1" for row in bins))

    def test_selectors_use_top_root_config_and_never_mix_rows(self):
        for nv in (1, 2):
            self.result(nv)
            self.result(nv, selector="best_val_ssd", error_scale=3.0)
        # A selector-specific variant must not appear in the other selector.
        self.result(1, variant="v12_extra", selector="best_val_ssd")
        self.collect()
        main_rows = read_rows(self.root / "analysis" / "table1_comparison__qtdb_train_qtdb_test__all.csv")
        ssd_root = self.root / "validation_ssd"
        ssd_rows = read_rows(ssd_root / "analysis" / "table1_comparison__qtdb_train_qtdb_test__all.csv")
        self.assertEqual(len(main_rows), 2)
        self.assertEqual(len(ssd_rows), 3)
        for row in main_rows:
            self.assertIn("/configured_v12/best_loss/", row["metrics_file"])
            self.assertNotIn("/validation_ssd/", row["metrics_file"])
        for row in ssd_rows:
            configured_name = "configured_" + row["model"].removeprefix("lstm_")
            self.assertIn(f"/{configured_name}/best_val_ssd/", row["metrics_file"])
            self.assertIn("/validation_ssd/", row["metrics_file"])
        main = read_rows(self.root / "analysis" / "table1_comparison__official_nv1_nv2.csv")
        ssd = read_rows(ssd_root / "analysis" / "table1_comparison__official_nv1_nv2.csv")
        self.assertEqual(len(main), 1)
        self.assertEqual(len(ssd), 1)
        self.assertAlmostEqual(float(ssd[0]["SSD_mean"]) / float(main[0]["SSD_mean"]), 9.0, places=4)

    def test_no_results_raise_a_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "No nv1/nv2 result PKLs"):
            collect_results(self.root)

    def test_recollection_without_amplitudes_preserves_and_marks_old_bins(self):
        for selector in ("best_loss", "best_val_ssd"):
            for nv in (1, 2):
                self.result(nv, selector=selector)
        self.collect(self.amplitudes())
        reports = (self.root, self.root / "validation_ssd")
        old_bins = {}
        for report_root in reports:
            paths = list((report_root / "analysis").glob("robustness*.csv"))
            paths.extend(report_root.glob("*/controlled_tests/**/*.csv"))
            for path in paths:
                old_bins[path] = (path.read_bytes(), path.stat().st_mtime_ns)
        for nv in (1, 2):
            self.result(nv, error_scale=2.0)
        output = self.collect()
        self.assertIn("retained and NOT regenerated", output)
        for path, (content, timestamp) in old_bins.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, timestamp)
        for report_root in reports:
            manifest = json.loads((report_root / "analysis" / "report_manifest.json").read_text())
            self.assertEqual(manifest["robustness_noise_versions"], [])
            self.assertEqual(len(manifest["active_result_names"]), 2)
            retained = manifest["retained_not_regenerated"]
            self.assertIn("analysis/robustness_comparison__official_nv1_nv2.csv", retained)
            self.assertIn("analysis/robustness_comparison__qtdb__all.csv", retained)
            self.assertFalse(any("controlled_tests/" in path for path in manifest["regenerated"]))
        main = read_rows(self.root / "analysis" / "table1_comparison__official_nv1_nv2.csv")[0]
        ssd = read_rows(reports[1] / "analysis" / "table1_comparison__official_nv1_nv2.csv")[0]
        self.assertAlmostEqual(float(main["SSD_mean"]) / float(ssd["SSD_mean"]), 4.0, places=4)

    def test_recollection_filters_removed_results_and_current_amplitude_versions(self):
        self.result(1)
        nv2 = self.result(2)
        removed = self.result(1, variant="v12_removed")
        amplitudes = self.amplitudes()
        self.collect(amplitudes)
        analysis = self.root / "analysis"
        old_nv2_bins = (analysis / "robustness_comparison__qtdb__nv2.csv").read_bytes()
        old_combined_bins = (analysis / "robustness_comparison__official_nv1_nv2.csv").read_bytes()
        removed.unlink()
        self.collect({1: amplitudes[1]})
        tables = read_rows(analysis / "table1_comparison__qtdb_train_qtdb_test__all.csv")
        self.assertEqual(len(tables), 2)
        self.assertEqual({row["model"] for row in tables}, {"lstm_v12"})
        bins = read_rows(analysis / "robustness_comparison__qtdb__all.csv")
        self.assertEqual(len(bins), 4)
        self.assertEqual({(row["model"], row["noise_version"]) for row in bins}, {("lstm_v12", "nv1")})
        self.assertEqual((analysis / "robustness_comparison__qtdb__nv2.csv").read_bytes(), old_nv2_bins)
        self.assertEqual((analysis / "robustness_comparison__official_nv1_nv2.csv").read_bytes(), old_combined_bins)
        manifest = json.loads((analysis / "report_manifest.json").read_text())
        self.assertEqual(manifest["robustness_noise_versions"], ["nv1"])
        self.assertIn("analysis/robustness_comparison__qtdb__all.csv", manifest["regenerated"])
        self.assertIn("analysis/robustness_comparison__qtdb__nv2.csv", manifest["retained_not_regenerated"])

        # Stale source summaries also cannot repopulate a now-empty noise table.
        nv2.unlink()
        self.collect({1: amplitudes[1]})
        self.assertEqual(read_rows(analysis / "table1_comparison__qtdb_train_qtdb_test__nv2.csv"), [])
        self.assertEqual(read_rows(analysis / "table1_comparison__official_nv1_nv2.csv"), [])

    def test_missing_selector_with_old_reports_marks_every_report_retained(self):
        self.result(1)
        ssd = self.result(1, selector="best_val_ssd")
        self.collect()
        ssd.unlink()
        output = self.collect()
        self.assertIn("Skipping best_val_ssd reports", output)
        self.assertIn("retained and NOT regenerated", output)
        manifest = json.loads((self.root / "validation_ssd" / "analysis" / "report_manifest.json").read_text())
        self.assertEqual(manifest["active_result_names"], [])
        self.assertEqual(manifest["regenerated"], [])
        self.assertTrue(manifest["retained_not_regenerated"])


if __name__ == "__main__":
    unittest.main()
