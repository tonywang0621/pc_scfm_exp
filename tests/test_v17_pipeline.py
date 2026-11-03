import csv
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from v17_experiments import APP, VARIANTS, SUITES, build_config, check_output_root, parse_args, select_variants, file_digest
from v17_results import contrasts, metric_rows
from test_v17_model import SMALL


class PipelineTests(unittest.TestCase):
    def cli(self, arguments, success=True):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", str(APP / "v17_experiments.py"), *arguments],
                                env=env, text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, (result.stdout+result.stderr)[-12000:])
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
        return result

    def test_read_only_dry_run_and_factor_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "absent"
            result = self.cli(["--suite", "all", "--dry-run", "--run-root", str(root)])
            self.assertIn("16 variants x 2 noise versions x 1 seeds", result.stdout)
            self.assertFalse(root.exists())
        self.assertEqual(select_variants(parse_args([])), SUITES["core"])
        self.cli(["--set", "model.v17_flow=false", "--dry-run"], success=False)
        self.cli(["--pkl-file", "one.pkl", "--dry-run"], success=False)
        self.cli(["--set", "training.batch_size=1.5", "--dry-run"], success=False)
        self.cli(["--set", "model.cfm_unet_time_dim=15", "--dry-run"], success=False)
        for root in (".", "data", "runs/v12_performance", "runs/v16_performance", ".agents"):
            with self.assertRaises(ValueError):
                check_output_root(PROJECT / root)
        with self.assertRaises(ValueError):
            check_output_root(Path("/tmp/output"), [Path("/tmp/output/input.pkl")])

    def test_missing_comparison_is_explicit(self):
        result = contrasts([dict(model="v17", seed="seed3407", nv="nv1", SSD=1, MAD=2, PRD=3, CosSim=.9)])
        self.assertTrue(all(row["status"] == "missing_pair" for row in result))
        self.assertFalse(any("SSD_delta" in row for row in result))

    def test_all_variants_both_noise_versions_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix="v17_pipeline_") as directory:
            root = Path(directory)
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            t = np.linspace(0, 1, 512, dtype=np.float32)
            clean = np.sin(12*np.pi*t)[None, :, None]*np.linspace(.8, 1.2, 8, dtype=np.float32)[:, None, None]
            for nv in (1, 2):
                noisy = clean + (.1*nv*np.cos(np.pi*t))[None, :, None]
                with (raw / f"dataset_bw_nv{nv}.pkl").open("wb") as handle:
                    pickle.dump([noisy, clean, noisy[:4], clean[:4]], handle)
                np.save(raw / f"rnd_test_nv{nv}.npy", np.array([.3, .7, 1.2, 1.7]))
            out = root / "runs"
            args = ["--suite", "all", "--device", "cpu", "--epochs", "1", "--data-root", str(root / "data"),
                    "--run-root", str(out), "--set", "training.batch_size=2"]
            for override in SMALL:
                args += ["--set", override]
            self.cli(args)
            manifests = list(out.glob("*/nv*/seed*/manifest.json"))
            self.assertEqual(len(manifests), 32)
            self.assertTrue(all(json.loads(p.read_text())["complete"] for p in manifests))
            for prefix in (out, out / "validation_loss"):
                with (prefix / "analysis/table1_comparison__official_nv1_nv2.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 16)
                self.assertTrue(all(int(row["SSD_count"]) == 8 for row in rows))
                self.assertTrue((prefix / "analysis/robustness_comparison__official_nv1_nv2.csv").is_file())
                with (prefix / "analysis/paired_ablation_contrasts.csv").open(newline="") as handle:
                    pairs = list(csv.DictReader(handle))
                self.assertTrue(all(row["status"] == "available" for row in pairs))
            for path in out.glob("v17*/nv*/seed*/checkpoint/parameter_audit.json"):
                self.assertTrue(json.loads(path.read_text())["passed"])
            checkpoint = out / "v17/nv1/seed3407/checkpoint/best_model.pt"
            digest = file_digest(checkpoint)
            repeated = self.cli(args)
            self.assertEqual(repeated.stdout.count("Matching completed run found"), 32)
            self.assertEqual(file_digest(checkpoint), digest)
            changed = self.cli([*args, "--set", "training.lr=0.0001"], success=False)
            self.assertIn("use a new --run-root", changed.stderr)
            # Collection is independent of the selected training suite and epochs.
            self.cli(["--collect-only", "--run-root", str(out), "--data-root", str(root / "data")])
            self.assertEqual(file_digest(checkpoint), digest)
            manifest = json.loads((out / "analysis/report_manifest.json").read_text())
            self.assertIn("analysis/paired_ablation_contrasts.csv", manifest["regenerated"])
            self.assertNotIn("analysis/paired_ablation_contrasts.csv", manifest["retained_not_regenerated"])
            with checkpoint.open("ab") as handle:
                handle.write(b"tampered")
            changed = self.cli([*args, "--eval-only"], success=False)
            self.assertIn("Completed artifact missing or changed", changed.stderr)


if __name__ == "__main__":
    unittest.main()
