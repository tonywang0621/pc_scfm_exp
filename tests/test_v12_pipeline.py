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
from v12_experiments import APP, CORE, MODULES, VARIANTS, build_config, check_output_root, file_digest, parse_args, select_variants, source_digests


class LauncherTests(unittest.TestCase):
    def run_cli(self, arguments, success=True):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", str(APP / "v12_experiments.py"), *arguments],
                                env=env, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_suite_excludes_original_v7_and_defaults_to_all(self):
        self.assertEqual(len(VARIANTS), 8)
        self.assertNotIn("v7", VARIANTS)
        self.assertEqual(select_variants(parse_args([])), tuple(VARIANTS))
        self.assertEqual(set(CORE) | set(MODULES), set(VARIANTS))
        self.run_cli(["--models", "v7", "--dry-run"], success=False)
        for name in VARIANTS:
            config = build_config(name, 3407, 1, Path("unused.pkl"))
            self.assertEqual(config.model_name, "lstm_temporal_gate_v12_ecg")
            self.assertEqual(config.training.train_epochs, 30)
        with self.assertRaisesRegex(ValueError, "original v7"):
            build_config("v12", 3407, 1, Path("unused.pkl"), ["model.v12_local_gate=false"])

    def test_protected_roots_and_read_only_dry_run(self):
        for name in ("runs/mecge_table1_repro", "runs/v11_performance", "data/new", "src", "."):
            with self.assertRaises(ValueError):
                check_output_root(PROJECT / name)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            result = self.run_cli(["--dry-run", "--run-root", str(output)])
            self.assertIn("8 variants x 2 noise versions x 1 seeds", result.stdout)
            self.assertEqual(result.stdout.count("--ssd-output-pkl"), 16)
            self.assertFalse(output.exists())

    def test_new_model_does_not_change_legacy_v11_source_inventory(self):
        from v11_experiments import source_digests as legacy_digests
        self.assertFalse(any("lstm_v12" in name for name in legacy_digests()))
        sources = source_digests()
        self.assertIn("src/models/lstm_v12/__init__.py", sources)
        self.assertIn("src/mecge_table1_official_result_metrics.py", sources)

    def test_core_training_and_two_selectors_skip_collision_and_collection(self):
        with tempfile.TemporaryDirectory(prefix="v12_pipeline_") as directory:
            root = Path(directory)
            data = root / "data/raw"
            data.mkdir(parents=True)
            rng = np.random.default_rng(13)
            time = np.linspace(0, 1, 512, dtype=np.float32)
            clean = np.sin(12 * np.pi * time)[None, :, None] * np.ones((12, 1, 1), dtype=np.float32)
            clean += rng.normal(0, 0.02, clean.shape).astype(np.float32)
            for nv in (1, 2):
                noisy = clean + (0.1 * nv * np.cos(np.pi * time))[None, :, None]
                with (data / f"dataset_bw_nv{nv}.pkl").open("wb") as handle:
                    pickle.dump([noisy, clean, noisy[:6], clean[:6]], handle)
                np.save(data / f"rnd_test_nv{nv}.npy", np.array([0.3, 0.7, 1.2, 1.7, 0.6, 1.0]))
            output = root / "runs"
            arguments = ["--suite", "core", "--device", "cpu", "--epochs", "1",
                         "--set", "training.batch_size=2", "--data-root", str(root / "data"),
                         "--run-root", str(output),
                         "--set", "model.dense_channel=8", "--set", "model.lstm_hidden=4",
                         "--set", "model.num_tscblocks=1", "--set", "model.cfm_unet_base_channels=8",
                         "--set", "model.cfm_unet_channel_mults=[1,2]", "--set", "model.cfm_unet_time_dim=16"]
            self.run_cli(arguments)
            self.assertFalse((output / "v7").exists())
            manifests = list(output.glob("*/nv*/seed*/manifest.json"))
            self.assertEqual(len(manifests), 6)
            self.assertTrue(all(json.loads(p.read_text())["complete"] for p in manifests))
            for prefix in (output, output / "validation_ssd"):
                with (prefix / "analysis/table1_comparison__official_nv1_nv2.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)
                self.assertEqual({r["model"] for r in rows}, {"lstm_" + name for name in CORE})
                self.assertTrue(all(int(r["SSD_count"]) == 12 for r in rows))
            checkpoint = output / "v12/nv1/seed3407/checkpoint/best_model.pt"
            original_hash = file_digest(checkpoint)
            repeated = self.run_cli(arguments)
            self.assertEqual(repeated.stdout.count("Matching completed run found; skipping."), 6)
            self.assertEqual(file_digest(checkpoint), original_hash)
            rejected = self.run_cli([*arguments, "--set", "training.lr=0.0001"], success=False)
            self.assertIn("Choose a new --run-root", rejected.stderr)
            self.assertEqual(file_digest(checkpoint), original_hash)
            reports = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("analysis/*.csv")}
            self.run_cli(["--collect-only", "--run-root", str(output), "--data-root", str(root / "data")])
            self.assertEqual(reports, {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("analysis/*.csv")})
            self.assertEqual(file_digest(checkpoint), original_hash)
            self.run_cli([*arguments, "--models", "v12"], success=False)  # mutually exclusive with suite
            # --eval-only can regenerate predictions, but cannot bless altered weights.
            with checkpoint.open("ab") as handle:
                handle.write(b"tampered")
            changed = self.run_cli([*arguments, "--eval-only"], success=False)
            self.assertIn("Completed checkpoint changed", changed.stderr)


if __name__ == "__main__":
    unittest.main()
