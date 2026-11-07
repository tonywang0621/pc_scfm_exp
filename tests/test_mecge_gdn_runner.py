import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import mecge_gdn_experiments as runner


class RunnerTests(unittest.TestCase):
    def test_default_plan_is_six_fixed_protocol_jobs(self):
        args = runner.parse_args([])
        self.assertEqual(args.models, ["m0", "m1", "m2"])
        commands = [runner.training_command(args, model, nv) for model in args.models for nv in runner.versions(args)]
        self.assertEqual(len(commands), 6)
        for cmd in commands:
            for key, value in (("--max-epochs", "30"), ("--early-stopping-patience", "none"), ("--n-type", "bw")):
                self.assertEqual(cmd[cmd.index(key) + 1], value)
            self.assertIn("seed3407", cmd[cmd.index("--output-pkl") + 1])

    def test_dry_run_does_not_write_or_launch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "run") as run:
            root = Path(tmp) / "output"
            with contextlib.redirect_stdout(io.StringIO()) as out:
                runner.main(["--dry-run", "--run-root", str(root)])
            self.assertEqual(out.getvalue().count("--backbone"), 6)
            self.assertFalse(root.exists())
            run.assert_not_called()

    def test_config_retains_official_model_and_training_values(self):
        import yaml
        args = runner.parse_args([])
        original = yaml.safe_load((args.mecge_dir / "config/MECGE_phase.yaml").read_text())
        for variant in runner.VARIANTS:
            config = runner.resolved_config(args, variant)
            self.assertEqual(config["model"], original["model"])
            self.assertEqual(config["train"]["batch_size"], 96)
            self.assertEqual(config["train"]["lr"], 1e-4)
            self.assertIsNone(config["train"]["early_stopping_patience_epochs"])

    def test_no_ambiguous_shared_dataset_for_two_versions(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--pkl-file", "/tmp/dataset.pkl"])

    def test_protected_output_root_rejected(self):
        with self.assertRaises(ValueError):
            runner.main(["--dry-run", "--run-root", str(PROJECT / "references")])

    def test_manifest_rejects_changed_inputs_and_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(["--run-root", tmp + "/run", "--data-root", tmp + "/data", "--nv", "1"])
            data = runner.dataset_path(args, 1)
            data.parent.mkdir(parents=True)
            data.write_bytes(b"dataset")
            amplitude = args.data_root / "raw/rnd_test_nv1.npy"
            amplitude.write_bytes(b"amplitude")
            p, config, identity, skip = runner.prepare_job(args, "m0", 1, {})
            self.assertFalse(skip)
            p["checkpoint"].mkdir(parents=True)
            p["result"].parent.mkdir(parents=True)
            artifacts = [p["result"], p["checkpoint"] / "best_model.pt"]
            for artifact in artifacts:
                artifact.write_bytes(b"artifact")
            runner.atomic_json(p["manifest"], dict(identity=identity, complete=True,
                artifacts={str(a): runner.digest(a) for a in artifacts}))
            self.assertTrue(runner.prepare_job(args, "m0", 1, {})[-1])
            data.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                runner.prepare_job(args, "m0", 1, {})
            data.write_bytes(b"dataset")
            artifacts[0].write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "artifact"):
                runner.prepare_job(args, "m0", 1, {})

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for result integration")
    def test_existing_collectors_on_six_synthetic_results(self):
        import numpy as np
        import pickle
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(["--run-root", tmp + "/run", "--data-root", tmp + "/data"])
            raw = args.data_root / "raw"
            raw.mkdir(parents=True)
            # Interior points: one sample in each existing alpha bin.
            for nv in (1, 2):
                np.save(raw / f"rnd_test_nv{nv}.npy", [0.4, 0.8, 1.2, 1.8])
                for variant in runner.VARIANTS:
                    path = runner.paths(args.run_root, variant, nv)["result"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    clean = np.tile(np.array([1., 2., 3., 4.], dtype=np.float32), (4, 1))[:, :, None]
                    with path.open("wb") as handle:
                        pickle.dump([clean + 0.5, clean, clean.copy()], handle)
            runner.collect(args)
            analysis = args.run_root / "analysis"
            with (analysis / "table1_comparison__official_nv1_nv2.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            for row in rows:
                for metric in ("SSD", "PRD", "MAD"):
                    self.assertEqual(float(row[metric + "_mean"]), 0.)
                self.assertAlmostEqual(float(row["CosSim_mean"]), 1., places=6)
                self.assertEqual(int(row["SSD_count"]), 8)
            with (analysis / "robustness_comparison__official_nv1_nv2.csv").open() as handle:
                bins = list(csv.DictReader(handle))
            self.assertEqual(len(bins), 12)
            self.assertTrue(all(int(row["SSD_count"]) == 2 for row in bins))


if __name__ == "__main__":
    unittest.main()
