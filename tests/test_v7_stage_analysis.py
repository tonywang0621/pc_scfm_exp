"""Stage extraction, official metric scope, and real-model plot/export checks."""

import csv
import hashlib
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import analyze_v7_stages as analysis
from models import get_model
from utils_ecg import prd_mecge_official


class StageAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = OmegaConf.load(analysis.DEFAULT_CONFIG)
        cls.settings = OmegaConf.to_container(cls.config.model, resolve=True)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def model(self):
        torch.manual_seed(14)
        return get_model(analysis.MODEL_NAME, **self.settings).eval()

    def fixtures(self, clean=True):
        time = np.arange(512, dtype=np.float32) / 512
        target = np.stack([0.3 * np.sin(10 * np.pi * time + i) + 0.02 * i for i in range(4)])
        noisy = target + 0.15 * np.cos(2 * np.pi * time)[None]
        dataset = self.work / "dataset_bw_nv1.pkl"
        with dataset.open("wb") as handle:
            pickle.dump({"X_test": noisy[..., None], "y_test": target[..., None] if clean else None}, handle)
        checkpoint = self.work / "best_model.pt"
        torch.save(self.model().state_dict(), checkpoint)
        return dataset, checkpoint

    def test_stage_extraction_matches_normal_inference_and_preserves_train_flag(self):
        model = self.model().train()
        x = torch.linspace(-0.25, 0.35, 1024).reshape(2, 1, 512)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        waves, difference = analysis.extract_stages(model, x)
        self.assertTrue(model.training)
        self.assertEqual(difference, 0)
        model.eval()
        with torch.no_grad():
            expected = model.denoising(x).numpy()[:, 0]
            coarse, _, _, residual, direct = model.core._restore_components(x)
        np.testing.assert_array_equal(waves["flow_final"], expected)
        np.testing.assert_array_equal(waves["lstm_pre_flow"], coarse.numpy()[:, 0])
        np.testing.assert_array_equal(waves["lstm_pre_flow"], (direct + model.core.residual_refine_scale * residual).numpy()[:, 0])
        np.testing.assert_array_equal(waves["flow_correction"], waves["flow_final"] - waves["lstm_pre_flow"])
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)

    def test_normalization_is_inverted_for_both_stages(self):
        model = self.model()
        model.core.norm = "1"
        x = torch.linspace(-0.2, 0.6, 1024).reshape(2, 1, 512)
        waves, difference = analysis.extract_stages(model, x)
        with torch.no_grad():
            factor = model.core._norm_factor(x)
            coarse = model.core._restore_components(x * factor)[0]
        np.testing.assert_array_equal(waves["lstm_pre_flow"], (coarse / factor).numpy()[:, 0])
        self.assertEqual(difference, 0)
        with self.assertRaisesRegex(ValueError, "normalization"):
            analysis.extract_stages(model, torch.zeros_like(x))

    def test_parity_mismatch_is_not_silently_exported(self):
        model = self.model().train()
        x = torch.ones(1, 1, 512)
        with patch.object(model, "denoising", return_value=torch.full_like(x, 99)), self.assertRaises(AssertionError):
            analysis.extract_stages(model, x)
        self.assertTrue(model.training)

    def test_official_prd_uses_complete_test_mean_even_for_a_subset(self):
        clean = np.array([[1., 2., 3.], [8., 9., 10.], [-3., -2., -1.]], dtype=np.float32)
        prediction = clean + np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
        expected = prd_mecge_official(clean, prediction)
        for indices in ([0], [2, 0], [0, 1, 2]):
            actual = analysis.stage_metrics(clean[indices], prediction[indices], float(np.mean(clean)))
            np.testing.assert_array_equal(actual["PRD"], expected[indices])
        subset_formula = prd_mecge_official(clean[:1], prediction[:1])[0]
        self.assertNotAlmostEqual(float(expected[0]), float(subset_formula), places=3)
        values = analysis.stage_metrics(clean, prediction, float(np.mean(clean)))
        np.testing.assert_allclose(values["MAD"], [0.1, 0.2, 0.3], atol=1e-6)

    def test_dataset_layouts_and_bad_result_file(self):
        arrays = np.arange(40, dtype=np.float32).reshape(4, 10)
        for data in (arrays, arrays[:, None], arrays[..., None]):
            np.testing.assert_array_equal(analysis.as_nt(data), arrays)
        path = self.work / "input.pkl"
        with path.open("wb") as handle:
            pickle.dump([None, None, arrays[:, None], arrays[..., None]], handle)
        noisy, clean = analysis.load_test_data(path)
        np.testing.assert_array_equal(noisy, clean)
        with path.open("wb") as handle:
            pickle.dump([arrays, arrays, arrays], handle)
        with self.assertRaisesRegex(ValueError, "result PKL"):
            analysis.load_test_data(path)
        for bad in (np.zeros((2, 2, 10)), np.zeros((0, 10)), np.full((2, 10), np.nan)):
            with self.assertRaises(ValueError):
                analysis.as_nt(bad)

    def test_strict_checkpoint_and_restricted_model_configuration(self):
        _, checkpoint = self.fixtures()
        state = torch.load(checkpoint, weights_only=True)
        torch.save({"model_state_dict": state}, checkpoint)
        model, _ = analysis.load_v7(analysis.DEFAULT_CONFIG, checkpoint, "cpu")
        self.assertFalse(model.training)
        missing = next(k for k in state if k.startswith("core.residual_flow."))
        del state[missing]
        torch.save(state, checkpoint)
        with self.assertRaises(RuntimeError):
            analysis.load_v7(analysis.DEFAULT_CONFIG, checkpoint, "cpu")
        config = OmegaConf.create(OmegaConf.to_container(self.config))
        path = self.work / "config.yaml"
        for key, value in (("exp_name", "lstm_v13"), ("model.cfm_inference_steps", 4),
                           ("model.cfm_inference_start_noise_scale", 0.1)):
            modified = OmegaConf.create(OmegaConf.to_container(config))
            OmegaConf.update(modified, key, value)
            OmegaConf.save(modified, path)
            with self.assertRaises(ValueError):
                analysis.load_v7(path, checkpoint, "cpu")

    def test_cli_end_to_end_png_arrays_metrics_and_manifest(self):
        from PIL import Image
        dataset, checkpoint = self.fixtures()
        original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        output = self.work / "analysis"
        command = [sys.executable, "-B", str(PROJECT / "src/analyze_v7_stages.py"),
                   "--checkpoint", str(checkpoint), "--dataset-pkl", str(dataset),
                   "--indices", "2", "0", "2", "--batch-size", "1", "--sampling-rate", "250",
                   "--nv", "1", "--seed", "3407", "--output-dir", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "manifest.json").read_text())
        self.assertTrue(report["complete"])
        self.assertEqual(report["sample_indices"], [2, 0])
        self.assertEqual(report["training_seed_label"], 3407)
        self.assertEqual(report["resolved_config"]["seed"], 42)
        self.assertEqual(report["inference_parity"]["max_abs_difference"], 0)
        noisy, clean = analysis.load_test_data(dataset)
        self.assertEqual(report["full_test_clean_mean"], float(np.mean(clean)))
        with (output / "stage_metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["sample_index"] for r in rows], ["2", "0"])
        for row in rows:
            index = int(row["sample_index"])
            folder = output / f"sample_{index:06d}"
            for name in ("waveform_comparison.png", "flow_correction.png"):
                with Image.open(folder / name) as img:
                    self.assertGreater(img.width, 1000)
                    img.verify()
            with np.load(folder / "waveforms.npz", allow_pickle=False) as waves:
                np.testing.assert_array_equal(waves["clean"], clean[index])
                np.testing.assert_array_equal(waves["noisy"], noisy[index])
                np.testing.assert_allclose(waves["time_seconds"], np.arange(512) / 250)
                np.testing.assert_array_equal(waves["flow_correction"], waves["flow_final"] - waves["lstm_pre_flow"])
                self.assertAlmostEqual(float(row["CosSim_delta"]), float(row["CosSim_flow_final"]) - float(row["CosSim_lstm_pre_flow"]))
        for name, digest in report["artifacts_sha256"].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), original_hash)
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("existing results", repeated.stderr)

    def test_no_reference_and_failed_run_are_explicit(self):
        dataset, checkpoint = self.fixtures(clean=False)
        output = self.work / "no_reference"
        analysis.main(["--checkpoint", str(checkpoint), "--dataset-pkl", str(dataset), "--output-dir", str(output)])
        report = json.loads((output / "manifest.json").read_text())
        self.assertFalse(report["clean_available"])
        self.assertIsNone(report["training_seed_label"])
        self.assertIsNone(report["sampling_rate_hz"])
        with (output / "stage_metrics.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["CosSim_delta"], "")
        failed = self.work / "failed"
        with patch.object(analysis, "plot_sample", side_effect=RuntimeError("plot failure")), self.assertRaises(RuntimeError):
            analysis.main(["--checkpoint", str(checkpoint), "--dataset-pkl", str(dataset), "--output-dir", str(failed)])
        report = json.loads((failed / "manifest.json").read_text())
        self.assertFalse(report["complete"])
        self.assertIn("plot failure", report["error"])

    def test_input_errors_fail_before_writing_results(self):
        dataset, checkpoint = self.fixtures()
        for options in (["--indices", "99"], ["--nv", "2"]):
            output = self.work / "not_created"
            with self.assertRaises(ValueError):
                analysis.main(["--checkpoint", str(checkpoint), "--dataset-pkl", str(dataset), "--output-dir", str(output), *options])
            self.assertFalse(output.exists())
        for options in (["--indices", "-1"], ["--sampling-rate", "nan"], ["--batch-size", "0"]):
            with self.assertRaises(SystemExit):
                analysis.parse_args(["--checkpoint", "x", "--dataset-pkl", "y", *options])


if __name__ == "__main__":
    unittest.main()
