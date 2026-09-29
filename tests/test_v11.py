import csv
import importlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from models import get_model
import models.lstm_v11
from v11_experiments import APP, CORE, MODULES, VARIANTS, V7_CONFIG, build_config, check_output_root, file_digest


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, variant):
        torch.manual_seed(3407)
        config = build_config(variant, 3407, 1, Path("unused.pkl"))
        return get_model(config.model_name, **OmegaConf.to_container(config.model, resolve=True))

    def test_every_variant_forward_backward_and_checkpoint(self):
        generator = torch.Generator().manual_seed(12)
        clean = 0.2 * torch.randn((2, 1, 512), generator=generator)
        noisy = clean + torch.linspace(-0.3, 0.3, 512).view(1, 1, -1)
        mask = torch.ones_like(clean)
        mask[:, :, -16:] = 0
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant)
                model.train()
                # The legacy loss expects a 2-D mask; v11 accepts either shape.
                valid = mask if variant.startswith("v11") else mask.squeeze(1)
                loss = model.compute_loss((noisy, clean, valid), torch.device("cpu"))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                torch.optim.AdamW(model.parameters(), lr=5e-5).step()
                model.eval()
                expected = model.denoising(noisy)
                self.assertEqual(expected.shape, noisy.shape)
                self.assertTrue(torch.isfinite(expected).all())
                self.assertTrue(torch.equal(expected, model.denoising(noisy)))
                restored = self.model(variant).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                self.assertTrue(torch.equal(expected, restored.denoising(noisy)))

    def test_module_ablation_preserves_shared_initialization(self):
        full = self.model("v11").state_dict()
        for variant in ("v11_no_feature_dapp", "v11_no_unet_dapp", "v11_no_resconv", "v11_no_dual_head"):
            model = self.model(variant)
            state = model.state_dict()
            self.assertLess(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in self.model("v11").parameters()))
            for key, value in state.items():
                self.assertTrue(torch.equal(value, full[key]), f"{variant}: changed shared initializer {key}")

    def test_direct_ablation_has_same_parameters(self):
        full, direct = self.model("v11").state_dict(), self.model("v11_direct").state_dict()
        self.assertEqual(full.keys(), direct.keys())
        self.assertTrue(all(torch.equal(full[key], direct[key]) for key in full))

    def test_straight_path_matches_target_velocity(self):
        core = self.model("v11").core
        core.train_noise_scale = 0.0
        core.zero_start_prob = 1.0
        target = torch.randn(2, 1, 512)

        class Oracle(torch.nn.Module):
            def forward(self, state, condition, time):
                return target

        core.residual_flow = Oracle()
        loss = core._flow_matching_loss(torch.zeros(2, 6, 512), target)
        self.assertEqual(loss.item(), 0.0)

    def test_v11_import_does_not_change_legacy_registry_rng_or_outputs(self):
        code = '''
import importlib
import torch
from omegaconf import OmegaConf
from models import get_model
torch.set_num_threads(1)
factory = importlib.import_module("models.factory")
registry = dict(factory.__MODELS__)
config = OmegaConf.load(CONFIG)
torch.manual_seed(3407)
first = get_model(config.model_name, **OmegaConf.to_container(config.model)).eval()
x = torch.linspace(-0.5, 0.5, 512).view(1, 1, -1)
y = first.denoising(x)
rng = torch.get_rng_state().clone()
import models.lstm_v11
assert torch.equal(rng, torch.get_rng_state())
assert all(factory.__MODELS__[name] is cls for name, cls in registry.items())
torch.manual_seed(3407)
second = get_model(config.model_name, **OmegaConf.to_container(config.model)).eval()
assert first.state_dict().keys() == second.state_dict().keys()
assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
assert torch.equal(y, second.denoising(x))
print("legacy state and outputs are bitwise identical")
'''
        code = "CONFIG = " + repr(str(APP / "configs" / V7_CONFIG)) + "\n" + code
        subprocess.run([sys.executable, "-B", "-c", code], cwd=APP, check=True, capture_output=True, text=True)

    def test_legacy_output_root_is_protected(self):
        with self.assertRaises(ValueError):
            check_output_root(PROJECT / "runs/mecge_table1_repro")
        with self.assertRaises(ValueError):
            check_output_root(PROJECT / "data/new_run")


class PipelineTests(unittest.TestCase):
    def run_cli(self, args, success=True):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-B", str(APP / "v11_experiments.py"), *args],
            env=env, capture_output=True, text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_synthetic_train_evaluate_aggregate_and_collision_protection(self):
        with tempfile.TemporaryDirectory(prefix="v11_pipeline_") as directory:
            root = Path(directory)
            data = root / "data/raw"
            data.mkdir(parents=True)
            rng = np.random.default_rng(7)
            time = np.linspace(0, 1, 512, dtype=np.float32)
            clean = np.sin(12 * np.pi * time)[None, :, None] * np.ones((12, 1, 1), dtype=np.float32)
            clean += rng.normal(0, 0.02, clean.shape).astype(np.float32)
            for nv in (1, 2):
                noisy = clean + (0.1 * nv * np.cos(np.pi * time))[None, :, None]
                with (data / f"dataset_bw_nv{nv}.pkl").open("wb") as handle:
                    pickle.dump([noisy, clean, noisy[:6], clean[:6]], handle)
                np.save(data / f"rnd_test_nv{nv}.npy", np.array([0.3, 0.7, 1.2, 1.7, 0.6, 1.0]))
            outputs = root / "runs"
            arguments = [
                "--models", "v11", "--nv", "all", "--device", "cpu",
                "--epochs", "1", "--set", "training.batch_size=2",
                "--data-root", str(root / "data"), "--run-root", str(outputs),
            ]
            self.run_cli(arguments)
            table = outputs / "analysis/table1_comparison__official_nv1_nv2.csv"
            with table.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertFalse((outputs / "v7").exists())
            self.assertTrue(all(int(row["SSD_count"]) == 12 for row in rows))
            self.assertTrue((outputs / "analysis/robustness_comparison__official_nv1_nv2.csv").is_file())
            self.assertFalse((outputs / "analysis/improvement_vs_v7.csv").exists())
            manifests = list(outputs.glob("*/nv*/seed*/manifest.json"))
            self.assertEqual(len(manifests), 2)
            self.assertTrue(all(json.loads(path.read_text())["complete"] for path in manifests))
            checkpoint = outputs / "v11/nv1/seed3407/checkpoint/best_model.pt"
            original_hash = file_digest(checkpoint)
            repeated = self.run_cli(arguments)
            self.assertEqual(repeated.stdout.count("Matching completed run found; skipping."), 2)
            self.assertEqual(file_digest(checkpoint), original_hash)
            collision = self.run_cli([*arguments, "--set", "training.lr=0.0001"], success=False)
            self.assertIn("Choose a new --run-root", collision.stderr)
            self.assertEqual(file_digest(checkpoint), original_hash)

            # Independently check the combined table against the legacy collector.
            reference = root / "legacy_analysis"
            subprocess.run([
                sys.executable, "-B", str(APP / "mecge_table1_collect_official_protocol.py"),
                "--official-results-dir", str(outputs / "official_results"),
                "--rnd-test-nv1", str(data / "rnd_test_nv1.npy"),
                "--rnd-test-nv2", str(data / "rnd_test_nv2.npy"),
                "--output-dir", str(reference),
            ], check=True, capture_output=True)
            with (reference / "table1_comparison__official_nv1_nv2.csv").open(newline="") as handle:
                expected_rows = list(csv.DictReader(handle))
            self.assertEqual(rows, expected_rows)
            robustness = "robustness_comparison__official_nv1_nv2.csv"
            self.assertEqual((outputs / "analysis" / robustness).read_bytes(), (reference / robustness).read_bytes())
            for noise in ("nv1", "nv2", "all"):
                subprocess.run([
                    sys.executable, "-B", str(APP / "mecge_table1_collect_results.py"),
                    "--run-root", str(outputs), "--noise-version", noise,
                    "--output-dir", str(reference),
                ], check=True, capture_output=True)
                for name in (f"table1_comparison__qtdb_train_qtdb_test__{noise}.csv",
                             f"robustness_comparison__qtdb__{noise}.csv"):
                    actual = outputs / "analysis" / name
                    self.assertEqual(actual.read_bytes(), (reference / name).read_bytes())
                    with actual.open(newline="") as handle:
                        records = list(csv.DictReader(handle))
                    expected = (2 if noise == "all" else 1) * (4 if name.startswith("robustness") else 1)
                    self.assertEqual(len(records), expected)
                    self.assertTrue(all(Path(row["metrics_file"]).is_file() for row in records))
            self.assertEqual(len(list((outputs / "lstm_v11/results").rglob("metrics_per_window.csv"))), 2)
            reports = {path.name: path.read_bytes() for path in (outputs / "analysis").glob("*.csv")}
            self.run_cli(["--collect-only", "--run-root", str(outputs), "--data-root", str(root / "data")])
            self.assertEqual(reports, {path.name: path.read_bytes() for path in (outputs / "analysis").glob("*.csv")})
            self.assertEqual(file_digest(checkpoint), original_hash)

    def test_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not_created"
            result = self.run_cli(["--suite", "all", "--dry-run", "--run-root", str(root)])
            self.assertIn("11 variants x 2 noise versions x 1 seeds", result.stdout)
            self.assertFalse(root.exists())

    def test_original_v7_is_not_a_training_variant(self):
        self.assertNotIn("v7", VARIANTS)
        self.assertEqual(len(CORE), 6)
        self.assertEqual(len(MODULES), 6)
        self.assertEqual(set(CORE) | set(MODULES), set(VARIANTS))
        self.run_cli(["--models", "v7", "--dry-run"], success=False)

    def test_collection_without_amplitudes_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "official_results"
            results.mkdir()
            clean = np.arange(16, dtype=np.float32).reshape(2, 8, 1)
            with (results / "lstm_v11__qtdb_train_qtdb_test__nv1__seed3407.pkl").open("wb") as handle:
                pickle.dump([clean, clean, clean + 0.1], handle)
            result = self.run_cli(["--collect-only", "--run-root", str(root), "--skip-robustness"])
            self.assertIn("Skipping official nv1+nv2 reports", result.stdout)
            with (root / "analysis/table1_comparison__qtdb_train_qtdb_test__nv1.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            self.assertFalse((root / "analysis/table1_comparison__official_nv1_nv2.csv").exists())
            self.assertFalse((root / "analysis/improvement_vs_v7.csv").exists())
            self.assertFalse(list(root.rglob("best_model.pt")))
        self.run_cli(["--baseline-results-dir", "unused", "--dry-run"], success=False)


if __name__ == "__main__":
    unittest.main()
