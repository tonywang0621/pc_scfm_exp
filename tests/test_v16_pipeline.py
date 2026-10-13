"""Four-metric selection, official metrics, exact resume, real v7 warm-start smoke."""
import copy
import csv
import json
import pickle
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import mecge_table1_run_v16_model as runner


class TinyModel(torch.nn.Module):
    def __init__(self, gain=0.85, static=False):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(gain))
        self.frozen = torch.nn.Parameter(torch.tensor(0.125), requires_grad=False)
        self.static = static
        self.loss_components = {}

    def denoising(self, noisy):
        return self.gain * noisy

    def stages(self, noisy):
        final = self.denoising(noisy)
        pre = final + 0.01
        return {"pre_flow": pre, "final": final, "flow_correction": final-pre,
                "diagnostics": {"example_gate": torch.full((len(noisy),), 0.2)}}

    def compute_loss(self, batch, device):
        noisy, clean = (value.to(device) for value in batch)
        jitter = torch.rand(()) + random.random() + np.random.random()
        loss = ((self.denoising(noisy) - clean).square().mean() + 0.001*jitter*self.gain)
        if self.static:
            loss = loss * 0
        self.loss_components = {"original_v7_loss": float(loss.detach()),
                                "raw_cos": 0., "weighted_raw_cos": 0., "total_loss": float(loss.detach())}
        return loss

    def configure_finetuning(self, **kwargs):
        return [{"name": "flow", "params": [self.gain], "lr": kwargs["lr_flow"]}]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def settings(epochs=3):
    return {"device": "cpu", "finetune_scope": "adapter", "lr_adapter": 0.01,
            "lr_flow": 0.01, "optimizer": "AdamW", "betas": [0.8, 0.99], "weight_decay": 0.0,
            "grad_clip_norm": 0.5, "epochs": epochs}


def loaders():
    noisy = torch.linspace(-1, 1, 70).reshape(7, 1, 10)
    data = TensorDataset(noisy, noisy*0.7 + 0.01)
    return (DataLoader(data, batch_size=2, shuffle=True, drop_last=True),
            DataLoader(data, batch_size=3, drop_last=True))


class V16PipelineTests(unittest.TestCase):
    def assert_rng_equal(self, first, second):
        self.assertEqual(first["python"], second["python"])
        self.assertEqual(first["numpy"][0], second["numpy"][0])
        np.testing.assert_array_equal(first["numpy"][1], second["numpy"][1])
        self.assertEqual(first["numpy"][2:], second["numpy"][2:])
        self.assertTrue(torch.equal(first["torch_cpu"], second["torch_cpu"]))

    def test_protection_rejects_any_regression_equality_nonfinite_and_tie_is_earlier(self):
        baseline = dict(ssd=3., mad=.3, prd=35., cossim=.94)
        good = dict(ssd=2.9, mad=.29, prd=34., cossim=.941)
        self.assertTrue(runner.feasible(good, baseline))
        self.assertFalse(runner.feasible(baseline, baseline))
        for key in runner.METRICS:
            bad = dict(good)
            bad[key] = baseline[key] + (1e-10 if key != "cossim" else -1e-10)
            self.assertFalse(runner.feasible(bad, baseline))
            bad[key] = float("nan")
            self.assertFalse(runner.feasible(bad, baseline))
        selected = {"epoch": 2, "metrics": good}
        self.assertFalse(runner.prefer_candidate(good, 3, selected))
        self.assertTrue(runner.prefer_candidate({**good, "cossim": .942}, 3, selected))
        self.assertTrue(runner.prefer_candidate({**good, "mad": .28}, 3, selected))
        self.assertFalse(runner.prefer_candidate({**good, "ssd": 2.8}, 3, selected))

    def test_full_validation_official_prd_partial_batches_and_rng_preserved(self):
        noisy = torch.tensor([[1., 2., 4., 3.], [10., 11., 7., 9.], [-2., 1., 3., 0.]])[:, None]
        clean = noisy * .7 + torch.tensor([0., .1, -.1, .2])
        model, reference = TinyModel(), TinyModel()
        model.train()
        before = runner.capture_rng_state()
        expected = runner.metric_arrays(clean[:, 0].numpy(), (noisy[:, 0]*.85).numpy())
        for size in (1, 2, 3):
            result = runner.validate(model, DataLoader(TensorDataset(noisy, clean), batch_size=size,
                                                       drop_last=True), "cpu", reference)
            self.assertEqual(result["count"], 3)
            self.assertEqual(result["parity_max_abs"], 0)
            for key in runner.METRICS:
                self.assertAlmostEqual(result["metrics"][key], float(np.asarray(expected[key], np.float64).mean()), places=9)
            self.assertEqual(len(result["windows"]), 3)
            self.assertEqual(result["diagnostics"]["example_gate"]["count"], 3)
        self.assertTrue(model.training)
        self.assert_rng_equal(before, runner.capture_rng_state())

    def test_initial_parity_must_match_real_reference(self):
        _, val = loaders()
        with self.assertRaises(AssertionError):
            runner.validate(TinyModel(.8), val, "cpu", TinyModel(.9))

    def test_correction_identity_and_clipped_label_oracle(self):
        clean = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 0.]])
        pre = np.zeros_like(clean)
        final = np.array([[2., 0.], [-1., 0.], [0., 0.], [0., 0.]])
        values = runner.correction_geometry(clean, pre, final)
        np.testing.assert_allclose(values["shrink_oracle_alpha"], [.5, 0, 0, 0])
        np.testing.assert_allclose(values["shrink_oracle_ssd"], [0, 1, 1, 0])
        np.testing.assert_allclose(values["alignment_cosine"], [1, -1, 0, 0])
        np.testing.assert_allclose(values["ssd_identity_residual"], 0, atol=1e-14)

    def test_gate_grid_matches_direct_waveform_enumeration_with_cross_terms(self):
        import itertools
        torch.manual_seed(5)
        pre = torch.randn(3, 1, 11, dtype=torch.float64)
        limited = torch.randn(3, 2, 11, dtype=torch.float64)
        # Deliberately correlated directions; omitting the Gram cross term fails.
        limited[:, 1] = limited[:, 0] * .8 + limited[:, 1] * .2
        target = pre + .04*torch.randn_like(pre)
        logits = torch.randn(3, 3, dtype=torch.float64)
        maxima = torch.tensor([.1, .44, .35], dtype=torch.float64)
        offsets = torch.full((3, 3), .137, dtype=torch.float64)
        def output(offset):
            g = maxima * torch.sigmoid(logits+offset)
            return pre[:, 0] + ((1-g[:, 2])*g[:, 0])[:, None]*limited[:, 0] - (g[:, 2]*g[:, 1])[:, None]*limited[:, 1]
        final = output(offsets)[:, None]
        stage = dict(pre_flow=pre, final=final, limited_flow=limited, base_logits=logits,
                     gate_maxima=maxima, adapter_offsets=offsets)
        actual = runner.bounded_gate_grid(stage, target[:, 0].numpy(), .5)
        candidates = [((final-target)**2).sum((1, 2)).numpy()]
        for offset in itertools.product(np.linspace(-.5, .5, runner.GATE_GRID_POINTS), repeat=3):
            candidates.append(((output(torch.tensor(offset))-target[:, 0])**2).sum(1).numpy())
        np.testing.assert_allclose(actual["gate_grid_best_ssd"], np.min(candidates, axis=0), atol=1e-14)
        self.assertTrue(np.all(actual["gate_grid_ssd_delta"] <= 0))

    def test_frozen_parameter_mutation_stops_before_checkpoint_selection(self):
        class BadModel(TinyModel):
            def compute_loss(self, *args):
                self.frozen.add_(1.)
                return super().compute_loss(*args)
        train, val = loaders()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Frozen v7 parameters changed"):
                runner.train_model(BadModel(), TinyModel(), train, val, settings(1), directory, {"epochs": 1})
            self.assertFalse((Path(directory)/"best_model.pt").exists())

    def test_sampler_order_independent_of_model_random_draws(self):
        first, _ = loaders()
        second, _ = loaders()
        runner.set_epoch_sampler(first, 3407, 3)
        expected = [batch[0] for batch in first]
        torch.rand(10000)
        runner.set_epoch_sampler(second, 3407, 3)
        actual = [batch[0] for batch in second]
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_static_baseline_never_creates_candidate_or_claims_success(self):
        train, val = loaders()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = runner.train_model(TinyModel(static=True), TinyModel(), train, val,
                                        settings(1), path, {"epochs": 1})
            self.assertEqual(result["status"], "no_feasible_checkpoint")
            self.assertTrue(result["complete"])
            self.assertIsNone(result["selected_checkpoint"])
            self.assertFalse((path / "best_model.pt").exists())
            for name in ("baseline.pt", "model_last.pt", "training_state.pt", "loss_history.csv",
                         "validation/epoch_000_windows.csv", "validation/epoch_001_diagnostics.json"):
                self.assertTrue((path / name).is_file(), name)
            self.assertEqual(result["baseline_metrics"]["val_count"], 7)
            with self.assertRaises(FileExistsError):
                runner.train_model(TinyModel(), TinyModel(), train, val, settings(1), path, {"epochs": 1})

    def test_training_exact_resume_and_identity_baseline_tampering_guards(self):
        train, val = loaders()
        identity = {"epochs": 3, "config_hash": "constant", "seed": 987}
        with tempfile.TemporaryDirectory() as directory:
            whole, partial = Path(directory)/"whole", Path(directory)/"partial"
            seed_all(987)
            model = TinyModel()
            expected = runner.train_model(model, TinyModel(), train, val, settings(), whole, identity)
            final_weights = copy.deepcopy(model.state_dict())
            final_rng = runner.capture_rng_state()
            seed_all(987)
            original = runner.save_training_state
            def save_then_interrupt(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("simulated interruption")
            with patch.object(runner, "save_training_state", side_effect=save_then_interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    runner.train_model(TinyModel(), TinyModel(), train, val, settings(), partial, identity)
            seed_all(999)
            resumed = TinyModel()
            actual = runner.train_model(resumed, TinyModel(), train, val, settings(), partial, identity, resume=True)
            self.assertEqual(expected, actual)
            for key, value in final_weights.items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]))
            self.assert_rng_equal(final_rng, runner.capture_rng_state())
            self.assertEqual((whole/"loss_history.csv").read_bytes(), (partial/"loss_history.csv").read_bytes())
            for epoch in range(4):
                self.assertEqual((whole/f"validation/epoch_{epoch:03d}_windows.csv").read_bytes(),
                                 (partial/f"validation/epoch_{epoch:03d}_windows.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "epoch budget"):
                runner.load_training_state(partial, {**identity, "epochs": 4}, actual["baseline"])
            (partial/"baseline.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "baseline checkpoint changed"):
                runner.load_training_state(partial, identity, actual["baseline"])

    def test_selection_saves_feasible_candidate_and_never_picks_best_single_metric(self):
        train, val = loaders()
        baseline = dict(ssd=3., mad=.3, prd=35., cossim=.94)
        sequence = [dict(ssd=2.9, mad=.29, prd=34., cossim=.941),
                    dict(ssd=2.8, mad=.28, prd=34., cossim=.942),
                    dict(ssd=2., mad=.31, prd=30., cossim=.99)]
        original = runner.validate
        weights = []
        def validation(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            if len(args) == 3:  # initial call includes the frozen reference
                return result
            weights.append(copy.deepcopy(model.state_dict()))
            result["metrics"] = sequence[len(weights)-1]
            return result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.object(runner, "frozen_reference_metrics", return_value=baseline), \
                    patch.object(runner, "validate", side_effect=validation):
                result = runner.train_model(TinyModel(), TinyModel(), train, val, settings(), path, {"epochs": 3})
            self.assertEqual(result["status"], "selected")
            self.assertEqual(result["selected_checkpoint"]["epoch"], 2)
            best = torch.load(path/"best_model.pt", weights_only=True)
            self.assertTrue(torch.equal(best["gain"], weights[1]["gain"]))
            loaded = runner.load_training_state(path, {"epochs": 3}, baseline)
            self.assertEqual(loaded["selected"]["metrics"], sequence[1])

    def test_invalid_training_settings_and_conflicting_modes(self):
        cfg = OmegaConf.create({"model_name": "lstm_v16_ecg", "training": {}})
        self.assertEqual(runner.validated_settings(cfg, "cpu")["epochs"], 10)
        for key, value in (("train_epochs", True), ("batch_size", 1.5), ("lr_flow", 0),
                           ("lr_adapter", float("nan")), ("scheduler", "ExponentialLR"),
                           ("early_stopping_patience_epochs", 1), ("grad_clip_norm", -1),
                           ("betas", [0.8, 1]), ("finetune_scope", "unknown"),
                           ("selection_metric", "test_cossim")):
            with self.subTest(key=key):
                bad = OmegaConf.create({"model_name": "lstm_v16_ecg", "training": {key: value}})
                with self.assertRaises(ValueError):
                    runner.validated_settings(bad, "cpu")

    def test_actual_v7_warm_start_training_and_eval_only_export(self):
        import models.lstm_v16
        from models import get_model
        config = OmegaConf.load(PROJECT/"src/configs/mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml")
        config.model_name = "lstm_v16_ecg"
        config.training = {"train_epochs": 1, "batch_size": 2, "scheduler": "none",
                           "lr_adapter": 1e-4, "lr_flow": 1e-5}
        config.model.v16_gate_mode = "correction"
        rng = np.random.default_rng(15)
        clean = rng.normal(0, .2, (4, 512, 1)).astype(np.float32)
        noisy = clean + .03*np.sin(np.arange(512))[None, :, None].astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            OmegaConf.save(config, path/"config.yaml")
            with (path/"dataset.pkl").open("wb") as handle:
                pickle.dump([noisy, clean, noisy[:2], clean[:2]], handle)
            reference = get_model("lstm_dualpath_dapp_cfm_unet_bd_ecg", **OmegaConf.to_container(config.model))
            torch.save(reference.state_dict(), path/"v7.pt")
            arguments = ["--config", str(path/"config.yaml"), "--dataset-pkl", str(path/"dataset.pkl"),
                         "--init-checkpoint", str(path/"v7.pt"), "--checkpoint-dir", str(path/"checkpoints"),
                         "--log-dir", str(path/"logs"), "--output-pkl", str(path/"v16.pkl"),
                         "--baseline-output-pkl", str(path/"baseline.pkl"), "--device", "cpu"]
            for protected in (path/"v7.pt", path/"config.yaml", path/"checkpoints/training_state.pt"):
                bad_arguments = list(arguments)
                bad_arguments[bad_arguments.index("--output-pkl")+1] = str(protected)
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    runner.main(bad_arguments)
            result = runner.main(arguments)
            self.assertTrue(result["complete"])
            self.assertEqual(result["baseline_metrics"]["val_count"], 2)
            with np.load(path/"checkpoints/validation/epoch_000_waveforms.npz") as snapshot:
                np.testing.assert_array_equal(snapshot["original_train_index"], [3, 2])
                np.testing.assert_array_equal(snapshot["noisy"], noisy[[3, 2]].transpose(0, 2, 1))
            provenance = json.loads((path/"checkpoints/validation/final_waveforms.json").read_text())
            self.assertEqual(provenance["retained_count"], 2)
            self.assertEqual(provenance["full_validation_count"], 2)
            self.assertEqual(provenance["waveform_sha256"], runner.sha256(path/"checkpoints/validation/final_waveforms.npz"))
            saved_state = (path/"checkpoints/training_state.pt").read_bytes()
            baseline_bytes = (path/"baseline.pkl").read_bytes()
            rerun = runner.main(arguments + ["--skip-train"])
            self.assertEqual(result, rerun)
            self.assertEqual(saved_state, (path/"checkpoints/training_state.pt").read_bytes())
            self.assertEqual(baseline_bytes, (path/"baseline.pkl").read_bytes())
            self.assertEqual((path/"v16.pkl").exists(), result["status"] == "selected")
            with (path/"baseline.pkl").open("rb") as handle:
                original, target, prediction = pickle.load(handle)
            np.testing.assert_array_equal(original, noisy[:2])
            np.testing.assert_array_equal(target, clean[:2])
            self.assertEqual(prediction.shape, (2, 512, 1))
            self.assertTrue(np.isfinite(prediction).all())
            with (path/"checkpoints/loss_history.csv").open() as handle:
                row = next(csv.DictReader(handle))
            for field in ("reconstruction", "stochastic_fm", "cfm_auxiliary", "lr_adapter",
                          "grad_norm_adapter", "first_batch_grad_reconstruction_adapter", "val_stage1_cossim",
                          "val_diag_alignment_cosine", "frozen_parameters_sha256"):
                self.assertIn(field, row)


if __name__ == "__main__":
    unittest.main()
