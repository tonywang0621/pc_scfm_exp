"""Numerical objectives, historical control, gradients and v7 inference parity."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from models import get_model
import models.lstm_v14


class V14ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.h = OmegaConf.to_container(OmegaConf.load(
            PROJECT / "src/configs/mecge_table1_repro_lstm_v13.yaml").model, resolve=True)
        cls.h.update(v12_local_gate=False, v13_raw_cos_weight=0.05)
        generator = torch.Generator().manual_seed(121)
        cls.clean = 0.2 * torch.randn(2, 1, 512, generator=generator)
        cls.noisy = cls.clean + torch.linspace(-0.3, 0.3, 512).view(1, 1, -1)

    def model(self, **overrides):
        torch.manual_seed(3407)
        return get_model("lstm_relative_balanced_v14_ecg", **dict(self.h, **overrides))

    def test_relative_loss_original_units_matches_hand_computation(self):
        core = self.model().core
        clean = torch.tensor([[1., 3.], [0.1, 0.3]], dtype=torch.float64)
        restored = clean + 0.1
        factor = torch.tensor([2., 5.], dtype=torch.float64).view(2, 1, 1)
        actual = core._relative_reconstruction_loss(
            clean * factor.squeeze(-1), restored * factor.squeeze(-1), factor)
        expected = (0.01 / (5.0 + 0.01) + 0.01 / (0.05 + 0.01)) / 2
        self.assertAlmostEqual(actual.item(), expected, places=14)
        # Equal absolute error receives more supervision in a low-energy window.
        first = core._relative_reconstruction_loss(clean[:1], restored[:1], torch.ones(1, 1, 1))
        second = core._relative_reconstruction_loss(clean[1:], restored[1:], torch.ones(1, 1, 1))
        self.assertGreater(second.item(), first.item())

    def test_masks_exclude_padding_and_empty_windows_with_finite_gradients(self):
        core = self.model().core
        clean = torch.tensor([[1., 2., 900.], [0., 900., 900.], [900., 900., 900.]], dtype=torch.float64)
        restored = torch.tensor([[2., 4., -900.], [0.1, -900., -900.], [-900., -900., -900.]],
                                dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
        factor = torch.ones(3, 1, 1, dtype=torch.float64)
        value = core._relative_reconstruction_loss(clean, restored, factor, mask)
        self.assertAlmostEqual(value.item(), (2.5 / 2.51 + 0.01 / 0.01) / 2, places=14)
        self.assertEqual(value.item(), core._relative_reconstruction_loss(clean, restored, factor, mask[:, None]).item())
        value.backward()
        self.assertTrue(torch.isfinite(restored.grad).all())
        self.assertEqual(restored.grad[~mask].abs().sum().item(), 0)
        empty = core._relative_reconstruction_loss(clean, restored, factor, torch.zeros_like(mask))
        self.assertEqual(empty.item(), 0)

    def test_zero_clean_has_bounded_weight_and_amplitude_inflation_is_penalized(self):
        core = self.model().core
        clean = torch.zeros(2, 4)
        prediction = torch.ones_like(clean, requires_grad=True)
        factor = torch.ones(2, 1, 1)
        loss = core._relative_reconstruction_loss(clean, prediction, factor)
        self.assertAlmostEqual(loss.item(), 100.0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        clean.fill_(0.5)
        self.assertEqual(core._relative_reconstruction_loss(clean, clean, factor).item(), 0)
        self.assertGreater(core._relative_reconstruction_loss(clean, 2 * clean, factor).item(), 0)

    def test_schedule_and_no_curriculum_ablation(self):
        model = self.model().train()
        getters = (model.tail_weight, model.extra_mse_weight, model.raw_cos_weight, model.relative_weight)
        terminal = (0.01, 0.25, 0.05, 0.02)
        self.assertEqual([method() for method in getters], [0., 0., 0., 0.])
        for epoch, scale in ((1, 0), (20, 0), (21, 0.1), (25, 0.5), (30, 1)):
            model.set_epoch(epoch, 30)
            for method, maximum in zip(getters, terminal):
                self.assertAlmostEqual(method(), maximum * scale)
        model.eval()
        model.set_epoch(1, 30)
        self.assertEqual([method() for method in getters], list(terminal))
        immediate = self.model(v14_curriculum=False).train()
        immediate.set_epoch(1, 30)
        self.assertEqual([immediate.tail_weight(), immediate.extra_mse_weight(),
                          immediate.raw_cos_weight(), immediate.relative_weight()], list(terminal))
        model.train().set_epoch(1, 5)
        self.assertAlmostEqual(model.relative_weight(), 0.004)

    def test_v13_control_preserves_parameters_rng_loss_and_gradients(self):
        h = dict(self.h, v13_raw_cos_weight=0.01)
        torch.manual_seed(3407)
        previous = get_model("lstm_metric_balanced_v13_ecg", **h)
        previous_rng = torch.get_rng_state().clone()
        control = self.model(v14_relative_weight=0.0, v13_raw_cos_weight=0.01)
        self.assertTrue(torch.equal(previous_rng, torch.get_rng_state()))
        self.assertEqual(previous.state_dict().keys(), control.state_dict().keys())
        for key, value in previous.state_dict().items():
            self.assertTrue(torch.equal(value, control.state_dict()[key]), key)
        for training, epoch in ((True, 1), (True, 21), (True, 30), (False, 1)):
            outputs, rngs = [], []
            for model in (previous, control):
                model.train(training)
                model.set_epoch(epoch, 30)
                torch.manual_seed(88)
                outputs.append(model.compute_loss((self.noisy, self.clean), "cpu"))
                rngs.append(torch.get_rng_state().clone())
            self.assertTrue(torch.equal(*outputs))
            self.assertTrue(torch.equal(*rngs))
            if training and epoch == 30:
                for loss in outputs:
                    loss.backward()
                for (name, p), (_, q) in zip(previous.named_parameters(), control.named_parameters()):
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        self.assertTrue(torch.equal(p.grad, q.grad), name)

    def test_v7_parameter_count_initialization_and_inference_are_unchanged(self):
        torch.manual_seed(3407)
        original = get_model("lstm_dualpath_dapp_cfm_unet_bd_ecg", **self.h).eval()
        original_rng = torch.get_rng_state().clone()
        model = self.model().eval()
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(original.state_dict().keys(), model.state_dict().keys())
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        self.assertEqual(sum(p.numel() for p in original.parameters()), sum(p.numel() for p in model.parameters()))
        with torch.no_grad():
            self.assertTrue(torch.equal(original.denoising(self.noisy), model.denoising(self.noisy)))

    def test_all_seven_variants_train_reload_and_denoise_deterministically(self):
        variants = ({}, {"v14_relative_weight": 0.0}, {"v13_raw_cos_weight": 0.0},
                    {"v12_tail_weight": 0.0}, {"v13_extra_mse_weight": 0.0},
                    {"v14_curriculum": False}, {"v14_relative_weight": 0.0, "v13_raw_cos_weight": 0.01})
        for overrides in variants:
            with self.subTest(overrides=overrides):
                model = self.model(**overrides).train()
                model.set_epoch(30, 30)
                loss = model.compute_loss((self.noisy, self.clean), "cpu")
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                torch.optim.AdamW(model.parameters(), lr=5e-5).step()
                model.eval()
                other = self.model(**overrides).eval()
                other.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    prediction = model.denoising(self.noisy)
                    self.assertEqual(prediction.shape, self.clean.shape)
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertTrue(torch.equal(prediction, model.denoising(self.noisy)))
                    self.assertTrue(torch.equal(prediction, other.denoising(self.noisy)))

    def test_relative_loss_is_added_once(self):
        full, no_relative = self.model().eval().core, self.model(v14_relative_weight=0).eval().core
        clean, noisy, factor = self.clean[:, 0], self.noisy[:, 0], torch.ones(2, 1, 1)
        difference = full._ecg_loss(clean, noisy, factor) - no_relative._ecg_loss(clean, noisy, factor)
        expected = 0.02 * full._relative_reconstruction_loss(clean, noisy, factor)
        self.assertTrue(torch.allclose(difference, expected, atol=1e-6))

    def test_invalid_settings_fail(self):
        for overrides in ({"v14_relative_weight": -1}, {"v14_relative_weight": float("nan")},
                          {"v14_energy_floor": 0}, {"v14_energy_floor": float("inf")},
                          {"v14_relative_ramp_epochs": 1.5}, {"v14_relative_ramp_epochs": True},
                          {"v14_curriculum": "false"}, {"v12_local_gate": True},
                          {"v12_feature_dapp": False}, {"v12_dual_head": False}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.model(**overrides)

    def test_import_is_opt_in(self):
        code = """
import torch
import models.lstm_v13
import importlib
factory = importlib.import_module('models.factory')
registry = dict(factory.__MODELS__)
assert 'lstm_relative_balanced_v14_ecg' not in registry
rng = torch.get_rng_state().clone()
import models.lstm_v14
assert torch.equal(rng, torch.get_rng_state())
assert set(factory.__MODELS__) - set(registry) == {'lstm_relative_balanced_v14_ecg'}
assert all(factory.__MODELS__[key] is value for key, value in registry.items())
"""
        subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT / "src",
                       check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
