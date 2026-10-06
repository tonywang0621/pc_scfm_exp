"""CPU checks for v13's objectives and exact v12 loss-only control."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from models import get_model
import models.lstm_v13
from utils_ecg import cosine_similarity

V12_CONFIG = PROJECT / "src/configs/mecge_table1_repro_lstm_v12.yaml"
VARIANTS = {
    "v13": {},
    "v13_no_extra_mse": {"v13_extra_mse_weight": 0.0},
    "v13_no_raw_cos": {"v13_raw_cos_weight": 0.0},
    "v13_neither": {"v13_extra_mse_weight": 0.0, "v13_raw_cos_weight": 0.0},
    "v13_no_topk": {"v12_tail_weight": 0.0},
    "v13_no_local_gate": {"v12_local_gate": False},
}


class V13ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.h = OmegaConf.to_container(OmegaConf.load(V12_CONFIG).model, resolve=True)
        cls.h["v12_tail_weight"] = 0.01
        generator = torch.Generator().manual_seed(12)
        cls.clean = 0.2 * torch.randn((2, 1, 512), generator=generator)
        cls.noisy = cls.clean + torch.linspace(-0.3, 0.3, 512).view(1, 1, -1)
        cls.mask = torch.ones_like(cls.clean)
        cls.mask[..., -16:] = 0

    def model(self, variant="v13", **kwargs):
        torch.manual_seed(3407)
        settings = dict(self.h, **VARIANTS[variant])
        settings.update(kwargs)
        return get_model("lstm_metric_balanced_v13_ecg", **settings)

    def test_defaults_preserve_delayed_topk_without_explicit_v12_keys(self):
        settings = {key: value for key, value in self.h.items()
                    if key not in {"v12_tail_weight", "v12_local_gate"}}
        model = get_model("lstm_metric_balanced_v13_ecg", **settings).eval()
        self.assertEqual(model.tail_weight(), 0.01)
        self.assertIsNotNone(model.core.temporal_gate)
        self.assertEqual(model.metric_weights(), {"extra_mse": 0.25, "raw_cos": 0.01})

    def test_all_six_variants_gradients_determinism_and_checkpoint(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant).train()
                model.set_epoch(30, 30)
                loss = model.compute_loss((self.noisy, self.clean, self.mask), torch.device("cpu"))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                torch.optim.AdamW(model.parameters(), lr=5e-5).step()
                model.eval()
                prediction = model.denoising(self.noisy)
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertTrue(torch.equal(prediction, model.denoising(self.noisy)))
                duplicate = self.model(variant).eval()
                duplicate.load_state_dict(model.state_dict(), strict=True)
                self.assertTrue(torch.equal(prediction, duplicate.denoising(self.noisy)))

    def test_neither_matches_v12_parameters_rng_loss_and_gradients_exactly(self):
        torch.manual_seed(3407)
        v12 = get_model("lstm_temporal_gate_v12_ecg", **self.h)
        init_rng = torch.get_rng_state().clone()
        v13 = self.model("v13_neither")
        self.assertTrue(torch.equal(init_rng, torch.get_rng_state()))
        self.assertEqual(v12.state_dict().keys(), v13.state_dict().keys())
        for key, value in v12.state_dict().items():
            self.assertTrue(torch.equal(value, v13.state_dict()[key]), key)
        for training in (True, False):
            for epoch in (1, 21, 30):
                with self.subTest(training=training, epoch=epoch):
                    outputs, rngs = [], []
                    for model in (v12, v13):
                        model.train(training)
                        model.set_epoch(epoch, 30)
                        torch.manual_seed(100)
                        outputs.append(model.compute_loss((self.noisy, self.clean, self.mask), torch.device("cpu")))
                        rngs.append(torch.get_rng_state().clone())
                    self.assertTrue(torch.equal(*outputs))
                    self.assertTrue(torch.equal(*rngs))
                    if training and epoch == 30:
                        for loss in outputs:
                            loss.backward()
                        for (key, first), (_, second) in zip(v12.named_parameters(), v13.named_parameters()):
                            if first.grad is None:
                                self.assertIsNone(second.grad, key)
                            else:
                                self.assertTrue(torch.equal(first.grad, second.grad), key)
        v12.eval()
        v13.eval()
        self.assertTrue(torch.equal(v12.denoising(self.noisy), v13.denoising(self.noisy)))

    def test_full_has_no_extra_parameters_or_inference_change(self):
        v13 = self.model().eval()
        torch.manual_seed(3407)
        v12 = get_model("lstm_temporal_gate_v12_ecg", **self.h).eval()
        self.assertEqual(v13.state_dict().keys(), v12.state_dict().keys())
        for key, value in v12.state_dict().items():
            self.assertTrue(torch.equal(value, v13.state_dict()[key]), key)
        self.assertTrue(torch.equal(v12.denoising(self.noisy), v13.denoising(self.noisy)))

    def test_metric_schedule_and_fixed_validation_objective(self):
        model = self.model().train()
        self.assertEqual(model.metric_weights(), {"extra_mse": 0.0, "raw_cos": 0.0})
        for epoch, scale in ((1, 0), (20, 0), (21, 0.1), (25, 0.5), (30, 1)):
            model.set_epoch(epoch, 30)
            self.assertAlmostEqual(model.metric_weights()["extra_mse"], 0.25 * scale)
            self.assertAlmostEqual(model.metric_weights()["raw_cos"], 0.01 * scale)
            self.assertAlmostEqual(model.tail_weight(), 0.01 * scale)
        for epoch, scale in ((1, 0.2), (5, 1)):
            model.set_epoch(epoch, 5)
            self.assertAlmostEqual(model.metric_weights()["extra_mse"], 0.25 * scale)
        losses = []
        model.eval()
        for epoch in (1, 21, 30):
            model.set_epoch(epoch, 30)
            self.assertEqual(model.metric_weights(), {"extra_mse": 0.25, "raw_cos": 0.01})
            torch.manual_seed(111)
            losses.append(model.compute_loss((self.noisy, self.clean, self.mask), torch.device("cpu")))
        self.assertTrue(all(torch.equal(losses[0], value) for value in losses[1:]))

    def test_raw_cosine_matches_official_formula_and_is_offset_sensitive(self):
        core = self.model().core
        clean = torch.tensor([[-1., 0., 1.], [1., 3., 7.]], dtype=torch.float64)
        restored = clean + torch.tensor([[2.], [-0.5]], dtype=torch.float64)
        factor = torch.tensor([2., 5.], dtype=torch.float64).view(2, 1, 1)
        actual = core._raw_cosine_loss(clean * factor.squeeze(-1), restored * factor.squeeze(-1), factor)
        expected = float(np.mean(1 - cosine_similarity(clean.numpy(), restored.numpy())))
        self.assertAlmostEqual(float(actual), expected, places=14)
        self.assertGreater(float(actual), 0.1)
        centered = torch.nn.functional.cosine_similarity(clean - clean.mean(-1, keepdim=True),
                                                         restored - restored.mean(-1, keepdim=True))
        self.assertTrue(torch.allclose(centered, torch.ones_like(centered)))

    def test_original_amplitude_mse_masks_and_padding_gradients(self):
        core = self.model().core
        clean = torch.zeros(3, 4, dtype=torch.float64)
        restored = torch.tensor([[2., 6., 100., 100.], [8., 100., 100., 100.],
                                 [100., 100., 100., 100.]], dtype=torch.float64, requires_grad=True)
        factor = torch.tensor([2., 4., 10.], dtype=torch.float64).view(3, 1, 1)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        loss = core._extra_mse_loss(clean, restored, factor, mask)
        self.assertAlmostEqual(float(loss.detach()), (1. + 9. + 4.) / 3)
        self.assertEqual(float(loss.detach()), float(core._extra_mse_loss(clean, restored, factor, mask.unsqueeze(1)).detach()))
        loss.backward()
        self.assertEqual(float(restored.grad.masked_select(~mask.bool()).abs().sum()), 0.0)
        zero_loss = core._extra_mse_loss(clean, restored, factor, torch.zeros_like(mask))
        self.assertEqual(float(zero_loss.detach()), 0.0)

    def test_cosine_masks_zero_reference_and_zero_prediction_have_finite_gradients(self):
        core = self.model().core
        clean = torch.tensor([[1., 2., 100., 100.], [0., 0., 0., 0.],
                              [0., 0., 0., 0.], [1., 2., 3., 4.]], dtype=torch.float64)
        restored = torch.tensor([[2., 1., -100., -100.], [4., 2., 3., 1.],
                                 [0., 0., 0., 0.], [0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
        factor = torch.ones(4, 1, 1, dtype=torch.float64)
        loss = core._raw_cosine_loss(clean, restored, factor, mask)
        self.assertAlmostEqual(float(loss.detach()), ((1 - 4 / (5 + 1e-10)) + 1) / 2, places=14)
        self.assertEqual(float(loss.detach()), float(core._raw_cosine_loss(clean, restored, factor, mask.unsqueeze(1)).detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(restored.grad).all())
        self.assertGreater(float(restored.grad[3].abs().sum()), 0)
        self.assertEqual(float(restored.grad[1:3].abs().sum()), 0)
        self.assertEqual(float(restored.grad[0, 2:].abs().sum()), 0)
        for dtype in (torch.float16, torch.float32, torch.float64):
            prediction = torch.zeros(2, 4, dtype=dtype, requires_grad=True)
            target = torch.zeros_like(prediction)
            empty = core._raw_cosine_loss(target, prediction, torch.ones(2, 1, 1, dtype=dtype))
            self.assertEqual(float(empty.detach()), 0.0)
            empty.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_weighted_terms_are_added_once(self):
        core = self.model().eval().core
        control = self.model("v13_neither").eval().core
        clean, restored = self.clean.squeeze(1), self.noisy.squeeze(1)
        factor = torch.full((2, 1, 1), 2.0)
        difference = core._ecg_loss(clean, restored, factor, valid_mask=self.mask) - control._ecg_loss(
            clean, restored, factor, valid_mask=self.mask)
        expected = (0.25 * core._extra_mse_loss(clean, restored, factor, self.mask)
                    + 0.01 * core._raw_cosine_loss(clean, restored, factor, self.mask))
        self.assertTrue(torch.allclose(difference, expected, atol=1e-6))

    def test_invalid_weights_ramp_and_ambiguous_legacy_cos_are_rejected(self):
        for field in ("v13_extra_mse_weight", "v13_raw_cos_weight"):
            for value in (-0.01, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.model(**{field: value})
        for value in (0, -1, 1.5, float("inf"), float("nan"), True, "10"):
            with self.subTest(ramp=value), self.assertRaises(ValueError):
                self.model(v13_ramp_epochs=value)
        with self.assertRaisesRegex(ValueError, "centered"):
            self.model(loss_fn=self.h["loss_fn"] + "+cos")
        self.model(v13_raw_cos_weight=0.0, loss_fn=self.h["loss_fn"] + "+cos")

    def test_import_is_opt_in_and_does_not_change_existing_registration_or_rng(self):
        code = """
import importlib
import torch
import models.lstm_v12
factory = importlib.import_module('models.factory')
registry = dict(factory.__MODELS__)
assert 'lstm_metric_balanced_v13_ecg' not in registry
torch.manual_seed(12)
rng = torch.get_rng_state().clone()
import models.lstm_v13
assert torch.equal(rng, torch.get_rng_state())
assert all(factory.__MODELS__[key] is value for key, value in registry.items())
assert set(factory.__MODELS__) - set(registry) == {'lstm_metric_balanced_v13_ecg'}
"""
        subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT / "src", check=True,
                       capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
