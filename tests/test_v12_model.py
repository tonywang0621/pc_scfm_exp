"""CPU regression checks for v12; uses synthetic signals, never research data."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from models import get_model
from models.lstm_v12 import BoundedTemporalGateOffsets

V7_CONFIG = PROJECT / "src/configs/mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml"
VARIANTS = {
    "v12": {},
    "v7_delayed_topk": {"v12_local_gate": False, "v12_tail_weight": 0.01},
    "v12_delayed_topk": {"v12_tail_weight": 0.01},
    "v12_no_smoothing": {"v12_gate_smoothing": False},
    "v12_no_feature_dapp": {"v12_feature_dapp": False},
    "v12_no_unet_dapp": {"v12_unet_dapp": False},
    "v12_no_resconv": {"v12_resconv": False},
    "v12_no_dual_head": {"v12_dual_head": False},
}


class V12ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.h = OmegaConf.to_container(OmegaConf.load(V7_CONFIG).model, resolve=True)
        generator = torch.Generator().manual_seed(12)
        cls.clean = 0.2 * torch.randn((2, 1, 512), generator=generator)
        cls.noisy = cls.clean + torch.linspace(-0.3, 0.3, 512).view(1, 1, -1)
        cls.mask = torch.ones_like(cls.clean)
        cls.mask[..., -16:] = 0

    def model(self, variant="v12", **kwargs):
        torch.manual_seed(3407)
        settings = dict(self.h, **VARIANTS[variant])
        settings.update(kwargs)
        return get_model("lstm_temporal_gate_v12_ecg", **settings)

    def test_all_variants_forward_backward_metadata_and_state_reload(self):
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
                if model.core.temporal_gate is not None:
                    gradient = model.core.temporal_gate.net[-1].weight.grad
                    self.assertIsNotNone(gradient)
                    self.assertGreater(float(gradient.abs().sum()), 0)
                torch.optim.AdamW(model.parameters(), lr=5e-5).step()
                model.eval()
                expected = model.denoising(self.noisy)
                self.assertEqual(expected.shape, self.noisy.shape)
                self.assertTrue(torch.isfinite(expected).all())
                self.assertTrue(torch.equal(expected, model.denoising(self.noisy)))
                restored, metadata = model.denoising_with_metadata(self.noisy)
                self.assertTrue(torch.equal(expected, restored))
                for gate in ("cfm_refine_gate", "cfm_baseline_gate", "cfm_consistency_blend"):
                    self.assertEqual(metadata[gate].shape, (1,))
                    self.assertTrue(torch.isfinite(metadata[gate + "_temporal_std"]).all())
                duplicate = self.model(variant).eval()
                duplicate.load_state_dict(model.state_dict(), strict=True)
                self.assertTrue(torch.equal(expected, duplicate.denoising(self.noisy)))

    def test_v7_initial_state_rng_predictions_and_train_loss_are_exact(self):
        torch.manual_seed(3407)
        legacy = get_model("lstm_dualpath_dapp_cfm_unet_bd_ecg", **self.h)
        rng = torch.get_rng_state().clone()
        legacy_state = legacy.state_dict()
        for local in (True, False):
            with self.subTest(local_gate=local):
                model = self.model(v12_local_gate=local)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                state = model.state_dict()
                self.assertTrue(all(torch.equal(v, state[k]) for k, v in legacy_state.items()))
                if not local:
                    self.assertEqual(legacy_state.keys(), state.keys())
                legacy.eval()
                model.eval()
                self.assertTrue(torch.equal(legacy.denoising(self.noisy), model.denoising(self.noisy)))
                legacy.train()
                model.train()
                torch.manual_seed(100)
                old_loss = legacy.compute_loss((self.noisy, self.clean, self.mask.squeeze(1)), torch.device("cpu"))
                torch.manual_seed(100)
                new_loss = model.compute_loss((self.noisy, self.clean, self.mask), torch.device("cpu"))
                self.assertTrue(torch.equal(old_loss, new_loss))

    def test_ablation_shared_initialization_and_removal(self):
        full_model = self.model()
        full = full_model.state_dict()
        full_count = sum(p.numel() for p in full_model.parameters())
        for variant in VARIANTS:
            model = self.model(variant)
            self.assertTrue(all(torch.equal(value, full[key]) for key, value in model.state_dict().items()), variant)
            if variant in {"v7_delayed_topk", "v12_no_feature_dapp", "v12_no_unet_dapp", "v12_no_resconv", "v12_no_dual_head"}:
                self.assertLess(sum(p.numel() for p in model.parameters()), full_count)
        core = self.model("v12_no_dual_head").core
        self.assertIsNone(core.dual_noise_head)
        self.assertFalse({"dual", "dual_noise"}.intersection(core.loss_fn))
        self.assertIn("cfm", core.loss_fn)
        core.eval()
        with torch.no_grad():
            coarse, _, baseline, delta, direct = core._restore_components(self.noisy)
        self.assertTrue(torch.equal(coarse, direct))
        self.assertTrue(torch.equal(baseline, self.noisy - coarse))
        self.assertEqual(float(delta.abs().max()), 0.0)

    def test_temporal_gates_bounds_nonconstant_response_and_gradient(self):
        model = self.model().eval()
        condition = torch.randn(2, 6, 512)
        branch = model.core.temporal_gate
        self.assertEqual(float(branch(condition).abs().max()), 0.0)
        initial = model.core._cfm_gates(condition)
        self.assertTrue(all(torch.equal(g, g[..., :1].expand_as(g)) for g in initial))
        with torch.no_grad():
            branch.net[-1].weight.normal_(mean=0, std=2)
        offset = branch(condition)
        self.assertLessEqual(float(offset.abs().max()), branch.bound)
        gates = model.core._cfm_gates(condition)
        maxima = (model.core.cfm_refine_gate_max, model.core.cfm_baseline_gate_max,
                  model.core.cfm_consistency_blend_max)
        for gate, maximum in zip(gates, maxima):
            self.assertEqual(gate.shape, (2, 1, 512))
            self.assertGreater(float(gate.min()), 0)
            self.assertLess(float(gate.max()), maximum)
            self.assertGreater(float(gate.std(dim=-1).mean()), 0)
        changed = model.core._cfm_gates(condition.roll(31, dims=-1))
        self.assertFalse(torch.equal(gates[0], changed[0]))
        sum(g.mean() for g in gates).backward()
        self.assertGreater(float(branch.net[0].weight.grad.abs().sum()), 0)

    def test_smoothing_and_disabled_ablation(self):
        branch = BoundedTemporalGateOffsets(smooth_kernel=33)
        signal = torch.ones(2, 3, 512)
        signal[..., 1::2] = -1
        smooth = branch.smooth(signal)
        self.assertLess(float(smooth.diff(dim=-1).abs().mean()), float(signal.diff(dim=-1).abs().mean()))
        self.assertTrue(torch.equal(branch.smooth(torch.ones_like(signal)), torch.ones_like(signal)))
        branch.smoothing = False
        self.assertTrue(torch.equal(branch.smooth(signal), signal))
        for kernel in (0, 2, -3):
            with self.assertRaises(ValueError):
                BoundedTemporalGateOffsets(smooth_kernel=kernel)

    def test_tail_schedule_and_validation_objective(self):
        model = self.model("v12_delayed_topk").train()
        for epoch, expected in ((1, 0), (20, 0), (21, 0.001), (25, 0.005), (30, 0.01)):
            model.set_epoch(epoch, 30)
            self.assertAlmostEqual(model.tail_weight(), expected)
        for epoch, expected in ((1, 0.002), (5, 0.01)):
            model.set_epoch(epoch, 5)
            self.assertAlmostEqual(model.tail_weight(), expected)
        model.eval()
        losses = []
        for epoch in (1, 25, 30):
            model.set_epoch(epoch, 30)
            self.assertEqual(model.tail_weight(), 0.01)
            torch.manual_seed(111)
            with torch.no_grad():
                losses.append(model.compute_loss((self.noisy, self.clean, self.mask), torch.device("cpu")))
        self.assertTrue(all(torch.equal(losses[0], value) for value in losses[1:]))
        for epoch, total in ((0, 30), (31, 30), (1, 0)):
            with self.assertRaises(ValueError):
                model.set_epoch(epoch, total)
        with self.assertRaisesRegex(ValueError, "legacy 'max'"):
            self.model("v12_delayed_topk", loss_fn=self.h["loss_fn"] + "+max")

    def test_tail_loss_masks_and_weight_application(self):
        core = self.model("v12_delayed_topk").core
        clean = torch.zeros(2, 4)
        restored = torch.tensor([[1., 3., 100., 100.], [5., 100., 100., 100.]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(float(core._tail_loss(clean, restored, mask)), 3.5)
        self.assertEqual(float(core._tail_loss(clean, restored, torch.zeros_like(mask))), 0.0)
        self.assertEqual(float(core._tail_loss(clean, restored, mask.unsqueeze(1))), 3.5)
        clean = self.clean.squeeze(1)
        restored = self.noisy.squeeze(1)
        factor = torch.ones((2, 1, 1))
        core.train()
        core.set_epoch(20, 30)
        before = core._ecg_loss(clean, restored, factor)
        core.set_epoch(30, 30)
        after = core._ecg_loss(clean, restored, factor)
        self.assertTrue(torch.allclose(after - before, 0.01 * core._tail_loss(clean, restored), atol=1e-6))

    def test_2d_and_3d_masks_and_signals(self):
        model = self.model().eval()
        torch.manual_seed(88)
        first = model.compute_loss((self.noisy, self.clean, self.mask), torch.device("cpu"))
        torch.manual_seed(88)
        second = model.compute_loss((self.noisy.squeeze(1), self.clean.squeeze(1), self.mask.squeeze(1)), torch.device("cpu"))
        self.assertTrue(torch.equal(first, second))
        with self.assertRaises(ValueError):
            model.compute_loss((self.noisy, self.clean, self.mask[..., :-1]), torch.device("cpu"))

    def test_import_is_opt_in_and_preserves_legacy_registry_rng(self):
        code = """
import importlib
import torch
from models import get_model
factory = importlib.import_module('models.factory')
registry = dict(factory.__MODELS__)
assert 'lstm_temporal_gate_v12_ecg' not in registry
torch.manual_seed(12)
rng = torch.get_rng_state().clone()
import models.lstm_v12
assert torch.equal(rng, torch.get_rng_state())
assert all(factory.__MODELS__[key] is value for key, value in registry.items())
assert set(factory.__MODELS__) - set(registry) == {'lstm_temporal_gate_v12_ecg'}
"""
        subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT / "src", check=True,
                       capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
