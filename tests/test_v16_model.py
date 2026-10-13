"""CPU contract checks for strict v7 parity and bounded adapter experiments."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from models import get_model
from models.lstm_v16 import CONDITION_FEATURE_COUNT, FEATURE_NAMES, GateAdapter, initialize_from_v7

V7_CONFIG = PROJECT / "src/configs/mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml"


class V16ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.h = OmegaConf.to_container(OmegaConf.load(V7_CONFIG).model, resolve=True)
        torch.manual_seed(3407)
        cls.reference = get_model("lstm_dualpath_dapp_cfm_unet_bd_ecg", **cls.h).eval()
        cls.temp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temp.name) / "best_model.pt"
        torch.save(cls.reference.state_dict(), cls.checkpoint)
        generator = torch.Generator().manual_seed(113)
        cls.clean = 0.2 * torch.randn(2, 1, 512, generator=generator)
        cls.noisy = cls.clean + torch.linspace(-0.2, 0.4, 512).view(1, 1, -1)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self, mode="correction", **settings):
        torch.manual_seed(37)
        model = get_model("lstm_v16_ecg", **dict(self.h, v16_gate_mode=mode, **settings))
        initialize_from_v7(model, self.checkpoint)
        return model

    def test_three_modes_have_exact_initial_inference_parity(self):
        expected = self.reference.denoising(self.noisy)
        for mode in ("constant", "condition", "correction"):
            with self.subTest(mode=mode):
                model = self.model(mode).eval()
                self.assertTrue(torch.equal(expected, model.denoising(self.noisy)))
                self.assertTrue(torch.equal(expected, model.stages(self.noisy)["final"]))
                for key, value in self.reference.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                provenance = initialize_from_v7(model, self.checkpoint)
                self.assertEqual(provenance["checkpoint_sha256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
                self.assertEqual(provenance["loaded_v7_tensor_count"], len(self.reference.state_dict()))

    def test_original_full_loss_and_rng_are_preserved_at_initialization(self):
        model = self.model().eval()
        losses, rng = [], []
        mask = torch.ones(2, 512)
        mask[:, -8:] = 0
        for item in (self.reference, model):
            torch.manual_seed(191)
            losses.append(item.compute_loss((self.noisy, self.clean, mask), "cpu"))
            rng.append(torch.get_rng_state().clone())
        self.assertTrue(torch.equal(losses[0], losses[1]))
        self.assertTrue(torch.equal(rng[0], rng[1]))
        self.assertAlmostEqual(sum(float(value.detach()) for value in model.loss_component_tensors.values()),
                               float(losses[1].detach()), places=5)
        self.assertFalse(model.loss_component_tensors["stochastic_fm"].requires_grad)
        self.assertFalse(model.loss_component_tensors["dual_noise_remainder"].requires_grad)
        self.assertTrue(model.loss_component_tensors["reconstruction"].requires_grad)
        self.assertIn("detail_cfm_path_consistency", model.loss_components)
        self.assertEqual(model.loss_components["total_loss"], float(losses[1].detach()))

    def test_construction_preserves_v7_rng_and_condition_correction_capacity(self):
        states = []
        models = []
        for name, mode in (("lstm_dualpath_dapp_cfm_unet_bd_ecg", None),
                           ("lstm_v16_ecg", "constant"), ("lstm_v16_ecg", "condition"),
                           ("lstm_v16_ecg", "correction")):
            torch.manual_seed(41)
            kwargs = self.h if mode is None else dict(self.h, v16_gate_mode=mode)
            models.append(get_model(name, **kwargs))
            states.append(torch.get_rng_state().clone())
        self.assertTrue(all(torch.equal(states[0], value) for value in states[1:]))
        condition = models[2].core.gate_adapter
        correction = models[3].core.gate_adapter
        self.assertEqual(condition.state_dict().keys(), correction.state_dict().keys())
        for key in condition.state_dict():
            self.assertTrue(torch.equal(condition.state_dict()[key], correction.state_dict()[key]))
        self.assertEqual(models[2].finetuning_summary()["trainable_parameters"],
                         models[3].finetuning_summary()["trainable_parameters"])

    def test_condition_control_masks_correction_features_and_features_are_detached(self):
        condition, correction = self.model("condition"), self.model("correction")
        with torch.no_grad():
            for model in (condition, correction):
                adapter = model.core.gate_adapter
                adapter.net[0].weight.fill_(0.1)
                adapter.net[0].bias.zero_()
                adapter.net[-1].weight.fill_(0.1)
        features = torch.ones(2, len(FEATURE_NAMES), requires_grad=True)
        changed = features.detach().clone()
        changed[:, CONDITION_FEATURE_COUNT:] = 4
        a = condition.core.gate_adapter(features)
        self.assertTrue(torch.equal(a, condition.core.gate_adapter(changed)))
        self.assertFalse(torch.equal(correction.core.gate_adapter(features), correction.core.gate_adapter(changed)))
        a.sum().backward()
        self.assertIsNone(features.grad)
        parts = correction.stages(self.noisy)
        self.assertEqual(parts["adapter_features"].shape, (2, len(FEATURE_NAMES)))
        self.assertFalse(parts["adapter_features"].requires_grad)

    def test_adapter_only_update_keeps_every_v7_parameter_and_stage_one_fixed(self):
        model = self.model().train()
        groups = model.configure_finetuning()
        self.assertEqual([(group["name"], group["lr"]) for group in groups], [("adapter", 1e-4)])
        frozen = {key: value.clone() for key, value in model.state_dict().items()
                  if not key.startswith("core.gate_adapter.")}
        self.assertTrue(model.core.gate_adapter.training)
        self.assertFalse(model.core.residual_flow.training)
        self.assertFalse(model.core.tsc_blocks.training)
        self.assertFalse(model.core.cfm_adaptive_gate.training)
        before = model.core._restore_components(self.noisy)[0]
        loss = model.compute_loss((self.noisy, self.clean), "cpu")
        loss.backward()
        gradients = []
        for name, parameter in model.named_parameters():
            if name.startswith("core.gate_adapter."):
                if parameter.grad is not None:
                    gradients.append(parameter.grad)
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(any(torch.count_nonzero(gradient) for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        torch.optim.AdamW(groups, weight_decay=0).step()
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items()))
        self.assertTrue(torch.equal(before, model.core._restore_components(self.noisy)[0]))
        self.assertFalse(model.core.gate_adapter.is_zero_output())
        model.eval().train()
        self.assertFalse(model.core.residual_flow.training)

    def test_optional_flow_update_never_unfreezes_original_gates_or_stage_one(self):
        model = self.model("condition")
        groups = model.configure_finetuning(scope="adapter_flow", lr_adapter=2e-4, lr_flow=3e-5)
        self.assertEqual([(group["name"], group["lr"]) for group in groups], [("adapter", 2e-4), ("flow", 3e-5)])
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        model.train()
        self.assertTrue(model.core.residual_flow.training)
        self.assertFalse(model.core.cfm_adaptive_gate.training)
        self.assertFalse(model.core.tsc_blocks.training)
        model.compute_loss((self.noisy[:1], self.clean[:1]), "cpu").backward()
        self.assertTrue(model.loss_component_tensors["stochastic_fm"].requires_grad)
        self.assertFalse(model.loss_component_tensors["dual_noise_remainder"].requires_grad)
        flow_gradients = []
        for name, parameter in model.named_parameters():
            if name.startswith("core.residual_flow."):
                if parameter.grad is not None:
                    flow_gradients.append(parameter.grad)
            elif not name.startswith("core.gate_adapter."):
                self.assertIsNone(parameter.grad, name)
                self.assertFalse(parameter.requires_grad, name)
        self.assertTrue(any(torch.count_nonzero(gradient) for gradient in flow_gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in flow_gradients))

    def test_stages_paths_budgets_bounds_and_counterfactuals(self):
        for normalization in (False, "1"):
            with self.subTest(normalization=normalization):
                model = self.model("constant", norm=normalization)
                with torch.no_grad():
                    model.core.gate_adapter.offset.copy_(torch.tensor((1000., -1000., 1000.)))
                model.train()
                parts = model.stages(self.noisy)
                self.assertTrue(model.training)
                self.assertFalse(model.core.residual_flow.training)
                self.assertTrue(torch.equal(parts["final"], model.denoising(self.noisy)))
                self.assertTrue(torch.equal(parts["flow_correction"], parts["final"] - parts["pre_flow"]))
                self.assertTrue(torch.allclose(parts["effective_flow"].sum(1, keepdim=True), parts["flow_correction"], atol=2e-7))
                self.assertTrue((parts["adapter_offsets"].abs() <= 0.5).all())
                maxima = torch.tensor([model.core.cfm_refine_gate_max, model.core.cfm_baseline_gate_max,
                                       model.core.cfm_consistency_blend_max]).view(1, 3, 1)
                self.assertTrue(((parts["gates"] > 0) & (parts["gates"] < maxima)).all())
                self.assertTrue(((parts["budget_scales"] >= 0) & (parts["budget_scales"] <= 1)).all())
                self.assertTrue(torch.allclose(parts["limited_flow"], parts["projected_flow"] * parts["budget_scales"], atol=2e-7))
                for key, value in parts["diagnostics"].items():
                    self.assertEqual(value.shape, (2,), key)
                    self.assertTrue(torch.isfinite(value).all(), key)
                gc, gb, blend = parts["gates"].split(1, dim=1)
                cp = parts["pre_flow"] + gc * parts["projected_flow"][:, :1]
                bp = self.noisy - ((self.noisy - parts["pre_flow"]) + gb * parts["projected_flow"][:, 1:2])
                self.assertTrue(torch.allclose(parts["budget_bypass_both"], cp + blend * (bp - cp), atol=2e-7))

    def test_strict_checkpoint_checks_before_any_mutation(self):
        model = self.model()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        baseline = dict(self.reference.state_dict())
        key = next(key for key in baseline if key.startswith("core.residual_flow."))
        invalid = Path(self.temp.name) / "invalid.pt"
        for kind in ("missing", "unexpected", "shape", "dtype", "nonfinite"):
            state = dict(baseline)
            if kind == "missing":
                del state[key]
            elif kind == "unexpected":
                state["core.gate_adapter.offset"] = torch.zeros(3)
            elif kind == "shape":
                state[key] = torch.zeros(1)
            elif kind == "dtype":
                state[key] = state[key].double()
            else:
                state[key] = torch.full_like(state[key], float("nan"))
            torch.save(state, invalid)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                initialize_from_v7(model, invalid)
            self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in original.items()))
        torch.save({"model_state_dict": baseline, "epoch": 30}, invalid)
        self.assertEqual(initialize_from_v7(model, invalid)["checkpoint_container"], "model_state_dict")
        with torch.no_grad():
            model.core.gate_adapter.net[-1].bias.add_(0.1)
        with self.assertRaisesRegex(ValueError, "zero output"):
            initialize_from_v7(model, self.checkpoint)

    def test_reject_invalid_experimental_configuration(self):
        for kwargs in ({"mode": "unknown"}, {"hidden": True}, {"hidden": 0},
                       {"bound": 0}, {"bound": float("nan")}, {"bound": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GateAdapter(**kwargs)
        model = self.model()
        for kwargs in ({"scope": "joint"}, {"lr_adapter": 0}, {"lr_flow": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.configure_finetuning(**kwargs)
        for kwargs in ({"cfm_inference_steps": 2}, {"cfm_inference_shots": True},
                       {"cfm_use_adaptive_gate": "false"}, {"cfm_inference_start_noise_scale": 0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)


if __name__ == "__main__":
    unittest.main()
