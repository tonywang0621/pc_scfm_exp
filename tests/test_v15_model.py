"""Meaningful CPU checks for v7 warm-start, loss contrasts and frozen stage one."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from models import get_model
from models.lstm_v15 import initialize_from_v7, parameter_group

V7_CONFIG = PROJECT / "src/configs/mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml"


class V15ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.h = OmegaConf.to_container(OmegaConf.load(V7_CONFIG).model, resolve=True)
        torch.manual_seed(3407)
        cls.reference = get_model("lstm_dualpath_dapp_cfm_unet_bd_ecg", **cls.h).eval()
        cls.temp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temp.name) / "best_model.pt"
        torch.save(cls.reference.state_dict(), cls.checkpoint)
        g = torch.Generator().manual_seed(13)
        cls.clean = torch.randn(2, 1, 512, generator=g) * 0.2
        cls.noisy = cls.clean + torch.linspace(-0.2, 0.4, 512).view(1, 1, -1)
        cls.mask = torch.ones_like(cls.clean)
        cls.mask[..., -16:] = 0

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self, **kwargs):
        settings = dict(self.h, **kwargs)
        torch.manual_seed(71)
        model = get_model("lstm_finetune_v15_ecg", **settings)
        initialize_from_v7(model, self.checkpoint)
        return model

    def test_checkpoint_parity_with_and_without_zero_local_gate(self):
        expected = self.reference.denoising(self.noisy)
        for local in (False, True):
            with self.subTest(local=local):
                model = self.model(v12_local_gate=local).eval()
                self.assertTrue(torch.equal(model.denoising(self.noisy), expected))
                for key, value in self.reference.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                provenance = initialize_from_v7(model, self.checkpoint)
                self.assertEqual(provenance["checkpoint_sha256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
                self.assertEqual(provenance["loaded_v7_tensor_count"], len(self.reference.state_dict()))
                self.assertEqual(bool(provenance["new_local_gate_tensor_count"]), local)

    def test_strict_checkpoint_rejects_missing_extra_bad_shape_nonfinite_before_mutation(self):
        model = self.model()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        key = next(key for key in original if key.startswith("core.residual_flow."))
        path = Path(self.temp.name) / "invalid.pt"
        for kind in ("missing", "unexpected", "shape", "nan"):
            state = dict(original)
            if kind == "missing":
                del state[key]
            elif kind == "unexpected":
                state["core.unrelated.weight"] = torch.ones(1)
            elif kind == "shape":
                state[key] = torch.zeros(1)
            else:
                state[key] = torch.full_like(state[key], float("nan"))
            torch.save(state, path)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                initialize_from_v7(model, path)
            self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in original.items()))
        torch.save({"model_state_dict": original, "epoch": 30}, path)
        self.assertEqual(initialize_from_v7(model, path)["checkpoint_container"], "model_state_dict")
        local = self.model(v12_local_gate=True)
        with torch.no_grad():
            local.core.temporal_gate.net[-1].bias.add_(0.1)
        with self.assertRaisesRegex(ValueError, "zero output"):
            initialize_from_v7(local, self.checkpoint)

    def test_zero_aux_control_matches_original_v7_loss_rng_and_gradients(self):
        model = self.model(v12_tail_weight=0, v13_raw_cos_weight=0, v13_extra_mse_weight=0).train()
        reference = get_model("lstm_dualpath_dapp_cfm_unet_bd_ecg", **self.h).train()
        nn.Module.load_state_dict(reference, self.reference.state_dict(), strict=True)
        losses, states = [], []
        for item in (reference, model):
            torch.manual_seed(4321)
            losses.append(item.compute_loss((self.noisy, self.clean, self.mask), "cpu"))
            states.append(torch.get_rng_state().clone())
            losses[-1].backward()
        self.assertTrue(torch.equal(losses[0], losses[1]))
        self.assertTrue(torch.equal(states[0], states[1]))
        for name, parameter in reference.named_parameters():
            other = dict(model.named_parameters())[name]
            if parameter.grad is None:
                self.assertIsNone(other.grad, name)
            else:
                self.assertTrue(torch.equal(parameter.grad, other.grad), name)
        self.assertEqual(model.loss_components["original_v7_loss"], float(losses[1].detach()))

    def test_aux_losses_constant_from_first_epoch_added_once_and_logged(self):
        model = self.model(v13_extra_mse_weight=0.25)
        for training in (True, False):
            model.train(training)
            for epoch in (1, 5, 10):
                model.set_epoch(epoch, 10)
                self.assertEqual(model.tail_weight(), 0.01)
                self.assertEqual(model.metric_weights(), {"raw_cos": 0.01, "extra_mse": 0.25})
        model.train()
        control = self.model(v12_tail_weight=0, v13_raw_cos_weight=0, v13_extra_mse_weight=0).train()
        losses = []
        for item in (control, model):
            torch.manual_seed(113)
            losses.append(item.compute_loss((self.noisy, self.clean, self.mask), "cpu"))
        c = model.loss_components
        self.assertAlmostEqual(float((losses[1] - losses[0]).detach()),
                               c["raw_cos"] * 0.01 + c["topk"] * 0.01 + c["extra_mse"] * 0.25, places=6)
        self.assertAlmostEqual(c["original_v7_loss"], float(losses[0].detach()), places=6)
        self.assertEqual(c["total_loss"], float(losses[1].detach()))
        for name, weight in (("raw_cos", 0.01), ("topk", 0.01), ("extra_mse", 0.25)):
            self.assertAlmostEqual(c["weighted_" + name], c[name] * weight, places=7)

    def test_parameter_groups_use_real_decoder_names_and_cover_trainables_once(self):
        model = self.model(v12_local_gate=True)
        groups = model.configure_finetuning()
        self.assertEqual([(g["name"], g["lr"]) for g in groups],
                         [("backbone", 1e-6), ("decoder", 3e-6), ("flow", 1e-5)])
        by_id = {id(p): name for name, p in model.named_parameters()}
        all_ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(by_id))
        decoder = [by_id[id(p)] for group in groups if group["name"] == "decoder" for p in group["params"]]
        self.assertTrue(any(name.startswith("core.phase_decoder.") for name in decoder))
        self.assertTrue(any(name.startswith("core.mask_decoder.") for name in decoder))
        self.assertTrue(any(name.startswith("core.dual_noise_head.") for name in decoder))
        for group in groups:
            self.assertTrue(all(parameter_group(by_id[id(p)]) == group["name"] for p in group["params"]))
        with self.assertRaises(ValueError):
            model.configure_finetuning(scope="invalid")
        with self.assertRaises(ValueError):
            model.configure_finetuning(lr_flow=float("nan"))

    def test_flow_only_freezes_parameters_dropout_outputs_and_detaches_stage_one(self):
        model = self.model(v12_local_gate=True)
        groups = model.configure_finetuning(scope="flow_only")
        self.assertEqual([g["name"] for g in groups], ["flow"])
        frozen = {name: value.clone() for name, value in model.state_dict().items() if parameter_group(name) != "flow"}
        model.train()
        self.assertFalse(model.core.tsc_blocks.training)
        self.assertFalse(model.core.mask_decoder.training)
        self.assertTrue(model.core.residual_flow.training)
        before = model.core._restore_components(self.noisy)[0]
        self.assertFalse(before.requires_grad)
        self.assertTrue(torch.equal(before, model.core._restore_components(self.noisy)[0]))
        loss = model.compute_loss((self.noisy, self.clean), "cpu")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter_group(name) != "flow":
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad, name)
        torch.optim.AdamW(groups).step()
        self.assertTrue(torch.equal(before, model.core._restore_components(self.noisy)[0]))
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()))
        model.eval().train()
        self.assertFalse(model.core.tsc_blocks.training)
        model.configure_finetuning(scope="joint")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(model.core.tsc_blocks.training)

    def test_stages_match_denoising_and_report_original_amplitude_budget_diagnostics(self):
        for normalization in (False, "1"):
            model = self.model(norm=normalization, v12_local_gate=True).eval()
            expected = model.denoising(self.noisy)
            stages = model.stages(self.noisy.squeeze(1))
            self.assertTrue(torch.equal(expected, stages["final"]))
            self.assertTrue(torch.equal(stages["final"] - stages["pre_flow"], stages["flow_correction"]))
            self.assertEqual(stages["pre_flow"].shape, self.noisy.shape)
            self.assertFalse(stages["final"].requires_grad)
            for key, value in stages["diagnostics"].items():
                self.assertEqual(value.shape, (2,), key)
                self.assertTrue(torch.isfinite(value).all(), key)
            diagnostics = stages["diagnostics"]
            for name in ("clean", "baseline"):
                scale = diagnostics[name + "_budget_scale"]
                self.assertTrue(((scale >= 0) & (scale <= 1)).all())
                self.assertTrue(torch.equal(diagnostics[name + "_budget_limited"], (scale < 1).float()))
                self.assertTrue(torch.allclose(diagnostics[name + "_limited_abs_mean"],
                                               diagnostics[name + "_projected_abs_mean"] * scale, atol=1e-6))
            self.assertTrue(torch.equal(diagnostics["flow_correction_abs_mean"],
                                        stages["flow_correction"].abs().mean((1, 2))))
            model.train()
            self.assertTrue(torch.equal(model.stages(self.noisy)["final"], expected))
            self.assertTrue(model.training)

    def test_all_eight_experimental_conditions_have_finite_gradients(self):
        variants = [
            dict(v12_tail_weight=0, v13_raw_cos_weight=0),
            dict(v12_tail_weight=0), dict(v13_raw_cos_weight=0), {},
            dict(v13_extra_mse_weight=0.25), dict(v12_local_gate=True),
            dict(v12_local_gate=True, v13_extra_mse_weight=0.25),
            dict(scope="flow_only"),
        ]
        for index, settings in enumerate(variants):
            with self.subTest(condition=index):
                settings = dict(settings)
                scope = settings.pop("scope", "joint")
                model = self.model(**settings).train()
                model.configure_finetuning(scope=scope)
                model.set_epoch(1, 10)
                loss = model.compute_loss((self.noisy[:1], self.clean[:1]), "cpu")
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_invalid_yaml_types_are_rejected_before_model_construction(self):
        cases = []
        for key in ("cfm_inference_steps", "cfm_inference_shots", "mad_topk"):
            cases.extend((key, value) for value in (True, "1", 1.5, 0, -1, float("nan")))
        for key in ("v12_local_gate", "v12_gate_smoothing", "v12_dual_head", "use_dapp",
                    "cfm_use_adaptive_gate", "cfm_unet_use_attention"):
            cases.extend((key, value) for value in ("false", 0, 1, None))
        for key in ("v12_tail_weight", "v13_raw_cos_weight", "v13_extra_mse_weight"):
            cases.extend((key, value) for value in (True, "0.01", -0.1, float("nan"), float("inf")))
        cases.extend(("norm", value) for value in (True, 1, None, "false", float("nan")))
        for key, value in cases:
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                get_model("lstm_finetune_v15_ecg", **dict(self.h, **{key: value}))
        for key in ("cfm_inference_steps", "cfm_inference_shots"):
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "one-step"):
                get_model("lstm_finetune_v15_ecg", **dict(self.h, **{key: 2}))
        model = self.model()
        for rate in (True, "0.00001", 0, float("nan")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                model.configure_finetuning(lr_flow=rate)

    def test_invalid_normalization_fails_training_and_regular_inference(self):
        for mode in ("1", "2"):
            model = self.model(norm=mode)
            zeros = torch.zeros_like(self.noisy[:1])
            with self.subTest(mode=mode, path="loss"), self.assertRaisesRegex(ValueError, "normalization factor"):
                model.compute_loss((zeros, zeros), "cpu")
            with self.subTest(mode=mode, path="denoising"), self.assertRaisesRegex(ValueError, "normalization factor"):
                model.eval().denoising(zeros)


if __name__ == "__main__":
    unittest.main()
