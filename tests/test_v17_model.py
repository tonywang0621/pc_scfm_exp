import copy
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from models.lstm_v17 import CooperativeGate, linear_flow_path
from models.lstm_v17.compact import compact_parameters, MeanFreeBiasConv1d
from models.lstm_v17.audit import audit_parameters
from mecge_table1_run_v17_model import make_model
from v17_experiments import VARIANTS, build_config

SMALL = ["model.dense_channel=8", "model.lstm_hidden=4", "model.num_tscblocks=1",
         "model.dual_noise_head_hidden=8", "model.cfm_unet_base_channels=8",
         "model.cfm_unet_channel_mults=[1,2]", "model.cfm_unet_time_dim=16",
         "model.v17_tcn_channels=8"]


def model_for(variant="v17", overrides=()):
    return make_model(build_config(variant, 3407, 1, Path("unused.pkl"), [*SMALL, *overrides]))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        t = torch.linspace(0, 1, 512).view(1, 1, -1)
        cls.clean = torch.cat((torch.sin(12*torch.pi*t), .8*torch.sin(10*torch.pi*t)), 0)
        cls.noisy = cls.clean + .4*torch.cos(torch.pi*t)

    def test_every_variant_forward_loss_and_gradient_participation(self):
        for name in VARIANTS:
            with self.subTest(name=name):
                torch.manual_seed(3407)
                model = model_for(name).eval()
                prediction = model.denoising(self.noisy)
                self.assertEqual(prediction.shape, self.noisy.shape)
                self.assertTrue(torch.isfinite(prediction).all())
                state = {k: v.clone() for k, v in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                strict = name not in {"v7_max_control", "v12_topk_control"}
                report = audit_parameters(model, self.noisy, self.clean, strict=strict)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))
                if strict:
                    self.assertFalse(report["dormant_parameter_tensors"])

    def test_compact_control_preserves_legacy_inference_and_loss(self):
        torch.manual_seed(15)
        reference = model_for("v12_topk_control").eval()
        torch.manual_seed(15)
        compact = model_for("v12_topk_compact").eval()
        with torch.no_grad():
            torch.testing.assert_close(reference(self.noisy), compact(self.noisy), rtol=3e-5, atol=3e-6)
            torch.manual_seed(77)
            a = reference.compute_loss((self.noisy, self.clean), "cpu")
            torch.manual_seed(77)
            b = compact.compute_loss((self.noisy, self.clean), "cpu")
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        self.assertGreater(compact.core.compaction["removed"], 500)
        self.assertFalse(any("bias_hh" in key for key, _ in compact.named_parameters()))
        self.assertTrue(any("bias_hh" in key for key, _ in compact.named_buffers()))

    def test_compaction_on_nontrivial_weights_and_norm_offsets(self):
        reference = model_for("v12_topk_control").eval()
        with torch.no_grad():
            for name, p in reference.named_parameters():
                if "bias" in name:
                    p.add_(0.03*torch.randn_like(p))
        compact = copy.deepcopy(reference).eval()
        compact.core.h.__dict__ = compact.core.h
        compact_parameters(compact.core)
        with torch.no_grad():
            torch.testing.assert_close(reference(self.noisy), compact(self.noisy), rtol=6e-5, atol=6e-6)

    def test_group_centered_bias_preserves_normalization(self):
        for groups in (1, 2, 8):
            conv = torch.nn.Conv1d(3, 8, 3, padding=1).double()
            norm = torch.nn.GroupNorm(groups, 8).double()
            compact = MeanFreeBiasConv1d(conv, groups)
            x = torch.randn(2, 3, 35, dtype=torch.float64)
            torch.testing.assert_close(norm(conv(x)), norm(compact(x)), rtol=1e-12, atol=1e-12)
            expected = 8-groups
            self.assertEqual(sum(p.numel() for key, p in compact.named_parameters() if key != "weight"), expected)

    def test_gate_centering_bounds_and_local_switch(self):
        gate = CooperativeGate(5, 3, bound=.5)
        x = torch.randn(2, 5, 512)
        gates, offset = gate(x)
        self.assertTrue((gates > 0).all() and (gates < 1).all())
        self.assertLess(float(offset.abs().max()), .5)
        torch.testing.assert_close(offset.mean(-1), torch.zeros(2, 3), atol=1e-7, rtol=0)
        gate.local = False
        gates, offset = gate(x)
        self.assertEqual(int(torch.count_nonzero(offset)), 0)
        torch.testing.assert_close(gates, gates[..., :1].expand_as(gates))

    def test_distinct_endpoints_and_final_fusion_identity(self):
        model = model_for().eval()
        stage = model.stages(self.noisy)
        torch.testing.assert_close(stage["baseline_candidate"], self.noisy-stage["baseline_hat"])
        ec, eb = self.clean-stage["pre_flow"], stage["baseline_candidate"]-self.clean
        torch.testing.assert_close(ec+eb, stage["baseline_candidate"]-stage["pre_flow"])
        self.assertGreater(float((ec+eb).abs().mean()), .01)
        gates = stage["gates"]
        expected = (gates[:, :1]*(stage["pre_flow"]+stage["flow"])
                    + (1-gates[:, :1])*((1-gates[:, 1:2])*stage["pre_flow"]
                                          + gates[:, 1:2]*stage["baseline_candidate"]))
        self.assertEqual(stage["flow"].shape[1], 1)
        self.assertEqual(gates.shape[1], 2)
        torch.testing.assert_close(stage["final"], expected)
        torch.testing.assert_close(stage["final"], model(self.noisy))

    def test_bridge_target_matches_exact_path_derivative(self):
        target, noise = torch.randn(3, 2, 11, dtype=torch.float64), torch.randn(3, 2, 11, dtype=torch.float64)
        t = torch.tensor([.13, .48, .86], dtype=torch.float64)
        eps = 1e-6
        _, velocity = linear_flow_path(target, t, .02, noise)
        before, _ = linear_flow_path(target, t-eps, .02, noise)
        after, _ = linear_flow_path(target, t+eps, .02, noise)
        torch.testing.assert_close(velocity, (after-before)/(2*eps), rtol=1e-8, atol=1e-9)
        start, _ = linear_flow_path(target, torch.zeros_like(t), .02, noise)
        end, _ = linear_flow_path(target, torch.ones_like(t), .02, noise)
        torch.testing.assert_close(start, torch.zeros_like(target))
        torch.testing.assert_close(end, target)

    def test_euler_uses_matching_state_and_left_endpoint_time(self):
        model = model_for()
        class ConstantVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, state, condition, t):
                self.calls.append((state.detach().clone(), t.detach().clone()))
                return torch.ones_like(state)*2
        flow = ConstantVelocity()
        model.core.residual_flow, model.core.steps = flow, 4
        result = model.core.integrate(torch.randn(2, 3, 512))
        torch.testing.assert_close(result, torch.ones_like(result)*2)
        for index, (state, t) in enumerate(flow.calls):
            torch.testing.assert_close(state, torch.ones_like(state)*index*.5)
            torch.testing.assert_close(t, torch.ones_like(t)*index/4)

    def test_ablations_remove_parameters_and_input_columns(self):
        main, condition, noflow, nodual, fixed = [model_for(name) for name in
            ("v17", "v17_condition_gate", "v17_no_flow", "v17_no_dual_head", "v17_fixed_gate")]
        self.assertEqual(main.core.gate.features[0].in_channels, 4)
        self.assertEqual(condition.core.gate.features[0].in_channels, 3)
        self.assertIsNone(noflow.core.residual_flow)
        self.assertEqual(noflow.core.gate.logits.out_channels, 1)
        self.assertIsNone(nodual.core.dual_noise_head)
        self.assertEqual(nodual.core.residual_flow.input.in_channels, 3)
        self.assertIsNone(fixed.core.gate)
        self.assertFalse(any("fixed_gates" in k for k, _ in fixed.named_parameters()))

    def test_direct_regression_preserves_initial_inference_without_inactive_fm_parameters(self):
        torch.manual_seed(13)
        flow = model_for().eval()
        torch.manual_seed(13)
        direct = model_for("v17_direct").eval()
        torch.testing.assert_close(flow(self.noisy), direct(self.noisy), rtol=2e-5, atol=2e-6)
        self.assertFalse(any("time_proj" in name or "time_embedding" in name
                             for name, _ in direct.core.residual_flow.named_parameters()))
        self.assertEqual(direct.core.residual_flow.input.in_channels, 3)
        direct.compute_loss((self.noisy, self.clean), "cpu")
        self.assertNotIn("fm_velocity", direct.core.loss_components)
        self.assertIn("flow_endpoint", direct.core.loss_components)

    def test_baseline_fusion_ablation_removes_inactive_gate_but_keeps_head_diagnostics(self):
        model = model_for("v17_no_baseline_fusion").eval()
        stage = model.stages(self.noisy)
        self.assertEqual(stage["gates"].shape[1], 1)
        torch.testing.assert_close(stage["baseline_candidate"], self.noisy-stage["baseline_hat"])
        torch.testing.assert_close(stage["final"], stage["pre_flow"] + stage["gates"]*stage["flow"])

    def test_topk_added_once_and_invalid_shapes_rejected(self):
        torch.manual_seed(7)
        main = model_for().eval()
        torch.manual_seed(7)
        control = model_for("v17_no_topk").eval()
        torch.manual_seed(9)
        a = main.compute_loss((self.noisy, self.clean), "cpu")
        torch.manual_seed(9)
        b = control.compute_loss((self.noisy, self.clean), "cpu")
        topk = (main(self.noisy)-self.clean).abs().topk(8, -1).values.mean()*.05
        torch.testing.assert_close(a-b, topk, rtol=1e-4, atol=1e-6)
        with self.assertRaises(ValueError):
            main(torch.randn(2, 2, 512))


if __name__ == "__main__":
    unittest.main()
