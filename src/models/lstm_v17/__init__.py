"""v17: independent dual-head candidates, conditional flow, shared local gates.

Opt-in registration: legacy model implementations and checkpoints are untouched.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..factory import register_model
from ..mecg_e import ECGDenoisingModel, ConditionalResidualCFMRefiner1d
from ..mambattention import (
    DualPathDAPPMambAttentionCore, LightweightLSTMContextBlock,
    ConditionalResidualFlowUNetRefiner1d,
)
from .compact import compact_parameters


class CooperativeGate(nn.Module):
    """One feature extractor and logit projection for global and local control.

The temporal offset has zero mean; it has no independent constant bias.
Only independent primitive waveforms enter the first linear convolution.
"""
    def __init__(self, channels, outputs, hidden=16, local=True, bound=0.5, kernel=33, initial=None):
        super().__init__()
        if hidden < 1 or kernel < 1 or kernel % 2 != 1 or not math.isfinite(bound) or bound <= 0:
            raise ValueError("Invalid cooperative gate width, smoothing kernel or bound")
        self.local, self.bound, self.kernel = local, bound, kernel
        self.features = nn.Sequential(nn.Conv1d(channels, hidden, 7, padding=3), nn.SiLU(),
                                      nn.Conv1d(hidden, hidden, 7, padding=12, dilation=4), nn.SiLU())
        self.logits = nn.Conv1d(hidden, outputs, 1)
        nn.init.normal_(self.logits.weight, std=0.01)
        initial = initial if initial is not None else [0.25] * outputs
        with torch.no_grad():
            self.logits.bias.copy_(torch.tensor([math.log(x / (1-x)) for x in initial]))

    def forward(self, condition):
        logits = self.logits(self.features(condition))
        global_logits = logits.mean(-1, keepdim=True)
        local = torch.zeros_like(logits)
        if self.local:
            local = torch.tanh(logits - global_logits)
            radius = self.kernel // 2
            local = F.avg_pool1d(F.pad(local, (radius, radius), mode="replicate"), self.kernel, stride=1)
            # Half-bound before centering guarantees |offset| <= bound.
            local = 0.5 * self.bound * (local - local.mean(-1, keepdim=True))
        return torch.sigmoid(global_logits + local), local


def linear_flow_path(target, t, bridge_scale=0.0, noise=None):
    """Zero-source stochastic interpolant and its exact path derivative."""
    tau = t[:, None, None]
    state, velocity = tau * target, target
    if bridge_scale:
        if noise is None:
            noise = torch.randn_like(target)
        state = state + bridge_scale * torch.sin(math.pi * tau) * noise
        velocity = velocity + bridge_scale * math.pi * torch.cos(math.pi * tau) * noise
    return state, velocity


class CooperativeV17Core(DualPathDAPPMambAttentionCore):
    def __init__(self, config):
        h = config["model"]
        if h.get("fea", "pha") != "pha" or h.get("time_frequency_transform", "stft") != "stft":
            raise ValueError("v17 currently supports the official single-lead STFT/phase configuration")
        if h.get("pcscfm_enabled", False):
            raise ValueError("v17 does not instantiate the separate PC-SCFM policy")
        super().__init__(config, block_cls=LightweightLSTMContextBlock)
        self.dual = bool(h.get("v17_dual_head", True))
        self.independent = bool(h.get("v17_independent_paths", True)) and self.dual
        self.use_flow = bool(h.get("v17_flow", True))
        self.use_fm = self.use_flow and bool(h.get("v17_fm_objective", True))
        self.correction_gate = bool(h.get("v17_correction_gate", True)) and self.use_flow
        self.steps = int(h.get("cfm_inference_steps", 1))
        self.bridge = float(h.get("v17_bridge_scale", 0.0))
        self.fm_weight = float(h.get("lambda_cfm", 0.17))
        self.coarse_weight = float(h.get("v17_coarse_weight", 0.25))
        self.path_weight = float(h.get("v17_path_weight", 0.12))
        self.tail = float(h.get("lambda_max", 0.05))
        self.topk = int(h.get("mad_topk", 8))
        self.condition_channels = 3 if self.dual else 2
        if not self.use_flow and not self.independent:
            raise ValueError("A no-flow model needs independent candidates for a meaningful blend gate")
        if self.steps < 1 or self.topk < 1 or self.bridge < 0:
            raise ValueError("Invalid v17 steps, top-k, or bridge scale")
        if self.use_flow and self.fm_weight <= 0:
            raise ValueError("v17 flow requires velocity supervision; use the explicit no-flow ablation")
        if self.tail < 0 or self.coarse_weight <= 0 or self.path_weight < 0:
            raise ValueError("Invalid v17 loss weights")
        if "max" in self.loss_fn:
            self.loss_fn.remove("max")  # Report top-k separately, apply exactly once.
        if not self.dual:
            self.dual_noise_head = None
        if self.use_flow:
            backbone = h.get("v17_backbone", "unet")
            common = dict(condition_channels=self.condition_channels, state_channels=1,
                          time_dim=int(h.get("cfm_unet_time_dim", 96)),
                          groups=int(h.get("cfm_groups", 8)), dropout=float(h.get("cfm_dropout", 0.06)))
            if backbone == "unet":
                self.residual_flow = ConditionalResidualFlowUNetRefiner1d(
                    **common, output_channels=1, aux_output_channels=0,
                    base_channels=int(h.get("cfm_unet_base_channels", 32)),
                    channel_mults=tuple(h.get("cfm_unet_channel_mults", [1, 2, 4])),
                    pool_scales=tuple(h.get("cfm_unet_pool_scales", [3, 5, 9, 15])),
                    use_attention=False, mid_dilations=tuple(h.get("cfm_unet_mid_dilations", [1, 2, 4, 8])))
            elif backbone == "tcn":
                self.residual_flow = ConditionalResidualCFMRefiner1d(
                    **common, channels=int(h.get("v17_tcn_channels", 64)),
                    blocks=int(h.get("cfm_blocks", 6)), dilations=tuple(h.get("cfm_dilations", [1, 2, 4, 8])))
            else:
                raise ValueError("v17_backbone must be unet or tcn")
        else:
            self.residual_flow = None
        # No zero-filled feature slots, inactive output rows, or fallback scalars.
        initial_gates = ([0.25] if self.use_flow else []) + ([0.10] if self.independent else [])
        self.gate = CooperativeGate(self.condition_channels + int(self.correction_gate),
                                    len(initial_gates), initial=initial_gates,
                                    hidden=int(h.get("v17_gate_hidden", 16)),
                                    local=bool(h.get("v17_local_gate", True)),
                                    bound=float(h.get("v17_gate_bound", 0.5)),
                                    kernel=int(h.get("v17_gate_smooth_kernel", 33)))
        if bool(h.get("v17_fixed_gate", False)):
            self.gate = None
            self.register_buffer("fixed_gates", torch.tensor(initial_gates).view(1, -1, 1))
        self.compaction = compact_parameters(self)
        if self.use_flow and not self.use_fm:
            from .direct import collapse_zero_state_time
            if self.steps != 1:
                raise ValueError("The direct control matches one-Euler-step inference")
            collapse_zero_state_time(self.residual_flow)
            self.compaction["after"] = sum(p.numel() for p in self.parameters())
            self.compaction["removed"] = self.compaction["before"] - self.compaction["after"]
            self.compaction["operations"].append("direct control only: fold fixed time and remove zero state inputs")
        self.loss_components = {}

    def _predict_dual_noise(self, encoded, length):
        if self.dual:
            return super()._predict_dual_noise(encoded, length)
        zeros = encoded.new_zeros((len(encoded), 1, length))
        return zeros, zeros

    def _condition(self, noisy, coarse, baseline):
        items = [noisy, coarse]
        if self.dual:
            items.append(baseline)
        return torch.cat(items, 1)

    def integrate(self, condition):
        if not self.use_fm:
            return self.residual_flow(condition)
        state = condition.new_zeros((len(condition), 1, condition.shape[-1]))
        # Explicit Euler uses the state and time at the same left endpoint.
        for step in range(self.steps):
            t = condition.new_full((len(condition),), step / self.steps)
            state = state + self.residual_flow(state, condition, t) / self.steps
        return state

    def normalized_stages(self, noisy):
        coarse, spectrum, baseline, residual, direct = self._restore_components(noisy)
        other = noisy - baseline if self.dual else coarse
        condition = self._condition(noisy, coarse, baseline)
        flow = self.integrate(condition) if self.use_flow else torch.zeros_like(noisy)
        gate_input = torch.cat((condition, flow), 1) if self.correction_gate else condition
        if self.gate is None:
            gates = self.fixed_gates.expand(len(noisy), -1, noisy.shape[-1])
            offsets = torch.zeros_like(gates)
        else:
            gates, offsets = self.gate(gate_input)
        # Two residual endpoints to the same clean target obey
        # delta_b = (other - coarse) - delta_c. Learn delta_c only.
        # Three old gates similarly produce only two final mixture coefficients.
        # Use exactly two identifiable controls: flow trust and coarse blend.
        trust = gates[:, :1] if self.use_flow else torch.zeros_like(noisy)
        blend = gates[:, int(self.use_flow):int(self.use_flow)+1] if self.independent else torch.zeros_like(noisy)
        clean_path, baseline_path = coarse + flow, other
        coarse_fusion = (1-blend)*coarse + blend*other
        final = trust*clean_path + (1-trust)*coarse_fusion
        return dict(final=final, pre_flow=coarse, baseline_candidate=other, direct=direct,
                    baseline_hat=baseline, residual_hat=residual, clean_path=clean_path,
                    baseline_path=baseline_path, coarse_fusion=coarse_fusion,
                    flow=flow, gates=gates, local_offsets=offsets,
                    condition=condition, spectrum=spectrum)

    def stages(self, noisy):
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError("v17 expects [B,1,T] or [B,T]")
        factor = self._norm_factor(noisy)
        stages = self.normalized_stages(noisy * factor)
        for key in ("final", "pre_flow", "baseline_candidate", "direct", "baseline_hat", "residual_hat",
                    "clean_path", "baseline_path", "coarse_fusion", "flow"):
            stages[key] = stages[key] / factor
        return stages

    def restore(self, noisy):
        return self.stages(noisy)["final"]

    def restore_with_metadata(self, noisy, valid_mask=None):
        stage = self.stages(noisy)
        metadata = {"gates_mean": stage["gates"].detach().mean(-1),
                    "gates_temporal_std": stage["gates"].detach().std(-1, unbiased=False)}
        return stage["final"], metadata

    def forward(self, clean_audio, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio, clean_audio = noisy_audio.unsqueeze(1), clean_audio.unsqueeze(1)
        if noisy_audio.ndim != 3 or noisy_audio.shape[1] != 1 or noisy_audio.shape != clean_audio.shape:
            raise ValueError("v17 requires matching single-lead clean/noisy arrays")
        if valid_mask is not None:
            valid_mask = valid_mask.to(noisy_audio.device).reshape(len(noisy_audio), -1)
            if valid_mask.shape != noisy_audio[:, 0].shape:
                raise ValueError("Invalid valid_mask shape")
        factor = self._norm_factor(noisy_audio)
        noisy, clean = noisy_audio * factor, clean_audio * factor
        stage = self.normalized_stages(noisy)
        masked = lambda value: self._masked_mean(value.squeeze(1), valid_mask)
        components = {"reconstruction": self._ecg_loss(clean[:, 0], stage["final"][:, 0], factor,
                                                      predicted_com=stage["spectrum"], valid_mask=valid_mask),
                      "coarse": self.coarse_weight * masked((stage["pre_flow"] - clean).square())}
        if self.dual:
            components["dual_baseline"] = self.lambda_dual_baseline * masked((stage["baseline_hat"] - (noisy-clean)).square())
            components["dual_residual"] = self.lambda_dual_residual * masked((stage["residual_hat"] - (clean-stage["direct"].detach())).square())
        if self.use_fm:
            target = (clean-stage["pre_flow"].detach()).detach()
            t = torch.rand(len(clean), device=clean.device, dtype=clean.dtype)
            # Include the deployment source endpoint without a noisy-start mismatch.
            t = torch.where(torch.rand_like(t) < 0.25, torch.zeros_like(t), t)
            state, velocity = linear_flow_path(target, t, self.bridge)
            prediction = self.residual_flow(state, stage["condition"].detach(), t)
            components["fm_velocity"] = self.fm_weight * masked((prediction-velocity).square())
        if self.path_weight and self.use_flow:
            components["flow_endpoint"] = self.path_weight * masked((stage["clean_path"]-clean).square())
        if self.tail:
            error = (stage["final"][:, 0] - clean[:, 0]).abs()
            count = min(self.topk, error.shape[-1])
            if valid_mask is not None:
                error = error.masked_fill(~valid_mask.bool(), 0)
                counts = valid_mask.bool().sum(-1).clamp(max=count)
                values = error.topk(count, -1).values.sum(-1) / counts.clamp_min(1)
                tail = values.sum() / (counts > 0).sum().clamp_min(1)
            else:
                tail = error.topk(count, -1).values.mean()
            components["topk"] = self.tail * tail
        loss = sum(components.values())
        self.loss_components = {name: float(value.detach()) for name, value in components.items()}
        return loss


@register_model("lstm_cooperative_v17_ecg")
class LSTMCooperativeV17Denoiser(ECGDenoisingModel):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = CooperativeV17Core({"model": kwargs})

    def set_epoch(self, epoch, total_epochs):
        if not 1 <= epoch <= total_epochs:
            raise ValueError("Invalid training epoch")

    def tail_weight(self):
        return self.core.tail

    def stages(self, noisy):
        return self.core.stages(noisy)


@register_model("lstm_v17_compact_control_ecg")
class CompactV12Control(ECGDenoisingModel):
    """Matched old cooperation rules with algebraic compaction only."""
    def __init__(self, **kwargs):
        from ..lstm_v12 import TemporalGateV12Core
        nn.Module.__init__(self)
        self.core = TemporalGateV12Core({"model": kwargs})
        if self.core.cfm_adaptive_gate is None:
            raise ValueError("The compact legacy control requires adaptive gates")
        if self.core.lambda_ecg_noise_aux or self.core.lambda_gaussian_noise_aux:
            raise ValueError("Cannot remove auxiliary channels while their objectives are active")
        before = sum(p.numel() for p in self.parameters())
        for name in ("cfm_refine_gate_raw", "cfm_baseline_gate_raw", "cfm_consistency_blend_raw"):
            delattr(self.core, name)
        old = self.core.residual_flow.output[-1]
        output = nn.Conv1d(old.in_channels, 2, old.kernel_size, padding=old.padding)
        with torch.no_grad():
            output.weight.copy_(old.weight[:2])
            output.bias.copy_(old.bias[:2])
        self.core.residual_flow.output[-1] = output
        self.core.cfm_unet_aux_channels = 0
        self.core.compaction = compact_parameters(self.core)
        self.core.compaction.update(before=before, after=sum(p.numel() for p in self.parameters()))
        self.core.compaction["removed"] = before - self.core.compaction["after"]

    def set_epoch(self, epoch, total_epochs):
        self.core.set_epoch(epoch, total_epochs)

    def tail_weight(self):
        return float(self.core.h.get("lambda_max", 0.05))
