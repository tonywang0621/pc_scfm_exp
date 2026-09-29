"""Opt-in v11 model; importing the legacy models package does not register it."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .factory import register_model
from .mecg_e import ECGDenoisingModel
from .mambattention import (
    ConditionalResidualFlowUNetRefiner1d,
    DualPathDAPPMambAttentionCore,
    LightweightLSTMContextBlock,
    ResidualFlowDualPathDAPPMambAttentionCore,
)


class ResidualBandGate(nn.Module):
    def __init__(self, channels, hidden, initial, maximum, adaptive=True):
        super().__init__()
        initial = torch.tensor(initial, dtype=torch.float32)
        maximum = torch.tensor(maximum, dtype=torch.float32)
        if not torch.all((initial > 0) & (initial < maximum)):
            raise ValueError("Gate initial values must be positive and below their maxima.")
        self.register_buffer("maximum", maximum.view(1, 2, 1))
        self.register_buffer("fixed", initial.view(1, 2, 1))
        self.net = None
        if adaptive:
            self.net = nn.Sequential(
                nn.LayerNorm(channels * 2),
                nn.Linear(channels * 2, hidden),
                nn.PReLU(hidden),
                nn.Linear(hidden, 2),
            )
            nn.init.zeros_(self.net[-1].weight)
            with torch.no_grad():
                self.net[-1].bias.copy_(torch.logit(initial / maximum))

    def forward(self, condition):
        if self.net is None:
            return self.fixed.expand(condition.shape[0], -1, -1)
        stats = torch.cat([condition.abs().mean(-1), condition.std(-1)], dim=-1)
        return self.maximum * torch.sigmoid(self.net(stats)).unsqueeze(-1)


class SingleResidualLSTMCore(DualPathDAPPMambAttentionCore):
    # Reuse the exact v7 conditioning, budget, and spectral-loss definitions.
    _flow_condition = ResidualFlowDualPathDAPPMambAttentionCore._flow_condition
    _limit_delta = ResidualFlowDualPathDAPPMambAttentionCore._limit_delta
    _multi_resolution_stft_loss = ResidualFlowDualPathDAPPMambAttentionCore._multi_resolution_stft_loss

    def __init__(self, config):
        super().__init__(config, block_cls=LightweightLSTMContextBlock)
        h = self.h
        self.use_cfm = bool(h.get("v11_use_cfm", True))
        self.use_band_split = bool(h.get("v11_use_band_split", True))
        self.use_dual_head = bool(h.get("v11_use_dual_head", True))
        self.inference_steps = int(h.get("cfm_inference_steps", 1))
        if self.inference_steps < 1:
            raise ValueError("cfm_inference_steps must be positive.")
        if not self.use_cfm and self.inference_steps != 1:
            raise ValueError("Direct residual regression requires one inference step.")
        self.train_noise_scale = float(h.get("cfm_train_noise_scale", 0.05))
        self.zero_start_prob = float(h.get("cfm_zero_start_prob", 0.5))
        if self.train_noise_scale < 0 or not 0 <= self.zero_start_prob <= 1:
            raise ValueError("Invalid CFM noise scale or zero-start probability.")
        if float(h.get("cfm_bridge_noise_scale", 0.0)) != 0:
            raise ValueError("v11 uses a straight CFM path; bridge noise must be zero.")
        self.lambda_cfm = float(h.get("lambda_cfm", 0.17))
        self.lambda_cfm_stft = float(h.get("lambda_cfm_stft", 0.01))
        self.low_budget = float(h.get("v11_low_delta_budget", 0.95))
        self.high_budget = float(h.get("v11_high_delta_budget", 0.08))

        self.residual_flow = ConditionalResidualFlowUNetRefiner1d(
            condition_channels=6,
            state_channels=1,
            output_channels=1,
            base_channels=int(h.get("cfm_unet_base_channels", 32)),
            channel_mults=tuple(h.get("cfm_unet_channel_mults", [1, 2, 4])),
            time_dim=int(h.get("cfm_unet_time_dim", 96)),
            groups=int(h.get("cfm_groups", 8)),
            dropout=float(h.get("cfm_dropout", 0.06)),
            pool_scales=tuple(h.get("cfm_unet_pool_scales", [3, 5, 9, 15])),
            use_attention=False,
            mid_dilations=tuple(h.get("cfm_unet_mid_dilations", [1, 2, 4, 8])),
            aux_output_channels=0,
        )
        self.band_gate = ResidualBandGate(
            channels=6,
            hidden=int(h.get("cfm_gate_hidden", 24)),
            initial=[h.get("v11_low_gate_init", 0.02), h.get("v11_high_gate_init", 0.017)],
            maximum=[h.get("v11_low_gate_max", 0.20), h.get("v11_high_gate_max", 0.10)],
            adaptive=bool(h.get("v11_adaptive_gate", True)),
        )

        # Remove ablated modules after construction to preserve shared initialization.
        if not bool(h.get("v11_feature_dapp", True)):
            self.feature_dapp = nn.Identity()
        if not bool(h.get("v11_unet_dapp", True)):
            self.residual_flow.first_skip_dapp = nn.Identity()
        if not bool(h.get("v11_resconv", True)):
            for block in self.tsc_blocks:
                block.context = nn.Identity()
        if not self.use_dual_head:
            self.dual_noise_head = None

    def _predict_dual_noise(self, encoded, length):
        if self.use_dual_head:
            return super()._predict_dual_noise(encoded, length)
        zeros = encoded.new_zeros((encoded.shape[0], 1, length))
        return zeros, zeros

    def _restore_components(self, noisy_audio):
        restored, spectrum, baseline, delta, direct = super()._restore_components(noisy_audio)
        if not self.use_dual_head:
            noisy = noisy_audio.unsqueeze(1) if noisy_audio.ndim == 2 else noisy_audio
            baseline = noisy - restored
        return restored, spectrum, baseline, delta, direct

    def _flow_matching_loss(self, condition, target, valid_mask=None):
        start = self.train_noise_scale * torch.randn_like(target)
        zero = torch.rand((target.shape[0], 1, 1), device=target.device) < self.zero_start_prob
        start = torch.where(zero, torch.zeros_like(start), start)
        t = torch.rand((target.shape[0],), device=target.device)
        state = (1.0 - t[:, None, None]) * start + t[:, None, None] * target
        velocity = self.residual_flow(state, condition, t)
        loss = F.mse_loss(velocity, target - start, reduction="none").squeeze(1)
        return self._masked_mean(loss, valid_mask)

    def _integrate(self, condition):
        residual = condition.new_zeros((condition.shape[0], 1, condition.shape[-1]))
        for step in range(self.inference_steps):
            t = condition.new_full((condition.shape[0],), (step + 0.5) / self.inference_steps)
            residual = residual + self.residual_flow(residual, condition, t) / self.inference_steps
        return residual

    def _refine(self, noisy, coarse, baseline):
        condition = self._flow_condition(noisy, coarse, baseline_hat=baseline)
        residual = self._integrate(condition)
        gates = self.band_gate(condition)
        reference = noisy - coarse
        if self.use_band_split:
            low = self._baseline_projection(residual)
            high = residual - low
            correction = (
                gates[:, :1] * self._limit_delta(low, reference, self.low_budget)
                + gates[:, 1:] * self._limit_delta(high, reference, self.high_budget)
            )
        else:
            # A shared scalar amplitude removes frequency selectivity, not capacity.
            amplitude = gates.mean(dim=1, keepdim=True)
            correction = amplitude * self._limit_delta(residual, reference, self.low_budget)
        return coarse + correction, residual, condition, gates

    def restore_with_metadata(self, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if noisy_audio.ndim != 3 or noisy_audio.shape[1] != 1:
            raise ValueError("v11 expects single-lead input [B, 1, T] or [B, T].")
        factor = self._norm_factor(noisy_audio)
        noisy = noisy_audio * factor
        coarse, _, baseline, _, _ = self._restore_components(noisy)
        restored, residual, _, gates = self._refine(noisy, coarse, baseline)
        metadata = {
            "v11_low_gate": gates[:, :1].detach(),
            "v11_high_gate": gates[:, 1:].detach(),
            "v11_residual_abs_mean": residual.detach().abs().mean(-1),
        }
        self.last_metadata = metadata
        return restored / factor, metadata

    def restore_one_shot(self, noisy_audio, return_com=False):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        coarse, spectrum, baseline, _, _ = self._restore_components(noisy_audio)
        restored, _, _, _ = self._refine(noisy_audio, coarse, baseline)
        return (restored, spectrum) if return_com else restored

    def _reconstruction_losses(self, clean, restored, residual, valid_mask):
        h = self.h
        output = restored.squeeze(1)
        loss = clean.new_tensor(0.0)
        recon = F.smooth_l1_loss(output, clean, reduction="none")
        loss = loss + float(h.get("lambda_cfm_recon", 0.72)) * self._masked_mean(recon, valid_mask)
        if self.lambda_cfm_stft > 0:
            loss = loss + self.lambda_cfm_stft * self._multi_resolution_stft_loss(clean, output, valid_mask)
        low_error = self._baseline_projection(restored - clean.unsqueeze(1))
        loss = loss + float(h.get("lambda_cfm_lf", 0.035)) * self._masked_mean(
            F.smooth_l1_loss(low_error, torch.zeros_like(low_error), reduction="none").squeeze(1),
            valid_mask,
        )
        derivative_error = (output - clean).diff(dim=-1)
        derivative_mask = None if valid_mask is None else valid_mask[..., 1:] * valid_mask[..., :-1]
        loss = loss + float(h.get("lambda_cfm_deriv", 0.03)) * self._masked_mean(
            F.smooth_l1_loss(derivative_error, torch.zeros_like(derivative_error), reduction="none"),
            derivative_mask,
        )
        curvature = residual.diff(n=2, dim=-1).abs().squeeze(1)
        curvature_mask = None
        if valid_mask is not None:
            curvature_mask = valid_mask[..., 2:] * valid_mask[..., 1:-1] * valid_mask[..., :-2]
        loss = loss + float(h.get("lambda_cfm_residual_smooth", 0.08)) * self._masked_mean(curvature, curvature_mask)
        return loss

    def forward(self, clean_audio, noisy_audio, valid_mask=None):
        factor = self._norm_factor(noisy_audio)
        clean = (clean_audio * factor).squeeze(1)
        noisy = noisy_audio * factor
        if valid_mask is not None:
            valid_mask = valid_mask.to(noisy.device)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.squeeze(1)
        coarse, spectrum, baseline, delta, direct = self._restore_components(noisy)
        restored, residual, condition, _ = self._refine(noisy, coarse, baseline)
        loss = self._ecg_loss(clean, restored.squeeze(1), factor, predicted_com=spectrum, valid_mask=valid_mask)
        if self.use_dual_head:
            baseline_loss = F.mse_loss(baseline, noisy - clean.unsqueeze(1), reduction="none")
            delta_loss = F.mse_loss(delta, clean.unsqueeze(1) - direct, reduction="none")
            loss = loss + self.lambda_dual_baseline * self._masked_mean(baseline_loss.squeeze(1), valid_mask)
            loss = loss + self.lambda_dual_residual * self._masked_mean(delta_loss.squeeze(1), valid_mask)
        if self.use_cfm:
            target = clean.unsqueeze(1) - coarse.detach()
            loss = loss + self.lambda_cfm * self._flow_matching_loss(condition.detach(), target.detach(), valid_mask)
        return loss + self._reconstruction_losses(clean, restored, residual, valid_mask)


@register_model("lstm_single_residual_v11_ecg")
class LSTMSingleResidualV11Denoiser(ECGDenoisingModel):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = SingleResidualLSTMCore({"model": kwargs})
