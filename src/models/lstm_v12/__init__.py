"""Opt-in v12: the complete v7 dual-flow model with bounded temporal gates.

Import this module explicitly to register the model. Legacy model classes and
configuration files are deliberately not modified.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..factory import register_model
from ..mecg_e import ECGDenoisingModel
from ..mambattention import (
    LightweightLSTMContextBlock,
    UNetResidualFlowDualPathDAPPMambAttentionCore,
)


class BoundedTemporalGateOffsets(nn.Module):
    """Three local logit offsets, bounded before optional moving averaging."""

    def __init__(self, channels=6, hidden=16, kernel_size=7, dilations=(1, 4),
                 bound=0.5, smooth_kernel=33, smoothing=True):
        super().__init__()
        if hidden < 1 or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("Temporal gate hidden width must be positive; kernel must be positive and odd.")
        if smooth_kernel < 1 or smooth_kernel % 2 != 1:
            raise ValueError("v12_gate_smooth_kernel must be positive and odd.")
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("v12_gate_offset_bound must be finite and positive.")
        if not dilations or any(int(d) != d or d < 1 for d in dilations):
            raise ValueError("v12_gate_dilations must contain positive integers.")
        self.bound = float(bound)
        self.smooth_kernel = int(smooth_kernel)
        self.smoothing = bool(smoothing)
        layers = []
        for dilation in dilations:
            layers.extend([
                nn.Conv1d(channels, hidden, kernel_size,
                          padding=int(dilation) * (kernel_size // 2), dilation=int(dilation)),
                nn.SiLU(),
            ])
            channels = hidden
        layers.append(nn.Conv1d(hidden, 3, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def smooth(self, values):
        if not self.smoothing or self.smooth_kernel == 1:
            return values
        radius = self.smooth_kernel // 2
        return F.avg_pool1d(F.pad(values, (radius, radius), mode="replicate"),
                            self.smooth_kernel, stride=1)

    def forward(self, condition):
        return self.bound * self.smooth(torch.tanh(self.net(condition)))


class TemporalGateV12Core(UNetResidualFlowDualPathDAPPMambAttentionCore):
    def __init__(self, config):
        # All original modules are initialized in their original order. Forking
        # only the new branch also preserves the subsequent training RNG stream.
        super().__init__(config, block_cls=LightweightLSTMContextBlock)
        h = self.h
        self.use_local_gate = bool(h.get("v12_local_gate", True))
        self.use_dual_head = bool(h.get("v12_dual_head", True))
        self.tail_final_weight = float(h.get("v12_tail_weight", 0.0))
        self.tail_ramp_epochs = int(h.get("v12_tail_ramp_epochs", 10))
        self.tail_topk = int(h.get("mad_topk", 8))
        if not math.isfinite(self.tail_final_weight) or self.tail_final_weight < 0:
            raise ValueError("v12_tail_weight must be finite and nonnegative.")
        if self.tail_ramp_epochs < 1 or self.tail_topk < 1:
            raise ValueError("v12_tail_ramp_epochs and mad_topk must be positive.")
        if self.tail_final_weight > 0 and "max" in self.loss_fn:
            raise ValueError("Delayed top-k loss cannot be combined with the legacy 'max' loss token.")
        self.current_epoch = 0
        self.total_epochs = None
        with torch.random.fork_rng(devices=[]):
            self.temporal_gate = BoundedTemporalGateOffsets(
                channels=self.cfm_condition_channels,
                hidden=int(h.get("v12_gate_hidden", 16)),
                kernel_size=int(h.get("v12_gate_kernel_size", 7)),
                dilations=tuple(h.get("v12_gate_dilations", [1, 4])),
                bound=float(h.get("v12_gate_offset_bound", 0.5)),
                smooth_kernel=int(h.get("v12_gate_smooth_kernel", 33)),
                smoothing=bool(h.get("v12_gate_smoothing", True)),
            )

        # Ablation removal follows construction, so surviving parameters match.
        if not self.use_local_gate:
            self.temporal_gate = None
        if not bool(h.get("v12_feature_dapp", True)):
            self.feature_dapp = nn.Identity()
        if not bool(h.get("v12_unet_dapp", True)):
            self.residual_flow.first_skip_dapp = nn.Identity()
        if not bool(h.get("v12_resconv", True)):
            for block in self.tsc_blocks:
                block.context = nn.Identity()
        if not self.use_dual_head:
            self.dual_noise_head = None
            # The head has two output channels (baseline and clean residual).
            # Removing both also removes both associated supervision terms.
            self.loss_fn = [token for token in self.loss_fn if token not in {"dual", "dual_noise"}]

    def set_epoch(self, epoch, total_epochs):
        """Set a one-based training epoch; ramp over the final configured epochs."""
        if int(epoch) != epoch or int(total_epochs) != total_epochs or not 1 <= epoch <= total_epochs:
            raise ValueError("Require integer epochs with 1 <= epoch <= total_epochs.")
        self.current_epoch, self.total_epochs = int(epoch), int(total_epochs)

    def tail_weight(self):
        # Validation always uses the terminal objective, independent of the
        # epoch, for checkpoint and ReduceLROnPlateau comparisons.
        if not self.training:
            return self.tail_final_weight
        if self.total_epochs is None:
            return 0.0
        ramp = min(self.tail_ramp_epochs, self.total_epochs)
        progress = (self.current_epoch - (self.total_epochs - ramp)) / ramp
        return self.tail_final_weight * max(0.0, min(1.0, progress))

    def _cfm_gates(self, condition):
        if self.temporal_gate is None:
            return super()._cfm_gates(condition)
        if self.cfm_adaptive_gate is not None:
            stats = torch.cat([condition.abs().mean(-1), condition.std(-1)], dim=-1)
            logits = self.cfm_adaptive_gate.net(stats).unsqueeze(-1)
        else:
            logits = torch.stack([self.cfm_refine_gate_raw, self.cfm_baseline_gate_raw,
                                  self.cfm_consistency_blend_raw]).view(1, 3, 1)
        gates = torch.sigmoid(logits + self.temporal_gate(condition))
        return (self.cfm_refine_gate_max * gates[:, 0:1],
                self.cfm_baseline_gate_max * gates[:, 1:2],
                self.cfm_consistency_blend_max * gates[:, 2:3])

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

    def _tail_loss(self, clean_audio, restored_audio, valid_mask=None):
        # Match the existing v7_max normalized-domain absolute-error definition.
        error = (restored_audio - clean_audio).abs()
        count = min(self.tail_topk, error.shape[-1])
        if valid_mask is None:
            return error.topk(count, dim=-1).values.mean()
        mask = valid_mask.to(device=error.device, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask.squeeze(1)
        values = error.masked_fill(~mask, 0.0).topk(count, dim=-1).values
        valid_count = mask.sum(dim=-1).clamp(max=count)
        per_window = values.sum(dim=-1) / valid_count.clamp_min(1)
        # Completely padded windows contribute neither error nor sample count.
        return per_window.sum() / (valid_count > 0).sum().clamp_min(1)

    def _ecg_loss(self, clean_audio, restored_audio, norm_factor,
                  predicted_com=None, aux_history=None, valid_mask=None):
        loss = super()._ecg_loss(clean_audio, restored_audio, norm_factor,
                                 predicted_com=predicted_com, aux_history=aux_history,
                                 valid_mask=valid_mask)
        weight = self.tail_weight()
        if weight > 0:
            loss = loss + weight * self._tail_loss(clean_audio, restored_audio, valid_mask)
        return loss

    def forward(self, clean_audio, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.ndim == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.ndim != 3 or noisy_audio.shape[1] != 1 or clean_audio.shape != noisy_audio.shape:
            raise ValueError("v12 expects matching single-lead inputs [B, 1, T] or [B, T].")
        if valid_mask is not None:
            if valid_mask.ndim == 3 and valid_mask.shape[1] == 1:
                valid_mask = valid_mask.squeeze(1)
            if valid_mask.shape != noisy_audio[:, 0].shape:
                raise ValueError("valid_mask must have shape [B, T] or [B, 1, T].")
        return super().forward(clean_audio, noisy_audio, valid_mask=valid_mask)

    def restore_with_metadata(self, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if noisy_audio.ndim != 3 or noisy_audio.shape[1] != 1:
            raise ValueError("v12 expects single-lead input [B, 1, T] or [B, T].")
        factor = self._norm_factor(noisy_audio)
        noisy = noisy_audio * factor
        coarse, _, baseline, delta, _ = self._restore_components(noisy)
        restored, flow, condition, _, _ = self._refine_from_base(noisy, coarse, baseline_hat=baseline)
        metadata = {
            "baseline_hat_abs_mean": baseline.detach().abs().mean(-1),
            "residual_delta_abs_mean": delta.detach().abs().mean(-1),
            "cfm_clean_delta_abs_mean": flow[:, :1].detach().abs().mean(-1),
            "cfm_baseline_delta_abs_mean": flow[:, 1:2].detach().abs().mean(-1),
        }
        gates = self._cfm_gates(condition)
        for name, values in zip(("cfm_refine_gate", "cfm_baseline_gate", "cfm_consistency_blend"), gates):
            values = values.detach()
            metadata[name] = values.mean().view(1)
            metadata[name + "_min"] = values.amin().view(1)
            metadata[name + "_max"] = values.amax().view(1)
            metadata[name + "_temporal_std"] = values.std(dim=-1, unbiased=False).mean().view(1)
        self.last_metadata = metadata
        return restored / factor, metadata


@register_model("lstm_temporal_gate_v12_ecg")
class LSTMTemporalGateV12Denoiser(ECGDenoisingModel):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = TemporalGateV12Core({"model": kwargs})

    def set_epoch(self, epoch, total_epochs):
        self.core.set_epoch(epoch, total_epochs)

    def tail_weight(self):
        return self.core.tail_weight()
