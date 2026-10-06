"""Opt-in v13: v12 gates with delayed MSE and uncentered cosine objectives.

This changes training objectives only. The initial weights are experimental
hypotheses, not settings established to improve the held-out ECG metrics.
"""

import math

import torch
import torch.nn as nn

from ..factory import register_model
from ..lstm_v12 import TemporalGateV12Core
from ..mecg_e import ECGDenoisingModel


class MetricBalancedV13Core(TemporalGateV12Core):
    """Preserve v12 parameters/inference; reinforce amplitude and raw direction."""

    def __init__(self, config):
        settings = dict(config["model"])
        settings.setdefault("v12_local_gate", True)
        settings.setdefault("v12_tail_weight", 0.01)
        extra_mse = float(settings.get("v13_extra_mse_weight", 0.25))
        raw_cos = float(settings.get("v13_raw_cos_weight", 0.01))
        ramp = settings.get("v13_ramp_epochs", 10)
        for name, weight in (("v13_extra_mse_weight", extra_mse),
                             ("v13_raw_cos_weight", raw_cos)):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if (isinstance(ramp, bool) or not isinstance(ramp, (int, float))
                or not math.isfinite(ramp) or int(ramp) != ramp or ramp < 1):
            raise ValueError("v13_ramp_epochs must be a positive integer.")
        if raw_cos > 0 and "cos" in settings.get("loss_fn", "").split("+"):
            raise ValueError("Do not combine v13 raw cosine with the legacy centered 'cos' loss.")
        super().__init__({"model": settings})
        self.extra_mse_final_weight = extra_mse
        self.raw_cos_final_weight = raw_cos
        self.metric_ramp_epochs = int(ramp)

    def metric_weights(self):
        """Validation uses terminal weights, independent of the training epoch."""
        if not self.training:
            scale = 1.0
        elif self.total_epochs is None:
            scale = 0.0
        else:
            ramp = min(self.metric_ramp_epochs, self.total_epochs)
            progress = (self.current_epoch - (self.total_epochs - ramp)) / ramp
            scale = max(0.0, min(1.0, progress))
        return {"extra_mse": scale * self.extra_mse_final_weight,
                "raw_cos": scale * self.raw_cos_final_weight}

    @staticmethod
    def _metric_inputs(clean_audio, restored_audio, norm_factor, valid_mask=None):
        # The official scores are computed in original amplitude, even if the
        # network was configured to normalize its input. Promote half precision
        # so the official cosine epsilon does not underflow to zero.
        dtype = torch.float32 if restored_audio.dtype in (torch.float16, torch.bfloat16) else restored_audio.dtype
        factor = norm_factor.squeeze(-1).to(device=restored_audio.device, dtype=dtype)
        clean = clean_audio.to(dtype=dtype) / factor
        restored = restored_audio.to(dtype=dtype) / factor
        mask = None
        if valid_mask is not None:
            mask = valid_mask.to(device=restored.device, dtype=torch.bool)
            if mask.ndim == 3 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            if mask.shape != restored.shape:
                raise ValueError("valid_mask must match the metric inputs [B, T] or [B, 1, T].")
            clean = clean.masked_fill(~mask, 0.0)
            restored = restored.masked_fill(~mask, 0.0)
        return clean, restored, mask

    def _extra_mse_loss(self, clean_audio, restored_audio, norm_factor, valid_mask=None):
        clean, restored, mask = self._metric_inputs(clean_audio, restored_audio, norm_factor, valid_mask)
        return self._masked_mean((restored - clean).square(), mask)

    def _raw_cosine_loss(self, clean_audio, restored_audio, norm_factor, valid_mask=None):
        """Uncentered per-window cosine, with the official additive epsilon.

        All-masked and zero-clean-energy windows have no reference direction and
        are excluded from this training term; MSE still supervises zero-clean
        windows. Official evaluation is unchanged. vector_norm has a finite
        backward at zero, unlike a naive sqrt(sum(square)) implementation.
        """
        clean, restored, _ = self._metric_inputs(clean_audio, restored_audio, norm_factor, valid_mask)
        clean_norm = torch.linalg.vector_norm(clean, dim=-1)
        restored_norm = torch.linalg.vector_norm(restored, dim=-1)
        cosine = (clean * restored).sum(dim=-1) / (clean_norm * restored_norm + 1.0e-10)
        eligible = clean_norm > 0
        error = torch.where(eligible, 1.0 - cosine, torch.zeros_like(cosine))
        return error.sum() / eligible.sum().clamp_min(1)

    def _ecg_loss(self, clean_audio, restored_audio, norm_factor,
                  predicted_com=None, aux_history=None, valid_mask=None):
        loss = super()._ecg_loss(clean_audio, restored_audio, norm_factor,
                                 predicted_com=predicted_com, aux_history=aux_history,
                                 valid_mask=valid_mask)
        weights = self.metric_weights()
        # Avoid even zero-weight additions: the neither ablation preserves the
        # exact v12 delayed-top-k loss and RNG trajectory, not just its formula.
        if weights["extra_mse"] > 0:
            loss = loss + weights["extra_mse"] * self._extra_mse_loss(
                clean_audio, restored_audio, norm_factor, valid_mask)
        if weights["raw_cos"] > 0:
            loss = loss + weights["raw_cos"] * self._raw_cosine_loss(
                clean_audio, restored_audio, norm_factor, valid_mask)
        return loss


@register_model("lstm_metric_balanced_v13_ecg")
class LSTMMetricBalancedV13Denoiser(ECGDenoisingModel):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = MetricBalancedV13Core({"model": kwargs})

    def set_epoch(self, epoch, total_epochs):
        self.core.set_epoch(epoch, total_epochs)

    def tail_weight(self):
        return self.core.tail_weight()

    def metric_weights(self):
        return self.core.metric_weights()

    def extra_mse_weight(self):
        return self.metric_weights()["extra_mse"]

    def raw_cos_weight(self):
        return self.metric_weights()["raw_cos"]
