"""V14: v7 inference with delayed, relative-energy-balanced reconstruction.

This is a training-recipe candidate, not a measured improvement or a new
inference backbone. Keep this package opt-in to preserve legacy run manifests.
"""

import math

import torch
import torch.nn as nn

from ..factory import register_model
from ..lstm_v13 import MetricBalancedV13Core
from ..mecg_e import ECGDenoisingModel


class RelativeBalancedV14Core(MetricBalancedV13Core):
    """Preserve v7's inference and balance absolute and per-window error.

    Relative error uses clean energy only, never the prediction-dependent
    denominator of official PRD. The positive energy floor bounds the inverse
    weight but does not cap the error itself or guarantee any test metric.
    """

    def __init__(self, config):
        settings = dict(config["model"])
        settings.setdefault("v12_local_gate", False)
        settings.setdefault("v12_tail_weight", 0.01)
        settings.setdefault("v13_extra_mse_weight", 0.25)
        settings.setdefault("v13_raw_cos_weight", 0.05)
        if settings["v12_local_gate"] is not False:
            raise ValueError("v14 retains v7 global gates; v12_local_gate must be false.")
        for key in ("v12_feature_dapp", "v12_unet_dapp", "v12_resconv", "v12_dual_head"):
            if settings.get(key, True) is not True:
                raise ValueError(f"v14 requires {key}=true; use the historical module ablations separately.")
        weight = float(settings.get("v14_relative_weight", 0.02))
        floor = float(settings.get("v14_energy_floor", 0.01))
        ramp = settings.get("v14_relative_ramp_epochs", 10)
        curriculum = settings.get("v14_curriculum", True)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("v14_relative_weight must be finite and nonnegative.")
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("v14_energy_floor must be finite and positive.")
        if (isinstance(ramp, bool) or not isinstance(ramp, (int, float))
                or not math.isfinite(ramp) or int(ramp) != ramp or ramp < 1):
            raise ValueError("v14_relative_ramp_epochs must be a positive integer.")
        if not isinstance(curriculum, bool):
            raise ValueError("v14_curriculum must be a boolean.")
        super().__init__({"model": settings})
        self.relative_final_weight = weight
        self.relative_energy_floor = floor
        self.relative_ramp_epochs = int(ramp)
        self.use_curriculum = curriculum

    def tail_weight(self):
        if not self.use_curriculum:
            return self.tail_final_weight
        return super().tail_weight()

    def metric_weights(self):
        if not self.use_curriculum:
            return {"extra_mse": self.extra_mse_final_weight,
                    "raw_cos": self.raw_cos_final_weight}
        return super().metric_weights()

    def relative_weight(self):
        if not self.training or not self.use_curriculum:
            return self.relative_final_weight
        if self.total_epochs is None:
            return 0.0
        ramp = min(self.relative_ramp_epochs, self.total_epochs)
        progress = (self.current_epoch - (self.total_epochs - ramp)) / ramp
        return self.relative_final_weight * max(0.0, min(1.0, progress))

    def _relative_reconstruction_loss(self, clean_audio, restored_audio,
                                      norm_factor, valid_mask=None):
        clean, restored, mask = self._metric_inputs(
            clean_audio, restored_audio, norm_factor, valid_mask)
        if mask is None:
            mse = (restored - clean).square().mean(dim=-1)
            energy = clean.square().mean(dim=-1)
            return (mse / (energy + self.relative_energy_floor)).mean()
        count = mask.sum(dim=-1)
        mse = (restored - clean).square().sum(dim=-1) / count.clamp_min(1)
        energy = clean.square().sum(dim=-1) / count.clamp_min(1)
        errors = mse / (energy + self.relative_energy_floor)
        eligible = count > 0
        return torch.where(eligible, errors, torch.zeros_like(errors)).sum() / eligible.sum().clamp_min(1)

    def _ecg_loss(self, clean_audio, restored_audio, norm_factor,
                  predicted_com=None, aux_history=None, valid_mask=None):
        loss = super()._ecg_loss(clean_audio, restored_audio, norm_factor,
                                 predicted_com=predicted_com, aux_history=aux_history,
                                 valid_mask=valid_mask)
        weight = self.relative_weight()
        if weight > 0:
            loss = loss + weight * self._relative_reconstruction_loss(
                clean_audio, restored_audio, norm_factor, valid_mask)
        return loss


@register_model("lstm_relative_balanced_v14_ecg")
class LSTMRelativeBalancedV14Denoiser(ECGDenoisingModel):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = RelativeBalancedV14Core({"model": kwargs})

    def set_epoch(self, epoch, total_epochs):
        self.core.set_epoch(epoch, total_epochs)

    def tail_weight(self):
        return self.core.tail_weight()

    def extra_mse_weight(self):
        return self.core.metric_weights()["extra_mse"]

    def raw_cos_weight(self):
        return self.core.metric_weights()["raw_cos"]

    def relative_weight(self):
        return self.core.relative_weight()
