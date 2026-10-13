"""V15: strictly initialized v7 with constant metric losses and fine-tuning groups.

Import explicitly to register the model; legacy registrations are unchanged.
"""

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from ..factory import register_model
from ..lstm_v13 import MetricBalancedV13Core
from ..mambattention import UNetResidualFlowDualPathDAPPMambAttentionCore
from ..mecg_e import ECGDenoisingModel


FLOW_MODULES = {"residual_flow", "cfm_adaptive_gate", "temporal_gate"}
FLOW_SCALARS = {"cfm_refine_gate_raw", "cfm_baseline_gate_raw", "cfm_consistency_blend_raw"}
DECODER_MODULES = {"mask_decoder", "phase_decoder", "complex_decoder", "decoder", "dual_noise_head"}


def parameter_group(name):
    """Classify the actual core parameter names, including original gate scalars."""
    root = name.removeprefix("core.").split(".", 1)[0]
    if root in FLOW_MODULES or root in FLOW_SCALARS:
        return "flow"
    if root in DECODER_MODULES:
        return "decoder"
    return "backbone"


class FineTuneV15Core(MetricBalancedV13Core):
    def __init__(self, config):
        settings = dict(config["model"])
        settings.setdefault("v12_local_gate", False)
        settings.setdefault("v12_tail_weight", 0.01)
        settings.setdefault("v13_raw_cos_weight", 0.01)
        settings.setdefault("v13_extra_mse_weight", 0.0)
        settings.setdefault("cfm_inference_steps", 1)
        settings.setdefault("cfm_inference_shots", 1)
        settings.setdefault("cfm_inference_start_noise_scale", 0.0)
        boolean_keys = (
            "v12_local_gate", "v12_dual_head", "v12_feature_dapp", "v12_unet_dapp", "v12_resconv",
            "v12_gate_smoothing", "cfm_use_adaptive_gate", "cfm_project_baseline_delta",
            "cfm_unet_use_attention", "use_dapp", "use_time_attention", "use_freq_attention",
            "learnable_frft_order", "pcscfm_enabled",
        )
        for key in boolean_keys:
            if key in settings and type(settings[key]) is not bool:
                raise ValueError(f"{key} must be a YAML boolean, not a string or number.")
        for key in ("cfm_inference_steps", "cfm_inference_shots", "mad_topk"):
            value = settings.get(key, 8 if key == "mad_topk" else 1)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        for key in ("v12_tail_weight", "v13_raw_cos_weight", "v13_extra_mse_weight",
                    "cfm_inference_start_noise_scale"):
            value = settings[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{key} must be a finite nonnegative number.")
        normalization = settings.get("norm", False)
        if normalization is not False and not (
                isinstance(normalization, str) and normalization in {"1", "2"}):
            raise ValueError("norm must be false, '1', or '2'.")
        for key in ("v12_dual_head", "v12_feature_dapp", "v12_unet_dapp", "v12_resconv"):
            if not settings.get(key, True):
                raise ValueError(f"v15 preserves the complete v7 architecture: {key} must be true.")
        if (settings["cfm_inference_steps"] != 1 or settings["cfm_inference_shots"] != 1
                or settings["cfm_inference_start_noise_scale"] != 0.0):
            raise ValueError("v15 requires deterministic one-step, one-shot, zero-start inference.")
        super().__init__({"model": settings})
        self.finetuning_scope = "joint"
        self.loss_components = {}
        self._ecg_components = {}

    def tail_weight(self):
        return self.tail_final_weight

    def metric_weights(self):
        return {"extra_mse": self.extra_mse_final_weight, "raw_cos": self.raw_cos_final_weight}

    def _norm_factor(self, noisy):
        factor = super()._norm_factor(noisy)
        if self.norm is not False and (not torch.isfinite(factor).all() or not (factor > 0).all()):
            raise ValueError("Invalid normalization factor (possibly an all-zero input).")
        return factor

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "finetuning_scope", "joint") == "flow_only":
            # requires_grad alone does not disable LSTM/context dropout. Keep
            # every stage-one child in eval, including future stage-one modules.
            for name, child in self.named_children():
                if name not in FLOW_MODULES:
                    child.eval()
        return self

    def _restore_components(self, noisy_audio):
        if self.finetuning_scope == "flow_only":
            with torch.no_grad():
                return tuple(value.detach() for value in super()._restore_components(noisy_audio))
        return super()._restore_components(noisy_audio)

    def _ecg_loss(self, clean_audio, restored_audio, norm_factor,
                  predicted_com=None, aux_history=None, valid_mask=None):
        # Call the original objective directly, then add each term exactly once.
        # Raw diagnostics use this SAME forward output and consume no RNG.
        loss = UNetResidualFlowDualPathDAPPMambAttentionCore._ecg_loss(
            self, clean_audio, restored_audio, norm_factor,
            predicted_com=predicted_com, aux_history=aux_history, valid_mask=valid_mask)
        raw = {
            "raw_cos": self._raw_cosine_loss(clean_audio, restored_audio, norm_factor, valid_mask),
            "topk": self._tail_loss(clean_audio, restored_audio, valid_mask),
            "extra_mse": self._extra_mse_loss(clean_audio, restored_audio, norm_factor, valid_mask),
        }
        weights = {"raw_cos": self.raw_cos_final_weight, "topk": self.tail_final_weight,
                   "extra_mse": self.extra_mse_final_weight}
        self._ecg_components = {name: value.detach() for name, value in raw.items()}
        for name, value in raw.items():
            weighted = weights[name] * value
            self._ecg_components["weighted_" + name] = weighted.detach()
            if weights[name] > 0:
                loss = loss + weighted
        return loss

    def forward(self, clean_audio, noisy_audio, valid_mask=None):
        total = super().forward(clean_audio, noisy_audio, valid_mask=valid_mask)
        values = {name: float(value) for name, value in self._ecg_components.items()}
        values["total_loss"] = float(total.detach())
        # Includes original reconstruction, dual-head and stochastic CFM losses;
        # no second forward pass or second bridge-noise draw is performed.
        values["original_v7_loss"] = values["total_loss"] - sum(
            values["weighted_" + name] for name in ("raw_cos", "topk", "extra_mse"))
        self.loss_components = values
        return total


@register_model("lstm_finetune_v15_ecg")
class LSTMFineTuneV15Denoiser(ECGDenoisingModel):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = FineTuneV15Core({"model": kwargs})

    def set_epoch(self, epoch, total_epochs):
        self.core.set_epoch(epoch, total_epochs)

    def tail_weight(self):
        return self.core.tail_weight()

    def metric_weights(self):
        return self.core.metric_weights()

    @property
    def loss_components(self):
        return self.core.loss_components

    def configure_finetuning(self, scope="joint", lr_backbone=1e-6,
                             lr_decoder=3e-6, lr_flow=1e-5):
        if scope not in {"joint", "flow_only"}:
            raise ValueError("scope must be joint or flow_only.")
        rates = {"backbone": lr_backbone, "decoder": lr_decoder, "flow": lr_flow}
        if any(isinstance(rate, bool) or not isinstance(rate, (int, float))
               or not math.isfinite(rate) or rate <= 0 for rate in rates.values()):
            raise ValueError("Fine-tuning learning rates must be finite and positive.")
        grouped = {name: [] for name in rates}
        for name, parameter in self.named_parameters():
            group = parameter_group(name)
            trainable = scope == "joint" or group == "flow"
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None
            if trainable:
                grouped[group].append(parameter)
        self.core.finetuning_scope = scope
        self.train(self.training)
        return [{"name": name, "params": parameters, "lr": float(rates[name])}
                for name, parameters in grouped.items() if parameters]

    @torch.no_grad()
    def stages(self, noisy):
        """Return original-amplitude stages and per-window deterministic diagnostics.

        Each diagnostics tensor has shape [B]. ``*_budget_scale`` is the exact
        per-window scalar multiplying the projected channel, in [0,1];
        ``*_budget_limited`` is its 0/1 indicator, whose dataset mean is the
        fraction of windows hitting that budget (not a per-time-point clamp).
        Raw/projected/limited channel magnitudes precede gates. Applied channel
        magnitudes include their gate but precede consistency blending.
        ``flow_correction_abs_mean`` measures the final NET blended correction.
        """
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if noisy.ndim != 3 or noisy.shape[1] != 1 or not torch.isfinite(noisy).all():
            raise ValueError("stages requires finite single-lead [B,1,T] or [B,T] input.")
        was_training = self.training
        self.eval()
        try:
            core = self.core
            factor = core._norm_factor(noisy)
            if not torch.isfinite(factor).all() or not (factor > 0).all():
                raise ValueError("Invalid normalization factor (possibly an all-zero input).")
            normalized = noisy * factor
            pre, _, baseline, _, _ = core._restore_components(normalized)
            final, raw_flow, condition, _, _ = core._refine_from_base(normalized, pre, baseline_hat=baseline)
            gates = core._cfm_gates(condition)
            diagnostics = {}
            for name, values in zip(("clean_gate", "baseline_gate", "consistency_blend"), gates):
                for suffix, reduced in (("mean", values.mean(-1)), ("min", values.amin(-1)),
                                        ("max", values.amax(-1)),
                                        ("temporal_std", values.std(-1, unbiased=False))):
                    diagnostics[name + "_" + suffix] = reduced.reshape(-1)
            reference = normalized - pre
            reference_level = (reference.abs().mean(-1, keepdim=True)
                               + 0.25 * reference.std(-1, keepdim=True)).clamp_min(1e-6)
            for index, name, budget in ((0, "clean", core.cfm_clean_delta_budget),
                                        (1, "baseline", core.cfm_baseline_delta_budget)):
                raw = raw_flow[:, index:index + 1]
                if name == "clean":
                    projected = raw - (1 - core.cfm_clean_lowpass_keep) * core._baseline_projection(raw)
                else:
                    projected = core._baseline_projection(raw) if core.cfm_project_baseline_delta else raw
                scale = ((float(budget) * reference_level / projected.abs().mean(-1, keepdim=True)
                          .clamp_min(1e-6)).clamp(max=1.0)
                         if budget > 0 else torch.ones_like(reference_level))
                limited = core._limit_delta(projected, reference, budget)
                for label, values in (("raw", raw), ("projected", projected),
                                      ("limited", limited), ("applied", gates[index] * limited)):
                    diagnostics[f"{name}_{label}_abs_mean"] = (values / factor).abs().mean((1, 2))
                diagnostics[name + "_budget_scale"] = scale.reshape(-1)
                diagnostics[name + "_budget_limited"] = (scale < 1).to(pre.dtype).reshape(-1)
            pre, final = pre / factor, final / factor
            correction = final - pre
            diagnostics["flow_correction_abs_mean"] = correction.abs().mean((1, 2))
            diagnostics["flow_correction_abs_max"] = correction.abs().amax((1, 2))
            outputs = {"pre_flow": pre, "final": final, "flow_correction": correction,
                       "diagnostics": diagnostics}
            if not all(torch.isfinite(value).all() for value in [pre, final, correction, *diagnostics.values()]):
                raise ValueError("Non-finite stage output or flow diagnostic.")
            return outputs
        finally:
            self.train(was_training)


def initialize_from_v7(model, checkpoint_path):
    """Strictly load all v7 tensors, permitting only a NEW zero-output local gate.

    Shapes, keys and finite tensor values are checked before any state mutation.
    This deliberately bypasses legacy permissive flow checkpoint loading.
    """
    if not isinstance(model, LSTMFineTuneV15Denoiser):
        raise TypeError("initialize_from_v7 requires LSTMFineTuneV15Denoiser.")
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    container = "raw_state_dict"
    if isinstance(state, Mapping):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                state, container = state[key], key
                break
    if not isinstance(state, Mapping) or not state or not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint must contain a v7 tensor state_dict.")
    current = model.state_dict()
    local_keys = {key for key in current if key.startswith("core.temporal_gate.")}
    expected = set(current) - local_keys
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(f"Not a complete v7 checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != current[key].shape:
            raise ValueError(f"Checkpoint tensor shape/type mismatch: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
    if model.core.temporal_gate is not None:
        final_layer = model.core.temporal_gate.net[-1]
        if torch.count_nonzero(final_layer.weight).item() or torch.count_nonzero(final_layer.bias).item():
            raise ValueError("A new local gate must have zero output before v7 initialization.")
    combined = dict(state)
    combined.update({key: current[key] for key in local_keys})
    nn.Module.load_state_dict(model, combined, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"checkpoint_path": str(path), "checkpoint_sha256": digest.hexdigest(),
            "checkpoint_container": container, "loaded_v7_tensor_count": len(expected),
            "new_local_gate_tensor_count": len(local_keys), "strict_v7_load": True}
