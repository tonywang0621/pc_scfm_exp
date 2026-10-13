"""Opt-in v16: bounded correction-aware calibration of a frozen v7.

The v7 reconstruction, flow matching and auxiliary objectives are inherited
unchanged. Only an adapter, and optionally the residual U-Net, may be trained.
"""

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..factory import register_model
from ..mambattention import LightweightLSTMContextBlock, UNetResidualFlowDualPathDAPPMambAttentionCore
from ..mecg_e import ECGDenoisingModel


CONDITION_FEATURE_COUNT = 12
FEATURE_NAMES = tuple(
    [f"condition_abs_mean_{i}" for i in range(6)]
    + [f"condition_std_{i}" for i in range(6)]
    + [f"{stage}_{stat}_{channel}" for stage in ("raw", "projected", "limited")
       for stat in ("abs_mean", "std") for channel in ("clean", "baseline")]
    + ["clean_budget_scale", "baseline_budget_scale", "raw_antisign_error",
       "baseline_head_disagreement"]
)


class GateAdapter(nn.Module):
    def __init__(self, mode="correction", hidden=24, bound=0.5):
        super().__init__()
        if mode not in {"constant", "condition", "correction"}:
            raise ValueError("v16_gate_mode must be constant, condition, or correction.")
        if type(hidden) is not int or hidden < 1:
            raise ValueError("v16_gate_hidden must be a positive integer.")
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound <= 0:
            raise ValueError("v16_gate_logit_bound must be finite and positive.")
        self.mode, self.bound = mode, float(bound)
        if mode == "constant":
            self.offset = nn.Parameter(torch.zeros(3))
        else:
            # B and C have identical parameter shapes and initialization. Do not
            # normalize across feature slots: zeros in B are an intentional mask.
            self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.SiLU(), nn.Linear(hidden, 3))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        features = features.detach()
        if self.mode == "constant":
            raw = self.offset.unsqueeze(0).expand(features.shape[0], -1)
        else:
            if self.mode == "condition":
                features = torch.cat((features[:, :CONDITION_FEATURE_COUNT],
                                      torch.zeros_like(features[:, CONDITION_FEATURE_COUNT:])), dim=1)
            raw = self.net(features)
        return self.bound * torch.tanh(raw)

    def is_zero_output(self):
        tensors = (self.offset,) if self.mode == "constant" else (self.net[-1].weight, self.net[-1].bias)
        return all(torch.count_nonzero(value).item() == 0 for value in tensors)


class CorrectionGateV16Core(UNetResidualFlowDualPathDAPPMambAttentionCore):
    def __init__(self, config):
        settings = dict(config["model"])
        for key in ("cfm_inference_steps", "cfm_inference_shots"):
            if type(settings.get(key, 1)) is not int or settings.get(key, 1) != 1:
                raise ValueError("v16 requires one-step, one-shot inference.")
            settings[key] = 1
        if settings.get("cfm_inference_start_noise_scale", 0.0) != 0.0:
            raise ValueError("v16 requires zero-start inference.")
        settings["cfm_inference_start_noise_scale"] = 0.0
        for key in ("cfm_use_adaptive_gate", "cfm_project_baseline_delta", "cfm_unet_use_attention",
                    "use_dapp", "use_time_attention", "use_freq_attention", "learnable_frft_order"):
            if key in settings and type(settings[key]) is not bool:
                raise ValueError(f"{key} must be a YAML boolean.")
        if settings.get("pcscfm_enabled", False):
            raise ValueError("v16 uses the original v7 inference path, not pcscfm.")
        super().__init__({"model": settings}, block_cls=LightweightLSTMContextBlock)
        # The new branch must not consume the legacy initialization RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.gate_adapter = GateAdapter(settings.get("v16_gate_mode", "correction"),
                                            settings.get("v16_gate_hidden", 24),
                                            settings.get("v16_gate_logit_bound", 0.5))
        self.finetuning_scope = "adapter"
        self.loss_components = {}
        self.loss_component_tensors = {}
        self._loss_detail = {}

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "gate_adapter"):
            for name, child in self.named_children():
                trainable = name == "gate_adapter" or (name == "residual_flow" and self.finetuning_scope == "adapter_flow")
                child.train(bool(mode and trainable))
        return self

    def _restore_components(self, noisy_audio):
        with torch.no_grad():
            return tuple(value.detach() for value in super()._restore_components(noisy_audio))

    def _base_gate_logits(self, condition):
        with torch.no_grad():
            if self.cfm_adaptive_gate is not None:
                stats = torch.cat((condition.abs().mean(-1), condition.std(-1)), dim=-1)
                return self.cfm_adaptive_gate.net(stats).detach()
            return torch.stack((self.cfm_refine_gate_raw, self.cfm_baseline_gate_raw,
                                self.cfm_consistency_blend_raw)).view(1, 3).expand(condition.shape[0], -1).detach()

    def _project_and_limit(self, raw, reference):
        clean = raw[:, :1]
        clean = clean - (1.0 - self.cfm_clean_lowpass_keep) * self._baseline_projection(clean)
        baseline = self._baseline_projection(raw[:, 1:2]) if self.cfm_project_baseline_delta else raw[:, 1:2]
        projected = torch.cat((clean, baseline), dim=1)
        reference_level = (reference.abs().mean(-1, keepdim=True)
                           + 0.25 * reference.std(-1, keepdim=True)).clamp_min(1e-6)
        scales, limited = [], []
        for delta, budget in ((clean, self.cfm_clean_delta_budget), (baseline, self.cfm_baseline_delta_budget)):
            scale = ((float(budget) * reference_level / delta.abs().mean(-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
                     if budget > 0 else torch.ones_like(reference_level))
            scales.append(scale)
            limited.append(self._limit_delta(delta, reference, budget))
        return projected, torch.cat(limited, dim=1), torch.cat(scales, dim=1)

    def _adapter_features(self, condition, raw, projected, limited, scales):
        values = [condition.abs().mean(-1), condition.std(-1)]
        for stage in (raw, projected, limited):
            values.extend((stage.abs().mean(-1), stage.std(-1)))
        values.extend((scales.squeeze(-1), (raw[:, :1] + raw[:, 1:2]).abs().mean(-1),
                       (condition[:, 3:4] - condition[:, 2:3]).abs().mean(-1)))
        return torch.cat(values, dim=1).detach()

    def refinement_parts(self, noisy, base, baseline_hat=None):
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if base.ndim == 2:
            base = base.unsqueeze(1)
        condition = self._flow_condition(noisy, base, baseline_hat=baseline_hat)
        raw = self._integrate_residual_flow(condition)
        projected, limited, scales = self._project_and_limit(raw, noisy - base)
        features = self._adapter_features(condition, raw, projected, limited, scales)
        logits = self._base_gate_logits(condition)
        offsets = self.gate_adapter(features)
        maxima = logits.new_tensor((self.cfm_refine_gate_max, self.cfm_baseline_gate_max,
                                    self.cfm_consistency_blend_max)).view(1, 3)
        gates = (maxima * torch.sigmoid(logits + offsets)).unsqueeze(-1)
        clean_gate, baseline_gate, blend = gates.split(1, dim=1)
        gated = torch.cat((clean_gate * limited[:, :1], baseline_gate * limited[:, 1:2]), dim=1)
        # Preserve original operation order for exact zero-offset parity.
        clean_path = base + gated[:, :1]
        baseline_path = noisy - ((noisy - base) + gated[:, 1:2])
        final = clean_path + blend * (baseline_path - clean_path)
        return {"final": final, "raw_flow": raw, "condition": condition,
                "projected_flow": projected, "limited_flow": limited, "budget_scales": scales,
                "gated_flow": gated,
                "effective_flow": torch.cat(((1.0 - blend) * gated[:, :1], -blend * gated[:, 1:2]), dim=1),
                "clean_path": clean_path, "baseline_path": baseline_path,
                "gates": gates, "base_logits": logits, "adapter_offsets": offsets,
                "adapter_features": features}

    def _refine_from_base(self, noisy_audio, base_restored, baseline_hat=None):
        parts = self.refinement_parts(noisy_audio, base_restored, baseline_hat)
        return tuple(parts[key] for key in ("final", "raw_flow", "condition", "clean_path", "baseline_path"))

    def _ecg_loss(self, *args, **kwargs):
        value = super()._ecg_loss(*args, **kwargs)
        self.loss_component_tensors["reconstruction"] = value
        return value

    def _flow_matching_loss(self, *args, **kwargs):
        # Call exactly once: logging must never draw a second start/time/bridge.
        value = super()._flow_matching_loss(*args, **kwargs)
        self.loss_component_tensors["stochastic_fm"] = value
        return value

    def _noise_auxiliary_loss(self, *args, **kwargs):
        value = super()._noise_auxiliary_loss(*args, **kwargs)
        self.loss_component_tensors["noise_auxiliary"] = value
        return value

    def _cfm_auxiliary_losses(self, clean_audio, noisy_audio, base_restored, refined,
                             flow_hat, clean_path, baseline_path, valid_mask=None):
        value = super()._cfm_auxiliary_losses(clean_audio, noisy_audio, base_restored, refined,
                                             flow_hat, clean_path, baseline_path, valid_mask)
        self.loss_component_tensors["cfm_auxiliary"] = value
        # These diagnostic subterms overlap the aggregate; they are floats only
        # and do not keep a duplicate autograd graph or invoke stochastic code.
        with torch.no_grad():
            terms = {}
            if self.lambda_cfm_recon > 0:
                terms["final_reconstruction"] = self.lambda_cfm_recon * self._masked_mean(
                    F.smooth_l1_loss(refined.squeeze(1), clean_audio, reduction="none"), valid_mask)
            if self.lambda_cfm_lf > 0:
                target = self._baseline_projection(noisy_audio - clean_audio.unsqueeze(1))
                predicted = self._baseline_projection(noisy_audio - refined)
                terms["low_frequency"] = self.lambda_cfm_lf * self._masked_mean(
                    F.smooth_l1_loss(predicted, target, reduction="none").squeeze(1), valid_mask)
            if self.lambda_cfm_deriv > 0:
                derivative_mask = valid_mask[..., 1:] * valid_mask[..., :-1] if valid_mask is not None else None
                terms["derivative"] = self.lambda_cfm_deriv * self._masked_mean(
                    F.smooth_l1_loss(refined.squeeze(1).diff(dim=-1), clean_audio.diff(dim=-1), reduction="none"),
                    derivative_mask)
            if self.lambda_cfm_residual_smooth > 0:
                smooth_mask = (valid_mask[..., 2:] * valid_mask[..., 1:-1] * valid_mask[..., :-2]
                               if valid_mask is not None else None)
                curvature = flow_hat[..., 2:] - 2.0 * flow_hat[..., 1:-1] + flow_hat[..., :-2]
                terms["raw_flow_smoothness"] = self.lambda_cfm_residual_smooth * self._masked_mean(
                    curvature.abs().sum(dim=1), smooth_mask)
            if self.lambda_cfm_clean_baseline_consistency > 0:
                terms["path_consistency"] = self.lambda_cfm_clean_baseline_consistency * self._masked_mean(
                    F.smooth_l1_loss(clean_path, baseline_path, reduction="none").squeeze(1), valid_mask)
            for name, term in terms.items():
                self._loss_detail["detail_cfm_" + name] = float(term)
            if self.lambda_cfm_stft > 0:
                self._loss_detail["detail_cfm_stft_remainder"] = float(value.detach() - sum(terms.values()))
        return value

    def forward(self, clean_audio, noisy_audio, valid_mask=None):
        # Discard the previous batch graph before computing the next batch.
        self.loss_component_tensors = {}
        self._loss_detail = {}
        if valid_mask is not None and valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(1)
        total = super().forward(clean_audio, noisy_audio, valid_mask=valid_mask)
        # Stage one (including both heads) is always frozen. Detach the
        # subtraction diagnostic to avoid reporting cancellation roundoff as
        # a trainable dual-head gradient.
        self.loss_component_tensors["dual_noise_remainder"] = (total - sum(self.loss_component_tensors.values())).detach()
        self.loss_components = {name: float(value.detach()) for name, value in self.loss_component_tensors.items()}
        self.loss_components.update(self._loss_detail)
        self.loss_components["total_loss"] = float(total.detach())
        return total

    def restore_with_metadata(self, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if noisy_audio.ndim != 3 or noisy_audio.shape[1] != 1:
            raise ValueError("v16 expects single-lead [B,1,T] input.")
        factor = self._norm_factor(noisy_audio)
        noisy = noisy_audio * factor
        base, _, baseline, _, _ = self._restore_components(noisy)
        parts = self.refinement_parts(noisy, base, baseline)
        metadata = {name: parts["gates"][:, index].detach().mean().view(1)
                    for index, name in enumerate(("cfm_refine_gate", "cfm_baseline_gate", "cfm_consistency_blend"))}
        self.last_metadata = metadata
        return parts["final"] / factor, metadata


@register_model("lstm_v16_ecg")
@register_model("lstm_v16")
class LSTMV16Denoiser(ECGDenoisingModel):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = CorrectionGateV16Core({"model": kwargs})
        self.configure_finetuning()

    @property
    def loss_components(self):
        return self.core.loss_components

    @property
    def loss_component_tensors(self):
        return self.core.loss_component_tensors

    def configure_finetuning(self, scope="adapter", lr_adapter=1e-4, lr_flow=1e-5):
        if scope not in {"adapter", "adapter_flow"}:
            raise ValueError("scope must be adapter or adapter_flow.")
        rates = {"adapter": lr_adapter, "flow": lr_flow}
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for value in rates.values()):
            raise ValueError("Learning rates must be finite and positive.")
        groups = {name: [] for name in rates}
        for name, parameter in self.named_parameters():
            group = ("adapter" if name.startswith("core.gate_adapter.") else
                     "flow" if scope == "adapter_flow" and name.startswith("core.residual_flow.") else None)
            parameter.requires_grad_(group is not None)
            if group is None:
                parameter.grad = None
            else:
                groups[group].append(parameter)
        self.core.finetuning_scope = scope
        self.train(self.training)
        return [{"name": name, "params": parameters, "lr": float(rates[name])}
                for name, parameters in groups.items() if parameters]

    def finetuning_summary(self):
        counts = {"adapter": 0, "flow": 0, "frozen": 0}
        for name, parameter in self.named_parameters():
            group = "frozen" if not parameter.requires_grad else ("adapter" if name.startswith("core.gate_adapter.") else "flow")
            counts[group] += parameter.numel()
        return {"scope": self.core.finetuning_scope, "gate_mode": self.core.gate_adapter.mode,
                "logit_bound": self.core.gate_adapter.bound, "parameter_counts": counts,
                "trainable_parameters": counts["adapter"] + counts["flow"], "total_parameters": sum(counts.values())}

    @torch.no_grad()
    def stages(self, noisy):
        """Return deterministic stages; all waveforms except condition are raw amplitude.

        Gates/scales have [B,C,1], features/logits [B,C], scalar diagnostics [B].
        effective_flow includes mixing and the baseline channel's minus sign.
        Counterfactual budget bypasses reuse the actual gates, without refitting.
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
                raise ValueError("Invalid normalization factor.")
            normalized = noisy * factor
            base, _, baseline, dual_residual, direct = core._restore_components(normalized)
            parts = core.refinement_parts(normalized, base, baseline)
            gates, scales = parts["gates"], parts["budget_scales"]
            diagnostics = {}
            for index, name in enumerate(("clean_gate", "baseline_gate", "consistency_blend")):
                diagnostics[name] = gates[:, index, 0]
                diagnostics[name + "_mean"] = gates[:, index, 0]
                diagnostics[name + "_logit_offset"] = parts["adapter_offsets"][:, index]
            for index, name in enumerate(("clean", "baseline")):
                diagnostics[name + "_budget_scale"] = scales[:, index, 0]
                diagnostics[name + "_budget_limited"] = (scales[:, index, 0] < 1).to(base.dtype)
                for stage in ("raw", "projected", "limited", "gated", "effective"):
                    wave = parts[stage + "_flow"][:, index:index + 1] / factor
                    diagnostics[name + "_" + stage + "_abs_mean"] = wave.abs().mean((1, 2))
            for mode in ("clean", "baseline", "both"):
                dc = parts["projected_flow"][:, :1] if mode in {"clean", "both"} else parts["limited_flow"][:, :1]
                db = parts["projected_flow"][:, 1:2] if mode in {"baseline", "both"} else parts["limited_flow"][:, 1:2]
                cp = base + gates[:, :1] * dc
                bp = normalized - ((normalized - base) + gates[:, 1:2] * db)
                parts["budget_bypass_" + mode] = cp + gates[:, 2:3] * (bp - cp)
            parts.update(pre_flow=base, direct=direct, baseline_hat=baseline, dual_residual=dual_residual)
            waveform_keys = ("final", "pre_flow", "direct", "raw_flow", "projected_flow", "limited_flow",
                             "gated_flow", "effective_flow", "clean_path", "baseline_path", "baseline_hat",
                             "dual_residual", "budget_bypass_clean", "budget_bypass_baseline", "budget_bypass_both")
            for key in waveform_keys:
                parts[key] = parts[key] / factor
            parts["flow_correction"] = parts["final"] - parts["pre_flow"]
            parts["gate_maxima"] = noisy.new_tensor((core.cfm_refine_gate_max, core.cfm_baseline_gate_max,
                                                     core.cfm_consistency_blend_max))
            diagnostics["flow_correction_abs_mean"] = parts["flow_correction"].abs().mean((1, 2))
            diagnostics["flow_correction_abs_max"] = parts["flow_correction"].abs().amax((1, 2))
            diagnostics["raw_antisign_error"] = (parts["raw_flow"][:, :1] + parts["raw_flow"][:, 1:2]).abs().mean((1, 2))
            diagnostics["baseline_head_disagreement"] = (parts["baseline_hat"] - (noisy - parts["pre_flow"])).abs().mean((1, 2))
            parts["diagnostics"] = diagnostics
            if not all(torch.isfinite(value).all() for value in [v for v in parts.values() if torch.is_tensor(v)] + list(diagnostics.values())):
                raise ValueError("Non-finite v16 stage diagnostics.")
            return parts
        finally:
            self.train(was_training)


def initialize_from_v7(model, checkpoint_path):
    """Load every v7 tensor strictly, allowing only a new zero-output adapter."""
    if not isinstance(model, LSTMV16Denoiser):
        raise TypeError("initialize_from_v7 requires LSTMV16Denoiser.")
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
    new_keys = {key for key in current if key.startswith("core.gate_adapter.")}
    expected = set(current) - new_keys
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(f"Not a complete v7 checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise ValueError(f"Checkpoint tensor shape/type mismatch: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
    if not model.core.gate_adapter.is_zero_output():
        raise ValueError("A new gate adapter must have zero output before v7 initialization.")
    combined = dict(state)
    combined.update({key: current[key] for key in new_keys})
    nn.Module.load_state_dict(model, combined, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"checkpoint_path": str(path), "checkpoint_sha256": digest.hexdigest(),
            "checkpoint_container": container, "loaded_v7_tensor_count": len(expected),
            "new_adapter_tensor_count": len(new_keys), "strict_v7_load": True}
