import math

import torch
import torch.nn as nn

try:
    from .factory import register_model
    from .mecg_e import ECGDenoisingModel, MECGECore, TSMambaBlock
except ImportError:
    from factory import register_model
    from mecg_e import ECGDenoisingModel, MECGECore, TSMambaBlock


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.0):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)

    def forward(self, x):
        x_norm = self.layernorm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return out


class MambAttentionBlock(TSMambaBlock):
    def __init__(self, h):
        super().__init__(h)
        self.use_time_attention = h.get("use_time_attention", True)
        self.use_freq_attention = h.get("use_freq_attention", True)
        self.attention_position = h.get("attention_position", "before_mamba")
        if self.attention_position not in {"before_mamba", "after_mamba"}:
            raise ValueError(
                "attention_position must be either 'before_mamba' or 'after_mamba'."
            )
        self.attention = AttentionModule(
            dim=h.dense_channel,
            n_head=h.get("attention_heads", 8),
            dropout=h.get("attention_dropout", 0.0),
        )

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        if self.use_time_attention and self.attention_position == "before_mamba":
            x = self.attention(x) + x
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        if self.use_time_attention and self.attention_position == "after_mamba":
            x = self.attention(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        if self.use_freq_attention and self.attention_position == "before_mamba":
            x = self.attention(x) + x
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        if self.use_freq_attention and self.attention_position == "after_mamba":
            x = self.attention(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


@register_model("mambattention_ecg")
class MambAttentionECGDenoiser(ECGDenoisingModel):
    block_cls = MambAttentionBlock


@register_model("mambattention_stfrft_ecg")
class MambAttentionSTFrFTECGDenoiser(MambAttentionECGDenoiser):
    pass


@register_model("mambattention_stfrft_lf_morph_ecg")
class MambAttentionSTFrFTLFMorphECGDenoiser(MambAttentionSTFrFTECGDenoiser):
    pass


class BaselineAwareGatedMambAttentionCore(MECGECore):
    def __init__(self, config, block_cls=MambAttentionBlock):
        super().__init__(config, block_cls=block_cls)
        h = self.h
        gate_hidden = h.get("baseline_gate_hidden", max(16, h.dense_channel // 2))
        self.baseline_gate_min = h.get("baseline_gate_min", 0.0)
        self.baseline_gate_max = h.get("baseline_gate_max", 1.2)
        self.baseline_gate_smooth = h.get("baseline_gate_smooth", True)
        self.baseline_blend_max = h.get("baseline_blend_max", 0.4)
        self.baseline_gate = nn.Sequential(
            nn.LayerNorm(h.dense_channel + 6),
            nn.Linear(h.dense_channel + 6, gate_hidden),
            nn.PReLU(gate_hidden),
            nn.Linear(gate_hidden, 1),
        )
        self.baseline_blend = nn.Sequential(
            nn.LayerNorm(h.dense_channel + 6),
            nn.Linear(h.dense_channel + 6, gate_hidden),
            nn.PReLU(gate_hidden),
            nn.Linear(gate_hidden, 1),
        )
        gate_init = h.get("baseline_gate_init", 0.85)
        gate_init = (gate_init - self.baseline_gate_min) / (self.baseline_gate_max - self.baseline_gate_min)
        gate_init = min(max(gate_init, 1.0e-4), 1.0 - 1.0e-4)
        blend_init = h.get("baseline_blend_init", 0.05)
        blend_init = blend_init / self.baseline_blend_max
        blend_init = min(max(blend_init, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.zeros_(self.baseline_gate[-1].weight)
        nn.init.constant_(self.baseline_gate[-1].bias, math.log(gate_init / (1.0 - gate_init)))
        nn.init.zeros_(self.baseline_blend[-1].weight)
        nn.init.constant_(self.baseline_blend[-1].bias, math.log(blend_init / (1.0 - blend_init)))

    def _gate_features(self, encoded, noisy_audio):
        pooled = encoded.mean(dim=(2, 3))
        low = self._baseline_projection(noisy_audio.unsqueeze(1)).squeeze(1)
        high = noisy_audio - low
        peak_to_peak = noisy_audio.amax(dim=-1, keepdim=True) - noisy_audio.amin(dim=-1, keepdim=True)
        low_abs = low.abs().mean(dim=-1, keepdim=True)
        high_abs = high.abs().mean(dim=-1, keepdim=True)
        noisy_abs = noisy_audio.abs().mean(dim=-1, keepdim=True)
        low_ratio = low_abs / noisy_abs.clamp_min(1.0e-6)
        high_ratio = high_abs / noisy_abs.clamp_min(1.0e-6)
        stats = torch.cat(
            [
                low_abs,
                high_abs,
                low_ratio,
                high_ratio,
                noisy_audio.std(dim=-1, keepdim=True),
                peak_to_peak,
            ],
            dim=-1,
        )
        return torch.cat([pooled, stats], dim=-1)

    def _gated_baseline_restore(self, noisy_audio):
        if noisy_audio.ndim == 3:
            noisy_audio = noisy_audio.squeeze(1)

        direct_restored, _ = super().restore_one_shot(noisy_audio, return_com=True)
        direct_restored = direct_restored.squeeze(1)
        encoded, _, _ = self._encode_noisy(noisy_audio)
        gate_features = self._gate_features(encoded, noisy_audio)
        gate_unit = torch.sigmoid(self.baseline_gate(gate_features))
        gate = self.baseline_gate_min + (self.baseline_gate_max - self.baseline_gate_min) * gate_unit
        blend = self.baseline_blend_max * torch.sigmoid(self.baseline_blend(gate_features))
        baseline = noisy_audio - direct_restored
        if self.baseline_gate_smooth:
            baseline = self._baseline_projection(baseline.unsqueeze(1)).squeeze(1)
        gated_restored = noisy_audio - gate * baseline
        restored = direct_restored + blend * (gated_restored - direct_restored)
        return restored.unsqueeze(1), {
            "baseline_gate": gate.detach(),
            "baseline_blend": blend.detach(),
        }

    def restore_one_shot(self, noisy_audio, return_com=False):
        restored, _ = self._gated_baseline_restore(noisy_audio)
        if return_com:
            return restored, None
        return restored

    def restore_with_metadata(self, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if noisy_audio.shape[1] != 1:
            raise ValueError(f"MambAttention-STFrFT-BAG expects single-lead input shaped [B, 1, T], got {noisy_audio.shape}.")
        norm_factor = self._norm_factor(noisy_audio)
        noisy_audio_norm = (noisy_audio * norm_factor).squeeze(1)
        restored, metadata = self._gated_baseline_restore(noisy_audio_norm)
        self.last_metadata = metadata
        return restored / norm_factor, metadata


@register_model("mambattention_stfrft_bag_ecg")
class MambAttentionSTFrFTBaselineAwareGatedECGDenoiser(ECGDenoisingModel):
    block_cls = MambAttentionBlock

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = BaselineAwareGatedMambAttentionCore({"model": kwargs}, block_cls=self.block_cls)
