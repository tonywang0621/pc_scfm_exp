import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LightweightDAPP2d(nn.Module):
    def __init__(self, channels, time_scales=(3, 5, 9, 15), freq_scales=(1, 3)):
        super().__init__()
        self.scales = tuple((int(t), int(f)) for t in time_scales for f in freq_scales)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(channels, channels, 1),
                    nn.PReLU(channels),
                )
                for scale in self.scales
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (len(self.scales) + 1), channels, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = [x]
        for projection in self.projections:
            pooled = projection(x)
            features.append(F.interpolate(pooled, size=size, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(features, dim=1)) + x


class DualPathDAPPMambAttentionCore(MECGECore):
    def __init__(self, config, block_cls=MambAttentionBlock):
        super().__init__(config, block_cls=block_cls)
        h = self.h
        self.use_dapp = h.get("use_dapp", True)
        if self.use_dapp:
            self.feature_dapp = LightweightDAPP2d(
                h.dense_channel,
                time_scales=h.get("dapp_time_scales", [3, 5, 9, 15]),
                freq_scales=h.get("dapp_freq_scales", [1, 3]),
            )
        else:
            self.feature_dapp = nn.Identity()

        head_hidden = h.get("dual_noise_head_hidden", h.dense_channel)
        self.dual_noise_head = nn.Sequential(
            nn.Conv2d(h.dense_channel, head_hidden, 3, padding=1),
            nn.InstanceNorm2d(head_hidden, affine=True),
            nn.PReLU(head_hidden),
            nn.Conv2d(head_hidden, 2, 1),
        )
        # Start as the parent STFrFT model and let training learn a bounded residual correction.
        nn.init.zeros_(self.dual_noise_head[-1].weight)
        nn.init.zeros_(self.dual_noise_head[-1].bias)

        self.residual_refine_scale = h.get("residual_refine_scale", 0.5)
        self.teacher_weight_l1 = h.get("lambda_teacher_l1", 0.0)
        self.teacher_weight_mse = h.get("lambda_teacher_mse", 0.0)
        self.lambda_prd = h.get("lambda_prd", 0.4)
        self.lambda_dual_baseline = h.get("lambda_dual_baseline", 0.2)
        self.lambda_dual_residual = h.get("lambda_dual_residual", 0.1)

    def _encode_noisy(self, noisy_audio):
        encoded, noisy_mag_4d, noisy_pha = super()._encode_noisy(noisy_audio)
        return self.feature_dapp(encoded), noisy_mag_4d, noisy_pha

    def _predict_dual_noise(self, encoded, length):
        predicted = self.dual_noise_head(encoded).mean(dim=-1)
        predicted = F.interpolate(predicted, size=length, mode="linear", align_corners=False)
        baseline_hat = predicted[:, 0:1, :]
        residual_delta_hat = predicted[:, 1:2, :]
        return baseline_hat, residual_delta_hat

    def _restore_components(self, noisy_audio):
        if noisy_audio.ndim == 3:
            noisy_audio = noisy_audio.squeeze(1)
        x, noisy_mag_4d, noisy_pha = self._encode_noisy(noisy_audio)
        mag_g = (noisy_mag_4d * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)

        if self.fea == "cpx":
            com_d = self.complex_decoder(x).permute(0, 3, 2, 1)
            com_g = torch.stack(
                (mag_g * torch.cos(noisy_pha), mag_g * torch.sin(noisy_pha)), dim=-1
            )
            pha_g = torch.angle(torch.complex((com_g + com_d)[..., 0], (com_g + com_d)[..., 1]))
            direct_restored = self._mag_pha_inverse(mag_g, pha_g)
        elif self.fea == "pha":
            pha_g = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
            com_g = torch.stack(
                (mag_g * torch.cos(pha_g), mag_g * torch.sin(pha_g)), dim=-1
            )
            direct_restored = self._mag_pha_inverse(mag_g, pha_g)
        else:
            b, channels, frames = self.encoder(noisy_audio.unsqueeze(1)).shape
            com_d = self.complex_decoder(x).permute(0, 1, 3, 2).reshape(b, channels, frames)
            direct_restored = self.decoder(com_d).squeeze(1)
            _, _, com_g = self._mag_pha_transform_loss(direct_restored)

        if direct_restored.shape[-1] != noisy_audio.shape[-1]:
            direct_restored = F.interpolate(
                direct_restored.unsqueeze(1),
                size=noisy_audio.shape[-1],
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        baseline_hat, residual_delta_hat = self._predict_dual_noise(x, noisy_audio.shape[-1])
        restored = direct_restored.unsqueeze(1) + self.residual_refine_scale * residual_delta_hat
        return restored, com_g, baseline_hat, residual_delta_hat, direct_restored.unsqueeze(1)

    def restore_one_shot(self, noisy_audio, return_com=False):
        restored, com_g, _, _, _ = self._restore_components(noisy_audio)
        if return_com:
            return restored, com_g
        return restored

    def restore_with_metadata(self, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if noisy_audio.shape[1] != 1:
            raise ValueError(
                f"MambAttention-STFrFT dual-path DAPP expects single-lead input shaped [B, 1, T], got {noisy_audio.shape}."
            )
        norm_factor = self._norm_factor(noisy_audio)
        noisy_audio_norm = (noisy_audio * norm_factor).squeeze(1)
        restored, _, baseline_hat, residual_delta_hat, _ = self._restore_components(noisy_audio_norm)
        metadata = {
            "baseline_hat_abs_mean": baseline_hat.detach().abs().mean(dim=-1),
            "residual_delta_abs_mean": residual_delta_hat.detach().abs().mean(dim=-1),
        }
        self.last_metadata = metadata
        return restored / norm_factor, metadata

    def _prd_loss(self, clean_audio, restored_audio, valid_mask=None):
        error = restored_audio - clean_audio
        target = clean_audio
        if valid_mask is not None:
            mask = valid_mask.squeeze(1).to(error.device, dtype=error.dtype)
            error = error * mask
            target = target * mask
        numerator = error.pow(2).sum(dim=-1)
        denominator = target.pow(2).sum(dim=-1).clamp_min(1.0e-8)
        return torch.sqrt(numerator / denominator).mean()

    def _teacher_loss(self, restored_audio, teacher_audio, norm_factor, valid_mask=None):
        if teacher_audio is None or self.teacher_weight_l1 + self.teacher_weight_mse <= 0:
            return restored_audio.new_tensor(0.0)
        if teacher_audio.ndim == 3:
            teacher_audio = teacher_audio.squeeze(1)
        teacher_audio = teacher_audio * norm_factor.squeeze(1)
        loss = restored_audio.new_tensor(0.0)
        if self.teacher_weight_l1:
            l1 = F.l1_loss(restored_audio, teacher_audio, reduction="none")
            loss = loss + self.teacher_weight_l1 * self._masked_mean(l1 / norm_factor.squeeze(-1), valid_mask)
        if self.teacher_weight_mse:
            mse = F.mse_loss(restored_audio, teacher_audio, reduction="none")
            loss = loss + self.teacher_weight_mse * self._masked_mean(
                mse / norm_factor.squeeze(-1).pow(2), valid_mask
            )
        return loss

    def forward(self, clean_audio, noisy_audio, valid_mask=None, teacher_audio=None):
        norm_factor = self._norm_factor(noisy_audio)
        clean_audio = (clean_audio * norm_factor).squeeze(1)
        noisy_audio = noisy_audio * norm_factor
        if valid_mask is not None:
            valid_mask = valid_mask.to(noisy_audio.device)

        restored, com_g, baseline_hat, residual_delta_hat, direct_restored = self._restore_components(noisy_audio)
        restored_audio = restored.squeeze(1)
        direct_restored = direct_restored.squeeze(1)
        loss = self._ecg_loss(
            clean_audio,
            restored_audio,
            norm_factor,
            predicted_com=com_g,
            valid_mask=valid_mask,
        )
        if "prd" in self.loss_fn:
            loss = loss + self.lambda_prd * self._prd_loss(clean_audio, restored_audio, valid_mask)
        if "dual" in self.loss_fn or "dual_noise" in self.loss_fn:
            baseline_target = noisy_audio - clean_audio.unsqueeze(1)
            residual_target = clean_audio.unsqueeze(1) - direct_restored.unsqueeze(1)
            baseline_loss = F.mse_loss(baseline_hat, baseline_target, reduction="none")
            residual_loss = F.mse_loss(residual_delta_hat, residual_target, reduction="none")
            loss = loss + self.lambda_dual_baseline * self._masked_mean(baseline_loss.squeeze(1), valid_mask)
            loss = loss + self.lambda_dual_residual * self._masked_mean(residual_loss.squeeze(1), valid_mask)
        if "teacher" in self.loss_fn:
            loss = loss + self._teacher_loss(restored_audio, teacher_audio, norm_factor, valid_mask)
        return loss


DistilledResidualDualNoiseMambAttentionCore = DualPathDAPPMambAttentionCore


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


@register_model("mambattention_stfrft_eddm_distill_ecg")
class MambAttentionSTFrFTEDDMDistilledDualNoiseECGDenoiser(ECGDenoisingModel):
    block_cls = MambAttentionBlock

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = DistilledResidualDualNoiseMambAttentionCore(
            {"model": kwargs},
            block_cls=self.block_cls,
        )

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        valid_mask = batch[2].to(device) if len(batch) > 2 else None
        teacher_audio = batch[3].to(device) if len(batch) > 3 else None
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)
        if teacher_audio is not None and teacher_audio.ndim == 2:
            teacher_audio = teacher_audio.unsqueeze(1)
        return self.core(clean, noisy, valid_mask=valid_mask, teacher_audio=teacher_audio)


@register_model("mambattention_stfrft_dualpath_dapp_ecg")
class MambAttentionSTFrFTDualPathDAPPECGDenoiser(ECGDenoisingModel):
    block_cls = MambAttentionBlock

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = DualPathDAPPMambAttentionCore(
            {"model": kwargs},
            block_cls=self.block_cls,
        )
