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


class ResidualFlowTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t):
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualFlowBlock1d(nn.Module):
    def __init__(self, channels, time_dim, dilation=1, groups=8, dropout=0.0):
        super().__init__()
        group_count = min(int(groups), int(channels))
        while channels % group_count != 0:
            group_count -= 1
        padding = int(dilation)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(group_count, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(group_count, channels)
        self.time_proj = nn.Linear(time_dim, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x + self.time_proj(time_emb).unsqueeze(-1))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.silu(x + residual)


class ConditionalResidualFlowRefiner1d(nn.Module):
    def __init__(
        self,
        condition_channels=5,
        state_channels=1,
        output_channels=1,
        channels=48,
        blocks=4,
        time_dim=96,
        groups=8,
        dropout=0.0,
        dilations=(1, 2, 4, 8),
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            ResidualFlowTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv1d(condition_channels + state_channels, channels, 3, padding=1)
        dilation_values = tuple(dilations) if dilations else (1,)
        self.blocks = nn.ModuleList(
            [
                ResidualFlowBlock1d(
                    channels,
                    time_dim,
                    dilation=dilation_values[index % len(dilation_values)],
                    groups=groups,
                    dropout=dropout,
                )
                for index in range(int(blocks))
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, output_channels, 3, padding=1),
        )

    def forward(self, residual_state, condition, t):
        if t.ndim == 0:
            t = t.expand(residual_state.shape[0])
        time_emb = self.time_embedding(t)
        x = self.input(torch.cat([residual_state, condition], dim=1))
        for block in self.blocks:
            x = block(x, time_emb)
        return self.output(x)


class CFMAdaptiveGate(nn.Module):
    def __init__(
        self,
        condition_channels,
        hidden,
        refine_gate_init,
        refine_gate_max,
        baseline_gate_init,
        baseline_gate_max,
        blend_init,
        blend_max,
    ):
        super().__init__()
        self.refine_gate_max = float(refine_gate_max)
        self.baseline_gate_max = float(baseline_gate_max)
        self.blend_max = float(blend_max)
        feature_dim = int(condition_channels) * 2
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden)),
            nn.PReLU(int(hidden)),
            nn.Linear(int(hidden), 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        init_values = [
            self._logit_ratio(refine_gate_init, self.refine_gate_max),
            self._logit_ratio(baseline_gate_init, self.baseline_gate_max),
            self._logit_ratio(blend_init, self.blend_max),
        ]
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor(init_values, dtype=torch.float32))

    @staticmethod
    def _logit_ratio(value, max_value):
        ratio = min(max(float(value) / max(float(max_value), 1.0e-6), 1.0e-6), 1.0 - 1.0e-6)
        return math.log(ratio / (1.0 - ratio))

    def forward(self, condition):
        stats = torch.cat(
            [
                condition.abs().mean(dim=-1),
                condition.std(dim=-1),
            ],
            dim=-1,
        )
        gates = torch.sigmoid(self.net(stats))
        clean_gate = self.refine_gate_max * gates[:, 0:1].unsqueeze(-1)
        baseline_gate = self.baseline_gate_max * gates[:, 1:2].unsqueeze(-1)
        blend = self.blend_max * gates[:, 2:3].unsqueeze(-1)
        return clean_gate, baseline_gate, blend


class UNetDAPP1d(nn.Module):
    def __init__(self, channels, pool_scales=(3, 5, 9, 15), use_global_pool=True):
        super().__init__()
        self.pool_scales = tuple(int(scale) for scale in pool_scales)
        self.use_global_pool = bool(use_global_pool)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(scale),
                    nn.Conv1d(channels, channels, 1),
                    nn.SiLU(),
                )
                for scale in self.pool_scales
            ]
        )
        self.global_projection = (
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, channels, 1),
                nn.SiLU(),
            )
            if self.use_global_pool
            else None
        )
        feature_count = 1 + len(self.pool_scales) + int(self.use_global_pool)
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * feature_count, channels, 1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        length = x.shape[-1]
        features = [x]
        for projection in self.projections:
            pooled = projection(x)
            features.append(F.interpolate(pooled, size=length, mode="linear", align_corners=False))
        if self.global_projection is not None:
            pooled = self.global_projection(x)
            features.append(F.interpolate(pooled, size=length, mode="linear", align_corners=False))
        return self.fuse(torch.cat(features, dim=1)) + x


class UNetConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8, dropout=0.0):
        super().__init__()
        group_count = min(int(groups), int(out_channels))
        while out_channels % group_count != 0:
            group_count -= 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(group_count, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x + self.time_proj(time_emb).unsqueeze(-1))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.silu(x + residual)


class UNetDownBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8, dropout=0.0):
        super().__init__()
        self.block = UNetConvBlock1d(in_channels, out_channels, time_dim, groups=groups, dropout=dropout)
        self.pool = nn.Conv1d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, time_emb):
        skip = self.block(x, time_emb)
        return self.pool(skip), skip


class UNetUpBlock1d(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim, groups=8, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = UNetConvBlock1d(
            out_channels + skip_channels,
            out_channels,
            time_dim,
            groups=groups,
            dropout=dropout,
        )

    def forward(self, x, skip, time_emb):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1), time_emb)


class UNetSelfAttention1d(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(int(channels), int(heads), batch_first=True)

    def forward(self, x):
        residual = x
        sequence = self.norm(x).transpose(1, 2)
        attended, _ = self.attn(sequence, sequence, sequence, need_weights=False)
        return residual + attended.transpose(1, 2)


class ConditionalResidualFlowUNetRefiner1d(nn.Module):
    def __init__(
        self,
        condition_channels=6,
        state_channels=2,
        output_channels=2,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_dim=192,
        groups=8,
        dropout=0.0,
        pool_scales=(3, 5, 9, 15),
        attention_heads=4,
        aux_output_channels=0,
    ):
        super().__init__()
        self.output_channels = int(output_channels)
        self.aux_output_channels = int(aux_output_channels)
        self.time_embedding = nn.Sequential(
            ResidualFlowTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        channels = [int(base_channels) * int(mult) for mult in channel_mults]
        self.input = nn.Conv1d(condition_channels + state_channels, channels[0], 3, padding=1)
        self.downs = nn.ModuleList()
        in_channels = channels[0]
        for out_channels in channels:
            self.downs.append(
                UNetDownBlock1d(
                    in_channels,
                    out_channels,
                    time_dim,
                    groups=groups,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.first_skip_dapp = UNetDAPP1d(channels[0], pool_scales=pool_scales, use_global_pool=True)
        self.mid_block1 = UNetConvBlock1d(channels[-1], channels[-1], time_dim, groups=groups, dropout=dropout)
        self.mid_attention = UNetSelfAttention1d(channels[-1], heads=attention_heads)
        self.mid_block2 = UNetConvBlock1d(channels[-1], channels[-1], time_dim, groups=groups, dropout=dropout)
        self.ups = nn.ModuleList()
        current = channels[-1]
        for skip_channels in reversed(channels):
            self.ups.append(
                UNetUpBlock1d(
                    current,
                    skip_channels,
                    skip_channels,
                    time_dim,
                    groups=groups,
                    dropout=dropout,
                )
            )
            current = skip_channels
        self.output = nn.Sequential(
            nn.GroupNorm(1, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], self.output_channels + self.aux_output_channels, 3, padding=1),
        )

    def forward(self, residual_state, condition, t):
        if t.ndim == 0:
            t = t.expand(residual_state.shape[0])
        time_emb = self.time_embedding(t)
        x = self.input(torch.cat([residual_state, condition], dim=1))
        skips = []
        for index, down in enumerate(self.downs):
            x, skip = down(x, time_emb)
            if index == 0:
                skip = self.first_skip_dapp(skip)
            skips.append(skip)
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_emb)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, time_emb)
        return self.output(x)


class ResidualFlowDualPathDAPPMambAttentionCore(DualPathDAPPMambAttentionCore):
    def __init__(self, config, block_cls=MambAttentionBlock):
        super().__init__(config, block_cls=block_cls)
        h = self.h
        self.cfm_inference_steps = int(h.get("cfm_inference_steps", 2))
        self.cfm_train_noise_scale = float(h.get("cfm_train_noise_scale", 0.05))
        self.cfm_zero_start_prob = float(h.get("cfm_zero_start_prob", 0.5))
        self.cfm_bridge_noise_scale = float(h.get("cfm_bridge_noise_scale", 0.02))
        self.cfm_condition_channels = 6
        self.cfm_use_adaptive_gate = bool(h.get("cfm_use_adaptive_gate", True))
        self.cfm_clean_delta_budget = float(h.get("cfm_clean_delta_budget", 0.6))
        self.cfm_baseline_delta_budget = float(h.get("cfm_baseline_delta_budget", 0.8))
        self.cfm_project_baseline_delta = bool(h.get("cfm_project_baseline_delta", True))
        self.cfm_clean_lowpass_keep = float(h.get("cfm_clean_lowpass_keep", 0.25))
        self.cfm_refine_gate_max = float(h.get("cfm_refine_gate_max", 0.6))
        self.cfm_baseline_gate_max = float(h.get("cfm_baseline_gate_max", 0.6))
        self.cfm_consistency_blend_max = float(h.get("cfm_consistency_blend_max", 0.5))
        self.lambda_cfm = float(h.get("lambda_cfm", 0.2))
        self.lambda_cfm_baseline = float(h.get("lambda_cfm_baseline", self.lambda_cfm))
        self.lambda_cfm_recon = float(h.get("lambda_cfm_recon", 0.25))
        self.lambda_cfm_stft = float(h.get("lambda_cfm_stft", 0.05))
        self.lambda_cfm_lf = float(h.get("lambda_cfm_lf", 0.03))
        self.lambda_cfm_deriv = float(h.get("lambda_cfm_deriv", 0.03))
        self.lambda_cfm_residual_smooth = float(h.get("lambda_cfm_residual_smooth", 0.005))
        self.lambda_cfm_clean_baseline_consistency = float(h.get("lambda_cfm_clean_baseline_consistency", 0.05))
        gate_init = float(h.get("cfm_refine_gate_init", 0.15))
        gate_ratio = min(max(gate_init / max(self.cfm_refine_gate_max, 1.0e-6), 1.0e-6), 1.0 - 1.0e-6)
        self.cfm_refine_gate_raw = nn.Parameter(
            torch.tensor(math.log(gate_ratio / (1.0 - gate_ratio)), dtype=torch.float32)
        )
        baseline_gate_init = float(h.get("cfm_baseline_gate_init", gate_init))
        baseline_gate_ratio = min(max(baseline_gate_init / max(self.cfm_baseline_gate_max, 1.0e-6), 1.0e-6), 1.0 - 1.0e-6)
        self.cfm_baseline_gate_raw = nn.Parameter(
            torch.tensor(math.log(baseline_gate_ratio / (1.0 - baseline_gate_ratio)), dtype=torch.float32)
        )
        blend_init = float(h.get("cfm_consistency_blend_init", 0.25))
        blend_ratio = min(max(blend_init / max(self.cfm_consistency_blend_max, 1.0e-6), 1.0e-6), 1.0 - 1.0e-6)
        self.cfm_consistency_blend_raw = nn.Parameter(
            torch.tensor(math.log(blend_ratio / (1.0 - blend_ratio)), dtype=torch.float32)
        )
        if self.cfm_use_adaptive_gate:
            self.cfm_adaptive_gate = CFMAdaptiveGate(
                condition_channels=self.cfm_condition_channels,
                hidden=int(h.get("cfm_gate_hidden", 32)),
                refine_gate_init=gate_init,
                refine_gate_max=self.cfm_refine_gate_max,
                baseline_gate_init=baseline_gate_init,
                baseline_gate_max=self.cfm_baseline_gate_max,
                blend_init=blend_init,
                blend_max=self.cfm_consistency_blend_max,
            )
        else:
            self.cfm_adaptive_gate = None
        self.residual_flow = ConditionalResidualFlowRefiner1d(
            condition_channels=self.cfm_condition_channels,
            state_channels=2,
            output_channels=2,
            channels=int(h.get("cfm_channels", 48)),
            blocks=int(h.get("cfm_blocks", 4)),
            time_dim=int(h.get("cfm_time_dim", 96)),
            groups=int(h.get("cfm_groups", 8)),
            dropout=float(h.get("cfm_dropout", 0.0)),
            dilations=tuple(h.get("cfm_dilations", [1, 2, 4, 8])),
        )

    def _cfm_refine_gate(self):
        return self.cfm_refine_gate_max * torch.sigmoid(self.cfm_refine_gate_raw)

    def _cfm_baseline_gate(self):
        return self.cfm_baseline_gate_max * torch.sigmoid(self.cfm_baseline_gate_raw)

    def _cfm_consistency_blend(self):
        return self.cfm_consistency_blend_max * torch.sigmoid(self.cfm_consistency_blend_raw)

    def _cfm_gates(self, condition):
        if self.cfm_adaptive_gate is not None:
            return self.cfm_adaptive_gate(condition)
        batch = condition.shape[0]
        clean_gate = self._cfm_refine_gate().view(1, 1, 1).expand(batch, 1, 1)
        baseline_gate = self._cfm_baseline_gate().view(1, 1, 1).expand(batch, 1, 1)
        blend = self._cfm_consistency_blend().view(1, 1, 1).expand(batch, 1, 1)
        return clean_gate, baseline_gate, blend

    def _limit_delta(self, delta, reference, budget_ratio):
        if budget_ratio <= 0:
            return delta
        reference_budget = (
            reference.abs().mean(dim=-1, keepdim=True)
            + 0.25 * reference.std(dim=-1, keepdim=True)
        ).clamp_min(1.0e-6)
        delta_level = delta.abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-6)
        scale = (float(budget_ratio) * reference_budget / delta_level).clamp(max=1.0)
        return delta * scale

    def _flow_condition(self, noisy_audio, base_restored, baseline_hat=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if base_restored.ndim == 2:
            base_restored = base_restored.unsqueeze(1)
        estimated_baseline = noisy_audio - base_restored
        if baseline_hat is None:
            baseline_hat = estimated_baseline
        low_noisy = self._baseline_projection(noisy_audio)
        high_base = base_restored - self._baseline_projection(base_restored)
        return torch.cat([noisy_audio, base_restored, estimated_baseline, baseline_hat, low_noisy, high_base], dim=1)

    def _sample_flow_start(self, target_residual):
        start = self.cfm_train_noise_scale * torch.randn_like(target_residual)
        if self.cfm_zero_start_prob > 0:
            zero_mask = (
                torch.rand((target_residual.shape[0], 1, 1), device=target_residual.device)
                < self.cfm_zero_start_prob
            )
            start = torch.where(zero_mask, torch.zeros_like(start), start)
        return start

    def _flow_matching_loss(self, condition, target_residual, channel_weight=1.0, valid_mask=None):
        start = self._sample_flow_start(target_residual)
        t = torch.rand((target_residual.shape[0],), device=target_residual.device)
        t_view = t.view(-1, 1, 1)
        bridge_noise = self.cfm_bridge_noise_scale * torch.sin(torch.pi * t_view) * torch.randn_like(target_residual)
        residual_state = (1.0 - t_view) * start + t_view * target_residual + bridge_noise
        target_velocity = target_residual - start
        predicted_velocity = self.residual_flow(residual_state, condition, t)
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        weight = torch.as_tensor(channel_weight, device=loss.device, dtype=loss.dtype).view(1, -1, 1)
        return self._masked_mean((loss * weight).sum(dim=1), valid_mask)

    def _integrate_residual_flow(self, condition, steps=None):
        steps = int(steps or self.cfm_inference_steps)
        steps = max(steps, 1)
        residual = condition.new_zeros((condition.shape[0], 2, condition.shape[-1]))
        dt = 1.0 / steps
        for step in range(steps):
            t_value = (step + 0.5) / steps
            t = torch.full((condition.shape[0],), t_value, device=condition.device, dtype=condition.dtype)
            residual = residual + dt * self.residual_flow(residual, condition, t)
        return residual

    def _refine_from_base(self, noisy_audio, base_restored, baseline_hat=None):
        condition = self._flow_condition(noisy_audio, base_restored, baseline_hat=baseline_hat)
        flow_hat = self._integrate_residual_flow(condition)
        clean_delta_hat = flow_hat[:, 0:1]
        baseline_delta_hat = flow_hat[:, 1:2]
        estimated_baseline = noisy_audio - base_restored
        clean_low = self._baseline_projection(clean_delta_hat)
        clean_delta_hat = clean_delta_hat - (1.0 - self.cfm_clean_lowpass_keep) * clean_low
        if self.cfm_project_baseline_delta:
            baseline_delta_hat = self._baseline_projection(baseline_delta_hat)
        clean_delta_hat = self._limit_delta(clean_delta_hat, estimated_baseline, self.cfm_clean_delta_budget)
        baseline_delta_hat = self._limit_delta(baseline_delta_hat, estimated_baseline, self.cfm_baseline_delta_budget)
        clean_gate, baseline_gate, consistency_blend = self._cfm_gates(condition)
        clean_path = base_restored + clean_gate * clean_delta_hat
        baseline_path = noisy_audio - (estimated_baseline + baseline_gate * baseline_delta_hat)
        refined = clean_path + consistency_blend * (baseline_path - clean_path)
        return refined, flow_hat, condition, clean_path, baseline_path

    def restore_one_shot(self, noisy_audio, return_com=False):
        restored, com_g, baseline_hat, _, _ = self._restore_components(noisy_audio)
        refined, _, _, _, _ = self._refine_from_base(noisy_audio, restored, baseline_hat=baseline_hat)
        if return_com:
            return refined, com_g
        return refined

    def restore_with_metadata(self, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if noisy_audio.shape[1] != 1:
            raise ValueError(
                f"MambAttention-STFrFT dual-path DAPP CFM expects single-lead input shaped [B, 1, T], got {noisy_audio.shape}."
            )
        norm_factor = self._norm_factor(noisy_audio)
        noisy_audio_norm = noisy_audio * norm_factor
        base_restored, _, baseline_hat, residual_delta_hat, _ = self._restore_components(noisy_audio_norm)
        refined, flow_hat, _, _, _ = self._refine_from_base(noisy_audio_norm, base_restored, baseline_hat=baseline_hat)
        metadata = {
            "baseline_hat_abs_mean": baseline_hat.detach().abs().mean(dim=-1),
            "residual_delta_abs_mean": residual_delta_hat.detach().abs().mean(dim=-1),
            "cfm_clean_delta_abs_mean": flow_hat[:, 0:1].detach().abs().mean(dim=-1),
            "cfm_baseline_delta_abs_mean": flow_hat[:, 1:2].detach().abs().mean(dim=-1),
        }
        clean_gate, baseline_gate, consistency_blend = self._cfm_gates(
            self._flow_condition(noisy_audio_norm, base_restored, baseline_hat=baseline_hat)
        )
        metadata["cfm_refine_gate"] = clean_gate.detach().mean(dim=0).view(1)
        metadata["cfm_baseline_gate"] = baseline_gate.detach().mean(dim=0).view(1)
        metadata["cfm_consistency_blend"] = consistency_blend.detach().mean(dim=0).view(1)
        self.last_metadata = metadata
        return refined / norm_factor, metadata

    def _multi_resolution_stft_loss(self, clean_audio, restored_audio, valid_mask=None):
        if self.lambda_cfm_stft <= 0:
            return clean_audio.new_tensor(0.0)
        loss = clean_audio.new_tensor(0.0)
        configs = self.h.get("cfm_stft_configs", [[32, 4, 32], [64, 8, 64], [128, 16, 128]])
        for n_fft, hop_size, win_size in configs:
            n_fft = int(n_fft)
            hop_size = int(hop_size)
            win_size = int(win_size)
            if clean_audio.shape[-1] < win_size:
                continue
            clean_spec = torch.stft(
                clean_audio,
                n_fft=n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=torch.hann_window(win_size, device=clean_audio.device, dtype=clean_audio.dtype),
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            restored_spec = torch.stft(
                restored_audio,
                n_fft=n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=torch.hann_window(win_size, device=restored_audio.device, dtype=restored_audio.dtype),
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            loss = loss + F.smooth_l1_loss(torch.abs(restored_spec), torch.abs(clean_spec))
        return loss / max(len(configs), 1)

    def _cfm_auxiliary_losses(self, clean_audio, noisy_audio, base_restored, refined, flow_hat, clean_path, baseline_path, valid_mask=None):
        loss = clean_audio.new_tensor(0.0)
        if self.lambda_cfm_recon > 0:
            recon = F.smooth_l1_loss(refined.squeeze(1), clean_audio, reduction="none")
            loss = loss + self.lambda_cfm_recon * self._masked_mean(recon, valid_mask)
        if self.lambda_cfm_stft > 0:
            loss = loss + self.lambda_cfm_stft * self._multi_resolution_stft_loss(clean_audio, refined.squeeze(1), valid_mask)
        if self.lambda_cfm_lf > 0:
            target_baseline = self._baseline_projection(noisy_audio - clean_audio.unsqueeze(1))
            predicted_baseline = self._baseline_projection(noisy_audio - refined)
            lf_loss = F.smooth_l1_loss(predicted_baseline, target_baseline, reduction="none")
            loss = loss + self.lambda_cfm_lf * self._masked_mean(lf_loss.squeeze(1), valid_mask)
        if self.lambda_cfm_deriv > 0:
            clean_derivative = clean_audio[..., 1:] - clean_audio[..., :-1]
            refined_derivative = refined.squeeze(1)[..., 1:] - refined.squeeze(1)[..., :-1]
            deriv_loss = F.smooth_l1_loss(refined_derivative, clean_derivative, reduction="none")
            derivative_mask = valid_mask[..., 1:] * valid_mask[..., :-1] if valid_mask is not None else None
            loss = loss + self.lambda_cfm_deriv * self._masked_mean(deriv_loss, derivative_mask)
        if self.lambda_cfm_residual_smooth > 0:
            residual_curvature = flow_hat[..., 2:] - 2.0 * flow_hat[..., 1:-1] + flow_hat[..., :-2]
            smooth_mask = valid_mask[..., 2:] * valid_mask[..., 1:-1] * valid_mask[..., :-2] if valid_mask is not None else None
            loss = loss + self.lambda_cfm_residual_smooth * self._masked_mean(residual_curvature.abs().sum(dim=1), smooth_mask)
        if self.lambda_cfm_clean_baseline_consistency > 0:
            consistency = F.smooth_l1_loss(clean_path, baseline_path, reduction="none")
            loss = loss + self.lambda_cfm_clean_baseline_consistency * self._masked_mean(consistency.squeeze(1), valid_mask)
        return loss

    def forward(self, clean_audio, noisy_audio, valid_mask=None):
        norm_factor = self._norm_factor(noisy_audio)
        clean_audio = (clean_audio * norm_factor).squeeze(1)
        noisy_audio = noisy_audio * norm_factor
        if valid_mask is not None:
            valid_mask = valid_mask.to(noisy_audio.device)

        base_restored, com_g, baseline_hat, residual_delta_hat, direct_restored = self._restore_components(noisy_audio)
        refined, flow_hat, condition, clean_path, baseline_path = self._refine_from_base(
            noisy_audio,
            base_restored,
            baseline_hat=baseline_hat,
        )
        refined_audio = refined.squeeze(1)
        direct_restored = direct_restored.squeeze(1)

        loss = self._ecg_loss(
            clean_audio,
            refined_audio,
            norm_factor,
            predicted_com=com_g,
            valid_mask=valid_mask,
        )
        if "dual" in self.loss_fn or "dual_noise" in self.loss_fn:
            baseline_target = noisy_audio - clean_audio.unsqueeze(1)
            residual_target = clean_audio.unsqueeze(1) - direct_restored.unsqueeze(1)
            baseline_loss = F.mse_loss(baseline_hat, baseline_target, reduction="none")
            residual_loss = F.mse_loss(residual_delta_hat, residual_target, reduction="none")
            loss = loss + self.lambda_dual_baseline * self._masked_mean(baseline_loss.squeeze(1), valid_mask)
            loss = loss + self.lambda_dual_residual * self._masked_mean(residual_loss.squeeze(1), valid_mask)
        if "cfm" in self.loss_fn or "flow_matching" in self.loss_fn:
            estimated_baseline = noisy_audio - base_restored.detach()
            true_baseline = noisy_audio - clean_audio.unsqueeze(1)
            target_clean_residual = clean_audio.unsqueeze(1) - base_restored.detach()
            target_baseline_residual = true_baseline - estimated_baseline
            target_residual = torch.cat([target_clean_residual, target_baseline_residual], dim=1)
            channel_weight = [self.lambda_cfm, self.lambda_cfm_baseline]
            loss = loss + self._flow_matching_loss(
                condition.detach(),
                target_residual.detach(),
                channel_weight=channel_weight,
                valid_mask=valid_mask,
            )
            loss = loss + self._cfm_auxiliary_losses(
                clean_audio,
                noisy_audio,
                base_restored,
                refined,
                flow_hat,
                clean_path,
                baseline_path,
                valid_mask=valid_mask,
            )
        return loss


class UNetResidualFlowDualPathDAPPMambAttentionCore(ResidualFlowDualPathDAPPMambAttentionCore):
    def __init__(self, config, block_cls=MambAttentionBlock):
        super().__init__(config, block_cls=block_cls)
        h = self.h
        self.cfm_unet_aux_channels = int(h.get("cfm_unet_aux_channels", 2))
        self.lambda_ecg_noise_aux = float(h.get("lambda_ecg_noise_aux", 0.35))
        self.lambda_gaussian_noise_aux = float(h.get("lambda_gaussian_noise_aux", 0.10))
        self.gaussian_aux_scale = float(h.get("gaussian_aux_scale", 0.2))
        self.residual_flow = ConditionalResidualFlowUNetRefiner1d(
            condition_channels=self.cfm_condition_channels,
            state_channels=2,
            output_channels=2,
            base_channels=int(h.get("cfm_unet_base_channels", 64)),
            channel_mults=tuple(h.get("cfm_unet_channel_mults", [1, 2, 4])),
            time_dim=int(h.get("cfm_unet_time_dim", 192)),
            groups=int(h.get("cfm_groups", 8)),
            dropout=float(h.get("cfm_dropout", 0.0)),
            pool_scales=tuple(h.get("cfm_unet_pool_scales", [3, 5, 9, 15])),
            attention_heads=int(h.get("cfm_unet_attention_heads", 4)),
            aux_output_channels=self.cfm_unet_aux_channels,
        )

    def _split_unet_output(self, output):
        velocity = output[:, :2]
        auxiliary = output[:, 2:] if output.shape[1] > 2 else None
        return velocity, auxiliary

    def _flow_matching_loss(self, condition, target_residual, channel_weight=1.0, valid_mask=None):
        start = self._sample_flow_start(target_residual)
        t = torch.rand((target_residual.shape[0],), device=target_residual.device)
        t_view = t.view(-1, 1, 1)
        bridge_noise = self.cfm_bridge_noise_scale * torch.sin(torch.pi * t_view) * torch.randn_like(target_residual)
        residual_state = (1.0 - t_view) * start + t_view * target_residual + bridge_noise
        target_velocity = target_residual - start
        predicted_velocity, _ = self._split_unet_output(self.residual_flow(residual_state, condition, t))
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        weight = torch.as_tensor(channel_weight, device=loss.device, dtype=loss.dtype).view(1, -1, 1)
        return self._masked_mean((loss * weight).sum(dim=1), valid_mask)

    def _integrate_residual_flow(self, condition, steps=None):
        steps = int(steps or self.cfm_inference_steps)
        steps = max(steps, 1)
        residual = condition.new_zeros((condition.shape[0], 2, condition.shape[-1]))
        dt = 1.0 / steps
        for step in range(steps):
            t_value = (step + 0.5) / steps
            t = torch.full((condition.shape[0],), t_value, device=condition.device, dtype=condition.dtype)
            velocity, _ = self._split_unet_output(self.residual_flow(residual, condition, t))
            residual = residual + dt * velocity
        return residual

    def _noise_auxiliary_loss(self, condition, clean_audio, noisy_audio, base_restored, valid_mask=None):
        if self.cfm_unet_aux_channels < 2 or self.lambda_ecg_noise_aux + self.lambda_gaussian_noise_aux <= 0:
            return clean_audio.new_tensor(0.0)
        true_baseline = noisy_audio - clean_audio.unsqueeze(1)
        gaussian_noise = torch.randn_like(true_baseline)
        t = torch.rand((true_baseline.shape[0],), device=true_baseline.device)
        t_view = t.view(-1, 1, 1)
        sigma = self.gaussian_aux_scale * torch.sqrt((t_view * (1.0 - t_view)).clamp_min(1.0e-8))
        xt = clean_audio.unsqueeze(1) + t_view * true_baseline + sigma * gaussian_noise
        aux_state = torch.cat([xt - base_restored.detach(), noisy_audio - xt], dim=1)
        _, auxiliary = self._split_unet_output(self.residual_flow(aux_state, condition.detach(), t))
        if auxiliary is None or auxiliary.shape[1] < 2:
            return clean_audio.new_tensor(0.0)
        baseline_loss = F.mse_loss(auxiliary[:, 0:1], true_baseline.detach(), reduction="none")
        gaussian_loss = F.mse_loss(auxiliary[:, 1:2], gaussian_noise.detach(), reduction="none")
        loss = clean_audio.new_tensor(0.0)
        if self.lambda_ecg_noise_aux > 0:
            loss = loss + self.lambda_ecg_noise_aux * self._masked_mean(baseline_loss.squeeze(1), valid_mask)
        if self.lambda_gaussian_noise_aux > 0:
            loss = loss + self.lambda_gaussian_noise_aux * self._masked_mean(gaussian_loss.squeeze(1), valid_mask)
        return loss

    def forward(self, clean_audio, noisy_audio, valid_mask=None):
        norm_factor = self._norm_factor(noisy_audio)
        clean_audio = (clean_audio * norm_factor).squeeze(1)
        noisy_audio = noisy_audio * norm_factor
        if valid_mask is not None:
            valid_mask = valid_mask.to(noisy_audio.device)

        base_restored, com_g, baseline_hat, residual_delta_hat, direct_restored = self._restore_components(noisy_audio)
        refined, flow_hat, condition, clean_path, baseline_path = self._refine_from_base(
            noisy_audio,
            base_restored,
            baseline_hat=baseline_hat,
        )
        refined_audio = refined.squeeze(1)
        direct_restored = direct_restored.squeeze(1)

        loss = self._ecg_loss(
            clean_audio,
            refined_audio,
            norm_factor,
            predicted_com=com_g,
            valid_mask=valid_mask,
        )
        if "dual" in self.loss_fn or "dual_noise" in self.loss_fn:
            baseline_target = noisy_audio - clean_audio.unsqueeze(1)
            residual_target = clean_audio.unsqueeze(1) - direct_restored.unsqueeze(1)
            baseline_loss = F.mse_loss(baseline_hat, baseline_target, reduction="none")
            residual_loss = F.mse_loss(residual_delta_hat, residual_target, reduction="none")
            loss = loss + self.lambda_dual_baseline * self._masked_mean(baseline_loss.squeeze(1), valid_mask)
            loss = loss + self.lambda_dual_residual * self._masked_mean(residual_loss.squeeze(1), valid_mask)
        if "cfm" in self.loss_fn or "flow_matching" in self.loss_fn:
            estimated_baseline = noisy_audio - base_restored.detach()
            true_baseline = noisy_audio - clean_audio.unsqueeze(1)
            target_clean_residual = clean_audio.unsqueeze(1) - base_restored.detach()
            target_baseline_residual = true_baseline - estimated_baseline
            target_residual = torch.cat([target_clean_residual, target_baseline_residual], dim=1)
            channel_weight = [self.lambda_cfm, self.lambda_cfm_baseline]
            loss = loss + self._flow_matching_loss(
                condition.detach(),
                target_residual.detach(),
                channel_weight=channel_weight,
                valid_mask=valid_mask,
            )
            loss = loss + self._cfm_auxiliary_losses(
                clean_audio,
                noisy_audio,
                base_restored,
                refined,
                flow_hat,
                clean_path,
                baseline_path,
                valid_mask=valid_mask,
            )
        if "noise_aux" in self.loss_fn:
            loss = loss + self._noise_auxiliary_loss(
                condition,
                clean_audio,
                noisy_audio,
                base_restored,
                valid_mask=valid_mask,
            )
        return loss


class GatedDualPathDAPPMambAttentionCore(DualPathDAPPMambAttentionCore):
    def __init__(self, config, block_cls=MambAttentionBlock):
        super().__init__(config, block_cls=block_cls)
        h = self.h
        self.baseline_refine_gate_max = h.get("baseline_refine_gate_max", 0.5)
        baseline_refine_gate_init = h.get("baseline_refine_gate_init", 0.08)
        gate_ratio = baseline_refine_gate_init / self.baseline_refine_gate_max
        gate_ratio = min(max(gate_ratio, 1.0e-6), 1.0 - 1.0e-6)
        self.baseline_refine_gate_raw = nn.Parameter(
            torch.tensor(math.log(gate_ratio / (1.0 - gate_ratio)), dtype=torch.float32)
        )

    def _baseline_refine_gate(self):
        return self.baseline_refine_gate_max * torch.sigmoid(self.baseline_refine_gate_raw)

    def _restore_components(self, noisy_audio):
        restored, com_g, baseline_hat, residual_delta_hat, direct_restored = super()._restore_components(noisy_audio)
        noisy = noisy_audio.squeeze(1) if noisy_audio.ndim == 3 else noisy_audio
        baseline_clean = noisy.unsqueeze(1) - self._baseline_projection(baseline_hat)
        baseline_gate = self._baseline_refine_gate()
        restored = restored + baseline_gate * (baseline_clean - direct_restored)
        return restored, com_g, baseline_hat, residual_delta_hat, direct_restored

    def restore_with_metadata(self, noisy_audio, valid_mask=None):
        restored, metadata = super().restore_with_metadata(noisy_audio, valid_mask=valid_mask)
        metadata = dict(metadata or {})
        metadata["baseline_refine_gate"] = self._baseline_refine_gate().detach().view(1)
        self.last_metadata = metadata
        return restored, metadata


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


@register_model("mambattention_stfrft_dualpath_dapp_h4_ecg")
class MambAttentionSTFrFTDualPathDAPPH4ECGDenoiser(MambAttentionSTFrFTDualPathDAPPECGDenoiser):
    pass


@register_model("mambattention_stfrft_dualpath_dapp_h16_ecg")
class MambAttentionSTFrFTDualPathDAPPH16ECGDenoiser(MambAttentionSTFrFTDualPathDAPPECGDenoiser):
    pass


@register_model("mambattention_stfrft_dualpath_dapp_h32_ecg")
class MambAttentionSTFrFTDualPathDAPPH32ECGDenoiser(MambAttentionSTFrFTDualPathDAPPECGDenoiser):
    pass


@register_model("mambattention_stfrft_dualpath_dapp_v2_ecg")
class MambAttentionSTFrFTDualPathDAPPV2ECGDenoiser(ECGDenoisingModel):
    block_cls = MambAttentionBlock

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = GatedDualPathDAPPMambAttentionCore(
            {"model": kwargs},
            block_cls=self.block_cls,
        )


@register_model("mambattention_stfrft_dualpath_dapp_cfm_residual_ecg")
class MambAttentionSTFrFTDualPathDAPPCFMResidualECGDenoiser(ECGDenoisingModel):
    block_cls = MambAttentionBlock

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = ResidualFlowDualPathDAPPMambAttentionCore(
            {"model": kwargs},
            block_cls=self.block_cls,
        )

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=False)
        if not strict:
            return result

        allowed_missing_prefixes = (
            "core.residual_flow.",
            "core.cfm_refine_gate_raw",
            "core.cfm_baseline_gate_raw",
            "core.cfm_consistency_blend_raw",
            "core.cfm_adaptive_gate.",
        )
        allowed_unexpected_keys = {
            "core.baseline_refine_gate_raw",
        }
        unexpected = [
            key
            for key in result.unexpected_keys
            if key not in allowed_unexpected_keys
        ]
        missing = [
            key
            for key in result.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{self.__class__.__name__}:\n"
                + "\n".join([f"\tMissing key(s): {missing}", f"\tUnexpected key(s): {unexpected}"])
            )
        return result


@register_model("mambattention_stfrft_dualpath_dapp_cfm_unet_bd_ecg")
class MambAttentionSTFrFTDualPathDAPPCFMUNetBaselineDominantECGDenoiser(ECGDenoisingModel):
    block_cls = MambAttentionBlock

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.core = UNetResidualFlowDualPathDAPPMambAttentionCore(
            {"model": kwargs},
            block_cls=self.block_cls,
        )
