import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


def _odd_kernel(size, limit):
    size = int(size)
    limit = int(limit)
    if limit % 2 == 0:
        limit -= 1
    return max(1, min(size if size % 2 == 1 else size + 1, limit))


class MultiScaleConvStem(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernels):
        super().__init__()
        branch_channels = max(4, hidden_channels // len(kernels))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, int(kernel), padding=int(kernel) // 2),
                    nn.GroupNorm(1, branch_channels),
                    nn.SiLU(),
                )
                for kernel in kernels
            ]
        )
        self.project = nn.Conv1d(branch_channels * len(kernels), hidden_channels, 1)

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DiagonalStateSpaceBlock(nn.Module):
    """Tiny diagonal SSM-style smoother implemented as learned exponential kernels."""

    def __init__(self, channels, state_dim=8, kernel_size=129, dropout=0.0):
        super().__init__()
        self.channels = int(channels)
        self.state_dim = int(state_dim)
        self.kernel_size = int(kernel_size)
        self.in_proj = nn.Conv1d(channels, channels * 2, 1)
        self.decay_logits = nn.Parameter(torch.linspace(-3.0, -0.5, self.state_dim).repeat(channels, 1))
        self.mix_logits = nn.Parameter(torch.zeros(channels, self.state_dim))
        self.out_proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(float(dropout))

    def _kernels(self, length, device, dtype):
        kernel_size = _odd_kernel(self.kernel_size, length)
        center = kernel_size // 2
        distance = torch.arange(kernel_size, device=device, dtype=dtype).sub(center).abs()
        decay = torch.sigmoid(self.decay_logits).to(device=device, dtype=dtype).clamp(0.01, 0.99)
        basis = decay.unsqueeze(-1).pow(distance.view(1, 1, -1))
        basis = basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        mix = torch.softmax(self.mix_logits.to(device=device, dtype=dtype), dim=-1)
        kernels = (mix.unsqueeze(-1) * basis).sum(dim=1)
        return kernels.unsqueeze(1), center

    def forward(self, x):
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        kernels, padding = self._kernels(x.shape[-1], x.device, x.dtype)
        u = F.conv1d(u, kernels, padding=padding, groups=self.channels)
        return residual + self.dropout(self.out_proj(F.silu(u) * torch.sigmoid(gate)))


class DownsampledAttentionBlock(nn.Module):
    def __init__(self, channels, heads=2, context_length=64, dropout=0.0):
        super().__init__()
        self.context_length = int(context_length)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        residual = x
        pooled = F.adaptive_avg_pool1d(x, self.context_length).transpose(1, 2)
        attended, _ = self.attn(self.norm(pooled), self.norm(pooled), self.norm(pooled), need_weights=False)
        attended = self.proj(attended).transpose(1, 2)
        attended = F.interpolate(attended, size=x.shape[-1], mode="linear", align_corners=False)
        return residual + attended


class DepthwiseDilatedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.0):
        super().__init__()
        padding = int(dilation) * (int(kernel_size) - 1) // 2
        self.norm = nn.GroupNorm(1, int(channels))
        self.depthwise = nn.Conv1d(
            int(channels),
            int(channels),
            int(kernel_size),
            padding=padding,
            dilation=int(dilation),
            groups=int(channels),
        )
        self.pointwise = nn.Conv1d(int(channels), int(channels), 1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        residual = x
        x = self.depthwise(self.norm(x))
        x = self.pointwise(F.silu(x))
        return residual + self.dropout(x)


class FrequencyGuidanceGate(nn.Module):
    def __init__(self, channels, n_fft=64, hop_size=8, win_size=64, low_bins=4, strength=0.20):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_size)
        self.win_size = int(win_size)
        self.low_bins = int(low_bins)
        self.strength = float(strength)
        self.net = nn.Sequential(
            nn.Conv1d(1, max(8, channels // 2), 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(max(8, channels // 2), channels, 1),
        )

    def forward(self, features, noisy):
        signal = noisy.squeeze(1)
        window = torch.hann_window(self.win_size, device=noisy.device, dtype=noisy.dtype)
        spec = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=window,
            center=True,
            return_complex=True,
        )
        low_bins = max(1, min(self.low_bins, spec.shape[1]))
        low_energy = torch.log1p(spec[:, :low_bins].abs().mean(dim=1, keepdim=True))
        low_energy = F.interpolate(low_energy, size=features.shape[-1], mode="linear", align_corners=False)
        gate = torch.tanh(self.net(low_energy)) * self.strength
        return features * (1.0 + gate)


class MorphologyProtectionGate(nn.Module):
    def __init__(self, strength=0.18, kernel_size=17):
        super().__init__()
        self.strength = float(strength)
        self.kernel_size = int(kernel_size)
        self.net = nn.Sequential(
            nn.Conv1d(3, 8, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(8, 1, 1),
        )

    def forward(self, noisy):
        slope = F.pad((noisy[..., 1:] - noisy[..., :-1]).abs(), (1, 0))
        curvature = F.pad((noisy[..., 2:] - 2 * noisy[..., 1:-1] + noisy[..., :-2]).abs(), (1, 1))
        energy = F.avg_pool1d(noisy.abs(), kernel_size=9, stride=1, padding=4, count_include_pad=False)
        stats = torch.cat([slope, curvature, energy], dim=1)
        scale = stats.detach().mean(dim=-1, keepdim=True).clamp_min(1.0e-4)
        gate = torch.sigmoid(self.net(stats / scale))
        kernel_size = _odd_kernel(self.kernel_size, noisy.shape[-1])
        gate = F.avg_pool1d(gate, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)
        return 1.0 - self.strength * gate


class BaselineSentryCore(nn.Module):
    def __init__(
        self,
        hidden_channels=32,
        conv_kernels=(7, 15, 31),
        ssm_layers=2,
        ssm_state_dim=8,
        ssm_kernel_size=129,
        dilated_conv_enabled=False,
        dilated_conv_kernel_size=5,
        dilated_conv_dilations=(1, 2, 4, 8),
        attention_heads=2,
        attention_context=64,
        attention_blocks=1,
        dropout=0.05,
        flow_enabled=False,
        flow_hidden_channels=16,
        baseline_kernel_size=129,
        baseline_delta_budget=0.25,
        delta_gate_init=0.05,
        delta_gate_max=0.20,
        frequency_gate_enabled=False,
        frequency_gate_strength=0.20,
        frequency_n_fft=64,
        frequency_hop_size=8,
        frequency_win_size=64,
        frequency_low_bins=4,
        morphology_gate_enabled=False,
        morphology_gate_strength=0.18,
        morphology_gate_kernel_size=17,
        **kwargs,
    ):
        super().__init__()
        self.flow_enabled = bool(flow_enabled)
        self.morphology_gate_enabled = bool(morphology_gate_enabled)
        self.baseline_kernel_size = int(baseline_kernel_size)
        self.baseline_delta_budget = float(baseline_delta_budget)
        self.delta_gate_max = float(delta_gate_max)
        gate_ratio = min(max(float(delta_gate_init) / max(self.delta_gate_max, 1.0e-6), 1.0e-6), 1.0 - 1.0e-6)
        self.delta_gate_raw = nn.Parameter(torch.logit(torch.tensor(gate_ratio)))

        self.stem = MultiScaleConvStem(1, int(hidden_channels), conv_kernels)
        blocks = []
        for _ in range(int(ssm_layers)):
            blocks.append(
                DiagonalStateSpaceBlock(
                    int(hidden_channels),
                    state_dim=int(ssm_state_dim),
                    kernel_size=int(ssm_kernel_size),
                    dropout=float(dropout),
                )
            )
        if bool(dilated_conv_enabled):
            for dilation in dilated_conv_dilations:
                blocks.append(
                    DepthwiseDilatedResidualBlock(
                        int(hidden_channels),
                        kernel_size=int(dilated_conv_kernel_size),
                        dilation=int(dilation),
                        dropout=float(dropout),
                    )
                )
        for _ in range(int(attention_blocks)):
            blocks.append(
                DownsampledAttentionBlock(
                    int(hidden_channels),
                    heads=int(attention_heads),
                    context_length=int(attention_context),
                    dropout=float(dropout),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.frequency_gate = (
            FrequencyGuidanceGate(
                int(hidden_channels),
                n_fft=int(frequency_n_fft),
                hop_size=int(frequency_hop_size),
                win_size=int(frequency_win_size),
                low_bins=int(frequency_low_bins),
                strength=float(frequency_gate_strength),
            )
            if bool(frequency_gate_enabled)
            else None
        )
        self.morphology_gate = (
            MorphologyProtectionGate(
                strength=float(morphology_gate_strength),
                kernel_size=int(morphology_gate_kernel_size),
            )
            if self.morphology_gate_enabled
            else None
        )
        self.baseline_head = nn.Sequential(
            nn.GroupNorm(1, int(hidden_channels)),
            nn.SiLU(),
            nn.Conv1d(int(hidden_channels), int(hidden_channels), 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(int(hidden_channels), 1, 1),
        )
        if self.flow_enabled:
            self.flow_refiner = nn.Sequential(
                nn.Conv1d(4, int(flow_hidden_channels), 7, padding=3),
                nn.SiLU(),
                nn.Conv1d(int(flow_hidden_channels), int(flow_hidden_channels), 9, padding=8, dilation=2),
                nn.SiLU(),
                nn.Conv1d(int(flow_hidden_channels), 1, 3, padding=1),
            )
        else:
            self.flow_refiner = None

    def _smooth(self, x, kernel_size):
        kernel_size = _odd_kernel(kernel_size, x.shape[-1])
        if kernel_size <= 1:
            return x
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)

    def _limit_delta(self, delta, reference):
        scale = reference.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-4)
        return torch.tanh(delta / scale) * scale * self.baseline_delta_budget

    def predict_baseline(self, noisy):
        features = self.blocks(self.stem(noisy))
        if self.frequency_gate is not None:
            features = self.frequency_gate(features, noisy)
        return self._smooth(self.baseline_head(features), self.baseline_kernel_size)

    def predict_delta(self, noisy, baseline):
        if self.flow_refiner is None:
            return torch.zeros_like(baseline)
        restored = noisy - baseline
        low_noisy = self._smooth(noisy, self.baseline_kernel_size)
        condition = torch.cat([noisy, baseline, restored, low_noisy], dim=1)
        delta = self._smooth(self.flow_refiner(condition), self.baseline_kernel_size)
        return self._limit_delta(delta, baseline)

    def forward(self, noisy):
        baseline = self.predict_baseline(noisy)
        delta = self.predict_delta(noisy, baseline)
        gate = self.delta_gate_max * torch.sigmoid(self.delta_gate_raw)
        correction = baseline + gate * delta
        if self.morphology_gate is not None:
            correction = correction * self.morphology_gate(noisy)
        return noisy - correction


@register_model("baseline_sentry_lite")
@register_model("baseline_sentry_flow")
@register_model("physio_freq_sentry_flow")
class BaselineSentryDenoiser(nn.Module):
    def __init__(
        self,
        loss_fn="time+baseline+lf+morph+smooth",
        lambda_baseline=0.15,
        lambda_lf=0.08,
        lambda_morph=0.025,
        lambda_smooth=0.03,
        lambda_cfm_baseline=0.03,
        **kwargs,
    ):
        super().__init__()
        self.core = BaselineSentryCore(**kwargs)
        self.loss_fn = str(loss_fn).split("+")
        self.lambda_baseline = float(lambda_baseline)
        self.lambda_lf = float(lambda_lf)
        self.lambda_morph = float(lambda_morph)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_cfm_baseline = float(lambda_cfm_baseline)

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected ECG shaped [B, T] or [B, 1, T], got {tuple(x.shape)}.")
        restored = self.core(x)
        return restored.squeeze(1) if squeeze else restored

    @torch.no_grad()
    def denoising(self, x):
        return self.forward(x)

    def _masked_mean(self, value, valid_mask=None):
        if valid_mask is None:
            return value.mean()
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.to(device=value.device, dtype=value.dtype)
        return (value * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        valid_mask = batch[2].to(device) if len(batch) > 2 else None
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)

        baseline = self.core.predict_baseline(noisy)
        delta = self.core.predict_delta(noisy, baseline)
        gate = self.core.delta_gate_max * torch.sigmoid(self.core.delta_gate_raw)
        correction = baseline + gate * delta
        if self.core.morphology_gate is not None:
            correction = correction * self.core.morphology_gate(noisy)
        pred = noisy - correction
        true_baseline = noisy - clean

        loss = torch.zeros((), device=device, dtype=pred.dtype)
        if "time" in self.loss_fn:
            loss = loss + self._masked_mean(F.l1_loss(pred, clean, reduction="none"), valid_mask)
        if "baseline" in self.loss_fn:
            loss = loss + self.lambda_baseline * self._masked_mean(
                F.smooth_l1_loss(baseline, true_baseline, reduction="none"), valid_mask
            )
        if "lf" in self.loss_fn:
            lf_residual = self.core._smooth(pred - clean, self.core.baseline_kernel_size)
            loss = loss + self.lambda_lf * self._masked_mean(lf_residual.abs(), valid_mask)
        if "morph" in self.loss_fn:
            pred_diff = pred[..., 1:] - pred[..., :-1]
            clean_diff = clean[..., 1:] - clean[..., :-1]
            derivative_mask = valid_mask[..., 1:] if valid_mask is not None else None
            loss = loss + self.lambda_morph * self._masked_mean(
                F.l1_loss(pred_diff, clean_diff, reduction="none"), derivative_mask
            )
        if "smooth" in self.loss_fn:
            curvature = baseline[..., 2:] - 2 * baseline[..., 1:-1] + baseline[..., :-2]
            loss = loss + self.lambda_smooth * curvature.abs().mean()
        if "cfm" in self.loss_fn and self.core.flow_enabled:
            target_delta = true_baseline - baseline.detach()
            target_delta = self.core._smooth(target_delta, self.core.baseline_kernel_size)
            loss = loss + self.lambda_cfm_baseline * self._masked_mean(
                F.smooth_l1_loss(delta, target_delta, reduction="none"), valid_mask
            )
        return loss
