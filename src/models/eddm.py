import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionCoefficientEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.scalar_dim = max(self.dim // 2, 2)

    def _embed_scalar(self, value):
        half = self.scalar_dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=value.device, dtype=torch.float32) * -scale)
        emb = value.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.scalar_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, coefficients):
        if coefficients.ndim != 2 or coefficients.shape[1] != 2:
            raise ValueError(f"Expected diffusion coefficients shaped [B, 2], got {tuple(coefficients.shape)}.")
        emb = torch.cat(
            [
                self._embed_scalar(coefficients[:, 0]),
                self._embed_scalar(coefficients[:, 1]),
            ],
            dim=1,
        )
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb[:, : self.dim]


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8, dropout=0.0):
        super().__init__()
        group_count = min(groups, out_channels)
        while out_channels % group_count != 0:
            group_count -= 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(group_count, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x + self.time_proj(time_emb).unsqueeze(-1))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.silu(x + residual)


class DeepAggregationPyramidPooling1d(nn.Module):
    def __init__(self, channels, pool_scales=(3, 5, 9, 15), use_global_pool=True):
        super().__init__()
        self.pool_scales = tuple(pool_scales)
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
        self.global_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels, 1),
            nn.SiLU(),
        ) if self.use_global_pool else None
        feature_count = len(self.pool_scales) + 1 + int(self.use_global_pool)
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


class SelfAttention1d(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=int(channels),
            num_heads=int(heads),
            batch_first=True,
        )

    def forward(self, x):
        residual = x
        sequence = self.norm(x).transpose(1, 2)
        attended, _ = self.attn(sequence, sequence, sequence, need_weights=False)
        return residual + attended.transpose(1, 2)


class DownBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.block = ConvBlock1d(in_channels, out_channels, time_dim, dropout=dropout)
        self.pool = nn.Conv1d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, time_emb):
        skip = self.block(x, time_emb)
        return self.pool(skip), skip


class UpBlock1d(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ConvBlock1d(out_channels + skip_channels, out_channels, time_dim, dropout=dropout)

    def forward(self, x, skip, time_emb):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1), time_emb)


class EDDMNoiseUNet1d(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_dim=256,
        dropout=0.0,
        pool_scales=(3, 5, 9, 15),
        attention_heads=4,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            DiffusionCoefficientEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        channels = [base_channels * mult for mult in channel_mults]
        self.input = nn.Conv1d(in_channels, channels[0], 3, padding=1)
        self.downs = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            self.downs.append(DownBlock1d(in_ch, out_ch, time_dim, dropout=dropout))
            in_ch = out_ch

        self.mid = nn.ModuleList(
            [
                ConvBlock1d(channels[-1], channels[-1], time_dim, dropout=dropout),
                SelfAttention1d(channels[-1], heads=attention_heads),
                ConvBlock1d(channels[-1], channels[-1], time_dim, dropout=dropout),
            ]
        )
        self.first_skip_dapp = DeepAggregationPyramidPooling1d(
            channels[0],
            pool_scales=pool_scales,
            use_global_pool=True,
        )

        self.ups = nn.ModuleList()
        current = channels[-1]
        for skip_ch in reversed(channels):
            out_ch = skip_ch
            self.ups.append(UpBlock1d(current, skip_ch, out_ch, time_dim, dropout=dropout))
            current = out_ch

        self.output = nn.Sequential(
            nn.GroupNorm(1, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], 1, 3, padding=1),
        )

    def forward(self, x, coefficients):
        time_emb = self.time_mlp(coefficients)
        x = self.input(x)
        skips = []
        for index, down in enumerate(self.downs):
            x, skip = down(x, time_emb)
            if index == 0:
                skip = self.first_skip_dapp(skip)
            skips.append(skip)
        for layer in self.mid:
            if isinstance(layer, ConvBlock1d):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, time_emb)
        return self.output(x)


@register_model("eddm")
class EDDMDenoiser(nn.Module):
    """EDDM dual-path ECG denoising diffusion model (Li et al. 2025, IEEE TIM).

    Implements the paper's dual-path forward process (Eq. 5): x_t = x_0 +
    alpha_bar_t * e + beta_bar_t * eps, where e = noisy-clean is the fixed
    colored (ECG) noise for the whole trajectory and eps ~ N(0, I) is
    resampled per step. Per-step coefficients alpha_t (colored-noise branch)
    and beta_t (Gaussian branch) both follow the paper's "linear decrease"
    strategy (Section IV-C2); alpha_t is additionally normalized so that
    alpha_bar_T = sum(alpha_1..T) = 1, which the paper requires so that the
    diffusion endpoint x_T equals the noisy ECG plus a small residual
    Gaussian perturbation (x_T = x_tilde + eps), not pure Gaussian noise as
    in standard DDPM. beta_t is scaled by `gaussian_scale` since the paper
    does not give an absolute magnitude for the white-noise branch.

    The reverse process (Algorithm 2 / Eq. 6-7) is a *stochastic* ancestral
    sampler -- x_{t-1} = x_t - alpha_t*e_hat - (beta_t^2/beta_bar_t)*eps_hat
    + sigma_t*eps', with posterior variance sigma_t^2 = beta_t^2 *
    beta_bar_{t-1}^2 / beta_bar_t^2 (Eq. 6) -- unlike a DDIM-style
    deterministic "predict-x0-then-re-add-noise" shortcut. This stochastic
    step is what makes the paper's "EDDM-k" multifold ensemble (repeating
    the reverse process k times and averaging, see `num_shots`) meaningful
    at all: a deterministic reverse process would produce identical repeats.

    Note: the paper does not have a public reference implementation, so the
    specific closed-form used to construct linearly-decreasing alpha_t/beta_t
    (normalized to sum to 1) is this port's best-effort reconstruction of the
    stated design constraints, not a verified copy of released code.
    """

    def __init__(
        self,
        timesteps=50,
        num_shots=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_dim=256,
        dropout=0.0,
        gaussian_scale=0.2,
        ecg_noise_weight=1.0,
        gaussian_noise_weight=1.0,
        pool_scales=(3, 5, 9, 15),
        attention_heads=4,
        **kwargs,
    ):
        super().__init__()
        self.timesteps = int(timesteps)
        self.num_shots = max(int(num_shots), 1)
        self.gaussian_scale = float(gaussian_scale)
        self.ecg_noise_weight = float(ecg_noise_weight)
        self.gaussian_noise_weight = float(gaussian_noise_weight)
        self.ecg_noise_model = EDDMNoiseUNet1d(
            in_channels=2,
            base_channels=int(base_channels),
            channel_mults=tuple(channel_mults),
            time_dim=int(time_dim),
            dropout=float(dropout),
            pool_scales=tuple(pool_scales),
            attention_heads=int(attention_heads),
        )
        self.white_noise_model = EDDMNoiseUNet1d(
            in_channels=2,
            base_channels=int(base_channels),
            channel_mults=tuple(channel_mults),
            time_dim=int(time_dim),
            dropout=float(dropout),
            pool_scales=tuple(pool_scales),
            attention_heads=int(attention_heads),
        )

        T = self.timesteps
        step_index = torch.arange(1, T + 1, dtype=torch.float64)
        # Linearly decreasing positive weights over t=1..T, normalized to
        # sum to 1 (Eq. 5's alpha_bar_T = 1 requirement).
        linear_decreasing_weights = (T - step_index + 1) / (T * (T + 1) / 2.0)
        alpha_t = linear_decreasing_weights.to(torch.float32)
        beta_t = (self.gaussian_scale * linear_decreasing_weights).to(torch.float32)

        alpha_bar = torch.cumsum(alpha_t, dim=0)
        beta_bar = torch.sqrt(torch.cumsum(beta_t.pow(2), dim=0))
        beta_bar_prev = F.pad(beta_bar[:-1], (1, 0), value=0.0)
        posterior_var = (
            beta_t.pow(2) * beta_bar_prev.pow(2) / beta_bar.pow(2).clamp_min(1.0e-12)
        )

        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("beta_t", beta_t)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("beta_bar", beta_bar)
        self.register_buffer("posterior_var", posterior_var)

    def _predict(self, xt, noisy, t):
        model_input = torch.cat([xt, noisy], dim=1)
        coefficients = torch.stack([self.alpha_bar[t], self.beta_bar[t]], dim=1)
        return self.ecg_noise_model(model_input, coefficients), self.white_noise_model(model_input, coefficients)

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        current = x
        # Algorithm 2 iterates every one of the T diffusion steps -- unlike
        # DDIM-style samplers, the per-step alpha_t/beta_t coefficients tie
        # each reverse step to its specific neighbor, so steps cannot be
        # skipped without breaking Eq. (6)-(7).
        for t_value in reversed(range(self.timesteps)):
            t = torch.full((x.shape[0],), t_value, device=x.device, dtype=torch.long)
            ecg_noise_hat, gaussian_noise_hat = self._predict(current, x, t)
            alpha_t = self.alpha_t[t].view(-1, 1, 1)
            beta_t = self.beta_t[t].view(-1, 1, 1)
            beta_bar_t = self.beta_bar[t].view(-1, 1, 1)
            sigma_t = self.posterior_var[t].clamp_min(0.0).sqrt().view(-1, 1, 1)
            noise = torch.randn_like(current) if t_value > 0 else torch.zeros_like(current)
            # Eq. (7): x_{t-1} = x_t - alpha_t*e_hat - (beta_t^2/beta_bar_t)*eps_hat + sigma_t*eps'
            current = (
                current
                - alpha_t * ecg_noise_hat
                - (beta_t.pow(2) / beta_bar_t.clamp_min(1.0e-12)) * gaussian_noise_hat
                + sigma_t * noise
            )
        return current.squeeze(1) if squeeze else current

    @torch.no_grad()
    def denoising(self, x):
        if self.num_shots <= 1:
            return self.forward(x)
        # Paper's "EDDM-k" multifold ensemble (Section IV-C2): average k
        # independent stochastic reverse-process runs.
        samples = torch.stack([self.forward(x) for _ in range(self.num_shots)], dim=0)
        return samples.mean(dim=0)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        valid_mask = batch[2].to(device) if len(batch) > 2 else None
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)

        batch_size = noisy.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        ecg_noise = noisy - clean
        gaussian_noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bar[t].view(-1, 1, 1)
        beta_bar = self.beta_bar[t].view(-1, 1, 1)
        xt = clean + alpha_bar * ecg_noise + beta_bar * gaussian_noise

        pred_ecg_noise, pred_gaussian_noise = self._predict(xt, noisy, t)

        if valid_mask is not None:
            valid_mask = valid_mask.to(device, dtype=clean.dtype)
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(1)
            denom = valid_mask.sum().clamp_min(1.0)

            def masked_mse(a, b):
                return ((a - b).pow(2) * valid_mask).sum() / denom

            ecg_loss = masked_mse(pred_ecg_noise, ecg_noise)
            gaussian_loss = masked_mse(pred_gaussian_noise, gaussian_noise)
        else:
            ecg_loss = F.mse_loss(pred_ecg_noise, ecg_noise)
            gaussian_loss = F.mse_loss(pred_gaussian_noise, gaussian_noise)

        # Eq. (8): L(theta) = E[||eps - eps_theta||^2] + E[||e - e_theta||^2]
        # (equal, unweighted combination of the two noise-prediction losses).
        return (
            self.ecg_noise_weight * ecg_loss
            + self.gaussian_noise_weight * gaussian_loss
        )
