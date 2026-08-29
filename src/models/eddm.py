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
    strategy (Section IV-C2), normalized to the paper's two endpoint
    constraints:
      sum(alpha_1..T)  = 1  ->  alpha_bar_T = 1  (Eq. 5)
      sum(beta_1..T^2) = 1  ->  beta_bar_T  = 1  (Section III-B: x_T =
                                                  x_tilde + eps exactly, not
                                                  a scaled-down perturbation)
    `gaussian_scale` (default 1.0 = paper-faithful) rescales beta_bar_T, i.e.
    the std of the white noise present at the diffusion endpoint.

    The reverse process (Algorithm 2 / Eq. 6-7) is a *stochastic* ancestral
    sampler -- x_{t-1} = x_t - alpha_t*e_hat - (beta_t^2/beta_bar_t)*eps_hat
    + sigma_t*eps', with posterior variance sigma_t^2 = beta_t^2 *
    beta_bar_{t-1}^2 / beta_bar_t^2 (Eq. 6) -- unlike a DDIM-style
    deterministic "predict-x0-then-re-add-noise" shortcut. This stochastic
    step is what makes the paper's "EDDM-k" multifold ensemble (repeating
    the reverse process k times and averaging, see `num_shots`) meaningful
    at all: a deterministic reverse process would produce identical repeats.

    Verified against the paper: dual-path diffusion (Eq. 5), reverse update
    (Eq. 7) x_{t-1} = x_t - alpha_t*e_hat - (beta_t^2/beta_bar_t)*eps_hat +
    sigma_t*eps' with sigma_t^2 = beta_t^2 * beta_bar_{t-1}^2 / beta_bar_t^2
    (Eq. 6), L2 dual-noise-prediction loss (Eq. 8, unweighted), conditioning
    on (x_t, alpha_bar_t, beta_bar_t, x_tilde), T=50, two 1-D U-Nets with
    four DownBlocks + self-attention mid-block + a DAPP module on the first
    skip connection, RAdam / lr 1e-5 / batch 64, and the "EDDM-k" multifold
    ensemble (Sec. IV-C2, `num_shots`).

    Not verifiable (no public reference implementation): the exact closed
    form of the linear-decrease alpha_t/beta_t ramp, the DAPP pool scales
    (Fig. 3 shows 3/5/9 + global, the text says "five pooling layers" -- we
    keep 3/5/9/15 + global, matching the DeepFilter multi-kernel design the
    paper cites), and the U-Net channel widths / attention-head count (the
    paper gives none).
    """

    def __init__(
        self,
        timesteps=50,
        num_shots=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_dim=256,
        dropout=0.0,
        gaussian_scale=1.0,
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
        # "Linear decrease" schedule (Sec. IV-C2): both sequences ramp down
        # linearly over t = 1..T. Normalisations enforce the paper's endpoint
        # constraints -- sum(alpha_t) = 1 (alpha_bar_T = 1, Eq. 5) and
        # sum(beta_t^2) = 1 (beta_bar_T = 1, so x_T = x_tilde + eps exactly).
        linear_decrease = T - step_index + 1.0  # T, T-1, ..., 1
        alpha_t = (linear_decrease / linear_decrease.sum()).to(torch.float32)
        beta_t = (
            self.gaussian_scale * linear_decrease / linear_decrease.pow(2).sum().sqrt()
        ).to(torch.float32)

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
        samples = self.denoising_shots(x)
        return samples.mean(dim=0)

    @torch.no_grad()
    def denoising_shots(self, x):
        return torch.stack([self.forward(x) for _ in range(self.num_shots)], dim=0)

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


@register_model("eddm_flow_matching")
class EDDMFlowMatchingDenoiser(nn.Module):
    """EDDM-style conditional flow matching denoiser.

    This keeps EDDM's two 1-D U-Nets, DAPP first skip, self-attention
    bottleneck, and conditioning on the current state plus the observed noisy
    ECG. The stochastic diffusion reverse process is replaced by a
    deterministic flow sampler over the same dual components:

        x_t = clean + alpha_bar(t) * e + beta_bar(t) * eps
        e = noisy - clean

    The loss supervises the vector field directly, not PRD/SSD/MAD/CosSim.
    """

    def __init__(
        self,
        timesteps=8,
        num_shots=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_dim=256,
        dropout=0.0,
        pool_scales=(3, 5, 9, 15),
        attention_heads=4,
        loss_type="smooth_l1",
        ecg_weight=1.0,
        gaussian_weight=0.15,
        gaussian_inference_weight=0.10,
        recon_weight=0.25,
        lowfreq_weight=0.10,
        lowfreq_kernel_size=129,
        no_gaussian_weight=0.50,
        endpoint_weight=0.50,
        inference_lowpass_ecg=True,
        inference_highfreq_weight=0.10,
        sampler="heun",
        velocity_clip=8.0,
        state_clip=8.0,
        output_blend=1.0,
        gaussian_scale=0.35,
        **kwargs,
    ):
        super().__init__()
        self.timesteps = max(int(timesteps), 1)
        self.num_shots = max(int(num_shots), 1)
        self.loss_type = str(loss_type).lower()
        self.ecg_weight = float(ecg_weight)
        self.gaussian_weight = float(gaussian_weight)
        self.gaussian_inference_weight = float(gaussian_inference_weight)
        self.recon_weight = float(recon_weight)
        self.lowfreq_weight = float(lowfreq_weight)
        self.lowfreq_kernel_size = int(lowfreq_kernel_size)
        self.no_gaussian_weight = float(no_gaussian_weight)
        self.endpoint_weight = float(endpoint_weight)
        self.inference_lowpass_ecg = bool(inference_lowpass_ecg)
        self.inference_highfreq_weight = float(inference_highfreq_weight)
        self.sampler = str(sampler).lower()
        self.velocity_clip = float(velocity_clip)
        self.state_clip = float(state_clip)
        self.output_blend = float(output_blend)
        self.gaussian_scale = float(gaussian_scale)
        self.ecg_velocity_model = EDDMNoiseUNet1d(
            in_channels=2,
            base_channels=int(base_channels),
            channel_mults=tuple(channel_mults),
            time_dim=int(time_dim),
            dropout=float(dropout),
            pool_scales=tuple(pool_scales),
            attention_heads=int(attention_heads),
        )
        self.gaussian_velocity_model = EDDMNoiseUNet1d(
            in_channels=2,
            base_channels=int(base_channels),
            channel_mults=tuple(channel_mults),
            time_dim=int(time_dim),
            dropout=float(dropout),
            pool_scales=tuple(pool_scales),
            attention_heads=int(attention_heads),
        )

        step_index = torch.arange(1, self.timesteps + 1, dtype=torch.float64)
        linear_decrease = self.timesteps - step_index + 1.0
        alpha_t = (linear_decrease / linear_decrease.sum()).to(torch.float32)
        beta_t = (
            self.gaussian_scale * linear_decrease / linear_decrease.pow(2).sum().sqrt()
        ).to(torch.float32)
        alpha_bar = torch.cumsum(alpha_t, dim=0)
        beta_bar = torch.sqrt(torch.cumsum(beta_t.pow(2), dim=0))
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=0.0)
        beta_bar_prev = F.pad(beta_bar[:-1], (1, 0), value=0.0)

        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("beta_t", beta_t)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("beta_bar", beta_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("beta_bar_prev", beta_bar_prev)

    def _robust_scale(self, x):
        scale = x.detach().abs().mean(dim=-1, keepdim=True)
        scale = scale + 0.25 * x.detach().std(dim=-1, keepdim=True)
        return scale.clamp_min(1.0e-4)

    def _clip_by_robust_scale(self, x, multiple):
        if multiple <= 0:
            return x
        bound = float(multiple) * self._robust_scale(x)
        return torch.clamp(x, min=-bound, max=bound)

    def _smooth_1d(self, x, kernel_size):
        if kernel_size <= 1:
            return x
        kernel_size = min(kernel_size, x.shape[-1] if x.shape[-1] % 2 == 1 else x.shape[-1] - 1)
        kernel_size = max(kernel_size, 1)
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)

    def _project_inference_ecg_noise(self, ecg_noise):
        if not self.inference_lowpass_ecg:
            return ecg_noise
        low = self._smooth_1d(ecg_noise, self.lowfreq_kernel_size)
        high = ecg_noise - low
        return low + self.inference_highfreq_weight * high

    def _loss_elementwise(self, prediction, target):
        if self.loss_type == "l1":
            return F.l1_loss(prediction, target, reduction="none")
        if self.loss_type == "mse":
            return F.mse_loss(prediction, target, reduction="none")
        return F.smooth_l1_loss(prediction, target, reduction="none")

    def _coefficients(self, t):
        if t.dtype == torch.long:
            return torch.stack([self.alpha_bar[t], self.beta_bar[t]], dim=1)
        t = t.float().clamp(0.0, 1.0)
        return torch.stack([t, self.gaussian_scale * torch.sqrt(t.clamp_min(0.0))], dim=1)

    def _predict_components(self, xt, noisy, t):
        xt = self._clip_by_robust_scale(torch.nan_to_num(xt), self.state_clip)
        model_input = torch.cat([xt, noisy], dim=1)
        coefficients = self._coefficients(t)
        ecg_velocity = self.ecg_velocity_model(model_input, coefficients)
        gaussian_velocity = self.gaussian_velocity_model(model_input, coefficients)
        return (
            self._clip_by_robust_scale(torch.nan_to_num(ecg_velocity), self.velocity_clip),
            self._clip_by_robust_scale(torch.nan_to_num(gaussian_velocity), self.velocity_clip),
        )

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        noisy = x
        current = x
        for t_value in reversed(range(self.timesteps)):
            t = torch.full((x.shape[0],), t_value, device=x.device, dtype=torch.long)
            ecg_velocity, gaussian_velocity = self._predict_components(current, noisy, t)
            ecg_velocity = self._project_inference_ecg_noise(ecg_velocity)
            alpha_delta = (self.alpha_bar[t] - self.alpha_bar_prev[t]).view(-1, 1, 1)
            beta_delta = (self.beta_bar[t] - self.beta_bar_prev[t]).view(-1, 1, 1)
            delta = alpha_delta * ecg_velocity
            if self.gaussian_inference_weight > 0:
                delta = delta + self.gaussian_inference_weight * beta_delta * gaussian_velocity
            if self.sampler == "heun" and t_value > 0:
                provisional = self._clip_by_robust_scale(torch.nan_to_num(current - delta), self.state_clip)
                prev_t = torch.full((x.shape[0],), t_value - 1, device=x.device, dtype=torch.long)
                ecg_velocity_2, gaussian_velocity_2 = self._predict_components(provisional, noisy, prev_t)
                ecg_velocity_2 = self._project_inference_ecg_noise(ecg_velocity_2)
                corrected_delta = alpha_delta * 0.5 * (ecg_velocity + ecg_velocity_2)
                if self.gaussian_inference_weight > 0:
                    corrected_delta = corrected_delta + self.gaussian_inference_weight * beta_delta * 0.5 * (
                        gaussian_velocity + gaussian_velocity_2
                    )
                current = current - corrected_delta
            else:
                current = current - delta
            current = self._clip_by_robust_scale(torch.nan_to_num(current), self.state_clip)
        output = noisy + self.output_blend * (current - noisy)
        return output.squeeze(1) if squeeze else output

    @torch.no_grad()
    def denoising(self, x):
        if self.num_shots <= 1:
            return self.forward(x)
        return torch.stack([self.forward(x) for _ in range(self.num_shots)], dim=0).mean(dim=0)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        valid_mask = batch[2].to(device) if len(batch) > 2 else None
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)

        batch_size = noisy.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        ecg_noise = self._clip_by_robust_scale(
            torch.nan_to_num(noisy - clean),
            self.velocity_clip,
        )
        gaussian_noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bar[t].view(-1, 1, 1)
        beta_bar = self.beta_bar[t].view(-1, 1, 1)
        xt = clean + alpha_bar * ecg_noise + beta_bar * gaussian_noise
        predicted_ecg, predicted_gaussian = self._predict_components(xt, noisy, t)
        ecg_loss = self._loss_elementwise(predicted_ecg, ecg_noise)
        gaussian_loss = self._loss_elementwise(predicted_gaussian, gaussian_noise)
        loss = self.ecg_weight * ecg_loss + self.gaussian_weight * gaussian_loss

        if self.recon_weight > 0:
            clean_hat = xt - alpha_bar * predicted_ecg - beta_bar * predicted_gaussian
            recon_loss = self._loss_elementwise(clean_hat, clean)
            loss = loss + self.recon_weight * recon_loss

        if self.lowfreq_weight > 0:
            predicted_low = self._smooth_1d(predicted_ecg, self.lowfreq_kernel_size)
            target_low = self._smooth_1d(ecg_noise, self.lowfreq_kernel_size)
            lowfreq_loss = self._loss_elementwise(predicted_low, target_low)
            loss = loss + self.lowfreq_weight * lowfreq_loss

        if self.no_gaussian_weight > 0:
            xt_no_gaussian = clean + alpha_bar * ecg_noise
            predicted_ecg_ng, _ = self._predict_components(xt_no_gaussian, noisy, t)
            clean_hat_ng = xt_no_gaussian - alpha_bar * predicted_ecg_ng
            no_gaussian_loss = self._loss_elementwise(predicted_ecg_ng, ecg_noise)
            no_gaussian_recon = self._loss_elementwise(clean_hat_ng, clean)
            loss = loss + self.no_gaussian_weight * (no_gaussian_loss + self.recon_weight * no_gaussian_recon)

        if self.endpoint_weight > 0:
            endpoint_t = torch.full((batch_size,), self.timesteps - 1, device=device, dtype=torch.long)
            predicted_endpoint_ecg, _ = self._predict_components(noisy, noisy, endpoint_t)
            endpoint_clean_hat = noisy - predicted_endpoint_ecg
            endpoint_loss = self._loss_elementwise(predicted_endpoint_ecg, ecg_noise)
            endpoint_recon = self._loss_elementwise(endpoint_clean_hat, clean)
            endpoint_low = self._loss_elementwise(
                self._smooth_1d(predicted_endpoint_ecg, self.lowfreq_kernel_size),
                self._smooth_1d(ecg_noise, self.lowfreq_kernel_size),
            )
            loss = loss + self.endpoint_weight * (
                endpoint_loss
                + self.recon_weight * endpoint_recon
                + self.lowfreq_weight * endpoint_low
            )

        if valid_mask is not None:
            valid_mask = valid_mask.to(device, dtype=clean.dtype)
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(1)
            return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        return loss.mean()
