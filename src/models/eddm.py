import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
    from .mecg_e import MambaBlock
except ImportError:
    from factory import register_model
    from mecg_e import MambaBlock


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


class RBFCoefficientEmbedding(nn.Module):
    """Small KAN-inspired nonlinear basis embedding for flow coefficients.

    This is intentionally much lighter than replacing the U-Net with KAN
    layers: it only enriches alpha/beta conditioning with learnable RBF bases.
    """

    def __init__(self, dim, basis_count=16):
        super().__init__()
        self.dim = int(dim)
        self.basis_count = int(basis_count)
        centers = torch.linspace(0.0, 1.0, self.basis_count)
        self.register_buffer("centers", centers)
        self.log_width = nn.Parameter(torch.zeros(2, self.basis_count))
        self.proj = nn.Sequential(
            nn.Linear(2 + 2 * self.basis_count, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, coefficients):
        if coefficients.ndim != 2 or coefficients.shape[1] != 2:
            raise ValueError(f"Expected coefficients shaped [B, 2], got {tuple(coefficients.shape)}.")
        values = coefficients.float().clamp(0.0, 1.0)
        width = F.softplus(self.log_width).clamp_min(1.0e-3)
        basis = torch.exp(
            -((values.unsqueeze(-1) - self.centers.view(1, 1, -1)) ** 2)
            * width.view(1, 2, -1)
        )
        features = torch.cat([values, basis.flatten(1)], dim=1)
        return self.proj(features)


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


class MambaBottleneck1d(nn.Module):
    def __init__(
        self,
        channels,
        n_layer=1,
        d_state=16,
        d_conv=4,
        expand=2,
        norm_epsilon=1.0e-5,
        gate_init=0.10,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.mamba = MambaBlock(
            int(channels),
            n_layer=int(n_layer),
            bidirectional=False,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            norm_epsilon=float(norm_epsilon),
        )
        gate_init = min(max(float(gate_init), 1.0e-6), 1.0 - 1.0e-6)
        self.gate_raw = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init)), dtype=torch.float32))

    def forward(self, x):
        residual = x
        sequence = self.norm(x).transpose(1, 2)
        out = self.mamba(sequence).transpose(1, 2)
        gate = torch.sigmoid(self.gate_raw)
        return residual + gate * out


class DilatedConvBottleneck1d(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 4, 8), dropout=0.0, gate_init=0.15):
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in tuple(dilations):
            dilation = int(dilation)
            self.branches.append(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    nn.SiLU(),
                    nn.Conv1d(
                        int(channels),
                        int(channels),
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=int(channels),
                    ),
                    nn.Conv1d(int(channels), int(channels), kernel_size=1),
                    nn.Dropout(float(dropout)),
                )
            )
        self.fuse = nn.Sequential(
            nn.GroupNorm(1, int(channels) * len(self.branches)),
            nn.SiLU(),
            nn.Conv1d(int(channels) * len(self.branches), int(channels), kernel_size=1),
        )
        gate_init = min(max(float(gate_init), 1.0e-6), 1.0 - 1.0e-6)
        self.gate_raw = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init)), dtype=torch.float32))

    def forward(self, x):
        if not self.branches:
            return x
        features = [branch(x) for branch in self.branches]
        residual = self.fuse(torch.cat(features, dim=1))
        return x + torch.sigmoid(self.gate_raw) * residual


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
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_dim=256,
        dropout=0.0,
        pool_scales=(3, 5, 9, 15),
        attention_heads=4,
        mamba_layers=0,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_norm_epsilon=1.0e-5,
        mamba_gate_init=0.10,
        bottleneck_dilated_conv=False,
        bottleneck_dilations=(1, 2, 4, 8),
        bottleneck_dilated_gate_init=0.15,
        coefficient_embedding="sinusoidal",
        coefficient_basis_count=16,
    ):
        super().__init__()
        if str(coefficient_embedding).lower() in {"rbf", "kan", "rbf_kan"}:
            embedding = RBFCoefficientEmbedding(time_dim, basis_count=int(coefficient_basis_count))
        else:
            embedding = DiffusionCoefficientEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            embedding,
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

        mid_layers = [
            ConvBlock1d(channels[-1], channels[-1], time_dim, dropout=dropout),
            SelfAttention1d(channels[-1], heads=attention_heads),
        ]
        if bool(bottleneck_dilated_conv):
            mid_layers.append(
                DilatedConvBottleneck1d(
                    channels[-1],
                    dilations=tuple(bottleneck_dilations),
                    dropout=float(dropout),
                    gate_init=float(bottleneck_dilated_gate_init),
                )
            )
        if int(mamba_layers) > 0:
            mid_layers.append(
                MambaBottleneck1d(
                    channels[-1],
                    n_layer=int(mamba_layers),
                    d_state=int(mamba_d_state),
                    d_conv=int(mamba_d_conv),
                    expand=int(mamba_expand),
                    norm_epsilon=float(mamba_norm_epsilon),
                    gate_init=float(mamba_gate_init),
                )
            )
        mid_layers.append(ConvBlock1d(channels[-1], channels[-1], time_dim, dropout=dropout))
        self.mid = nn.ModuleList(mid_layers)
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
            nn.Conv1d(channels[0], int(out_channels), 3, padding=1),
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
        highfreq_weight=0.08,
        baseline_smooth_weight=0.05,
        baseline_spectrum_weight=0.05,
        spectrum_n_fft=1024,
        spectrum_low_hz=0.8,
        sampling_rate=360,
        lowfreq_kernel_size=129,
        lowfreq_kernel_sizes=None,
        no_gaussian_weight=0.50,
        endpoint_weight=0.50,
        inference_lowpass_ecg=True,
        inference_highfreq_weight=0.10,
        sampler="heun",
        shared_backbone=False,
        final_endpoint_blend=0.25,
        final_lowpass_baseline=True,
        mamba_layers=0,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_norm_epsilon=1.0e-5,
        mamba_gate_init=0.10,
        bottleneck_dilated_conv=False,
        bottleneck_dilations=(1, 2, 4, 8),
        bottleneck_dilated_gate_init=0.15,
        coefficient_embedding="sinusoidal",
        coefficient_basis_count=16,
        antithetic_weight=0.0,
        antithetic_ecg_weight=1.0,
        antithetic_gaussian_weight=0.25,
        aux_warmup_steps=0,
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
        self.highfreq_weight = float(highfreq_weight)
        self.baseline_smooth_weight = float(baseline_smooth_weight)
        self.baseline_spectrum_weight = float(baseline_spectrum_weight)
        self.spectrum_n_fft = int(spectrum_n_fft)
        self.spectrum_low_hz = float(spectrum_low_hz)
        self.sampling_rate = float(sampling_rate)
        self.lowfreq_kernel_size = int(lowfreq_kernel_size)
        if lowfreq_kernel_sizes is None:
            lowfreq_kernel_sizes = [self.lowfreq_kernel_size]
        self.lowfreq_kernel_sizes = tuple(int(kernel) for kernel in lowfreq_kernel_sizes)
        self.no_gaussian_weight = float(no_gaussian_weight)
        self.endpoint_weight = float(endpoint_weight)
        self.inference_lowpass_ecg = bool(inference_lowpass_ecg)
        self.inference_highfreq_weight = float(inference_highfreq_weight)
        self.sampler = str(sampler).lower()
        self.shared_backbone = bool(shared_backbone)
        self.final_endpoint_blend = float(final_endpoint_blend)
        self.final_lowpass_baseline = bool(final_lowpass_baseline)
        self.antithetic_weight = float(antithetic_weight)
        self.antithetic_ecg_weight = float(antithetic_ecg_weight)
        self.antithetic_gaussian_weight = float(antithetic_gaussian_weight)
        self.aux_warmup_steps = int(aux_warmup_steps)
        self.velocity_clip = float(velocity_clip)
        self.state_clip = float(state_clip)
        self.output_blend = float(output_blend)
        self.gaussian_scale = float(gaussian_scale)
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long))
        if self.shared_backbone:
            self.dual_velocity_model = EDDMNoiseUNet1d(
                in_channels=2,
                out_channels=2,
                base_channels=int(base_channels),
                channel_mults=tuple(channel_mults),
                time_dim=int(time_dim),
                dropout=float(dropout),
                pool_scales=tuple(pool_scales),
                attention_heads=int(attention_heads),
                mamba_layers=int(mamba_layers),
                mamba_d_state=int(mamba_d_state),
                mamba_d_conv=int(mamba_d_conv),
                mamba_expand=int(mamba_expand),
                mamba_norm_epsilon=float(mamba_norm_epsilon),
                mamba_gate_init=float(mamba_gate_init),
                bottleneck_dilated_conv=bool(bottleneck_dilated_conv),
                bottleneck_dilations=tuple(bottleneck_dilations),
                bottleneck_dilated_gate_init=float(bottleneck_dilated_gate_init),
                coefficient_embedding=str(coefficient_embedding),
                coefficient_basis_count=int(coefficient_basis_count),
            )
            self.ecg_velocity_model = None
            self.gaussian_velocity_model = None
        else:
            self.dual_velocity_model = None
            self.ecg_velocity_model = EDDMNoiseUNet1d(
                in_channels=2,
                out_channels=1,
                base_channels=int(base_channels),
                channel_mults=tuple(channel_mults),
                time_dim=int(time_dim),
                dropout=float(dropout),
                pool_scales=tuple(pool_scales),
                attention_heads=int(attention_heads),
                mamba_layers=int(mamba_layers),
                mamba_d_state=int(mamba_d_state),
                mamba_d_conv=int(mamba_d_conv),
                mamba_expand=int(mamba_expand),
                mamba_norm_epsilon=float(mamba_norm_epsilon),
                mamba_gate_init=float(mamba_gate_init),
                bottleneck_dilated_conv=bool(bottleneck_dilated_conv),
                bottleneck_dilations=tuple(bottleneck_dilations),
                bottleneck_dilated_gate_init=float(bottleneck_dilated_gate_init),
                coefficient_embedding=str(coefficient_embedding),
                coefficient_basis_count=int(coefficient_basis_count),
            )
            self.gaussian_velocity_model = EDDMNoiseUNet1d(
                in_channels=2,
                out_channels=1,
                base_channels=int(base_channels),
                channel_mults=tuple(channel_mults),
                time_dim=int(time_dim),
                dropout=float(dropout),
                pool_scales=tuple(pool_scales),
                attention_heads=int(attention_heads),
                mamba_layers=int(mamba_layers),
                mamba_d_state=int(mamba_d_state),
                mamba_d_conv=int(mamba_d_conv),
                mamba_expand=int(mamba_expand),
                mamba_norm_epsilon=float(mamba_norm_epsilon),
                mamba_gate_init=float(mamba_gate_init),
                bottleneck_dilated_conv=bool(bottleneck_dilated_conv),
                bottleneck_dilations=tuple(bottleneck_dilations),
                bottleneck_dilated_gate_init=float(bottleneck_dilated_gate_init),
                coefficient_embedding=str(coefficient_embedding),
                coefficient_basis_count=int(coefficient_basis_count),
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

    def _multi_scale_lowpass(self, x):
        if not self.lowfreq_kernel_sizes:
            return self._smooth_1d(x, self.lowfreq_kernel_size)
        lowpasses = [self._smooth_1d(x, kernel_size) for kernel_size in self.lowfreq_kernel_sizes]
        return torch.stack(lowpasses, dim=0).mean(dim=0)

    def _project_inference_ecg_noise(self, ecg_noise):
        if not self.inference_lowpass_ecg:
            return ecg_noise
        low = self._multi_scale_lowpass(ecg_noise)
        high = ecg_noise - low
        return low + self.inference_highfreq_weight * high

    def _loss_elementwise(self, prediction, target):
        if self.loss_type == "l1":
            return F.l1_loss(prediction, target, reduction="none")
        if self.loss_type == "mse":
            return F.mse_loss(prediction, target, reduction="none")
        return F.smooth_l1_loss(prediction, target, reduction="none")

    def _baseline_spectrum_loss(self, predicted_baseline, target_baseline):
        if self.baseline_spectrum_weight <= 0:
            return predicted_baseline.new_tensor(0.0)
        n_fft = max(int(self.spectrum_n_fft), predicted_baseline.shape[-1])
        max_bin = int(self.spectrum_low_hz * n_fft / max(self.sampling_rate, 1.0)) + 1
        max_bin = min(max(max_bin, 2), n_fft // 2 + 1)
        predicted_spec = torch.fft.rfft(predicted_baseline, n=n_fft, dim=-1)[..., :max_bin]
        target_spec = torch.fft.rfft(target_baseline, n=n_fft, dim=-1)[..., :max_bin]
        predicted_mag = torch.log1p(predicted_spec.abs())
        target_mag = torch.log1p(target_spec.abs())
        return F.smooth_l1_loss(predicted_mag, target_mag, reduction="none").mean(dim=-1, keepdim=True)

    def _aux_ramp(self):
        if not self.training or self.aux_warmup_steps <= 0:
            return 1.0
        step = float(self.training_step.item())
        return min(max(step / max(float(self.aux_warmup_steps), 1.0), 0.0), 1.0)

    def _coefficients(self, t):
        if t.dtype == torch.long:
            return torch.stack([self.alpha_bar[t], self.beta_bar[t]], dim=1)
        t = t.float().clamp(0.0, 1.0)
        return torch.stack([t, self.gaussian_scale * torch.sqrt(t.clamp_min(0.0))], dim=1)

    def _predict_components(self, xt, noisy, t):
        xt = self._clip_by_robust_scale(torch.nan_to_num(xt), self.state_clip)
        model_input = torch.cat([xt, noisy], dim=1)
        coefficients = self._coefficients(t)
        if self.shared_backbone:
            dual_velocity = self.dual_velocity_model(model_input, coefficients)
            ecg_velocity = dual_velocity[:, 0:1]
            gaussian_velocity = dual_velocity[:, 1:2]
        else:
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
        if self.final_endpoint_blend > 0:
            endpoint_t = torch.full((x.shape[0],), self.timesteps - 1, device=x.device, dtype=torch.long)
            endpoint_ecg_noise, _ = self._predict_components(noisy, noisy, endpoint_t)
            endpoint_ecg_noise = self._project_inference_ecg_noise(endpoint_ecg_noise)
            flow_baseline = noisy - current
            blended_baseline = flow_baseline + self.final_endpoint_blend * (endpoint_ecg_noise - flow_baseline)
            if self.final_lowpass_baseline:
                blended_baseline = self._project_inference_ecg_noise(blended_baseline)
            current = noisy - blended_baseline
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
        if self.training:
            self.training_step += 1
        aux_ramp = self._aux_ramp()

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
            loss = loss + aux_ramp * self.recon_weight * recon_loss
        else:
            clean_hat = None

        if self.highfreq_weight > 0:
            if clean_hat is None:
                clean_hat = xt - alpha_bar * predicted_ecg - beta_bar * predicted_gaussian
            clean_high = clean - self._multi_scale_lowpass(clean)
            clean_hat_high = clean_hat - self._multi_scale_lowpass(clean_hat)
            highfreq_loss = self._loss_elementwise(clean_hat_high, clean_high)
            loss = loss + aux_ramp * self.highfreq_weight * highfreq_loss

        if self.lowfreq_weight > 0:
            lowfreq_loss = 0.0
            for kernel_size in self.lowfreq_kernel_sizes:
                predicted_low = self._smooth_1d(predicted_ecg, kernel_size)
                target_low = self._smooth_1d(ecg_noise, kernel_size)
                lowfreq_loss = lowfreq_loss + self._loss_elementwise(predicted_low, target_low)
            loss = loss + aux_ramp * self.lowfreq_weight * (lowfreq_loss / max(len(self.lowfreq_kernel_sizes), 1))

        if self.baseline_spectrum_weight > 0:
            spectrum_loss = self._baseline_spectrum_loss(predicted_ecg, ecg_noise)
            loss = loss + aux_ramp * self.baseline_spectrum_weight * spectrum_loss

        if self.baseline_smooth_weight > 0 and predicted_ecg.shape[-1] > 2:
            predicted_curvature = predicted_ecg[..., 2:] - 2.0 * predicted_ecg[..., 1:-1] + predicted_ecg[..., :-2]
            target_curvature = ecg_noise[..., 2:] - 2.0 * ecg_noise[..., 1:-1] + ecg_noise[..., :-2]
            smooth_loss = self._loss_elementwise(predicted_curvature, target_curvature)
            loss = loss + aux_ramp * self.baseline_smooth_weight * F.pad(smooth_loss, (1, 1))

        if self.no_gaussian_weight > 0:
            xt_no_gaussian = clean + alpha_bar * ecg_noise
            predicted_ecg_ng, _ = self._predict_components(xt_no_gaussian, noisy, t)
            clean_hat_ng = xt_no_gaussian - alpha_bar * predicted_ecg_ng
            no_gaussian_loss = self._loss_elementwise(predicted_ecg_ng, ecg_noise)
            no_gaussian_recon = self._loss_elementwise(clean_hat_ng, clean)
            loss = loss + aux_ramp * self.no_gaussian_weight * (no_gaussian_loss + self.recon_weight * no_gaussian_recon)

        if self.endpoint_weight > 0:
            endpoint_t = torch.full((batch_size,), self.timesteps - 1, device=device, dtype=torch.long)
            predicted_endpoint_ecg, _ = self._predict_components(noisy, noisy, endpoint_t)
            endpoint_clean_hat = noisy - predicted_endpoint_ecg
            endpoint_loss = self._loss_elementwise(predicted_endpoint_ecg, ecg_noise)
            endpoint_recon = self._loss_elementwise(endpoint_clean_hat, clean)
            endpoint_high = self._loss_elementwise(
                endpoint_clean_hat - self._multi_scale_lowpass(endpoint_clean_hat),
                clean - self._multi_scale_lowpass(clean),
            )
            endpoint_low = self._loss_elementwise(
                self._multi_scale_lowpass(predicted_endpoint_ecg),
                self._multi_scale_lowpass(ecg_noise),
            )
            endpoint_spectrum = self._baseline_spectrum_loss(predicted_endpoint_ecg, ecg_noise)
            loss = loss + aux_ramp * self.endpoint_weight * (
                endpoint_loss
                + self.recon_weight * endpoint_recon
                + self.highfreq_weight * endpoint_high
                + self.lowfreq_weight * endpoint_low
                + self.baseline_spectrum_weight * endpoint_spectrum
            )

        if self.antithetic_weight > 0:
            xt_anti = clean + alpha_bar * ecg_noise - beta_bar * gaussian_noise
            predicted_ecg_anti, predicted_gaussian_anti = self._predict_components(xt_anti, noisy, t)
            anti_ecg = self._loss_elementwise(predicted_ecg_anti, ecg_noise)
            anti_gaussian = self._loss_elementwise(predicted_gaussian_anti, -gaussian_noise)
            anti_consistency = self._loss_elementwise(predicted_ecg_anti, predicted_ecg.detach())
            loss = loss + aux_ramp * self.antithetic_weight * (
                self.antithetic_ecg_weight * (anti_ecg + anti_consistency)
                + self.antithetic_gaussian_weight * anti_gaussian
            )

        if valid_mask is not None:
            valid_mask = valid_mask.to(device, dtype=clean.dtype)
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(1)
            return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        return loss.mean()


@register_model("eddm_flow_matching_mamba")
class EDDMFlowMatchingMambaDenoiser(EDDMFlowMatchingDenoiser):
    """EDDM-flow-matching variant with a gated Mamba bottleneck.

    Intended for the QTDB+NSTDB baseline-wander setting where the shared
    dual-output UNet keeps complexity below EDDM-1, while one lightweight
    Mamba block adds long-range sequence modeling inspired by MECG-E.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("shared_backbone", True)
        kwargs.setdefault("timesteps", 24)
        kwargs.setdefault("sampler", "heun")
        kwargs.setdefault("mamba_layers", 1)
        kwargs.setdefault("mamba_expand", 2)
        kwargs.setdefault("mamba_gate_init", 0.08)
        kwargs.setdefault("bottleneck_dilated_conv", True)
        kwargs.setdefault("bottleneck_dilated_gate_init", 0.12)
        kwargs.setdefault("coefficient_embedding", "rbf_kan")
        kwargs.setdefault("coefficient_basis_count", 16)
        kwargs.setdefault("antithetic_weight", 0.25)
        kwargs.setdefault("aux_warmup_steps", 1000)
        kwargs.setdefault("gaussian_inference_weight", 0.0)
        kwargs.setdefault("inference_lowpass_ecg", True)
        super().__init__(**kwargs)
