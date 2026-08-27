import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


class _ClassicalECGFilter(nn.Module):
    requires_training = False

    def __init__(
        self,
        sampling_rate=250,
        cutoff_hz=0.5,
        filter_kind="fir",
        fir_order=56,
        fir_window="kaiser",
        kaiser_beta=8.6,
        fir_design="fixed_order",
        kaiser_factor=4.5,
        transition_width_hz=0.07,
        iir_order=1,
        iir_method="butterworth",
        zero_phase=True,
        allow_causal_fallback=False,
        short_signal_mode="error",
        post_lowpass_hz=None,
        **kwargs,
    ):
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.cutoff_hz = float(cutoff_hz)
        self.filter_kind = str(filter_kind)
        self.fir_order = int(fir_order)
        self.fir_window = str(fir_window)
        self.kaiser_beta = float(kaiser_beta)
        self.fir_design = str(fir_design)
        self.kaiser_factor = float(kaiser_factor)
        self.transition_width_hz = float(transition_width_hz)
        self.iir_order = int(iir_order)
        self.iir_method = str(iir_method).lower()
        self.zero_phase = bool(zero_phase)
        self.allow_causal_fallback = bool(allow_causal_fallback)
        self.short_signal_mode = str(short_signal_mode)
        self.post_lowpass_hz = None if post_lowpass_hz is None else float(post_lowpass_hz)
        self._dummy = nn.Parameter(torch.zeros(()))

        if self.filter_kind == "fir":
            self.b, self.a = self._design_fir(self.cutoff_hz, pass_zero=False)
            self.sos = None
        elif self.filter_kind == "iir":
            self.b, self.a, self.sos = self._design_iir(self.cutoff_hz, btype="highpass")
        else:
            raise ValueError(f"Unsupported filter_kind={filter_kind!r}.")

        self.post_b = self.post_a = self.post_sos = None
        if self.post_lowpass_hz is not None and self.post_lowpass_hz < self.sampling_rate / 2.0:
            if self.filter_kind == "fir":
                self.post_b, self.post_a = self._design_fir(self.post_lowpass_hz, pass_zero=True)
                self.post_sos = None
            else:
                self.post_b, self.post_a, self.post_sos = self._design_iir(self.post_lowpass_hz, btype="lowpass")

    def _design_fir(self, cutoff_hz, pass_zero):
        if self.fir_design == "kaiserord":
            nyq_rate = self.sampling_rate / 2.0
            width = self.transition_width_hz / nyq_rate
            ripple_db = (round(-20 * np.log10(0.001)) + 1) / self.kaiser_factor
            numtaps, beta = signal.kaiserord(ripple_db, width)
            window = ("kaiser", beta)
            self.fir_order = int(numtaps - 1)
        else:
            numtaps = self.fir_order + 1
            window = (self.fir_window, self.kaiser_beta) if self.fir_window == "kaiser" else self.fir_window
        b = signal.firwin(
            numtaps,
            cutoff_hz,
            pass_zero=pass_zero,
            fs=self.sampling_rate,
            window=window,
        ).astype(np.float32)
        return b, np.array([1.0], dtype=np.float32)

    def _design_iir(self, cutoff_hz, btype):
        kwargs = {
            "N": self.iir_order,
            "Wn": cutoff_hz,
            "btype": btype,
            "fs": self.sampling_rate,
            "output": "sos",
        }
        if self.iir_method == "butterworth":
            sos = signal.butter(**kwargs)
        elif self.iir_method == "chebyshev1":
            sos = signal.cheby1(rp=0.5, **kwargs)
        elif self.iir_method == "chebyshev2":
            sos = signal.cheby2(rs=40.0, **kwargs)
        elif self.iir_method == "elliptic":
            sos = signal.ellip(rp=0.5, rs=40.0, **kwargs)
        else:
            raise ValueError(f"Unsupported iir_method={self.iir_method!r}.")
        b, a = signal.sos2tf(sos)
        return b.astype(np.float32), a.astype(np.float32), sos.astype(np.float32)

    def _filter_1d_with(self, x, b, a, sos):
        padlen = 3 * (max(len(a), len(b)) - 1)
        original_length = x.shape[-1]
        if (
            self.zero_phase
            and original_length <= padlen
            and self.short_signal_mode in {"deepfilter_pad", "mirror_constant_pad"}
        ):
            diff = padlen - original_length + 1
            pad_tail = np.repeat(x[..., -1:], diff, axis=-1)
            x = np.concatenate([np.flip(x, axis=-1), x, pad_tail], axis=-1)
            slice_start = original_length
            slice_end = slice_start + original_length
        else:
            slice_start = None
            slice_end = None

        if sos is not None:
            if self.zero_phase and x.shape[-1] > padlen:
                filtered = signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)
                return filtered[..., slice_start:slice_end] if slice_start is not None else filtered
            if self.zero_phase and not self.allow_causal_fallback:
                raise ValueError(
                    f"zero_phase=True requires windows longer than padlen={padlen}, "
                    f"got length {x.shape[-1]}. Increase this baseline's paper-specific "
                    "dataset.window_size, filter the full record before windowing, or explicitly "
                    "set model.allow_causal_fallback=True to accept a non-paper-faithful causal filter."
                )
            return signal.sosfilt(sos, x, axis=-1).astype(np.float32)

        if self.zero_phase and x.shape[-1] > padlen:
            filtered = signal.filtfilt(b, a, x, axis=-1).astype(np.float32)
            return filtered[..., slice_start:slice_end] if slice_start is not None else filtered
        if self.zero_phase and not self.allow_causal_fallback:
            raise ValueError(
                f"zero_phase=True requires windows longer than padlen={padlen}, "
                f"got length {x.shape[-1]}. Increase this baseline's paper-specific "
                "dataset.window_size, filter the full record before windowing, or explicitly "
                "set model.allow_causal_fallback=True to accept a non-paper-faithful causal filter."
            )
        return signal.lfilter(b, a, x, axis=-1).astype(np.float32)

    def _filter_1d(self, x):
        x = self._filter_1d_with(x, self.b, self.a, self.sos)
        if self.post_b is not None:
            x = self._filter_1d_with(x, self.post_b, self.post_a, self.post_sos)
        return x

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected ECG shaped [B, T] or [B, 1, T], got {tuple(x.shape)}.")

        device = x.device
        dtype = x.dtype
        filtered = self._filter_1d(x.detach().cpu().numpy())
        output = torch.from_numpy(filtered).to(device=device, dtype=dtype)
        return output.squeeze(1) if squeeze else output

    @torch.no_grad()
    def denoising(self, x):
        return self.forward(x)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        pred = self.forward(noisy)
        loss = F.mse_loss(pred, clean)
        return loss.detach() + self._dummy * 0.0


@register_model("fir_filter")
class FIRFilterDenoiser(_ClassicalECGFilter):
    def __init__(self, **kwargs):
        kwargs.setdefault("filter_kind", "fir")
        super().__init__(**kwargs)


@register_model("iir_filter")
class IIRFilterDenoiser(_ClassicalECGFilter):
    def __init__(self, **kwargs):
        kwargs.setdefault("filter_kind", "iir")
        super().__init__(**kwargs)
