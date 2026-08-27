import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


@register_model("drnn")
class DRNNDenoiser(nn.Module):
    """Deep recurrent denoising network following Antczak's DRNN layout.

    Verified against Antczak (2018), "Deep Recurrent Neural Networks for ECG
    Signal Denoising":
      - architecture (Fig. 2 / Section 3.1 "best network overall"): one
        unidirectional 64-unit LSTM layer, then two ReLU dense layers of 64
        units each, then one linear output layer; input and output width 1.
        No dropout / no L2 (Section 4 lists these only as future work), no
        residual connection, no bidirectionality (Section 2.1: "input signal
        to LSTM ... applied one sample at a time").
      - loss: MSE (Section 3.1). Optimiser: Adam, batch 64 (Section 3.1).

    NOT reproduced -- this is a DRNN *architecture-only* baseline, the option
    explicitly allowed in notes/experiment_實驗設計.txt:
      - the paper's synthetic dynamical-model (McSharry ECGSYN) pretraining
        followed by real-data fine-tuning (Sections 2.2 / 3.2), which is the
        paper's main contribution.

    Task-defined differences, shared by every baseline here: training data is
    PTB-XL Lead II (paper: PTB aVL lead + synthetic), the noise is baseline
    wander (paper: additive white noise, SNR-based), windows are 512 samples
    (paper: 600), normalisation is endpoint-centering (paper: zero-mean), at
    360 Hz (paper: 512 Hz synthetic / native PTB).
    """

    def __init__(
        self,
        input_size=1,
        lstm_hidden_sizes=(64,),
        hidden_size=None,
        lstm_layers=None,
        dense_layers=(64, 64),
        dropout=0.0,
        residual=False,
        output_size=1,
        **kwargs,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.input_size = int(input_size)
        self.output_size = int(output_size)

        if lstm_hidden_sizes is None:
            if hidden_size is None:
                hidden_size = 64
            if lstm_layers is None:
                lstm_layers = 1
            lstm_hidden_sizes = [int(hidden_size)] * int(lstm_layers)
        elif isinstance(lstm_hidden_sizes, int):
            lstm_hidden_sizes = [int(lstm_hidden_sizes)]
        else:
            lstm_hidden_sizes = [int(width) for width in lstm_hidden_sizes]
        if not lstm_hidden_sizes:
            raise ValueError("lstm_hidden_sizes must contain at least one LSTM layer.")

        recurrent_layers = []
        in_features = self.input_size
        for hidden_width in lstm_hidden_sizes:
            recurrent_layers.append(
                nn.LSTM(
                    input_size=in_features,
                    hidden_size=hidden_width,
                    num_layers=1,
                    batch_first=True,
                )
            )
            in_features = hidden_width
        self.recurrent_layers = nn.ModuleList(recurrent_layers)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        layers = []
        for width in dense_layers:
            layers.append(nn.Linear(in_features, int(width)))
            layers.append(nn.ReLU())
            in_features = int(width)
        layers.append(nn.Linear(in_features, self.output_size))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 3 or x.shape[1] != self.input_size:
            raise ValueError(f"Expected ECG shaped [B, T] or [B, 1, T], got {tuple(x.shape)}.")

        sequence = x.transpose(1, 2)
        features = sequence
        for index, layer in enumerate(self.recurrent_layers):
            features, _ = layer(features)
            if index != len(self.recurrent_layers) - 1:
                features = self.dropout(features)
        restored = self.head(features).transpose(1, 2)
        if self.residual:
            restored = x + restored
        return restored.squeeze(1) if squeeze else restored

    @torch.no_grad()
    def denoising(self, x):
        return self.forward(x)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        valid_mask = batch[2].to(device) if len(batch) > 2 else None
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)

        pred = self.forward(noisy)
        loss = F.mse_loss(pred, clean, reduction="none")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=loss.dtype)
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        else:
            loss = loss.mean()
        return loss
