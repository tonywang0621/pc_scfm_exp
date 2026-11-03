"""Function-preserving removal of known parameter redundancies (before training).

This is a parameterization audit, not a proof that every remaining neuron is
statistically indispensable. Optimizer trajectories need not be preserved.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..mambattention import ResidualDilatedConvModule


class CompactContext(ResidualDilatedConvModule):
    def forward(self, x):
        residual = x
        x = self.layernorm(x).transpose(1, 2)
        for block in self.blocks:
            x = x + block(x)
        return residual + self.out(x).transpose(1, 2)


class MeanFreeBiasConv1d(nn.Conv1d):
    """Keep C-G identifiable bias coordinates before G-group normalization."""
    def __init__(self, source, groups):
        super().__init__(source.in_channels, source.out_channels, source.kernel_size,
                         source.stride, source.padding, source.dilation, source.groups,
                         bias=False, padding_mode=source.padding_mode,
                         device=source.weight.device, dtype=source.weight.dtype)
        self.weight = source.weight
        self.bias_groups = groups
        width = source.out_channels // groups
        if width > 1:
            centered = source.bias.detach().reshape(groups, width)
            centered = centered - centered.mean(-1, keepdim=True)
            self.bias_coordinates = nn.Parameter(centered[:, :-1].clone())
        else:
            self.register_parameter("bias_coordinates", None)

    def forward(self, x):
        bias = None
        if self.bias_coordinates is not None:
            b = self.bias_coordinates
            bias = torch.cat((b, -b.sum(-1, keepdim=True)), -1).flatten()
        return self._conv_forward(x, self.weight, bias)


@torch.no_grad()
def compact_parameters(core):
    """Mutate a newly built core, recording each exact algebraic reduction."""
    before = sum(p.numel() for p in core.parameters())
    events = []
    mask = core.mask_decoder.mask_conv
    if isinstance(mask[0], nn.ConvTranspose2d) and isinstance(mask[1], nn.Conv2d):
        a, b = mask[0], mask[1]
        if b.kernel_size != (1, 1) or a.groups != 1 or b.groups != 1:
            raise ValueError("Unsupported mask decoder for exact folding")
        folded = nn.ConvTranspose2d(a.in_channels, b.out_channels, a.kernel_size,
                                   a.stride, a.padding, a.output_padding, bias=False,
                                   dilation=a.dilation, device=a.weight.device, dtype=a.weight.dtype)
        folded.weight.copy_(torch.einsum("iohw,jo->ijhw", a.weight, b.weight[..., 0, 0]))
        # Both biases are spatial constants removed by the following IN.
        if not isinstance(mask[2], nn.InstanceNorm2d) or mask[2].track_running_stats:
            raise ValueError("Mask folding requires input-statistics InstanceNorm")
        core.mask_decoder.mask_conv = nn.Sequential(folded, *list(mask.children())[2:])
        events.append("fold mask transpose/pointwise convolutions and remove IN-null bias")

    for block in core.tsc_blocks:
        for axis in ("time", "freq"):
            lstm, norm = getattr(block, axis + "_lstm"), getattr(block, axis + "_norm")
            if lstm.num_layers != 1:
                raise ValueError("v17 compaction expects single-layer bidirectional LSTMs")
            for suffix in ("", "_reverse"):
                w = getattr(lstm, "weight_ih_l0" + suffix)
                b = getattr(lstm, "bias_ih_l0" + suffix)
                if norm.elementwise_affine:
                    b.add_(w @ norm.bias)
                    w.mul_(norm.weight.unsqueeze(0))
                other = "bias_hh_l0" + suffix
                b.add_(getattr(lstm, other))
                fixed = torch.zeros_like(getattr(lstm, other))
                delattr(lstm, other)
                lstm.register_buffer(other, fixed)
            lstm._init_flat_weights()
            setattr(block, axis + "_norm", nn.LayerNorm(norm.normalized_shape, eps=norm.eps,
                                                       elementwise_affine=False))
        context = block.context
        if isinstance(context, ResidualDilatedConvModule):
            scale = context.res_scale.sigmoid()
            context.out.weight.mul_(scale)
            context.out.bias.mul_(scale)
            del context.res_scale
            context.__class__ = CompactContext
    events.append("fold input LayerNorm affine and paired LSTM biases; absorb context gain")

    # Remove a shared linear layer immediately followed by per-block linears.
    flow = getattr(core, "residual_flow", None)
    embedding = getattr(flow, "time_embedding", None)
    if isinstance(embedding, nn.Sequential) and isinstance(embedding[-1], nn.Linear):
        last = embedding[-1]
        for block in flow.modules():
            projection = getattr(block, "time_proj", None)
            if isinstance(projection, nn.Linear):
                weight = projection.weight.detach().clone()
                projection.weight.copy_(weight @ last.weight)
                projection.bias.add_(weight @ last.bias)
        embedding[-1] = nn.Identity()
        events.append("fold shared final time linear into per-block time projections")

    for block in list(core.modules()):
        if isinstance(getattr(block, "time_proj", None), nn.Linear) and hasattr(block, "norm1"):
            projection = block.time_proj
            if projection.bias is not None:
                block.norm1.bias.add_(projection.bias)
                projection.register_parameter("bias", None)
        skip = getattr(block, "skip", None)
        if isinstance(skip, nn.Conv1d) and skip.bias is not None and hasattr(block, "norm2"):
            block.norm2.bias.add_(skip.bias)
            skip.register_parameter("bias", None)
        if isinstance(block, nn.Sequential):
            children = list(block.children())
            for index, (a, b) in enumerate(zip(children, children[1:])):
                if isinstance(a, (nn.Conv2d, nn.ConvTranspose2d)) and isinstance(b, nn.InstanceNorm2d):
                    if not b.track_running_stats:
                        a.register_parameter("bias", None)
                if isinstance(a, nn.Conv1d) and isinstance(b, nn.GroupNorm) and a.bias is not None:
                    block[index] = MeanFreeBiasConv1d(a, b.num_groups)
                if (isinstance(a, nn.Conv1d) and isinstance(b, nn.Conv1d)
                        and b.kernel_size == (1,) and b.groups == 1 and a.bias is not None):
                    # Covers depthwise -> pointwise in residual context blocks.
                    b.bias.add_(b.weight[..., 0] @ a.bias)
                    a.register_parameter("bias", None)
        for conv_name, norm_name in (("conv1", "norm1"), ("conv2", "norm2")):
            conv, norm = getattr(block, conv_name, None), getattr(block, norm_name, None)
            if isinstance(conv, nn.Conv1d) and isinstance(norm, nn.GroupNorm) and conv.bias is not None:
                setattr(block, conv_name, MeanFreeBiasConv1d(conv, norm.num_groups))
    events.append("remove normalization-null bias coordinates and duplicate additive offsets")
    after = sum(p.numel() for p in core.parameters())
    return {"before": before, "after": after, "removed": before - after, "operations": events}
