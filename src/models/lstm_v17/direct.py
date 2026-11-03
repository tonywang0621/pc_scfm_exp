"""Collapse the deployed one-Euler-step U-Net into a time-free direct control."""
import torch
from torch import nn
from torch.nn import functional as F


class DirectBlock(nn.Module):
    def forward(self, x, unused_time=None):
        residual = self.skip(x)
        x = self.dropout(F.silu(self.norm1(self.conv1(x))))
        return F.silu(self.norm2(self.conv2(x)) + residual)


class DirectUNet(nn.Module):
    def forward(self, condition):
        x = self.input(condition)
        skips = []
        for index, down in enumerate(self.downs):
            x, skip = down(x, None)
            skips.append(self.first_skip_dapp(skip) if index == 0 else skip)
        x = self.mid_block2(self.mid_attention(self.mid_block1(x)))
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, None)
        return self.output(x)


@torch.no_grad()
def collapse_zero_state_time(flow):
    """Exact inference mapping at state=0,t=0; optimizer parameterization differs."""
    if not hasattr(flow, "downs"):
        raise ValueError("The matched direct control currently supports the U-Net backbone")
    embedding = flow.time_embedding(flow.input.weight.new_zeros(1))
    for block in list(flow.modules()):
        if hasattr(block, "time_proj") and hasattr(block, "norm1"):
            block.norm1.bias.add_(block.time_proj(embedding)[0])
            del block.time_proj
            block.__class__ = DirectBlock
    old = flow.input
    conv = nn.Conv1d(old.in_channels-1, old.out_channels, old.kernel_size,
                     stride=old.stride, padding=old.padding, bias=old.bias is not None,
                     device=old.weight.device, dtype=old.weight.dtype)
    conv.weight.copy_(old.weight[:, 1:])
    if old.bias is not None:
        conv.bias.copy_(old.bias)
    flow.input = conv
    del flow.time_embedding
    flow.__class__ = DirectUNet
