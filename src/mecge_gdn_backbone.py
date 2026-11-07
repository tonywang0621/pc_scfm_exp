"""Official GDN-2 mixers inside the unchanged MECG-E time/frequency scaffold."""
import copy
import importlib
from pathlib import Path
import sys
import types

import torch
from torch import nn


def load_official_gdn(directory):
    """Load the token mixer without importing Lit-GPT's language-model stack."""
    source = Path(directory).resolve() / "lit_gpt"
    if not (source / "gdn2.py").is_file():
        raise FileNotFoundError(f"Official GDN-2 layer missing: {source / 'gdn2.py'}")
    name = "_mecge_official_gdn2"
    if name in sys.modules:
        if list(sys.modules[name].__path__) != [str(source)]:
            raise RuntimeError("Cannot mix two GDN-2 reference directories in one process")
    else:
        package = types.ModuleType(name)
        package.__path__ = [str(source)]
        sys.modules[name] = package
    return importlib.import_module(name + ".gdn2").GatedDeltaNet2


class GDNDirection(nn.Module):
    def __init__(self, norm, width, layer_idx, mixer_cls):
        super().__init__()
        self.norm = copy.deepcopy(norm)
        # MECG-E width=32; 4 x 32 gives an internal width of 128, matching
        # the original Mamba expansion factor 4 (not a parameter-count match).
        self.mixer = mixer_cls(
            hidden_size=width, num_heads=4, head_dim=32, expand_v=1,
            mode="chunk", use_short_conv=True, conv_size=4, conv_bias=False,
            allow_neg_eigval=False, norm_eps=1e-5, layer_idx=layer_idx,
        )

    def forward(self, x):
        # Match the original one-layer MambaBlock: pre-norm mixer + residual.
        y, _, _ = self.mixer(self.norm(x), use_cache=False)
        return y + x


class BidirectionalGDN2(nn.Module):
    def __init__(self, original, width, layer_idx, mixer_cls):
        super().__init__()
        self.forward_direction = GDNDirection(original.forward_blocks[0].norm, width, layer_idx, mixer_cls)
        self.backward_direction = GDNDirection(original.backward_blocks[0].norm, width, layer_idx + 1, mixer_cls)

    def forward(self, x):
        forward = self.forward_direction(x)
        backward = self.backward_direction(x.flip(1)).flip(1)
        return torch.cat((forward, backward), dim=-1)


def install_backbone(model, variant, directory, mixer_cls=None):
    if variant not in {"m0", "m1", "m2"}:
        raise ValueError(f"Unknown MECG-E backbone: {variant}")
    if variant == "m0":
        return model
    mixer_cls = mixer_cls or load_official_gdn(directory)
    # Additional initialization must not change the training loader's CPU RNG.
    # Official pipeline constructs the model on CPU, then moves it to CUDA.
    with torch.random.fork_rng(devices=[]):
        for index, block in enumerate(model.TSMamba):
            width = block.tlinear.out_channels
            block.time_mamba = BidirectionalGDN2(block.time_mamba, width, index * 4, mixer_cls)
            if variant == "m2":
                block.freq_mamba = BidirectionalGDN2(block.freq_mamba, width, index * 4 + 2, mixer_cls)
    return model
