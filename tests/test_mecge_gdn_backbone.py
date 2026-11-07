"""CPU adapter checks; real official kernels are checked by mecge_gdn_smoke.py."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from mecge_gdn_backbone import BidirectionalGDN2, install_backbone


@unittest.skipUnless(HAS_TORCH, "PyTorch required for adapter checks")
class BackboneTests(unittest.TestCase):
    def fixture(self):
        class Mixer(nn.Module):
            def __init__(self, hidden_size, **kwargs):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(hidden_size))

            def forward(self, x, **kwargs):
                return x.cumsum(1) * self.scale, None, None

        class Original(nn.Module):
            def __init__(self):
                super().__init__()
                self.forward_blocks = [types.SimpleNamespace(norm=nn.Identity())]
                self.backward_blocks = [types.SimpleNamespace(norm=nn.Identity())]

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.time_mamba, self.freq_mamba = Original(), Original()
                self.tlinear, self.flinear = nn.Conv1d(8, 4, 1), nn.Conv1d(8, 4, 1)

        model = nn.Module()
        model.TSMamba = nn.ModuleList([Block() for _ in range(4)])
        model.encoder = nn.Linear(4, 4)
        return model, Mixer

    def test_direction_reversal_residual_and_gradient(self):
        model, mixer = self.fixture()
        layer = BidirectionalGDN2(model.TSMamba[0].time_mamba, 4, 0, mixer)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_()
        y = layer(x)
        expected = torch.cat((x + x.cumsum(1), x + x.flip(1).cumsum(1).flip(1)), -1)
        torch.testing.assert_close(y, expected)
        self.assertEqual(tuple(y.shape), (2, 3, 8))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_only_selected_modules_change_and_rng_is_preserved(self):
        for variant in ("m0", "m1", "m2"):
            model, mixer = self.fixture()
            encoder = model.encoder
            old = [(b.time_mamba, b.freq_mamba, b.tlinear, b.flinear) for b in model.TSMamba]
            state = torch.random.get_rng_state().clone()
            self.assertIs(install_backbone(model, variant, "/unused", mixer), model)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
            self.assertIs(model.encoder, encoder)
            for block, (time, freq, tlinear, flinear) in zip(model.TSMamba, old):
                self.assertEqual(block.time_mamba is time, variant == "m0")
                self.assertEqual(block.freq_mamba is freq, variant != "m2")
                self.assertIs(block.tlinear, tlinear)
                self.assertIs(block.flinear, flinear)


if __name__ == "__main__":
    unittest.main()
