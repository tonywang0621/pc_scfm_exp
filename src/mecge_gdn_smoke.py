"""GPU acceptance check for the actual official M0/M1/M2 models (no training run)."""
import argparse
import json
from pathlib import Path
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mecge-dir", type=Path, default=project / "references/MECG-E")
    parser.add_argument("--gdn-dir", type=Path, default=project / "references/GatedDeltaNet-2")
    args = parser.parse_args()
    import torch
    import yaml
    from mecge_gdn_backbone import install_backbone, BidirectionalGDN2

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Official Mamba/GDN-2 kernel smoke test requires CUDA")
    sys.path.insert(0, str(args.mecge_dir.resolve() / "mamba"))
    sys.path.insert(0, str(args.mecge_dir.resolve()))
    from models.MECGE import MECGE
    config = yaml.safe_load((args.mecge_dir / "config/MECGE_phase.yaml").read_text())
    torch.manual_seed(3407)
    original = MECGE(config)
    baseline = {k: v.clone() for k, v in original.state_dict().items()}
    assert install_backbone(original, "m0", args.gdn_dir) is original
    del original
    for variant in ("m0", "m1", "m2"):
        torch.manual_seed(3407)
        model = install_backbone(MECGE(config), variant, args.gdn_dir)
        # All retained parameters must be exactly the original initialization.
        for name, value in model.state_dict().items():
            if name in baseline:
                torch.testing.assert_close(value, baseline[name], rtol=0, atol=0)
        expected = {"m0": 0, "m1": 4, "m2": 8}[variant]
        assert sum(isinstance(m, BidirectionalGDN2) for m in model.modules()) == expected
        model.to(args.device).train()
        clean = torch.randn(2, 1, 512, device=args.device)
        t = torch.arange(512, device=args.device) / 360
        noisy = clean + 0.3 * torch.sin(2 * torch.pi * 0.3 * t)
        loss = model(clean, noisy)
        assert loss.ndim == 0 and torch.isfinite(loss), (variant, loss)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), variant
        if variant != "m0":
            for name, parameter in model.named_parameters():
                if ".mixer." in name:
                    assert parameter.grad is not None, (variant, name)
        torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.8, 0.99)).step()
        model.eval()
        with torch.no_grad():
            prediction = model.denoising(noisy)
        assert prediction.shape == clean.shape and torch.isfinite(prediction).all(), variant
        print(json.dumps(dict(variant=variant, loss=loss.item(), shape=list(prediction.shape),
                              parameters=sum(p.numel() for p in model.parameters()), status="passed")), flush=True)
        del model, loss, prediction, gradients
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
