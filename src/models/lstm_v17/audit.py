"""Non-destructive, multi-update parameter participation audit."""
import copy

import torch


def audit_parameters(model, noisy, clean, updates=3, strict=True):
    # manual_seed seeds every visible CUDA device, including devices not holding
    # this model. Preserve all their streams as well as the CPU stream.
    devices = list(range(torch.cuda.device_count()))
    # Work on a copy; audit optimization must not change training initialization.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(17017)
        probe = copy.deepcopy(model).train()
        # Legacy AttrDict aliases __dict__ to itself; deepcopy loses that alias.
        for module in probe.modules():
            h = getattr(module, "h", None)
            if isinstance(h, dict) and hasattr(h, "__dict__"):
                h.__dict__ = h
        probe.set_epoch(1, 30)
        parameters = dict(probe.named_parameters())
        seen = {name: torch.zeros_like(p, dtype=torch.bool) for name, p in parameters.items()}
        connected = set()
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=0)
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            loss = probe.compute_loss((noisy, clean), noisy.device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Parameter audit encountered non-finite loss")
            loss.backward()
            for name, parameter in parameters.items():
                gradient = parameter.grad
                if gradient is not None:
                    connected.add(name)
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError(f"Non-finite audit gradient: {name}")
                    seen[name] |= gradient != 0
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 0.5)
            optimizer.step()
        rows = [{"name": name, "shape": list(p.shape), "parameters": p.numel(),
                 "connected": name in connected, "observed_nonzero_coordinates": int(seen[name].sum()),
                 "observed_nonzero_fraction": float(seen[name].float().mean())}
                for name, p in parameters.items()]
        dormant = [r["name"] for r in rows if r["observed_nonzero_coordinates"] == 0]
        result = {"parameters": sum(p.numel() for p in parameters.values()),
                  "trainable_parameters": sum(p.numel() for p in parameters.values() if p.requires_grad),
                  "updates": updates, "dormant_parameter_tensors": dormant, "passed": not dormant,
                  "compaction": getattr(model.core, "compaction", None), "per_parameter": rows,
                  "scope": "Finite probes test participation, not mathematical/statistical irredundancy."}
        if strict and dormant:
            raise RuntimeError(f"Inactive parameter tensors after {updates} audit updates: {dormant}")
        return result
