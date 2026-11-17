"""Launch constraints and numerical gate for experimental shared-prefix training."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def configure_prefix_environment() -> None:
    required = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    for name, expected in required.items():
        if os.environ.get(name, expected) != expected:
            raise ValueError(f"shared_prefix_training requires {name}={expected}")
    os.environ.update(required)
    for name, value in {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }.items():
        os.environ.setdefault(name, value)


def _values(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().double().reshape(-1).cpu()
    if isinstance(value, dict):
        items = [_values(value[key]) for key in sorted(value)]
    elif isinstance(value, (list, tuple)):
        items = [_values(item) for item in value]
    elif isinstance(value, (float, int)):
        return torch.tensor([value], dtype=torch.float64)
    else:
        raise ValueError(f"Unsupported prefix preflight value: {type(value).__name__}")
    return torch.cat(items) if items else torch.empty(0, dtype=torch.float64)


def check_forward_parity(native: object, shared: object) -> dict[str, float]:
    try:
        if not isinstance(native, dict) or not isinstance(shared, dict):
            raise ValueError("Expected native per-datum output dictionaries")
        a = _values(native.get("per_datum", []))
        b = _values(shared.get("per_datum", []))
    except (ValueError, TypeError):
        a = b = torch.empty(0, dtype=torch.float64)
    valid = bool(
        a.numel()
        and a.shape == b.shape
        and torch.isfinite(a).all()
        and torch.isfinite(b).all()
        and torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    )
    # Coordinate failure before any rank enters backward. A single local raise
    # would leave successful peers blocked in their next training collective.
    if dist.is_initialized():
        device = "cuda" if dist.get_backend() == "nccl" else "cpu"
        accepted = torch.tensor(int(valid), device=device)
        dist.all_reduce(accepted, op=dist.ReduceOp.MIN)
        valid = bool(accepted.item())
    if not valid:
        raise ValueError(
            "Native/shared forward values failed the numerical preflight on a trainer rank"
        )
    delta = b - a
    return {
        "values": a.numel(),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / a.norm().clamp_min(1e-30)),
    }
