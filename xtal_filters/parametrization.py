from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

Kind = Literal["resistance", "capacitance", "inductance", "generic"]


def raw_to_positive(raw: torch.Tensor, kind: Kind, eps: float = 1e-30) -> torch.Tensor:
    """Map unconstrained raw to strictly positive physical values (SI)."""
    y = F.softplus(raw)
    if kind == "capacitance":
        return y + eps
    if kind == "inductance":
        return y + eps
    if kind == "resistance":
        return y + eps
    return raw


def initialize_raw(value: float, kind: Kind) -> float:
    """Inverse-ish mapping: pick raw so softplus(raw) ≈ value for positive kinds."""
    if kind in ("resistance", "capacitance", "inductance"):
        v = max(value, 1e-30)
        # softplus(x) = log(1+exp(x)); for large x, x ≈ log(2*exp(x))... simpler: use log(exp(v)-1) for v>0
        if v > 20:
            return v  # softplus(v) ≈ v
        return math.log(math.expm1(v))
    return float(value)


def clamp_to_bounds(x: torch.Tensor, min_v: float | None, max_v: float | None) -> torch.Tensor:
    if min_v is None and max_v is None:
        return x
    lo = -math.inf if min_v is None else min_v
    hi = math.inf if max_v is None else max_v
    return torch.clamp(x, lo, hi)
