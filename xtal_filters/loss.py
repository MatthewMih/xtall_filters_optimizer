from __future__ import annotations

import torch


def masked_response_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """
    pred, target, mask: (n_freq,), mask bool
    """
    d = pred - target
    w = mask.to(dtype=pred.dtype)
    denom = torch.clamp(w.sum(), min=1.0)
    if loss_type == "l1":
        return (d.abs() * w).sum() / denom
    if loss_type == "l2":
        return ((d ** 2) * w).sum() / denom
    raise ValueError(loss_type)


def masked_weighted_response_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
    freq_weight: torch.Tensor,
) -> torch.Tensor:
    """Same as masked_response_loss but each frequency point scaled by freq_weight >= 0."""
    d = pred - target
    m = mask.to(dtype=pred.dtype)
    fw = freq_weight.to(dtype=pred.dtype)
    w = m * fw
    denom = torch.clamp(w.sum(), min=1e-30)
    if loss_type == "l1":
        return (d.abs() * w).sum() / denom
    if loss_type == "l2":
        return ((d ** 2) * w).sum() / denom
    raise ValueError(loss_type)
