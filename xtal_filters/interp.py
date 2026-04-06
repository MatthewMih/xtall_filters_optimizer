from __future__ import annotations

import torch


def linear_interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Differentiable linear interpolation. x shape (n,); xp, fp 1D ascending, same length.
    """
    if xp.ndim != 1 or fp.ndim != 1 or xp.shape[0] != fp.shape[0]:
        raise ValueError("xp and fp must be 1D same length")
    xc = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, xc, right=False) - 1
    idx = idx.clamp(0, xp.shape[0] - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    t = (xc - x0) / (x1 - x0 + 1e-300)
    return (1 - t) * y0 + t * y1


def shifted_target_values(
    f_eval: torch.Tensor,
    f_target: torch.Tensor,
    y_target: torch.Tensor,
    delta_f: torch.Tensor,
    delta_y: torch.Tensor,
    pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    y_shifted(f) = interp(y_target, f - delta_f) + delta_y where defined;
    outside [f_target[0], f_target[-1]] after shift, match pred (detached) so loss mask is clean.
    """
    fq = f_eval - delta_f
    valid = (fq >= f_target[0]) & (fq <= f_target[-1])
    y = linear_interp1d(fq, f_target, y_target) + delta_y
    tgt = torch.where(valid, y, pred.detach())
    return tgt, valid


def shifted_target_raw(
    f_eval: torch.Tensor,
    f_target: torch.Tensor,
    y_target: torch.Tensor,
    delta_f: torch.Tensor,
    delta_y: torch.Tensor,
) -> torch.Tensor:
    """
    target_shifted(f) = interp(y_target, f - delta_f) + delta_y на всей сетке (без подстановки pred вне полосы).
    Для весов и пика shifted; clamp интерполяции как в linear_interp1d.
    """
    fq = f_eval - delta_f
    return linear_interp1d(fq, f_target, y_target) + delta_y
