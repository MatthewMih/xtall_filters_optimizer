from __future__ import annotations

import torch


def linear_freq_grid(
    f_min: float,
    f_max: float,
    num_points: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    return torch.linspace(f_min, f_max, num_points, device=device, dtype=dtype)
