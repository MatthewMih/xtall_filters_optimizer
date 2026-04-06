from __future__ import annotations

import torch


def complex_dtype(name: str) -> torch.dtype:
    if name == "complex64":
        return torch.complex64
    if name == "complex128":
        return torch.complex128
    raise ValueError(f"Unknown complex dtype: {name}")


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    raise ValueError(f"Unknown device: {name}")


def float_dtype_for_device(device: torch.device) -> torch.dtype:
    """MPS does not support float64; use float32 there."""
    if device.type == "mps":
        return torch.float32
    return torch.float64
