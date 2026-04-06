from __future__ import annotations

from typing import Any

import torch


def z_resistor(R: torch.Tensor, omega: torch.Tensor, z_dtype: torch.dtype) -> torch.Tensor:
    return R.to(z_dtype) + 0j


def z_capacitor(C: torch.Tensor, omega: torch.Tensor, z_dtype: torch.dtype) -> torch.Tensor:
    # Z = 1/(j omega C)
    o = omega.to(torch.float64)
    c = C.to(torch.float64)
    inv_z_imag = o * c  # 1/Xc magnitude for imag part: Z = -j/(omega C)
    z_imag = -1.0 / torch.clamp(inv_z_imag, min=1e-300)
    return z_imag.to(z_dtype) * 1j


def z_inductor(L: torch.Tensor, omega: torch.Tensor, z_dtype: torch.dtype) -> torch.Tensor:
    o = omega.to(torch.float64)
    l = L.to(torch.float64)
    return (o * l).to(z_dtype) * 1j


def z_parallel(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    return (z1 * z2) / (z1 + z2 + eps)


def z_crystal_bvd(
    Rm: torch.Tensor,
    Lm: torch.Tensor,
    Cm: torch.Tensor,
    Cp: torch.Tensor,
    omega: torch.Tensor,
    z_dtype: torch.dtype,
) -> torch.Tensor:
    """Butterworth–Van Dyke: (Rm+jωLm+1/jωCm) || 1/jωCp."""
    zm = z_resistor(Rm, omega, z_dtype) + z_inductor(Lm, omega, z_dtype) + z_capacitor(Cm, omega, z_dtype)
    zcp = z_capacitor(Cp, omega, z_dtype)
    return z_parallel(zm, zcp)


def z_crystal_lcc(
    L: torch.Tensor,
    C1: torch.Tensor,
    C2: torch.Tensor,
    omega: torch.Tensor,
    z_dtype: torch.dtype,
) -> torch.Tensor:
    """Notebook model: (jωL + 1/jωC1) || 1/jωC2."""
    zs = z_inductor(L, omega, z_dtype) + z_capacitor(C1, omega, z_dtype)
    z2 = z_capacitor(C2, omega, z_dtype)
    return z_parallel(zs, z2)


def z_impedance(R: torch.Tensor, X: torch.Tensor, omega: torch.Tensor, z_dtype: torch.dtype) -> torch.Tensor:
    return R.to(z_dtype) + 1j * X.to(z_dtype)


def branch_impedance(
    etype: str,
    resolved: dict[str, torch.Tensor],
    omega: torch.Tensor,
    z_dtype: torch.dtype,
) -> torch.Tensor:
    if etype == "Resistor":
        return z_resistor(resolved["R"], omega, z_dtype)
    if etype == "Capacitor":
        return z_capacitor(resolved["C"], omega, z_dtype)
    if etype == "Inductor":
        return z_inductor(resolved["L"], omega, z_dtype)
    if etype == "Impedance":
        return z_impedance(resolved["R"], resolved["X"], omega, z_dtype)
    if etype == "Crystal":
        return z_crystal_bvd(
            resolved["Rm"], resolved["Lm"], resolved["Cm"], resolved["Cp"], omega, z_dtype
        )
    if etype == "CrystalLCC":
        return z_crystal_lcc(resolved["L"], resolved["C1"], resolved["C2"], omega, z_dtype)
    if etype == "VoltageSource":
        return torch.zeros(omega.shape, dtype=z_dtype, device=omega.device)
    raise ValueError(f"Unknown element type: {etype}")


def resolve_params(params_template: dict[str, Any], phys: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in params_template.items():
        if isinstance(v, str):
            out[k] = phys[v]
        else:
            out[k] = torch.as_tensor(float(v), dtype=torch.float64)
    return out


def expected_params(etype: str) -> frozenset[str]:
    m = {
        "Resistor": frozenset({"R"}),
        "Capacitor": frozenset({"C"}),
        "Inductor": frozenset({"L"}),
        "Impedance": frozenset({"R", "X"}),
        "Crystal": frozenset({"Rm", "Lm", "Cm", "Cp"}),
        "CrystalLCC": frozenset({"L", "C1", "C2"}),
        "VoltageSource": frozenset({"E"}),
    }
    if etype not in m:
        raise ValueError(etype)
    return m[etype]
