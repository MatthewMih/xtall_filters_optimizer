from __future__ import annotations

import torch

from xtal_filters.circuit import CircuitTopology
from xtal_filters.elements import resolve_params


def extract_branch_current_voltage(
    x: torch.Tensor,
    top: CircuitTopology,
    elem2idx: dict[str, int],
    node2idx: dict[str, int],
    load_name: str,
    num_elems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return phasor I and V across load resistor (V at node1 - V at node2)."""
    el = next(e for e in top.elements if e.name == load_name)
    if el.etype != "Resistor":
        raise ValueError("load_element must be a Resistor")
    i_idx = elem2idx[load_name]
    I = x[:, i_idx]
    n1, n2 = el.node1, el.node2
    v1 = x[:, num_elems + node2idx[n1]]
    v2 = x[:, num_elems + node2idx[n2]]
    V = v1 - v2
    return I, V


def avg_power_watts(I: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """P = 0.5 * Re(V * conj(I)) — same real form as the reference notebook."""
    return 0.5 * (V.real * I.real + V.imag * I.imag)


def power_to_dbm(P_w: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """dBm = 10 * log10(P / 1 mW)."""
    return 10.0 * torch.log10(torch.clamp(P_w, min=eps) / 1e-3)


def matched_thevenin_available_power_watts(
    top: CircuitTopology,
    phys: dict[str, torch.Tensor],
    voltage_source_name: str,
    series_resistor_name: str,
) -> torch.Tensor:
    """
    Максимальная средняя мощность от Thevenin (E, R) при согласовании R_load = R.

    Параметр E в VoltageSource — пиковая амплитуда фазора (как в MNA и в
    avg_power_watts = 0.5·Re(V·I*)). Тогда P_avail = E²/(8R). Формула E²/(4R)
    соответствовала бы RMS-амплитуде E_rms = E_peak/√2.
    """
    vs = next(e for e in top.elements if e.name == voltage_source_name)
    rs = next(e for e in top.elements if e.name == series_resistor_name)
    if vs.etype != "VoltageSource":
        raise ValueError("voltage_source must be a VoltageSource element")
    if rs.etype != "Resistor":
        raise ValueError("input_series_resistor must be a Resistor element")
    E = resolve_params(vs.params, phys)["E"]
    R = resolve_params(rs.params, phys)["R"]
    return (E * E) / (8.0 * torch.clamp(R, min=1e-300))


def response_dbm_curve(
    x: torch.Tensor,
    top: CircuitTopology,
    elem2idx: dict[str, int],
    node2idx: dict[str, int],
    num_elems: int,
    phys: dict[str, torch.Tensor],
    *,
    relative_to_input_power: bool = False,
    input_series_resistor: str | None = None,
) -> torch.Tensor:
    I, V = extract_branch_current_voltage(x, top, elem2idx, node2idx, top.load_element_name, num_elems)
    P_load = avg_power_watts(I, V)
    dbm_load = power_to_dbm(P_load)
    if not relative_to_input_power:
        return dbm_load
    if not input_series_resistor:
        raise ValueError("input_series_resistor required when relative_to_input_power is true")
    P_avail = matched_thevenin_available_power_watts(
        top, phys, top.voltage_source_name, input_series_resistor
    )
    dbm_ref = power_to_dbm(P_avail)
    return dbm_load - dbm_ref
