from __future__ import annotations

import torch

from xtal_filters.circuit import CircuitTopology, build_node_incidence
from xtal_filters.elements import branch_impedance, resolve_params


def _node_index_map(nodes: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(nodes)}


def assemble_mna(
    top: CircuitTopology,
    phys: dict[str, torch.Tensor],
    f_hz: torch.Tensor,
    z_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build MNA system A x = b for each frequency in f_hz.
    Unknowns: [I_branch_0 .. I_branch_{B-1}, V_node_0 .. V_node_{N-1}] (same order as notebook).
    Returns A, b with shape (n_freq, n_eq, n_eq) and (n_freq, n_eq).
    """
    elems = top.elements
    nodes_list = top.nodes
    num_elems = len(elems)
    num_nodes = len(nodes_list)
    elem_names = [e.name for e in elems]
    elem2idx = {name: i for i, name in enumerate(elem_names)}
    node2idx = _node_index_map(nodes_list)
    n_eq = num_elems + num_nodes

    device = f_hz.device
    omega = 2 * torch.pi * f_hz.to(torch.float64)
    n_freq = f_hz.shape[0]

    incidence = build_node_incidence(top)

    A = torch.zeros((n_freq, n_eq, n_eq), dtype=z_dtype, device=device)
    b = torch.zeros((n_freq, n_eq), dtype=z_dtype, device=device)

    # KCL: one row per node except GND
    row = 0
    for node_name in nodes_list:
        if node_name == top.gnd_name:
            continue
        e1, e2 = incidence[node_name]
        for en in e1:
            A[:, row, elem2idx[en]] = -1
        for en in e2:
            A[:, row, elem2idx[en]] = 1
        row += 1

    # Branch equations
    for elem_idx, el in enumerate(elems):
        irow = num_nodes - 1 + elem_idx
        resolved = resolve_params(el.params, phys)
        z = branch_impedance(el.etype, resolved, omega, z_dtype)
        n1, n2 = el.node1, el.node2
        if el.etype == "VoltageSource":
            A[:, irow, num_elems + node2idx[n1]] = -1
            A[:, irow, num_elems + node2idx[n2]] = 1
            b[:, irow] = resolved["E"].to(z_dtype)
        else:
            A[:, irow, elem_idx] = -z
            A[:, irow, num_elems + node2idx[n1]] = 1
            A[:, irow, num_elems + node2idx[n2]] = -1

    # GND: last row
    last = n_eq - 1
    A[:, last, num_elems + node2idx[top.gnd_name]] = 1
    b[:, last] = 0

    return A, b


def solve_mna_batch(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A: (n_freq, n, n), b: (n_freq, n) -> x: (n_freq, n)"""
    return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
