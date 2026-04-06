from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xtal_filters.elements import expected_params


@dataclass
class ElementEntry:
    name: str
    etype: str
    node1: str
    node2: str
    params: dict[str, Any]


@dataclass
class CircuitTopology:
    nodes: list[str]
    elements: list[ElementEntry]
    voltage_source_name: str
    load_element_name: str
    gnd_name: str


def build_topology(cfg: dict[str, Any]) -> CircuitTopology:
    nodes = list(cfg["nodes"])
    gnd = "GND"
    if gnd not in nodes:
        raise ValueError("nodes must contain GND")
    els: list[ElementEntry] = []
    for e in cfg["elements"]:
        et = e["type"]
        params = e.get("params", {})
        expected = expected_params(et)
        if set(params.keys()) != expected:
            raise ValueError(f"Element {e['name']}: params must be exactly {expected}, got {set(params.keys())}")
        els.append(
            ElementEntry(
                name=e["name"],
                etype=et,
                node1=e["node1"],
                node2=e["node2"],
                params=params,
            )
        )
    return CircuitTopology(
        nodes=nodes,
        elements=els,
        voltage_source_name=cfg["voltage_source"],
        load_element_name=cfg["load_element"],
        gnd_name=gnd,
    )


def build_node_incidence(top: CircuitTopology) -> dict[str, tuple[list[str], list[str]]]:
    """For each node: (elements where node is node1, elements where node is node2)."""
    nmap: dict[str, tuple[list[str], list[str]]] = {n: ([], []) for n in top.nodes}
    for el in top.elements:
        nmap[el.node1][0].append(el.name)
        nmap[el.node2][1].append(el.name)
    return nmap
