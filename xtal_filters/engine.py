from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from xtal_filters.circuit import CircuitTopology, build_topology
from xtal_filters.dtypes import complex_dtype
from xtal_filters.mna import assemble_mna, solve_mna_batch
from xtal_filters.parameters import ParameterRegistry
from xtal_filters.response import response_dbm_curve


class ACAnalysis(nn.Module):
    """Differentiable AC sweep: build MNA from JSON topology + parameter registry."""

    def __init__(self, cfg: dict[str, Any], device: torch.device, z_dtype_name: str = "complex64"):
        super().__init__()
        self.cfg = cfg
        self.top = build_topology(cfg)
        self.registry = ParameterRegistry.from_config_list(cfg["parameters"])
        self.device = device
        self.z_dtype = complex_dtype(z_dtype_name)
        self.elem_names = [e.name for e in self.top.elements]
        self.elem2idx = {n: i for i, n in enumerate(self.elem_names)}
        self.node2idx = {n: i for i, n in enumerate(self.top.nodes)}
        self.num_elems = len(self.elem_names)

        vs = cfg["voltage_source"]
        ld = cfg["load_element"]
        names = set(self.elem_names)
        if vs not in names:
            raise ValueError(f"voltage_source {vs} not in elements")
        if ld not in names:
            raise ValueError(f"load_element {ld} not in elements")

        resp = cfg.get("response") or {}
        self._relative_to_input_power = bool(resp.get("relative_to_input_power", False))
        self._input_series_resistor = resp.get("input_series_resistor")
        if self._relative_to_input_power:
            if not self._input_series_resistor:
                raise ValueError(
                    "response.relative_to_input_power true requires response.input_series_resistor (имя Resistor — внутреннее сопротивление генератора для P_avail=E²/(4R))"
                )
            if self._input_series_resistor not in names:
                raise ValueError(f"input_series_resistor {self._input_series_resistor!r} not in elements")
            ser_el = next(e for e in self.top.elements if e.name == self._input_series_resistor)
            if ser_el.etype != "Resistor":
                raise ValueError("response.input_series_resistor must be a Resistor element")

    def forward(self, f_hz: torch.Tensor) -> torch.Tensor:
        """Returns dBm curve (n_freq,) float64 on same device as f_hz.

        При ``response.relative_to_input_power``: значение = dBm(нагрузка) − dBm(P_avail),
        где P_avail = E²/(4R) — доступная мощность генератора (Thevenin E, внутреннее R ветви
        ``input_series_resistor``) при согласовании R_load = R.
        """
        f_hz = f_hz.to(self.device)
        phys = self.registry.physical_dict()
        phys = {k: v.to(self.device) for k, v in phys.items()}
        A, b = assemble_mna(self.top, phys, f_hz, self.z_dtype)
        x = solve_mna_batch(A, b)
        dbm = response_dbm_curve(
            x,
            self.top,
            self.elem2idx,
            self.node2idx,
            self.num_elems,
            phys,
            relative_to_input_power=self._relative_to_input_power,
            input_series_resistor=self._input_series_resistor,
        )
        return dbm.real.to(torch.float64)
