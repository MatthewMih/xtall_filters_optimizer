from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from xtal_filters.parametrization import Kind, initialize_raw, raw_to_positive


@dataclass
class ParamSpec:
    name: str
    value: float
    trainable: bool
    kind: Kind
    min: float | None = None
    max: float | None = None


class ParameterRegistry(nn.Module):
    """Named physical parameters; trainable ones stored as raw nn.Parameters."""

    def __init__(self, params: list[ParamSpec], param_dtype: torch.dtype = torch.float64):
        super().__init__()
        self._specs: dict[str, ParamSpec] = {p.name: p for p in params}
        self._buffers: dict[str, torch.Tensor] = {}
        self._trainable_names: list[str] = []

        for p in params:
            if p.trainable:
                raw0 = initialize_raw(p.value, p.kind)
                self.register_parameter(
                    f"_raw_{p.name}", nn.Parameter(torch.tensor(raw0, dtype=param_dtype))
                )
                self._trainable_names.append(p.name)
            else:
                t = torch.tensor(float(p.value), dtype=param_dtype)
                self.register_buffer(f"_fixed_{p.name}", t)

    def physical_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, spec in self._specs.items():
            if spec.trainable:
                raw = getattr(self, f"_raw_{name}")
                x = raw_to_positive(raw, spec.kind)
                if spec.min is not None or spec.max is not None:
                    lo = spec.min if spec.min is not None else -1e300
                    hi = spec.max if spec.max is not None else 1e300
                    x = torch.clamp(x, lo, hi)
                out[name] = x
            else:
                buf = getattr(self, f"_fixed_{name}")
                out[name] = buf
        return out

    def trainable_physical(self) -> dict[str, torch.Tensor]:
        d = self.physical_dict()
        return {k: d[k] for k in self._trainable_names}

    def load_physical_values(self, values: dict[str, Any]) -> None:
        """Set internal state from physical SI values (e.g. params snapshot JSON). Unknown keys ignored."""
        for name, spec in self._specs.items():
            if name not in values:
                continue
            val = float(values[name])
            if spec.trainable:
                raw0 = initialize_raw(val, spec.kind)
                getattr(self, f"_raw_{name}").data.fill_(raw0)
            else:
                getattr(self, f"_fixed_{name}").fill_(val)

    @staticmethod
    def from_config_list(
        raw_list: list[dict[str, Any]], param_dtype: torch.dtype = torch.float64
    ) -> "ParameterRegistry":
        specs: list[ParamSpec] = []
        for d in raw_list:
            kind = d.get("kind", "generic")
            if kind not in ("resistance", "capacitance", "inductance", "generic"):
                kind = "generic"
            specs.append(
                ParamSpec(
                    name=d["name"],
                    value=float(d["value"]),
                    trainable=bool(d.get("trainable", False)),
                    kind=kind,  # type: ignore[arg-type]
                    min=float(d["min"]) if d.get("min") is not None else None,
                    max=float(d["max"]) if d.get("max") is not None else None,
                )
            )
        return ParameterRegistry(specs, param_dtype=param_dtype)
