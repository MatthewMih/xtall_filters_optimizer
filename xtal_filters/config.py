from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(d: dict[str, Any]) -> None:
    for key in ("parameters", "nodes", "elements", "sweep"):
        if key not in d:
            raise KeyError(f"Missing key: {key}")
    if "GND" not in d["nodes"]:
        raise ValueError('nodes must include "GND"')
    if "load_element" not in d:
        raise KeyError("load_element: name of resistor used as load probe")
    if "voltage_source" not in d:
        raise KeyError("voltage_source: element name of the voltage source")
    for k in ("f_min", "f_max", "num_points"):
        if k not in d["sweep"]:
            raise KeyError(f"sweep.{k} required")
