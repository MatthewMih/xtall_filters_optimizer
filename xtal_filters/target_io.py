from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from xtal_filters.config import load_json, validate_schema
from xtal_filters.dtypes import pick_device
from xtal_filters.engine import ACAnalysis
from xtal_filters.sweep import linear_freq_grid
from xtal_filters.viz import plot_response, response_vertical_axis_label


def generate_target_artifacts(circuit_json: str | Path, out_dir: str | Path, device: str = "cpu") -> Path:
    """Build circuit from JSON, sweep, save freqs_hz + dbm + plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_json(circuit_json)
    validate_schema(cfg)
    dev = pick_device(device)
    zname = cfg.get("sweep", {}).get("complex_dtype", "complex64")
    sw = cfg["sweep"]
    f_hz = linear_freq_grid(float(sw["f_min"]), float(sw["f_max"]), int(sw["num_points"]), dev)
    model = ACAnalysis(cfg, device=dev, z_dtype_name=zname).to(dev)
    with torch.no_grad():
        dbm = model(f_hz).cpu().numpy()
    f_np = f_hz.cpu().numpy()
    target_path = out_dir / "target.npz"
    np.savez(target_path, freqs_hz=f_np, dbm=dbm)
    resp = cfg.get("response") or {}
    meta = {"circuit": str(circuit_json), "sweep": sw, "response": resp}
    (out_dir / "target_meta.json").write_text(
        __import__("json").dumps(meta, indent=2),
        encoding="utf-8",
    )
    if resp.get("relative_to_input_power"):
        plot_title = "Target: dBm(load) − dBm(E²/(4R)) — относительно согласованной мощности генератора"
    else:
        plot_title = "Target dBm (load)"
    plot_response(
        out_dir / "target_plot.png",
        f_np,
        dbm,
        title=plot_title,
        ylabel=response_vertical_axis_label(cfg),
    )
    return target_path
