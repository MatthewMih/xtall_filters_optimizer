#!/usr/bin/env python3
"""Пересборка GIF из каталога прогона оптимизации с узкой осью Y.

Использует params_frames/step_*.json (нужен params_snapshot_every > 0 при прогоне),
пересчитывает отклик через ACAnalysis и xtal_filters.viz.

Верх оси Y: max(идеальный target на сетке sweep) + margin; низ: по умолчанию −20 dBm.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

# пакет из корня репозитория
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from xtal_filters.config import load_json, validate_schema
from xtal_filters.dtypes import pick_device
from xtal_filters.engine import ACAnalysis
from xtal_filters.sweep import linear_freq_grid
from xtal_filters.viz import (
    render_frame_to_array,
    response_vertical_axis_label,
    save_final_plot,
    save_gif,
    y_lim_zoom_above_ideal,
)


def _sorted_snapshots(params_dir: Path) -> list[Path]:
    pat = re.compile(r"^step_(\d+)\.json$")

    def key(p: Path) -> int:
        m = pat.match(p.name)
        if not m:
            return -1
        return int(m.group(1))

    paths = [p for p in params_dir.iterdir() if p.is_file() and pat.match(p.name)]
    paths.sort(key=key)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON схемы (как при optimize)")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Каталог прогона (target.npz, params_frames/)",
    )
    ap.add_argument(
        "--out-gif",
        default=None,
        help="Куда сохранить GIF (по умолчанию: <run-dir>/optimization_yzoom.gif)",
    )
    ap.add_argument(
        "--y-bottom",
        type=float,
        default=-20.0,
        help="Нижний предел оси Y, dBm (по умолчанию −20)",
    )
    ap.add_argument(
        "--y-margin",
        type=float,
        default=3.0,
        help="Запас сверху от max(идеальный target), dB",
    )
    ap.add_argument("--duration-ms", type=int, default=120, help="Длительность кадра GIF")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--save-final",
        default=None,
        metavar="PATH",
        help="Опционально сохранить final.png с тем же Y (последний кадр)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    params_dir = run_dir / "params_frames"
    if not params_dir.is_dir():
        raise SystemExit(f"Нет каталога {params_dir} — задайте params_snapshot_every в optimization.")

    cfg = load_json(args.config)
    validate_schema(cfg)

    target_npz = run_dir / "target.npz"
    if not target_npz.is_file():
        raise SystemExit(f"Нет {target_npz}")

    device = pick_device(args.device)
    fd = torch.float64
    zname = cfg.get("sweep", {}).get("complex_dtype", "complex64")
    sw = cfg["sweep"]
    f_opt = linear_freq_grid(float(sw["f_min"]), float(sw["f_max"]), int(sw["num_points"]), device, dtype=fd)

    data = np.load(target_npz)
    f_t = torch.as_tensor(data["freqs_hz"], dtype=fd, device=device)
    y_t = torch.as_tensor(data["dbm"], dtype=fd, device=device)

    f_opt_np = f_opt.detach().cpu().numpy()
    f_t_np = f_t.detach().cpu().numpy()
    y_t_np = y_t.detach().cpu().numpy()
    y_lim = y_lim_zoom_above_ideal(f_opt_np, f_t_np, y_t_np, args.y_bottom, args.y_margin)
    plot_ylabel = response_vertical_axis_label(cfg)

    analysis = ACAnalysis(cfg, device=device, z_dtype_name=zname).to(device)

    frames: list[np.ndarray] = []
    paths = _sorted_snapshots(params_dir)
    if not paths:
        raise SystemExit(f"В {params_dir} нет step_*.json")

    for p in paths:
        snap = json.loads(p.read_text(encoding="utf-8"))
        analysis.registry.load_physical_values(snap)
        with torch.no_grad():
            pred = analysis(f_opt).detach().cpu().numpy()
        step = int(snap.get("step", 0))
        loss = float(snap.get("loss", 0.0))
        df_v = float(snap.get("delta_f_hz", 0.0))
        dy_v = float(snap.get("delta_y_db", 0.0))
        arr = render_frame_to_array(
            f_opt_np,
            pred,
            f_t_np,
            y_t_np,
            df_v,
            dy_v,
            step,
            loss,
            y_lim=y_lim,
            ylabel=plot_ylabel,
        )
        frames.append(arr)

    out_gif = Path(args.out_gif) if args.out_gif else (run_dir / "optimization_yzoom.gif")
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    save_gif(out_gif, frames, duration_ms=args.duration_ms)
    print("Wrote", out_gif, "ylim=", y_lim, "frames=", len(frames))

    if args.save_final:
        last = json.loads(paths[-1].read_text(encoding="utf-8"))
        analysis.registry.load_physical_values(last)
        with torch.no_grad():
            y_final = analysis(f_opt).detach().cpu().numpy()
        save_final_plot(
            args.save_final,
            f_opt_np,
            f_t_np,
            y_t_np,
            y_final,
            float(last.get("delta_f_hz", 0.0)),
            float(last.get("delta_y_db", 0.0)),
            y_lim=y_lim,
            ylabel=plot_ylabel,
        )
        print("Wrote", args.save_final)


if __name__ == "__main__":
    main()
