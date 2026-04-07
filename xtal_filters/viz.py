from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def response_vertical_axis_label(cfg: dict[str, Any]) -> str:
    """Подпись оси Y (абсолютный dBm на нагрузке или относительно P_avail=E²/(8R), E — пик)."""
    r = cfg.get("response") or {}
    if r.get("relative_to_input_power"):
        return "dB (load − matched gen.)"
    return "dBm (load)"


def shifted_target_curve_np(f_hz: np.ndarray, f_tgt_hz: np.ndarray, y_tgt: np.ndarray, delta_f_hz: float, delta_y_db: float) -> np.ndarray:
    fq = f_hz - delta_f_hz
    return np.interp(fq, f_tgt_hz, y_tgt) + delta_y_db


def _target_on_freq_grid(f_hz: np.ndarray, f_tgt_hz: np.ndarray, y_tgt: np.ndarray) -> np.ndarray:
    """Target dBm на той же оси частот, что и pred (важно для совпадения линий на графике)."""
    return np.interp(f_hz, f_tgt_hz, y_tgt)


def ideal_target_dbm_on_grid(f_hz: np.ndarray, f_tgt_hz: np.ndarray, y_tgt_dbm: np.ndarray) -> np.ndarray:
    """Идеальный target (как на графике), интерполированный на сетку частот оптимизации."""
    return _target_on_freq_grid(f_hz, f_tgt_hz, y_tgt_dbm)


def y_lim_zoom_above_ideal(
    f_hz: np.ndarray,
    f_tgt_hz: np.ndarray,
    y_tgt_dbm: np.ndarray,
    y_bottom_dbm: float = -20.0,
    margin_db: float = 3.0,
) -> tuple[float, float]:
    """Ось Y: низ фиксированный, верх = max(ideal target на сетке) + margin (для set_ylim(bottom, top))."""
    y_ideal = ideal_target_dbm_on_grid(f_hz, f_tgt_hz, y_tgt_dbm)
    top = float(np.max(y_ideal) + margin_db)
    return (y_bottom_dbm, top)


def render_frame_to_array(
    f_hz: np.ndarray,
    pred_dbm: np.ndarray,
    f_tgt_hz: np.ndarray,
    y_tgt_dbm: np.ndarray,
    delta_f_hz: float,
    delta_y_db: float,
    step: int,
    loss: float,
    dpi: int = 100,
    y_lim: tuple[float, float] | None = None,
    ylabel: str = "dBm",
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    x_mhz = f_hz / 1e6
    y_tgt_on_f = _target_on_freq_grid(f_hz, f_tgt_hz, y_tgt_dbm)

    # target (ideal) — зелёный; сдвинутый target — оранжевый; current — синий
    ax.plot(
        x_mhz,
        y_tgt_on_f,
        linestyle=(0, (4, 2)),
        color="#2ca02c",
        linewidth=2.4,
        label="target (ideal)",
        zorder=2,
        alpha=0.95,
    )
    y_shifted = shifted_target_curve_np(f_hz, f_tgt_hz, y_tgt_dbm, delta_f_hz, delta_y_db)
    ax.plot(
        x_mhz,
        y_shifted,
        linestyle=(0, (8, 3, 2, 3)),
        color="#ff7f0e",
        linewidth=2.0,
        label="target shifted",
        zorder=3,
        alpha=0.95,
    )
    ax.plot(x_mhz, pred_dbm, "-", color="#1f77b4", linewidth=1.6, label="current", zorder=4)

    ax.set_xlabel("f (MHz)")
    ax.set_ylabel(ylabel)
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=8, loc="best", framealpha=0.92)
    ax.set_title(f"step={step}  loss={loss:.4f}  Δf={delta_f_hz:.3g} Hz  Δy={delta_y_db:.4g} dB", fontsize=9)
    fig.tight_layout()
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi)
    plt.close(fig)
    bio.seek(0)
    im = Image.open(bio).convert("RGB")
    return np.array(im)


def save_final_plot(
    path: Path | str,
    f_hz: np.ndarray,
    f_tgt_hz: np.ndarray,
    y_tgt: np.ndarray,
    y_final: np.ndarray,
    delta_f_hz: float,
    delta_y_db: float,
    y_initial: np.ndarray | None = None,
    y_lim: tuple[float, float] | None = None,
    ylabel: str = "dBm (load)",
) -> None:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    x_mhz = f_hz / 1e6
    y_tgt_on_f = _target_on_freq_grid(f_hz, f_tgt_hz, y_tgt)
    if y_initial is not None:
        ax.plot(
            x_mhz,
            y_initial,
            "--",
            color="#7f7f7f",
            linewidth=1.5,
            label="initial (pre-opt)",
            zorder=1,
            alpha=0.9,
        )
    ax.plot(
        x_mhz,
        y_tgt_on_f,
        linestyle=(0, (4, 2)),
        color="#2ca02c",
        linewidth=2.2,
        label="target (ideal)",
        zorder=2,
        alpha=0.95,
    )
    y_s = shifted_target_curve_np(f_hz, f_tgt_hz, y_tgt, delta_f_hz, delta_y_db)
    ax.plot(
        x_mhz,
        y_s,
        linestyle=(0, (8, 3, 2, 3)),
        color="#ff7f0e",
        linewidth=2.0,
        label="target shifted",
        zorder=3,
        alpha=0.95,
    )
    ax.plot(x_mhz, y_final, "-", color="#1f77b4", linewidth=1.8, label="final", zorder=4)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(ylabel)
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_gif(path: Path | str, frames: list[np.ndarray], duration_ms: int = 100) -> None:
    path = Path(path)
    if not frames:
        return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)


def plot_response(path: Path | str, f_hz: np.ndarray, dbm: np.ndarray, title: str = "", ylabel: str = "dBm") -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(f_hz / 1e6, dbm, "-", color="C0")
    ax.set_xlabel("f (MHz)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
