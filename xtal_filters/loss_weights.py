from __future__ import annotations

from typing import Any

import torch

from xtal_filters.interp import linear_interp1d

DYNAMIC_LOSS_WEIGHT_MODES = frozenset({"shifted_pred_max_decay"})


def compute_shifted_pred_max_weights(
    pred: torch.Tensor,
    y_shifted_raw: torch.Tensor,
    mask: torch.Tensor,
    spec: dict[str, Any],
) -> torch.Tensor:
    """
    На каждой частоте: уровень для веса = max(pred_dBm, target_shifted_dBm).
    Пик эталона: глобальный максимум target_shifted только по точкам с mask True.
    w = w_peak * 10^(-(y_peak - level) / slope_db), clamp [w_min, w_peak].
    По умолчанию slope_db=60 → вес 0.1 на 60 dB ниже пика shifted.
    """
    slope_db = float(spec.get("slope_db", 60.0))
    if slope_db <= 0:
        raise ValueError("slope_db must be positive")
    w_peak = float(spec.get("w_peak", 1.0))
    w_min = float(spec.get("w_min", 1e-8))
    dt, dev = pred.dtype, pred.device
    neg_inf = torch.tensor(-1e300, device=dev, dtype=dt)
    y_for_max = torch.where(mask, y_shifted_raw, neg_inf)
    y_peak = torch.max(y_for_max)
    level = torch.maximum(pred, y_shifted_raw)
    delta_db = y_peak - level
    ten = torch.tensor(10.0, device=dev, dtype=dt)
    w = w_peak * torch.pow(ten, -(delta_db / slope_db))
    return torch.clamp(w, min=w_min, max=w_peak)


def build_frequency_loss_weights(
    f_opt: torch.Tensor,
    f_target: torch.Tensor,
    y_target: torch.Tensor,
    spec: dict[str, Any] | None,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Per-frequency nonnegative weights on the optimization grid f_opt.
    spec None or {"mode": "uniform"} -> all ones.

    two_band: high weight between f_pass_min_hz and f_pass_max_hz.
    target_above_db: high weight where interpolated target dBm exceeds db_threshold
    (удобно для полосового фильтра без ручного указания границ полосы).
    target_level_decay: w = w_peak * 10^(-(y_peak - y)/slope_db), clamped to w_min;
    slope_db — на сколько dB ниже пика уменьшается вес в 10 раз (по умолчанию 20).
    shifted_pred_max_decay: веса считаются в forward (см. compute_shifted_pred_max_weights); здесь возвращаются единицы.
    """
    n = f_opt.shape[0]
    ones = torch.ones((n,), device=device, dtype=dtype)
    if not spec:
        return ones
    mode = spec.get("mode", "uniform")
    if mode == "uniform":
        return ones
    if mode in DYNAMIC_LOSS_WEIGHT_MODES:
        return ones

    w_pass = float(spec.get("w_pass", 1.0))
    w_stop = float(spec.get("w_stop", 1.0))
    if w_pass < 0 or w_stop < 0:
        raise ValueError("w_pass and w_stop must be nonnegative")

    if mode == "two_band":
        lo = float(spec["f_pass_min_hz"])
        hi = float(spec["f_pass_max_hz"])
        if lo > hi:
            raise ValueError("f_pass_min_hz must be <= f_pass_max_hz")
        in_band = (f_opt >= lo) & (f_opt <= hi)
        return torch.where(in_band, f_opt.new_tensor(w_pass), f_opt.new_tensor(w_stop)).to(dtype)

    if mode == "target_above_db":
        thr = float(spec["db_threshold"])
        y_on_grid = linear_interp1d(f_opt, f_target, y_target)
        in_pass = y_on_grid > thr
        return torch.where(in_pass, f_opt.new_tensor(w_pass), f_opt.new_tensor(w_stop)).to(dtype)

    if mode == "target_level_decay":
        # Вес 1 у пика target; на каждые slope_db ниже пика — вес ×0.1 (логарифмическая ось dBm).
        y_on_grid = linear_interp1d(f_opt, f_target, y_target)
        y_peak = torch.max(y_on_grid)
        delta_db = y_peak - y_on_grid
        slope_db = float(spec.get("slope_db", 20.0))
        if slope_db <= 0:
            raise ValueError("target_level_decay.slope_db must be positive")
        w_peak = float(spec.get("w_peak", 1.0))
        w_min = float(spec.get("w_min", 1e-8))
        w = w_peak * torch.pow(
            torch.tensor(10.0, device=device, dtype=dtype), -(delta_db / slope_db)
        )
        return torch.clamp(w, min=w_min, max=w_peak)

    raise ValueError(f"Unknown loss_weighting.mode: {mode}")
