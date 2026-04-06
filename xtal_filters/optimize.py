from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from xtal_filters.dtypes import pick_device
from xtal_filters.engine import ACAnalysis
from xtal_filters.interp import shifted_target_raw, shifted_target_values
from xtal_filters.loss import masked_weighted_response_loss
from xtal_filters.loss_weights import (
    build_frequency_loss_weights,
    compute_shifted_pred_max_weights,
)
from xtal_filters.sweep import linear_freq_grid


@dataclass
class OptimizationConfig:
    device: str = "cpu"
    complex_dtype: str = "complex64"
    optimizer: str = "adam"
    lr: float = 1e-2
    num_steps: int = 200
    log_every: int = 10
    gif_every: int = 20
    loss_type: str = "l1"
    lambda_y_shift: float = 0.01
    enable_delta_f: bool = True
    enable_delta_y: bool = True
    adam_then_lbfgs: bool = False
    lbfgs_steps: int = 20
    lbfgs_lr: float = 0.1
    seed: int | None = 42
    loss_weighting: dict[str, Any] | None = None
    params_snapshot_every: int = 0


def _load_opt_section(cfg: dict[str, Any]) -> OptimizationConfig:
    o = cfg.get("optimization", {})
    return OptimizationConfig(
        device=o.get("device", "cpu"),
        complex_dtype=o.get("complex_dtype", "complex64"),
        optimizer=o.get("optimizer", "adam"),
        lr=float(o.get("lr", 1e-2)),
        num_steps=int(o.get("num_steps", 200)),
        log_every=int(o.get("log_every", 10)),
        gif_every=int(o.get("gif_every", 20)),
        loss_type=o.get("loss_type", "l1"),
        lambda_y_shift=float(o.get("lambda_y_shift", 0.01)),
        enable_delta_f=bool(o.get("enable_delta_f", True)),
        enable_delta_y=bool(o.get("enable_delta_y", True)),
        adam_then_lbfgs=bool(o.get("adam_then_lbfgs", False)),
        lbfgs_steps=int(o.get("lbfgs_steps", 20)),
        lbfgs_lr=float(o.get("lbfgs_lr", 0.1)),
        seed=o.get("seed"),
        loss_weighting=o.get("loss_weighting"),
        params_snapshot_every=int(o.get("params_snapshot_every", 0)),
    )


class OptimizationProblem(nn.Module):
    def __init__(
        self,
        analysis: ACAnalysis,
        f_target: torch.Tensor,
        y_target: torch.Tensor,
        f_opt: torch.Tensor,
        opt: OptimizationConfig,
    ):
        super().__init__()
        self.analysis = analysis
        self.register_buffer("f_target", f_target)
        self.register_buffer("y_target", y_target)
        self.register_buffer("f_opt", f_opt)
        self.cfg = opt
        self.delta_f = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.delta_y = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        if not opt.enable_delta_f:
            self.delta_f.requires_grad_(False)
        if not opt.enable_delta_y:
            self.delta_y.requires_grad_(False)

        lw = opt.loss_weighting
        self._dynamic_shifted_pred_weights = bool(
            lw and lw.get("mode") == "shifted_pred_max_decay"
        )
        fw = build_frequency_loss_weights(
            f_opt, f_target, y_target, opt.loss_weighting, f_opt.device, dtype=torch.float64
        )
        self.register_buffer("freq_loss_weight", fw)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.analysis(self.f_opt)
        df = self.delta_f.squeeze() if self.cfg.enable_delta_f else self.f_opt.new_zeros(())
        dy = self.delta_y.squeeze() if self.cfg.enable_delta_y else self.f_opt.new_zeros(())
        tgt, mask = shifted_target_values(self.f_opt, self.f_target, self.y_target, df, dy, pred)
        if self._dynamic_shifted_pred_weights and self.cfg.loss_weighting:
            y_raw = shifted_target_raw(self.f_opt, self.f_target, self.y_target, df, dy)
            fw = compute_shifted_pred_max_weights(
                pred, y_raw, mask, self.cfg.loss_weighting
            )
        else:
            fw = self.freq_loss_weight
        loss_resp = masked_weighted_response_loss(pred, tgt, mask, self.cfg.loss_type, fw)
        pen_y = dy.abs() if self.cfg.enable_delta_y else pred.new_tensor(0.0)
        loss = loss_resp + self.cfg.lambda_y_shift * pen_y
        return loss, loss_resp, pred, mask


def run_optimization(
    circuit_cfg: dict[str, Any],
    target_path: str | Path,
    out_dir: str | Path,
    opt_override: OptimizationConfig | None = None,
) -> dict[str, Any]:
    from tqdm import tqdm

    from xtal_filters.viz import render_frame_to_array, response_vertical_axis_label, save_final_plot, save_gif

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(target_path), out_dir / "target.npz")

    opt_cfg = _load_opt_section(circuit_cfg) if opt_override is None else opt_override
    if opt_cfg.seed is not None:
        torch.manual_seed(opt_cfg.seed)
        np.random.seed(opt_cfg.seed)

    device = pick_device(opt_cfg.device)
    zname = circuit_cfg.get("sweep", {}).get("complex_dtype", opt_cfg.complex_dtype)

    data = np.load(target_path)
    f_t = torch.as_tensor(data["freqs_hz"], dtype=torch.float64, device=device)
    y_t = torch.as_tensor(data["dbm"], dtype=torch.float64, device=device)

    sw = circuit_cfg["sweep"]
    f_opt = linear_freq_grid(float(sw["f_min"]), float(sw["f_max"]), int(sw["num_points"]), device)
    plot_ylabel = response_vertical_axis_label(circuit_cfg)

    analysis = ACAnalysis(circuit_cfg, device=device, z_dtype_name=zname).to(device)
    with torch.no_grad():
        initial_response = analysis(f_opt).clone()

    prob = OptimizationProblem(analysis, f_t, y_t, f_opt, opt_cfg).to(device)

    params: list[nn.Parameter] = list(prob.analysis.registry.parameters())
    if opt_cfg.enable_delta_f:
        params.append(prob.delta_f)
    if opt_cfg.enable_delta_y:
        params.append(prob.delta_y)

    log_rows: list[dict[str, Any]] = []
    gif_frames: list[Any] = []

    snap_every = opt_cfg.params_snapshot_every
    params_dir = out_dir / "params_frames"
    if snap_every > 0:
        params_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(params, lr=opt_cfg.lr)

    for it in tqdm(range(opt_cfg.num_steps), desc="optimize", unit="step"):
        optimizer.zero_grad()
        loss, loss_resp, pred, mask = prob()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_log = prob.analysis(f_opt)
            lv = float(loss.cpu())
            df_v = float(prob.delta_f.cpu()) if opt_cfg.enable_delta_f else 0.0
            dy_v = float(prob.delta_y.cpu()) if opt_cfg.enable_delta_y else 0.0

        if it % opt_cfg.log_every == 0 or it == opt_cfg.num_steps - 1:
            log_rows.append(
                {
                    "step": it,
                    "loss": lv,
                    "loss_resp": float(loss_resp.detach().cpu()),
                    "delta_f_hz": df_v,
                    "delta_y_db": dy_v,
                }
            )

        if snap_every > 0 and (it % snap_every == 0 or it == opt_cfg.num_steps - 1):
            snap = {k: float(v.detach().cpu()) for k, v in prob.analysis.registry.physical_dict().items()}
            snap["step"] = it
            snap["loss"] = lv
            snap["delta_f_hz"] = df_v
            snap["delta_y_db"] = dy_v
            (params_dir / f"step_{it:06d}.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")

        if it % opt_cfg.gif_every == 0 or it == opt_cfg.num_steps - 1:
            arr = render_frame_to_array(
                f_opt.detach().cpu().numpy(),
                pred_log.detach().cpu().numpy(),
                f_t.detach().cpu().numpy(),
                y_t.detach().cpu().numpy(),
                df_v,
                dy_v,
                it,
                lv,
                ylabel=plot_ylabel,
            )
            gif_frames.append(arr)

    if opt_cfg.adam_then_lbfgs:
        lbfgs = torch.optim.LBFGS(params, lr=opt_cfg.lbfgs_lr, max_iter=opt_cfg.lbfgs_steps)

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            l, _, _, _ = prob()
            l.backward()
            return l

        lbfgs.step(closure)

    with torch.no_grad():
        final_response = analysis(f_opt).clone()

    phys = {k: float(v.detach().cpu()) for k, v in prob.analysis.registry.physical_dict().items()}
    trainable = {k: float(v.detach().cpu()) for k, v in prob.analysis.registry.trainable_physical().items()}

    (out_dir / "parameters_final.json").write_text(json.dumps(phys, indent=2), encoding="utf-8")
    (out_dir / "trainable.json").write_text(json.dumps(trainable, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "responses.npz",
        freqs_hz=f_opt.cpu().numpy(),
        initial=initial_response.cpu().numpy(),
        final=final_response.cpu().numpy(),
    )
    (out_dir / "optimization_log.json").write_text(json.dumps(log_rows, indent=2), encoding="utf-8")

    df_final = float(prob.delta_f.detach().cpu()) if opt_cfg.enable_delta_f else 0.0
    dy_final = float(prob.delta_y.detach().cpu()) if opt_cfg.enable_delta_y else 0.0
    save_final_plot(
        out_dir / "final.png",
        f_opt.cpu().numpy(),
        f_t.cpu().numpy(),
        y_t.cpu().numpy(),
        final_response.cpu().numpy(),
        df_final,
        dy_final,
        ylabel=plot_ylabel,
    )
    if gif_frames:
        save_gif(out_dir / "optimization.gif", gif_frames, duration_ms=120)

    return {"output_dir": str(out_dir), "final_loss": log_rows[-1]["loss"] if log_rows else None}
