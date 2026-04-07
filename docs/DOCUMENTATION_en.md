# Technical documentation (English)

[Russian version](DOCUMENTATION.md)

Intro for readers: [README.md](../README.md) (English) · [README_ru.md](../README_ru.md) (Russian)

Repository: [github.com/MatthewMih/crystal-rf-filter-optimizer](https://github.com/MatthewMih/crystal-rf-filter-optimizer)

---

# `xtal_filters` reference (JSON, CLI, API)

Research-oriented **PyTorch** framework for **linear AC circuits** described in **JSON**, comparing magnitude responses in **dBm at the load** and **differentiable optimization** of parameters (Adam, optional LBFGS) with plots and GIF export.

## Features

- Circuit definition: `GND` node, …, branches between `node1` and `node2`, shared named parameters (one name → one scalar, reusable across elements).
- Elements: `Resistor`, `Capacitor`, `Inductor`, `Impedance` (R + jX), `VoltageSource`, `Crystal` (**BVD** model: Rm, Lm, Cm, Cp), `CrystalLCC` (parallel L–C1 and C2 branches).
- Solver: **MNA** (as in the reference notebook), `torch.linalg.solve`, batched over frequency, `complex64` / `complex128`.
- Compute device: **`cpu`**, **`cuda`**.
- **`target` mode:** compute reference curve and save `target.npz`.
- **`optimize` mode:** fit trainable parameters to target, learnable `delta_f` / `delta_y` shifts for target, L1/L2 in dBm, penalty on `|delta_y|`, linear target interpolation, mask outside target frequency span.

## Setup

From the repository root:

```bash
python3 -m pip install -e .
```

or dependencies only:

```bash
python3 -m pip install -r requirements.txt
```

Requires: Python ≥ 3.10, PyTorch 2.x, NumPy, Matplotlib, Pillow.

Run modules from the project root (or with the package installed):

```bash
cd /path/to/crystal-rf-filter-optimizer
python3 -m xtal_filters --help
```

## Package layout `xtal_filters/`

| Module | Role |
|--------|------|
| `config.py` | JSON load, basic validation |
| `parameters.py`, `parametrization.py` | Parameter registry, softplus + clamp for positive R/L/C |
| `elements.py` | Branch impedances ω → Z(ω) |
| `circuit.py` | Topology from JSON |
| `mna.py` | MNA matrix and RHS assembly |
| `engine.py` | `ACAnalysis`: circuit + sweep → dBm curve |
| `response.py` | Power on load resistor, dBm conversion |
| `interp.py` | Shifted target and mask |
| `loss.py` | L1/L2 loss in dBm |
| `optimize.py` | Optimization loop and artifacts |
| `viz.py` | PNG/GIF; optional fixed Y (`y_lim`), helpers for “zoom” to ideal target |
| `loss_weights.py` | Per-frequency loss weights (`two_band`, `target_above_db`, `target_level_decay`, `shifted_pred_max_decay`) |
| `target_io.py` | Target generation to disk |
| `cli.py` | Command-line interface |

Helper script (not part of the package): [`scripts/rebuild_optimization_gif_yzoom.py`](../scripts/rebuild_optimization_gif_yzoom.py) — rebuild GIF from `params_frames/` with a narrow Y axis.

Example in [`examples/`](../examples/): **`ladder_ideal.json`** (reference, \(R_m=0\)) and **`ladder_optimize.json`** (lossy circuit + `optimization` section).

## JSON circuit format

### Top-level required fields

| Field | Description |
|------|-------------|
| `nodes` | Node names; must include **`GND`**. |
| `voltage_source` | Name of the `VoltageSource` element (EMF in volts). |
| `load_element` | Name of a **`Resistor`** — power and dBm are computed for this branch. |
| `parameters` | Parameter list (below). |
| `elements` | Branch list (below). |
| `sweep` | Frequency sweep (below). |

### `parameters`

Each object:

- `name` — unique name.
- `value` — initial SI value (Ω, F, H, V for `E`, etc.).
- `trainable` — `true` / `false` (only trainable entries are optimized).
- `kind` — `resistance` \| `capacitance` \| `inductance` \| `generic` (`generic` is not forced through softplus for sign).
- optional `min`, `max` — after softplus mapping the value is **clamped** to this interval.

The same `name` in `params` across elements refers to **one** scalar (shared crystals, Rs/Rl, etc.).

### `elements`

Each object:

- `type` — one of the types in the table below.
- `name` — unique branch name.
- `node1`, `node2` — nodes; **positive current** in equations flows from `node1` to `node2`.
- `params` — map of type field names → **string** parameter name from `parameters` **or** a numeric constant.

`params` keys are fixed per type:

| `type` | `params` |
|--------|----------|
| `Resistor` | `R` |
| `Capacitor` | `C` |
| `Inductor` | `L` |
| `Impedance` | `R`, `X` |
| `VoltageSource` | `E` (V) |
| `Crystal` | `Rm`, `Lm`, `Cm`, `Cp` (BVD: series Rm–L–Cm, parallel Cp) |
| `CrystalLCC` | `L`, `C1`, `C2` |

### `sweep`

| Field | Description |
|------|-------------|
| `f_min`, `f_max` | Frequency bounds, Hz |
| `num_points` | Number of points (linear grid, ≥ 2) |
| `complex_dtype` | optional: `complex64` (default) or `complex128` |

### `response` (optional)

Defines units for the **load** (`load_element`) curve everywhere: `ACAnalysis`, `target`, `optimize`, yzoom script.

| Field | Description |
|------|-------------|
| `relative_to_input_power` | `false` (default) — **dBm** at load. `true` — **dB difference**: dBm(load) − dBm(\(P_\mathrm{avail}\)), with \(P_\mathrm{avail} = E^2/(8R)\): maximum **average** power at match if **\(E\)** is the **peak** phasor amplitude in `VoltageSource` (consistent with \(P_\mathrm{load} = \frac{1}{2}\mathrm{Re}(VI^*)\) on the load). For **RMS** amplitude it would be \(E_\mathrm{rms}^2/(4R)\). Same as \(10\log_{10}(P_\mathrm{load}/P_\mathrm{avail})\). |
| `input_series_resistor` | Branch name of the **`Resistor`** whose \(R\) appears in \(E^2/(8R)\) (same as in the netlist, e.g. `Rs` / `Rport`). Required if `relative_to_input_power` is `true`. \(E\) comes from `voltage_source`. |

```json
"response": {
  "relative_to_input_power": true,
  "input_series_resistor": "Rs"
}
```

Reference `target.npz` and the optimization JSON must use the **same** `response` section or comparisons are meaningless.

### `optimization` (matching mode only)

Same JSON as the “filter #2” netlist. Main fields:

| Field | Description |
|------|-------------|
| `device` | `cpu` \| `cuda` |
| `lr` | Initial Adam learning rate |
| `lr_schedule` | optional: `"cosine"` — cosine LR schedule (`torch.optim.lr_scheduler.CosineAnnealingLR`) from `lr` to `lr_min` over `num_steps` |
| `lr_min` | LR floor for `lr_schedule: "cosine"` (default `0`) |
| `num_steps` | Adam steps |
| `log_every` | Log every k-th step |
| `gif_every` | GIF frame interval |
| `params_snapshot_every` | Save `params_frames/step_*.json` every N steps (0 = off) |
| `loss_type` | `l1` or `l2` (difference in dBm on the grid) |
| `lambda_y_shift` | Penalty weight `λ·|delta_y|` (dB) |
| `enable_delta_f`, `enable_delta_y` | Learnable target shifts in frequency (Hz) and dBm |
| `adam_then_lbfgs` | Run LBFGS after Adam (`true`/`false`) |
| `lbfgs_steps`, `lbfgs_lr` | LBFGS settings |
| `seed` | RNG seed (or `null`) |
| `output_dir` | Result directory (if CLI `--out` is omitted) |

Optional `complex_dtype` (as in `sweep`) to override.

### Per-frequency loss weights (`loss_weighting`)

Uniform averaging often over-emphasizes **skirts** (large gradients). To **weight the passband more**, set per-point weights (“how much this frequency matters”).

Optional block in `optimization`:

```json
"loss_weighting": {
  "mode": "two_band",
  "f_pass_min_hz": 10698500,
  "f_pass_max_hz": 10700000,
  "w_pass": 15,
  "w_stop": 1
}
```

- **`two_band`** — weight `w_pass` inside \([f_\mathrm{pass,min}, f_\mathrm{pass,max}]\), else `w_stop`. Use when passband edges are known.
- **`target_above_db`** — weight `w_pass` where **interpolated target** exceeds `db_threshold`, else `w_stop`. No manual Hz: passband follows the target shape.

```json
"loss_weighting": {
  "mode": "target_above_db",
  "db_threshold": -35,
  "w_pass": 15,
  "w_stop": 1
}
```

- **`target_level_decay`** — weight from **reference level** (interpolated target on the optimization grid). At the global target maximum, weight `w_peak` (default 1). Every **`slope_db`** dB below that peak, weight ×0.1 (linear progression on the **log** dBm axis). Example: `slope_db: 20` → 20 dB below peak → weight 0.1; 40 dB below → 0.01. Parameters: `slope_db` (required, > 0), `w_peak`, `w_min` (floor).

```json
"loss_weighting": {
  "mode": "target_level_decay",
  "slope_db": 20,
  "w_peak": 1.0,
  "w_min": 1e-8
}
```

- **`shifted_pred_max_decay`** — at frequency `f`, level for weighting is **`max(target_shifted(f), current(f))` in dBm**; reference peak is **`max target_shifted`** over points where the loss mask is true. Same formula: `w = w_peak · 10^(-(y_peak - level)/slope_db)`, then clamp. Default **`slope_db`: 60** → weight **0.1** at **60 dB** below shifted-target peak. Weights **recomputed every forward** (depend on `pred` and `delta_f`, `delta_y`).

```json
"loss_weighting": {
  "mode": "shifted_pred_max_decay",
  "slope_db": 60,
  "w_peak": 1.0,
  "w_min": 1e-8
}
```

Guidelines: try `w_pass / w_stop` around **5…20**; for `target_above_db`, set threshold **below** passband plateau but **above** stopband floor (e.g. −30…−45 dBm depending on target). For `target_level_decay`, reduce `slope_db` to pull weight faster away from the peak. Too aggressive weights can effectively remove skirts from the objective.

Module: [`xtal_filters/loss_weights.py`](../xtal_filters/loss_weights.py).

### Parameter snapshots and progress

- `params_snapshot_every` (integer, > 0) — every N steps write JSON with all physical parameters to `output_dir/params_frames/step_XXXXXX.json` (plus `step`, `loss`, `delta_f_hz`, `delta_y_db`).
- Terminal progress uses **tqdm** (`optimize`).

## Response and dBm

Average power on the load resistor matches the reference notebook phasor convention:

\(P = \frac{1}{2}\,\mathrm{Re}(V \cdot I^*)\) (real-valued equivalent in code).

**dBm** (power relative to 1 mW):

\(\mathrm{dBm} = 10 \log_{10}(P_{\mathrm{W}} / 10^{-3})\).

Load resistance is arbitrary, set by the `load_element` resistor parameter.

With **`response.relative_to_input_power`: true** the vertical axis is **difference** dBm(load) − dBm(\(E^2/(8R)\)) for **peak** \(E\) in MNA (not absolute mW). Axis label: “dB (load − matched gen.)”.

## Command line

Working directory: repository root (`cd /path/to/crystal-rf-filter-optimizer`).

### Example: ~10.7 MHz ladder (reference → lossy optimization → yzoom)

**1. Reference curve** (`ladder_ideal.json`, crystals with \(R_m=0\)) — `target` mode:

```bash
python3 -m xtal_filters target \
  --config examples/ladder_ideal.json \
  --out examples/ladder_target \
  --device cpu
```

Output in `examples/ladder_target/`: `target.npz`, `target_plot.png`, `target_meta.json`.

**2. Fit lossy netlist** (`ladder_optimize.json`: trainable capacitors and \(R_\mathrm{port}\), fixed \(R_m,L_m,C_m,C_p\), `loss_weighting`, `params_snapshot_every` for yzoom):

```bash
python3 -m xtal_filters optimize \
  --config examples/ladder_optimize.json \
  --target examples/ladder_target/target.npz
```

By default artifacts go to `optimization.output_dir` in JSON (`examples/ladder_run`), or pass `--out <dir>`.

**3. Y-zoom GIF / PNG** (requires `params_frames/step_*.json`):

```bash
python3 scripts/rebuild_optimization_gif_yzoom.py \
  --config examples/ladder_optimize.json \
  --run-dir examples/ladder_run \
  --save-final examples/ladder_run/final_yzoom.png
```

Default GIF path: `<run-dir>/optimization_yzoom.gif`. Flags: `--y-bottom`, `--y-margin`, `--out-gif`, `--duration-ms`, `--device`.

### Pipeline

1. `ladder_ideal.json` → `target` → `examples/ladder_target/target.npz`.  
2. `ladder_optimize.json` → `optimize` with that `target.npz` → `examples/ladder_run/`.  
3. Optionally `rebuild_optimization_gif_yzoom.py` on the run directory.

## Optimization artifacts (`output_dir`)

| File | Contents |
|------|----------|
| `target.npz` | Copy of input target |
| `responses.npz` | `freqs_hz`, `initial`, `final` — dBm before/after |
| `parameters_final.json` | All parameters in SI units |
| `trainable.json` | Trainable subset only |
| `optimization_log.json` | Step history: loss, `delta_f_hz`, `delta_y_db`, … |
| `params_frames/` | If `params_snapshot_every` > 0 — per-step JSON (needed for yzoom script) |
| `final.png` | Summary plot: **initial** (pre-optimization), ideal target, shifted target, **final** |
| `optimization.gif` | Animation, auto Y scale |
| `optimization_yzoom.gif` | Not written by the optimizer; created by `rebuild_optimization_gif_yzoom.py` |
| `final_yzoom.png` | Same, when `--save-final` is passed to the rebuild script |

## Python API

```python
import torch
from xtal_filters import ACAnalysis, generate_target_artifacts, run_optimization
from xtal_filters.config import load_json

cfg = load_json("examples/ladder_ideal.json")
model = ACAnalysis(cfg, device=torch.device("cpu"))
f = torch.linspace(10.696e6, 10.702e6, 256, dtype=torch.float64)
dbm = model(f)  # (n_freq,) dBm at load

generate_target_artifacts("examples/ladder_ideal.json", "examples/ladder_target")

cfg2 = load_json("examples/ladder_optimize.json")
run_optimization(cfg2, "examples/ladder_target/target.npz", "examples/ladder_run")
```

## Notes

- **CUDA:** if `torch.linalg.solve` fails on your PyTorch build, switch to **`cpu`**.
- Target shift: outside the original target frequency span, points are excluded from the averaged loss via a mask (see `interp.py`).
- No SPICE in the optimization loop — custom MNA only.
