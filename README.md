**[Русская версия](README_ru.md)**

# Crystal RF filter optimizer

PyTorch-based tool to **optimize ladder and similar crystal filter networks** so their **measured-style frequency response** (power on a load, in dBm or dB relative to a matched generator) **tracks a target curve** — typically the response of an **ideal** design.

**Repository:** [github.com/MatthewMih/crystal-rf-filter-optimizer](https://github.com/MatthewMih/crystal-rf-filter-optimizer)  
**Full reference (JSON schema, CLI, API, loss weighting — Russian):** [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)

---

## Why this exists

Classic ladder design flows (e.g. **Dishal** and similar) usually assume **ideal crystals**: infinite Q, no series loss. Real quartz resonators have **finite Q** and a **motional resistance** \(R_m\) (Butterworth–Van Dyke model). That alone changes the passband: it gets **narrower**, the **ripple/flatness** worsens, and you see **extra insertion loss** compared with the textbook curve.

This project lets you:

1. Build a **target** response from a JSON circuit (often “ideal” or fixed-nominal crystals).  
2. Describe a **more realistic** network (same topology, trainable capacitors / port R, fixed crystal motional parameters).  
3. Run **differentiable optimization** (Adam, optional LBFGS) so the **non-ideal** filter’s response **matches the target shape** in dB over a frequency sweep.

You still need physically realizable parts and sensible starting values; the optimizer **tweaks nominated values**, it does not replace filter theory.

---

## Example (after optimization)

Narrow Y-axis plot: **ideal target**, **shifted target** (learnable small frequency / level alignment), and **optimized** ladder response (~10.7 MHz example, motional resistance included). Vertical scale is **dB on load relative to available power from a matched Thevenin source** (see documentation).

![Optimized ladder vs ideal target (yzoom)](docs/assets/ladder_optimization_example.png)

*Figure: `examples/ladder_nonideal_opt`-style run with passband-focused loss weighting; asset copied from `examples/ladder_opt_slope40/final_yzoom.png`.*

---

## Features (short)

- JSON circuit: nodes, branches, shared named parameters.  
- Elements: `Resistor`, `Capacitor`, `Inductor`, `Impedance`, `VoltageSource`, **`Crystal` (BVD: Rm, Lm, Cm, Cp)**, `CrystalLCC`.  
- **MNA** AC solver, batched over frequency, `torch.linalg.solve`, autodiff through parameters.  
- **`target` mode:** save `target.npz` + plot.  
- **`optimize` mode:** L1/L2 on dB, optional **frequency-dependent loss weights**, learnable target shifts `delta_f` / `delta_y`, GIF / parameter snapshots.  
- Helper script: [`scripts/rebuild_optimization_gif_yzoom.py`](scripts/rebuild_optimization_gif_yzoom.py) for a **zoomed** animation.

---

## Requirements

Python ≥ 3.10, PyTorch 2.x, NumPy, Matplotlib, Pillow (see `requirements.txt`).

---

## Install

```bash
cd crystal-rf-filter-optimizer
python3 -m pip install -e .
```

---

## Quick start (ladder ~10.7 MHz)

From the repo root:

```bash
# 1) Ideal / reference response → target.npz
python3 -m xtal_filters target \
  --config examples/ladder_10p696_10p702MHz.json \
  --out examples/ladder_10p7MHz_out \
  --device cpu

# 2) Optimize non-ideal ladder to match that target
python3 -m xtal_filters optimize \
  --config examples/ladder_nonideal_opt.json \
  --target examples/ladder_10p7MHz_out/target.npz \
  --out examples/my_run

# 3) Optional: rebuild GIF with a tight Y window (needs params snapshots in JSON)
python3 scripts/rebuild_optimization_gif_yzoom.py \
  --config examples/ladder_nonideal_opt.json \
  --run-dir examples/my_run \
  --save-final examples/my_run/final_yzoom.png
```

Use **`optimization.device`: `cpu` or `cuda`**. Apple **MPS is not supported** for this AC solver (complex linear algebra).

---

## Documentation

| Document | Content |
|----------|---------|
| [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) | JSON schema, `response` / `optimization`, loss weighting modes, dBm definitions, artifacts, Python API |
| [README_ru.md](README_ru.md) | This page in Russian |

---

## License

[MIT](LICENSE)
