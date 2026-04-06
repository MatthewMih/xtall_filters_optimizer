from __future__ import annotations

import argparse
from pathlib import Path

from xtal_filters.config import load_json, validate_schema
from xtal_filters.optimize import run_optimization
from xtal_filters.target_io import generate_target_artifacts


def main() -> None:
    p = argparse.ArgumentParser(description="Quartz filter JSON AC analysis and optimization")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("target", help="Generate target response from circuit JSON")
    t.add_argument("--config", required=True, help="Path to circuit JSON")
    t.add_argument("--out", required=True, help="Output directory")
    t.add_argument("--device", default="cpu")

    o = sub.add_parser("optimize", help="Match circuit to saved target .npz")
    o.add_argument("--config", required=True, help="Path to circuit JSON (includes optimization section)")
    o.add_argument("--target", required=True, help="Path to target.npz (freqs_hz, dbm)")
    o.add_argument(
        "--out",
        required=False,
        default=None,
        help="Output dir (default: optimization.output_dir from JSON)",
    )

    args = p.parse_args()
    if args.cmd == "target":
        generate_target_artifacts(args.config, args.out, device=args.device)
        print("Wrote target.npz and target_plot.png to", args.out)
        return
    if args.cmd == "optimize":
        cfg = load_json(args.config)
        validate_schema(cfg)
        out = args.out or cfg.get("optimization", {}).get("output_dir")
        if not out:
            raise SystemExit("Provide --out or optimization.output_dir in JSON")
        run_optimization(cfg, args.target, out)
        print("Optimization done. Artifacts in", out)
        return


if __name__ == "__main__":
    main()
