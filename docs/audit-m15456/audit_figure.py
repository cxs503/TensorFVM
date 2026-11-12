#!/usr/bin/env python3
"""Companion to audit_double_mach.py: field-level check + report figure.

Runs the shipped double-Mach case and a textbook-datum counterfactual on the
same 120x30 grid, saves the density/pressure fields as .npz and draws the
audit figure (two density maps + the wall-row density profile).  Every number
plotted is read back from the arrays, nothing is hand-typed.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_double_mach import lab_velocity  # noqa: E402


def run(case_mod, corrected: bool):
    cfg = case_mod.DoubleMachConfig(nx=120, ny=30, t_end=0.2, cfl=0.3)
    if corrected:
        good = lab_velocity(10.0, 60.0)
        orig = case_mod._post_shock_state
        case_mod._post_shock_state = lambda *a, **k: {
            "rho": good["rho"], "u": good["u"], "v": good["v"], "p": good["p"],
            "shock_speed": good["front_speed_x"], "mn": 10.0}
        try:
            solver, meta = case_mod.double_mach_reflection(cfg)
        finally:
            case_mod._post_shock_state = orig
    else:
        solver, meta = case_mod.double_mach_reflection(cfg)
    solver.run(cfg.t_end)
    prim = solver.primitive()
    return {k: v.numpy() for k, v in prim.items()}, meta


def stats(prim, xc, yc):
    rho = prim["rho"]
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    win = (X > 0.35) & (X < 3.5) & (Y > 0.1) & (Y < 0.9)   # drop BC/corner band
    return {
        "max_rho_full": float(rho.max()),
        "argmax_xy": [float(X.ravel()[rho.argmax()]), float(Y.ravel()[rho.argmax()])],
        "max_rho_window": float(rho[win].max()),
        "argmax_window_xy": [float(X[win][rho[win].argmax()]), float(Y[win][rho[win].argmax()])],
        "max_p_full": float(prim["p"].max()),
        "min_rho": float(rho.min()),
        "wall_row_max": float(rho[:, 0].max()),
        "wall_row_mean": float(rho[:, 0].mean()),
        "wall_row_min": float(rho[:, 0].min()),
    }


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    case_mod = importlib.import_module("tensorfvm.compressible.cases")

    res = {}
    fields = {}
    for tag, corrected in (("shipped", False), ("textbook", True)):
        p, meta = run(case_mod, corrected)
        fields[tag] = p
        res[tag] = stats(p, meta["xc"].numpy(), meta["yc"].numpy())
    np.savez_compressed(out / "dm_fields_120x30.npz",
                        **{f"{t}_{k}": v for t, p in fields.items() for k, v in p.items()})
    (out / "dm_field_stats.json").write_text(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xc = np.linspace(0, 4, 120, endpoint=False) + 4 / 240
    yc = np.linspace(0, 1, 30, endpoint=False) + 1 / 60
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for a, tag, title in ((ax[0], "shipped", "shipped datum (Riemann-solved v>0)"),
                          (ax[1], "textbook", "textbook datum (M_n=10, v<0)")):
        rho = fields[tag]["rho"]
        cs = a.contourf(X, Y, rho, levels=40, cmap="inferno")
        a.set_title(f"rho, 120x30, t=0.2\n{title}\nmax={res[tag]['max_rho_full']:.2f} "
                    f"wall-mean={res[tag]['wall_row_mean']:.2f}", fontsize=9)
        a.set_xlabel("x"); a.set_ylabel("y"); a.set_aspect("equal")
        fig.colorbar(cs, ax=a, shrink=0.85)
    ax[2].plot(xc, fields["shipped"]["rho"][:, 0], "o-", ms=3, label="shipped datum")
    ax[2].plot(xc, fields["textbook"]["rho"][:, 0], "s-", ms=3, label="textbook datum")
    ax[2].axhline(8.0, color="k", ls="--", lw=1, label="post-shock rho2 = 8.0")
    ax[2].axhline(1.4, color="grey", ls=":", lw=1, label="pre-shock rho1 = 1.4")
    ax[2].set_xlabel("x along wall (y=0)"); ax[2].set_ylabel("rho")
    ax[2].set_title("wall-row density: no reflection vs. reflected stem", fontsize=9)
    ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    fig.suptitle("m15456 audit of m15331 CASE-DOUBLE-MACH (TensorFVM)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_dm_audit.png", dpi=140)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())