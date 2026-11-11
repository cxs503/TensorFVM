#!/usr/bin/env python
"""Campaign driver for node P447W2-N01 (FVM two-phase VOF three-case face).

    PYTHONPATH=src python docs/P447W2-N01-vof-three-case/run_campaign.py --leg all

Each leg writes its own JSON as soon as it finishes (``artifacts/static.json``,
``artifacts/zalesak.json``, ``artifacts/dam_break.json``), and the merged
acceptance record ``artifacts/vof_threecase.json`` is rebuilt from whatever legs
exist.  A crashed or killed leg therefore costs one leg, not the campaign.

Figures are produced from the in-memory fields at the end of each leg and
written to ``artifacts/figures/``.  Everything is float64 torch; nothing is
transcribed by hand -- the report renderer reads the JSON only.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_num_threads(max(1, int(os.environ.get("TP_THREADS", "8"))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorfvm.vof_threecase.campaign import (
    DAM_GRID_SEQUENCE,
    _strip,
    dambreak_sweep,
    static_bubble_sweep,
    verdicts,
    write_summary,
    zalesak_sweep,
)
from tensorfvm.vof_threecase.dambreak import DamBreakMMConfig, run_dam_break
from tensorfvm.vof_threecase.statics import StaticBubbleConfig, run_static_bubble
from tensorfvm.vof_threecase.zalesak import NotchConfig, notch_mask, run_long_rotation
from tensorfvm.vof_phase0.anchors import (
    MARTIN_MOYCE_FRONT_T,
    MARTIN_MOYCE_FRONT_X,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(ART, "figures")

PROD_STATIC = StaticBubbleConfig(cells=64, mu_liquid=1.0e-2)


def _ensure_dirs():
    os.makedirs(FIG, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}", flush=True)
    return path


# --------------------------------------------------------------------------- #
# leg 1: static bubble
# --------------------------------------------------------------------------- #
def leg_static():
    cfgs = [StaticBubbleConfig(cells=c, mu_liquid=1.0e-2) for c in (32, 64, 96, 128)]
    for mu in (1.0e-3, 3.0e-2, 1.0e-1, 1.0):
        cfgs.append(StaticBubbleConfig(cells=64, mu_liquid=mu, mu_gas=mu))
    cfgs.append(StaticBubbleConfig(cells=64, mu_liquid=1.0e-2, csf_balance="legacy"))
    records = static_bubble_sweep(cfgs)

    # a field snapshot of the production run (alpha + spurious-current magnitude)
    prod = run_static_bubble(PROD_STATIC)
    solver = prod["_solver"]
    alpha = solver.alpha.detach().numpy()
    u = solver.u.detach().numpy()
    v = solver.v.detach().numpy()
    umag = 0.5 * (np.abs(u[:, :-1]) + np.abs(u[:, 1:]))
    vmag = 0.5 * (np.abs(v[:-1, :]) + np.abs(v[1:, :]))
    speed = np.sqrt(umag ** 2 + vmag ** 2)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    im0 = ax[0].imshow(alpha, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    ax[0].set_title(f"alpha, N={prod['cells']}, t={prod['time_end']:.3g} s")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)
    umax = float(speed.max()) or 1.0
    im1 = ax[1].imshow(speed, origin="lower", extent=[0, 1, 0, 1], cmap="magma",
                       vmin=0.0, vmax=umax)
    ax[1].set_title(r"$|u|$ [m/s], max = %.2e" % umax)
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    for a in ax:
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
    _save(fig, "fig1_static_bubble_fields.png")

    balanced = [r for r in records if r["csf_balance"] == "balanced"]
    legacy = [r for r in records if r["csf_balance"] == "legacy"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for r in balanced:
        ax[0].semilogy(r["history_step"], r["history_ratio"],
                       label=f"N={r['cells']}, $\\mu$={r['mu_liquid']:.0e}")
    for r in legacy:
        ax[0].semilogy(r["history_step"], r["history_ratio"], "--", color="crimson",
                       label="legacy CSF (control)")
    ax[0].axhline(0.01, color="k", ls=":", label="criterion 0.01")
    ax[0].set_xlabel("time step")
    ax[0].set_ylabel(r"$\max|u|/\sqrt{\sigma/\rho_l R}$")
    ax[0].set_title("spurious-current history")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)

    grids = sorted({r["cells"] for r in balanced if abs(r["mu_liquid"] - 1e-2) < 1e-12})
    ratios = [next(r["spurious_ratio"] for r in balanced
                   if r["cells"] == g and abs(r["mu_liquid"] - 1e-2) < 1e-12) for g in grids]
    ax[1].loglog(grids, ratios, "o-", label="balanced CSF")
    ax[1].axhline(0.01, color="k", ls=":", label="criterion 0.01")
    ax[1].set_xlabel("cells per direction")
    ax[1].set_ylabel(r"$\max|u|/u_\sigma$")
    ax[1].set_title("grid dependence ($\\mu_l=10^{-2}$ Pa s)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    _save(fig, "fig2_static_bubble_current.png")

    out = {"static_bubble": records}
    with open(os.path.join(ART, "static.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- #
# leg 2: Zalesak
# --------------------------------------------------------------------------- #
def leg_zalesak():
    cfgs = []
    for cells in (64, 128, 256):
        for deg in (360.0, 1000.0):
            cfgs.append(NotchConfig(cells=cells, scheme="cicsam", degrees=deg))
    for cells in (64, 128):
        cfgs.append(NotchConfig(cells=cells, scheme="plic", degrees=1000.0))
        cfgs.append(NotchConfig(cells=cells, scheme="upwind", degrees=360.0))
    records = zalesak_sweep(cfgs)

    fields = {}
    for scheme, deg in (("cicsam", 360.0), ("cicsam", 1000.0), ("plic", 360.0),
                        ("upwind", 360.0)):
        rec = run_long_rotation(NotchConfig(cells=128, scheme=scheme, degrees=deg))
        fields[(scheme, deg)] = (rec["_exact"].numpy(), rec["_field"].numpy())

    mask = notch_mask(128).numpy()
    keys = list(fields)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.6))
    for ax, k in zip(np.atleast_1d(axes), keys):
        exact, field = fields[k]
        ax.imshow(field, origin="lower", extent=[0, 1, 0, 1], cmap="viridis",
                  vmin=0, vmax=1)
        ax.contour(np.linspace(0.004, 0.996, 128), np.linspace(0.004, 0.996, 128),
                   exact, levels=[0.5], colors="w", linewidths=0.6)
        ax.contour(np.linspace(0.004, 0.996, 128), np.linspace(0.004, 0.996, 128),
                   mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.6,
                   linestyles=":")
        ax.set_title(f"{k[0]}, {k[1]:.0f} deg, N=128", fontsize=9)
        ax.set_xlabel("x")
    _save(fig, "fig3_zalesak_fields.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for scheme, marker in (("cicsam", "o"), ("plic", "s"), ("upwind", "^")):
        for deg, ls in ((360.0, "-"), (1000.0, "--")):
            pts = sorted([(r["cells"], r["notch_error_relative"])
                          for r in records
                          if r["scheme"] == scheme and abs(r["degrees"] - deg) < 1e-9])
            if pts:
                ax[0].loglog([p[0] for p in pts], [max(p[1], 1e-6) for p in pts],
                             marker=marker, ls=ls, label=f"{scheme} {deg:.0f} deg")
    ax[0].axhline(0.05, color="k", ls=":", label="5 % (360 deg)")
    ax[0].axhline(0.15, color="gray", ls=":", label="15 % (1000 deg)")
    ax[0].set_xlabel("cells per direction")
    ax[0].set_ylabel("notch-area relative error")
    ax[0].set_title("notch shape error")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    for scheme, marker in (("cicsam", "o"), ("plic", "s"), ("upwind", "^")):
        for deg, ls in ((360.0, "-"), (1000.0, "--")):
            pts = sorted([(r["cells"], r["area_error_contour_vs_initial"])
                          for r in records
                          if r["scheme"] == scheme and abs(r["degrees"] - deg) < 1e-9])
            if pts:
                ax[1].loglog([p[0] for p in pts], [max(p[1], 1e-8) for p in pts],
                             marker=marker, ls=ls, label=f"{scheme} {deg:.0f} deg")
    ax[1].set_xlabel("cells per direction")
    ax[1].set_ylabel("enclosed-area error vs initial contour")
    ax[1].set_title("global shape error (for comparison)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    _save(fig, "fig4_zalesak_notch.png")

    out = {"zalesak": records}
    with open(os.path.join(ART, "zalesak.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- #
# leg 3: dam break
# --------------------------------------------------------------------------- #
def leg_dam():
    cfgs = [DamBreakMMConfig(ny=ny, nx=nx, aspect=2.0, column_width=0.05715,
                             length=0.4, height=0.2) for ny, nx in DAM_GRID_SEQUENCE]
    records = dambreak_sweep(cfgs)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(MARTIN_MOYCE_FRONT_T, MARTIN_MOYCE_FRONT_X, "k^--",
               label="Martin & Moyce (1952)", ms=5)
    for r in records:
        ax[0].plot(r["history_t_star"], r["history_x_star"], lw=1.2,
                   label=f"FVM {r['ny']}x{r['nx']}")
    ax[0].set_xlabel(r"$t^* = t\sqrt{g/W}$")
    ax[0].set_ylabel(r"$x^* = (x_f - W)/W$")
    ax[0].set_title("surge front")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for r in records:
        ax[1].plot([s["t_star"] for s in r["samples"]],
                   [100 * s["relative_error"] for s in r["samples"]], "o-",
                   label=f"{r['ny']}x{r['nx']}")
    ax[1].axhline(10, color="k", ls=":", label="+10 %")
    ax[1].axhline(-10, color="k", ls=":", label="-10 %")
    ax[1].set_xlabel(r"$t^*$")
    ax[1].set_ylabel("front error vs experiment [%]")
    ax[1].set_title("error at the criterion times")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    _save(fig, "fig5_dam_break_front.png")

    # finest-grid field + free-surface profile
    finest = run_dam_break(DamBreakMMConfig(ny=DAM_GRID_SEQUENCE[-1][0],
                                            nx=DAM_GRID_SEQUENCE[-1][1],
                                            aspect=2.0, column_width=0.05715,
                                            length=0.4, height=0.2))
    alpha = finest["_alpha_final"].numpy()
    solver = finest["_solver"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    im = ax[0].imshow(alpha, origin="lower",
                      extent=[0, 0.4, 0, 0.2], cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax[0], shrink=0.85)
    ax[0].set_title(f"alpha at t = {finest['t_end']:.2f} s "
                    f"({finest['ny']}x{finest['nx']})")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    x = (np.arange(alpha.shape[1]) + 0.5) * (0.4 / alpha.shape[1])
    surf = alpha.argmax(axis=0)
    has = alpha.max(axis=0) > 0.5
    ax[1].plot(x[has], surf[has] * (0.2 / alpha.shape[0]), "b.-", ms=2,
               label="interface cell (alpha = 0.5)")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("free surface y [m]")
    ax[1].set_title("free-surface profile at the final time")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    _save(fig, "fig6_dam_break_fields.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in records:
        ax.semilogy(r["history_t"], np.maximum(r["history_mass_drift"], 1e-18),
                    label=f"{r['ny']}x{r['nx']}, max {r['mass_drift_max']:.1e}")
    ax.axhline(1e-3, color="k", ls=":", label="criterion 0.1 %")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$|M(t)-M(0)|/M(0)$")
    ax.set_title("global mass ledger (dam break)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    _save(fig, "fig7_dam_break_mass.png")

    out = {"dam_break": [{**_strip(r)} for r in records]}
    with open(os.path.join(ART, "dam_break.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    return out


def merge():
    payload = {"node": "P447W2-N01",
               "title": "FVM two-phase VOF three-case verification face",
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for leg, key in (("static", "static_bubble"), ("zalesak", "zalesak"),
                     ("dam_break", "dam_break")):
        path = os.path.join(ART, leg + ".json")
        if os.path.exists(path):
            with open(path) as fh:
                payload.update(json.load(fh))
    out = os.path.join(ART, "vof_threecase.json")
    write_summary(out, payload)
    print(f"[merge] {out}  all_pass={payload['all_pass']}", flush=True)
    for k, c in payload["criteria"].items():
        print(f"  {k}: {'PASS' if c.get('pass') else 'FAIL'}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", default="all",
                    choices=["static", "zalesak", "dam", "all", "merge"])
    args = ap.parse_args()
    _ensure_dirs()
    t0 = time.time()
    if args.leg in ("static", "all"):
        leg_static()
    if args.leg in ("zalesak", "all"):
        leg_zalesak()
    if args.leg in ("dam", "all"):
        leg_dam()
    merge()
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()