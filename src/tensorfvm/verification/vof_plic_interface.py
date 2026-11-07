"""Minimal verification surface for the PLIC interface transport (two-phase VOF).

Scope
-----
This module is the *minimal face* a two-phase VOF capability has to clear before
any free-surface ship result can be believed.  Four things are measured, each
against an exact reference or a declared criterion:

1. **1-D step interface transport** — a sharp step advected through a periodic
   grid; quantified by the 10 %–90 % interface width (in cells), the L1 error
   against the exact translated step and the peak retention, for the geometric
   PLIC scheme, its 1-D restriction and a first-order donor-cell reference.
2. **Zalesak slotted-disk rotation** (Zalesak 1979, one full revolution in a
   divergence-free rigid-rotation field) — area (volume) error, shape error,
   peak retention and distorted fraction.  Geometry is *reused* from
   :func:`tensorfvm.verification.vof_rotation.zalesak_disk`
   (``domain = 1``, centre ``(0.5, 0.75)``, ``R = 0.15``, slot half-width
   ``0.05``), i.e. the same geometry口径 as the plan-13 P2 fluid leg; the
   provenance is stated in the report so the two legs cannot drift apart.
3. **Global mass conservation** — the per-step relative volume change
   ``|Σα_new − Σα_old| / Σα_old``; criterion ``< 1e-6`` per step.
4. **Negative evidence** — the input guards (:func:`validate_alpha`,
   :func:`validate_velocity`, :func:`validate_directional_speed`) must *reject*
   out-of-range volume fractions, a non-positive time step and a negative
   transport-rate magnitude instead of silently repairing them.

The donor-cell reference is the first-order stencil of
:class:`tensorfvm.two_phase_vof.VOFTransport` (``scheme="upwind"``) re-expressed
with **periodic** boundary conditions, so both schemes see the same BCs and the
comparison isolates the interface reconstruction.  The library's own transport
uses transmissive zero-gradient ghosts, which does not conserve mass for a
non-uniform field; that difference is measured and reported separately instead
of being mixed into the scheme comparison.

Usage
-----
    python -m tensorfvm.verification.vof_plic_interface --output <dir>
    python -m tensorfvm.verification.vof_plic_interface --quick --output <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from ..two_phase_vof import VOFTransport
from ..vof_plic import (
    advect_1d_step,
    advect_plic,
    interface_width_cells,
    validate_alpha,
    validate_directional_speed,
    validate_velocity,
)
from .vof_rotation import rotation_velocity, zalesak_disk

DOMAIN = 1.0
CFL = 0.4
SHIFT = 0.3
MASS_TOL = 1e-6
AREA_TOL = 0.05  # 5 %

__all__ = [
    "donor_advect_periodic",
    "exact_step_field",
    "one_d_step_case",
    "zalesak_case",
    "negative_evidence",
    "run",
    "main",
]


# ---------------------------------------------------------------------------
# reference schemes and exact solutions
# ---------------------------------------------------------------------------
def donor_advect_periodic(
    alpha: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    order: str = "xy",
) -> torch.Tensor:
    """First-order donor-cell (upwind) step with **periodic** boundaries.

    This is the stencil of :class:`tensorfvm.two_phase_vof.VOFTransport` with
    ``scheme="upwind"`` — the face value is the upwind cell's own volume fraction
    — re-expressed with periodic wrapping.  It is the fair first-order baseline
    for :func:`tensorfvm.vof_plic.advect_plic`: identical velocity field, identical
    boundaries, no interface reconstruction.
    """
    a = alpha.to(torch.float64)
    for axis in (1, 0) if order == "xy" else (0, 1):
        work = a if axis == 1 else a.T.contiguous()
        vel = (u if axis == 1 else v).to(torch.float64)
        if axis != 1:
            vel = vel.T.contiguous()
        ny, nx = work.shape
        if vel.shape != (ny, nx + 1):
            raise ValueError(f"face velocity shape {vel.shape} != {(ny, nx + 1)}")
        padded = torch.cat((work[:, -1:], work, work[:, :1]), dim=1)
        cour = vel * float(dt) / float(dx if axis == 1 else dy)
        C = cour.abs()
        donor = torch.where(cour >= 0.0, padded[:, :-1], padded[:, 1:])
        flux = torch.where(cour >= 0.0, C * donor, -C * donor)
        net = flux[:, 1 : nx + 1] - flux[:, 0:nx]
        out = (work - net).clamp(0.0, 1.0)
        a = out if axis == 1 else out.T.contiguous()
    return a


def exact_step_field(n: int, shift: float, dx: float) -> torch.Tensor:
    """Exact periodic translation of a step at ``x = 0.5`` by ``shift``."""
    xc = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    xm = torch.remainder(xc - shift, DOMAIN)
    return ((xm >= 0.0) & (xm < 0.5)).to(torch.float64).reshape(1, n)


def _metrics(field: torch.Tensor, exact: torch.Tensor) -> Dict[str, float]:
    err = (field - exact).abs()
    peak = float(field.max().item())
    peak_ref = float(exact.max().item())
    return {
        "l1": float(err.mean().item()),
        "linf": float(err.max().item()),
        "min": float(field.min().item()),
        "max": peak,
        "peak_retention": peak / peak_ref if peak_ref > 0 else float("nan"),
        "distorted_fraction": float((err > 0.05).to(torch.float64).mean().item()),
    }


# ---------------------------------------------------------------------------
# 1-D step interface transport
# ---------------------------------------------------------------------------
def one_d_step_case(
    n: int, cfl: float = CFL, shift: float = SHIFT, library_baseline: bool = True
) -> Dict[str, object]:
    """Advect a step interface and quantify the resulting numerical diffusion."""
    dx = DOMAIN / n
    nsteps = max(1, int(round(shift / (cfl * dx))))
    dt = shift / nsteps  # |u| = 1 → the step displacement is the time step
    cour = dt / dx
    u = torch.ones((1, n + 1), dtype=torch.float64)
    v = torch.zeros((2, n), dtype=torch.float64)

    a0 = torch.zeros(1, n, dtype=torch.float64)
    a0[0, : n // 2] = 1.0
    exact = exact_step_field(n, nsteps * dt, dx)

    out: Dict[str, object] = {
        "cells": n,
        "steps": nsteps,
        "courant": cour,
        "shift": nsteps * dt,
        "interface_cells_exact": 1.0,
    }

    # --- geometric PLIC (2-D code path, used as the 1-D transport) ---
    a = a0.clone()
    worst_step = 0.0
    for k in range(nsteps):
        vol0 = float(a.sum().item())
        a = advect_plic(a, u, v, dt, dx, dx, order="xy" if k % 2 == 0 else "yx", clip=False)
        worst_step = max(worst_step, abs(float(a.sum().item()) - vol0) / vol0)
    out["plic"] = dict(
        _metrics(a, exact),
        interface_width_cells=interface_width_cells(a),
        volume_drift=float(a.sum().item() - a0.sum().item()),
        worst_step_relative_drift=worst_step,
    )

    # --- PLIC 1-D restriction (exact for a linear interface) ---
    b = a0.clone()
    worst_1d = 0.0
    for _ in range(nsteps):
        vol0 = float(b.sum().item())
        b = advect_1d_step(b, 1.0, dt, dx)
        worst_1d = max(worst_1d, abs(float(b.sum().item()) - vol0) / vol0)
    out["plic_1d"] = dict(
        _metrics(b, exact),
        interface_width_cells=interface_width_cells(b),
        volume_drift=float(b.sum().item() - a0.sum().item()),
        worst_step_relative_drift=worst_1d,
    )

    # --- first-order donor-cell reference (periodic, same BCs) ---
    c = a0.clone()
    worst_donor = 0.0
    for k in range(nsteps):
        vol0 = float(c.sum().item())
        c = donor_advect_periodic(c, u, v, dt, dx, dx, order="xy" if k % 2 == 0 else "yx")
        worst_donor = max(worst_donor, abs(float(c.sum().item()) - vol0) / vol0)
    out["donor_cell"] = dict(
        _metrics(c, exact),
        interface_width_cells=interface_width_cells(c),
        volume_drift=float(c.sum().item() - a0.sum().item()),
        worst_step_relative_drift=worst_donor,
    )
    # analytic 10-90 % width of the donor-cell (pure diffusion) profile
    D = 0.5 * dx * (1.0 - cour)  # (u dx / 2)(1 - C), u = 1
    t = nsteps * dt
    out["donor_cell"]["interface_width_theory"] = (
        2.563 * math.sqrt(2.0 * D * t) / dx if D > 0 else 0.0
    )

    if library_baseline:
        # the shipped transport with its transmissive zero-gradient ghosts
        t_lib = VOFTransport(1, n, dx, dx, scheme="upwind")
        d = a0.clone()
        for _ in range(nsteps):
            d = t_lib.advect(d, u, v, dt)
        out["library_upwind_transmissive"] = dict(
            _metrics(d, exact),
            interface_width_cells=interface_width_cells(d),
            volume_drift=float(d.sum().item() - a0.sum().item()),
            volume_drift_relative=float(
                (d.sum().item() - a0.sum().item()) / a0.sum().item()
            ),
            note="transmissive BCs: net boundary exchange is not zero for a "
            "non-uniform field, so this drift is a BC artefact, not a flux error",
        )
    out["fields"] = {
        "exact": exact.reshape(-1).tolist(),
        "plic": a.reshape(-1).tolist(),
        "donor_cell": c.reshape(-1).tolist(),
    }
    return out


# ---------------------------------------------------------------------------
# Zalesak slotted-disk rotation
# ---------------------------------------------------------------------------
def zalesak_case(n: int, cfl: float = CFL, revolutions: int = 1) -> Dict[str, object]:
    """``revolutions`` rigid-body revolutions of the Zalesak slotted disk.

    The rigid rotation is an exact advection, so after an **integer** number of
    revolutions the discrete initial field *is* the exact solution — no
    interpolation of a reference field is involved anywhere in the comparison.
    Non-integer revolutions are refused: the exact field would then have to be
    resampled at rotated cell centres, which injects an O(Δx) sampling error into
    the very quantity being judged.
    """
    if int(revolutions) != revolutions or revolutions < 1:
        raise ValueError(
            "revolutions must be a positive integer (the exact solution after a "
            "partial revolution is a resampled field, not the initial field)"
        )
    revolutions = int(revolutions)
    dx = DOMAIN / n
    alpha0 = zalesak_disk(n)
    u, v, dx_u = rotation_velocity(n)
    if abs(dx_u - dx) > 1e-14:
        raise ValueError(f"geometry mismatch: dx {dx_u} != {dx}")
    speed = float(u.abs().max().item())
    nsteps = max(1, int(round(revolutions / (cfl * dx / speed))))
    dt = revolutions / nsteps
    exact = alpha0.clone()

    out: Dict[str, object] = {
        "cells": n,
        "steps": nsteps,
        "dt": dt,
        "courant_max": speed * dt / dx,
        "geometry": {
            "source": "tensorfvm.verification.vof_rotation.zalesak_disk",
            "domain": DOMAIN,
            "centre": [0.5, 0.75],
            "radius": 0.15,
            "slot_half_width": 0.05,
        },
    }
    volume0 = float(alpha0.sum().item())

    def _run(scheme: str):
        a = alpha0.clone()
        worst = 0.0
        raw_min, raw_max = float("inf"), float("-inf")
        for k in range(nsteps):
            vol_old = float(a.sum().item())
            if scheme == "plic":
                a = advect_plic(
                    a, u, v, dt, dx, dx,
                    order="xy" if k % 2 == 0 else "yx", clip=True,
                )
            else:
                a = donor_advect_periodic(
                    a, u, v, dt, dx, dx, order="xy" if k % 2 == 0 else "yx"
                )
            worst = max(worst, abs(float(a.sum().item()) - vol_old) / vol_old)
            raw_min = min(raw_min, float(a.min().item()))
            raw_max = max(raw_max, float(a.max().item()))
        area_err = abs(float(a.sum().item()) - volume0) / volume0
        return dict(
            _metrics(a, exact),
            area_error=area_err,
            worst_step_relative_drift=worst,
            raw_min=raw_min,
            raw_max=raw_max,
            field=a.detach().clone(),
        )

    for scheme in ("plic", "donor_cell"):
        res = _run(scheme)
        out[scheme] = {k: v for k, v in res.items() if k != "field"}
        out[scheme + "_field"] = res["field"]
    out["exact_field"] = exact.detach().clone()
    return out


# ---------------------------------------------------------------------------
# negative evidence (input guards)
# ---------------------------------------------------------------------------
def negative_evidence() -> Dict[str, Dict[str, object]]:
    """Exercise every input guard; each case must raise ``ValueError``."""
    good = torch.zeros(2, 4, dtype=torch.float64)
    good[0, :2] = 1.0
    u = torch.zeros((2, 5), dtype=torch.float64)
    v = torch.zeros((3, 4), dtype=torch.float64)
    cases = {
        "alpha_above_one": lambda: validate_alpha(good + 1.5),
        "alpha_below_zero": lambda: validate_alpha(good - 0.25),
        "alpha_nan": lambda: validate_alpha(
            torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float64)
        ),
        "dt_non_positive": lambda: validate_velocity(u, v, 0.0, 0.1, 0.1),
        "dt_negative": lambda: validate_velocity(u, v, -1e-3, 0.1, 0.1),
        "cfl_above_one": lambda: validate_velocity(
            u + 10.0, v, 0.1, 0.1, 0.1
        ),
        "negative_directional_speed": lambda: validate_directional_speed(-0.5),
    }
    report: Dict[str, Dict[str, object]] = {}
    for name, fn in cases.items():
        try:
            fn()
        except ValueError as exc:
            report[name] = {"rejected": True, "message": str(exc)[:160]}
        else:  # pragma: no cover - a guard regressed
            report[name] = {"rejected": False, "message": "no exception raised"}
    # positive control: signed (negative) face velocities are physical
    ref = good.clone()
    u_neg = -0.2 * torch.ones((2, 5), dtype=torch.float64)
    try:
        validate_velocity(u_neg, v, 0.1, 0.1, 0.1)
        adv = advect_plic(ref, u_neg, v, 0.1, 0.1, 0.1)
        report["negative_velocity_accepted"] = {
            "rejected": False,
            "message": "signed back-flow accepted by design",
            "mass_drift": float(adv.sum().item() - ref.sum().item()),
        }
    except ValueError as exc:  # pragma: no cover
        report["negative_velocity_accepted"] = {"rejected": True, "message": str(exc)}
    return report


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _write_figures(results: Dict[str, object], outdir: Path) -> List[Dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figs: List[Dict[str, str]] = []
    one_d = results["one_d_step"]
    reference = one_d["resolutions"][str(results["parameters"]["one_d_cells"][-1])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    n = reference["cells"]
    x = (np.arange(n) + 0.5) / n
    axes[0].plot(x, reference["fields"]["exact"], "k-", lw=2.0, label="exact (translated step)")
    axes[0].plot(
        x, reference["fields"]["plic"], "o-", ms=3.5, lw=1.2, label="PLIC (geometric)"
    )
    axes[0].plot(
        x, reference["fields"]["donor_cell"], "s--", ms=3.0, lw=1.0, label="donor cell"
    )
    axes[0].set_xlim(0.2, 0.9)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\alpha$")
    axes[0].set_title(f"1-D step interface after {reference['steps']} steps (N={n})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    res = [one_d["resolutions"][str(c)] for c in results["parameters"]["one_d_cells"]]
    ns = [r["cells"] for r in res]
    axes[1].plot(ns, [r["plic"]["interface_width_cells"] for r in res], "o-", label="PLIC")
    axes[1].plot(
        ns, [r["plic_1d"]["interface_width_cells"] for r in res], "^--", label="PLIC (1-D)"
    )
    axes[1].plot(
        ns, [r["donor_cell"]["interface_width_cells"] for r in res], "s-", label="donor cell"
    )
    axes[1].plot(
        ns,
        [r["donor_cell"]["interface_width_theory"] for r in res],
        "k:",
        label=r"donor theory $2.563\sqrt{2Dt}/\Delta x$",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("cells")
    axes[1].set_ylabel("10 %–90 % interface width [cells]")
    axes[1].set_title("interface sharpness vs resolution")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = outdir / "fig-1d-step-interface.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": "fig-1d-step-interface.png",
            "caption": "1-D step transport: PLIC keeps a fixed sub-cell transition "
            "while donor cell diffuses (right: measured 10–90 % width vs grid, "
            "and the analytic donor-cell diffusion width).",
        }
    )

    zres = results["zalesak"]["resolutions"]
    nkey = str(results["parameters"]["zalesak_cells"][-1])
    z = zres[nkey]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panels = [
        (z["exact_field"], "exact"),
        (z["plic_field"], "PLIC"),
        (z["donor_cell_field"], "donor cell"),
    ]
    for ax, (field, title) in zip(axes, panels):
        im = ax.imshow(
            np.asarray(field), origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1,
            cmap="viridis",
        )
        ax.set_title(f"Zalesak, N={z['cells']}: {title}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, fraction=0.025, label=r"$\alpha$")
    path = outdir / "fig-zalesak-fields.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    figs.append(
        {
            "name": "fig-zalesak-fields.png",
            "caption": "Slotted disk after one rigid-body revolution: the geometric "
            "reconstruction preserves the disk and the slot, the donor-cell scheme "
            "smears both.",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ns = [zres[str(c)]["cells"] for c in results["parameters"]["zalesak_cells"]]
    width = 0.35
    idx = np.arange(len(ns))
    axes[0].bar(
        idx - width / 2, [100 * zres[str(c)]["plic"]["area_error"] for c in ns], width,
        label="PLIC",
    )
    axes[0].bar(
        idx + width / 2,
        [100 * zres[str(c)]["donor_cell"]["area_error"] for c in ns],
        width,
        label="donor cell",
    )
    axes[0].axhline(100 * AREA_TOL, color="r", ls="--", lw=1.0, label="5 % criterion")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels([str(n) for n in ns])
    axes[0].set_xlabel("cells")
    axes[0].set_ylabel("area error [%]")
    axes[0].set_title("Zalesak area (volume) error")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].plot(
        ns, [zres[str(c)]["plic"]["peak_retention"] for c in ns], "o-", label="PLIC"
    )
    axes[1].plot(
        ns,
        [zres[str(c)]["donor_cell"]["peak_retention"] for c in ns],
        "s-",
        label="donor cell",
    )
    axes[1].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("cells")
    axes[1].set_ylabel(r"$\max\alpha / \max\alpha_0$")
    axes[1].set_title("peak retention (interface sharpness)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = outdir / "fig-zalesak-metrics.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": "fig-zalesak-metrics.png",
            "caption": "Zalesak area error against the 5 % criterion (left) and peak "
            "retention (right) for both schemes.",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for scheme in ("plic", "donor_cell"):
        drift = zres[nkey][scheme]["worst_step_relative_drift"]
        ax.bar(
            [scheme], [drift], color="#2b7bba" if scheme == "plic" else "#c0504d"
        )
    ax.set_yscale("log")
    ax.axhline(MASS_TOL, color="r", ls="--", lw=1.0, label=r"$10^{-6}$/step criterion")
    ax.set_ylabel("worst per-step relative volume change")
    ax.set_title(f"Mass ledger, Zalesak N={zres[nkey]['cells']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = outdir / "fig-mass-ledger.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": "fig-mass-ledger.png",
            "caption": "Worst per-step relative volume change of both schemes against "
            "the 1e-6 per-step mass-conservation criterion.",
        }
    )
    return figs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _criteria(results: Dict[str, object]) -> Dict[str, object]:
    one_d = results["one_d_step"]
    z = results["zalesak"]
    one_d_cells = results["parameters"]["one_d_cells"]
    z_cells = results["parameters"]["zalesak_cells"]

    widths = {
        str(c): one_d["resolutions"][str(c)]["plic"]["interface_width_cells"]
        for c in one_d_cells
    }
    w_donor = {
        str(c): one_d["resolutions"][str(c)]["donor_cell"]["interface_width_cells"]
        for c in one_d_cells
    }
    l1_plic = {str(c): one_d["resolutions"][str(c)]["plic"]["l1"] for c in one_d_cells}
    l1_donor = {
        str(c): one_d["resolutions"][str(c)]["donor_cell"]["l1"] for c in one_d_cells
    }
    mass = {
        "one_d_worst_step_drift": max(
            one_d["resolutions"][str(c)]["plic"]["worst_step_relative_drift"]
            for c in one_d_cells
        ),
        "zalesak_worst_step_drift": max(
            z["resolutions"][str(c)]["plic"]["worst_step_relative_drift"]
            for c in z_cells
        ),
    }
    area = {
        str(c): z["resolutions"][str(c)]["plic"]["area_error"] for c in z_cells
    }
    area_donor = {
        str(c): z["resolutions"][str(c)]["donor_cell"]["area_error"] for c in z_cells
    }
    peak = {str(c): z["resolutions"][str(c)]["plic"]["peak_retention"] for c in z_cells}
    peak_donor = {
        str(c): z["resolutions"][str(c)]["donor_cell"]["peak_retention"] for c in z_cells
    }
    neg = results["negative_evidence"]
    neg_ok = all(v.get("rejected") for k, v in neg.items() if k != "negative_velocity_accepted")

    return {
        "C1_interface_sharpness": {
            "plic_width_cells": widths,
            "donor_width_cells": w_donor,
            "plic_l1": l1_plic,
            "donor_l1": l1_donor,
            "pass": bool(all(w <= 2.0 for w in widths.values())),
            "criterion": "PLIC 10-90 % interface width <= 2 cells at every resolution",
        },
        "C2_zalesak_area_error": {
            "plic_area_error": area,
            "donor_area_error": area_donor,
            "pass": bool(all(a < AREA_TOL for a in area.values())),
            "criterion": "area (volume) error < 5 % after one revolution",
        },
        "C3_mass_conservation": {
            "plic_worst_step_relative_drift": mass,
            "pass": bool(all(v < MASS_TOL for v in mass.values())),
            "criterion": "worst per-step relative volume change < 1e-6",
        },
        "C4_negative_evidence": {
            "guards": neg,
            "pass": bool(neg_ok),
            "criterion": "every invalid input (alpha outside [0,1], dt <= 0, CFL > 1, "
            "negative directional speed) raises ValueError",
        },
        "C5_peak_retention": {
            "plic_peak_retention": peak,
            "donor_peak_retention": peak_donor,
            "criterion": "peak retention reported (no criterion in the task book; "
            "PLIC must beat donor cell at every resolution)",
            "pass": bool(all(peak[str(c)] > peak_donor[str(c)] for c in z_cells)),
        },
    }


def _markdown(results: Dict[str, object], criteria: Dict[str, object]) -> str:
    one_d = results["one_d_step"]
    z = results["zalesak"]
    one_d_cells = results["parameters"]["one_d_cells"]
    z_cells = results["parameters"]["zalesak_cells"]
    lines: List[str] = []
    add = lines.append
    add("# PLIC interface transport — minimal verification surface (two-phase VOF)\n")
    add(
        "Module under test: `src/tensorfvm/vof_plic.py`. Reproduce:\n\n"
        "```bash\npython -m tensorfvm.verification.vof_plic_interface --output <dir>\n"
        "pytest tests/test_vof_plic_interface.py -q\n```\n"
    )
    add("## 1. What is measured and why\n")
    add(
        "The shipped algebraic transport (`two_phase_vof.VOFTransport`) is flux-form "
        "and conservative but has no interface reconstruction; its own artefacts "
        "record Zalesak peak retention 0.19/0.31/0.52 and relative L1 1.19–1.77 at "
        "32²/64²/128², i.e. the disk is smeared over many cells. A two-phase VOF "
        "capability cannot be validated on momentum alone, so this node measures the "
        "interface transport itself with a geometric PLIC reconstruction.\n"
    )
    add("## 2. Method\n")
    add(
        "* Reconstruction: Youngs 3×3 normal + plane constant from the exact cut-area "
        "inverse (`cut_area`/`plane_constant`, bisection to ~1e-13).\n"
        "* Transport: directional (Lie) split, flux = exact reconstructed liquid volume "
        "in the swept strip of the upwind cell; sweep order alternated each step; "
        "periodic boundaries (the rigid-rotation field is divergence-free, exactly and "
        "discretely).\n"
        "* Reference: first-order donor cell with identical velocity field and identical "
        "periodic BCs (`donor_advect_periodic`), plus the shipped "
        "`VOFTransport(scheme='upwind')` for provenance.\n"
        "* Geometry provenance: Zalesak geometry is imported from "
        "`verification/vof_rotation.py::zalesak_disk` (domain 1, centre (0.5, 0.75), "
        "R = 0.15, slot half-width 0.05) — same口径 as the plan-13 P2 fluid leg.\n"
    )
    add("## 3. Results — 1-D step interface\n")
    add(
        "| N | scheme | L1 | LinF | 10–90 % width [cells] | peak retention | "
        "worst per-step drift |\n|---|---|---|---|---|---|---|"
    )
    for c in one_d_cells:
        r = one_d["resolutions"][str(c)]
        for scheme, label in (
            ("plic", "PLIC"),
            ("plic_1d", "PLIC 1-D"),
            ("donor_cell", "donor cell"),
        ):
            m = r[scheme]
            add(
                f"| {r['cells']} | {label} | {m['l1']:.4e} | {m['linf']:.3f} | "
                f"{m['interface_width_cells']:.0f} | {m['peak_retention']:.3f} | "
                f"{m['worst_step_relative_drift']:.2e} |"
            )
    add("")
    add("## 4. Results — Zalesak slotted disk (one revolution)\n")
    add(
        "| N | scheme | area error | L1 | LinF | peak retention | distorted fraction | "
        "worst per-step drift |\n|---|---|---|---|---|---|---|---|"
    )
    for c in z_cells:
        r = z["resolutions"][str(c)]
        for scheme, label in (("plic", "PLIC"), ("donor_cell", "donor cell")):
            m = r[scheme]
            add(
                f"| {r['cells']} | {label} | {100 * m['area_error']:.3f} % | "
                f"{m['l1']:.3e} | {m['linf']:.3f} | {m['peak_retention']:.3f} | "
                f"{m['distorted_fraction']:.3f} | {m['worst_step_relative_drift']:.2e} |"
            )
    add("")
    add("## 5. Criteria\n")
    for name, c in criteria.items():
        status = "PASS" if c.get("pass") else "FAIL"
        add(f"* **{name}: {status}** — {c['criterion']}")
        add(f"  * `{json.dumps({k: v for k, v in c.items() if k not in ('guards',)})}`")
    add("")
    add("## 6. Negative evidence\n")
    add("| case | rejected | message |\n|---|---|---|")
    for name, c in criteria["C4_negative_evidence"]["guards"].items():
        add(f"| {name} | {c['rejected']} | {c['message']} |")
    add("")
    add("## 7. Limitations / roadmap\n")
    add(
        "* Surface tension (CSF) and phase change are **out of scope** for this node; "
        "they are declared roadmap items: CSF is audited separately "
        "(`docs/two-phase-csf-balance`, `docs/two-phase-surface-tension-audit`), phase "
        "change has no module yet.\n"
        "* PLIC is 2-D on a uniform Cartesian grid; a 3-D general polygon-clipping "
        "reconstruction and an unstructured-mesh variant are not implemented.\n"
        "* Dimensional splitting is Lie (two sweeps, alternating order), so it is "
        "formally first order in time; a Strang split or an unsplit geometric flux is "
        "the next accuracy step.\n"
        "* Boundedness is a property of the geometric flux; the residual ~1e-14 "
        "excursions come from float roundoff in the plane-constant inversion.\n"
    )
    return "\n".join(lines) + "\n"


def run(output: str | Path, quick: bool = False) -> Dict[str, object]:
    """Run the whole verification surface and write artefacts into ``output``."""
    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    one_d_cells = [32, 64] if quick else [32, 64, 128]
    z_cells = [32, 64] if quick else [64, 128]
    revolutions = 0.25 if quick else 1.0

    one_d = {"resolutions": {}}
    for n in one_d_cells:
        one_d["resolutions"][str(n)] = one_d_step_case(n)
    z = {"resolutions": {}}
    for n in z_cells:
        z["resolutions"][str(n)] = zalesak_case(n, revolutions=revolutions)

    results: Dict[str, object] = {
        "parameters": {
            "one_d_cells": one_d_cells,
            "zalesak_cells": z_cells,
            "cfl": CFL,
            "one_d_shift": SHIFT,
            "zalesak_revolutions": revolutions,
            "mass_tolerance_per_step": MASS_TOL,
            "area_tolerance": AREA_TOL,
            "quick": quick,
        },
        "one_d_step": one_d,
        "zalesak": z,
        "negative_evidence": negative_evidence(),
    }
    criteria = _criteria(results)
    results["criteria"] = criteria
    results["figures"] = _write_figures(results, outdir)

    # JSON without the heavy fields
    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if not k.endswith("_field")}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    (outdir / "summary.json").write_text(json.dumps(_strip(results), indent=2) + "\n")
    (outdir / "report.md").write_text(_markdown(results, criteria))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/m4956-vof-plic-interface")
    parser.add_argument("--quick", action="store_true", help="small grids, 1/4 revolution")
    args = parser.parse_args(argv)
    torch.set_grad_enabled(False)
    results = run(args.output, quick=args.quick)
    passed = [k for k, v in results["criteria"].items() if v.get("pass")]
    failed = [k for k, v in results["criteria"].items() if not v.get("pass")]
    print(json.dumps({"pass": passed, "fail": failed}, indent=2))
    print(f"artefacts written to {args.output}")
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())