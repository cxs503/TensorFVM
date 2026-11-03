"""Verification-debt payment: Darcy--Forchheimer porous channel vs closed-form anchors.

Produces ``docs/porous-darcy-forchheimer/evidence.json`` plus the report figures.

Anchors (no external data required, all programmatically derivable):

1. ``porous=None`` reproduces ``SimpleSolver`` bit for bit (extension is inert).
2. Darcy--Brinkman limit (``beta=0``): the wall-normal velocity profile and the
   pressure gradient match the exact Brinkman channel solution
   ``u = A[1 - cosh((y-h)/l)/cosh(h/l)]``, ``dp/dx = -mu*A/K``.
3. ``K -> infinity`` (negligible resistance) recovers the plane-Poiseuille
   profile ``6 Ub y/H (1 - y/H)`` and its ``dp/dx = -12 mu Ub/H^2``.
4. Forchheimer (``beta > 0``): ``|dp/dx|`` is strictly increasing in ``beta`` and
   the central-region velocity obeys the bulk law ``-dp/dx = (mu/K)u + rho beta u^2``.

Run: ``PYTHONPATH=src python scripts/verify_porous_drag.py``
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.porous_drag import (PorousChannelSolver, PorousMedia,  # noqa: E402
                                   brinkman_profile, brinkman_pressure_gradient,
                                   forchheimer_bulk_velocity)
from tensorfvm.solver import SimpleSolver, SolverConfig  # noqa: E402

OUT = ROOT / "docs" / "porous-darcy-forchheimer"

# Channel geometry shared by every case: H = 20 mm, L = 6 H.
HEIGHT, LENGTH = 0.02, 0.12
INLET = 5e-4          # m/s
RE = 1.0              # -> mu = rho Ub H / Re = 1e-2 Pa s
RHO = 1000.0


def config(ny=16) -> SolverConfig:
    return SolverConfig(nx=3 * ny, ny=ny, length=LENGTH, height=HEIGHT,
                        inlet_velocity=INLET, density=RHO, reynolds=RE,
                        cylinder_radius=None, tolerance=1e-8,
                        max_iterations=4000, velocity_relaxation=0.7)


def measure(result, cfg):
    """Cell-centred u(y) in the fully developed window and its dp/dx."""
    u, _ = result.cell_center_velocity()
    u, p = u.numpy(), result.p.numpy()
    x, y = result.x.numpy(), result.y.numpy()
    region = (x >= 2.0 * HEIGHT) & (x <= 0.98 * cfg.length)
    profile = u[:, region].mean(axis=1)
    px = p[:, region].mean(axis=0)
    xr = x[region] - x[region].mean()
    slope = float(np.dot(xr, px) / np.dot(xr, xr))
    return dict(y=y, u=profile, dpdx=slope, x_window=[float(x[region].min()),
                                                     float(x[region].max())])


def darcy_brinkman_case(ny=16, permeability=1e-5, form_drag=0.0):
    cfg = config(ny)
    media = PorousMedia(permeability_m2=permeability, form_drag_1_per_m=form_drag)
    started = time.perf_counter()
    solver = PorousChannelSolver(cfg, media)
    result = solver.solve()
    elapsed = time.perf_counter() - started
    data = measure(result, cfg)
    mu = cfg.viscosity
    exact_u = brinkman_profile(torch.as_tensor(data["y"]), INLET, HEIGHT, permeability).numpy()
    exact_dpdx = brinkman_pressure_gradient(INLET, HEIGHT, permeability, mu)
    peak = float(np.max(np.abs(exact_u)))
    profile_l2 = float(np.sqrt(np.mean((data["u"] - exact_u) ** 2)) / peak)
    profile_linf = float(np.max(np.abs(data["u"] - exact_u)) / peak)
    dpdx_error = abs(data["dpdx"] - exact_dpdx) / abs(exact_dpdx)
    h = 0.5 * HEIGHT
    bulk = np.abs(data["y"] - h) <= 0.5 * h
    u_bulk = float(data["u"][bulk].mean())
    forchheimer_prediction = forchheimer_bulk_velocity(
        data["dpdx"], mu, RHO, permeability, form_drag)
    law_residual = abs(u_bulk - forchheimer_prediction) / forchheimer_prediction
    return dict(
        permeability_m2=permeability, form_drag_1_per_m=form_drag,
        grid=f"{cfg.nx}x{cfg.ny}", viscosity_pa_s=mu, converged=bool(result.converged),
        iterations=len(result.history), elapsed_s=elapsed,
        boundaries_s_inv=1.0 / math.sqrt(permeability), half_height_over_sqrt_K=h / math.sqrt(permeability),
        profile_l2_relative_error=profile_l2, profile_linf_relative_error=profile_linf,
        dpdx_simulated=data["dpdx"], dpdx_exact=exact_dpdx, dpdx_relative_error=dpdx_error,
        mean_velocity=float(np.mean(data["u"])), bulk_velocity=u_bulk,
        forchheimer_bulk_law_prediction=forchheimer_prediction,
        forchheimer_bulk_law_residual=law_residual,
        fields=dict(y_m=data["y"].tolist(), u_sim=data["u"].tolist(),
                    u_exact=exact_u.tolist()),
        residual=result.history[-1] if result.history else {})


def inert_extension_check(ny=8):
    """``porous=None`` must reproduce the production solver exactly."""
    cfg = config(ny)
    cfg.max_iterations = 60
    plain = SimpleSolver(cfg).solve()
    wrapped = PorousChannelSolver(cfg, None).solve()
    return dict(max_abs_u=float((plain.u - wrapped.u).abs().max()),
                max_abs_v=float((plain.v - wrapped.v).abs().max()),
                max_abs_p=float((plain.p - wrapped.p).abs().max()),
                bitwise_identical=bool(torch.equal(plain.u, wrapped.u)
                                       and torch.equal(plain.v, wrapped.v)
                                       and torch.equal(plain.p, wrapped.p)))


def no_resistance_limit(ny=16, permeability=1.0):
    """K -> infinity: the sink vanishes and Poiseuille flow must be recovered."""
    cfg = config(ny)
    result = PorousChannelSolver(cfg, PorousMedia(permeability_m2=permeability)).solve()
    data = measure(result, cfg)
    exact = 6.0 * INLET * data["y"] / HEIGHT * (1.0 - data["y"] / HEIGHT)
    exact_dpdx = -12.0 * cfg.viscosity * INLET / HEIGHT ** 2
    peak = float(np.max(np.abs(exact)))
    return dict(permeability_m2=permeability, poiseuille_profile_l2_relative_error=float(
        np.sqrt(np.mean((data["u"] - exact) ** 2)) / peak),
        poiseuille_dpdx_relative_error=abs(data["dpdx"] - exact_dpdx) / abs(exact_dpdx),
        dpdx_simulated=data["dpdx"], dpdx_exact=exact_dpdx)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = dict(schema="tensorfvm.porous-darcy-forchheimer/1",
                    geometry_m=dict(length=LENGTH, height=HEIGHT),
                    inlet_velocity_m_s=INLET, density_kg_m3=RHO, reynolds=RE)
    print("inert extension ...", flush=True)
    evidence["inert_extension"] = inert_extension_check()
    print("  bitwise_identical:", evidence["inert_extension"]["bitwise_identical"], flush=True)

    print("no-resistance limit ...", flush=True)
    evidence["no_resistance_limit"] = no_resistance_limit()

    cases = []
    for permeability in (1e-5, 4e-6, 1e-6):
        print(f"Darcy-Brinkman K={permeability:g} ...", flush=True)
        case = darcy_brinkman_case(permeability=permeability)
        print(f"  converged={case['converged']} iters={case['iterations']} "
              f"L2={case['profile_l2_relative_error']:.3e} "
              f"dpdx={case['dpdx_relative_error']:.3e} t={case['elapsed_s']:.1f}s", flush=True)
        cases.append(case)
    evidence["darcy_brinkman_cases"] = cases

    forchheimer = []
    for beta in (0.0, 5e3, 2e4, 8e4):
        print(f"Forchheimer beta={beta:g} ...", flush=True)
        case = darcy_brinkman_case(permeability=1e-6, form_drag=beta)
        print(f"  dpdx={case['dpdx_simulated']:.1f} Pa/m bulk_law_residual="
              f"{case['forchheimer_bulk_law_residual']:.3%}", flush=True)
        forchheimer.append(case)
    evidence["forchheimer_cases"] = forchheimer
    magnitudes = [abs(c["dpdx_simulated"]) for c in forchheimer]
    evidence["forchheimer_monotone"] = bool(
        all(b > a for a, b in zip(magnitudes, magnitudes[1:])))
    evidence["forchheimer_dpdx_magnitudes"] = magnitudes
    evidence["forchheimer_bulk_law_residual"] = [c["forchheimer_bulk_law_residual"]
                                                 for c in forchheimer]

    worst_profile = max(c["profile_l2_relative_error"] for c in cases)
    worst_dpdx = max(c["dpdx_relative_error"] for c in cases)
    evidence["acceptance"] = dict(
        inert_extension=evidence["inert_extension"]["bitwise_identical"],
        poiseuille_profile_under_3pct=evidence["no_resistance_limit"][
            "poiseuille_profile_l2_relative_error"] < 0.03,
        poiseuille_dpdx_under_3pct=evidence["no_resistance_limit"][
            "poiseuille_dpdx_relative_error"] < 0.03,
        brinkman_profile_under_3pct=worst_profile < 0.03,
        brinkman_dpdx_under_3pct=worst_dpdx < 0.03,
        forchheimer_monotone=evidence["forchheimer_monotone"],
    )
    evidence["acceptance"]["all_passed"] = all(evidence["acceptance"].values())
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=1))
    figures(evidence)
    print(json.dumps({k: v for k, v in evidence["acceptance"].items()}, indent=1))
    print("worst brinkman profile L2 =", f"{worst_profile:.3e}",
          "worst dpdx =", f"{worst_dpdx:.3e}")
    return 0 if evidence["acceptance"]["all_passed"] else 2


def figures(evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    marks = ("o", "s", "^")
    for case, mark in zip(evidence["darcy_brinkman_cases"], marks):
        y, sim = np.array(case["fields"]["y_m"]), np.array(case["fields"]["u_sim"])
        exact = np.array(case["fields"]["u_exact"])
        ax.plot(sim * 1e3, y * 1e3, mark, ms=3.2, mfc="none",
                label=f"FVM $K$={case['permeability_m2']:.0e} m$^2$")
        ax.plot(exact * 1e3, y * 1e3, "-", lw=1.0)
    ax.set_xlabel("$u$ [mm/s]"); ax.set_ylabel("$y$ [mm]")
    ax.set_title("Darcy–Brinkman channel: FVM (symbols) vs exact (lines)")
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout()
    fig.savefig(OUT / "brinkman-profiles.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ks = np.array([c["permeability_m2"] for c in evidence["darcy_brinkman_cases"]])
    sim = np.array([abs(c["dpdx_simulated"]) for c in evidence["darcy_brinkman_cases"]])
    exact = np.array([abs(c["dpdx_exact"]) for c in evidence["darcy_brinkman_cases"]])
    ax.loglog(ks, exact, "-", lw=1.0, label="exact $\\mu A/K$")
    ax.loglog(ks, sim, "o", ms=5, mfc="none", label="FVM")
    for k, s, e in zip(ks, sim, exact):
        ax.annotate(f"{abs(s-e)/e:.1%}", (k, s), textcoords="offset points",
                    xytext=(6, -10), fontsize=7)
    ax.set_xlabel("$K$ [m$^2$]"); ax.set_ylabel("$|\\mathrm{d}p/\\mathrm{d}x|$ [Pa/m]")
    ax.set_title("Darcy pressure gradient: simulation vs exact"); ax.legend(fontsize=8)
    ax.invert_xaxis(); fig.tight_layout()
    fig.savefig(OUT / "darcy-pressure-gradient.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    betas = np.array([c["form_drag_1_per_m"] for c in evidence["forchheimer_cases"]])
    mags = np.array([abs(c["dpdx_simulated"]) for c in evidence["forchheimer_cases"]])
    u_bar = evidence["forchheimer_cases"][0]["mean_velocity"]
    mu = evidence["forchheimer_cases"][0]["viscosity_pa_s"]
    K = evidence["forchheimer_cases"][0]["permeability_m2"]
    law = mu / K * u_bar + RHO * betas * u_bar ** 2
    ax.plot(betas, law, "-", lw=1.0, label="Forchheimer law (mean $u$)")
    ax.plot(betas, mags, "s", ms=5, mfc="none", label="FVM")
    ax.set_xlabel("$\\beta$ [1/m]"); ax.set_ylabel("$|\\mathrm{d}p/\\mathrm{d}x|$ [Pa/m]")
    ax.set_title("Forchheimer regime: $K=10^{-6}$ m$^2$"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "forchheimer-law.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())