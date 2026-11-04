#!/usr/bin/env python3
"""Audit the transient (Chorin projection) path of ``TwoPhaseSolver``.

Read-only numerical audit of three claimed defects plus one fixed gate:

A. the documented analytic anchor ``initialize_hydrostatic_pressure`` is
   discontinuous at the interface (the liquid branch omits the gas-column
   offset rho_g*g*(H-h_i));
B. the projection solves for the *total* pressure from div(u*)/dt and drops
   the carried p^n from the predictor, so a divergence-free free fall is an
   admissible solution: a resting hydrostatic column accelerates;
C. ``surface_tension`` never enters the momentum predictor, so a static drop
   carries no Young-Laplace jump;
D. sealed domains (all-Neumann pressure Poisson) used to run silently; they
   now raise ``ValueError`` (negative control, fixed).

Writes ``evidence.json``, two PNG figures and ``report.md`` next to this file
(or into --out).  Nothing in ``src/`` is modified.
"""
import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from tensorfvm.two_phase_solver import (  # noqa: E402
    GridConfig, TwoPhaseConfig, TwoPhaseSolver)

RHO_L, RHO_G, G = 1000.0, 1.0, 9.81


def build(n=16, gravity=G, sigma=0.0, surface_tension=False,
          top_atmosphere=True, mu=1e-3, drop_radius=None):
    tp = TwoPhaseConfig(enabled=True, rho_liquid=RHO_L, rho_gas=RHO_G,
                        mu_liquid=mu, mu_gas=mu, gravity=gravity,
                        sigma=sigma, surface_tension=surface_tension,
                        top_atmosphere=top_atmosphere)
    solver = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dx
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dy
    if drop_radius is None:
        yy = y.reshape(-1, 1).expand(n, n)
        solver.set_alpha(torch.where(yy < 0.5, torch.ones_like(yy),
                                     torch.zeros_like(yy)))
        solver.initialize_hydrostatic_pressure(0.5)
        return solver, None
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (y[:, None] - 0.5) ** 2)
    alpha = torch.clamp((drop_radius - rr) / max(solver.dx, solver.dy) + 0.5, 0.0, 1.0)
    solver.set_alpha(alpha)
    return solver, rr


# ---------------------------------------------------------------- A: anchor
def audit_anchor(n=64):
    solver, _ = build(n=n)
    p = solver.p
    # cell rows straddling the interface face y = 0.5 (face between n/2-1, n/2)
    jlo, jhi = n // 2 - 1, n // 2
    jump = float(p[jhi, 0] - p[jlo, 0])
    # face-consistent value for a continuous profile: -rho_face*g*dy
    rho_face = 0.5 * (RHO_L + RHO_G)
    expected = -rho_face * G * solver.dy
    exact_jump = RHO_G * G * (1.0 - 0.5)     # gas-column offset, Pa
    # residual of the discrete hydrostatic balance, per v-face, at the interface
    residual = solver.hydrostatic_balance_residual()
    dpdy = torch.zeros_like(solver.v)
    dpdy[1:-1, :] = (p[1:, :] - p[:-1, :]) / solver.dy
    rfield = (dpdy + solver._rho_face_v() * G)[:, 0]
    return {
        "max_residual_after_init_Pa_per_m": residual["max_residual"],
        "rms_residual_after_init_Pa_per_m": residual["rms_residual"],
        "interface_dp_dy_Pa_per_m": float(dpdy[jhi, 0]),
        "interface_balance_required_Pa_per_m": float(-rho_face * G),
        "interface_jump_Pa": jump,
        "interface_jump_expected_Pa": expected,
        "missing_gas_offset_Pa": exact_jump,
        "profile_continuous": abs(jump - expected) < 1e-9,
        "residual_profile_col0": [float(v) for v in rfield],
    }


# ------------------------------------------------------- B: resting column
def audit_resting_column(steps=10, dt=1e-5, sizes=(16, 32, 64)):
    out = {}
    for n in sizes:
        solver, _ = build(n=n)
        hist = []
        for _ in range(steps):
            solver.step(dt)
            hist.append({"v_max": float(solver.v.abs().max()),
                         "u_max": float(solver.u.abs().max())})
        out[str(n)] = hist
    return {"dt": dt, "steps": steps, "history": out}


# --------------------------------------------------------------- C: Laplace
def audit_laplace(n=32, radius=0.2, sigma=0.07, steps=10, dt=1e-5, mu=1e-2):
    solver, rr = build(n=n, gravity=0.0, sigma=sigma, surface_tension=True,
                       mu=mu, drop_radius=radius)
    for _ in range(steps):
        solver.step(dt)
    inside = float(solver.p[rr < 0.5 * radius].mean())
    outside = float(solver.p[rr > 1.6 * radius].mean())
    dp = inside - outside
    exact = sigma / radius
    return {"dp_measured_Pa": dp, "dp_exact_Pa": exact,
            "relative_error": abs(dp - exact) / exact,
            "u_max": float(solver.u.abs().max())}


# ---------------------------------------------------- D: sealed-domain gate
def audit_sealed_domain(**kw):
    solver, _ = build(top_atmosphere=False, **kw)
    try:
        solver.step(1e-5)
        return {"raised_ValueError": False}
    except ValueError as exc:
        return {"raised_ValueError": True, "message": str(exc)}


def plot(anchor, rest, laplace, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for n, hist in rest["history"].items():
        v = [h["v_max"] for h in hist]
        ax.semilogy(range(1, len(v) + 1), v, marker="o", label=f"n={n}")
    ax.set_xlabel("step (dt = 1e-5 s, gravity 9.81)")
    ax.set_ylabel(r"$\max|v|$  [m/s]")
    ax.set_title("B  resting column: velocity does not stay at rest")
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    y = (torch.arange(len(anchor["residual_profile_col0"]), dtype=torch.float64)
         + 0.5)[: len(anchor["residual_profile_col0"])]
    prof = anchor["residual_profile_col0"]
    ax.plot(prof, range(len(prof)), marker="o", ms=3)
    ax.axvline(0.0, color="k", lw=.8)
    ax.set_xlabel(r"$dp/dy + \rho_{\rm face} g$  [Pa/m]")
    ax.set_ylabel("v-face index (0 = bottom wall)")
    ax.set_title("A  anchor: interface mismatch of "
                 f"{anchor['interface_jump_Pa'] - anchor['interface_jump_expected_Pa']:.2f} Pa split")

    ax = axes[2]
    ax.bar(["CSF in predictor", "exact $\\sigma/R$"],
           [laplace["dp_measured_Pa"], laplace["dp_exact_Pa"]],
           color=["tab:red", "tab:green"])
    ax.set_ylabel("pressure jump [Pa]")
    ax.set_title("C  static drop, $\\sigma/R$ = %.3f Pa" % laplace["dp_exact_Pa"])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).with_suffix("")))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    anchor = audit_anchor()
    rest = audit_resting_column()
    laplace = audit_laplace()
    sealed = audit_sealed_domain()

    growth = {}
    for n, hist in rest["history"].items():
        growth[n] = hist[-1]["v_max"] / max(hist[0]["v_max"], 1e-300)
    evidence = {"anchor": anchor, "resting_column": rest,
                "resting_growth_factor": growth, "laplace": laplace,
                "sealed_domain": sealed,
                "verdicts": {
                    "A_anchor_continuous": anchor["profile_continuous"],
                    "B_resting_column_stays_at_rest":
                        all(hist[-1]["v_max"] < 1e-9
                            for hist in rest["history"].values()),
                    "C_laplace_within_3pct":
                        laplace["relative_error"] < 0.03,
                    "D_sealed_domain_rejected": sealed["raised_ValueError"],
                }}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    plot(anchor, rest, laplace, out / "two_phase_projection_audit.png")

    lines = [
        "# Two-phase transient projection audit",
        "",
        "| 项 | 判据 | 实测 | 判定 |",
        "|---|---|---|---|",
        f"| A 静水锚点连续 | 界面跳变 = ρ_face·g·dy = "
        f"{anchor['interface_jump_expected_Pa']:.2f} Pa | "
        f"{anchor['interface_jump_Pa']:.2f} Pa (缺 {anchor['missing_gas_offset_Pa']:.2f} Pa) | "
        f"{'PASS' if anchor['profile_continuous'] else 'FAIL'} |",
        f"| B 静止分层柱保持静止 | max\\|v\\|<1e-9 | "
        + ", ".join(f"n={n}: {hist[-1]['v_max']:.2e}" for n, hist in rest["history"].items())
        + f" | {'PASS' if evidence['verdicts']['B_resting_column_stays_at_rest'] else 'FAIL'} |",
        f"| C Young-Laplace 压差 σ/R | 误差 <3% (σ/R={laplace['dp_exact_Pa']:.3f} Pa) | "
        f"{laplace['dp_measured_Pa']:.3e} Pa ({100 * laplace['relative_error']:.2f}%) | "
        f"{'PASS' if evidence['verdicts']['C_laplace_within_3pct'] else 'FAIL'} |",
        f"| D 全封闭域 fail-fast | 抛 ValueError | "
        f"{'抛 ValueError' if sealed['raised_ValueError'] else '静默发散'} | "
        f"{'PASS' if sealed['raised_ValueError'] else 'FAIL'} |",
    ]
    (out / "report-table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: evidence["verdicts"][k] for k in evidence["verdicts"]}, indent=2))
    print("growth", growth)


if __name__ == "__main__":
    main()
