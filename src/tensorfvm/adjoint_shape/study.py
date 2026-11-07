"""Verification + optimisation driver for the discrete-adjoint shape sensitivity.

Produces ``docs/adjoint-shape-sensitivity/{result.json, figures/*.png}`` and prints
the acceptance numbers, so the report can quote numbers that were *programmatically
injected* rather than transcribed.

Acceptance criteria (see the mission sheet):

(a) Taylor remainder slope in ``[1.7, 2.2]`` on a log-log ladder;
(b) adjoint gradient vs per-parameter central differences, relative error <= 5 %
    for every one of the eight parameters;
(c) adjoint gradient relative change over three grids <= 10 % (GCI style, L2);
(d) 20 L-BFGS-B steps reduce ``C_D`` by >= 3 %, reproduced on a refined grid to
    within 20 %;
(e) path equivalence: ``C_D`` from the direct factorisation path equals the
    GMRES/ILU path to <= 1e-10 (relative), plus the analytic plane-Poiseuille
    benchmark with the expected second-order mesh convergence;
(f) pytest suite ``tests/test_adjoint_shape_sensitivity.py`` (>= 12 cases).

Run as ``python -m tensorfvm.adjoint_shape.study [--out DIR]``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import scipy

from . import __version__
from .adjoint import (
    central_difference_gradient,
    gradient_grid_convergence,
    taylor_remainder,
)
from .channel import ChannelGrid, ChannelSolver, poiseuille
from .geometry import N_PARAMS, PARAM_NAMES, FourierBody
from .optimize import DEFAULT_BOUND, drag_reduction, evaluate_shape

#: grid ladder (coarse -> fine); the last one is the "refined" grid of criterion (d)
GRID_LADDER = ((64, 32), (96, 48), (128, 64))

#: fixed study configuration: half-diameter body, off-centre so that no shape
#: parameter is structurally inactive (a centred body makes the sin modes exactly
#: insensitive, which would make a *relative* gradient error ill conditioned)
CONFIG = {
    "radius0": 0.25,
    "center": (2.0, 0.85),
    "reynolds_diameter": 20.0,
    "eta": 1.0e-3,
    "delta_cells": 0.3,
    "subcell_quad": 4,
    "subcell_cells": 1.0,
    "eps_upwind": 0.05,
    "channel_length": 6.0,
    "channel_height": 2.0,
    "bound": DEFAULT_BOUND,
}


def jsonable(obj):
    """Recursively convert numpy containers/scalars so ``json.dump`` cannot fail."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def make_solver(nx: int, ny: int, with_body: bool = True) -> ChannelSolver:
    """Solver on the study configuration (``nu = U D / Re`` at ``U = 1``)."""
    grid = ChannelGrid(nx=nx, ny=ny, length=CONFIG["channel_length"],
                       height=CONFIG["channel_height"])
    nu = 2.0 * CONFIG["radius0"] * 1.0 / CONFIG["reynolds_diameter"]
    force = 12.0 * nu * 1.0 / grid.height**2
    body = None
    if with_body:
        body = FourierBody(
            center=CONFIG["center"],
            radius0=CONFIG["radius0"],
            delta=CONFIG["delta_cells"] * min(grid.dx, grid.dy),
            quad=CONFIG["subcell_quad"],
            subcell=CONFIG["subcell_cells"] * min(grid.dx, grid.dy),
        )
    return ChannelSolver(
        grid,
        body,
        nu=nu,
        force=force,
        eta=CONFIG["eta"],
        eps_upwind=CONFIG["eps_upwind"],
    )


def base_point(scale: float = 0.02, seed: int = 7) -> np.ndarray:
    """Small, asymmetric base shape -- keeps the Taylor test out of the symmetric point."""
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(N_PARAMS)


# --------------------------------------------------------------------------
def poiseuille_check(nx, ny):
    """Analytic plane-Poiseuille benchmark (body-free) and its mesh convergence."""
    solver = make_solver(nx, ny, with_body=False)
    theta = np.zeros(N_PARAMS)
    U, info = solver.solve(theta)
    g = solver.grid
    Uu = U[: g.nu].reshape(g.ny, g.nx)
    exact = np.array(
        [poiseuille(g.y_u(j), g.height, solver.force, solver.nu) for j in range(g.ny)]
    )
    err = float(np.abs(Uu - exact[:, None]).max())
    u_m = solver.bulk_velocity(U)
    return {
        "grid": [nx, ny],
        "max_abs_error": err,
        "bulk_velocity": float(u_m),
        "bulk_velocity_error": float(abs(u_m - 1.0)),
        "residual": info["relative_residual"],
        "newton_iterations": info["iterations"],
    }


def path_equivalence(nx, ny, theta):
    """Criterion (e): direct factorisation vs GMRES/ILU on the *same* discrete system."""
    solver = make_solver(nx, ny)
    U_direct, info = solver.solve(theta)
    cd_direct = solver.drag_coefficient(theta, U_direct)
    U_gmres, gmres_info = solver.solve_reference(theta, U0=U_direct)
    cd_gmres = solver.drag_coefficient(theta, U_gmres)
    rel = abs(cd_gmres - cd_direct) / abs(cd_direct)
    return {
        "grid": [nx, ny],
        "C_D_direct": cd_direct,
        "C_D_gmres": cd_gmres,
        "relative_difference": float(rel),
        "gmres_flag": gmres_info["gmres_flag"],
        "gmres_residual": gmres_info["residual_norm"],
        "newton_relative_residual": info["relative_residual"],
    }


def normalisation_audit(nx, ny, theta):
    """Check the sensitivity path against a *global* momentum budget.

    ``F_drive = F_body + F_wall`` must hold discretely; the deficit is the audit
    margin of the drag used in the gradient.
    """
    solver = make_solver(nx, ny)
    U, info = solver.solve(theta)
    bal = solver.momentum_balance(theta, U)
    return {
        "grid": [nx, ny],
        **{k: float(v) for k, v in bal.items()},
        "relative_budget_defect": float(
            abs(bal["residual"]) / max(bal["F_drive"], 1e-300)
        ),
        "newton_relative_residual": info["relative_residual"],
        "bulk_velocity": float(solver.bulk_velocity(U)),
        "C_D": float(solver.drag_coefficient(theta, U)),
    }


# --------------------------------------------------------------------------
def make_figures(outdir: Path, payload) -> list[str]:
    """Field plots, geometry, and the verification curves (PNG, small on purpose)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    names = []

    # --- geometry: nominal, perturbed, optimised -------------------------
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=110)
    phi = np.linspace(0.0, 2.0 * np.pi, 361)
    body = FourierBody(center=CONFIG["center"], radius0=CONFIG["radius0"], delta=0.01)
    shapes = [
        ("nominal", np.zeros(N_PARAMS), "k-"),
        ("start shape", np.asarray(payload["optimisation"]["theta0"]), "b--"),
        ("optimised", np.asarray(payload["optimisation"]["theta_opt"]), "r-"),
    ]
    for label, theta, style in shapes:
        r = body.radius(CONFIG["center"][0] + np.cos(phi), CONFIG["center"][1] + np.sin(phi), theta)
        ax.plot(r * np.cos(phi), r * np.sin(phi), style, lw=1.4, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("body shapes (x/y relative to centre)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = figdir / "shapes.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    names.append(str(path))

    # --- flow field: nominal vs optimised -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=110, sharey=True)
    for ax, key, title in zip(
        axes, ("field_nominal", "field_optimised"), ("nominal shape", "optimised shape")
    ):
        fld = payload[key]
        speed = np.asarray(fld["speed"])  # (ny, nx) cell centred
        yy = np.asarray(fld["y_cell"])
        xx = np.asarray(fld["x_cell"])
        im = ax.pcolormesh(xx, yy, speed, shading="nearest", cmap="viridis")
        ax.plot(
            np.cos(np.linspace(0, 2 * np.pi, 200)) * CONFIG["radius0"] + CONFIG["center"][0],
            np.sin(np.linspace(0, 2 * np.pi, 200)) * CONFIG["radius0"] + CONFIG["center"][1],
            "w--",
            lw=0.8,
        )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|u|")
    fig.tight_layout()
    path = figdir / "flow_field.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    names.append(str(path))

    # --- Taylor remainder + gradient comparison -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=110)
    ax = axes[0]
    rows = payload["taylor"]["rows"]
    eps = np.array([r["epsilon"] for r in rows])
    rem = np.array([r["remainder"] for r in rows])
    ax.loglog(eps, rem, "o-", label="remainder")
    ref = rem[0] * (eps / eps[0]) ** 2
    ax.loglog(eps, ref, "k--", lw=0.9, label=r"$\propto\epsilon^{2}$")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$|J(\theta+\epsilon d)-J-\epsilon\,\nabla J\cdot d|$")
    ax.set_title(f"Taylor test, slope = {payload['taylor']['slope']:.2f}", fontsize=8)
    ax.legend(fontsize=7)
    ax = axes[1]
    grad = np.asarray(payload["central_difference"]["adjoint"])
    best_eps = str(payload["central_difference"]["best_epsilon"])
    fd = np.asarray(payload["central_difference"]["by_epsilon"][best_eps]["fd"])
    idx = np.arange(N_PARAMS)
    ax.bar(idx - 0.2, grad, 0.4, label="adjoint")
    ax.bar(idx + 0.2, fd, 0.4, label="central diff.")
    ax.set_xticks(idx)
    ax.set_xticklabels(PARAM_NAMES, rotation=45, fontsize=6)
    ax.set_ylabel(r"$\partial C_D/\partial\theta_q$")
    ax.set_title(f"gradients (eps={best_eps})", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = figdir / "gradient_check.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    names.append(str(path))

    # --- optimisation + grid convergence --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=110)
    ax = axes[0]
    hist = payload["optimisation"]["history"]["J"]
    ax.plot(np.arange(len(hist)), hist, "o-")
    ax.set_xlabel("adjoint evaluation")
    ax.set_ylabel(r"$C_D$")
    ax.set_title(f"L-BFGS-B, reduction {100 * payload['optimisation']['reduction']:.2f}%", fontsize=8)
    ax = axes[1]
    g = np.asarray(payload["grid_convergence"]["gradients"])
    for q in range(N_PARAMS):
        ax.plot(np.arange(g.shape[0]), g[:, q], "o-", label=PARAM_NAMES[q], lw=1.0)
    ax.set_xlabel("grid (coarse -> fine)")
    ax.set_ylabel(r"$\partial C_D/\partial\theta_q$")
    ax.set_title("gradient mesh convergence", fontsize=8)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = figdir / "optimisation_and_convergence.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    names.append(str(path))
    return names


def field_snapshot(nx, ny, theta, U=None):
    """Cell-centred speed for the field figure."""
    solver = make_solver(nx, ny)
    if U is None:
        U, _ = solver.solve(theta)
    g = solver.grid
    Uu = U[: g.nu].reshape(g.ny, g.nx)
    Uv = U[g.nu : g.nu + g.nv].reshape(g.ny - 1, g.nx)
    uc = 0.5 * (Uu + np.roll(Uu, -1, axis=1))
    vc = np.zeros_like(uc)
    vc[1:-1, :] = 0.5 * (Uv[:-1, :] + Uv[1:, :])
    speed = np.hypot(uc, vc)
    return {
        "speed": speed.tolist(),
        "x_cell": ((np.arange(g.nx) + 0.5) * g.dx).tolist(),
        "y_cell": ((np.arange(g.ny) + 0.5) * g.dy).tolist(),
        "C_D": float(solver.drag_coefficient(theta, U)),
        "bulk_velocity": float(solver.bulk_velocity(U)),
    }


# --------------------------------------------------------------------------
def run(outdir: Path, jar: bool = True) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    theta0 = base_point()
    results: dict = {
        "mission": 5597,
        "repo": "TensorFVM",
        "module": "src/tensorfvm/adjoint_shape",
        "version": __version__,
        "config": CONFIG,
        "grid_ladder": [list(g) for g in GRID_LADDER],
        "base_point": theta0.tolist(),
        "param_names": list(PARAM_NAMES),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    # ---------------- (a) Taylor remainder ------------------------------
    coarse = make_solver(*GRID_LADDER[0])
    tay = taylor_remainder(
        coarse, theta0, epsilons=10.0 ** np.arange(-5.0, -0.9, 0.5)
    )
    results["taylor"] = {
        "slope": float(tay["slope"]),
        "rows": tay["rows"],
        "direction": tay["direction"].tolist(),
        "gradient": tay["gradient"].tolist(),
        "acceptance": bool(1.7 <= tay["slope"] <= 2.2),
        "grid": list(GRID_LADDER[0]),
    }
    print(f"(a) Taylor slope = {tay['slope']:.3f}")

    # ---------------- (b) central differences ---------------------------
    cd = central_difference_gradient(
        coarse, theta0, epsilons=(1.0e-4, 1.0e-3, 1.0e-2)
    )
    per_eps = {str(k): {"fd": v["fd"].tolist(),
                        "rel_error": v["rel_error"].tolist(),
                        "max_rel_error": v["max_rel_error"],
                        "mean_rel_error": v["mean_rel_error"]}
               for k, v in cd["by_epsilon"].items()}
    best_eps = min(cd["by_epsilon"].items(),
                   key=lambda kv: kv[1]["max_rel_error"])[0]
    results["central_difference"] = {
        "adjoint": cd["adjoint"].tolist(),
        "by_epsilon": per_eps,
        "best_epsilon": float(best_eps),
        "max_rel_error": per_eps[str(best_eps)]["max_rel_error"],
        "acceptance": bool(per_eps[str(best_eps)]["max_rel_error"] <= 0.05),
        "grid": list(GRID_LADDER[0]),
    }
    print(f"(b) max relative gradient error = "
          f"{per_eps[str(best_eps)]['max_rel_error']:.3e} (eps={best_eps})")

    # ---------------- (c) grid convergence ------------------------------
    solvers = [make_solver(*g) for g in GRID_LADDER]
    gc = gradient_grid_convergence(solvers, np.zeros(N_PARAMS))
    results["grid_convergence"] = {
        "gradients": [g.tolist() for g in gc["gradients"]],
        "relative_change": gc["relative_change"],
        "max_relative_change": float(max(gc["relative_change"])),
        "acceptance": bool(max(gc["relative_change"]) <= 0.10),
        "note": "gradients evaluated at the nominal shape theta = 0",
    }
    print(f"(c) gradient relative change {gc['relative_change']}")

    # ---------------- (d) drag-reduction loop ---------------------------
    start_shape = np.zeros(N_PARAMS)
    start_shape[0] = -0.09  # vertically stretched: a deliberately drag-heavy start
    opt_solver = make_solver(*GRID_LADDER[1])
    opt = drag_reduction(opt_solver, theta0=start_shape, maxiter=20,
                         bound=CONFIG["bound"])
    j_opt_fine, _, _ = evaluate_shape(make_solver(*GRID_LADDER[2]), opt["theta_opt"])
    j0_fine, _, _ = evaluate_shape(make_solver(*GRID_LADDER[2]), start_shape)
    red_coarse = opt["reduction"]
    red_fine = (j0_fine - j_opt_fine) / j0_fine
    results["optimisation"] = {
        "theta0": opt["theta0"].tolist(),
        "theta_opt": opt["theta_opt"].tolist(),
        "J0": opt["J0"],
        "J_final": opt["J_final"],
        "reduction": float(red_coarse),
        "history": opt["history"],
        "n_iterations": opt["n_iterations"],
        "grid": list(GRID_LADDER[1]),
        "refined_grid": list(GRID_LADDER[2]),
        "J0_refined": float(j0_fine),
        "J_opt_refined": float(j_opt_fine),
        "reduction_refined": float(red_fine),
        "reduction_relative_difference": float(
            abs(red_fine - red_coarse) / max(abs(red_coarse), 1e-300)
        ),
        "acceptance": bool(
            red_coarse >= 0.03
            and abs(red_fine - red_coarse) / max(abs(red_coarse), 1e-300) <= 0.20
        ),
    }
    print(f"(d) reduction {100 * red_coarse:.2f}% (refined {100 * red_fine:.2f}%)")

    # ---------------- (e) path equivalence + Poiseuille -----------------
    pe = path_equivalence(*GRID_LADDER[1], np.zeros(N_PARAMS))
    pois = [poiseuille_check(*g) for g in GRID_LADDER]
    orders = []
    for a, b in zip(pois[:-1], pois[1:]):
        h1 = CONFIG["channel_height"] / a["grid"][1]
        h2 = CONFIG["channel_height"] / b["grid"][1]
        if b["max_abs_error"] > 0.0 and a["max_abs_error"] > 0.0:
            orders.append(float(np.log(a["max_abs_error"] / b["max_abs_error"])
                                / np.log(h1 / h2)))
    results["path_equivalence"] = {
        **pe,
        "acceptance": bool(pe["relative_difference"] <= 1.0e-10),
    }
    results["poiseuille"] = {"cases": pois, "observed_orders": orders}
    print(f"(e) path-equivalence rel diff = {pe['relative_difference']:.3e}; "
          f"Poiseuille orders {['%.2f' % o for o in orders]}")

    # ---------------- normalisation audit -------------------------------
    results["momentum_audit"] = normalisation_audit(*GRID_LADDER[1], np.zeros(N_PARAMS))

    # ---------------- field snapshots -----------------------------------
    results["field_nominal"] = field_snapshot(*GRID_LADDER[1], np.zeros(N_PARAMS))
    results["field_optimised"] = field_snapshot(
        *GRID_LADDER[1], np.asarray(opt["theta_opt"])
    )

    # ---------------- figures -------------------------------------------
    figs = []
    if jar:
        try:
            figs = make_figures(outdir, results)
        except Exception as exc:  # pragma: no cover - plotting must not hide results
            results["figure_error"] = repr(exc)
    results["figures"] = figs
    results["wall_seconds"] = time.time() - t_start

    # ---------------- verdict -------------------------------------------
    checks = {
        "a_taylor_slope": results["taylor"]["acceptance"],
        "b_gradient_vs_central_difference": results["central_difference"]["acceptance"],
        "c_grid_convergence": results["grid_convergence"]["acceptance"],
        "d_drag_reduction": results["optimisation"]["acceptance"],
        "e_path_equivalence": results["path_equivalence"]["acceptance"],
    }
    results["criteria"] = {
        **checks,
        "all_but_pytest_pass": all(checks.values()),
    }
    (outdir / "result.json").write_text(
        json.dumps(jsonable(results), indent=1), encoding="utf-8"
    )
    return results


def main(argv=None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/adjoint-shape-sensitivity")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    res = run(Path(args.out), jar=not args.no_figures)
    print(json.dumps(res["criteria"], indent=1))
    return 0 if res["criteria"]["all_but_pytest_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())