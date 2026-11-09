"""Command line runner for the TensorFVM conjugate-heat-transfer verification set.

Each case writes one JSON ledger under ``runs/cht`` plus PNG figures under
``runs/cht/figures``.  Cases are deliberately separable so the expensive
cylinder runs can be sharded onto the compute cluster::

    python -m tensorfvm.cht.runner --case flat_plate
    python -m tensorfvm.cht.runner --case flat_plate_grid
    python -m tensorfvm.cht.runner --case annulus
    python -m tensorfvm.cht.runner --case cylinder --re 40

The active cases are:

``flat_plate``       conjugate plate, ``Nu_x`` against the Pohlhausen similarity
                     solution for Pr = 0.7 ... 7.0.
``flat_plate_grid``  three grids, observed order of convergence of ``Nu_x``.
``annulus``          two-layer radial conduction with an *exact* interface
                     temperature and flux; conformal and non-conformal interface
                     coupling, three grids.
``cylinder``         cross-flow around a hot conducting cylinder; mean Nusselt
                     number against the Churchill-Bernstein correlation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .flatplate import FlatPlateCHT
from .polar import ConjugateCylinder, PolarVorticitySolver
from .similarity import classical_correlation, pohlhausen_nusselt, solve_blasius

ROOT = Path(__file__).resolve().parents[3]  # repository root


def churchill_bernstein(re: float, pr: float) -> float:
    """Mean Nusselt number for cross-flow over a cylinder (Pr >= 0.2)."""
    return (0.3 + 0.62 * math.sqrt(re) * pr ** (1.0 / 3.0)
            / (1.0 + (0.4 / pr) ** (2.0 / 3.0)) ** 0.25
            * (1.0 + (re / 282000.0) ** 0.625) ** 0.8)


def _outdir(out: Path) -> Path:
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def _dump(out: Path, name: str, payload: dict) -> Path:
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


# ------------------------------------------------------------------ flat plate
def case_flat_plate(out: Path, *, quick: bool = False) -> dict:
    blasius = solve_blasius()
    prism = [0.7, 1.0, 5.0, 7.0]
    nx, ny = (120, 60) if quick else (200, 110)
    rows = []
    for pr in prism:
        case = FlatPlateCHT(re_l=1.0e5, pr=pr, solid_ratio=100.0, nx=nx,
                            ny_fluid=ny, ny_solid=16, blasius=blasius)
        res = case.solve()
        m = res.metrics
        rows.append({
            "prandtl": pr,
            "nusselt_similarity_coefficient": m["nusselt_similarity_coefficient"],
            "classical_coefficient_0.332": m["classical_coefficient_0.332"],
            "nu_max_rel_error": m["nu_max_rel_error"],
            "nu_l2_rel_error": m["nu_l2_rel_error"],
            "wall_temperature_nonuniformity": m["wall_temperature_nonuniformity"],
            "interface_local_flux_imbalance_rel": m["interface_local_flux_imbalance_rel"],
            "max_discrete_divergence": m["max_discrete_divergence"],
            "ledger": res.ledger,
        })
        if pr == prism[0]:
            _plot_flat_plate(out, res, pr)
    # isothermal control (no substrate) at Pr = 0.7
    ctrl = FlatPlateCHT.isothermal(re_l=1.0e5, pr=0.7, nx=nx, ny_fluid=ny, blasius=blasius)
    cres = ctrl.solve()
    payload = {
        "case": "flat_plate_cht",
        "reynolds_length": 1.0e5,
        "solid_ratio": 100.0,
        "grid": {"nx": nx, "ny_fluid": ny, "ny_solid": 16},
        "cases": rows,
        "isothermal_control": {
            "prandtl": 0.7,
            "nu_max_rel_error": cres.metrics["nu_max_rel_error"],
            "nu_l2_rel_error": cres.metrics["nu_l2_rel_error"],
            "ledger": cres.ledger,
        },
        "acceptance": {
            "nu_error_limit": 0.15,
            "all_within_limit": all(r["nu_max_rel_error"] < 0.15 for r in rows),
        },
    }
    return payload


def _plot_flat_plate(out: Path, res, pr: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(res.x, res.nu_x, "o", ms=3, label="FVM (conjugate)")
    ax.plot(res.x, res.nu_reference, "-", lw=1.5, label="Pohlhausen similarity")
    ax.set_xlabel("x / L")
    ax.set_ylabel(r"$Nu_x$")
    ax.set_title(f"flat plate, Pr = {pr}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(res.x[1:], np.abs(res.nu_error_rel[1:]), lw=1.2)
    ax.axhline(0.03, color="r", ls="--", lw=1, label="3 % line")
    ax.set_xlabel("x / L")
    ax.set_ylabel("relative error")
    ax.set_title("local Nusselt error")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    yf = res.fields["y_faces"]
    T = res.fields["T"]
    xf = res.fields["x_faces"]
    mesh = ax.imshow(T, origin="lower", aspect="auto", cmap="inferno",
                     extent=[xf[0], xf[-1], yf[0], yf[-1]])
    fig.colorbar(mesh, ax=ax, label="T")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title("temperature field (fluid + substrate)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "cht_flat_plate.png", dpi=130)
    plt.close(fig)


def case_flat_plate_grid(out: Path) -> dict:
    """Three grids; the order is measured on the *discretisation* increment.

    Comparing each grid with the similarity solution would freeze the observed
    order at zero as soon as the solution error is dominated by a
    grid-independent bias (the unheated inlet, the leading-edge treatment).  The
    order is therefore taken from the successive differences of ``Nu_x`` between
    the three grids (Richardson form), which is what a GCI study reports.
    """
    blasius = solve_blasius()
    levels = [(80, 44), (120, 66), (180, 99)]
    errors = []
    profiles = []
    stations = np.linspace(0.12, 0.92, 60)
    for nx, ny in levels:
        case = FlatPlateCHT(re_l=1.0e5, pr=0.7, solid_ratio=100.0, nx=nx,
                            ny_fluid=ny, ny_solid=12, blasius=blasius)
        res = case.solve()
        errors.append({
            "nx": nx, "ny_fluid": ny,
            "nu_l2_rel_error": res.metrics["nu_l2_rel_error"],
            "nu_max_rel_error": res.metrics["nu_max_rel_error"],
            "interface_local_flux_imbalance_rel": res.metrics[
                "interface_local_flux_imbalance_rel"],
        })
        profiles.append(np.interp(stations, res.x, res.nu_x))
    order = []
    for i in range(1, len(levels)):
        e1, e2 = errors[i - 1]["nu_l2_rel_error"], errors[i]["nu_l2_rel_error"]
        r = levels[i][0] / levels[i - 1][0]
        order.append(math.log(e1 / e2) / math.log(r) if e2 > 0 else float("nan"))
    increments = [profiles[0] - profiles[1], profiles[1] - profiles[2]]
    d1 = float(np.linalg.norm(increments[0]) / math.sqrt(stations.size))
    d2 = float(np.linalg.norm(increments[1]) / math.sqrt(stations.size))
    r_ratio = levels[1][0] / levels[0][0]
    order_increment = math.log(d1 / d2) / math.log(r_ratio) if d2 > 0 else float("nan")

    # pure diffusion operator order: 1-D conduction with an exact linear profile is
    # exact, so use the annular two-layer case (below) for the second order claim
    # wall-normal refinement alone: the direction that sets Nu_x
    ys = [44, 66, 99]
    yprof = []
    yerrmax = []
    for ny in ys:
        case = FlatPlateCHT(re_l=1.0e5, pr=0.7, solid_ratio=100.0, nx=180,
                            ny_fluid=ny, ny_solid=12, blasius=blasius)
        res = case.solve()
        yprof.append(np.interp(stations, res.x, res.nu_x))
        yerrmax.append(res.metrics["nu_l2_rel_error"])
    dy1 = float(np.linalg.norm(yprof[0] - yprof[1]) / math.sqrt(stations.size))
    dy2 = float(np.linalg.norm(yprof[1] - yprof[2]) / math.sqrt(stations.size))
    order_wall_normal = math.log(dy1 / dy2) / math.log(ys[1] / ys[0]) if dy2 > 0 else float("nan")
    return {
        "case": "flat_plate_grid_convergence",
        "wall_normal_refinement": {"ny_fluid": ys, "nu_l2_rel_error": yerrmax,
                                  "observed_order": order_wall_normal},
        "refinement_ratio": levels[-1][0] / levels[0][0],
        "levels": errors,
        "observed_order_vs_similarity": order,
        "nu_profile_increment_norms": {"coarse_medium": d1, "medium_fine": d2},
        "observed_order": [order_increment],
        "acceptance": {
            "order_at_least_2_uniform": bool(order_increment >= 2.0),
            "order_at_least_2_wall_normal": bool(order_wall_normal >= 2.0),
            "order_measured_on": "successive Nu_x profile increments",
        },
    }


# --------------------------------------------------------------------- annulus
def analytic_annulus(k_ratio: float, r_inner: float = 0.15, r_interface: float = 0.5,
                     r_outer: float = 12.0) -> dict:
    """Exact two-layer radial conduction solution (``T = A ln r + B`` per layer).

    Inner wall ``T = 1`` at ``r_inner``, outer wall ``T = 0`` at ``r_outer``, and
    flux continuity ``k_s dT_s/dr = k_f dT_f/dr`` at ``r_interface``.
    """
    c_in = math.log(r_interface / r_inner)
    c_out = math.log(r_interface / r_outer)
    a_s = 1.0 / (k_ratio * c_out - c_in)
    a_f = k_ratio * a_s
    t_interface = 1.0 + c_in * a_s
    return {"a_solid": a_s, "a_fluid": a_f, "interface_temperature": t_interface,
            "interface_flux_kappa_units": -a_s / r_interface,
            "note": "flux in units of kappa_solid * dT / L"}


def case_annulus(out: Path, *, quick: bool = False) -> dict:
    k_ratio = 100.0
    ref = analytic_annulus(k_ratio)
    levels = [(36, 48, 48), (54, 72, 72), (81, 108, 96)] if not quick else [(36, 48, 48)]
    rows = []
    for n_r, n_theta, n_theta_solid in levels:
        flow = PolarVorticitySolver(re=1.0, n_r=n_r, n_theta=n_theta, r_out=12.0,
                                    wall_cell=0.05)
        case = ConjugateCylinder(re=1.0, pr=0.7, solid_ratio=k_ratio,
                                 n_theta_solid=n_theta_solid, n_r_solid=24, flow=flow)
        psi = np.zeros((flow.n_node, flow.n_theta))
        res = case.couple(psi, outer_temperature=0.0, relax=0.8, max_iter=400, tol=1e-12)
        m = res.metrics
        nu = m["nusselt_mean"]
        ref_flux = case.kappa_solid * ref["interface_flux_kappa_units"]
        nu_exact = ref_flux / case.kappa_fluid
        rows.append({
            "n_r": n_r, "n_theta": n_theta, "n_theta_solid": n_theta_solid,
            "interface_temperature": m["wall_temperature_mean"],
            "interface_temperature_exact": ref["interface_temperature"],
            "interface_temperature_error": abs(m["wall_temperature_mean"]
                                               - ref["interface_temperature"]),
            "interface_flux_error_rel": abs(m["heat_rate_fluid"]
                                            / (ref_flux * 2.0 * math.pi * 0.5) - 1.0),
            "nusselt": nu, "nusselt_exact": nu_exact,
            "nusselt_error_rel": abs(nu - nu_exact) / abs(nu_exact),
            "interface_local_residual_rel": m["interface_local_residual_rel"],
            "interface_global_residual_rel": m["interface_global_residual_rel"],
            "coupling_iterations": m["coupling_iterations"],
            "conformal": m["conformal"],
        })
    order = []
    for i in range(1, len(rows)):
        r = rows[i]["n_r"] / rows[i - 1]["n_r"]
        e1, e2 = rows[i - 1]["nusselt_error_rel"], rows[i]["nusselt_error_rel"]
        order.append(math.log(e1 / e2) / math.log(r) if e2 > 0 else float("nan"))
    payload = {
        "case": "annulus_two_layer_conduction",
        "solid_to_fluid_conductivity_ratio": k_ratio,
        "exact": ref,
        "levels": rows,
        "observed_order": order,
        "acceptance": {
            "interface_flux_residual_limit": 1e-6,
            "interface_conserved": all(r["interface_local_residual_rel"] < 1e-6
                                       for r in rows),
            "order_at_least_2": bool(order and min(order) >= 2.0),
        },
    }
    return payload


# -------------------------------------------------------------------- cylinder
def case_cylinder(out: Path, *, re: float, n_r: int = 71, n_theta: int = 72,
                  r_out: float = 15.0, wall_cell: float = 8.0e-3,
                  dt: float = 2.0e-3, t_end: float = 60.0, solid_ratio: float = 100.0,
                  pr: float = 0.7) -> dict:
    t0 = time.time()
    flow = PolarVorticitySolver(re=re, n_r=n_r, n_theta=n_theta, r_out=r_out,
                                wall_cell=wall_cell)
    march = flow.march(t_end=t_end, dt=dt, tol_steady=1e-7, max_steps=40000,
                       t_ramp=4.0)
    cd, cl = flow.drag_lift(march["wall_vorticity"])
    case = ConjugateCylinder(re=re, pr=pr, solid_ratio=solid_ratio, flow=flow)
    res = case.couple(march["psi"], outer_temperature=None, relax=0.7, max_iter=200,
                      tol=1e-8)
    nu_cb = churchill_bernstein(re, pr)
    m = res.metrics
    payload = {
        "case": "cylinder_cross_flow_cht",
        "reynolds": re,
        "prandtl": pr,
        "mesh": {"n_r": n_r, "n_theta": n_theta, "r_out": r_out,
                 "wall_cell": wall_cell},
        "flow": {"steady": march["steady"], "steps": march["steps"],
                 "time": march["time"], "dt": dt, "drag_coefficient": cd,
                 "lift_coefficient": cl},
        "nusselt_mean": m["nusselt_mean"],
        "nusselt_mean_flux_weighted": m["nusselt_mean_flux_weighted"],
        "nusselt_churchill_bernstein": nu_cb,
        "nusselt_error_rel": abs(m["nusselt_mean"] - nu_cb) / nu_cb,
        "interface_local_residual_rel": m["interface_local_residual_rel"],
        "interface_global_residual_rel": m["interface_global_residual_rel"],
        "wall_temperature_mean": m["wall_temperature_mean"],
        "wall_temperature_ptp": m["wall_temperature_ptp"],
        "energy_rate_balance_rel": m["energy_rate_balance_rel"],
        "runtime_s": time.time() - t0,
        "acceptance": {
            "nusselt_error_limit": 0.15,
            "within_limit": abs(m["nusselt_mean"] - nu_cb) / nu_cb < 0.15,
            "interface_conserved": m["interface_local_residual_rel"] < 1e-6,
        },
    }
    _plot_cylinder(out, res, re, nu_cb)
    return payload


def _plot_cylinder(out: Path, res, re: float, nu_cb: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 4.2))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(np.degrees(res.theta), res.nusselt_local, lw=1.2)
    ax.set_xlabel("theta [deg]")
    ax.set_ylabel(r"$Nu_\theta$")
    ax.set_title(f"local Nusselt, Re = {re:g}")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(np.degrees(res.theta), res.wall_temperature, lw=1.2)
    ax.set_xlabel("theta [deg]")
    ax.set_ylabel(r"$T_w$")
    ax.set_title("conjugate wall temperature")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 3)
    T = res.fields["T_fluid"]
    r = np.asarray(res.fields.get("r_node", range(T.shape[0])), dtype=float)
    sel = slice(0, min(T.shape[0], 40))
    ax.plot(T[sel, 0], label="theta = 0")
    ax.plot(T[sel, T.shape[1] // 2], label="theta = pi")
    ax.set_xlabel("radial node")
    ax.set_ylabel("T")
    ax.set_title(f"radial profiles (CB Nu = {nu_cb:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / f"cht_cylinder_re{int(re)}.png", dpi=130)
    plt.close(fig)


def _max_spread(rows) -> bool:
    """True when the interface temperature is grid independent per ratio."""
    by_ratio = {}
    for r in rows:
        by_ratio.setdefault(r["conductivity_ratio"], []).append(r["interface_temperature"])
    return max(max(v) - min(v) for v in by_ratio.values()) < 1e-9


def case_slab(out: Path, *, ratios=(1.0, 0.1, 10.0), cells=(40, 80, 160)) -> dict:
    """Two-layer slab: exact interface flux continuity and grid independence.

    Conduction across a conductivity jump is piecewise linear, and the
    harmonic-mean face conductance reproduces it *exactly* -- so this case
    isolates the interface treatment: any residual of ``q_left - q_right`` or any
    grid dependence is a defect of the interface algebra, not of a discretisation
    error that could be blamed on the flow.
    """
    from .energy import solve_steady_energy

    rows = []
    for nx in cells:
        for ratio in ratios:
            kappa1 = 1.0
            kappa2 = ratio
            xf = np.linspace(0.0, 1.0, 2 * nx + 1)
            dx = np.diff(xf)
            kappa = np.empty((1, xf.size - 1))
            kappa[0, :nx] = kappa1
            kappa[0, nx:] = kappa2
            flux_e = np.zeros((1, xf.size))
            flux_n = np.zeros((2, xf.size - 1))
            bc = {"west": {"type": "dirichlet", "values": 0.0},
                  "east": {"type": "dirichlet", "values": 1.0}}
            T, info = solve_steady_energy(dx, np.array([1.0]), kappa, flux_e, flux_n, bc)
            q = -(kappa[0, nx] - kappa[0, nx - 1]) * 0.0
            # interface flux from each side (per unit area, positive in +x)
            cond = 1.0 / (0.5 * dx[nx - 1] / kappa1 + 0.5 * dx[nx] / kappa2)
            q_iface = cond * (T[0, nx - 1] - T[0, nx])
            q_left = kappa1 * (T[0, nx - 2] - T[0, nx - 1]) / dx[nx - 1]
            q_right = kappa2 * (T[0, nx] - T[0, nx + 1]) / dx[nx]
            exact_ratio = 1.0 / (1.0 + ratio)      # temperature drop split
            t_iface = T[0, nx - 1] - 0.5 * dx[nx - 1] * (q_iface / kappa1)
            rows.append({
                "nx_half": nx, "conductivity_ratio": ratio,
                "interface_temperature": float(t_iface),
                "interface_temperature_exact": float(0.5),
                "interface_flux": float(q_iface),
                "interface_flux_left": float(q_left),
                "interface_flux_right": float(q_right),
                "interface_flux_jump_rel": float(abs(q_left - q_right)
                                                 / max(abs(q_iface), 1e-300)),
                "ledger_imbalance_normalized": info["ledger"].normalized_imbalance,
                "picard_iterations": info["picard_iterations"],
                "expected_drop_ratio": exact_ratio,
            })
    return {
        "case": "two_layer_slab_conduction",
        "levels": rows,
        "acceptance": {
            "flux_jump_limit": 1e-6,
            "max_flux_jump_rel": max(r["interface_flux_jump_rel"] for r in rows),
            "all_conservative": all(r["interface_flux_jump_rel"] < 1e-6 for r in rows),
            "grid_independent": _max_spread(rows),
        },
    }


CASES = {
    "flat_plate": lambda out, a: case_flat_plate(out, quick=a.quick),
    "flat_plate_grid": lambda out, a: case_flat_plate_grid(out),
    "annulus": lambda out, a: case_annulus(out, quick=a.quick),
    "slab": lambda out, a: case_slab(out),
    "cylinder": lambda out, a: case_cylinder(out, re=a.re, n_r=a.n_r, n_theta=a.n_theta,
                                            dt=a.dt, t_end=a.t_end,
                                            solid_ratio=a.solid_ratio, pr=a.pr),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--out", default=str(ROOT / "runs" / "cht"))
    parser.add_argument("--re", type=float, default=40.0)
    parser.add_argument("--pr", type=float, default=0.7)
    parser.add_argument("--n-r", type=int, default=71)
    parser.add_argument("--n-theta", type=int, default=72)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--t-end", type=float, default=60.0)
    parser.add_argument("--solid-ratio", type=float, default=100.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    out = _outdir(Path(args.out))
    payload = CASES[args.case](out, args)
    name = payload["case"] if args.case != "cylinder" else f"{payload['case']}_re{int(args.re)}"
    path = _dump(out, name, payload)
    print(json.dumps({"wrote": str(path), "summary": {
        k: payload[k] for k in ("acceptance",) if k in payload}}))
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()