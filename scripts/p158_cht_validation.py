"""P158W3-N02 verification driver: conjugate heat transfer anchors for TensorFVM.

Two verification faces, every number written to ``docs/p158-cht-verification/artifacts/*.json`` so the
report can be filled programmatically (no hand-copied digits):

``--case flatplate``
    Conjugate flat plate with a *uniform bottom heat flux* through a conducting
    slab.  Anchors: (A1) local ``Nu_x`` against the uniform-heat-flux Pohlhausen
    similarity solution ``Nu_x = sqrt(Re_x)/phi(0)`` at five ``Re_x`` in
    ``[1e4, 1e6]``; (A2) the solid temperature field against the analytic
    conjugate profile ``T_w(x) - q'' y/k_s``; (A4) interface heat-flux
    conservation (local imbalance and the global fluid/solid balance); (A5) a
    three-grid GCI study of ``Nu_x``.

``--case cylinder``
    Forced convection over an isothermal circular cylinder, ``Re`` in
    ``{40, 100, 200}`` and ``Pr`` in ``{0.7, 7}``, solved with the spectral
    azimuthal / finite-volume radial solver (:mod:`tensorfvm.cht.cylinder_fourier`),
    compared with the Churchill-Bernstein and Hilpert correlations.  With
    ``--conjugate`` the core is a conducting solid (``k_s/k_f = 100``) and the
    interface flux/temperature continuity residuals are reported as well.

Usage::

    python scripts/p158_cht_validation.py --case flatplate
    python scripts/p158_cht_validation.py --case cylinder --re 40 --pr 0.7
    python scripts/p158_cht_validation.py --case cylinder --shard 2 --nshards 6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.cht.flatplate import FlatPlateCHT                       # noqa: E402
from tensorfvm.cht.similarity import (solve_uniform_flux,              # noqa: E402
                                      classical_uniform_flux_correlation)

OUT = ROOT / "docs" / "p158-cht-verification" / "artifacts"

# cylinder grids: name -> (n_r, n_theta, wall_cell, r_out)
GRIDS = {
    "coarse": (60, 64, 8.0e-3, 20.0),
    "medium": (80, 96, 4.0e-3, 25.0),
    "fine": (110, 128, 2.0e-3, 30.0),
}
# The step is limited by the *explicit* no-slip wall-vorticity coupling: the wall
# value enters the first-cell diffusion solve with weight dt*nu/dr^2, and the march
# is only robust for dt*nu/dr^2 <~ 0.3 (measured: 0.78 diverges after t ~ 6).
def _dt(re: float, wall_cell: float) -> float:
    return 0.3 * wall_cell ** 2 * re


DT_OVERRIDE: dict = {}
TEND = {40.0: 70.0, 100.0: 120.0, 200.0: 170.0}


def hilpert(re: float, pr: float) -> float:
    """Hilpert (1933) cross-flow correlation, 40 < Re < 4000."""
    return 0.683 * re ** 0.466 * pr ** (1.0 / 3.0)


def eckert_soehngen(re: float, pr: float) -> float:
    """Eckert & Soehngen low-Re laminar correlation (Re < 40-100)."""
    return 0.43 + 0.50 * re ** 0.5 * pr ** 0.38 * (pr / (pr + 4.0)) ** 0.25


# --------------------------------------------------------------- flat plate
def run_flatplate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"case": "flatplate_uniform_flux", "targets": [], "gci": {}}
    targets = [1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6]
    for re_target in targets:
        # put the target station in the middle of the plate: x = 0.5
        case = FlatPlateCHT(re_l=2.0 * re_target, pr=0.7, solid_ratio=10.0,
                            thickness=0.005, height=0.08, nx=240, ny_fluid=90,
                            ny_solid=10, wall_flux=1.0)
        res = case.solve(tol=1.0e-10, max_picard=400)
        i = int(np.argmin(np.abs(res.x - 0.5)))
        phi = solve_uniform_flux(0.7, blasius=case.blasius)
        payload["targets"].append({
            "re_x": re_target,
            "re_l": 2.0 * re_target,
            "nusselt": float(res.nu_x[i]),
            "nusselt_similarity": float(res.nu_reference[i]),
            "nusselt_classical": float(res.nu_correlation[i]),
            "rel_error_similarity": float(res.nu_x[i] / res.nu_reference[i] - 1.0),
            "rel_error_classical": float(res.nu_x[i] / res.nu_correlation[i] - 1.0),
            "solid_rel_error_max": float(res.metrics["solid_analytic_max_rel_error"]),
            "wall_flux_uniformity_rel": float(res.metrics["wall_flux_uniformity_rel"]),
            "interface_local_imbalance_rel": float(
                res.metrics["interface_local_flux_imbalance_rel"]),
            "interface_global_balance_rel": float(
                res.metrics["global_interface_balance_rel"]),
            "ledger_normalized_imbalance": float(res.ledger["normalized_imbalance"]),
            "max_discrete_divergence": float(res.metrics["max_discrete_divergence"]),
            "wall_temperature": float(res.t_wall[i]),
            "wall_temperature_analytic": float(res.metrics["wall_temperature_analytic"][i]),
            "grid": res.metrics["grid"],
            "phi0": float(phi.phi0),
            "classical_coefficient": float(classical_uniform_flux_correlation(0.7)),
        })
    # ---- GCI: a *self-similar* grid family (uniform y, both directions halved).
    # The stretched family used for the anchors (y_power = 2) is not a geometric
    # refinement -- halving ny divides every cell height by four -- so Richardson
    # is only meaningful for the uniform family used here.
    nu_seq = []
    for (nx, nyf, nys) in ((120, 48, 12), (240, 96, 24), (480, 192, 48)):
        case = FlatPlateCHT(re_l=2.0e5, pr=0.7, solid_ratio=10.0, thickness=0.005,
                            height=0.08, nx=nx, ny_fluid=nyf, ny_solid=nys,
                            wall_flux=1.0, y_power=1.0)
        res = case.solve(tol=1.0e-10, max_picard=400)
        i = int(np.argmin(np.abs(res.x - 0.5)))
        nu_seq.append(float(res.nu_x[i]))
    f1, f2, f3 = nu_seq
    r = 2.0
    d12, d23 = f2 - f1, f3 - f2
    order = float(math.log(abs(d12 / d23)) / math.log(r)) if d23 != 0.0 else float("nan")
    denom = r ** order - 1.0
    gci_fine = float(1.25 * abs(d23) / abs(f3) / denom)
    extrap = float(f3 + d23 / denom)
    payload["gci"] = {
        "grids": [[120, 48, 12], [240, 96, 24], [480, 192, 48]],
        "y_power": 1.0,
        "re_x": 1.0e5,
        "nusselt": nu_seq,
        "ratio": r,
        "observed_order": order,
        "gci_fine_rel": gci_fine,
        "richardson_extrapolated": extrap,
        "similarity_reference": float(1.0e5 ** 0.5 *
                                     solve_uniform_flux(0.7).coefficient),
        "errors_vs_similarity": [float(f / (1.0e5 ** 0.5 *
                                           solve_uniform_flux(0.7).coefficient) - 1.0)
                                 for f in nu_seq],
    }
    path = OUT / "flatplate.json"
    path.write_text(json.dumps(payload, indent=1))
    return payload


# ------------------------------------------------------------------ cylinder
def run_cylinder(re: float, pr: float, grid: str = "medium", conjugate: bool = False,
                 tag: str | None = None) -> dict:
    from tensorfvm.cht.cylinder_fourier import (FourierCylinderEnergy,
                                                FourierCylinderFlow)
    from tensorfvm.cht.runner import churchill_bernstein

    OUT.mkdir(parents=True, exist_ok=True)
    n_r, n_theta, wall_cell, r_out = GRIDS[grid]
    dt = DT_OVERRIDE.get((re, grid), round(_dt(re, wall_cell), 10))
    t_end = TEND[re]
    t0 = time.time()
    flow = FourierCylinderFlow(re=re, n_r=n_r, n_theta=n_theta, r_out=r_out,
                               wall_cell=wall_cell)
    march = flow.march(t_end=t_end, dt=dt, tol_steady=1e-7, t_ramp=10.0)
    cd, cl = flow.drag_lift(march["wall_vorticity"])
    energy = FourierCylinderEnergy(flow, pr=pr)
    t_wall = 1.0
    if conjugate:
        res = _conjugate_cylinder(flow, energy, pr=pr, solid_ratio=100.0)
    else:
        res = energy.solve(t_wall, t_outer=0.0, t_end=60.0)
    nu = res["nusselt_mean"]
    payload = {
        "case": tag or f"cyl_re{re:g}_pr{pr:g}_{grid}",
        "reynolds": re, "prandtl": pr, "grid_name": grid,
        "grid": {"n_r": n_r, "n_theta": n_theta, "wall_cell": wall_cell,
                 "r_out": r_out},
        "dt": dt, "t_end": t_end, "steps": march["steps"],
        "time_to_steady": march["time"], "steady": march["steady"],
        "drag_coefficient": float(cd), "lift_coefficient": float(cl),
        "wall_vorticity_extreme": flow.wall_shear_max(march["wall_vorticity"]),
        "nusselt": float(nu),
        "nusselt_local": [float(v) for v in res["nu_local"]],
        "theta_deg": [float(v * 180.0 / math.pi) for v in flow.grid.theta],
        "wall_temperature": (t_wall if not conjugate else
                             [float(v) for v in res["wall_temperature"]]),
        "reference_churchill_bernstein": float(churchill_bernstein(re, pr)),
        "reference_hilpert": float(hilpert(re, pr)),
        "reference_eckert_soehngen": float(eckert_soehngen(re, pr)),
        "rel_error_churchill_bernstein": float(nu / churchill_bernstein(re, pr) - 1.0),
        "rel_error_hilpert": float(nu / hilpert(re, pr) - 1.0),
        "rel_error_eckert_soehngen": float(nu / eckert_soehngen(re, pr) - 1.0),
        "energy_steps": res["steps"], "energy_dt": res["dt"],
        "energy_converged_delta": res["converged_delta"],
        "wall_time_s": time.time() - t0,
        "drag_history": [[float(a), float(b), float(c)]
                         for a, b, c in march["drag_history"]],
        "steady_history_tail": [float(v) for v in march["wall_vorticity_history"][-50:]],
        "temperature_field": _downsample(res["T"], 40, 24),
        "radial_nodes": [float(v) for v in flow.grid.r],
    }
    if conjugate:
        payload.update(res["conjugate"])
    name = payload["case"]
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1))
    return payload


def _downsample(field: np.ndarray, nr: int, nt: int) -> dict:
    ri = np.linspace(0, field.shape[0] - 1, nr).astype(int)
    tj = np.linspace(0, field.shape[1] - 1, nt).astype(int)
    return {"r_index": ri.tolist(), "theta_index": tj.tolist(),
            "values": field[np.ix_(ri, tj)].tolist()}


def _conjugate_cylinder(flow, energy, *, pr: float, solid_ratio: float = 100.0,
                        max_iter: int = 60, tol: float = 1.0e-9) -> dict:
    """Dirichlet-Neumann coupling with a conducting core of radius 0.15 D.

    The solid is the same annular laplacian used by the fluid solver, driven by an
    inner Dirichlet temperature and the interface flux; the fluid is re-solved
    with the updated wall temperature each sweep.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from tensorfvm.cht.cylinder_fourier import _Annulus

    g = flow.grid
    alpha_f = flow.nu / pr
    alpha_s = solid_ratio * alpha_f
    solid = _Annulus(0.15, 0.5, 40, (0.5 - 0.15) / 40.0, g.n_theta)
    lu = spla.splu(solid.L.tocsc())
    t_wall = np.ones(g.n_theta)
    history = []
    res = None
    for it in range(max_iter):
        res = energy.solve(t_wall, t_outer=0.0, t_end=60.0)
        q_fluid = res["q_wall"]                       # into the fluid
        # solid: inner Dirichlet T = 1, interface Neumann -q_fluid
        rhs = (-q_fluid * solid.r_face[-1] * solid.dtheta)[None, :] * np.eye(solid.ni)[:, -1:]
        rhs = rhs + solid.coef_w[0] * 1.0 * np.eye(solid.ni)[:, 0:1]
        t_solid = np.zeros((solid.n_node, g.n_theta))
        t_solid[0, :] = 1.0
        t_solid[1:-1, :] = lu.solve(rhs.ravel()).reshape(solid.ni, g.n_theta)
        # sign: L T = rhs with +laplacian convention -> flux enters as +q
        s1 = solid.r[1] - solid.r[0]
        s2 = solid.r[2] - solid.r[0]
        c0 = -(s1 + s2) / (s1 * s2)
        c1 = s2 / (s1 * (s2 - s1))
        c2 = -s1 / (s2 * (s2 - s1))
        slope = c0 * t_solid[0, :] + c1 * t_solid[1, :] + c2 * t_solid[2, :]
        q_solid = alpha_s * slope                     # outward from the solid core
        scale = max(float(np.max(np.abs(q_fluid))), 1e-30)
        local = float(np.max(np.abs(q_fluid - q_solid))) / scale
        glob = abs(float(np.sum(q_fluid)) - float(np.sum(q_solid))) / max(
            abs(float(np.sum(q_fluid))), 1e-30)
        history.append({"iteration": it, "local": local, "global": glob,
                        "wall_temperature_ptp": float(np.ptp(t_wall))})
        t_wall = t_wall + 0.7 * (t_solid[-1, :] - t_wall)
        if local < tol and glob < tol:
            break
    nu = res["nusselt_mean"]
    return {
        "nusselt_mean": nu,
        "wall_temperature": t_wall,
        "conjugate": {
            "solid_ratio": solid_ratio,
            "coupling_iterations": len(history),
            "interface_local_residual_rel": history[-1]["local"],
            "interface_global_residual_rel": history[-1]["global"],
            "wall_temperature_ptp": history[-1]["wall_temperature_ptp"],
            "coupling_history": history[-5:],
        },
    }


CASES = [("40", "0.7", "medium"), ("40", "7", "medium"),
         ("100", "0.7", "medium"), ("100", "7", "medium"),
         ("200", "0.7", "medium"), ("200", "7", "medium"),
         ("40", "7", "coarse"), ("100", "7", "coarse"),
         ("200", "7", "coarse")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["flatplate", "cylinder"], required=True)
    ap.add_argument("--re", type=float, default=40.0)
    ap.add_argument("--pr", type=float, default=0.7)
    ap.add_argument("--grid", default="medium", choices=sorted(GRIDS))
    ap.add_argument("--conjugate", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=None)
    args = ap.parse_args(argv)
    if args.case == "flatplate":
        payload = run_flatplate()
        print(json.dumps({"targets": len(payload["targets"]),
                          "order": payload["gci"]["observed_order"]}))
        return
    if args.shard is not None and args.nshards:
        re_s, pr_s, grid = CASES[args.shard % len(CASES)]
        payload = run_cylinder(float(re_s), float(pr_s), grid)
    else:
        payload = run_cylinder(args.re, args.pr, args.grid,
                               conjugate=args.conjugate, tag=args.tag)
    print(json.dumps({k: payload[k] for k in
                      ("case", "reynolds", "prandtl", "nusselt", "steady",
                       "drag_coefficient", "rel_error_churchill_bernstein")}))


if __name__ == "__main__":
    main()