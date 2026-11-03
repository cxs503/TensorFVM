"""Runner: second-order verification of the annular laminar backend.

Usage:  python scripts/verify_annular_order.py [--levels 8,16,32,64]

Writes into ``scripts/verification-annular-order/``:
  results.json   raw numbers (no hand-copied digits reach the report)
  report.md      publication-form verification report rendered from results.json
  profile.png    velocity profile vs the exact annular Poiseuille solution
  convergence.png  log-log error vs mesh spacing with the nominal 2nd-order slope
  field.png      axial-velocity cloud map and its absolute error map

Everything is deterministic, CPU-only, double precision.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.annular_order import (
    DP_DZ, ERROR_LIMIT, MU, ORDER_LIMIT, RI, RO, SELF_CHECK_LIMIT,
    acceptance, analytic_flow_rate, analytic_max_velocity, analytic_velocity,
    analytic_wall_shear,
    convergence, geometry_audit, quadrature_flow_rate, reference_self_check,
)
from tensorfvm.verification.core import jsonable, write_json

OUT = Path(__file__).resolve().parent / "verification-annular-order"


def strip(study):
    """Study without the raw field arrays (they go to the figures only)."""
    rows = []
    for row in study["rows"]:
        rows.append({k: v for k, v in row.items() if k != "arrays"})
    return dict(rows=rows, orders=study["orders"], geometry=study["geometry"],
                geometry_orders=study["geometry_orders"])


def figure_profile(study):
    row = study["rows"][-1]
    a = row["arrays"]
    r, u = a["radius_m"], a["u_num_m_s"]
    centers = a["centers_m"].reshape(r.size, 2)
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    grid = np.linspace(RI, RO, 400)
    ax.plot(grid, analytic_velocity(grid) * 1e3, "k-", lw=1.6, label="exact solution")
    for target, marker in ((0.0, "o"), (math.pi / 2, "s"), (math.pi, "^")):
        pick = np.abs(np.angle(np.exp(1j * (theta - target)))) < 0.6 * 2 * math.pi / row["ntheta"]
        order = np.argsort(r[pick])
        ax.plot(r[pick][order], u[pick][order] * 1e3, marker, ms=3.0, mfc="none",
                label=rf"FVM $\theta\approx{target:.2f}$ rad")
    ax.set_xlabel("radius r [m]")
    ax.set_ylabel(r"axial velocity u [mm/s]")
    ax.set_title(f"velocity profile, nr={row['nr']}, ntheta={row['ntheta']}")
    ax.legend(fontsize=7, loc="lower center")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(r, np.abs(a["u_num_m_s"] - a["u_exact_m_s"]) * 1e3, ".", ms=1.5, color="tab:red")
    ax.set_xlabel("radius r [m]")
    ax.set_ylabel(r"$|u_{num}-u_{exact}|$ [mm/s]")
    ax.set_title("pointwise absolute error")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "profile.png", dpi=110)
    plt.close(fig)


def figure_convergence(study):
    rows = study["rows"]
    h = np.array([row["spacing_m"] for row in rows])
    series = {
        "velocity L2": [next(m["value"] for m in row["metrics"] if m["name"] == "velocity_l2") for row in rows],
        "velocity Linf": [next(m["value"] for m in row["metrics"] if m["name"] == "velocity_linf") for row in rows],
        "flow rate": [next(m["value"] for m in row["metrics"] if m["name"] == "flow_rate") for row in rows],
        "inner wall shear": [next(m["value"] for m in row["metrics"] if m["name"] == "inner_wall_shear") for row in rows],
        "outer wall shear": [next(m["value"] for m in row["metrics"] if m["name"] == "outer_wall_shear") for row in rows],
        "max velocity": [next(m["value"] for m in row["metrics"] if m["name"] == "max_velocity") for row in rows],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    for name, values in series.items():
        ax.loglog(h, values, "o-", ms=3.5, lw=1.1, label=name)
    guide = series["velocity L2"][0] * (h / h[0]) ** 2
    ax.loglog(h, guide, "k--", lw=1.0, label="slope 2")
    ax.axhline(ERROR_LIMIT, color="tab:gray", ls=":", lw=1.0, label="3% acceptance limit")
    ax.set_xlabel("radial spacing h = (ro-ri)/nr [m]")
    ax.set_ylabel("relative error")
    ax.set_title("refinement study")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    geometry = study["geometry"]
    hg = np.array([(RO - RI) / g["nr"] for g in geometry])
    area = [abs(g["relative_area_error"]) for g in geometry]
    ax.loglog(hg, area, "s-", color="tab:purple", ms=3.5, label="polygon area defect")
    ax.loglog(hg, area[0] * (hg / hg[0]) ** 2, "k--", lw=1.0, label="slope 2")
    ax.set_xlabel("radial spacing h [m]")
    ax.set_ylabel("relative area error")
    ax.set_title("geometry: straight-edge annulus vs exact")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=110)
    plt.close(fig)


def figure_field(study):
    row = study["rows"][-1]
    a = row["arrays"]
    centers = a["centers_m"].reshape(-1, 2)
    r = a["radius_m"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    size = 2600.0 / max(row["cells"] ** 0.55, 1.0)
    ax = axes[0]
    sc = ax.scatter(centers[:, 0] * 1e2, centers[:, 1] * 1e2, c=a["u_num_m_s"] * 1e3,
                    s=size, marker="s", cmap="viridis")
    ax.add_patch(plt.Circle((0, 0), RI * 1e2, color="white", zorder=3))
    ax.add_patch(plt.Circle((0, 0), RO * 1e2, fill=False, color="k", lw=0.8, zorder=4))
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(f"axial velocity [mm/s], {row['cells']} cells")
    fig.colorbar(sc, ax=ax, fraction=0.046)

    ax = axes[1]
    err = np.abs(a["u_num_m_s"] - a["u_exact_m_s"]) * 1e3
    sc = ax.scatter(centers[:, 0] * 1e2, centers[:, 1] * 1e2, c=err, s=size, marker="s",
                    cmap="magma", vmin=0)
    ax.add_patch(plt.Circle((0, 0), RI * 1e2, color="white", zorder=3))
    ax.add_patch(plt.Circle((0, 0), RO * 1e2, fill=False, color="k", lw=0.8, zorder=4))
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title("absolute velocity error [mm/s]")
    fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "field.png", dpi=110)
    plt.close(fig)


def render(study, self_check, verdict, runtime):
    rows = study["rows"]
    orders = study["orders"]
    metric_names = [m["name"] for m in rows[0]["metrics"]]
    labels = {
        "velocity_linf": r"$\|u-u_{ex}\|_\infty/\max|u_{ex}|$",
        "velocity_l2": r"volume-weighted $\|u-u_{ex}\|_2/\max|u_{ex}|$",
        "flow_rate": r"$|Q-Q_{ex}|/Q_{ex}$",
        "inner_wall_shear": r"inner-wall shear $|\tau_i-\tau_{ex}|/|\tau_{ex}|$",
        "outer_wall_shear": r"outer-wall shear $|\tau_o-\tau_{ex}|/|\tau_{ex}|$",
        "max_velocity": r"$|u_{max}-u_{ex,max}|/u_{ex,max}$",
    }
    s = self_check["checks"]
    v = rows[-1]["values"]
    lines = []
    add = lines.append
    add("# Second-order verification of the annular laminar FVM backend")
    add("")
    add("**Case**: fully developed laminar annular Poiseuille flow, production backend "
        "`tensorfvm.annular` (fitted diffusion kernel, straight-edge polygonal annulus, "
        "no analytic initialisation).")
    add("")
    add("**Verdict**: "
        + ("PASS — every reported quantity is below the 3 % acceptance limit at the finest "
           f"level and the velocity error converges at observed order {verdict['l2_order']:.2f}."
           if verdict["passed"] else "FAIL — see the table below."))
    add("")
    add("## 1. Case, geometry and parameters")
    add("")
    add("| quantity | symbol | value |")
    add("|---|---|---|")
    add(f"| inner radius | ri | {RI:.4f} m |")
    add(f"| outer radius | ro | {RO:.4f} m |")
    add(f"| radius ratio | ri/ro | {RI / RO:.3f} |")
    add(f"| axial pressure gradient (volumetric source) | G | {DP_DZ:.4f} Pa/m |")
    add(f"| dynamic viscosity | mu | {MU:.4f} Pa s |")
    add(f"| density | rho | 1.0 kg/m3 (irrelevant: Stokes limit) |")
    add(f"| exact flow rate | Q | {s['flow_rate_m3_s']:.6e} m3/s |")
    add(f"| exact peak velocity at r = {s['max_velocity_radius_m'] * 1e3:.2f} mm | u_max | "
        f"{s['max_velocity_m_s'] * 1e3:.5f} mm/s |")
    add(f"| exact inner-wall shear | tau_i | {s['inner_wall_shear_pa']:.6e} Pa |")
    add(f"| exact outer-wall shear | tau_o | {s['outer_wall_shear_pa']:.6e} Pa |")
    add("")
    add("Reynolds number based on the hydraulic diameter is "
        f"{1.0 * s['max_velocity_m_s'] * 2 * (RO - RI) / MU:.1f}: the case is a pure "
        "diffusion problem without advection, so the analytic solution is the exact "
        "reference and no turbulence or literature scatter enters the comparison.")
    add("")
    add("## 2. Method and tool chain")
    add("")
    add("1. **Reference is machine-verified first.** The closed-form profile is checked "
        "against the axisymmetric momentum equation with a fourth-order stencil, both walls "
        "against no-slip, the closed-form flow rate against independent "
        "`scipy.integrate.quad`, and the analytic wall shears against the global equilibrium "
        "identity. Worst relative residual: "
        f"{self_check['worst_relative']:.2e} (limit {SELF_CHECK_LIMIT:.0e}).")
    add("2. **Geometry is audited before the numerics** on the actual production mesh: total "
        "polygon area versus pi(ro^2-ri^2) and the chord deficit of both straight-edge walls.")
    add("3. **Production refinement study**: `tensorfvm.annular.solve` at "
        f"nr = {', '.join(str(r['nr']) for r in rows)} with ntheta = 4 nr "
        f"(h = (ro-ri)/nr), double precision, scipy CG to a true residual gate of 1e-11.")
    add("4. **Error budget**: L-infinity and volume-weighted L2 velocity error, flow rate, "
        "both wall shears, peak velocity and its radial position, plus the discrete axial "
        "equilibrium residual (pressure source versus total wall traction).")
    add("")
    add("Reproduce with `python scripts/verify_annular_order.py` "
        f"(total wall time {runtime:.1f} s on CPU).")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 Geometry audit")
    add("")
    add("| nr | ntheta | cells | area error | inner perimeter deficit | outer perimeter deficit | min cell area [m2] |")
    add("|---|---|---|---|---|---|---|")
    for g in study["geometry"]:
        add(f"| {g['nr']} | {g['ntheta']} | {g['cells']} | {g['relative_area_error']:+.3e} | "
            f"{g['inner_perimeter_deficit']:+.3e} | {g['outer_perimeter_deficit']:+.3e} | "
            f"{g['min_cell_area_m2']:.3e} |")
    add("")
    add("The area defect decays as h^"
        f"{study['geometry_orders'][-1]:.2f}: the polygonal inner and outer walls are "
        "inscribed chords, so the numerical annulus is *smaller* than the true one by a "
        "second-order geometric term. This is the floor under any second-order error claim "
        "on this mesh family.")
    add("")
    add("### 3.2 Error against the exact solution")
    add("")
    header = "| nr | cells | " + " | ".join(labels[m] for m in metric_names) + " | wall equilibrium | solver residual |"
    add(header)
    add("|" + "---|" * (len(metric_names) + 4))
    for row in rows:
        cells = " | ".join(f"{next(m['value'] for m in row['metrics'] if m['name'] == name):.3e}"
                           for name in metric_names)
        add(f"| {row['nr']} | {row['cells']} | {cells} | "
            f"{row['wall_traction_balance_relative']:.1e} | {row['solver_relative_residual']:.1e} |")
    add("")
    add("### 3.3 Observed order of accuracy")
    add("")
    add("| quantity | orders between successive levels | last |")
    add("|---|---|---|")
    for name in metric_names:
        seq = ", ".join(f"{o:.2f}" for o in orders[name])
        add(f"| {labels[name]} | {seq} | {orders[name][-1]:.2f} |")
    add("")
    add("### 3.4 Integrated quantities at the finest level")
    add("")
    add("| quantity | numerical | exact | relative error |")
    add("|---|---|---|---|")
    add(f"| flow rate Q [m3/s] | {v['flow_rate_m3_s']:.6e} | {v['exact_flow_rate_m3_s']:.6e} | "
        f"{v['flow_rate_error']:+.3e} |")
    add(f"| inner-wall shear tau_i [Pa] | {v['inner_wall_shear_pa']:.6e} | "
        f"{v['exact_inner_wall_shear_pa']:.6e} | "
        f"{(v['inner_wall_shear_pa'] - v['exact_inner_wall_shear_pa']) / v['exact_inner_wall_shear_pa']:+.3e} |")
    add(f"| outer-wall shear magnitude |tau_o| [Pa] | {v['outer_wall_shear_pa']:.6e} | "
        f"{abs(v['exact_outer_wall_shear_pa']):.6e} | "
        f"{(v['outer_wall_shear_pa'] - abs(v['exact_outer_wall_shear_pa'])) / abs(v['exact_outer_wall_shear_pa']):+.3e} |")
    add(f"| peak velocity u_max [m/s] | {v['max_velocity_m_s']:.6e} | {v['exact_max_velocity_m_s']:.6e} | "
        f"{(v['max_velocity_m_s'] - v['exact_max_velocity_m_s']) / v['exact_max_velocity_m_s']:+.3e} |")
    add(f"| peak velocity radius [m] | {v['max_velocity_radius_m']:.6f} | "
        f"{v['exact_max_velocity_radius_m']:.6f} | "
        f"{(v['max_velocity_radius_m'] - v['exact_max_velocity_radius_m']) / v['exact_max_velocity_radius_m']:+.3e} |")
    add("")
    add("Cross-check of the reference itself: `scipy.integrate.quad` gives "
        f"{s['flow_rate_quadrature_m3_s']:.12e} m3/s versus the closed form "
        f"{s['flow_rate_m3_s']:.12e} m3/s, a relative difference of "
        f"{s['closed_form_vs_quadrature_flow_rate_relative']:.1e}.")
    add("")
    add("## 4. Figures")
    add("")
    add("![profile](profile.png)")
    add("")
    add("*Figure 1 — radial velocity profiles at three angular positions against the exact "
        "solution (left) and the pointwise absolute error (right) at the finest level.*")
    add("")
    add("![convergence](convergence.png)")
    add("")
    add("*Figure 2 — log-log error versus spacing h with the nominal slope-2 guide (left) and "
        "the geometric area defect of the polygonal annulus (right).*")
    add("")
    add("![field](field.png)")
    add("")
    add("*Figure 3 — axial-velocity cloud map on the production mesh (left) and the absolute "
        "error field (right); the white disc is the no-slip inner cylinder.*")
    add("")
    add("## 5. Notes, limitations and scope")
    add("")
    add("- **Scope**: this verifies the production diffusion operator and no-slip boundary "
        "handling on a body-fitted curvilinear mesh, in the Stokes limit. It does not "
        "qualify advection, pressure-velocity coupling, turbulence or GPU paths.")
    add("- The straight-edge inner/outer walls are chords, not arcs, so the discrete domain "
        "differs from the analytic one by an O(h^2) area defect; the measured velocity error "
        "contains that geometric term and cannot fall below it on this mesh family. A "
        "curved-wall variant would be needed to isolate the operator error further.")
    add("- Wall shear is read from the exported boundary-face momentum flux, which is the "
        "same quantity the discrete momentum balance uses; it is therefore a property of the "
        "solved system, not an independent post-processing gradient.")
    add("- The peak velocity is a single-cell quantity: its error is already "
        f"{next(m['value'] for m in rows[-1]['metrics'] if m['name'] == 'max_velocity'):.1e} "
        "but its observed order is not monotone (the argmax jumps between cells), so no "
        "order claim is made for it; the smooth L2 and linfin norms carry the order claim.")
    add("- The discrete axial equilibrium residual (pressure source minus total wall traction) "
        "is machine-zero by construction; it is reported as an implementation-consistency "
        "check, not as an accuracy statement.")
    add(f"- Runtime {runtime:.1f} s, linear iterations at the finest level "
        f"{rows[-1]['linear_iterations']}, matrix nonzeros {rows[-1]['matrix_nnz']}.")
    add("")
    add("## 6. Error verdict")
    add("")
    add("| quantity (finest level) | relative error | < 3 % ? |")
    add("|---|---|---|")
    for name in metric_names:
        value = next(m["value"] for m in rows[-1]["metrics"] if m["name"] == name)
        add(f"| {labels[name]} | {value:.3e} | {'yes' if value < ERROR_LIMIT else 'NO'} |")
    add(f"| velocity convergence order | {orders['velocity_l2'][-1]:.2f} (nominal 2) | "
        f"{'yes' if orders['velocity_l2'][-1] >= ORDER_LIMIT else 'NO'} |")
    add("")
    add("Overall: **" + ("PASS" if verdict["passed"] else "FAIL") + "**.")
    add("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="8,16,32,64",
                        help="comma separated radial counts; ntheta = 4 nr")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)
    levels = tuple((int(nr), 4 * int(nr)) for nr in args.levels.split(","))

    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    self_check = reference_self_check()
    if not self_check["passed"]:
        raise RuntimeError(f"analytic reference is not self-consistent: {self_check}")
    study = convergence(levels)
    runtime = time.perf_counter() - start

    if not args.skip_figures:
        figure_profile(study)
        figure_convergence(study)
        figure_field(study)

    verdict = acceptance(study)
    payload = dict(
        case=dict(inner_radius_m=RI, outer_radius_m=RO, pressure_gradient_pa_m=DP_DZ,
                  viscosity_pa_s=MU, analytic_flow_rate_m3_s=analytic_flow_rate(),
                  quadrature_flow_rate_m3_s=quadrature_flow_rate(),
                  analytic_max_velocity_m_s=analytic_max_velocity()[0],
                  analytic_max_velocity_radius_m=analytic_max_velocity()[1],
                  analytic_inner_wall_shear_pa=analytic_wall_shear()[0],
                  analytic_outer_wall_shear_pa=analytic_wall_shear()[1]),
        reference_self_check=self_check,
        study=strip(study),
        acceptance=verdict,
        runtime_s=runtime,
        error_limit=ERROR_LIMIT,
        order_limit=ORDER_LIMIT,
    )
    write_json(OUT / "results.json", jsonable(payload))
    (OUT / "report.md").write_text(render(study, self_check, verdict, runtime))
    print(json.dumps(dict(passed=verdict["passed"], l2_order=verdict["l2_order"],
                          runtime_s=round(runtime, 1),
                          finest=dict(nr=study["rows"][-1]["nr"],
                                      cells=study["rows"][-1]["cells"]),
                          metrics={m["name"]: m["value"] for m in study["rows"][-1]["metrics"]}),
                     indent=2))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())