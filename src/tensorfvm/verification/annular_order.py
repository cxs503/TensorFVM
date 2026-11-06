"""Design-order verification of the production annular laminar backend.

The case is the classical fully developed laminar annular Poiseuille flow: an
annulus ``ri < r < ro`` with no-slip walls, constant axial pressure gradient
``G = -dp/dz`` and constant molecular viscosity ``mu``.  The exact solution is

    u(r) = G/(4 mu) * (ro^2 - r^2 - K ln(ro/r)),   K = (ro^2 - ri^2)/ln(ro/ri)

with closed-form flow rate

    Q = pi G/(8 mu) * (ro^4 - ri^4 - (ro^2 - ri^2)^2/ln(ro/ri))

and wall shear stresses ``tau = mu du/dr`` whose weighted sum satisfies the
global equilibrium identity ``G*pi*(ro^2-ri^2) = 2 pi ri tau_i - 2 pi ro tau_o``.

``src/tensorfvm/annular.py`` solves exactly this problem with the production
fitted diffusion kernel on a straight-edge polygonal annulus, but it carried no
test and no documented order of accuracy.  This module supplies the missing
verification: the analytic reference is first machine-checked (momentum-equation
residual, no-slip, closed-form vs quadrature flow rate, equilibrium identity) so
that the yardstick itself is not trusted on faith, then the production backend is
refined (``ntheta = 4 nr``, ``h = (ro-ri)/nr``) and the observed convergence order
of every reported quantity is measured.

Nothing here modifies or replaces the solver: it only measures it.  All quantities
are deterministic, CPU-only and computed in double precision.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .core import Metric

# Production defaults of AnnularConfig; kept explicit so the report is reproducible.
RI, RO, DP_DZ, MU = 0.05, 0.10, 0.01, 0.001
REFINEMENT_LEVELS = ((8, 32), (16, 64), (32, 128), (64, 256))
ERROR_LIMIT = 0.03          # repo-wide acceptance limit (verification/core.py)
ORDER_LIMIT = 1.8           # asymptotic second-order band, 10% below nominal 2
SELF_CHECK_LIMIT = 1e-9     # analytic-reference self-consistency (5-pt stencil noise)


def _annular_config(nr, ntheta=None):
    from ..annular import AnnularConfig
    return AnnularConfig(ntheta=4 * nr if ntheta is None else ntheta, nr=nr,
                         inner_radius_m=RI, outer_radius_m=RO,
                         pressure_gradient_pa_m=DP_DZ, viscosity_pa_s=MU)


# --------------------------------------------------------------------------- #
# Closed-form reference and its independent self-verification
# --------------------------------------------------------------------------- #
def analytic_velocity(r, ri=RI, ro=RO, G=DP_DZ, mu=MU):
    """Exact axial velocity of annular Poiseuille flow at radius ``r``."""
    r = np.asarray(r, dtype=float)
    K = (ro * ro - ri * ri) / math.log(ro / ri)
    return G / (4.0 * mu) * (ro * ro - r * r - K * np.log(ro / r))


def analytic_flow_rate(ri=RI, ro=RO, G=DP_DZ, mu=MU):
    """Closed-form volumetric flow rate ``Q = integral u dA``."""
    return math.pi * G / (8.0 * mu) * (
        ro ** 4 - ri ** 4 - (ro ** 2 - ri ** 2) ** 2 / math.log(ro / ri))


def quadrature_flow_rate(ri=RI, ro=RO, G=DP_DZ, mu=MU):
    """Independent ``scipy.integrate.quad`` evaluation of the same integral."""
    import warnings
    from scipy.integrate import quad
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value, _ = quad(lambda r: 2.0 * math.pi * r * analytic_velocity(r, ri, ro, G, mu),
                        ri, ro, epsabs=1e-18, epsrel=1e-13, limit=200)
    return value


def analytic_wall_shear(ri=RI, ro=RO, G=DP_DZ, mu=MU):
    """``(tau_inner, tau_outer)`` = ``mu du/dr`` at the two walls (Pa)."""
    K = (ro * ro - ri * ri) / math.log(ro / ri)
    tau = lambda r: mu * G / (4.0 * mu) * (-2.0 * r + K / r)
    return tau(ri), tau(ro)


def analytic_max_velocity(ri=RI, ro=RO, G=DP_DZ, mu=MU):
    """Peak velocity and the radius where ``du/dr = 0`` (``r = sqrt(K/2)``)."""
    K = (ro * ro - ri * ri) / math.log(ro / ri)
    radius = math.sqrt(K / 2.0)
    return float(analytic_velocity(radius, ri, ro, G, mu)), radius


def reference_self_check(samples=201):
    """Verify the analytic yardstick itself before using it on the solver.

    Four independent statements are checked: (i) the profile satisfies the
    axisymmetric momentum equation ``mu (1/r) d/dr(r du/dr) = G``; (ii) both walls
    are no-slip; (iii) the closed-form flow rate equals independent quadrature;
    (iv) the analytic wall shears close the global equilibrium identity.
    """
    r = np.linspace(RI, RO, samples)
    u = analytic_velocity(r)
    # Five-point fourth-order stencils for du/dr and d2u/dr2, then (u'/r + u'').
    h = r[1] - r[0]
    d1 = (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[:-4]) / (12 * h)
    d2 = (-u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[:-4]) / (12 * h * h)
    momentum = MU * (d1 / r[2:-2] + d2)
    # The production backend adds a positive volumetric source G per unit volume,
    # i.e. it solves mu*lap(u) = -G; the residual is measured against that sign.
    momentum_error = float(np.max(np.abs(momentum + DP_DZ)) / DP_DZ)
    q_closed, q_quad = analytic_flow_rate(), quadrature_flow_rate()
    tau_i, tau_o = analytic_wall_shear()
    area = math.pi * (RO * RO - RI * RI)
    equilibrium = abs(2 * math.pi * RI * tau_i - 2 * math.pi * RO * tau_o - DP_DZ * area) / (DP_DZ * area)
    no_slip = max(abs(analytic_velocity(np.array([RI, RO])))) / analytic_max_velocity()[0]
    checks = {
        "momentum_equation_relative_residual": momentum_error,
        "no_slip_relative_residual": float(no_slip),
        "closed_form_vs_quadrature_flow_rate_relative": abs(q_closed - q_quad) / q_closed,
        "global_equilibrium_identity_relative_residual": float(equilibrium),
        "flow_rate_m3_s": q_closed,
        "flow_rate_quadrature_m3_s": q_quad,
        "max_velocity_m_s": analytic_max_velocity()[0],
        "max_velocity_radius_m": analytic_max_velocity()[1],
        "inner_wall_shear_pa": tau_i,
        "outer_wall_shear_pa": tau_o,
    }
    worst = max(momentum_error, float(no_slip), checks["closed_form_vs_quadrature_flow_rate_relative"], float(equilibrium))
    return dict(checks=checks, worst_relative=worst, passed=bool(worst < SELF_CHECK_LIMIT))


# --------------------------------------------------------------------------- #
# Geometry audit of the production mesh (numerical geometry before numerics)
# --------------------------------------------------------------------------- #
def _mesh_config(nr, ntheta=None, device="cpu"):
    """The exact SolverConfig the production ``annular.solve`` builds for the mesh."""
    from ..solver import SolverConfig
    c = _annular_config(nr, ntheta)
    cfg = SolverConfig(mesh_type="annular", nx=c.ntheta, ny=c.nr,
                       length=2 * c.outer_radius_m, height=2 * c.outer_radius_m,
                       cylinder_radius=None, inlet_velocity=0.01,
                       density=c.density_kg_m3,
                       reynolds=c.density_kg_m3 * 0.01 * 2 * c.outer_radius_m / c.viscosity_pa_s,
                       max_iterations=1, device=device)
    cfg.annular_config = c
    return cfg


def geometry_audit(nr, ntheta=None):
    """Straight-edge polygonal annulus metrics versus the exact annulus."""
    from ..annular import AnnularMesh
    config = _annular_config(nr, ntheta)
    mesh = AnnularMesh(_mesh_config(nr, ntheta))
    volume = mesh.volumes.detach().cpu().numpy()
    area = float(volume.sum())
    exact_area = math.pi * (RO * RO - RI * RI)
    lengths = mesh.face_lengths.detach().cpu().numpy()
    inner = mesh.inner_wall.detach().cpu().numpy()
    outer = mesh.outer_wall.detach().cpu().numpy()
    perimeter_inner = float(lengths[inner].sum())
    perimeter_outer = float(lengths[outer].sum())
    return dict(
        nr=nr, ntheta=int(config.ntheta), cells=int(volume.size),
        area_m2=area, exact_area_m2=exact_area,
        relative_area_error=(area - exact_area) / exact_area,
        inner_perimeter_m=perimeter_inner, outer_perimeter_m=perimeter_outer,
        inner_perimeter_deficit=(perimeter_inner - 2 * math.pi * RI) / (2 * math.pi * RI),
        outer_perimeter_deficit=(perimeter_outer - 2 * math.pi * RO) / (2 * math.pi * RO),
        min_cell_area_m2=float(volume.min()),
        cell_area_ratio=float(volume.max() / volume.min()),
        positive_volumes=bool((volume > 0).all()),
    )


# --------------------------------------------------------------------------- #
# Production run and its error budget
# --------------------------------------------------------------------------- #
def run_case(nr, ntheta=None, device="cpu", linear_backend="scipy"):
    """Solve one refinement level with the production backend and measure errors."""
    from ..annular import solve
    c = _annular_config(nr, ntheta)
    start = time.perf_counter()
    result = solve(c, device=device, linear_backend=linear_backend)
    elapsed = time.perf_counter() - start
    f = result["fields"]
    centers = f["cell_centers_m"]
    volume = f["cell_volumes_m2"].ravel()
    u_num = f["axial_velocity_m_s"].ravel()
    radius = np.linalg.norm(centers, axis=-1).ravel()
    u_ex = analytic_velocity(radius)
    u_ref = analytic_velocity(radius, ri=RI, ro=RO, G=DP_DZ, mu=MU)  # explicit, same call
    if not (np.array_equal(u_ex, u_ref) and np.isfinite(u_num).all()):
        raise RuntimeError("non-finite solution or inconsistent reference field")

    scale = analytic_max_velocity()[0]
    area = float(volume.sum())
    err = u_num - u_ex
    linf = float(np.max(np.abs(err)) / scale)
    l2 = float(math.sqrt(float((err ** 2 * volume).sum()) / area) / scale)

    q_num = float((u_num * volume).sum())
    q_ex = analytic_flow_rate()
    q_error = (q_num - q_ex) / q_ex

    # Discrete wall shear: the exported boundary-face momentum flux carries the
    # no-slip traction (w_wall = 0), in N per unit axial length.
    flux = f["axial_momentum_flux_n_m"].ravel()
    lengths = f["face_lengths_m"].ravel()
    inner = f["inner_wall_face_mask"]
    outer = f["outer_wall_face_mask"]
    tau_i_num = float(flux[inner].sum() / lengths[inner].sum())
    tau_o_num = float(flux[outer].sum() / lengths[outer].sum())
    tau_i_ex, tau_o_ex = analytic_wall_shear()
    # Sign convention: the exported boundary flux is positive along the outward normal on
    # both walls, so the magnitudes are compared (tau_o_ex < 0 is the decelerating gradient).
    mag_i_ex, mag_o_ex = abs(tau_i_ex), abs(tau_o_ex)
    # Global axial equilibrium of the discrete system: pressure source = wall traction.
    wall_balance = abs(flux[inner | outer].sum() - DP_DZ * area) / (DP_DZ * area)

    u_max_num = float(u_num.max())
    u_max_ex, r_max = analytic_max_velocity()
    u_max_location = float(radius[int(np.argmax(u_num))])

    metrics = [
        Metric("velocity_linf", linf, ERROR_LIMIT,
               "max |u_num-u_exact| / max|u_exact|").record(),
        Metric("velocity_l2", l2, ERROR_LIMIT,
               "volume-weighted L2 of velocity error / max|u_exact|").record(),
        Metric("flow_rate", abs(q_error), ERROR_LIMIT,
               "|Q_num-Q_exact|/Q_exact", "1").record(),
        Metric("inner_wall_shear", abs(tau_i_num - mag_i_ex) / mag_i_ex, ERROR_LIMIT,
               "|tau_inner_num|-|tau_exact| over |tau_exact|", "1").record(),
        Metric("outer_wall_shear", abs(tau_o_num - mag_o_ex) / mag_o_ex, ERROR_LIMIT,
               "|tau_outer_num|-|tau_exact| over |tau_exact|", "1").record(),
        Metric("max_velocity", abs(u_max_num - u_max_ex) / u_max_ex, ERROR_LIMIT,
               "|u_max_num-u_max_exact|/u_max_exact", "1").record(),
    ]
    if wall_balance > 1e-8:
        raise RuntimeError(f"discrete pressure/wall-traction equilibrium violated: {wall_balance}")

    return dict(
        nr=nr, ntheta=int(c.ntheta), cells=int(u_num.size),
        spacing_m=(RO - RI) / nr, angular_spacing_rad=2 * math.pi / int(c.ntheta),
        elapsed_s=elapsed,
        linear_iterations=int(result["linear_iterations"]),
        solver_relative_residual=float(result["diagnostics"]["true_production_relative_residual"]),
        matrix_nnz=int(result["diagnostics"]["full_matrix_nnz"]),
        metrics=metrics,
        wall_traction_balance_relative=float(wall_balance),
        radial_shear_correction_max_n_m=float(np.abs(f["nonorthogonal_momentum_flux_n_m"]).max()),
        arrays=dict(radius_m=radius, u_num_m_s=u_num, u_exact_m_s=u_ex, volume_m2=volume,
                    centers_m=centers, u_field_m_s=u_num.reshape(f["axial_velocity_m_s"].shape)),
        values=dict(l2=l2, linf=linf, flow_rate_m3_s=q_num, flow_rate_error=q_error,
                    inner_wall_shear_pa=tau_i_num, outer_wall_shear_pa=tau_o_num,
                    max_velocity_m_s=u_max_num, max_velocity_radius_m=u_max_location,
                    exact_flow_rate_m3_s=q_ex, exact_inner_wall_shear_pa=tau_i_ex,
                    exact_outer_wall_shear_pa=tau_o_ex, exact_max_velocity_m_s=u_max_ex,
                    exact_max_velocity_radius_m=r_max, area_m2=area),
    )


def _order(a, b, ha, hb):
    return math.log(a / b) / math.log(ha / hb) if a > 0 and b > 0 else float("nan")


def convergence(levels=REFINEMENT_LEVELS, device="cpu", linear_backend="scipy"):
    """Run the refinement sequence and report observed orders of accuracy."""
    rows = [run_case(nr, nt, device=device, linear_backend=linear_backend) for nr, nt in levels]
    orders = {}
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        for name in ("velocity_linf", "velocity_l2", "flow_rate", "inner_wall_shear",
                     "outer_wall_shear", "max_velocity"):
            va = next(m["value"] for m in a["metrics"] if m["name"] == name)
            vb = next(m["value"] for m in b["metrics"] if m["name"] == name)
            orders.setdefault(name, []).append(_order(va, vb, a["spacing_m"], b["spacing_m"]))
    geometry = [geometry_audit(nr, nt) for nr, nt in levels]
    geometry_orders = [_order(abs(a["relative_area_error"]), abs(b["relative_area_error"]),
                              1.0 / a["nr"], 1.0 / b["nr"]) for a, b in zip(geometry, geometry[1:])]
    return dict(rows=rows, orders=orders, geometry=geometry, geometry_orders=geometry_orders)


def acceptance(study):
    """Machine verdicts: accuracy at the finest level and asymptotic order."""
    finest = study["rows"][-1]
    verdicts = {m["name"]: bool(m["passed"]) for m in finest["metrics"]}
    l2_order = study["orders"]["velocity_l2"][-1]
    verdicts["asymptotic_second_order"] = bool(l2_order >= ORDER_LIMIT)
    return dict(per_metric=verdicts, l2_order=float(l2_order),
                passed=bool(all(verdicts.values())))