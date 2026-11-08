"""Thin-shear-layer marching RANS for flat-plate and axisymmetric bodies.

The solver integrates the steady, incompressible boundary-layer equations

    u du/dx + v du/dy = U_e dU_e/dx + (1/r) d/dy [ r (nu + nu_t) du/dy ]      (1)
    d(r u)/dx + d(r v)/dy = 0                                                  (2)

together with either the k-omega SST (Menter 2003) or the Spalart-Allmaras
(1994) transport equation(s), marching in ``x`` with a first-order implicit
streamwise discretisation and a second-order wall-normal discretisation.  ``r``
is the metric factor: ``r = 1`` for a 2-D boundary layer, ``r = R(x) + y`` on the
surface of an axisymmetric body of revolution and ``r = y`` in its wake.

Why a marching RANS and not the repository's 3-D solvers
-------------------------------------------------------
``docs/resistance-chain-form-factor`` records, machine-checked, that a resolved
double-body 3-D RANS of the KVLCC2 hull needs ~3e6 cells (~63 h of setup on this
node) and that the native SST path diverges at every affordable spacing.  The
thin-shear-layer form used here is the standard affordable alternative: it keeps
the full turbulence closure and the wall-normal transport, drops only the
streamwise elliptic coupling (which for attached, high-Re hull flow is a
well-documented approximation) and is *verifiable* against the analytic anchors
of :mod:`tensorfvm.rans_ship_resistance.anchors` -- which a 3-D run at an
affordable mesh is not.

Numerics
--------
* Wall-normal grid: ``y = Y(x) (e^{a s} - 1)/(e^a - 1)`` with
  ``s = j/(n_y - 1)``; the domain height follows ``1.35 delta_99`` monotonically,
  so the near-wall spacing stays a small fraction of the viscous sublayer.
* Streamwise term: backward Euler, ``u^n (u^{n+1} - u^n)/dx``.
* Momentum: tridiagonal implicit solve for ``u^{n+1}`` with the eddy viscosity
  frozen at the current Picard iterate; ``v`` follows from the discrete
  continuity equation (2) integrated from the wall (or the symmetry axis).
* Turbulence: implicit diffusion plus linearised sinks (production explicit and
  limited), 4-8 Picard iterations per station.
* Wall conditions: ``u = 0``, ``k = 0``, ``omega_wall = 10 * 6 nu/(beta1 y1^2)``
  (Menter 2003 recommendation), ``nu_tilde = 0`` (SA).  Outer conditions:
  ``u = U_e``, ``k = 1.5 (Tu U_e)^2`` with ``Tu = 1e-3``, ``omega = 5 U_e/L``,
  ``nu_tilde = 3 nu``.
* Trip: ``trip_x`` may be a station index in ``(0, nx)``; laminar until the trip,
  ``k`` and ``omega`` reset to a log-layer-consistent turbulent state there.
  ``trip_x=None`` keeps the laminar branch (the anchor C1/C8 runs);
  ``trip_x=0`` starts turbulent from the first station (the C2 runs).

Diagnostics returned for the acceptance criteria
------------------------------------------------
``cf``         local skin friction referenced to ``U_e^2`` (``2 tau_w/(rho U_e^2)``)
``ctotal``      ``2 tau_w/(rho U_inf^2)`` on the last station's reference speed
``theta``/``delta_star``/``H``  integral thicknesses (with the ``r/R`` weight when
               axisymmetric, so the von Karman identity below holds)
``karman_residual``  residual of the momentum-integral identity
               ``dtheta/dx + (H+2) theta/U_e dU_e/dx + theta/R dR/dx = C_f/2``
               (criterion C7)
``continuity_residual``  global mass-flux residual over the whole march
               (criterion C6)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .closures import (
    SA_CONSTANTS, SST_CONSTANTS, sst_blending, sst_coefficients, sst_eddy_viscosity,
    sst_production, sa_eddy_viscosity, sa_source_terms,
)
from .anchors import blasius_solution

__all__ = [
    "BoundaryLayerResult", "march_boundary_layer", "wake_drag",
]

MODELS = ("sst", "sa", "laminar")


@dataclass
class BoundaryLayerResult:
    """Container for one marching-RANS solution."""

    x: np.ndarray
    ue: np.ndarray
    cf: np.ndarray
    delta_star: np.ndarray
    theta: np.ndarray
    H: np.ndarray
    momentum_thickness: np.ndarray
    y: np.ndarray
    u: np.ndarray
    nu_t: np.ndarray
    k: Optional[np.ndarray]
    omega: Optional[np.ndarray]
    nu_tilde: Optional[np.ndarray]
    continuity_residual: float
    karman_residual: np.ndarray
    max_picard_change: float
    model: str
    axisymmetric: bool
    wall_shear: np.ndarray = field(default_factory=lambda: np.array([]))
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "axisymmetric": self.axisymmetric,
            "x": self.x.tolist(), "ue": self.ue.tolist(), "cf": self.cf.tolist(),
            "delta_star": self.delta_star.tolist(), "theta": self.theta.tolist(),
            "H": self.H.tolist(),
            "continuity_residual": self.continuity_residual,
            "max_picard_change": self.max_picard_change,
        }


def _turbulent_seed(y, ue, nu, x, model):
    """Log-layer consistent turbulent state used to start (or trip) the models.

    A direct modelling choice, documented as such: both closures admit the
    laminar state as an exact solution (Spalart-Allmaras in particular has a
    stable ``nu_tilde = 0`` fixed point), so a turbulence model must be *seeded*
    with a turbulent profile somewhere.  The seed uses the 1/7 power-law friction
    estimate ``C_f = 0.0592 Re_x^{-1/5}`` only to set the friction velocity scale;
    it is not used in any later step, so the skin friction that comes out of the
    march remains an independent prediction.

    ``k = u_tau^2/sqrt(beta*)``, ``omega = max(6 nu/(beta1 y^2), u_tau/(sqrt(beta*) kappa y))``
    and ``nu_tilde = max(3 nu, kappa y u_tau) exp(-(y/delta)^2)`` with
    ``delta = 0.37 x Re_x^{-1/5}``.
    """
    x = max(float(x), 1e-12)
    re_x = max(ue * x / nu, 1.0)
    cf = 0.0592 * re_x ** -0.2
    u_tau = max(ue * np.sqrt(cf / 2.0), 1e-12)
    delta = max(0.37 * x * re_x ** -0.2, 1e-12)
    y = np.asarray(y, dtype=float)
    if model == "sst":
        beta_star = SST_CONSTANTS["beta_star"]
        k = np.full_like(y, u_tau ** 2 / np.sqrt(beta_star))
        omega = np.maximum(6.0 * nu / (SST_CONSTANTS["beta1"] * np.maximum(y, 1e-30) ** 2),
                           u_tau / (np.sqrt(beta_star) * SST_CONSTANTS["kappa"]
                                    * np.maximum(y, 1e-30)))
        k[0] = 0.0
        return k, omega
    decay = np.exp(-(y / delta) ** 2)
    nu_tilde = np.maximum(3.0 * nu, SST_CONSTANTS["kappa"] * y * u_tau)
    nu_tilde = np.maximum(3.0 * nu, nu_tilde * decay)
    nu_tilde[0] = 0.0
    return nu_tilde


def _stretch(n: int, a: float) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return (np.exp(a * s) - 1.0) / (np.exp(a) - 1.0)


def _solve_tridiag(lower, diag, upper, rhs):
    """Solve a tridiagonal system.

    ``lower[j]`` multiplies ``phi[j-1]`` and ``upper[j]`` multiplies ``phi[j+1]``
    (the first entry of each is unused).
    """
    from scipy.linalg import solve_banded
    ab = np.zeros((3, len(diag)))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def march_boundary_layer(x: np.ndarray, ue: np.ndarray, nu: float, *,
                         model: str = "sst", axisymmetric: bool = False,
                         radius: Optional[np.ndarray] = None,
                         wake: bool = False, ny: int = 128, stretching: float = 8.0,
                         trip_x: Optional[int] = 0, tu: float = 1e-3,
                         picard: int = 6, picard_tol: float = 1e-6,
                         under_relaxation: float = 0.7,
                         wall_omega_factor: float = 10.0,
                         seed=None, turbulent_start: bool = False) -> BoundaryLayerResult:
    """March the thin-shear-layer RANS along ``x``.

    Parameters
    ----------
    x, ue : arrays
        Station positions [m] (strictly increasing) and edge velocity ``U_e`` [m/s].
    nu : float
        Molecular viscosity [m^2/s].
    model : {'sst', 'sa', 'laminar'}
        Closure; ``laminar`` keeps ``nu_t = 0`` and must reproduce Blasius.
    axisymmetric : bool
        Use the axisymmetric metric factor ``r``.
    radius : array, optional
        Body radius ``R(x)``; required when ``axisymmetric and not wake``.
    wake : bool
        Wake mode: no wall, symmetry at ``y = 0``, ``r = y``.
    ny : int
        Number of wall-normal nodes.
    trip_x : int or None
        Station index where turbulence is injected; ``None`` keeps the solution
        laminar, ``0`` starts turbulent from the first station.
    """
    x = np.asarray(x, dtype=float)
    ue = np.asarray(ue, dtype=float)
    if x.ndim != 1 or ue.shape != x.shape:
        raise ValueError("x and ue must be 1-D arrays of equal length")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if model not in MODELS:
        raise ValueError("model must be one of %s" % (MODELS,))
    if nu <= 0 or not np.isfinite(nu):
        raise ValueError("nu must be positive and finite")
    if axisymmetric and not wake:
        if radius is None:
            raise ValueError("axisymmetric surface marching requires radius(x)")
        radius = np.asarray(radius, dtype=float)
        if radius.shape != x.shape or np.any(radius <= 0):
            raise ValueError("radius must be positive and match x")
    if ny < 16:
        raise ValueError("ny must be >= 16")

    nx = len(x)
    sigma = _stretch(ny, stretching)
    gradient_ue = np.gradient(ue, x)

    # --- initial station ---------------------------------------------------
    delta0 = 5.0 * np.sqrt(nu * x[0] / ue[0]) if x[0] > 0 else 1e-3
    span = max(x[-1] - x[0], 1e-12)
    re_span = max(float(np.mean(ue)) * span / nu, 1.0)
    # a domain height sized for the *final* boundary layer keeps the wall-normal
    # grid fixed over the march: remeshing at every station is what destroys the
    # discrete mass balance (measured: 100% continuity residual at nx=160)
    y_estimate = 0.37 * span * re_span ** -0.2
    Y = max(2.5 * y_estimate, 2.5 * delta0)
    y = Y * sigma
    blasius = blasius_solution(eta_max=12.0)
    eta = y * np.sqrt(ue[0] / (nu * max(x[0], 1e-12)))
    u = ue[0] * np.interp(eta, blasius["eta"], blasius["fp"])
    u[0] = 0.0
    u[-1] = ue[0]
    if turbulent_start and model != "laminar":
        # turbulent restart: give the closure the velocity profile its own log
        # layer implies, otherwise a model that admits the laminar state as a
        # fixed point (Spalart-Allmaras) can stay on it
        re0 = max(ue[0] * x[0] / nu, 1.0)
        delta_t = 0.37 * x[0] * re0 ** -0.2
        u = np.where(y < delta_t, ue[0] * np.clip(y / delta_t, 0.0, 1.0) ** (1.0 / 7.0), ue[0])
        u[0] = 0.0 if not wake else u[1]
        u[-1] = ue[0]
    if seed is not None:
        # start from a supplied profile (used to hand a boundary layer over to a
        # wake march): the seed arrays are interpolated onto the initial grid
        seed_y, seed_u = np.asarray(seed[0], dtype=float), np.asarray(seed[1], dtype=float)
        order = np.argsort(seed_y)
        u = np.interp(y, seed_y[order], seed_u[order])
        u[0] = 0.0 if not wake else u[1]
        u[-1] = ue[0]

    k = np.full(ny, 1.5 * (tu * ue[0]) ** 2)
    omega = np.full(ny, 5.0 * ue[0] / max(x[0], 1e-6))
    nu_tilde = np.full(ny, 3.0 * nu)
    nu_t = np.zeros(ny)
    if model == "sst" and trip_x is not None:
        k, omega = _turbulent_seed(y, ue[0], nu, x[0], "sst")
        omega[0] = wall_omega_factor * 6.0 * nu / (SST_CONSTANTS["beta1"] * y[1] ** 2)
        nu_t = sst_eddy_viscosity(k, omega, np.gradient(u, y), np.ones(ny))
    elif model == "sa" and trip_x is not None:
        nu_tilde = _turbulent_seed(y, ue[0], nu, x[0], "sa")
        nu_t = sa_eddy_viscosity(nu_tilde, nu)

    cf = np.zeros(nx)
    theta = np.zeros(nx)
    delta_star = np.zeros(nx)
    h_shape = np.zeros(nx)
    wall_shear = np.zeros(nx)
    theta = np.zeros(nx)
    karman = np.zeros(nx)
    max_change_overall = 0.0
    mass_defect_prev = None
    continuity_residual = 0.0

    store_profiles = None
    nu_t_max_history = np.zeros(nx)
    transported_max_history = np.zeros(nx)
    continuity_history = np.zeros(nx)
    picard_history = np.zeros(nx)
    for n in range(nx):
        if n > 0:
            dx = x[n] - x[n - 1]
            # --- domain height: follow 1.35 delta_99, never shrink ----------
            inside = np.where(u < 0.99 * ue[n - 1])[0]
            delta99 = y[inside[-1]] * 1.05 if inside.size and inside[-1] > 0 else y[-1]
            Y_new = Y if delta99 < 0.7 * Y else 2.5 * delta99
            y_new = Y_new * sigma
            u = np.interp(y_new, y, u)
            k = np.interp(y_new, y, k)
            omega = np.interp(y_new, y, omega)
            nu_tilde = np.interp(y_new, y, nu_tilde)
            nu_t = np.interp(y_new, y, nu_t)
            y, Y = y_new, Y_new
        else:
            dx = None

        if wake:
            u[0] = u[1]
            u[-1] = ue[n]
            k[0] = k[1]
            omega[0] = omega[1]
            nu_tilde[0] = nu_tilde[1]
        else:
            u[0] = 0.0
            u[-1] = ue[n]
            k[0] = 0.0
            omega[0] = wall_omega_factor * 6.0 * nu / (SST_CONSTANTS["beta1"] * y[1] ** 2)
            omega[-1] = 5.0 * ue[n] / max(x[n] - x[0], 1e-6) if x[n] > x[0] else omega[-1]
            k[-1] = 1.5 * (tu * ue[n]) ** 2
            nu_tilde[0] = 0.0
            nu_tilde[-1] = 3.0 * nu

        # --- tripping ---------------------------------------------------
        if n > 0 and trip_x is not None and n == trip_x and model != "laminar":
            if model == "sst":
                k_seed, omega_seed = _turbulent_seed(y, ue[n], nu, x[n], "sst")
                k[1:-1] = np.maximum(k[1:-1], k_seed[1:-1])
                omega[1:-1] = np.maximum(omega[1:-1], omega_seed[1:-1])
            else:
                nu_tilde_seed = _turbulent_seed(y, ue[n], nu, x[n], "sa")
                nu_tilde[1:-1] = np.maximum(nu_tilde[1:-1], nu_tilde_seed[1:-1])

        if n == 0:
            # diagnostics on the initial profile
            pass
        else:
            u_old = u.copy()
            change = np.inf
            for _ in range(picard):
                if model == "laminar":
                    nu_t = np.zeros(ny)
                u_new, v = _solve_station(y, u, u_old, dx, ue[n], gradient_ue[n],
                                          nu, nu_t, wake, axisymmetric,
                                          radius[n] if axisymmetric and not wake else None)
                change_u = np.max(np.abs(u_new - u)) / max(np.max(np.abs(u_new)), 1e-30)
                relax = 1.0 if model == "laminar" else under_relaxation
                u = u + relax * (u_new - u)
                if not wake:
                    u[0] = 0.0
                du_dy = np.gradient(u, y)
                strain = np.abs(du_dy)
                if model == "sst":
                    k_new, omega_new, nu_t_new = _solve_sst(y, u, v, dx, ue[n], nu, k,
                                                            omega, strain, wake, nu_t)
                elif model == "sa":
                    nu_tilde_new, nu_t_new = _solve_sa(y, u, v, dx, ue[n], nu, nu_tilde,
                                                       strain, wake)
                else:
                    nu_t_new = np.zeros(ny)
                scale_t = max(np.max(np.abs(nu_t_new)), 1e-30)
                change_t = np.max(np.abs(nu_t_new - nu_t)) / scale_t
                nu_t = nu_t + relax * (nu_t_new - nu_t)
                if model == "sst":
                    k = k + relax * (k_new - k)
                    omega = omega + relax * (omega_new - omega)
                elif model == "sa":
                    nu_tilde = nu_tilde + relax * (nu_tilde_new - nu_tilde)
                change = max(change_u, change_t)
                if change < picard_tol:
                    break
            max_change_overall = max(max_change_overall, change)
            picard_history[n] = change

        du_dy = np.gradient(u, y)
        tau_w = nu * du_dy[0] if not wake else 0.0
        wall_shear[n] = tau_w
        cf[n] = 2.0 * tau_w / ue[n] ** 2
        metric = (np.ones(ny) if not axisymmetric else
                  (y / np.maximum(y + 0.0, 1e-30) if wake else (radius[n] + y) / radius[n]))
        if axisymmetric and wake:
            metric = np.ones(ny)  # wake: r = y, and theta is r-weighted separately
            weight = y
        else:
            weight = metric
        u_norm = u / ue[n]
        theta[n] = np.trapezoid(u_norm * (1.0 - u_norm) * weight, y)
        delta_star[n] = np.trapezoid((1.0 - u_norm) * weight, y)
        h_shape[n] = delta_star[n] / theta[n] if theta[n] > 0 else np.nan

        # global mass-flux defect (continuity diagnostic)
        if axisymmetric and wake:
            defect = np.trapezoid((ue[n] - u) * y, y)
        elif axisymmetric:
            defect = np.trapezoid((ue[n] - u) * (radius[n] + y), y)
        else:
            defect = np.trapezoid(ue[n] - u, y)

        if n > 0:
            v_edge = _edge_velocity(y, u, u_old, dx, axisymmetric, wake,
                                    radius[n] if axisymmetric and not wake else None)
            r_edge = _metric(np.array([y[-1]]), radius[n] if axisymmetric and not wake else None,
                             wake, axisymmetric)[0]
            # momentum-integral continuity: the growth rate of the velocity
            # defect equals the edge entrainment plus the acceleration term
            d_due = gradient_ue[n] * np.trapezoid(_metric(y, radius[n] if axisymmetric and not wake else None,
                                                          wake, axisymmetric), y)
            expected = ((defect - mass_defect_prev) / dx - d_due) / r_edge
            scale = max(abs(expected), abs(v_edge), 1e-12)
            continuity_history[n] = abs(expected - v_edge) / scale
            continuity_residual = max(continuity_residual, continuity_history[n])
        mass_defect_prev = defect
        nu_t_max_history[n] = np.max(nu_t)
        transported_max_history[n] = (np.max(k) if model == "sst" else
                                      np.max(nu_tilde) if model == "sa" else 0.0)
        store_profiles = (y.copy(), u.copy(), nu_t.copy(), k.copy(), omega.copy(), nu_tilde.copy())

    # --- von Karman momentum identity ------------------------------------
    dtheta_dx = np.gradient(theta, x)
    due_dx = np.gradient(ue, x)
    radius_factor = np.zeros(nx)
    if axisymmetric and not wake:
        radius_factor = theta / np.maximum(radius, 1e-30) * np.gradient(radius, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        karman = (dtheta_dx + (h_shape + 2.0) * theta / np.where(ue > 0, ue, np.nan) * due_dx
                  + radius_factor - 0.5 * cf)

    y_f, u_f, nut_f, k_f, om_f, nt_f = store_profiles
    return BoundaryLayerResult(
        x=x, ue=ue, cf=cf, delta_star=delta_star, theta=theta, H=h_shape,
        momentum_thickness=theta, y=y_f, u=u_f, nu_t=nut_f,
        k=k_f if model == "sst" else None,
        omega=om_f if model == "sst" else None,
        nu_tilde=nt_f if model == "sa" else None,
        continuity_residual=continuity_residual, karman_residual=karman,
        max_picard_change=max_change_overall, model=model,
        axisymmetric=axisymmetric, wall_shear=wall_shear,
        extra={"sigma": sigma, "Y": Y, "ny": float(ny),
               "nu_t_max_history": nu_t_max_history,
               "transported_max_history": transported_max_history,
               "continuity_history": continuity_history,
               "picard_history": picard_history},
    )


def _metric(y, radius, wake, axisymmetric):
    if not axisymmetric:
        return np.ones_like(y)
    if wake:
        return y
    return radius + y


def _solve_station(y, u, u_old, dx, ue, due_dx, nu, nu_t, wake, axisymmetric, radius):
    """Implicit streamwise momentum solve; returns ``(u, v)``."""
    ny = len(y)
    dy_half = np.diff(y)
    cell = np.empty(ny)
    cell[1:-1] = 0.5 * (dy_half[:-1] + dy_half[1:])
    cell[0] = 0.5 * dy_half[0]
    cell[-1] = 0.5 * dy_half[-1]
    r = _metric(y, radius, wake, axisymmetric)
    G = nu + nu_t
    G_face = 0.5 * (G[:-1] + G[1:])
    r_face = 0.5 * (r[:-1] + r[1:])

    lower = np.zeros(ny)
    diag = np.zeros(ny)
    upper = np.zeros(ny)
    rhs = np.zeros(ny)

    # convection v: use previous iterate v (from continuity of previous iterate)
    v = _continuity_velocity(y, u, u_old, dx, axisymmetric, wake, radius)

    for j in range(1, ny - 1):
        # trapezoidal (Crank-Nicolson) streamwise convection, linearised on the
        # Picard iterate ``u``
        coeff = 0.5 * (max(u_old[j], 0.0) + max(u[j], 0.0)) / dx
        coeff = max(coeff, 1e-12)
        diag[j] += coeff
        rhs[j] += coeff * u_old[j]
        if v[j] >= 0:
            diag[j] += v[j] / dy_half[j - 1]
            lower[j] += -v[j] / dy_half[j - 1]
        else:
            diag[j] += -v[j] / dy_half[j]
            upper[j] += v[j] / dy_half[j]
        d_lo = r_face[j - 1] * G_face[j - 1] / dy_half[j - 1]
        d_hi = r_face[j] * G_face[j] / dy_half[j]
        diag[j] += (d_lo + d_hi) / (r[j] * cell[j])
        lower[j] += -d_lo / (r[j] * cell[j])
        upper[j] += -d_hi / (r[j] * cell[j])
        rhs[j] += due_dx * ue

    # boundary rows
    if wake:
        # symmetry at y=0 : u_0 = u_1  -> row 0: diag - lower
        diag[0] = 1.0
        upper[0] = -1.0
        rhs[0] = 0.0
    else:
        diag[0] = 1.0
        upper[0] = 0.0
        rhs[0] = 0.0
    diag[-1] = 1.0
    lower[-1] = 0.0
    rhs[-1] = ue

    sol = _solve_tridiag(lower, diag, upper, rhs)
    return sol, v


def _continuity_velocity(y, u, u_old, dx, axisymmetric, wake, radius):
    """Discrete continuity: integrate ``d(r u)/dx + d(r v)/dy = 0`` from the wall."""
    r = _metric(y, radius, wake, axisymmetric)
    integrand = r * (u - u_old) / dx
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(y))])
    v = -cumulative / r
    v[0] = 0.0 if not wake else 0.0
    return v


def _edge_velocity(y, u, u_old, dx, axisymmetric, wake, radius):
    r = _metric(y, radius, wake, axisymmetric)
    integrand = r * (u - u_old) / dx
    total = np.trapezoid(integrand, y)
    return -total / r[-1]


def _solve_sst(y, u, v, dx, ue, nu, k, omega, strain, wake, nu_t):
    """Implicit k and omega transport with SST-2003 closure."""
    ny = len(y)
    dy_half = np.diff(y)
    cell = np.empty(ny)
    cell[1:-1] = 0.5 * (dy_half[:-1] + dy_half[1:])
    cell[0] = 0.5 * dy_half[0]
    cell[-1] = 0.5 * dy_half[-1]
    nu_t = np.maximum(nu_t, 0.0)
    d = np.maximum(y, 1e-30)
    dk = np.gradient(k, y)
    dom = np.gradient(omega, y)
    blend = sst_blending(k, omega, dk, dom, d, nu)
    coef = sst_coefficients(blend["f1"])
    nu_t_new = sst_eddy_viscosity(k, omega, strain, blend["f2"])
    production = sst_production(nu_t_new, strain, k, omega)

    # ---- k equation
    gamma_k = nu + coef["sigma_k"] * nu_t_new
    k_new = _transport(y, cell, dy_half, u, v, dx, gamma_k, k,
                       source=production,
                       sink=SST_CONSTANTS["beta_star"] * np.maximum(omega, 1e-30),
                       wall_zero=not wake, symmetry=wake,
                       outer_value=1.5 * (1e-3 * ue) ** 2)
    k_new = np.maximum(k_new, 0.0)

    # ---- omega equation
    gamma_w = nu + coef["sigma_omega"] * nu_t_new
    source_w = coef["gamma"] * production / np.maximum(nu_t_new, 1e-30)
    cross = 2.0 * (1.0 - blend["f1"]) * SST_CONSTANTS["sigma_omega2"] / np.maximum(omega, 1e-30) * dk * dom
    sink_w = 2.0 * coef["beta"] * np.maximum(omega, 1e-30)
    rhs_w = coef["beta"] * np.maximum(omega, 1e-30) ** 2
    omega_new = _transport(y, cell, dy_half, u, v, dx, gamma_w, omega,
                           source=source_w + cross + rhs_w, sink=sink_w,
                           wall_zero=False, symmetry=wake,
                           outer_value=5.0 * ue / max(y[-1], 1e-6))
    omega_new = np.maximum(omega_new, 1e-20)
    if not wake:
        omega_new[0] = 10.0 * 6.0 * nu / (SST_CONSTANTS["beta1"] * y[1] ** 2)
    return k_new, omega_new, nu_t_new


def _solve_sa(y, u, v, dx, ue, nu, nu_tilde, strain, wake):
    ny = len(y)
    dy_half = np.diff(y)
    cell = np.empty(ny)
    cell[1:-1] = 0.5 * (dy_half[:-1] + dy_half[1:])
    cell[0] = 0.5 * dy_half[0]
    cell[-1] = 0.5 * dy_half[-1]
    d = np.maximum(y, 1e-30)
    terms = sa_source_terms(nu_tilde, strain, d, nu)
    nu_t_new = sa_eddy_viscosity(nu_tilde, nu)
    gamma = (nu + np.maximum(nu_tilde, 0.0)) / SA_CONSTANTS["sigma"]
    grad = np.gradient(np.maximum(nu_tilde, 0.0), y)
    cb2_term = SA_CONSTANTS["cb2"] / SA_CONSTANTS["sigma"] * grad ** 2
    production_coefficient = np.where(
        np.abs(nu_tilde) > 1e-30, terms["production"] / np.where(np.abs(nu_tilde) > 1e-30, nu_tilde, 1.0),
        SA_CONSTANTS["cb1"] * terms["S_tilde"])
    # Newton linearisation of the quadratic destruction c_w1 f_w (nu_tilde/d)^2:
    # d/d(nu_tilde) = 2 c_w1 f_w nu_tilde/d^2, with the full quadratic returned to
    # the right-hand side.  A Picard factor of 1 instead of 2 does not converge to
    # the turbulent branch (measured: the field collapses to the trivial
    # nu_tilde = 0 root); see docs/RANS_verification.md.
    phi_old = np.maximum(nu_tilde, 0.0)
    new = _transport(y, cell, dy_half, u, v, dx, gamma, nu_tilde,
                     source=cb2_term + terms["destruction_coefficient"] * phi_old ** 2,
                     sink=None, production_linear=production_coefficient,
                     destructive_linear=2.0 * terms["destruction_coefficient"] * phi_old,
                     wall_zero=not wake, symmetry=wake, outer_value=3.0 * nu)
    new = np.maximum(new, 0.0)
    return new, nu_t_new


def _transport(y, cell, dy_half, u, v, dx, gamma, phi, *, source, sink=None,
               production_linear=None, destructive_linear=None, wall_zero=False,
               symmetry=False, outer_value=0.0):
    """Generic implicit wall-normal transport for a scalar on the marching grid.

    ``source`` is explicit [per unit volume], ``sink`` an explicit sink
    coefficient (multiplied by ``phi`` and treated implicitly), and
    ``production_linear``/``destructive_linear`` the linearised production and
    destruction coefficients of the SA form.
    """
    ny = len(y)
    lower = np.zeros(ny)
    diag = np.zeros(ny)
    upper = np.zeros(ny)
    rhs = np.zeros(ny)
    gamma_face = 0.5 * (gamma[:-1] + gamma[1:])
    # dual-time stabilisation: the streamwise march is a pseudo-time advance, so
    # a local pseudo-time term with the local residence time dx/u damps the
    # stiffness of the destruction term without changing the converged solution
    pseudo_inverse = max(np.max(u), 1e-12) / dx
    for j in range(1, ny - 1):
        coeff = max(u[j], 1e-12) / dx
        diag[j] += coeff + pseudo_inverse
        rhs[j] += (coeff + pseudo_inverse) * phi[j]
        if v[j] >= 0:
            diag[j] += v[j] / dy_half[j - 1]
            lower[j] += -v[j] / dy_half[j - 1]
        else:
            diag[j] += -v[j] / dy_half[j]
            upper[j] += v[j] / dy_half[j]
        d_lo = gamma_face[j - 1] / dy_half[j - 1]
        d_hi = gamma_face[j] / dy_half[j]
        diag[j] += (d_lo + d_hi) / cell[j]
        lower[j] += -d_lo / cell[j]
        upper[j] += -d_hi / cell[j]
        rhs[j] += source[j]
        if sink is not None:
            diag[j] += sink[j]
        if destructive_linear is not None:
            diag[j] += destructive_linear[j]
        if production_linear is not None:
            diag[j] -= production_linear[j]
    if symmetry:
        diag[0] = 1.0
        upper[0] = -1.0
        rhs[0] = 0.0
    elif wall_zero:
        diag[0] = 1.0
        upper[0] = 0.0
        rhs[0] = 0.0
    else:
        diag[0] = 1.0
        upper[0] = 0.0
        rhs[0] = phi[0]
    diag[-1] = 1.0
    lower[-1] = 0.0
    rhs[-1] = outer_value
    return _solve_tridiag(lower, diag, upper, rhs)


def wake_drag(result: BoundaryLayerResult, nu: float, wake_result: BoundaryLayerResult,
              u_inf: float, axisymmetric: bool) -> float:
    """Drag from the far-wake momentum defect of a wake solution."""
    y = wake_result.y
    u = wake_result.u
    if axisymmetric:
        return 2.0 * np.pi * np.trapezoid(u * (u_inf - u) * y, y) / u_inf ** 2
    return np.trapezoid(u * (u_inf - u), y) / u_inf ** 2