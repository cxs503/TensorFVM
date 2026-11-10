"""One-dimensional steady/unsteady drift-flux model for vertical gas-liquid pipe flow.

Model
-----
The drift-flux formulation (Wallis 1969; Zuber & Findlay 1965) replaces the two
momentum equations of the two-fluid model by a single mixture momentum balance
plus an algebraic relation between the phase velocities.  With

* ``alpha``        gas volume fraction (void fraction),
* ``j_g, j_l``     superficial (volumetric-flux) velocities of gas and liquid,
* ``j = j_g + j_l`` the mixture volumetric flux,
* ``C0``           distribution parameter,
* ``v_dj``         drift velocity (Wallis' bubble/slug correlations),

the closure used here is

.. math::

    j_g = C_0 \\alpha j + \\alpha (1-\\alpha) v_{dj},

which is the classical drift-flux relation and is *quadratic* in ``alpha``.
Solving it for a given pair ``(j_g, j_l)`` gives the closed-form void fraction

.. math::

    \\alpha = \\frac{B - \\sqrt{B^2 - 4 v_{dj} j_g}}{2 v_{dj}},
    \\qquad B = C_0 (j_g + j_l) + v_{dj},

(the smaller root is the physical branch).  For ``v_dj = 0`` the relation
degenerates to the homogeneous (no-slip) value ``alpha = j_g/(C0 j)``.

Gas is treated as an ideal gas, ``rho_g = p / (R_s T)``, so the gas expands as
the pressure falls along the pipe: this is what makes ``alpha(z)`` non-trivial.
The mixture momentum balance is used in its (standard, quasi-steady) drift-flux
form with the hydrostatic-acceleration term retained and wall friction omitted:

.. math::

    \\frac{dp}{dz} = -\\big[\\alpha \\rho_g + (1-\\alpha)\\rho_l\\big] g ,

which is integrated upward from the inlet pressure.  (Friction is deliberately
absent: the verification target is the *void-fraction profile*, which friction
does not enter; adding ``f j^2`` would only blur the comparison.)

Two independent solvers of the same equations are provided:

``wallis_profile``
    The verification reference.  The void fraction is the *closed form* above and
    the pressure profile is obtained by integrating the momentum ODE with
    ``scipy`` DOP853 at ``rtol = 1e-12`` -- i.e. the ODE part is exact to
    round-off, so the resulting ``alpha(z)`` is the (numerically) exact solution
    of the model.

``solve_drift_flux``
    A finite-volume solver: gas mass is transported conservatively in flux form
    on a uniform cell-centred grid with upwind face values, the pressure being
    integrated from the current void-fraction field by the trapezoidal rule on
    faces.  The scheme is marched in time to a steady state; the steady state is
    the drift-flux algebraic relation, and the discretisation error is what the
    benchmark measures (it must vanish at second order).

Degenerate limit (negative control)
-----------------------------------
``gas_mass_flux = 0`` removes the gas phase entirely: the gas mass flux is
identically zero, ``alpha == 0`` to round-off, and the liquid flows as a single
phase with ``v_l = j_l`` and ``dp/dz = -rho_l g``.

References
----------
G.B. Wallis, *One-Dimensional Two-Phase Flow*, McGraw-Hill (1969).
N. Zuber & J.A. Findlay, *J. Heat Transfer* 87 (1965) 453-468.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "DriftFluxConfig",
    "DriftFluxResult",
    "wallis_void_fraction",
    "wallis_drift_velocity",
    "wallis_profile",
    "solve_drift_flux",
    "homogeneous_void_fraction",
    "compare_profiles",
]

GRAVITY = 9.80665


def wallis_drift_velocity(kind: str, diameter: float, rho_liquid: float,
                          rho_gas: float, sigma: float,
                          gravity: float = GRAVITY) -> float:
    """Wallis (1969) drift velocity for the requested flow regime [m/s].

    ``"bubbly"``      ``1.53 (sigma g drho / rho_l^2)^{1/4}``  (small bubbles)
    ``"churn"``     ``0.35 sqrt(g D drho / rho_l)``         (churn/slug, the
                    standard Wallis value for a tube of diameter ``D``)
    """
    drho = max(rho_liquid - rho_gas, 0.0)
    if kind == "bubbly":
        return 1.53 * (sigma * gravity * drho / rho_liquid ** 2) ** 0.25
    if kind in ("churn", "slug"):
        return 0.35 * math.sqrt(gravity * diameter * drho / rho_liquid)
    raise ValueError(f"unknown drift-velocity correlation {kind!r}")


def wallis_void_fraction(j_g: float, j_l: float, v_dj: float,
                         distribution: float = 1.0) -> float:
    """Closed-form void fraction of the drift-flux relation (see module doc)."""
    j_g = max(float(j_g), 0.0)
    j_l = max(float(j_l), 0.0)
    if j_g <= 0.0:
        return 0.0
    b = distribution * (j_g + j_l) + v_dj
    if v_dj <= 0.0:
        return float(j_g / max(distribution * (j_g + j_l), 1e-300))
    disc = max(b * b - 4.0 * v_dj * j_g, 0.0)
    alpha = (b - math.sqrt(disc)) / (2.0 * v_dj)
    return float(min(max(alpha, 0.0), 1.0))


def homogeneous_void_fraction(j_g: float, j_l: float,
                              distribution: float = 1.0) -> float:
    """No-slip (homogeneous) limit ``alpha = j_g / (C0 (j_g + j_l))``."""
    denom = distribution * (max(j_g, 0.0) + max(j_l, 0.0))
    if denom <= 0.0:
        return 0.0
    return float(min(max(j_g / denom, 0.0), 1.0))


def wallis_gas_flux(alpha: float, j_l: float, rho_g: float, v_dj: float,
                    distribution: float = 1.0) -> float:
    """Gas mass flux ``G_g = rho_g j_g`` implied by ``alpha`` (drift-flux closure).

    Inverting the closure for ``j_g``:

    .. math::

        j_g = \\frac{C_0 \\alpha j_l + \\alpha(1-\\alpha) v_{dj}}
                    {1 - C_0 \\alpha},

    which is the form used by the finite-volume solver for its face fluxes.
    """
    a = min(max(alpha, 0.0), 1.0)
    denom = 1.0 - distribution * a
    if a <= 0.0:
        return 0.0
    if denom <= 1e-12:                      # alpha beyond the model's range
        return float("inf")
    j_g = (distribution * a * j_l + a * (1.0 - a) * v_dj) / denom
    return float(rho_g * j_g)


@dataclass
class DriftFluxConfig:
    """Geometry, fluid properties and boundary data of the pipe test case."""

    length: float = 3.0                 # pipe length [m]
    diameter: float = 0.05              # pipe diameter [m]
    n_cells: int = 200                  # FVM resolution
    p_bottom: float = 3.0e5             # inlet pressure [Pa]
    temperature: float = 293.15         # isothermal [K]
    gas_mass_flux: float = 0.5          # G_g at the inlet [kg/(m^2 s)]
    liquid_superficial_velocity: float = 0.5   # j_l [m/s]
    rho_liquid: float = 998.2           # [kg/m^3]
    mu_liquid: float = 1.002e-3         # [Pa s]
    sigma: float = 0.0728               # [N/m]
    r_gas: float = 287.05               # specific gas constant [J/(kg K)]
    p_vapour: float = 2339.0            # water vapour pressure at 20 C [Pa]
    distribution: float = 1.2           # Zuber-Findlay C0 (churn-turbulent)
    drift_kind: str = "churn"
    gravity: float = GRAVITY

    @property
    def dz(self) -> float:
        return self.length / self.n_cells

    @property
    def liquid_mass_flux(self) -> float:
        return self.rho_liquid * self.liquid_superficial_velocity

    @property
    def v_dj(self) -> float:
        """Drift velocity at the mean pressure (weak density dependence)."""
        rho_g = self.p_bottom / (self.r_gas * self.temperature)
        return wallis_drift_velocity(self.drift_kind, self.diameter,
                                     self.rho_liquid, rho_g, self.sigma,
                                     self.gravity)

    def validate(self) -> None:
        if self.length <= 0.0 or self.diameter <= 0.0:
            raise ValueError("length and diameter must be positive")
        if self.n_cells < 4:
            raise ValueError("need at least 4 cells")
        if self.p_bottom <= 0.0:
            raise ValueError("inlet pressure must be positive")
        if self.gas_mass_flux < 0.0 or self.liquid_superficial_velocity < 0.0:
            raise ValueError("phase fluxes must be non-negative")


@dataclass
class DriftFluxResult:
    """Solution on the cell centres plus the run descriptors."""

    z: np.ndarray
    alpha: np.ndarray
    pressure: np.ndarray
    rho_gas: np.ndarray
    v_gas: np.ndarray
    v_liquid: np.ndarray
    gas_mass_flux_profile: np.ndarray
    liquid_mass_flux_profile: np.ndarray
    config: dict
    steady: bool
    steps: int
    time: float
    residual: float
    gas_mass_drift: float
    liquid_mass_drift: float


# --------------------------------------------------------------------------- #
# reference solution: closed-form void fraction + high-precision momentum ODE
# --------------------------------------------------------------------------- #
def wallis_profile(cfg: DriftFluxConfig, n_samples: int = 2001,
                   rtol: float = 1e-12, atol: float = 1e-6) -> DriftFluxResult:
    """Verification reference ("Wallis analytic") profile.

    The void fraction is the closed form of :func:`wallis_void_fraction`; the
    pressure is integrated with DOP853 at ``rtol=1e-12``, so the ODE part is
    converged to round-off and the result is independent of the FVM grid.
    """
    cfg.validate()
    v_dj = cfg.v_dj
    j_l = cfg.liquid_superficial_velocity
    g_g = cfg.gas_mass_flux

    def alpha_of(p: float) -> float:
        if g_g <= 0.0:
            return 0.0
        rho_g = p / (cfg.r_gas * cfg.temperature)
        return wallis_void_fraction(g_g / rho_g, j_l, v_dj, cfg.distribution)

    def rhs(_z, p):
        p = max(float(p[0]), 1.0)
        a = alpha_of(p)
        rho_g = p / (cfg.r_gas * cfg.temperature)
        return [-(a * rho_g + (1.0 - a) * cfg.rho_liquid) * cfg.gravity]

    z_eval = np.linspace(0.0, cfg.length, n_samples)
    sol = solve_ivp(rhs, (0.0, cfg.length), [cfg.p_bottom], t_eval=z_eval,
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"reference integration failed: {sol.message}")
    p = sol.y[0]
    rho_g = np.maximum(p, 1.0) / (cfg.r_gas * cfg.temperature)
    alpha = np.array([alpha_of(v) for v in p])
    v_gas = np.array([wallis_gas_flux(a, j_l, r, v_dj, cfg.distribution) / r
                      for a, r in zip(alpha, rho_g)])
    v_liquid = np.where(alpha < 1.0,
                        np.maximum(g_g * 0.0 + j_l, 0.0) / np.maximum(1.0 - alpha, 1e-30),
                        0.0)
    gas_mass = alpha * rho_g * v_gas
    liquid_mass = (1.0 - alpha) * cfg.rho_liquid * v_liquid
    return DriftFluxResult(
        z=z_eval, alpha=alpha, pressure=p, rho_gas=rho_g,
        v_gas=v_gas, v_liquid=v_liquid,
        gas_mass_flux_profile=gas_mass,
        liquid_mass_flux_profile=liquid_mass,
        config=asdict(cfg), steady=True, steps=0, time=0.0,
        residual=float(np.max(np.abs(sol.y[0] - cfg.p_bottom)) * 0.0),
        gas_mass_drift=_relative_drift(gas_mass),
        liquid_mass_drift=_relative_drift(liquid_mass),
    )


def _relative_drift(profile: np.ndarray) -> float:
    """``max|f - mean(f)| / mean(f)`` -- a grid-independent mass-closure metric."""
    f = np.asarray(profile, dtype=float)
    if f.size == 0:
        return 0.0
    scale = float(np.mean(np.abs(f)))
    if scale <= 0.0:
        return 0.0
    return float(np.max(np.abs(f - np.mean(f))) / scale)


# --------------------------------------------------------------------------- #
# finite-volume solver
# --------------------------------------------------------------------------- #
def solve_drift_flux(cfg: DriftFluxConfig, t_end: float | None = None,
                     cfl: float = 0.4, max_steps: int = 200_000,
                     steady_tol: float = 1e-11,
                     check_every: int = 200) -> DriftFluxResult:
    """Conservative finite-volume drift-flux solver, marched to steady state.

    Gas mass ``alpha*rho_g`` is the conserved variable:

    .. math::

        (\\alpha\\rho_g)^{n+1}_i = (\\alpha\\rho_g)^{n}_i
            - \\frac{\\Delta t}{\\Delta z}\\big(F_{i+1/2} - F_{i-1/2}\\big),

    with the face gas mass flux from :func:`wallis_gas_flux` evaluated with an
    upwind void fraction and the face pressure.  The pressure is re-integrated
    from the updated void fraction each step (quasi-steady momentum), which is
    what closes the loop between expansion and void fraction.
    """
    cfg.validate()
    n, dz = cfg.n_cells, cfg.dz
    j_l = cfg.liquid_superficial_velocity
    v_dj, c0 = cfg.v_dj, cfg.distribution
    g_g = cfg.gas_mass_flux

    z = (np.arange(n) + 0.5) * dz

    def pressure_of(alpha):
        """Hydrostatic/mixture momentum integrated upward on the faces."""
        a_face = np.empty(n + 1)
        a_face[1:-1] = 0.5 * (alpha[:-1] + alpha[1:])
        a_face[0], a_face[-1] = alpha[0], alpha[-1]
        p = np.empty(n)
        p[0] = cfg.p_bottom - 0.5 * dz * (
            a_face[1] * cfg.p_bottom / (cfg.r_gas * cfg.temperature)
            + (1.0 - a_face[1]) * cfg.rho_liquid) * cfg.gravity
        for i in range(1, n):
            rho_i = p[i - 1] / (cfg.r_gas * cfg.temperature)
            mix_lo = a_face[i] * rho_i + (1.0 - a_face[i]) * cfg.rho_liquid
            mix_hi = a_face[i + 1] * rho_i + (1.0 - a_face[i + 1]) * cfg.rho_liquid
            p[i] = p[i - 1] - dz * 0.5 * (mix_lo + mix_hi) * cfg.gravity
            p[i] = max(p[i], 1.0)
        return p

    def phase_state(alpha, p):
        rho_g = p / (cfg.r_gas * cfg.temperature)
        flux = np.array([wallis_gas_flux(a, j_l, r, v_dj, c0)
                         for a, r in zip(alpha, rho_g)])
        v_g = np.where(alpha > 0.0, flux / np.maximum(alpha * rho_g, 1e-300), 0.0)
        v_l = np.where(alpha < 1.0, j_l / np.maximum(1.0 - alpha, 1e-30), 0.0)
        return rho_g, flux, v_g, v_l

    alpha = np.full(n, 0.0 if g_g <= 0.0 else
                    wallis_void_fraction(g_g * cfg.r_gas * cfg.temperature
                                         / cfg.p_bottom, j_l, v_dj, c0))
    p = pressure_of(alpha)

    rho_g_inlet = cfg.p_bottom / (cfg.r_gas * cfg.temperature)
    alpha_inlet = (0.0 if g_g <= 0.0 else
                   wallis_void_fraction(g_g / rho_g_inlet, j_l, v_dj, c0))

    if g_g <= 0.0:
        rho_g, flux, v_g, v_l = phase_state(alpha, p)
        return DriftFluxResult(
            z=z, alpha=alpha.copy(), pressure=p, rho_gas=rho_g,
            v_gas=v_g, v_liquid=v_l, gas_mass_flux_profile=flux,
            liquid_mass_flux_profile=(1.0 - alpha) * cfg.rho_liquid * v_l,
            config=asdict(cfg), steady=True, steps=0, time=0.0, residual=0.0,
            gas_mass_drift=0.0,
            liquid_mass_drift=_relative_drift((1.0 - alpha) * cfg.rho_liquid * v_l))

    rho_g, flux, v_g, v_l = phase_state(alpha, p)
    flow_through = cfg.length / max(float(np.max(v_g)), 1e-12)
    t_end = (40.0 * flow_through) if t_end is None else float(t_end)

    t, steps = 0.0, 0
    gas_mass = alpha * rho_g
    prev = gas_mass.copy()
    residual = float("inf")
    steady = False
    while t < t_end and steps < max_steps:
        v_max = float(np.max(v_g))
        dt = min(cfl * dz / max(v_max, 1e-12), t_end - t)
        if dt <= 0.0:
            break
        # upwind face gas mass flux (inlet: prescribed gas mass flux)
        face = np.empty(n + 1)
        face[0] = g_g
        face[1:n] = np.where(v_g[:-1] >= 0.0, flux[:-1], flux[1:])
        face[n] = flux[-1]                      # zero-gradient outlet
        gas_mass = gas_mass - dt / dz * (face[1:] - face[:-1])
        # inlet: Dirichlet void fraction (the inlet pressure is fixed, so the
        # inlet gas density -- and hence the conserved gas mass -- is fixed too)
        gas_mass[0] = alpha_inlet * rho_g_inlet
        p = pressure_of(np.clip(gas_mass / rho_g, 0.0, 1.0))
        p[0] = cfg.p_bottom
        rho_g = p / (cfg.r_gas * cfg.temperature)
        alpha = np.clip(gas_mass / np.maximum(rho_g, 1e-300), 0.0, 1.0)
        rho_g, flux, v_g, v_l = phase_state(alpha, p)
        t += dt
        steps += 1
        if steps % check_every == 0:
            scale = max(float(np.max(np.abs(flux))), 1e-300)
            residual = float(np.max(np.abs(gas_mass - prev))) / max(
                float(np.max(np.abs(gas_mass))), 1e-300) / (check_every * dt)
            prev = gas_mass.copy()
            if residual * scale < steady_tol:
                steady = True
                break

    liquid_mass_flux = (1.0 - alpha) * cfg.rho_liquid * v_l
    return DriftFluxResult(
        z=z, alpha=alpha, pressure=p, rho_gas=rho_g, v_gas=v_g, v_liquid=v_l,
        gas_mass_flux_profile=flux, liquid_mass_flux_profile=liquid_mass_flux,
        config=asdict(cfg), steady=steady, steps=steps, time=t,
        residual=residual,
        gas_mass_drift=_relative_drift(flux),
        liquid_mass_drift=_relative_drift(liquid_mass_flux),
    )


def compare_profiles(ref: DriftFluxResult, num: DriftFluxResult,
                     n_sample: int = 41) -> dict:
    """Max relative void-fraction deviation between two profiles at matched z.

    The comparison is made on a common z-grid by interpolating the reference
    (which is grid independent) onto the FVM cell centres.
    """
    z_num = num.z
    a_ref = np.interp(z_num, ref.z, ref.alpha)
    p_ref = np.interp(z_num, ref.z, ref.pressure)
    mask = a_ref > 1e-6
    if not np.any(mask):
        return {"max_rel_error": 0.0, "rms_rel_error": 0.0, "n_samples": 0}
    rel = np.abs(num.alpha[mask] - a_ref[mask]) / a_ref[mask]
    return {
        "max_rel_error": float(np.max(rel)),
        "rms_rel_error": float(np.sqrt(np.mean(rel ** 2))),
        "n_samples": int(mask.sum()),
        "max_abs_pressure_error": float(np.max(np.abs(num.pressure - p_ref))
                                        / max(np.max(p_ref), 1.0)),
    }