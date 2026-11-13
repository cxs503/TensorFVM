"""Round-7 diagnostic for ``CASE-FVM-PERIODIC-HILLS``: SA closure vs the channel reference.

Why this module exists
----------------------
Round 6 (docs/case-periodic-hills/ROUND6.md) established two facts about the
Re_b = 2800 hill run and left one question open:

* fact 1 -- the SA closure equilibrates (``nu_t/nu`` saturates at 44.1, asymptote
  45.05), so the round-5 growth 2.1 -> 21.1 was a transient, not a divergence;
* fact 2 -- with that closure the hill run is *over-diffusive*: ``x_r/H = 4.07``
  (-23.8 % vs Breuer's 5.34), the separation point is pushed from ``0.227`` to
  ``0.83`` and ``u/U_b`` at ``y/H = 0.5`` is 57 % low in the worst station;
* open question -- is ``nu_t/nu = 44`` a *defect of the implemented closure*, or
  the correct answer of a correct model on a wall-under-resolved mesh (first
  cell at ``y+ ~ 4`` for ``ny = 64``, see
  :func:`tensorfvm.verification.periodic_hill_round6.channel_y_plus_requirement`)?

The 2D hill run cannot answer that question: it mixes model error, grid error
and a statistical-convergence budget in one number.  This module separates the
model from the discretisation by carrying the *same* SA coefficients
(:class:`tensorfvm.periodic_hill_turbulence.SpalartAllmarasConfig`, imported,
not re-typed) into a problem that has an external reference: the fully developed
plane channel, whose eddy-viscosity field follows from the total-stress balance
of the log layer (Pope, *Turbulent Flows*, 2000, Sec. 7.1.2)

    tau/rho = (nu + nu_t) dU/dy = u_tau^2 (1 - y/h)      (linear total stress)
    =>  nu_t^ref(y) = kappa u_tau y (1 - y/h)            (log layer, y+ >> 1)

so both the *level* (peak ``nu_t/nu``) and the *shape* (log-layer slope
``kappa``) of the closure can be graded against literature values instead of
against itself.

What is solved
--------------
A steady 1D RANS channel on ``y in [0, h]`` (``h`` = channel half-height, which
for this solver is ``channel_top_m / 2`` -- the hill case's upper wall sits at
``3.036 H``, so the equivalent channel has ``h = 1.518 H``):

    d/dy [ (nu + nu_t) dU/dy ] = -G,      U(0) = U(h) = 0,   G = u_tau^2 / h

coupled to the SA transport equation in its *equilibrium* form (the channel is
fully developed, so the convective and streamwise-diffusive terms of a
one-dimensional column vanish identically):

    0 = cb1 S~ nu~ - cw1 fw nu~^2/d^2 + (1/sigma) d/dy [(nu + nu~) d nu~/dy]

Both are marched in pseudo-time with the solver's own coefficients and the same
implicit-destruction linearisation the 2D code uses, so a discrepancy between
this column and the 2D run is attributable to the 2D environment (grid, flux,
statistics) rather than to a different model.

Everything is numpy-only on <= 400 cells: the whole report costs well under a
second, i.e. it can gate CI without touching the run budget.
"""

from __future__ import annotations

import math

import numpy as np

from ..periodic_hill_turbulence import (
    SpalartAllmarasConfig,
    eddy_viscosity,
)

__all__ = [
    "LOG_KAPPA_REFERENCE",
    "LOG_B_REFERENCE",
    "stretched_cell_centres",
    "reference_eddy_viscosity",
    "channel_half_height_for_hill_case",
    "wall_resolution_plan",
    "solve_channel_column",
    "log_law_fit",
    "equilibrium_report",
]

#: von Karman constant and additive constant of the standard log law
#: ``u+ = (1/kappa) ln y+ + B`` (Pope 2000, Sec. 7.1.2; the *same* pair round 5
#: gated the channel on -- imported by value here because this module must stay
#: importable without the round-5 driver).
LOG_KAPPA_REFERENCE = 0.41
LOG_B_REFERENCE = 5.2

#: the hill benchmark's upper wall height in units of the hill height
#: (ERCOFTAC UFR 3-30 geometry ``3.036 H``; the solver default
#: ``PeriodicHillConfig.channel_top_m``).
CHANNEL_TOP_OVER_H = 3.036


def channel_half_height_for_hill_case(channel_top=CHANNEL_TOP_OVER_H):
    """Half-height of the plane channel equivalent to the hill case's duct."""
    return 0.5 * float(channel_top)


def stretched_cell_centres(ny, h, growth=1.0):
    """Cell-centre ``y`` for ``ny`` cells spanning ``[0, h]`` with symmetric stretching.

    ``growth = 1`` is uniform spacing.  Otherwise the spacing at the wall is
    ``delta_0`` and each successive cell is ``growth`` times thicker, the profile
    being mirrored about the mid-plane so both walls see ``delta_0`` (the
    standard wall-resolved channel mesh).  Returned together with ``delta_0``.
    """
    ny = int(ny)
    h = float(h)
    growth = float(growth)
    if ny < 4 or ny % 2 != 0:
        raise ValueError(f"ny must be an even integer >= 4, got {ny!r}")
    if not (math.isfinite(growth) and growth > 0.0):
        raise ValueError(f"invalid growth {growth!r}")
    nh = ny // 2
    if growth == 1.0:
        dy = np.full(nh, h / nh)
    else:
        # half-channel: delta_0 * (1 + r + ... + r^(nh-1)) = h
        ratios = growth ** np.arange(nh)
        dy = ratios * (h / ratios.sum())
    y = np.concatenate([np.cumsum(dy) - 0.5 * dy,
                        h + (np.cumsum(dy[::-1]) - 0.5 * dy[::-1])])
    return y, float(dy[0])


def reference_eddy_viscosity(y, u_tau, kappa=LOG_KAPPA_REFERENCE, h=None):
    """``nu_t^ref = kappa u_tau y (1 - y/h)`` -- the log-layer total-stress result.

    ``h`` is the wall-to-centre distance (the channel *half* height): the total
    stress decays linearly from ``u_tau^2`` at the wall to zero at the centre, so
    ``tau = u_tau^2 (1 - y/h)`` and ``nu_t = tau / (dU/dy) = kappa u_tau y (1 - y/h)``
    with the log-layer shear ``dU/dy = u_tau/(kappa y)`` (Pope 2000, Sec. 7.1.2).
    """
    y = np.asarray(y, dtype=float)
    h = float(h) if h is not None else float(y.max())
    return kappa * u_tau * y * (1.0 - y / h)


def wall_resolution_plan(reynolds=2800.0, bulk=1.0, u_tau=0.06,
                         channel_top=CHANNEL_TOP_OVER_H, y_plus_target=1.0,
                         growth=1.1, ny_uniform_reference=64):
    """Concrete mesh plan for ``y+ < 1`` -- the answer round 6 could only bound.

    Round 6's :func:`~tensorfvm.verification.periodic_hill_round6.channel_y_plus_requirement`
    showed the uniform ``ny = 64`` grid puts the first cell at ``y+ ~ 4`` and
    that reaching ``y+ < 1`` *uniformly* needs ``ny ~ 256`` ("16x the work").
    Stretching changes the conclusion by an order of magnitude: with wall
    clustering the same wall distance is reached with a few tens of cells.  This
    function returns both numbers so the choice of the next round is a number,
    not a guess.
    """
    nu = float(bulk) * 1.0 / float(reynolds)
    h = channel_half_height_for_hill_case(channel_top)
    u_tau = float(u_tau)
    delta_wall = float(y_plus_target) * nu / u_tau
    # geometric series: delta_0 * (r^nh - 1)/(r - 1) = h
    growth = float(growth)
    if growth == 1.0:
        nh_uniform = int(math.ceil(h / delta_wall))
        return {
            "nu": nu, "u_tau": u_tau, "half_height": h,
            "first_cell_for_y_plus_target": delta_wall,
            "growth": 1.0, "ny": 2 * nh_uniform, "ny_uniform": 2 * nh_uniform,
            "first_cell_height": h / nh_uniform,
            "first_cell_y_plus": u_tau * (h / nh_uniform) / nu,
            "first_cell_y_plus_at_uniform_reference":
                float(u_tau * (h / (0.5 * float(ny_uniform_reference))) / nu),
            "y_plus_target": float(y_plus_target),
            "saving_factor_vs_uniform": 1.0,
        }
    if growth <= 1.0:
        raise ValueError(f"growth must be > 1 for a clustered mesh, got {growth!r}")
    nh = 4
    while True:
        ratios = growth ** np.arange(nh)
        delta0 = h / ratios.sum()
        if delta0 <= delta_wall:
            break
        nh += 1
        if nh > 10000:
            raise RuntimeError("wall_resolution_plan failed to converge")
    ny = 2 * nh
    delta0_uniform = CHANNEL_TOP_OVER_H / float(ny_uniform_reference) / 2.0
    uniform_ny_needed = int(math.ceil(h / delta_wall))
    return {
        "nu": nu, "u_tau": u_tau, "half_height": h,
        "first_cell_for_y_plus_target": delta_wall,
        "growth": growth, "ny": ny, "ny_uniform": 2 * uniform_ny_needed,
        "first_cell_height": delta0,
        "first_cell_y_plus": u_tau * delta0 / nu,
        "first_cell_y_plus_at_uniform_reference":
            float(u_tau * delta0_uniform / nu),
        "y_plus_target": float(y_plus_target),
        "saving_factor_vs_uniform": 2 * uniform_ny_needed / float(ny),
    }


def _cell_shear(U, y, h):
    """Cell-centred ``|dU/dy|`` with no-slip at both walls.

    First/last cells use the exact wall gradient of the wall-adjacent interval
    (``U/d`` to the wall); interior cells use the central difference.
    """
    shear = np.empty(U.size)
    shear[0] = U[0] / y[0]
    shear[-1] = U[-1] / (h - y[-1])
    shear[1:-1] = (U[2:] - U[:-2]) / (y[2:] - y[:-2])
    return shear


def _tridiag_fv(n, dy, coef_interior, mu_wall_lo, mu_wall_hi):
    """Assemble the FV tridiagonal ``A`` for ``d/dy[mu dphi/dy]`` (Dirichlet 0 at walls)."""
    lower = np.zeros(n)
    main = np.zeros(n)
    upper = np.zeros(n)
    wall_lo = mu_wall_lo / (0.5 * dy[0])
    wall_hi = mu_wall_hi / (0.5 * dy[-1])
    main[0] = (coef_interior[0] + wall_lo) / dy[0]
    main[-1] = (coef_interior[-1] + wall_hi) / dy[-1]
    # the wall rows *do* couple to their neighbour (missing terms here made the
    # first cell decouple from the column, a bug this round's tests pin down)
    upper[0] = -coef_interior[0] / dy[0]
    lower[-1] = -coef_interior[-1] / dy[-1]
    for i in range(1, n - 1):
        main[i] = (coef_interior[i - 1] + coef_interior[i]) / dy[i]
        lower[i] = -coef_interior[i - 1] / dy[i]
        upper[i] = -coef_interior[i] / dy[i]
    return lower, main, upper


def solve_channel_column(ny=192, reynolds=2800.0, bulk=1.0, u_tau=0.06,
                         growth=1.05, config: SpalartAllmarasConfig | None = None,
                         iterations=40000, channel_top=CHANNEL_TOP_OVER_H,
                         tol=1.0e-10, relaxation=0.2):
    """Steady 1D RANS channel driven by ``u_tau``, closed by the solver's SA model.

    Pseudo-time march of the coupled (momentum, ``nu~``) column: momentum and the
    SA equilibrium are each solved by a finite-volume tridiagonal solve per
    Picard iteration, with the SA algebra (``fv1``, ``fv2``, ``S~``, ``r``, ``fw``,
    production, implicit-destruction linearisation) taken from the *same*
    constants object and the *same* ``eddy_viscosity`` used by the 2D solver.
    """
    cfg = config or SpalartAllmarasConfig()
    half = channel_half_height_for_hill_case(channel_top)   # h
    height = 2.0 * half                                     # the duct: y in [0, 2h]
    nu = float(bulk) / float(reynolds)
    y, delta0 = stretched_cell_centres(ny, half, growth)
    nh = ny // 2
    if growth == 1.0:
        spac = np.full(nh, half / nh)
    else:
        ratios = growth ** np.arange(nh)
        spac = ratios * (half / ratios.sum())
    dy = np.concatenate([spac, spac[::-1]])

    G = u_tau ** 2 / half            # d(tau)/dy = G, tau(2h/2) = 0 => tau_wall = u_tau^2
    U = np.zeros(ny)
    nu_tilde = np.full(ny, cfg.freestream_ratio * nu)
    d = np.minimum(y, height - y)
    center_dist = 0.5 * (dy[:-1] + dy[1:])

    def sa_algebra(nu_tilde, shear):
        nu_t, chi, fv1 = eddy_viscosity(nu_tilde, nu, cfg)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)
        sbar = np.maximum(np.abs(shear) + nu_tilde / (cfg.kappa ** 2 * d ** 2) * fv2, 1.0e-12)
        r = np.minimum(nu_tilde / (sbar * cfg.kappa ** 2 * d ** 2), cfg.r_clip)
        g = r + cfg.cw2 * (r ** 6 - r)
        fw = g * ((1.0 + cfg.cw3 ** 6) / (g ** 6 + cfg.cw3 ** 6)) ** (1.0 / 6.0)
        return nu_t, sbar, fw

    shear = _cell_shear(U, y, height)
    for _iter in range(int(iterations)):
        nu_t, sbar, fw = sa_algebra(nu_tilde, shear)
        # ---- momentum: d/dy[(nu + nu_t) dU/dy] = -G
        mu_int = 0.5 * (nu_t[:-1] + nu_t[1:]) + nu
        coef = mu_int / center_dist
        lower, main, upper = _tridiag_fv(ny, dy, coef, nu, nu)
        U = _thomas(lower, main, upper, np.full(ny, G))
        shear = _cell_shear(U, y, height)

        # ---- SA equilibrium: production = destruction + diffusion
        nu_t, sbar, fw = sa_algebra(nu_tilde, shear)
        production = cfg.cb1 * sbar * nu_tilde
        destruction_lin = cfg.cw1 * fw * nu_tilde / d ** 2
        mu_s = nu + np.maximum(nu_t, 0.0)
        coef_s = (0.5 * (mu_s[:-1] + mu_s[1:])) / center_dist
        lower, main, upper = _tridiag_fv(ny, dy, coef_s, nu, nu)
        main = main + destruction_lin
        new = _thomas(lower, main, upper, production)
        new = np.maximum(new, 0.0)
        if not np.all(np.isfinite(new)):
            raise FloatingPointError("1D SA column diverged")
        delta = float(np.max(np.abs(new - nu_tilde)))
        nu_tilde = (1.0 - relaxation) * nu_tilde + relaxation * new
        if delta < tol * max(float(np.max(nu_tilde)), nu):
            break

    nu_t, sbar, fw = sa_algebra(nu_tilde, shear)
    return {
        "iterations_used": _iter + 1,
        "y": y, "U": U, "nu_t": nu_t, "nu_tilde": nu_tilde, "d": d,
        "nu": nu, "u_tau": u_tau, "half_height": half, "height": height,
        "first_cell_height": delta0,
        "growth": growth, "ny": int(ny), "config": cfg, "shear": shear,
    }


def _thomas(lower, main, upper, rhs):
    """Tridiagonal solve via LAPACK banded storage (``n = O(10^2)``, called O(10^3) times)."""
    from scipy.linalg import solve_banded

    n = main.size
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = main
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def _thomas_python(lower, main, upper, rhs):
    """Reference implementation kept for the cross-check test."""
    n = main.size
    c = np.zeros(n)
    dvec = np.zeros(n)
    c[0] = upper[0] / main[0] if n > 1 else 0.0
    dvec[0] = rhs[0] / main[0]
    for i in range(1, n):
        m = main[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        dvec[i] = (rhs[i] - lower[i] * dvec[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dvec[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dvec[i] - c[i] * x[i + 1]
    return x


def log_law_fit(y_plus, u_plus, lo=30.0, hi=0.2):
    """Least-squares ``u+ = (1/kappa) ln y+ + B`` inside ``lo < y+ < hi * max(y+)``."""
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    mask = (y_plus > lo) & (y_plus < hi * y_plus.max())
    if mask.sum() < 3:
        return {"kappa": float("nan"), "B": float("nan"), "points": int(mask.sum())}
    slope, intercept = np.polyfit(np.log(y_plus[mask]), u_plus[mask], 1)
    return {"kappa": float(1.0 / slope), "B": float(intercept),
            "points": int(mask.sum())}


def equilibrium_report(ny=192, reynolds=2800.0, bulk=1.0, u_tau=0.06,
                       growth=1.05, config=None, iterations=40000):
    """Grade the merged SA closure on the channel: level, shape and log law."""
    col = solve_channel_column(ny=ny, reynolds=reynolds, bulk=bulk, u_tau=u_tau,
                               growth=growth, config=config, iterations=iterations)
    nu = col["nu"]
    y, nu_t = col["y"], col["nu_t"]
    ref = reference_eddy_viscosity(y, u_tau, h=col["half_height"])
    # only the log layer carries the reference (below y+ 30 the mixing-length
    # form is not the reference; above 0.3 h the wake matters)
    y_plus = u_tau * y / nu
    mask = (y_plus > 30.0) & (y < 0.3 * col["half_height"])

    nu_t_over_nu = nu_t / nu
    ref_over_nu = ref / nu
    deviation = float(np.max(np.abs(nu_t[mask] / ref[mask] - 1.0))) if mask.any() else float("nan")

    # wall shear from the SA column's own wall gradient (nu_t = 0 at the wall)
    u_tau_num = float(np.sqrt(nu * abs(col["shear"][0])))
    assert u_tau_num > 0.0
    u_plus = col["U"] / u_tau_num
    fit = log_law_fit(u_tau_num * y / nu, u_plus)
    return {
        "input": {"ny": ny, "reynolds": reynolds, "u_tau": u_tau,
                  "growth": growth, "height": col["height"],
                  "half_height": col["half_height"],
                  "first_cell_height": col["first_cell_height"],
                  "first_cell_y_plus": float(u_tau * col["first_cell_height"] / nu)},
        "peak_nu_t_over_nu": float(np.max(nu_t_over_nu)),
        "y_over_half_height_at_peak": float(y[int(np.argmax(nu_t_over_nu))] / col["half_height"]),
        "reference_peak_nu_t_over_nu": float(np.max(ref_over_nu)),
        "over_diffusion_ratio": float(np.max(nu_t_over_nu) / np.max(ref_over_nu)),
        "log_layer_max_relative_deviation": deviation,
        "log_layer_points": int(mask.sum()),
        "u_tau_numerical": u_tau_num,
        "u_tau_relative_error": abs(u_tau_num / u_tau - 1.0),
        "log_law_fit": fit,
        "log_law_kappa_error": (abs(fit["kappa"] / LOG_KAPPA_REFERENCE - 1.0)
                                if math.isfinite(fit["kappa"]) else float("nan")),
        "log_law_B_error": (abs(fit["B"] - LOG_B_REFERENCE)
                            if math.isfinite(fit["B"]) else float("nan")),
        "profile": {"y": y.tolist(), "nu_t_over_nu": nu_t_over_nu.tolist(),
                    "reference_nu_t_over_nu": ref_over_nu.tolist(),
                    "u_plus": u_plus.tolist(), "y_plus": (u_tau_num * y / nu).tolist()},
    }