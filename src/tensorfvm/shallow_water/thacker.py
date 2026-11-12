"""Thacker (1981) exact solution: nonlinear oscillation of a lake in a paraboloid.

Why this module exists
----------------------
The 1-D shallow-water benchmarks in this package are dam breaks: a rarefaction
(or a bore) plus, for the dry case, a moving shoreline.  A 2-D shallow-water
operator needs a reference that exercises what 1-D cannot: a genuinely
two-dimensional, *nonlinear*, *exact*, *moving-shoreline* solution.  Thacker's
paraboloid basin is that reference, and it is the standard verification case
for well-balanced shallow-water schemes.

Setup
-----
Bed ``z_b(r) = -H (1 - r^2/a^2)`` with ``r^2 = x^2 + y^2`` (a paraboloid bowl
whose still-water depth at the centre is ``H`` and whose still-water shoreline
is ``r = a``), inviscid hydrostatic shallow water, free surface
``eta = h + z_b``.  For an amplitude parameter ``A`` in ``(0, 1)``:

.. math::

    \\eta(x,y,t) &= \\left(\\frac{H\\sqrt{1-A^2}}{Q} - H\\right)
                  + \\frac{H r^2}{a^2}\\left(1 - \\frac{1-A^2}{Q^2}\\right),\\\\
    u = \\beta x,\\quad v = \\beta y,\\qquad
    \\beta(t) = \\frac{A\\,\\omega\\,\\sin(\\omega t)}{2Q(t)},\\\\
    Q(t) = 1 - A\\cos(\\omega t),\\qquad \\omega^2 = \\frac{8 g H}{a^2},

so the depth is ``h = eta - z_b``, a paraboloid that keeps its shape while its
curvature breathes; the shoreline (``h = 0``) is the circle

.. math:: s(t) = a\\sqrt{Q(t)/\\sqrt{1-A^2}},

and the total volume is the *time-independent* ``V = pi a^2 H / 2`` (evaluated
below, not tabulated).

Derivation (so that no step is a table lookup)
----------------------------------------------
The ansatz ``h = h_c(t) - c(t) r^2`` with a radial-linear velocity ``u = beta x,
v = beta y`` reduces the 2-D shallow-water equations exactly to

* continuity: ``dh_c/dt = -2 beta h_c`` and ``dc/dt = -4 beta c``;
* radial momentum: ``dbeta/dt + beta^2 + 2 g eta_2 = 0`` with ``eta_2`` the
  ``r^2`` coefficient of the surface,
* the shoreline moves with the fluid, ``ds/dt = beta s``, and ``h_c = c s^2``.

These three are the entire derivation; :func:`check_thacker_solution` then
substitutes the closed forms back into the **Cartesian conservative** PDEs by
finite differences and reports the residual, so a transcription or sign error
cannot survive as a green gate (the same discipline as
``shallow_water.exact.check_stoker_solution``).

References
----------
Thacker, W. C. (1981).  "Some exact solutions to the nonlinear shallow-water
wave equations." *Journal of Fluid Mechanics* **107**, 499-508.
doi:10.1017/S0022112081001882 (the paraboloid basin, eqs. 3.1-3.5).
Secondary (same closed forms used as a well-balanced CFD test):
Marche, Bonneton, Fabrie & Seguin (2007), *Int. J. Numer. Methods Fluids*
**56**, 1877; Delestre et al. (2013) SWASHES validation, subm. to *Computers &
Fluids*.
"""

from __future__ import annotations

import math

__all__ = ["ThackerBasin", "check_thacker_solution"]
import numpy as np


class ThackerBasin:
    """Exact solution of the parabolic-basin oscillation.

    Parameters
    ----------
    H : float
        Still-water depth at the basin centre (metres).
    a : float
        Still-water shoreline radius (metres).
    amplitude : float
        ``A`` in ``(0, 1)``: 0 is the linear limit (lake at rest), larger
        ``A`` means larger shoreline excursion.
    gravity : float
        ``g`` in m/s^2.
    """

    def __init__(self, H: float = 1.0, a: float = 1.0, amplitude: float = 0.5,
                 gravity: float = 9.81):
        H = float(H)
        a = float(a)
        amplitude = float(amplitude)
        self.gravity = float(gravity)
        if H <= 0.0 or a <= 0.0:
            raise ValueError("H and a must be positive")
        if not 0.0 <= amplitude < 1.0:
            raise ValueError("amplitude must satisfy 0 <= A < 1")
        self.H = H
        self.a = a
        self.amplitude = amplitude
        self.omega = math.sqrt(8.0 * self.gravity * H) / a
        self.period = 2.0 * math.pi / self.omega

    # ------------------------------------------------------------------ bed
    def bed(self, x, y):
        """Bed elevation ``z_b = -H (1 - r^2/a^2)`` (positive on the slope)."""
        r2 = np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2
        return -self.H * (1.0 - r2 / (self.a * self.a))

    def still_water_depth(self, x, y):
        """Lake-at-rest depth ``-z_b`` (the ``A -> 0`` limit of ``depth``)."""
        return np.maximum(-self.bed(x, y), 0.0)

    # ------------------------------------------------------------- solution
    def envelope(self, t: float) -> dict:
        """Time-dependent scalars ``Q``, ``beta``, ``h_c``, ``c``, ``s``."""
        A = self.amplitude
        tau = self.omega * float(t)
        q = 1.0 - A * math.cos(tau)
        root = math.sqrt(max(1.0 - A * A, 0.0))
        beta = A * self.omega * math.sin(tau) / (2.0 * q)
        h_c = self.H * root / q
        c = (self.H / (self.a * self.a)) * (1.0 - A * A) / (q * q)
        return {"Q": q, "beta": beta, "h_c": h_c, "c": c,
                "s": math.sqrt(h_c / c)}

    def depth(self, x, y, t: float):
        """Exact depth ``h(x, y, t)`` (clipped at zero outside the shoreline)."""
        e = self.envelope(t)
        r2 = np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2
        return np.maximum(e["h_c"] - e["c"] * r2, 0.0)

    def surface(self, x, y, t: float):
        """Exact free surface ``eta = h + z_b``."""
        return self.depth(x, y, t) + self.bed(x, y)

    def velocity(self, x, y, t: float):
        """Exact velocity components ``(u, v) = beta (x, y)``, zero when dry."""
        e = self.envelope(t)
        h = self.depth(x, y, t)
        wet = h > 0.0
        beta = e["beta"]
        u = np.where(wet, beta * np.asarray(x, dtype=float), 0.0)
        v = np.where(wet, beta * np.asarray(y, dtype=float), 0.0)
        return u, v

    def centre_depth(self, t: float) -> float:
        """Exact depth at the centre, ``h_c(t)``."""
        return self.envelope(t)["h_c"]

    def shoreline_radius(self, t: float) -> float:
        """Exact shoreline radius ``s(t)``."""
        return self.envelope(t)["s"]

    def max_shoreline_radius(self) -> float:
        """Largest shoreline radius over one period (for domain sizing)."""
        return self.a * ((1.0 + self.amplitude) / (1.0 - self.amplitude)) ** 0.25

    @property
    def volume(self) -> float:
        """Exact conserved volume ``pi a^2 H / 2`` (independent of ``t``)."""
        return math.pi * self.a * self.a * self.H / 2.0

    def initial_state(self, x, y):
        """``(h, u, v)`` at ``t = 0`` on the given grid."""
        return (self.depth(x, y, 0.0),) + self.velocity(x, y, 0.0)

    def summary(self) -> dict:
        return {
            "H": self.H,
            "a": self.a,
            "amplitude": self.amplitude,
            "g": self.gravity,
            "gravity": self.gravity,
            "omega": self.omega,
            "period": self.period,
            "volume": self.volume,
            "max_shoreline_radius": self.max_shoreline_radius(),
            "centre_depth_min": min(self.centre_depth(t) for t in
                                    (0.0, 0.5 * self.period)),
            "centre_depth_max": max(self.centre_depth(t) for t in
                                    (0.0, 0.5 * self.period)),
        }


def check_thacker_solution(basin: ThackerBasin, directions=(0.0, math.pi / 8.0,
                                                           math.pi / 4.0),
                           radial_fractions=(0.2, 0.5, 0.8),
                           time_fractions=(0.13, 0.37, 0.61, 0.88),
                           delta_fraction: float = 3e-5) -> dict:
    """Certify the closed forms by substituting them into the SWE residuals.

    The 2-D shallow-water equations in Cartesian conservative form

    ``h_t + (h u)_x + (h v)_y = 0``,
    ``(h u)_t + (h u^2 + g h^2/2)_x + (h u v)_y = -g h z_x``,
    ``(h v)_t + (h u v)_x + (h v^2 + g h^2/2)_y = -g h z_y``

    are evaluated with central differences on the analytic fields at a set of
    points inside the wet region and at a set of times inside one period.

    Returns ``{equation: max relative residual, ...}`` where the relative
    residual is ``|residual| / max|individual term|`` -- an O(1) scale, so the
    number does not depend on the units.  A closed form that is wrong anywhere
    (sign, factor, frequency) leaves a residual of order one here.
    """
    g = basin.gravity
    d = delta_fraction * basin.a
    out = {}
    worst = {"mass": 0.0, "x_momentum": 0.0, "y_momentum": 0.0}
    for tf in time_fractions:
        t = tf * basin.period
        e = basin.envelope(t)
        if abs(e["beta"]) < 1e-12:      # turning point: u = 0 exactly
            continue
        for theta in directions:
            for rf in radial_fractions:
                r = rf * e["s"]
                x = r * math.cos(theta)
                y = r * math.sin(theta)

                def fields(xx, yy, tt):
                    h = float(basin.depth(xx, yy, tt))
                    u, v = basin.velocity(xx, yy, tt)
                    return h, float(u), float(v)

                def ddt(xx, yy, tt):
                    hp, up, vp = fields(xx, yy, tt + d)
                    hm, um, vm = fields(xx, yy, tt - d)
                    return (hp - hm), (hp * up - hm * um), (hp * vp - hm * vm)

                def ddx(xx, yy, tt):
                    hp, up, vp = fields(xx + d, yy, tt)
                    hm, um, vm = fields(xx - d, yy, tt)
                    return (hp * up - hm * um), \
                           (hp * up * up + 0.5 * g * hp * hp
                            - hm * um * um - 0.5 * g * hm * hm), \
                           (hp * up * vp - hm * um * vm)

                def ddy(xx, yy, tt):
                    hp, up, vp = fields(xx, yy + d, tt)
                    hm, um, vm = fields(xx, yy - d, tt)
                    return (hp * vp - hm * vm), \
                           (hp * up * vp - hm * um * vm), \
                           (hp * vp * vp + 0.5 * g * hp * hp
                            - hm * vm * vm - 0.5 * g * hm * hm)

                h0, u0, v0 = fields(x, y, t)
                two_d = 2.0 * d
                dh, dhu, dhv = ddt(x, y, t)
                dhu_x, dhuu_x, dhuv_x = ddx(x, y, t)
                dhv_y, dhuv_y, dvv_y = ddy(x, y, t)
                # momentum flux divergence in x uses the h*u column; in y the
                # h*v column, hence the swaps below.
                fx_mass, fx_mom, fx_trans = dhu_x, dhuu_x, dhuv_x
                fy_mass, fy_trans, fy_mom = dhv_y, dhuv_y, dvv_y
                zx = float(2.0 * basin.H * x / (basin.a * basin.a))
                zy = float(2.0 * basin.H * y / (basin.a * basin.a))

                res_mass = (dh + fx_mass + fy_mass) / two_d
                res_momx = ((dhu + fx_mom + fy_trans) / two_d
                            + g * h0 * zx)
                res_momy = ((dhv + fx_trans + fy_mom) / two_d
                            + g * h0 * zy)
                # term scales (per unit length after the /2d above)
                scale_mass = max(abs(dh / two_d), abs(fx_mass / two_d),
                                 abs(fy_mass / two_d), 1e-300)
                scale_momx = max(abs(dhu / two_d), abs(fx_mom / two_d),
                                 abs(fy_trans / two_d),
                                 abs(g * h0 * zx), 1e-300)
                scale_momy = max(abs(dhv / two_d), abs(fx_trans / two_d),
                                 abs(fy_mom / two_d),
                                 abs(g * h0 * zy), 1e-300)
                worst["mass"] = max(worst["mass"], abs(res_mass) / scale_mass)
                worst["x_momentum"] = max(worst["x_momentum"],
                                          abs(res_momx) / scale_momx)
                worst["y_momentum"] = max(worst["y_momentum"],
                                          abs(res_momy) / scale_momy)
    out.update(worst)
    out["max_relative_residual"] = max(worst.values())
    out["delta"] = d
    out["n_points"] = len(directions) * len(radial_fractions) * len(
        time_fractions)
    return out