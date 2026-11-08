"""KVLCC2 geometry, its axisymmetric equivalent body, and Michell wave resistance.

Geometry discipline first
-------------------------
The repository's legacy "KVLCC2 model scale" convention
(``tensorfvm.kvlcc2_geometry``: ``L = 7 m, B = 0.936 m, T = 0.342 m, Cb = 0.708``)
is audited in ``docs/resistance-chain-form-factor`` and is *not* geometrically
similar to KVLCC2 (``L/B = 7.48`` against the real ``5.52``).  Comparing a total
resistance coefficient with a SIMMAN measurement through that convention would be
meaningless, so this module uses the real KVLCC2 particulars at model scale

    L_pp = 7.0 m, B = 1.2688 m, T = 0.4550 m, Cb = 0.8098, Cm = 0.9982,
    Cwp = 0.870, LCB = -1.688 % L_pp,  lambda = 320/7 = 45.714,

and verifies the coefficients of the parametric hull it builds (``Cb``, ``Cm``,
``Cwp``, ``LCB`` and the wetted surface) *before* any resistance number is
produced.

Construction
------------
The hull is a parametric family, because no trustworthy offset table exists in
this workspace:

* sectional area :math:`a(\\xi)` and waterline half breadth :math:`b(\\xi)` are
  power-law forms with independent fore/aft exponents, fitted by least squares to
  the volume, the waterplane area, the midship section coefficient and the LCB;
* the section shape is the power law
  :math:`y(\\xi,z) = b(\\xi)[1-(1-z/T)^{n(\\xi)}]^{1/n(\\xi)}` with :math:`n(\\xi)`
  recovered from the local fullness :math:`a/(2bT)`, which reproduces the
  KVLCC2 pattern of a waterplane fuller than the mean section (``Cwp > Cp``).

Two derived bodies follow:

* the **axisymmetric equivalent body** of revolution with the same sectional area,
  :math:`r(\\xi) = \\sqrt{a(\\xi)/\\pi}`, whose surface velocity comes from the
  axisymmetric source distribution :math:`q = U\\,\\mathrm{d}A/\\mathrm{d}x`
  evaluated on a regularised kernel (slender-body approximation);
* the **Michell thin-ship wave resistance** built from the same offsets, used for
  the wave-making term that no viscid RANS can supply.

Both are approximations and are labelled as such:

* the equivalent body reproduces the *viscous* resistance of a hull to roughly
  the accuracy of the Hughes form-factor method, but it has no cross flow, so its
  form factor is a lower bound for a full tanker stern;
* Michell's integral is a thin-ship result; for a tanker at ``B/L = 0.18`` it
  typically over-predicts the wave resistance by a few tens of percent.  The
  campaign therefore reports the wave term with an explicit band and shows how
  the total-resistance verdict changes when the band is applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from ..resistance_chain import HullPrincipal

__all__ = [
    "KVLCC2_MODEL_PARTICULARS", "KVLCC2_MODEL", "Kvlcc2Hull", "build_kvlcc2_hull",
    "equivalent_body_radius", "equivalent_body_velocity", "michell_wave_resistance",
    "KVLCC2_S_FULL_SCALE", "froude_number",
]

#: KVLCC2 model particulars (full scale 320 m, model scale ratio 320/7).
#: ``Cb``/``Cm``/``Cwp``/``LCB`` are the published KRISO/MOERI values.
KVLCC2_MODEL_PARTICULARS = dict(
    name="KVLCC2_model_L7", L=7.0, B=1.2688, T=0.4550,
    Cb=0.8098, Cm=0.9982, Cwp=0.870, lcb_pct=-1.688,
)

#: Published bare-hull wetted surface at full scale [m^2] (KVLCC2, KRISO).
KVLCC2_S_FULL_SCALE = 27578.0

#: :class:`HullPrincipal` record for the regression branches of the repository.
KVLCC2_MODEL = HullPrincipal(**KVLCC2_MODEL_PARTICULARS)


def _section_fullness(n: np.ndarray) -> np.ndarray:
    """``C_m(n) = int_0^1 [1 - (1-s)^n]^{1/n} ds`` for the power-law section."""
    s = np.linspace(0.0, 1.0, 2001)
    n = np.atleast_1d(np.asarray(n, dtype=float))
    out = np.empty_like(n)
    for i, ni in enumerate(n):
        out[i] = np.trapezoid((1.0 - (1.0 - s) ** ni) ** (1.0 / ni), s)
    return out if out.size > 1 else out[0]


_N_TABLE = np.linspace(1.2, 40.0, 400)
_CM_TABLE = _section_fullness(_N_TABLE)


def _n_for_fullness(fullness):
    """Invert :func:`_section_fullness` (falls back to the table ends)."""
    f = np.asarray(fullness, dtype=float)
    return np.interp(np.clip(f, _CM_TABLE.min(), _CM_TABLE.max()), _CM_TABLE, _N_TABLE)


@dataclass
class Kvlcc2Hull:
    """Parametric KVLCC2 hull with verified integral coefficients."""

    L: float = 7.0
    B: float = 1.2688
    T: float = 0.4550
    Cb: float = 0.8098
    Cm: float = 0.9982
    Cwp: float = 0.870
    lcb_pct: float = -1.688
    p_fore: float = 2.0
    p_aft: float = 2.0
    q_fore: float = 0.35
    q_aft: float = 0.35
    r_fore: float = 3.0
    r_aft: float = 3.0
    s_fore: float = 0.35
    s_aft: float = 0.35
    coefficients_check: Dict[str, float] = field(default_factory=dict)

    # -- primary shapes ---------------------------------------------------
    def _xi(self, x=None, n: int = 401) -> np.ndarray:
        if x is None:
            return np.linspace(0.0, 1.0, n)
        return np.asarray(x, dtype=float) / self.L

    def area_shape(self, xi) -> np.ndarray:
        """Sectional area ``a(xi)`` [m^2]."""
        xi = np.asarray(xi, dtype=float)
        s = 2.0 * xi - 1.0
        fore = s >= 0.0
        base = np.where(fore,
                        np.clip(1.0 - np.abs(s) ** self.p_fore, 0.0, None) ** self.q_fore,
                        np.clip(1.0 - np.abs(s) ** self.p_aft, 0.0, None) ** self.q_aft)
        return (self.Cm * self.B * self.T) * np.maximum(base, 0.0)

    def half_breadth_shape(self, xi) -> np.ndarray:
        """Waterline half breadth ``b(xi)`` [m]."""
        xi = np.asarray(xi, dtype=float)
        s = 2.0 * xi - 1.0
        fore = s >= 0.0
        base = np.where(fore,
                        np.clip(1.0 - np.abs(s) ** self.r_fore, 0.0, None) ** self.s_fore,
                        np.clip(1.0 - np.abs(s) ** self.r_aft, 0.0, None) ** self.s_aft)
        return (self.B / 2.0) * np.maximum(base, 0.0)

    def half_breadth(self, x, z=None) -> np.ndarray:
        """Section offsets ``y(x, z)``; ``z`` measured down from the waterline."""
        xi = self._xi(x)
        b = self.half_breadth_shape(xi)
        a = self.area_shape(xi)
        fullness = np.where(b > 1e-9, a / np.maximum(2.0 * b * self.T, 1e-12), 0.0)
        n = _n_for_fullness(np.clip(fullness, _CM_TABLE.min() * 1.001, self.Cm))
        if z is None:
            return b, n, fullness
        z = np.asarray(z, dtype=float)
        # z is the depth below the waterline, positive downwards (0 at the
        # waterline, T at the keel)
        zz = np.clip(z / self.T, 0.0, 1.0)
        shape = (1.0 - zz ** n[..., None]) ** (1.0 / n[..., None])
        return np.expand_dims(b, -1) * shape

    # -- integral quantities ---------------------------------------------
    def coefficients(self, n: int = 4001) -> Dict[str, float]:
        xi = np.linspace(0.0, 1.0, n)
        x = xi * self.L
        a = self.area_shape(xi)
        b = self.half_breadth_shape(xi)
        volume = float(np.trapezoid(a, x))
        waterplane = float(np.trapezoid(2.0 * b, x))
        midship = float(np.interp(0.5, xi, a))
        moment = float(np.trapezoid(a * (x - 0.5 * self.L), x)) / max(volume, 1e-30)
        return {
            "volume": volume,
            "Cb": volume / (self.L * self.B * self.T),
            "Cm": midship / (self.B * self.T),
            "Cwp": waterplane / (self.L * self.B),
            "LCB_pct": 100.0 * moment / self.L,
        }

    def section_perimeter(self, n_sections: int = 201, n_z: int = 401) -> Tuple[np.ndarray, np.ndarray]:
        """Wetted perimeter of each section and the resulting wetted surface."""
        xi = np.linspace(0.0, 1.0, n_sections)
        x = xi * self.L
        b = self.half_breadth_shape(xi)
        a = self.area_shape(xi)
        fullness = np.where(b > 1e-9, a / np.maximum(2.0 * b * self.T, 1e-12), 0.0)
        nn = _n_for_fullness(np.clip(fullness, _CM_TABLE.min() * 1.001, self.Cm))
        z = np.linspace(0.0, self.T, n_z)
        zz = z / self.T
        perimeter = np.zeros_like(xi)
        for i in range(len(xi)):
            if b[i] <= 1e-12:
                continue
            shape = (1.0 - zz ** nn[i]) ** (1.0 / nn[i])
            y = b[i] * shape
            dydz = np.gradient(y, z)
            perimeter[i] = 2.0 * np.trapezoid(np.sqrt(1.0 + dydz ** 2), z)
        surface = float(np.trapezoid(perimeter, x))
        return perimeter, surface

    def wetted_surface(self) -> float:
        return self.section_perimeter()[1]

    def verify(self) -> Dict[str, float]:
        """Fit the fore/aft exponents to the published coefficients."""
        from scipy.optimize import least_squares

        target = np.array([self.Cb, self.Cwp, self.lcb_pct])

        def residual(p):
            self.p_fore, self.p_aft, self.q_fore, self.q_aft, \
                self.r_fore, self.r_aft, self.s_fore, self.s_aft = p
            c = self.coefficients(n=1201)
            return np.array([(c["Cb"] - target[0]) / target[0] * 10.0,
                             (c["Cwp"] - target[1]) / target[1] * 10.0,
                             (c["LCB_pct"] - target[2]) / 3.0])

        x0 = np.array([2.0, 2.0, 0.35, 0.35, 3.0, 3.0, 0.35, 0.12])
        lower = np.array([1.2, 1.2, 0.05, 0.05, 1.2, 1.2, 0.05, 0.02])
        upper = np.array([8.0, 8.0, 4.0, 4.0, 12.0, 12.0, 4.0, 4.0])
        sol = least_squares(residual, x0, bounds=(lower, upper), xtol=1e-14,
                            ftol=1e-14, gtol=1e-14, max_nfev=400)
        self.p_fore, self.p_aft, self.q_fore, self.q_aft, \
            self.r_fore, self.r_aft, self.s_fore, self.s_aft = sol.x
        c = self.coefficients()
        self.coefficients_check = c
        return c


def build_kvlcc2_hull(verify: bool = True, **overrides) -> Kvlcc2Hull:
    """Build the parametric hull and (by default) fit it to the published data."""
    params = dict(KVLCC2_MODEL_PARTICULARS)
    params.pop("name", None)
    params.update(overrides)
    hull = Kvlcc2Hull(**params)
    if verify:
        hull.verify()
    return hull


def equivalent_body_radius(hull: Kvlcc2Hull, x) -> np.ndarray:
    """Radius of the equal-sectional-area body of revolution."""
    area = hull.area_shape(hull._xi(x))
    return np.sqrt(np.maximum(area, 0.0) / np.pi)


def equivalent_body_velocity(hull: Kvlcc2Hull, u_inf: float, n: int = 241
                             ) -> Dict[str, np.ndarray]:
    """Surface speed of the equivalent body from the axisymmetric source flow.

    Axisymmetric slender-body theory places a source distribution of strength
    ``q(x) = U dA/dx`` on the axis; for the axial perturbation on the surface,
    the kernel is regularised with the local radius,
    ``u'(x) = int q(xi) (x - xi) / (4 pi R^3) dxi`` with
    ``R^2 = (x - xi)^2 + r(x)^2``.  The body slope condition fixes ``q`` exactly,
    so only the axial velocity needs the integral.
    """
    x = np.linspace(1e-6 * hull.L, hull.L * (1.0 - 1e-6), n)
    r = equivalent_body_radius(hull, x)
    area = np.pi * r ** 2
    q = u_inf * np.gradient(area, x)
    dx = np.gradient(x)
    u_pert = np.zeros(n)
    for i in range(n):
        R = np.sqrt((x[i] - x) ** 2 + r[i] ** 2)
        u_pert[i] = np.sum(q * (x[i] - x) / (4.0 * np.pi * np.maximum(R, 1e-12) ** 3) * dx)
    ue = u_inf + u_pert
    ue = np.clip(ue, 0.05 * u_inf, 2.5 * u_inf)
    return {"x": x, "r": r, "ue": ue, "area": area, "q": q}


def michell_wave_resistance(hull: "Kvlcc2Hull", u_inf: float, *, n_x: int = 161,
                            n_z: int = 201, n_theta: int = 200, rho: float = 998.2,
                            g: float = 9.81) -> Dict[str, object]:
    """Michell thin-ship wave resistance of the hull offsets.

    ``R_W = (4 rho g^2 / (pi U^2)) int_0^{pi/2} [P^2 + Q^2] sec^3(theta) dtheta``
    with the centreplane amplitude

        ``P + iQ = int int (dy/dx) exp(-k z) exp(i k x cos theta) dz dx``,
        ``k = k0 sec^2 theta``, ``k0 = g/U^2``,

    ``z`` the depth below the waterline (positive downwards) and ``y`` the
    half breadth.  Because ``sec^2`` makes ``k`` grow without bound as
    ``theta -> pi/2``, the depth integral is evaluated on the *substituted* grid
    ``z = t/k`` so that ``exp(-k z) = exp(-t)`` is always resolved: a fixed
    depth grid leaves the waterline strip under-resolved and makes the
    ``theta`` integral diverge (measured: ``C_W`` grew by a factor 20 between
    ``n_z = 101`` and ``n_z = 401`` before this change).
    """
    import numpy as np

    x = np.linspace(-0.5 * hull.L, 0.5 * hull.L, n_x)
    xi = (x + 0.5 * hull.L) / hull.L
    x_body = xi * hull.L
    k0 = g / u_inf ** 2
    s = np.linspace(0.0, 1.0, n_theta)
    theta = 0.5 * np.pi * (1.0 - np.cos(0.5 * np.pi * s))   # clustered towards 0
    sec = 1.0 / np.cos(theta)
    k = k0 * sec ** 2
    dx = np.gradient(x)
    t_nodes = np.linspace(0.0, 24.0, 97)
    dt_k = np.gradient(t_nodes)
    amplitude = np.empty(n_theta, dtype=complex)
    for j in range(n_theta):
        phase = np.exp(1j * k[j] * np.cos(theta[j]) * x)
        zq = t_nodes / k[j]
        yq = hull.half_breadth(x_body, zq)   # exact offsets, no interpolation
        dyq = np.gradient(yq, x, axis=0)
        weight = np.exp(-t_nodes) * dt_k / k[j]
        amplitude[j] = np.sum((dyq * weight[None, :]).T @ (phase * dx))
    integrand = np.abs(amplitude) ** 2 * sec ** 3
    integral = np.trapezoid(integrand, theta)
    rw = 4.0 * rho * g ** 2 / (np.pi * u_inf ** 2) * integral
    return {"R_W": float(rw), "k0": float(k0), "theta": theta,
            "integrand": integrand, "amplitude": amplitude}

def froude_number(u: float, length: float, g: float = 9.81) -> float:
    return u / np.sqrt(g * length)