"""Surface radiation: two-dimensional view factors, gray-diffuse enclosures and
the long-wave sky exchange of the icing heat balance.

Scope and why the module is 2-D first
-------------------------------------
TensorFVM owns conduction, conjugate heat transfer and the ice-accretion surface
balance (:mod:`tensorfvm.cht.ice_conjugate`), whose long-wave term is written as
``q_lw = eps (sigma T_s^4 - L_down)`` with ``L_down`` from Idso-Jackson.  That
term had no verification at all (``docs/verification-catalog.md`` gap 5,
"Radiation: no heat transfer verification"), and the repository could not
compute exchange *between* surfaces at all -- only the sky term.

The engineering use cases in this repository are cross-sections: a deck beam, a
cargo-hold corner, an ice-channel wall, a radiatively cooled structural profile.
For those the natural model is a **2-D enclosure**: surfaces are straight strips
of unit depth in the ``(x, y)`` plane and the view factor is the classic
crossed-strings construction (Hottel, *Radiative Heat Transfer*, ch. 3), which is
*exact* for infinite strips -- no quadrature error and no singularity treatment.
The "area" in the enclosure equations is therefore the strip length (per unit
depth) and the view factor is the subtended-angle fraction.

Validity domain (measured, not assumed)
---------------------------------------
Crossed strings are exact for a pair of mutually visible strips and for a strip
enclosure whose boundary is traversed once around the radiating domain (a convex
cavity).  They are **invalid** when the wire frame wraps around a strip, i.e. for
an annular (doubly connected) domain: there each of the ``n`` outer strips
receives a signed contribution whose sum cancels to zero for an inner strip
(measured: inner-strip row sum of the annulus = 0.0 while the correct value is
1.0), so no orientation convention repairs it.  Use
:func:`circle_pair_form_factors` for concentric circles and
:func:`strip_view_factor_quadrature` for a single pair, and do not assemble an
annulus out of strips.

Orientation convention
----------------------
Each strip is traversed so that the radiating domain lies on its **left**; with
that single convention the signed crossed-strings difference telescopes, and the
rows of a closed convex enclosure sum to one to round-off.  Taking absolute
values instead double-counts concave boundaries (measured: a wrong 2.0).

Everything above is geometry and algebra: the anchors used by
``python -m tensorfvm.verification.radiation`` are exact to round-off.  Nothing
here is compared with a radiation *measurement*, because this repository holds no
radiometric data; the correlation spread between published clear-sky long-wave
models is reported as a band, never as a verified number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SIGMA",
    "Strip",
    "strip_form_factors",
    "strip_view_factor",
    "strip_view_factor_quadrature",
    "circle_pair_form_factors",
    "two_surface_exchange",
    "GrayEnclosure",
    "sky_equivalent_temperature",
    "sky_depression",
]

#: Stefan-Boltzmann constant [W m^-2 K^-4] (CODATA 2018).
SIGMA = 5.670374419e-8


@dataclass(frozen=True)
class Strip:
    """A straight strip of unit depth in the ``(x, y)`` plane.

    ``p0``/``p1`` are the end points.  Their order selects the side: the strip is
    traversed so that the radiating domain lies on its left (see the module
    docstring), which is what fixes the sign of the crossed-strings difference.
    """

    p0: Tuple[float, float]
    p1: Tuple[float, float]

    def __post_init__(self) -> None:
        if np.allclose(np.asarray(self.p0, float), np.asarray(self.p1, float)):
            raise ValueError("degenerate strip: p0 == p1")

    @property
    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.p0, float), np.asarray(self.p1, float)

    @property
    def length(self) -> float:
        a, b = self.points
        return float(np.hypot(*(b - a)))

    @property
    def midpoint(self) -> np.ndarray:
        a, b = self.points
        return 0.5 * (a + b)

    @property
    def tangent(self) -> np.ndarray:
        a, b = self.points
        return (b - a) / self.length

    @property
    def inward_normal(self) -> np.ndarray:
        """Unit normal pointing to the left of ``p0 -> p1`` (into the domain)."""
        t = self.tangent
        return np.array([-t[1], t[0]])


def _crossed_uncrossed(si: Strip, sj: Strip) -> Tuple[float, float]:
    a0, a1 = si.points
    b0, b1 = sj.points
    crossed = float(np.hypot(*(a0 - b1)) + np.hypot(*(a1 - b0)))
    uncrossed = float(np.hypot(*(a0 - b0)) + np.hypot(*(a1 - b1)))
    return crossed, uncrossed


def _exchange_area(si: Strip, sj: Strip) -> float:
    """``A_i F_ij`` (per unit depth) for strips whose domain lies on their left.

    The sign is global -- it is set by the orientation convention, not by the
    pair -- so the result is symmetric in ``(i, j)``, which is exactly the
    reciprocity theorem ``A_i F_ij = A_j F_ji``.
    """
    crossed, uncrossed = _crossed_uncrossed(si, sj)
    return -0.5 * (crossed - uncrossed)


def strip_view_factor(si: Strip, sj: Strip) -> float:
    """``F_{i -> j}`` for two mutually visible strips of unit depth."""
    return _exchange_area(si, sj) / si.length


def strip_form_factors(strips: Sequence[Strip],
                       visible: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """View-factor matrix of a strip enclosure, plus the symmetric exchange areas.

    Returns ``{"F": (n, n), "exchange": (n, n), "lengths": (n,)}`` with
    ``exchange[i, j] = A_i F_ij = A_j F_ji`` and ``F[i, j] = exchange[i, j] / lengths[i]``;
    the diagonal is zero because a flat strip does not see itself.

    ``visible`` (boolean mask, default all-True) declares which pairs see each
    other.  A convex enclosure needs no mask.  The mask cannot repair a wrapping
    (annular) wire frame -- that construction is invalid, see the module
    docstring -- it only removes pairs that are known to be blocked.
    """
    n = len(strips)
    if n < 2:
        raise ValueError("need at least two strips")
    lengths = np.array([s.length for s in strips], dtype=float)
    exchange = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = _exchange_area(strips[i], strips[j])
            exchange[i, j] = exchange[j, i] = value
    if visible is not None:
        visible = np.asarray(visible, dtype=bool)
        if visible.shape != (n, n):
            raise ValueError(f"visible must be ({n}, {n})")
        if not np.array_equal(visible, visible.T):
            raise ValueError("visible mask must be symmetric (mutual visibility)")
        np.fill_diagonal(visible, False)
        exchange = np.where(visible, exchange, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = exchange / lengths[:, None]
    return {"F": f, "exchange": exchange, "lengths": lengths}


def strip_view_factor_quadrature(si: Strip, sj: Strip, order: int = 96) -> float:
    """Independent reference ``F_{i -> j}`` by Gauss-Legendre quadrature of the kernel.

    In two dimensions

    .. math:: F_{ij} = \\frac{1}{2 L_i} \\int_{L_i}\\!\\int_{L_j}
              \\frac{\\cos\\theta_i \\cos\\theta_j}{r}\\, ds_j\\, ds_i ,

    which is used by the verification driver to cross-check the crossed-strings
    construction with a *different* method.  Quadrature is the weaker method: the
    integrand is singular when the strips touch (a shared corner or edge), where
    it converges at ``O(1/order)`` -- the reported agreement is then limited by
    this function, not by the crossed strings.
    """
    a0, a1 = si.points
    b0, b1 = sj.points
    ta, tb = si.tangent, sj.tangent
    na, nb = si.inward_normal, sj.inward_normal
    if order < 4:
        raise ValueError("order >= 4")
    xs, ws = np.polynomial.legendre.leggauss(order)
    la, lb = si.length, sj.length
    sa, wsa = 0.5 * la * (xs + 1.0), 0.5 * la * ws
    sb, wsb = 0.5 * lb * (xs + 1.0), 0.5 * lb * ws
    pa = a0 + np.outer(sa, ta)                       # (order, 2)
    pb = b0 + np.outer(sb, tb)                       # (order, 2)
    d = pb[None, :, :] - pa[:, None, :]              # (a, b, 2)
    r = np.linalg.norm(d, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_a = np.abs(d @ na) / r
        cos_b = np.abs((-d) @ nb) / r
        kernel = cos_a * cos_b / r
    integral = float(np.einsum("i,j,ij->", wsa, wsb, kernel))
    return integral / (2.0 * la)


def circle_pair_form_factors(r_inner: float, r_outer: float) -> Dict[str, np.ndarray]:
    """Exact 2-D enclosure: a circle of radius ``r_inner`` inside one of ``r_outer``.

    In two dimensions the view factor is the subtended-angle fraction, so the
    inner (convex) circle sees the whole outer circle, ``F_12 = 1``.  Reciprocity
    on per-unit-depth areas ``2 pi r`` gives ``F_21 = r1 / r2``, and because the
    inner side of the outer circle is concave it also sees itself,
    ``F_22 = 1 - r1 / r2``.
    """
    if not (r_inner > 0.0 and r_outer > r_inner):
        raise ValueError("need 0 < r_inner < r_outer")
    areas = np.array([2.0 * math.pi * r_inner, 2.0 * math.pi * r_outer])
    f21 = r_inner / r_outer
    f = np.array([[0.0, 1.0], [f21, 1.0 - f21]])
    return {"F": f, "exchange": f * areas[:, None], "lengths": areas}


def two_surface_exchange(area_1: float, area_2: float, eps_1: float, eps_2: float,
                         t_1: float, t_2: float, f_12: float = 1.0) -> float:
    """Closed-form net power leaving surface 1 towards surface 2 (two-surface enclosure).

    .. math:: Q_{12} = \\frac{\\sigma (T_1^4 - T_2^4)}
              {\\frac{1-\\epsilon_1}{\\epsilon_1 A_1} + \\frac{1}{A_1 F_{12}}
              + \\frac{1-\\epsilon_2}{\\epsilon_2 A_2}}

    (Siegel & Howell, *Thermal Radiation Heat Transfer*, two-surface enclosure).
    Valid when surface 1 is completely enclosed by surface 2 (``F_12 = 1``);
    areas are per unit depth in the 2-D case.
    """
    for e in (eps_1, eps_2):
        if not (0.0 < e <= 1.0):
            raise ValueError("emissivity must lie in (0, 1]")
    denom = ((1.0 - eps_1) / (eps_1 * area_1) + 1.0 / (area_1 * f_12)
             + (1.0 - eps_2) / (eps_2 * area_2))
    return SIGMA * (t_1 ** 4 - t_2 ** 4) / denom


class GrayEnclosure:
    """Gray-diffuse enclosure solved by the radiosity (net-radiation) method.

    Parameters
    ----------
    f : (n, n) view factors, rows summing to one.
    areas : (n,) per-unit-depth surface areas (strip lengths).
    emissivity : (n,) in ``(0, 1]``.
    temperatures : (n,) in kelvin; ignored (``nan`` allowed) for reradiating surfaces.
    reradiating : boolean mask of adiabatic surfaces (net flux identically zero).
    sigma : Stefan-Boltzmann constant (overridable for verification).

    The radiosity of surface ``i`` is ``J_i = eps_i sigma T_i^4 + (1 - eps_i) G_i``
    with irradiation ``G_i = sum_j F_ij J_j``; the net flux leaving the surface is
    ``q_i = J_i - G_i``, valid for black and gray surfaces alike.  A reradiating
    surface replaces its energy equation by ``q_i = 0``.  ``power`` is ``q_i A_i``.
    """

    def __init__(self, f: np.ndarray, areas: Sequence[float], emissivity: Sequence[float],
                 temperatures: Sequence[float], reradiating: Optional[Sequence[bool]] = None,
                 sigma: float = SIGMA):
        f = np.asarray(f, dtype=float)
        n = f.shape[0]
        if f.shape != (n, n):
            raise ValueError("f must be square")
        areas = np.asarray(areas, dtype=float)
        emissivity = np.asarray(emissivity, dtype=float)
        temperatures = np.asarray(temperatures, dtype=float)
        for arr, name in ((areas, "areas"), (emissivity, "emissivity"),
                          (temperatures, "temperatures")):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have length {n}")
        if np.any(areas <= 0.0):
            raise ValueError("areas must be positive")
        if np.any((emissivity <= 0.0) | (emissivity > 1.0)):
            raise ValueError("emissivity must lie in (0, 1]")
        if np.any(np.abs(f.sum(axis=1) - 1.0) > 1e-6):
            raise ValueError("view-factor rows must sum to one")
        self.f = f
        self.areas = areas
        self.emissivity = emissivity
        self.temperatures = temperatures
        self.sigma = float(sigma)
        self.reradiating = (np.zeros(n, dtype=bool) if reradiating is None
                            else np.asarray(reradiating, dtype=bool))
        if self.reradiating.shape != (n,):
            raise ValueError("reradiating must have length n")
        if np.any(~self.reradiating & ~np.isfinite(temperatures)):
            raise ValueError("temperature-specified surfaces need a finite temperature")

    def matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Linear system ``M J = b`` of the radiosity equations."""
        n = self.f.shape[0]
        m = np.eye(n) - (1.0 - self.emissivity)[:, None] * self.f
        b = self.emissivity * self.sigma * np.nan_to_num(self.temperatures) ** 4
        for i in np.nonzero(self.reradiating)[0]:
            m[i, :] = -self.f[i, :]
            m[i, i] += 1.0
            b[i] = 0.0
        return m, b

    def solve(self) -> Dict[str, np.ndarray]:
        m, b = self.matrix()
        j = np.linalg.solve(m, b)
        g = self.f @ j
        q = j - g
        return {"radiosity": j, "irradiation": g, "flux": q, "power": q * self.areas}

    def energy_residual(self, reference_power: Optional[float] = None) -> float:
        """``|sum_i q_i A_i|`` normalised by a *radiative* power scale (0 = closed).

        The scale defaults to ``sigma T_max^4 * sum(A)``, i.e. the largest power the
        enclosure could exchange.  Normalising by the largest *computed* surface
        power instead is useless in the one case the check matters most -- an
        isothermal enclosure, where every flux is zero and the ratio becomes
        ``0/0`` (measured: 1.0 for an exact solution).
        """
        res = self.solve()
        if reference_power is None:
            t_max = float(np.max(np.nan_to_num(self.temperatures)))
            reference_power = self.sigma * t_max ** 4 * float(np.sum(self.areas))
        if reference_power <= 0.0:
            raise ValueError("reference_power must be positive")
        return float(abs(np.sum(res["power"])) / reference_power)


def sky_depression(t_air_c: float, sigma: float = SIGMA) -> float:
    """``L_down / (sigma T_air^4)`` of the icing balance's Idso-Jackson sky model.

    The production correlation lives in
    :func:`tensorfvm.cht.ice_conjugate._sky_longwave`; this wrapper queries its
    *ratio* so a verification driver can check the physical ordering
    (``0.6 <= ratio < 1`` for a clear sky) without duplicating the correlation.
    """
    from .cht.ice_conjugate import _sky_longwave  # local: keep this module light

    t_air = float(t_air_c) + 273.15
    return float(_sky_longwave(t_air_c) / (sigma * t_air ** 4))


def sky_equivalent_temperature(l_down: float, sigma: float = SIGMA) -> float:
    """Radiative (equivalent black-body) sky temperature ``(L_down / sigma)^(1/4)`` in K."""
    if l_down <= 0.0:
        raise ValueError("L_down must be positive")
    return float((l_down / sigma) ** 0.25)