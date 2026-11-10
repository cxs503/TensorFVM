"""Exact conformal-mapping solution for a family of thick symmetric sections.

The Joukowski map

    zeta = z + a^2 / z                                                     (1)

sends the exterior of a circle ``|z - z_c| = R`` (``R = a + m`` with a real
centre offset ``z_c = -m``, so that the circle passes through the trailing edge
``z = a``) onto the exterior of a *cusped* symmetric aerofoil.  For that flow
the complex potential, the circulation that satisfies the Kutta condition and
the surface pressure are available in closed form, which makes the family an
exact anchor for the panel solver in :mod:`.section`:

    F(z)  = U e^{-i alpha} w + U e^{i alpha} R^2 / w
            + (i Gamma / (2 pi)) ln(w / R),      w = z - z_c             (2)
    Gamma = 4 pi U R sin(alpha - phi),           phi = arg(a - z_c)      (3)
    Cp    = 1 - |dF/dzeta|^2 / U^2,             dF/dzeta = (dF/dz)/(dzeta/dz)

with ``dzeta/dz = 1 - a^2/z^2``.  Equation (3) is *derived*, not fitted: at the
trailing edge ``dzeta/dz = 0``, so a finite velocity requires ``dF/dz = 0`` there,
which gives ``Gamma`` exactly (the step is repeated in the module docstring of
``tests/test_cavitation_rudder.py``).

Nothing in this module uses the panel solver, so agreement between the two is an
independent check of both.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["JoukowskiSection", "joukowski_for_thickness", "analytic_cp_circle"]


class JoukowskiSection:
    """A symmetric Joukowski section: ``zeta = z + a^2/z`` of a circle.

    Parameters
    ----------
    m : float
        centre offset (``z_c = -m``), which sets the thickness.
    a : float
        map parameter; the chord is ``~ 4a`` and the trailing edge sits at
        ``zeta = 2a``.
    """

    def __init__(self, m: float, a: float = 1.0):
        if m <= 0.0:
            raise ValueError("m must be positive")
        self.m, self.a = float(m), float(a)
        self.R = self.a + self.m
        self.z_c = -self.m

    # -- geometry ---------------------------------------------------------- #
    def point(self, theta):
        """Map circle points ``z = z_c + R exp(i theta)`` to the aerofoil."""
        th = np.asarray(theta, dtype=float)
        # the map is singular at the trailing edge (theta = 0): keep the sample a
        # hair away from it so that the 0/0 limit of dF/dzeta is not evaluated
        th = np.where(np.abs(th) < 1e-12, 1e-12, th)
        z = self.z_c + self.R * np.exp(1j * th)
        out = z + self.a ** 2 / z
        return out if out.ndim else complex(out)

    def dense_contour(self, n_dense: int = 2001) -> np.ndarray:
        """Dense contour ordered TE -> upper -> LE -> lower -> TE (closed)."""
        theta = math.pi * np.linspace(0.0, 1.0, n_dense)
        up = self.point(theta)
        lo = self.point(-theta)
        # clockwise: TE -> lower -> LE -> upper -> TE
        return np.column_stack([np.concatenate([lo.real, up.real[::-1][1:]]),
                                np.concatenate([lo.imag, up.imag[::-1][1:]])])

    def contour(self, n_half: int = 200) -> np.ndarray:
        """Panel nodes (arc-length resampled, leading edge refined), closed."""
        from .section import _resample_closed
        return _resample_closed(self.dense_contour(), 2 * n_half)

    @property
    def chord(self) -> float:
        """Chord length (LE at ``theta = pi``, TE at ``theta = 0``)."""
        le = self.point(math.pi)
        te = self.point(0.0)
        return float(abs(te - le))

    @property
    def thickness_ratio(self) -> float:
        """Max thickness / chord, from a dense surface sampling."""
        theta = np.linspace(0.0, math.pi, 2001)
        zeta = self.point(theta)
        return float(2.0 * np.max(zeta.imag) / self.chord)

    # -- aerodynamics ------------------------------------------------------ #
    def circulation(self, alpha_deg: float, u_inf: float = 1.0) -> float:
        """Kutta circulation, eq. (3)."""
        alpha = math.radians(alpha_deg)
        phi = math.atan2(0.0, self.a - self.z_c)      # arg(a - z_c) = 0 for real z_c > a
        return 4.0 * math.pi * u_inf * self.R * math.sin(alpha - phi)

    def cp(self, theta, alpha_deg: float, u_inf: float = 1.0) -> np.ndarray:
        """Exact ``Cp`` at the circle points ``theta`` (mapped to the aerofoil)."""
        alpha = math.radians(alpha_deg)
        gamma = self.circulation(alpha_deg, u_inf)
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        # the map is singular at the trailing edge (theta = 0): keep the sample a
        # hair away from it so that the 0/0 limit of dF/dzeta is not evaluated
        th = np.where(np.abs(th) < 1e-12, 1e-12, th)
        z = self.z_c + self.R * np.exp(1j * th)
        w = z - self.z_c
        dFdz = (u_inf * np.exp(-1j * alpha) - u_inf * np.exp(1j * alpha) * self.R ** 2 / w ** 2
                + 1j * gamma / (2.0 * math.pi * w))
        dzetadz = 1.0 - self.a ** 2 / z ** 2
        dFdzeta = dFdz / dzetadz
        speed2 = np.abs(dFdzeta) ** 2
        return 1.0 - speed2 / u_inf ** 2

    def cp_min(self, alpha_deg: float, u_inf: float = 1.0, n: int = 4001) -> float:
        theta = np.linspace(1e-6, math.pi - 1e-6, n)
        cp_up = self.cp(theta, alpha_deg, u_inf)
        cp_low = self.cp(-theta, alpha_deg, u_inf)
        return float(min(cp_up.min(), cp_low.min()))

    def cl(self, alpha_deg: float, u_inf: float = 1.0) -> float:
        """Exact lift coefficient ``Cl = 2 Gamma / (U c)`` (Kutta-Joukowski)."""
        return 2.0 * self.circulation(alpha_deg, u_inf) / (u_inf * self.chord)


def joukowski_for_thickness(thickness_ratio: float = 0.12, a: float = 1.0,
                            tol: float = 1e-10) -> JoukowskiSection:
    """Bisect the centre offset ``m`` so that the section has ``t/c`` requested."""
    lo, hi = 1e-4, 0.3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        tc = JoukowskiSection(mid, a).thickness_ratio
        if abs(tc - thickness_ratio) < tol:
            return JoukowskiSection(mid, a)
        if tc < thickness_ratio:
            lo = mid
        else:
            hi = mid
    raise RuntimeError("thickness bisection did not converge")


def analytic_cp_circle(theta, a: float = 1.0) -> np.ndarray:
    """``Cp`` of the unit circle in a uniform stream (no lift): ``1 - 4 sin^2``.

    Used as a smoke check of the mapping utilities: at ``m -> 0`` the Joukowski
    map degenerates to a circle of radius ``2a`` and the ``alpha = 0`` solution
    must reproduce ``1 - 4 sin^2(theta)``.
    """
    t = np.asarray(theta, dtype=float)
    return 1.0 - 4.0 * np.sin(t) ** 2