"""Radial Fourier body parametrisation with analytic shape derivatives.

The body is a star-shaped curve around a fixed centre ``(x0, y0)``::

    R(phi; theta) = r0 * (1 + sum_q theta_q * mode_q(phi))
    mode_q(phi)   = cos(k_q * phi)   or   sin(k_q * phi)

with ``k_q in {2, 3, 4, 5}`` and both cos/sin members, i.e. **eight** shape
parameters.  Modes ``k >= 2`` are used on purpose: the ``k = 1`` pair is a rigid
translation of the body (drag against a symmetric channel is then flat in that
direction and a relative-error gradient test would be ill conditioned).

The solid is not meshed.  It enters the flow through a *volume parametrisation*
(Brinkman penalisation, see :mod:`tensorfvm.adjoint_shape.channel`)::

    alpha(x; theta) = 0.5 * (1 - tanh(g(x; theta) / delta)),   g = r - R(phi)

``alpha -> 1`` inside the body, ``alpha -> 0`` outside, and ``delta`` is the
interface width (taken proportional to the mesh spacing by the driver).  The
geometry is therefore a smooth function of ``theta`` *on a fixed grid*::

    d alpha / d theta_q = 0.5 * sech^2(g / delta) / delta * dR / d theta_q

which is exactly what makes the discrete (hand-transposed) adjoint closed -- no
finite difference ever enters the sensitivity path.

Everything is vectorised over arbitrary arrays, so cell centres, ``u`` faces and
``v`` faces are all evaluated from the same closed form.
"""

from __future__ import annotations

import numpy as np

#: ``(wavenumber, trig)`` for the eight shape parameters, in fixed order.
MODES: tuple[tuple[int, str], ...] = (
    (2, "cos"),
    (2, "sin"),
    (3, "cos"),
    (3, "sin"),
    (4, "cos"),
    (4, "sin"),
    (5, "cos"),
    (5, "sin"),
)

#: Human readable parameter names, e.g. ``a2``/``b2`` for the ``k = 2`` pair.
PARAM_NAMES: tuple[str, ...] = tuple(
    f"{'a' if trig == 'cos' else 'b'}{k}" for k, trig in MODES
)

N_PARAMS = len(MODES)


def mode_value(index: int, phi):
    """``mode_q(phi)`` for parameter ``index`` (array in, array out)."""
    k, trig = MODES[index]
    return np.cos(k * phi) if trig == "cos" else np.sin(k * phi)


class FourierBody:
    """Star-shaped body ``R(phi; theta)`` with a smoothed characteristic field."""

    def __init__(
        self,
        center: tuple[float, float] = (2.0, 1.0),
        radius0: float = 0.5,
        delta: float = 0.1,
        quad: int = 1,
        subcell: float = 0.0,
    ) -> None:
        self.center = (float(center[0]), float(center[1]))
        self.radius0 = float(radius0)
        self.delta = float(delta)
        #: sub-cell quadrature order and box size (0 -> point sampling).  ``quad > 1``
        #: turns ``alpha`` into the *volume fraction* of the cell around the face
        #: centre, which is what makes ``C_D`` and its shape derivative converge
        #: under mesh refinement instead of carrying staircase noise.
        self.quad = int(quad)
        self.subcell = float(subcell)

    # -- geometry ---------------------------------------------------------
    def polar(self, x, y):
        """Radius ``r`` and polar angle ``phi`` of the points ``(x, y)``."""
        dx = np.asarray(x, dtype=float) - self.center[0]
        dy = np.asarray(y, dtype=float) - self.center[1]
        return np.hypot(dx, dy), np.arctan2(dy, dx)

    def radius(self, x, y, theta):
        """Body radius ``R(phi(x, y); theta)``."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (N_PARAMS,):
            raise ValueError(f"theta must have shape ({N_PARAMS},)")
        _, phi = self.polar(x, y)
        shape = np.ones_like(phi)
        for q in range(N_PARAMS):
            shape = shape + theta[q] * mode_value(q, phi)
        return self.radius0 * shape

    def radius_derivative(self, x, y):
        """``dR/dtheta_q``, shape ``(N_PARAMS, ...)`` (independent of theta)."""
        _, phi = self.polar(x, y)
        out = np.empty((N_PARAMS,) + np.shape(phi))
        for q in range(N_PARAMS):
            out[q] = self.radius0 * mode_value(q, phi)
        return out

    # -- penalisation indicator -------------------------------------------
    def indicator(self, x, y, theta):
        """Return ``(alpha, dalpha/dtheta)``: sub-cell average of :meth:`_indicator_at`."""
        if self.quad <= 1:
            return self._indicator_at(x, y, theta)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        offs = ((np.arange(self.quad, dtype=float) + 0.5) / self.quad - 0.5) * self.subcell
        alpha = np.zeros(np.shape(x))
        dalpha = np.zeros((N_PARAMS,) + np.shape(x))
        for oy in offs:
            for ox in offs:
                a, d = self._indicator_at(x + ox, y + oy, theta)
                alpha += a
                dalpha += d
        n = float(self.quad * self.quad)
        return alpha / n, dalpha / n

    def _indicator_at(self, x, y, theta):
        """Pointwise ``(alpha, dalpha/dtheta)``.

        ``alpha`` is a **compact-support** smoothstep of the signed distance
        ``g = r - R(phi)``::

            s     = clip(1/2 - g / (2 delta), 0, 1)
            alpha = s^2 (3 - 2 s)          (= 1 for g <= -delta, 0 for g >= +delta)
            dalpha/dtheta_q = 6 s (1 - s) / (2 delta) * dR/dtheta_q

        Compact support matters: with a ``tanh`` interface and
        ``1/eta = 1e3`` the exponential tail of ``alpha`` (a few percent two or
        three ``delta`` outside the surface) still acts as a visible extra
        blockage and inflates the drag by orders of magnitude.  The smoothstep is
        ``C^1`` in ``theta`` (its ``theta`` derivative vanishes at both ends of
        the ramp), which is all the discrete adjoint and a central difference
        need.
        """
        r, _ = self.polar(x, y)
        g = r - self.radius(x, y, theta)
        sc = np.clip(0.5 - g / (2.0 * self.delta), 0.0, 1.0)
        alpha = sc * sc * (3.0 - 2.0 * sc)
        dalpha = (6.0 * sc * (1.0 - sc) / (2.0 * self.delta)) * self.radius_derivative(
            x, y
        )
        return alpha, dalpha

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FourierBody(center={self.center}, radius0={self.radius0}, "
            f"delta={self.delta})"
        )