"""Flow over periodically arranged hills (ERCOFTAC UFR 3-30, Breuer et al. 2009).

Why this module exists
----------------------
The tree could already run a streamwise-periodic *plane channel*
(``channel_periodic``) and a fully periodic box (``periodic_mac``), but it could
not run the classical separated-flow benchmark that the ERCOFTAC database,
Breuer et al. (2009, Comput. Fluids 38:433, doi:10.1016/j.compfluid.2008.05.002)
and the whole RANS/LES closure literature use: a channel periodic in the
streamwise direction, bounded by a *curved* lower wall (a hill) and a flat upper
wall.  The separation bubble behind the hill crest, its reattachment length
``x_r/H`` and the mean velocity profiles are the standard acceptance metrics for
turbulence-model work, so the case has to be runnable in the native tree rather
than imported from a third-party code.

Geometry
--------
The contour is the six-cubic description of the ERCOFTAC UFR 3-30 test case
(ERCOFTAC Knowledge Base Wiki, "UFR 3-30 Test Case"; the same coefficients are
reproduced in Xiao et al., arXiv:1910.01264 App. A).  The published polynomials
are written in a normalised abscissa ``u = HILL_KAPPA * x/H``; ``HILL_KAPPA`` is
fixed here by requiring the six arcs to join and to land on ``y = 0`` at
``x/H = 1.929``, and is then checked against an independent published contour of
the same case (``tests/test_periodic_hill.py``).  Hill height is maximal at
``x/H = 0``, the wall is flat for ``x/H in [1.929, 7.071]`` and the shape is
mirrored about ``x/H = 4.5``.

Numerics
--------
Finite volume on a boundary-conforming ("terrain following") quadrilateral mesh::

    x = xi,   y = y_b(x) + eta * (H_top - y_b(x)),   xi periodic, eta in [0, 1]

so the streamwise faces are exactly vertical and every metric is analytic
(``eta_y = 1/Delta``, ``eta_x = -(1 - eta) y_b' / Delta``).  Cells are
trapezoids and face area vectors are built from the actual cell vertices, so the
discrete divergence, the discrete pressure gradient and the pressure Poisson
operator share the same face coefficients ``|A_f| / d_n`` and the projection is
consistent.

* predictor: explicit (blended central/upwind) convection, implicit diffusion
  solved exactly on the fixed FV operator, previous pressure;
* corrector: pressure-correction Poisson with periodic wrap in ``xi`` and
  ``dphi/dn = 0`` on both walls (in these coordinates the wall Neumann condition
  is literally ``p_eta = 0``);
* driven by a streamwise body force ``G`` adjusted every step so that the bulk
  velocity above the crest -- the ERCOFTAC definition
  ``u_b = 1/2.035 * int_h^{3.035h} u dy`` -- is held at ``U_b``, hence
  ``Re = U_b H / nu`` is a grid-independent input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "HILL_KAPPA",
    "PeriodicHillConfig",
    "PeriodicHillMesh",
    "PeriodicHillSolver",
    "hill_profile",
    "analytic_flat_channel",
]

#: Normalisation of the published hill-cubic abscissa, ``u = HILL_KAPPA * x/H``.
HILL_KAPPA = 27.96

#: (x/H start, x/H end, (a0, a1, a2, a3), clamp); y/H = a0 + a1 u + a2 u^2 + a3 u^3
_HILL_PIECES = (
    (0.0, 0.3214, (1.0, 0.0, 2.420e-4, -7.588e-5), "min"),
    (0.3214, 0.5, (0.8955, 3.484e-2, -3.629e-3, 6.749e-5), None),
    (0.5, 0.7143, (0.9213, 2.931e-2, -3.234e-3, 5.809e-5), None),
    (0.7143, 1.071, (1.445, -4.927e-2, 6.950e-4, -7.394e-6), None),
    (1.071, 1.429, (0.6401, 3.123e-2, -1.988e-3, 2.242e-5), None),
    (1.429, 1.929, (2.0139, -7.180e-2, 5.875e-4, 9.553e-7), "max"),
)


def _hill_arc(xn):
    """Leeward contour, ``xn = x/H in [0, 4.5]``; returns (y/H, d/dxn, d2/dxn2)."""
    xn = np.asarray(xn, dtype=float)
    out = np.zeros_like(xn)
    d1 = np.zeros_like(xn)
    d2 = np.zeros_like(xn)
    for index, (lo, hi, coeffs, clamp) in enumerate(_HILL_PIECES):
        last = index == len(_HILL_PIECES) - 1
        mask = (xn >= lo) & ((xn <= hi) if last else (xn < hi))
        if not np.any(mask):
            continue
        u = HILL_KAPPA * xn[mask]
        a0, a1, a2, a3 = coeffs
        val = a0 + a1 * u + a2 * u ** 2 + a3 * u ** 3
        dv = a1 + 2.0 * a2 * u + 3.0 * a3 * u ** 2
        d2v = 2.0 * a2 + 6.0 * a3 * u
        if clamp == "min":
            active = val < 1.0
            val = np.minimum(val, 1.0)
            dv = np.where(active, dv, 0.0)
            d2v = np.where(active, d2v, 0.0)
        elif clamp == "max":
            active = val > 0.0
            val = np.maximum(val, 0.0)
            dv = np.where(active, dv, 0.0)
            d2v = np.where(active, d2v, 0.0)
        out[mask] = val
        d1[mask] = dv * HILL_KAPPA
        d2[mask] = d2v * HILL_KAPPA ** 2
    return out, d1, d2


def hill_profile(x, height=1.0, length=9.0):
    """Hill contour ``y_b(x)`` plus ``dy_b/dx`` and ``d2y_b/dx2``.

    ``x`` is in metres; ``y_b`` in metres, derivatives per metre.  Periodic with
    period ``length`` and mirror-symmetric about ``length / 2``.
    """
    x = np.asarray(x, dtype=float)
    shape = x.shape
    flat = np.atleast_1d(x)
    xn = np.clip(np.mod(flat, length) / height, 0.0, length / height)
    reflected = xn > 0.5 * length / height
    s = np.where(reflected, length / height - xn, xn)
    y, d1, d2 = _hill_arc(s)
    d1 = np.where(reflected, -d1, d1)
    return (y * height).reshape(shape), (d1 / height).reshape(shape), (d2 / height).reshape(shape)


def analytic_flat_channel(y, height, body_force, nu):
    """Plane-Poiseuille solution, used to verify the solver with the hill removed."""
    y = np.asarray(y, dtype=float)
    return body_force / (2.0 * nu) * y * (height - y)


@dataclass
class PeriodicHillConfig:
    """Physical and numerical parameters of the periodic-hill case.

    ``reynolds`` is ``U_b * H / nu`` with ``U_b`` the ERCOFTAC bulk velocity
    above the crest (``height`` band from ``H`` to ``3.035 H``).
    """

    nx: int = 144
    ny: int = 72
    length_m: float = 9.0
    height_m: float = 1.0
    channel_top_m: float = 3.036
    reynolds: float = 2800.0
    bulk_velocity: float = 1.0
    density_kg_m3: float = 1.0
    hill_amplitude: float = 1.0
    time_step_s: float = 0.02
    convection_blend: float = 0.75
    body_force_relaxation: float = 0.5

    def __post_init__(self):
        if self.nx < 8 or self.ny < 4:
            raise ValueError("need nx >= 8 and ny >= 4")
        for name in ("length_m", "height_m", "channel_top_m", "reynolds",
                     "bulk_velocity", "density_kg_m3", "time_step_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid {name}: need finite > 0")
        if self.channel_top_m <= 2.0 * self.height_m:
            raise ValueError("channel_top_m must clear the crest by more than h")
        if not 0.0 <= self.convection_blend <= 1.0:
            raise ValueError("convection_blend must be in [0, 1]")
        if not 0.0 < self.body_force_relaxation <= 1.0:
            raise ValueError("body_force_relaxation must be in (0, 1]")

    @property
    def nu(self):
        return self.bulk_velocity * self.height_m / self.reynolds

    @property
    def dx(self):
        return self.length_m / self.nx

    def with_options(self, **kwargs):
        return PeriodicHillConfig(**{**self.__dict__, **kwargs})

    def hill(self, x):
        y, d1, d2 = hill_profile(x, self.height_m, self.length_m)
        return self.hill_amplitude * y, self.hill_amplitude * d1, self.hill_amplitude * d2


@dataclass
class PeriodicHillMesh:
    """Boundary-conforming mesh; every metric is analytic in (xi, eta)."""

    config: PeriodicHillConfig

    def __post_init__(self):
        c = self.config
        self.dx = c.length_m / c.nx
        self.deta = 1.0 / c.ny
        xv = np.linspace(0.0, c.length_m, c.nx + 1)
        etap = np.linspace(0.0, 1.0, c.ny + 1)
        self.x_vertices = xv
        self.eta_vertices = etap
        y_b, dy_b, _ = c.hill(xv)
        self.y_b, self.dy_b = y_b, dy_b
        self.delta_vertices = c.channel_top_m - y_b
        self.y_vertices = y_b[None, :] + etap[:, None] * self.delta_vertices[None, :]
        self._build()

    def _build(self):
        c = self.config
        ny, nx = c.ny, c.nx
        xv, yv = self.x_vertices, self.y_vertices
        x0, x1 = xv[:-1], xv[1:]
        y00, y10 = yv[:-1, :-1], yv[:-1, 1:]
        y01, y11 = yv[1:, :-1], yv[1:, 1:]
        self.xc = 0.5 * (x0 + x1)[None, :] * np.ones((ny, 1))
        self.yc = 0.25 * (y00 + y10 + y01 + y11)
        self.volume = (x1 - x0)[None, :] * 0.5 * ((y01 - y00) + (y11 - y10))
        eta_c = np.linspace(0.5 * self.deta, 1.0 - 0.5 * self.deta, ny)[:, None]
        _, dyb_c, _ = c.hill(self.xc[0])
        delta_c = np.broadcast_to((c.channel_top_m - c.hill(self.xc[0])[0])[None, :],
                                  (ny, nx)).copy()
        self.delta_cell = delta_c
        self.eta_cell = np.broadcast_to(eta_c, (ny, nx)).copy()
        self.eta_x = -(1.0 - self.eta_cell) * dyb_c[None, :] / delta_c
        self.eta_y = 1.0 / delta_c
        # east / west faces: exactly vertical, outward normal (1, 0)
        self.area_e = (yv[1:, 1:] - yv[:-1, 1:]).copy()
        self.x_e = np.broadcast_to(xv[1:][None, :], (ny, nx)).copy()
        self.y_e = 0.5 * (yv[:-1, 1:] + yv[1:, 1:])
        # north / south faces at eta levels j = 0..ny
        dy_face = yv[:, 1:] - yv[:, :-1]
        dx_face = np.broadcast_to((xv[1:] - xv[:-1])[None, :], (ny + 1, nx))
        self.area_n = np.hypot(dy_face, dx_face)
        self.unx_n = -dy_face / self.area_n
        self.uny_n = dx_face / self.area_n
        self.x_n = 0.5 * (xv[None, :-1] + xv[None, 1:]) * np.ones((ny + 1, 1))
        self.y_n = 0.5 * (yv[:, :-1] + yv[:, 1:])
        # projected distances cell-centre -> face-centre
        self.ip = (np.arange(nx) + 1) % nx
        self.im = (np.arange(nx) - 1) % nx
        dx_e = self.xc[:, self.ip] - self.xc
        # minimum image: the face between i = nx-1 and i = 0 is a +dx step across
        # the periodic seam, not a -(L - dx) step the other way round the domain
        dx_e = dx_e - c.length_m * np.round(dx_e / c.length_m)
        self.dn_e = np.abs(dx_e)
        dn_n = np.zeros((ny + 1, nx))
        dn_n[1:-1] = ((self.xc[1:] - self.xc[:-1]) * self.unx_n[1:-1]
                      + (self.yc[1:] - self.yc[:-1]) * self.uny_n[1:-1])
        dn_n[0] = -((self.x_n[0] - self.xc[0]) * self.unx_n[0]
                    + (self.y_n[0] - self.yc[0]) * self.uny_n[0])
        dn_n[-1] = ((self.x_n[-1] - self.xc[-1]) * self.unx_n[-1]
                    + (self.y_n[-1] - self.yc[-1]) * self.uny_n[-1])
        self.dn_n = dn_n
        if np.any(self.dn_e <= 0.0) or np.any(dn_n <= 0.0):
            raise ValueError("mesh has non-positive projected face distances")
        # non-orthogonality of the eta faces: the centre-to-centre vector is
        # ``d = alpha n + beta t`` with ``t = (-n_y, n_x)``.  On this
        # terrain-following mesh the slant is not small (|beta/alpha| up to
        # ~0.79 over the hill), so the two-point normal derivative
        # ``(phi_N - phi_P)/alpha`` alone is only first-order consistent; the
        # deferred correction below removes the tangential part.
        self.beta_n = np.zeros((ny + 1, nx))
        dy_d = self.yc[1:] - self.yc[:-1]
        dx_d = self.xc[1:] - self.xc[:-1]
        self.beta_n[1:-1] = (-dx_d * self.uny_n[1:-1] + dy_d * self.unx_n[1:-1])
        # east/west faces are vertical, so their non-orthogonality is simply the
        # centre-to-centre y offset (t = (0, 1))
        self.beta_e = self.yc[:, self.ip] - self.yc
        # wall tangent pointing downstream along each wall
        tau_b = np.stack([-self.uny_n[0], self.unx_n[0]], axis=-1)
        tau_t = np.stack([self.uny_n[-1], -self.unx_n[-1]], axis=-1)
        self.tau_bottom = tau_b * np.where(tau_b[:, 0] < 0.0, -1.0, 1.0)[:, None]
        self.tau_top = tau_t * np.where(tau_t[:, 0] < 0.0, -1.0, 1.0)[:, None]
        self._least_squares_weights()
        self._compact_weights()

    def _compact_weights(self):
        """Per-cell 2x2 inverse for the compact (face-normal) gradient.

        Each face supplies one equation ``(grad phi) . n_f = (phi_N - phi_P)/d_n``
        weighted by ``|A_f|``; the wall faces supply the Neumann condition
        ``(grad phi) . n_f = 0``.  Unlike the wide (2 dx) central least-squares
        stencil in :meth:`_least_squares_weights`, this operator is **not** blind
        to the odd-even (checkerboard) mode: a checkerboard has
        ``(phi_N - phi_P)/d_n = +-2 phi_P / d_n != 0``, so the pressure couples to
        it and the fractional-step scheme stays non-singular.
        """
        c = self.config
        ny, nx = c.ny, c.nx
        a11 = np.zeros((ny, nx))
        a12 = np.zeros((ny, nx))
        a22 = np.zeros((ny, nx))
        # east / west faces: n = (+-1, 0), area = area_e, distance = dn_e
        a11 += self.area_e / self.dn_e
        a11 += self.area_e[:, self.im] / self.dn_e[:, self.im]
        # eta faces: interior levels only (1..ny-1).  A Neumann wall face
        # carries no two-point information (zero jump), so feeding it into the
        # normalisation would bias the stencil towards a zero normal derivative
        # in the wall-adjacent cells and destroy exactness on linear fields.
        nx_n, ny_n, area_n = self.unx_n, self.uny_n, self.area_n
        interior = np.arange(1, ny)
        a = area_n[interior] / self.dn_n[interior]
        w11 = a * nx_n[interior] ** 2
        w12 = a * nx_n[interior] * ny_n[interior]
        w22 = a * ny_n[interior] ** 2
        for w, acc in ((w11, a11), (w12, a12), (w22, a22)):
            acc[:-1] += w          # face L is the north face of cell L-1
            acc[1:] += w           # ... and the south face of cell L
        det = a11 * a22 - a12 * a12
        if np.any(det <= 0.0):
            raise ValueError("degenerate compact-gradient stencil")
        self.cg_i11 = a22 / det
        self.cg_i12 = -a12 / det
        self.cg_i22 = a11 / det

    def _least_squares_weights(self):
        """Per-cell 2x2 inverse of the weighted least-squares gradient stencil."""
        c = self.config
        ny, nx = c.ny, c.nx
        a11 = np.zeros((ny, nx))
        a12 = np.zeros((ny, nx))
        a22 = np.zeros((ny, nx))
        for shift in (1, -1):
            ii = (np.arange(nx) + shift) % nx
            dx = self.xc[:, ii] - self.xc
            dx = dx - c.length_m * np.round(dx / c.length_m)      # minimum image
            dy = self.yc[:, ii] - self.yc
            w = 1.0 / np.maximum(dx * dx + dy * dy, 1e-300)
            a11 += w * dx * dx
            a12 += w * dx * dy
            a22 += w * dy * dy
        for shift in (1, -1):
            jj = np.arange(ny) + shift
            inside = (jj >= 0) & (jj <= ny - 1)
            jc = np.clip(jj, 0, ny - 1)
            level = np.where(shift > 0, jj, jj)
            level = np.clip(level, 0, ny)
            ins = inside[:, None]
            dx = np.where(ins, self.xc[jc] - self.xc, self.x_n[level] - self.xc)
            dy = np.where(ins, self.yc[jc] - self.yc, self.y_n[level] - self.yc)
            w = np.where(ins, 1.0 / np.maximum(dx * dx + dy * dy, 1e-300), 0.0)
            a11 += w * dx * dx
            a12 += w * dx * dy
            a22 += w * dy * dy
        det = a11 * a22 - a12 * a12
        if np.any(det <= 0.0):
            raise ValueError("degenerate least-squares stencil")
        self.lsq_i11 = a22 / det
        self.lsq_i12 = -a12 / det
        self.lsq_i22 = a11 / det


class PeriodicHillSolver:
    """Fractional-step finite-volume solver on a :class:`PeriodicHillMesh`.

    The state is three cell-centred fields ``u``, ``v`` (Cartesian components,
    shape ``(ny, nx)``) and ``p`` (kinematic pressure ``p/rho``).  Both walls are
    no-slip and impermeable; the streamwise direction wraps.
    """

    def __init__(self, config: PeriodicHillConfig):
        self.config = config
        self.mesh = PeriodicHillMesh(config)
        self.nx, self.ny = config.nx, config.ny
        self.n = self.nx * self.ny
        self.nu = config.nu
        self.u = np.zeros((self.ny, self.nx))
        self.v = np.zeros((self.ny, self.nx))
        self.p = np.zeros((self.ny, self.nx))
        self.body_force = 0.0
        self.steps = 0
        self._weight, self._weight_sum = self._bulk_weights()
        self._flux_e = None            # projected (divergence-free) face fluxes
        self._flux_n = None
        self._flux_steps = -1
        self._build(config.time_step_s)

    # -------------------------------------------------------------- geometry
    def _bulk_weights(self):
        c, m = self.config, self.mesh
        lo, hi = c.height_m, c.height_m + 2.035 * c.height_m
        height = m.volume / m.dx
        bottom = m.yc - 0.5 * height
        top = m.yc + 0.5 * height
        overlap = np.clip(np.minimum(top, hi) - np.maximum(bottom, lo), 0.0, None)
        weight = overlap / height
        return weight, float((weight * m.volume).sum())

    def bulk_velocity(self, u=None):
        """ERCOFTAC bulk velocity ``1/2.035 int_h^{3.035h} u dy`` (x-averaged)."""
        u = self.u if u is None else u
        return float((self._weight * u * self.mesh.volume).sum() / self._weight_sum)

    def _face_flux(self, u, v):
        """Outward normal volume flux ``|A| u.n`` on every face.

        ``flux_e[j, i]`` is the flux through the east face of cell ``(j, i)``
        (positive in +x, out of that cell); ``flux_n[j, i]`` is the flux through
        eta level ``j`` with its normal pointing up (+eta).  Wall faces
        (``flux_n[0]``, ``flux_n[-1]``) are impermeable and stay zero.
        """
        m = self.mesh
        uf_e = 0.5 * (u + u[:, m.ip])
        flux_e = m.area_e * uf_e
        flux_n = np.zeros((self.ny + 1, self.nx))
        uf = 0.5 * (u[1:] + u[:-1])
        vf = 0.5 * (v[1:] + v[:-1])
        flux_n[1:-1] = m.area_n[1:-1] * (uf * m.unx_n[1:-1] + vf * m.uny_n[1:-1])
        return flux_e, flux_n

    def divergence(self, u=None, v=None):
        """Discrete ``div u`` per cell (1/s) from the same face fluxes."""
        u = self.u if u is None else u
        v = self.v if v is None else v
        m = self.mesh
        flux_e, flux_n = self._face_flux(u, v)
        net = flux_e - flux_e[:, m.im] + flux_n[1:] - flux_n[:-1]
        return net / m.volume

    def _convection(self, phi, flux_e, flux_n):
        """Net convective outflow of ``phi`` per unit volume (blended scheme)."""
        m, beta = self.mesh, self.config.convection_blend
        out = np.zeros_like(phi)
        # east faces: between (j, i) and (j, i+1)
        face = beta * 0.5 * (phi + phi[:, m.ip]) + (1.0 - beta) * np.where(
            flux_e > 0.0, phi, phi[:, m.ip])
        contrib = flux_e * face / m.volume
        out += contrib
        out[:, m.ip] -= contrib
        # eta faces: level L between rows L-1 and L
        f = flux_n[1:-1]
        lower = phi[:-1]
        upper = phi[1:]
        face = beta * 0.5 * (lower + upper) + (1.0 - beta) * np.where(f > 0.0, lower, upper)
        contrib = f * face
        out[:-1] -= contrib / m.volume[:-1]
        out[1:] += contrib / m.volume[1:]
        return out

    def compact_gradient(self, field, nonorthogonal_sweeps=2):
        """Compact face-normal gradient on the terrain-following mesh.

        Every face supplies one equation ``(grad phi) . n_f = g_n``; the per-cell
        2x2 inverse of ``sum_f |A_f| / d_n * n_f n_f^T`` turns the set into a
        gradient.  This is the pressure-gradient operator of the projection: the
        momentum predictor, the velocity corrector and the pressure Poisson
        operator in :meth:`_build` all derive from the *same* face coefficients
        ``|A_f| / d_n``.  The wide central stencil of :meth:`gradient` does not
        have that property (it annihilates the odd-even mode exactly, see the
        module log), so it is kept only as a diagnostic.

        Two corrections relative to round 2 (mission 16672):

        * **south-face index shift (defect vi).**  The eta-face term was written
          as ``c_n[j] n[j] jump[j]`` with ``jump[j] = phi[j+1] - phi[j]``, i.e.
          every cell was handed its *north* jump as if it were the south one.
          Row 0 therefore received a wall face driven by an interior jump and
          row ``ny-1`` lost its south face altogether, so the operator was not
          even exact on a linear field (measured: ``|grad(0.7x + 0.3y) - (0.7,
          0.3)|_inf = 11.19`` on the 32x16 hill mesh).  The face jump now
          belongs to the *level* ``L`` it sits on, and a face contributes the
          same ``c_L n_L jump_L`` to both cells it separates (``L-1`` as north
          face, ``L`` as south face).
        * **non-orthogonal deferred correction.**  On this mesh
          ``d = alpha n + beta t`` with ``|beta/alpha|`` up to 0.79, so
          ``(phi_N - phi_P)/alpha`` differs from ``(grad phi).n`` at first
          order.  The tangential part is supplied by the previous gradient
          estimate and subtracted in ``nonorthogonal_sweeps`` explicit sweeps,

              g_n = jump/alpha - (beta/alpha) * g_t,

          where ``g_t`` is the average of ``grad phi . t`` over the two cells.
          Only the two-point difference stays implicit in the Poisson operator
          (making the correction implicit would degenerate the operator, see
          ``docs/case-periodic-hills/ROUND2.md`` §5.3).
        """
        m = self.mesh
        phi = np.asarray(field, dtype=float)
        ce = m.area_e / m.dn_e ** 2
        ce_w = m.area_e[:, m.im] / m.dn_e[:, m.im] ** 2
        cn = m.area_n / m.dn_n ** 2
        b1 = ce * (phi[:, m.ip] - phi)                  # east face, n = (+1, 0)
        b1 += ce_w * (phi - phi[:, m.im])               # west face, n = (-1, 0)
        jump = np.zeros_like(phi)
        jump[1:] = phi[1:] - phi[:-1]                   # level L: phi_L - phi_{L-1}
        # interior eta faces (levels 1..ny-1); the two wall levels are Neumann
        # (dp/dn = 0, zero jump) and carry no two-point information at all
        face_n = (cn[1:-1] * m.unx_n[1:-1], cn[1:-1] * m.uny_n[1:-1])
        js = jump[1:]
        c1 = face_n[0] * js
        c2 = face_n[1] * js
        b1[:-1] += c1
        b1[1:] += c1
        b2 = np.zeros_like(phi)
        b2[:-1] += c2
        b2[1:] += c2
        gx = m.cg_i11 * b1 + m.cg_i12 * b2
        gy = m.cg_i12 * b1 + m.cg_i22 * b2
        if int(nonorthogonal_sweeps) <= 0:
            return gx, gy
        # The tangential part is estimated from the *wide* least-squares
        # gradient (:meth:`gradient`), which is second-order accurate for smooth
        # fields on this skewed mesh (measured 0.9 % at 64x32); iterating from
        # the compact estimate itself does not converge because |beta/alpha|
        # reaches ~1 in the hill columns.  One corrector pass per sweep, with the
        # corrected gradient fed back, converges linearly (measured 2.5 % ->
        # 0.87 % -> 0.30 % on 32x16 -> 64x32 -> 128x64).
        pgx, pgy = self.gradient(phi)
        beta_e = m.beta_e
        for _ in range(int(nonorthogonal_sweeps)):
            n1 = b1.copy()
            n2 = b2.copy()
            g_t_e = 0.5 * (pgy + pgy[:, m.ip])
            d1 = -ce * beta_e * g_t_e
            n1 += d1                 # east face, normal (+1, 0)
            n1[:, m.ip] += d1        # ... same face, as the west face of cell i+1
            beta = m.beta_n[1:-1]
            tx = -m.uny_n[1:-1]
            ty = m.unx_n[1:-1]
            g_t = 0.5 * ((pgx[:-1] + pgx[1:]) * tx + (pgy[:-1] + pgy[1:]) * ty)
            dc1 = -cn[1:-1] * beta * m.unx_n[1:-1] * g_t
            dc2 = -cn[1:-1] * beta * m.uny_n[1:-1] * g_t
            n1[:-1] += dc1
            n1[1:] += dc1
            n2[:-1] += dc2
            n2[1:] += dc2
            gx = m.cg_i11 * n1 + m.cg_i12 * n2
            gy = m.cg_i12 * n1 + m.cg_i22 * n2
            pgx, pgy = gx, gy
        return gx, gy

    def gradient(self, field):
        """Weighted least-squares cell gradient (periodic in x, one-sided at walls)."""
        m = self.mesh
        ny, nx = self.ny, self.nx
        b1 = np.zeros((ny, nx))
        b2 = np.zeros((ny, nx))
        for shift in (1, -1):
            ii = (np.arange(nx) + shift) % nx
            dx = m.xc[:, ii] - m.xc
            dx = dx - self.config.length_m * np.round(dx / self.config.length_m)
            dy = m.yc[:, ii] - m.yc
            dphi = field[:, ii] - field
            w = 1.0 / np.maximum(dx * dx + dy * dy, 1e-300)
            b1 += w * dx * dphi
            b2 += w * dy * dphi
        for shift in (1, -1):
            jj = np.arange(ny) + shift
            inside = (jj >= 0) & (jj <= ny - 1)
            jc = np.clip(jj, 0, ny - 1)
            level = np.clip(jj, 0, ny)
            ins = inside[:, None]
            dx = np.where(ins, m.xc[jc] - m.xc, m.x_n[level] - m.xc)
            dy = np.where(ins, m.yc[jc] - m.yc, m.y_n[level] - m.yc)
            # the wall face carries the wall value for velocity (zero) and, for
            # pressure, the zero-normal-derivative condition (no jump)
            dphi = np.where(ins, field[jc] - field, 0.0)
            w = np.where(ins, 1.0 / np.maximum(dx * dx + dy * dy, 1e-300), 0.0)
            b1 += w * dx * dphi
            b2 += w * dy * dphi
        return (m.lsq_i11 * b1 + m.lsq_i12 * b2,
                m.lsq_i12 * b1 + m.lsq_i22 * b2)# ------------------------------------------------------------- operators
    def _cell(self, j, i):
        return j * self.nx + (i % self.nx)

    def _build(self, dt):
        """Assemble the fixed diffusion and pressure-correction operators.

        Face coefficients are ``c_f = |A_f| / d_n``.  The momentum operator is
        the volume-normalised Laplacian ``(1/Vol) sum_f c_f (phi_N - phi_P)``
        with a *Dirichlet* contribution from both no-slip walls, which carries
        the wall shear and therefore the whole momentum balance.  The pressure
        operator is the same face-coefficient sum **without** the volume
        normalisation, because the right-hand side is the net face *flux*
        (``sum_f A_f u_f``) rather than its volume average:

            sum_f c_f (p_P - p_N) = -(1/dt) sum_f A_f u*_f ,

        i.e. ``(diag_prs - off) p = -net/dt`` as used by :meth:`advance`.
        Wall faces are Neumann (``dp/dn = 0``) for the pressure and therefore
        do not appear in the pressure matrix at all.

        Round-2 fixes (mission 16503), both regression-tested in
        ``tests/test_periodic_hill.py``:

        * the diagonal accumulators were 1-D but written into ``(ny, nx)``
          arrays, so ``diag += coef`` broadcast row ``j``'s coefficients onto
          *every* row (diagonal ~``ny`` times too stiff, LU returned ``O(1e-3)``
          of the right-hand side);
        * ``c_n[0]`` / ``c_n[-1]`` (the two wall faces) were never filled, so
          the wall Dirichlet term was exactly zero and the channel free-slipped
          (plug flow, ``u_max / u_b = 1.003`` instead of ``1.5``).
        """
        from scipy import sparse
        from scipy.sparse import linalg as spla

        self.time_step_s = float(dt)
        m = self.mesh
        ny, nx = self.ny, self.nx
        n = self.n
        self.c_e = m.area_e / m.dn_e
        self.c_n = m.area_n / m.dn_n                     # all ny+1 eta levels
        vols = m.volume
        rows, cols, vals = [], [], []
        prows, pcols, pvals = [], [], []
        diag_mom = np.zeros((ny, nx))
        diag_prs = np.zeros((ny, nx))
        for j in range(ny):
            p = np.arange(j * nx, (j + 1) * nx)
            ce_e = self.c_e[j]
            ce_w = self.c_e[j, m.im]
            cn_s = self.c_n[j]                           # wall when j == 0
            cn_n = self.c_n[j + 1]                       # wall when j == ny - 1
            # momentum: (1/Vol) sum_f c_f, wall Dirichlet enters the diagonal only
            rows.append(p); cols.append(self._cell(j, np.arange(nx) + 1))
            vals.append(ce_e / vols[j])
            rows.append(p); cols.append(self._cell(j, np.arange(nx) - 1))
            vals.append(ce_w / vols[j])
            if j > 0:
                rows.append(p); cols.append(self._cell(j - 1, np.arange(nx)))
                vals.append(cn_s / vols[j])
            if j < ny - 1:
                rows.append(p); cols.append(self._cell(j + 1, np.arange(nx)))
                vals.append(cn_n / vols[j])
            diag_mom[j] = (ce_e + ce_w + cn_s + cn_n) / vols[j]
            # pressure: same c_f, no volume division, walls are Neumann
            prows.append(p); pcols.append(self._cell(j, np.arange(nx) + 1))
            pvals.append(ce_e)
            prows.append(p); pcols.append(self._cell(j, np.arange(nx) - 1))
            pvals.append(ce_w)
            if j > 0:
                prows.append(p); pcols.append(self._cell(j - 1, np.arange(nx)))
                pvals.append(cn_s)
            if j < ny - 1:
                prows.append(p); pcols.append(self._cell(j + 1, np.arange(nx)))
                pvals.append(cn_n)
            diag_prs[j] = (ce_e + ce_w
                           + (cn_s if j > 0 else 0.0) + (cn_n if j < ny - 1 else 0.0))
        off = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        # L = (1/Vol) sum_f c_f (phi_N - phi_P); wall Dirichlet already in diag_mom
        self.laplacian = off - sparse.diags(diag_mom.ravel())
        self.momentum = (sparse.diags(np.full(n, 1.0 / self.time_step_s))
                         - self.nu * self.laplacian).tocsc()
        self._momentum_lu = spla.splu(self.momentum)
        p_off = sparse.csr_matrix(
            (np.concatenate(pvals), (np.concatenate(prows), np.concatenate(pcols))),
            shape=(n, n))
        pressure = (sparse.diags(diag_prs.ravel()) - p_off).tocsr()
        # The operator has the constant vector in its null space (periodic in xi,
        # Neumann on both walls).  It is regularised with ~1e-10 * |diag| instead
        # of pinning a row: a pinned row would leave that one cell's continuity
        # equation unsatisfied, which is exactly what the mass-conservation gate
        # below measures.
        self._pressure_regularisation = 1e-10 * float(np.mean(diag_prs))
        self.pressure = pressure + sparse.identity(n, format="csr") * self._pressure_regularisation
        self._pressure_lu = spla.splu(self.pressure.tocsc())
        self._volume_flat = vols.ravel()

    def set_time_step(self, dt, rebuild=True):
        if rebuild or float(dt) != self.time_step_s:
            self._build(dt)

    # ------------------------------------------------------------------ step
    def advance(self, steps=1, statistics=None):
        """Advance ``steps`` fractional steps and leave a mass-conserving state.

        Convection uses the *projected* face fluxes of the previous step (they
        are divergence-free by construction), the pressure-correction Poisson is
        the compact face-coefficient operator of :meth:`_build`, and the
        correction is applied to the **face fluxes** as well as to the cell
        velocities::

            net_new = net - dt * sum_f c_f (phi_N - phi_P) = 0

        Correcting only the cell velocities (as round 1 did) cannot conserve mass
        here: the least-squares cell gradient annihilates the checkerboard mode
        exactly (measured 0.0), so it is not the adjoint of the face-averaged
        divergence and ``div(u)`` stays O(1).
        """
        m, c, dt = self.mesh, self.config, self.time_step_s
        for _ in range(int(steps)):
            ub = self.bulk_velocity()
            self.body_force += c.body_force_relaxation * (c.bulk_velocity - ub) / dt
            gpx, gpy = self.compact_gradient(self.p)
            if self._flux_e is None or self._flux_steps != self.steps:
                self.refresh_fluxes()
            flux_e, flux_n = self._flux_e, self._flux_n
            rhs_u = (self.u / dt - self._convection(self.u, flux_e, flux_n)
                     - gpx + self.body_force)
            rhs_v = (self.v / dt - self._convection(self.v, flux_e, flux_n) - gpy)
            us = self._momentum_lu.solve(rhs_u.ravel()).reshape(self.ny, self.nx)
            vs = self._momentum_lu.solve(rhs_v.ravel()).reshape(self.ny, self.nx)
            us_e, us_n = self._face_flux(us, vs)
            # Rhie-Chow damping (explicit): replace the *interpolated cell*
            # pressure gradient in the face flux by the *compact* face
            # difference.  Only the compact difference enters the implicit
            # operator below; if the interpolated gradient were made implicit as
            # well, the two would cancel at leading order (measured: both are
            # O(h) phi'' with equal magnitude and opposite sign) and the Poisson
            # operator would degenerate.
            p_n = self.p
            gp_x, gp_y = gpx, gpy
            rc_e = dt * (0.5 * (gp_x[:, m.ip] + gp_x) - (p_n[:, m.ip] - p_n) / m.dn_e)
            us_e = us_e + m.area_e * rc_e
            gp_nx = 0.5 * (gp_x[1:] + gp_x[:-1])
            gp_ny = 0.5 * (gp_y[1:] + gp_y[:-1])
            gp_dot_n = (gp_nx * m.unx_n[1:-1] + gp_ny * m.uny_n[1:-1])
            d_n = (p_n[1:] - p_n[:-1]) / m.dn_n[1:-1]
            us_n = us_n.copy()
            us_n[1:-1] = us_n[1:-1] + m.area_n[1:-1] * dt * (gp_dot_n - d_n)
            net = us_e - us_e[:, m.im] + us_n[1:] - us_n[:-1]
            # (diag_prs - off) phi = sum_f c_f (phi_P - phi_N) = -net/dt, so that
            # the flux correction below leaves net_new = net + dt*M(phi) = 0
            phi = self._pressure_lu.solve(-net.ravel() / dt).reshape(self.ny, self.nx)
            # flux correction: compact face differences of phi -> exact continuity
            self._flux_e = us_e - dt * self.c_e * (phi[:, m.ip] - phi)
            self._flux_n = np.zeros_like(us_n)
            self._flux_n[1:-1] = us_n[1:-1] - dt * self.c_n[1:-1] * (phi[1:] - phi[:-1])
            phix, phiy = self.compact_gradient(phi)
            self.u = us - dt * phix
            self.v = vs - dt * phiy
            self.p = self.p + phi
            self.steps += 1
            self._flux_steps = self.steps
            if statistics is not None:
                statistics.accumulate(self)
        return self

    def refresh_fluxes(self):
        """Recompute the face fluxes from the current cell velocities."""
        self._flux_e, self._flux_n = self._face_flux(self.u, self.v)
        self._flux_steps = self.steps

    def flux_divergence(self):
        """Discrete ``div u`` of the conservative (projected) face fluxes."""
        m = self.mesh
        flux_e, flux_n = self._flux_e, self._flux_n
        net = flux_e - flux_e[:, m.im] + flux_n[1:] - flux_n[:-1]
        return net / m.volume

    # --------------------------------------------------------- diagnostics
    def wall_shear(self, u=None, v=None):
        """Wall shear (kinematic, ``tau/rho``) on both walls at the face centres."""
        u = self.u if u is None else u
        v = self.v if v is None else v
        m = self.mesh
        tau_b = (u[0] * m.tau_bottom[:, 0] + v[0] * m.tau_bottom[:, 1]) / m.dn_n[0]
        tau_t = (u[-1] * m.tau_top[:, 0] + v[-1] * m.tau_top[:, 1]) / m.dn_n[-1]
        return tau_b, tau_t

    def reattachment(self, u=None, v=None):
        """Zero crossings of the bottom-wall shear.

        Returns ``(separation, reattachment)`` in units of the hill height,
        ``nan`` when no crossing is found (period counted from the crest at
        ``x/H = 0``).
        """
        tau_b, _ = self.wall_shear(u, v)
        x = self.mesh.x_n[0] / self.config.height_m
        crossings = []
        for i in range(len(x)):
            a, b = tau_b[i], tau_b[(i + 1) % len(x)]
            if a == 0.0:
                continue
            if a < 0.0 <= b:
                t = -a / (b - a)
                xc = x[i] + t * ((x[(i + 1) % len(x)] - x[i]) % self.config.length_m)
                crossings.append(xc)
        sep = next((c for c in crossings if c < 1.9), float("nan"))
        rea = next((c for c in crossings if 1.9 <= c <= 7.1), float("nan"))
        return sep, rea

    def profile_at(self, x_over_h):
        """Mean ``u`` vs ``y`` sampled on the vertical line ``x = x_over_h * H``.

        Returns ``(y/H, u/U_b)`` on the mesh rows; linear interpolation in x.
        """
        h = self.config.height_m
        x = x_over_h * h
        xv = self.mesh.xc[0]
        i = int(np.clip(np.searchsorted(xv, x) - 1, 0, self.nx - 1))
        x0, x1 = xv[i], xv[(i + 1) % self.nx]
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        u = (1.0 - t) * self.u[:, i] + t * self.u[:, (i + 1) % self.nx]
        y = ((1.0 - t) * self.mesh.yc[:, i] + t * self.mesh.yc[:, (i + 1) % self.nx]) / h
        return y, u / self.config.bulk_velocity


class Statistics:
    """Reynolds-averaged fields accumulated over a stationary window."""

    def __init__(self, solver):
        self.samples = 0
        self.u = np.zeros_like(solver.u)
        self.v = np.zeros_like(solver.u)
        self.body_force = 0.0

    def accumulate(self, solver):
        self.samples += 1
        self.u += solver.u
        self.v += solver.v
        self.body_force += solver.body_force

    def mean(self):
        n = max(self.samples, 1)
        return self.u / n, self.v / n, self.body_force / n