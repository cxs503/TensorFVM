"""Time-domain numerical wave tank on a **surface-fitted (sigma) grid**.

Why a different discretisation from :mod:`tensorfvm.wave_tank`
--------------------------------------------------------------
The legacy VOF tank recorded one blocking defect: the first-harmonic wave
amplitude decayed by ~80 % within one wavelength, i.e. the *numerical* scheme,
not the physics, set the wave field.  Every candidate cause was an interface
scheme (THINC re-steepening, cell-centred surface Dirichlet, upwind blend) --
dissipation that exists only because the free surface cuts through a Cartesian
mesh.

This module removes the root cause instead of tuning it: the vertical
coordinate is normalised by the *instantaneous* free surface,

    sigma in [0, 1],   y(x, sigma, t) = -h + sigma * H(x, t),   H = h + eta(x, t)

so the free surface **is** the top coordinate line.  No interface to
reconstruct, no cell cut by the surface, no air phase, and the surface pressure
Dirichlet ``p = 0`` sits exactly where the surface is.  Wave damping can then
only come from the momentum discretisation, which is what one wants to measure.

Continuity in the sigma frame is the plain cell divergence

    A_x[i+1] u[i+1,j] - A_x[i] u[i,j] + A_s (w[i,j+1] - w[i,j]) = 0,
    A_x = dsig*Hf,   A_s = dx,

because the grid-velocity terms of the moving sigma lines cancel identically:
the geometric conservation law holds *by construction* (the contravariant
vertical velocity through the surface line is zero).  Momentum is

    du/dt = -u du/dx - w du/dy + nu lap(u) - dp/dx          (rho = 1)
    dw/dt = -u dw/dx - w dw/dy + nu lap(w) - dp/dy - g

with a central/upwind blend on the advective derivatives.

One time step (:meth:`SigmaWaveTank.step`)
-----------------------------------------
1. rebuild the geometry ``H, Ax, Wx, Ws, y`` from ``eta`` (the sigma grid moves);
2. body mask -> open face areas ``ox`` (x-faces) and ``os`` (sigma-faces);
3. momentum predictor ``u*, w*`` (advection + gravity);
4. relaxation zone ``u* <- (1-b) u* + b u_target`` for ``x < L_r`` (waves2Foam
   recipe, smoothstep weight) and sponge ``u* <- u* (1 - s dt)`` for
   ``x > L - L_d``;
5. pressure projection: solve ``(B W^-1 B^T) p = B u*/dt`` with ``p = 0`` on the
   free surface, ``dp/dn = 0`` on the bottom, the walls and the body.  ``B`` is
   assembled from the *open* face areas, so the projection is an exact algebraic
   projection: after the correction ``max|div u|`` is at round-off even with a
   body inside the tank (:attr:`div_residual` records it every step);
6. ``eta += dt * w_surface`` -- mass conservation is a flux statement, so the
   water volume is conserved to round-off with no volume ledger;
7. body: pressure integral over the wetted, partially covered faces -> heave
   force and pitch moment -> added-mass-informed integration (surge fixed,
   heave and pitch free).

The last point is why a sigma grid is worth the complexity: the measured
elevation is a state variable, not something recovered from a smeared ``alpha``
field, so it cannot be quantised to ``dy``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from ..wave_tank.theory import GRAVITY
from .wave_target import WaveTarget

__all__ = ["TankConfig", "SigmaWaveTank", "points_in_polygon"]


def points_in_polygon(px, py, poly) -> np.ndarray:
    """Vectorised even-odd (ray crossing) point-in-polygon test."""
    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    inside = np.zeros(px.shape, dtype=bool)
    poly = np.asarray(poly, dtype=float)
    n = poly.shape[0]
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if y1 == y2:
            continue
        straddle = (y1 > py) != (y2 > py)
        if not straddle.any():
            continue
        xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddle & (px < xint)
    return inside


def polygon_properties(poly, density: float = 1000.0):
    """Area, centroid and second moment of a simple polygon (per unit width)."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area2 = float(np.sum(cross))
    area = 0.5 * area2
    if abs(area) < 1e-14:
        raise ValueError("degenerate section polygon")
    cx = float(np.sum((x + xn) * cross)) / (3.0 * area2)
    cy = float(np.sum((y + yn) * cross)) / (3.0 * area2)
    # second moment about the centroid (per unit width, rho = 1)
    i_o = 0.0
    for k in range(p.shape[0]):
        x1, y1 = p[k]
        x2, y2 = p[(k + 1) % p.shape[0]]
        cr = x1 * y2 - x2 * y1
        i_o += cr * (x1 * x1 + x1 * x2 + x2 * x2 + y1 * y1 + y1 * y2 + y2 * y2)
    i_o /= 12.0
    i_c = abs(i_o) - abs(area) * (cx * cx + cy * cy)
    return float(abs(area)), cx, cy, abs(i_c) * density


@dataclass
class TankConfig:
    length: float = 12.0          # tank length [m]
    depth: float = 1.0            # still-water depth h [m]
    nx: int = 240                 # columns
    ns: int = 32                  # sigma cells (uniform in sigma)
    gravity: float = GRAVITY
    density: float = 1000.0
    advection_blend: float = 0.15   # 0 = central (non-dissipative), 1 = upwind
    nu: float = 0.0                 # kinematic viscosity [m^2/s]
    relax_waves: float = 1.0        # relaxation-zone length in wavelengths
    damp_waves: float = 1.5         # sponge length in wavelengths (>= 1.5 required)
    damp_strength: float = 6.0      # sponge rate = damp_strength * omega [1/s]
    cg_rtol: float = 1e-11
    cg_maxiter: int = 4000
    mask_sub: int = 8               # face sub-samples for the body fractions


def _solve_spd(A, b, x0, rtol, maxiter):
    try:
        return cg(A, b, x0=x0, rtol=rtol, maxiter=maxiter)
    except TypeError:                                     # scipy < 1.12
        return cg(A, b, x0=x0, tol=rtol, maxiter=maxiter)


class SigmaWaveTank:
    """2-D vertical-plane numerical wave tank with heave/pitch body support."""

    def __init__(self, cfg: TankConfig):
        self.cfg = cfg
        self.nx = int(cfg.nx)
        self.ns = int(cfg.ns)
        self.dx = cfg.length / self.nx
        self.dsig = 1.0 / self.ns
        self.x = (np.arange(self.nx) + 0.5) * self.dx       # column centres
        self.xf = np.arange(self.nx + 1) * self.dx          # x-faces
        self.sigf = np.arange(self.ns + 1) * self.dsig
        self.sigc = (np.arange(self.ns) + 0.5) * self.dsig
        self.eta = np.zeros(self.nx)
        self.u = np.zeros((self.nx + 1, self.ns))
        self.w = np.zeros((self.nx, self.ns + 1))
        self.t = 0.0
        self.wave: WaveTarget | None = None
        self.body: dict | None = None
        self._build_grid()
        self.p = np.zeros((self.nx, self.ns))               # non-hydrostatic pressure
        self.ox = np.ones((self.nx + 1, self.ns))
        self.os = np.ones((self.nx, self.ns + 1))
        self.bfrac_x = np.zeros((self.nx + 1, self.ns))     # body-blocked only
        self.bfrac_s = np.zeros((self.nx, self.ns + 1))     # body-blocked only
        self.solid = np.zeros((self.nx, self.ns), dtype=bool)
        self.div_residual = 0.0
        self.volume0 = self.volume()
        self.volume_drift = 0.0

    # ------------------------------------------------------------------ grid
    def _build_grid(self):
        cfg = self.cfg
        self.H = cfg.depth + self.eta
        self.Hf = np.empty(self.nx + 1)
        self.Hf[1:-1] = 0.5 * (self.H[:-1] + self.H[1:])
        self.Hf[0] = self.H[0]
        self.Hf[-1] = self.H[-1]
        self.Ax = self.dsig * self.Hf                        # x-face area
        self.Wx = self.dx * self.dsig * self.Hf              # x-face dual volume
        self.V = self.dx * self.dsig * self.H                # cell volume
        self.Ws = self.V.copy()                              # sigma-face dual volume
        self.yf = -cfg.depth + np.outer(self.H, self.sigf)   # (nx, ns+1)
        self.yc = -cfg.depth + np.outer(self.H, self.sigc)   # (nx, ns)
        self.dy = np.broadcast_to(self.dsig * self.H[:, None], (self.nx, self.ns))

    def volume(self) -> float:
        return float(np.sum(self.dx * self.H))

    # ----------------------------------------------------------------- setup
    def set_wave(self, amp: float, period: float, phase0: float = 0.0,
                 order2: bool = True) -> WaveTarget:
        self.wave = WaveTarget(amp, period, self.cfg.depth, self.cfg.gravity,
                               phase0=phase0, order2=order2)
        return self.wave

    def set_body(self, polygon, mass=None, inertia=None, added_mass=(0.0, 0.0),
                 ref=None, damping=(0.0, 0.0), motion="free", forced=None):
        """Install a floating 2-D section.

        ``ref`` is the reference point for the pitch rotation and the moment; it
        defaults to the section's own vertical axis ``(cx, 0)``.  Rotating about
        the *tank origin* instead would sink the body whenever it pitches
        (``x * theta`` of vertical displacement), which is a bookkeeping error
        that looks exactly like a spurious pitch-heave coupling.
        """
        poly = np.asarray(polygon, dtype=float)
        area, cx, cy, inertia_c = polygon_properties(poly, self.cfg.density)
        if ref is None:
            ref = (float(cx), 0.0)
        ref = np.asarray(ref, dtype=float)
        if mass is None:
            mass = self.cfg.density * area
        if inertia is None:
            inertia = inertia_c + mass * ((cx - ref[0]) ** 2 + (cy - ref[1]) ** 2)
        self.body = {
            "poly": poly, "mass": float(mass), "inertia": float(inertia),
            "area": float(area), "cg": np.array([cx, cy]),
            "added": np.asarray(added_mass, dtype=float),
            "damping": np.asarray(damping, dtype=float),
            "motion": str(motion),
            "forced": forced,
            "ref": np.asarray(ref, dtype=float),
            "xi": np.zeros(2), "vel": np.zeros(2), "acc": np.zeros(2),
            "force": np.zeros(2), "moment": 0.0,
            "time_series": {"t": [], "heave": [], "pitch": [], "fx": [], "fy": [],
                            "m5": []},
        }
        return self.body

    def initialize_from_wave(self, t: float = 0.0):
        """Set eta, u, w to the analytic target wave (propagation test start)."""
        if self.wave is None:
            raise RuntimeError("set_wave() first")
        self.eta = np.asarray(self.wave.eta(self.x, t), dtype=float).copy()
        self._build_grid()
        ut, wt = self._target_fields(t)
        self.u = ut.copy()
        self.w = wt.copy()
        self.p = np.zeros((self.nx, self.ns))
        self.t = float(t)
        self.volume0 = self.volume()

    # -------------------------------------------------------------- body mask
    def body_polygon(self):
        """Section polygon in tank coordinates for the current heave/pitch.

        The rotation is taken about :attr:`body["ref"]`, *not* about the tank
        origin.  Rotating the absolute polygon about ``(0, 0)`` displaces a
        section sitting at ``x0`` vertically by ``-x0 * theta``; at
        ``x0 = 4 m`` and ``theta = 0.02 rad`` that is 8 cm of spurious heave,
        which the buoyancy then reads back as a real draft change and the
        heave/pitch coupling runs away.  That was the defect behind the
        diverging RAO case recorded in ``docs/wave-tank-td/README.md``.
        """
        b = self.body
        if b is None:
            return None
        th = float(b["xi"][1])
        c, s = math.cos(th), math.sin(th)
        rel = b["poly"] - b["ref"]                        # shift to reference
        p = rel @ np.array([[c, s], [-s, c]]) + b["ref"]   # rotate about ref
        p = p + np.array([0.0, float(b["xi"][0])])        # heave = pure vertical
        return p

    def _update_mask(self):
        nx, ns, cfg = self.nx, self.ns, self.cfg
        poly = self.body_polygon()
        ox_sub = np.ones((nx + 1, ns))
        os_sub = np.ones((nx, ns + 1))
        self.solid = np.zeros((nx, ns), dtype=bool)
        if poly is not None:
            sub = int(cfg.mask_sub)
            ssub = (np.arange(sub) + 0.5) / sub
            lo, hi = poly.min(axis=0), poly.max(axis=0)
            yf_f = -cfg.depth + np.outer(self.Hf, self.sigf)
            ylo, yhi = yf_f[:, :-1], yf_f[:, 1:]
            y = ylo[:, :, None] + ssub[None, None, :] * (yhi - ylo)[:, :, None]
            ok = (self.xf >= lo[0] - self.dx) & (self.xf <= hi[0] + self.dx)
            if ok.any():
                ii = np.nonzero(ok)[0]
                xx = np.broadcast_to(self.xf[ii, None, None], (ii.size, ns, sub))
                cnt = points_in_polygon(xx, y[ii], poly).reshape(ii.size, ns, sub).sum(-1)
                ox_sub[ii] = 1.0 - cnt / sub
            xs = self.xf[:-1][:, None] + ssub[None, :] * self.dx
            okj = (self.x >= lo[0] - self.dx) & (self.x <= hi[0] + self.dx)
            if okj.any():
                jj = np.nonzero(okj)[0]
                xx = np.broadcast_to(xs[jj][:, None, :], (jj.size, ns + 1, sub))
                yy = np.broadcast_to(self.yf[jj][:, :, None], (jj.size, ns + 1, sub))
                cnt = points_in_polygon(xx, yy, poly).reshape(jj.size, ns + 1, sub).sum(-1)
                os_sub[jj] = 1.0 - cnt / sub
            self.solid = points_in_polygon(
                np.broadcast_to(self.x[:, None], (nx, ns)), self.yc, poly).reshape(nx, ns)
        # topological closure: a face with exactly one fully solid neighbour is
        # blocked.  Grid-aligned sections put every sub-sample exactly on the
        # polygon edge, where a point-in-polygon test is ambiguous; this term
        # makes the mask exact for them and leaves cut cells to the sub-sampling.
        closed_x = np.zeros((nx + 1, ns), dtype=bool)
        closed_x[1:-1] = self.solid[1:, :] ^ self.solid[:-1, :]
        closed_s = np.zeros((nx, ns + 1), dtype=bool)
        closed_s[:, 1:-1] = self.solid[:, 1:] ^ self.solid[:, :-1]
        self.ox = np.where(closed_x, 0.0, ox_sub)
        self.os = np.where(closed_s, 0.0, os_sub)
        self.ox[0, :] = 1.0 if self.wave is not None else 0.0
        self.ox[-1, :] = 0.0
        self.os[:, 0] = 0.0
        closed = ((self.ox[:-1, :] <= 1e-12) & (self.ox[1:, :] <= 1e-12)
                  & (self.os[:, :-1] <= 1e-12) & (self.os[:, 1:] <= 1e-12))
        self.solid = self.solid | closed
        # Body-only blocked fractions.  ``ox``/``os`` also carry the *walls* (the
        # inlet, the outlet and the tank bottom are set to zero there), so a
        # moving-body flux built from ``1 - ox`` / ``1 - os`` lets the 12 m tank
        # bottom follow the section: the measured "added mass" then came out at a
        # basin-scale 1.1e4 kg/m instead of O(rho B^2) and the harmonic radiation
        # force lost its reactive part.  The walls are boundaries of the fluid
        # domain, not body surface; only the section may move them.
        self.bfrac_x = np.where(closed_x, 1.0, 1.0 - ox_sub)
        self.bfrac_s = np.where(closed_s, 1.0, 1.0 - os_sub)
        self.bfrac_x[0, :] = 0.0
        self.bfrac_x[-1, :] = 0.0
        self.bfrac_s[:, 0] = 0.0
        self.bfrac_s[:, -1] = 0.0

    # ------------------------------------------------------------- predictor
    def body_flux_gradient(self):
        """``d(body volume source)/d(heave rate)`` and ``.../d(pitch rate)``.

        The body enters the continuity statement *only* through the blocked
        face fractions (see :meth:`body_normal_flux`), so the source is exactly
        linear in the rigid-body velocity.  These two cell fields are that
        linear map's columns; they are what turns the body force into the
        discrete adjoint of the constraint (and, in :meth:`step`, into the
        implicit added-mass matrix).
        """
        A_block_x = self.bfrac_x * self.Ax[:, None]
        A_block_s = self.bfrac_s * self.dx
        ref = np.asarray(self.body["ref"], dtype=float)
        y_sig = -self.cfg.depth + np.outer(self.Hf, self.sigc)      # (nx+1, ns)
        x_face = np.broadcast_to(self.x[:, None], (self.nx, self.ns + 1))
        du_h = np.zeros_like(y_sig)
        dw_h = np.ones_like(x_face)
        du_p = -(y_sig - ref[1])
        dw_p = (x_face - ref[0])
        out = []
        for du, dw in ((du_h, dw_h), (du_p, dw_p)):
            out.append(A_block_x[1:, :] * du[1:, :] - A_block_x[:-1, :] * du[:-1, :]
                       + A_block_s[:, 1:] * dw[:, 1:] - A_block_s[:, :-1] * dw[:, :-1])
        return out

    def body_force_reaction(self, p=None):
        """Generalised fluid force that is the *discrete reaction* of the mask.

        ``F_k = -sum_cells q * d(source)/d(v_k)``.  This is the discrete adjoint
        of the body-velocity constraint, so the power identity

            F . v_body = -sum_cells q * source(v_body)

        holds at round-off (test ``test_reaction_force_consistent_with_flux``).
        Integrating a *separately reconstructed* pressure field over the
        stair-step wetted faces instead (what :meth:`body_force_interpolated`
        does, and what the previous campaign node used) breaks that identity and
        pumps energy into the coupling -- that is what made the free-body run
        diverge at ``A33/m ~ 2.7``.

        Returns ``(force[2], moment_about_ref)`` for the *non-hydrostatic*
        pressure ``q`` only; the hydrostatic + Froude-Krylov part comes from
        :meth:`hydrostatic_force`.
        """
        if self.body is None:
            return np.zeros(2), 0.0
        p = self.p if p is None else np.asarray(p, dtype=float)
        grads = self.body_flux_gradient()
        rho = self.cfg.density
        f = -rho * np.array([float((p * gk).sum()) for gk in grads])
        return np.array([f[0], 0.0]), float(f[1])

    def wetted_quadrature(self, sub: int = 8):
        """Quadrature weights of the wetted (rotated) section contour.

        Returns ``(points, sample points, normals, weights)``: the mid-points of
        ``sub`` pieces
        per polygon edge restricted to the submerged part, the outward normal
        of each piece, and ``dl``.  Because the elevation is interpolated
        linearly along an edge and the pressure of the model is
        ``g(eta(x) - y) + q`` (linear in ``y``, piecewise linear in ``x``), the
        contour integral on these weights is *exact* for the hydrostatic part --
        the waterplane restoring and the Froude-Krylov force of the section
        therefore carry no stair-step dependence at all, unlike the cut-cell
        face-fraction sum.
        """
        poly = np.asarray(self.body_polygon(), dtype=float)
        pts, nrm, wgt = [], [], []
        s = (np.arange(sub) + 0.5) / sub
        n = len(poly)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            d = b - a
            length = math.hypot(d[0], d[1])
            if length <= 0.0:
                continue
            n_out = np.array([d[1], -d[0]]) / length      # outward normal (CCW)
            p_mid = a[None, :] + s[:, None] * d[None, :]
            eta = self.eta_at(p_mid[:, 0])
            wet = p_mid[:, 1] < eta
            if not wet.any():
                continue
            pts.append(p_mid[wet])
            nrm.append(np.broadcast_to(n_out, p_mid[wet].shape))
            wgt.append(np.full(int(wet.sum()), length / sub))
        if not pts:
            z = np.zeros((0, 2))
            return z, z, z, np.zeros(0)
        pts, nrm, wgt = np.vstack(pts), np.vstack(nrm), np.concatenate(wgt)
        # ``q`` lives on cell centres and is *zero* inside the body (the solid rows
        # are masked out of the Poisson system), so the field must be sampled half
        # a cell outside the section; the moment arm stays on the exact contour.
        return pts, pts + 0.5 * self.dx * nrm, nrm, wgt

    def hydrostatic_force(self, quad=None):
        """Froude-Krylov + hydrostatic load ``-oint rho g (eta - y) n dl``.

        The tank's pressure is split as ``p = g(eta - y) + q`` (a surface-fitted
        sigma grid, see the module docstring), so the body load splits the same
        way.  The still-water Archimedes formula used by the previous node
        (``rho g * area(wetted at z = 0)``) misses the Froude-Krylov force
        entirely: it cannot see the incident elevation over the footprint, which
        is ``O(rho g B a) = 235 N/m`` here -- of the same order as the whole
        heave excitation.
        """
        if self.body is None:
            return np.zeros(2), 0.0
        pts, _ps, nrm, wgt = self.wetted_quadrature() if quad is None else quad
        if pts.shape[0] == 0:
            return np.zeros(2), 0.0
        rho, g = self.cfg.density, self.cfg.gravity
        ref = np.asarray(self.body["ref"], dtype=float)
        p = rho * g * (self.eta_at(pts[:, 0]) - pts[:, 1]) * wgt
        force = -np.array([float((p * nrm[:, 0]).sum()), float((p * nrm[:, 1]).sum())])
        r = pts - ref
        mom = -float(np.sum(p * (r[:, 0] * nrm[:, 1] - r[:, 1] * nrm[:, 0])))
        return force, mom

    def body_force_dynamic(self, p, quad=None):
        """``q``-part of the load: ``-oint rho q n dl`` -- a linear functional of q."""
        if self.body is None:
            return np.zeros(2), 0.0
        pts, ps, nrm, wgt = self.wetted_quadrature() if quad is None else quad
        if pts.shape[0] == 0:
            return np.zeros(2), 0.0
        rho = self.cfg.density
        ref = np.asarray(self.body["ref"], dtype=float)
        q = rho * np.asarray(self.pressure_at(ps[:, 0], ps[:, 1], p), dtype=float) * wgt
        force = -np.array([float((q * nrm[:, 0]).sum()), float((q * nrm[:, 1]).sum())])
        r = pts - ref
        mom = -float(np.sum(q * (r[:, 0] * nrm[:, 1] - r[:, 1] * nrm[:, 0])))
        return force, mom

    def body_loads(self, p=None, quad=None):
        """Total fluid load on the section: Froude-Krylov/hydrostatic + ``q``.

        Both terms are integrated on the *same* contour weights, so the load is
        one linear functional of the pressure field -- which is what the implicit
        free-body step needs to keep the added mass inside the mass matrix.
        """
        if self.body is None:
            return np.zeros(2), 0.0
        quad = self.wetted_quadrature() if quad is None else quad
        fh, mh = self.hydrostatic_force(quad)
        fq, mq = self.body_force_dynamic(self.p if p is None else p, quad)
        return fh + fq, mh + mq

    def discrete_added_mass(self):
        """The tank's own added-mass matrix ``A = -dF/d(acceleration)`` [kg/m].

        Obtained from the linearity of the projection: the pressure produced by
        a unit acceleration in each DOF is a solve of the *same* Poisson matrix
        with the body-flux pattern as the right-hand side, and the reaction of
        that pressure is the added mass.  It is therefore a measurement of the
        discretisation, not an input: the free-body time integration uses it as
        an implicit mass augmentation (so the coupling is added-mass free) and
        the potential-flow reference adjudicates its value.
        """
        if self.body is None:
            return np.zeros((2, 2))
        M, free = self._poisson_matrix()
        grads = self.body_flux_gradient()
        quad = self.wetted_quadrature()
        pk = []
        for gk in grads:
            rhs = -gk.ravel()
            rhs[~free] = 0.0
            pk.append(self._cg_solve(M, rhs))
        # dF_j / d(acceleration_k): the pressure produced by a unit acceleration
        # in DOF k, integrated with the same contour functional as the load
        g = np.zeros((2, 2))
        for k in range(2):
            f, m = self.body_force_dynamic(pk[k], quad)
            g[:, k] = (f[1], m)
        return -g

    def _ddx(self, f, uvel):
        cen = np.empty_like(f)
        cen[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.dx)
        cen[0] = (f[1] - f[0]) / self.dx
        cen[-1] = (f[-1] - f[-2]) / self.dx
        up = np.empty_like(f)
        up[1:-1] = np.where(uvel[1:-1] > 0.0,
                            (f[1:-1] - f[:-2]) / self.dx,
                            (f[2:] - f[1:-1]) / self.dx)
        up[0] = (f[1] - f[0]) / self.dx
        up[-1] = (f[-1] - f[-2]) / self.dx
        b = self.cfg.advection_blend
        return (1.0 - b) * cen + b * up

    def _ddy(self, f, wvel):
        dy1 = self.dsig * self.H                             # (nx,)
        dy = dy1[:, None]
        cen = np.empty_like(f)
        cen[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dy)
        cen[:, 0] = (f[:, 1] - f[:, 0]) / dy1
        cen[:, -1] = (f[:, -1] - f[:, -2]) / dy1
        up = np.empty_like(f)
        up[:, 1:-1] = np.where(wvel[:, 1:-1] > 0.0,
                               (f[:, 1:-1] - f[:, :-2]) / dy,
                               (f[:, 2:] - f[:, 1:-1]) / dy)
        up[:, 0] = (f[:, 1] - f[:, 0]) / dy1
        up[:, -1] = (f[:, -1] - f[:, -2]) / dy1
        b = self.cfg.advection_blend
        return (1.0 - b) * cen + b * up

    def _predict(self, dt: float):
        g = self.cfg.gravity
        u, w = self.u, self.w
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        wc = 0.5 * (w[:, :-1] + w[:, 1:])
        adv_u = uc * self._ddx(uc, uc) + wc * self._ddy(uc, wc)
        adv_w = uc * self._ddx(wc, uc) + wc * self._ddy(wc, wc)
        if self.cfg.nu > 0.0:
            dy = (self.dsig * self.H)[:, None]
            lap_u = np.zeros_like(uc)
            lap_u[1:-1] = (uc[2:] - 2 * uc[1:-1] + uc[:-2]) / self.dx ** 2
            lap_u[:, 1:-1] = lap_u[:, 1:-1] + (uc[:, 2:] - 2 * uc[:, 1:-1] + uc[:, :-2]) / dy ** 2
            lap_w = np.zeros_like(wc)
            lap_w[1:-1] = (wc[2:] - 2 * wc[1:-1] + wc[:-2]) / self.dx ** 2
            lap_w[:, 1:-1] = lap_w[:, 1:-1] + (wc[:, 2:] - 2 * wc[:, 1:-1] + wc[:, :-2]) / dy ** 2
            adv_u = adv_u - self.cfg.nu * lap_u
            adv_w = adv_w - self.cfg.nu * lap_w
        a_u = np.zeros_like(u)
        a_u[1:-1, :] = 0.5 * (adv_u[:-1, :] + adv_u[1:, :])
        a_w = np.zeros_like(w)
        a_w[:, 1:-1] = 0.5 * (adv_w[:, :-1] + adv_w[:, 1:])
        a_w[:, -1] = adv_w[:, -1]
        # The pressure is split as p = g(eta - y) + q with q the NON-hydrostatic
        # part (q = 0 on the free surface).  The barotropic part g*d(eta)/dx is
        # then an exact column quantity -- no sigma metric error -- and gravity is
        # cancelled analytically by the hydrostatic part in the vertical
        # momentum.  Only q is projected, which is the whole point: the sigma
        # pressure-gradient error can only pollute the small non-hydrostatic
        # remainder.
        detadx = np.zeros(self.nx + 1)
        detadx[1:-1] = (self.eta[1:] - self.eta[:-1]) / self.dx
        us = u - dt * (a_u + g * detadx[:, None])
        ws = w - dt * a_w
        us[0, :] = 0.0
        us[-1, :] = 0.0
        ws[:, 0] = 0.0
        return us, ws

    # ---------------------------------------------------------- forcing zones
    def relax_weight(self) -> np.ndarray:
        """Smoothstep weight 1 at the generation wall -> 0 at the zone end."""
        cfg, wave = self.cfg, self.wave
        if wave is None or cfg.relax_waves <= 0.0:
            return np.zeros(self.nx + 1)
        xr = cfg.relax_waves * wave.wavelength
        t = np.clip(self.xf / xr, 0.0, 1.0)
        return 1.0 - (3.0 * t ** 2 - 2.0 * t ** 3)

    def set_sponge(self, period: float):
        """Activate the absorbing beach for a *radiation* run (no incident wave).

        A forced-oscillation run in otherwise still water still needs the beach:
        without it the tank is a closed basin whose longitudinal seiche period is
        ``2L / sqrt(g h) = 7.7 s`` here, and a forcing at 0.8 s excites it
        resonantly (measured: the velocity grew linearly from 0 to 0.58 m/s in
        0.6 s and the harmonic fit never settled).  The sponge rate is scaled by
        the radiation frequency exactly as it is by the wave frequency in a wave
        run, so the beach length stays ``damp_waves`` radiation wavelengths.
        """
        from ..wave_tank.theory import dispersion_k
        omega = 2.0 * math.pi / float(period)
        k = dispersion_k(omega, self.cfg.depth, self.cfg.gravity)
        self._sponge_reference = (omega, 2.0 * math.pi / k)
        return self._sponge_reference

    def sponge_rate(self, x) -> np.ndarray:
        """Quadratic ramp of the absorbing-beach rate [1/s] along the tank."""
        cfg = self.cfg
        x = np.asarray(x, dtype=float)
        ref = (self.wave.omega, self.wave.wavelength) if self.wave is not None \
            else getattr(self, "_sponge_reference", None)
        if ref is None or cfg.damp_waves <= 0.0 or cfg.damp_strength <= 0.0:
            return np.zeros_like(x)
        omega, wavelength = ref
        x0 = cfg.length - cfg.damp_waves * wavelength
        s = np.clip((x - x0) / max(cfg.length - x0, 1e-12), 0.0, 1.0) ** 2
        return cfg.damp_strength * omega * s

    def _target_fields(self, t):
        wave = self.wave
        left = np.vstack([self.yc[:1], self.yc[:-1]])
        yf_u = 0.5 * (left + self.yc)                        # (nx, ns)
        yt_u = np.vstack([yf_u, self.yc[-1:]])               # (nx+1, ns)
        xt_u = np.broadcast_to(self.xf[:, None], (self.nx + 1, self.ns))
        ut, _ = wave.velocities(xt_u, yt_u, t)
        xt_w = np.broadcast_to(self.x[:, None], (self.nx, self.ns + 1))
        _, wt = wave.velocities(xt_w, self.yf, t)
        return ut, wt

    def _apply_zones(self, us, ws, t, dt):
        wrx = self.relax_weight()
        if wrx.any():
            ut, wt = self._target_fields(t)
            us[:] = wrx[:, None] * ut + (1.0 - wrx[:, None]) * us
            wrw = (0.5 * (wrx[:-1] + wrx[1:]))[:, None]
            ws[:] = wrw * wt + (1.0 - wrw) * ws
        rate_x = self.sponge_rate(self.xf)
        if rate_x.any():
            us *= np.clip(1.0 - dt * rate_x, 0.0, 1.0)[:, None]
            rate_c = self.sponge_rate(self.x)
            ws *= np.clip(1.0 - dt * rate_c, 0.0, 1.0)[:, None]
        return us, ws

    # ------------------------------------------------------------ projection
    def _poisson_matrix(self):
        nx, ns = self.nx, self.ns
        Aopen_x = self.ox * self.Ax[:, None]
        Aopen_s = self.os * self.dx
        cax = Aopen_x ** 2 / self.Wx[:, None]
        cas = Aopen_s ** 2 / self.Ws[:, None]
        idx = np.arange(nx * ns).reshape(nx, ns)
        ii, jj = np.meshgrid(np.arange(1, nx), np.arange(ns), indexing="ij")
        c = cax[1:nx, :]
        r1, r2 = idx[ii, jj], idx[ii - 1, jj]
        kk = np.arange(1, ns)
        ii2, kk2 = np.meshgrid(np.arange(nx), kk, indexing="ij")
        c2 = cas[ii2, kk2]
        ra, rb = idx[ii2, kk2], idx[ii2, kk2 - 1]
        ra3 = idx[np.arange(nx), ns - 1]
        # The free surface is a Dirichlet face half a cell above the top pressure
        # node: its dual volume is half a cell, so A/W = 2/dy (not 1/dy).  Both
        # the matrix entry and the velocity correction carry that factor 2, which
        # is what makes the discrete free-surface stiffness equal g.
        c3 = 2.0 * cas[np.arange(nx), ns]
        # Symmetric operator M = B W^-1 B^T (positive definite).  The projection is
        # u = u* - dt W^-1 B^T p with flux(u) = 0, hence  M p = -flux(u*)/dt.
        rows = np.concatenate([r1.ravel(), r1.ravel(), r2.ravel(), r2.ravel(),
                               ra.ravel(), ra.ravel(), rb.ravel(), rb.ravel(), ra3])
        cols = np.concatenate([r1.ravel(), r2.ravel(), r2.ravel(), r1.ravel(),
                               ra.ravel(), rb.ravel(), rb.ravel(), ra.ravel(), ra3])
        vals = np.concatenate([c.ravel(), -c.ravel(), c.ravel(), -c.ravel(),
                               c2.ravel(), -c2.ravel(), c2.ravel(), -c2.ravel(), c3])
        M = sparse.coo_matrix((vals, (rows, cols)), shape=(nx * ns, nx * ns)).tocsr()
        free = ~self.solid.ravel()
        M = sparse.diags(free.astype(float)) @ M + sparse.diags((~free).astype(float))
        return M, free

    def body_normal_flux(self, vel=None):
        """Divergence source produced by the moving section (per cell).

        The cut-cell mask alone makes the body a *static* obstacle: without this
        term a forced oscillation produces no fluid motion at all, so the tank
        can measure no added mass, no radiation damping and no radiation
        reaction (measured: a driven heave of 1 cm returned exactly zero force).
        The faces that the mask blocks are body surface, and the fluid must
        leave them at the body's normal velocity,

            v_x = -th' (z - z_ref),          v_z = h' + th' (x - x_ref),

        so the continuity statement gains ``A_blocked v_body . n``.  The pressure
        correction only acts on the open fraction, which keeps the projection
        exact (``max|div u|`` stays at round-off).
        """
        if self.body is None:
            return 0.0
        b = self.body
        vel = np.asarray(b["vel"] if vel is None else vel, dtype=float)
        ref = np.asarray(b["ref"], dtype=float)
        Hf = self.Hf
        y_sig = -self.cfg.depth + np.outer(Hf, self.sigc)          # (nx+1, ns)
        # xi = (heave, pitch): heave is vertical, so it only feeds the
        # horizontal faces; pitch feeds both through the rigid rotation.
        u_body = -vel[1] * (y_sig - ref[1])
        x_face = np.broadcast_to(self.x[:, None], (self.nx, self.ns + 1))
        w_body = vel[0] + vel[1] * (x_face - ref[0])
        A_block_x = self.bfrac_x * self.Ax[:, None]
        A_block_s = self.bfrac_s * self.dx
        return (A_block_x[1:, :] * u_body[1:, :] - A_block_x[:-1, :] * u_body[:-1, :]
                + A_block_s[:, 1:] * w_body[:, 1:] - A_block_s[:, :-1] * w_body[:, :-1])

    def _divergence(self, u, w):
        Aopen_x = self.ox * self.Ax[:, None]
        Aopen_s = self.os * self.dx
        return (Aopen_x[1:, :] * u[1:, :] - Aopen_x[:-1, :] * u[:-1, :]
                + Aopen_s[:, 1:] * w[:, 1:] - Aopen_s[:, :-1] * w[:, :-1])

    def _cg_solve(self, M, rhs, x0=None):
        if x0 is None:
            x0 = np.zeros_like(rhs)
        p, info = _solve_spd(M, np.asarray(rhs, dtype=float).ravel(),
                             np.asarray(x0, dtype=float).ravel(),
                             self.cfg.cg_rtol, self.cfg.cg_maxiter)
        self.cg_info = int(info)
        return p.reshape(self.nx, self.ns)

    def _correct(self, us, ws, p, dt):
        Aopen_x = self.ox * self.Ax[:, None]
        Aopen_s = self.os * self.dx
        coef_u = dt * Aopen_x / self.Wx[:, None]
        coef_s = dt * Aopen_s / self.Ws[:, None]
        u_new = us.copy()
        u_new[1:-1, :] -= coef_u[1:-1, :] * (p[1:, :] - p[:-1, :])
        w_new = ws.copy()
        if self.ns > 1:
            w_new[:, 1:-1] -= coef_s[:, 1:-1] * (p[:, 1:] - p[:, :-1])
        w_new[:, -1] = ws[:, -1] + 2.0 * coef_s[:, -1] * p[:, -1]   # p_out = 0, half cell
        w_new[:, 0] = 0.0
        flux_new = self._divergence(u_new, w_new)
        scale_field = self._divergence(us, ws)
        if self.body is not None:
            body_flux = self.body_normal_flux()
            flux_new = flux_new + body_flux
            scale_field = scale_field + body_flux
        if self.body is not None:
            # Direct forcing at the *fully blocked* faces: the fluid there must
            # move with the body (no penetration).  Without it those face values
            # are never touched by the projection, they are advected from the
            # solid interior, and they feed the momentum of the neighbouring
            # fluid cell -- measured: the face velocity next to the section's
            # waterline corner grew 0 -> 3.3 m/s in 4 s of a 1 cm forced heave
            # while the fluid field itself stayed bounded.
            vel = np.asarray(self.body["vel"], dtype=float)
            ref = np.asarray(self.body["ref"], dtype=float)
            y_sig = -self.cfg.depth + np.outer(self.Hf, self.sigc)
            x_face = np.broadcast_to(self.x[:, None], (self.nx, self.ns + 1))
            fill_u = np.zeros_like(u_new)
            fill_u[:, :] = (-vel[1] * (y_sig - ref[1]))
            fill_w = vel[0] + vel[1] * (x_face - ref[0])
            bu = (self.bfrac_x >= 1.0 - 1e-12)
            u_new[bu] = fill_u[bu]
            bw = (self.bfrac_s >= 1.0 - 1e-12)
            w_new[bw] = fill_w[bw]
        free = ~self.solid
        scale = max(float(np.abs(scale_field).max()), 1e-30)
        self.div_residual = (float(np.abs(flux_new[free]).max() / scale)
                             if free.any() else 0.0)
        self.p = p
        return u_new, w_new

    def _project(self, us, ws, dt):
        M, free = self._poisson_matrix()
        div = self._divergence(us, ws)
        if self.body is not None:
            div = div + self.body_normal_flux()
        rhs = -div.ravel() / dt
        rhs[~free] = 0.0
        p = self._cg_solve(M, rhs, self.p.ravel())
        return self._correct(us, ws, p, dt)

    def _step_free_body(self, us, ws, dt):
        """One time step with a *free* heave/pitch section (implicit coupling).

        The rigid-body acceleration is eliminated from the projection instead of
        being lagged: the body source is linear in the acceleration, so the
        pressure of this step is ``q = q0 + a3 q3 + a5 q5`` with the ``q_k``
        solved from the same Poisson matrix.  The body equation then closes on
        the same linear combination, giving the 2x2 system

            (M_body + A) a = F0 + F_ext,     A = -dF/d(a)   (discrete added mass)

        so the *whole* added mass the fluid can exert in this discretisation is
        inside the implicit mass matrix.  Nothing is left to the lagged coupling
        that made the explicit staggered version unstable at ``A33/m ~ 2.7``
        (previous node, honest FAIL in ``docs/wave-tank-td-seakeeping``).
        """
        b = self.body
        grav = self.cfg.gravity
        M, free = self._poisson_matrix()
        div_adv = self._divergence(us, ws)
        v_n = np.asarray(b["vel"], dtype=float)
        rhs0 = -(div_adv + self.body_normal_flux(v_n)).ravel() / dt
        rhs0[~free] = 0.0
        p0 = self._cg_solve(M, rhs0, self.p.ravel())
        grads = self.body_flux_gradient()
        pk = []
        for gk in grads:
            rhs = -gk.ravel()
            rhs[~free] = 0.0
            pk.append(self._cg_solve(M, rhs, p0.ravel()))
        rho = self.cfg.density
        quad = self.wetted_quadrature()
        fh, mh = self.hydrostatic_force(quad)
        Fq0, Mq0 = self.body_force_dynamic(p0, quad)
        # G[j][k] = dF_j/d(acceleration in DOF k) from the same contour
        # functional: the added mass the fluid can exert at this time step is
        # therefore *inside* the implicit mass matrix, not left to the lagged
        # coupling that destabilised the explicit version.
        G = np.zeros((2, 2))
        for k in range(2):
            fk, mk = self.body_force_dynamic(pk[k], quad)
            G[:, k] = (fk[1], mk)
        # weight and its moment about the reference point; the CG rides with the
        # body, so an off-centre CG really does couple heave and pitch
        th = float(b["xi"][1])
        dx_cg = b["cg"][0] - b["ref"][0]
        dz_cg = b["cg"][1] - b["ref"][1]
        c, sn = math.cos(th), math.sin(th)
        xg = b["ref"][0] + dx_cg * c - dz_cg * sn
        m_weight = -(xg - b["ref"][0]) * b["mass"] * grav
        damp = np.asarray(b.get("damping", np.zeros(2)), dtype=float)
        rhs = np.array([Fq0[1] + fh[1] - b["mass"] * grav - damp[0] * v_n[0],
                        Mq0 + mh + m_weight - damp[1] * v_n[1]])
        dcg = float(b["ref"][1] - b["cg"][1])
        m_body = np.array([[b["mass"], b["mass"] * dcg],
                           [b["mass"] * dcg, b["inertia"]]])
        acc = np.linalg.solve(m_body - G, rhs)
        vel = v_n + dt * acc
        p = p0 + acc[0] * pk[0] + acc[1] * pk[1]
        b["vel"] = vel
        b["acc"] = acc
        b["added_mass_discrete"] = (-G).tolist()
        u_new, w_new = self._correct(us, ws, p, dt)
        # the recorded load is the *fluid* load on the section (no weight), so a
        # wave-drift analysis of ``force[0]`` is not contaminated by gravity
        b["force"] = np.array([Fq0[0], Fq0[1]]) + fh + G @ acc
        b["moment"] = float(Mq0 + mh + float(G[1] @ acc))
        b["xi"] = b["xi"] + dt * vel
        b["xi"][1] = float(np.clip(b["xi"][1], -0.35, 0.35))
        return u_new, w_new

    # ------------------------------------------------------------------- step
    def step(self, dt: float, t=None):
        if t is None:
            t = self.t
        self._update_mask()
        us, ws = self._predict(dt)
        us, ws = self._apply_zones(us, ws, t, dt)
        mode = None if self.body is None else str(self.body.get("motion", "free"))
        if mode == "free":
            u_new, w_new = self._step_free_body(us, ws, dt)
        else:
            if self.body is not None:
                self._prescribe(dt)
            u_new, w_new = self._project(us, ws, dt)
            if self.body is not None:
                self.body["force"], self.body["moment"] = self.body_loads()
        self.u, self.w = u_new, w_new
        top_open = ~self.solid[:, -1]
        self.eta = self.eta + np.where(top_open, dt * w_new[:, -1], 0.0)
        # Absorbing beach: in an incompressible projection solver a velocity-only
        # sponge is undone by the pressure correction (mass is re-routed, the
        # beach only produces a set-up).  The surface itself must be relaxed, so
        # the beach acts as the open boundary it stands for.
        rate_c = self.sponge_rate(self.x)
        if rate_c.any():
            self.eta = self.eta * np.clip(1.0 - dt * rate_c, 0.0, 1.0)
        self.t = t + dt
        self._build_grid()
        self.volume_drift = self.volume() - self.volume0
        if self.body is not None:
            self._record_body(self.t)
        return self.t

    def total_pressure(self) -> np.ndarray:
        """Physical (kinematic) pressure ``g(eta - y) + q`` on cell centres."""
        g = self.cfg.gravity
        return g * (self.eta[:, None] - self.yc) + self.p

    # ----------------------------------------------------------- body forces
    def pressure_at(self, xq, yq, p=None):
        """Bilinear ``(x, sigma)`` interpolation of the projected pressure ``q``.

        The vertical coordinate is normalised by the local water column, so the
        interpolation follows the moving grid instead of smearing across it.
        Points outside the water column are clamped to the nearest interior
        sample; callers sample slightly outside the body for exactly that
        reason.
        """
        field = self.p if p is None else np.asarray(p, dtype=float)
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        yq = np.atleast_1d(np.asarray(yq, dtype=float))
        eta = self.eta_at(xq)
        height = self.cfg.depth + eta
        sig = np.clip((yq + self.cfg.depth) / np.maximum(height, 1e-12), 0.0, 1.0)
        # the top sigma node is the Dirichlet surface row: stay inside it
        sig = np.minimum(sig, 1.0 - 1e-6)
        fi = np.clip(np.searchsorted(self.x, xq) - 1, 0, self.nx - 2)
        tx = (xq - self.x[fi]) / self.dx
        fj = np.clip(sig * self.ns - 0.5, 0.0, self.ns - 1.0 - 1e-9)
        j0 = np.floor(fj).astype(int)
        tj = fj - j0
        p00 = field[fi, j0]
        p10 = field[fi + 1, j0]
        p01 = field[fi, j0 + 1]
        p11 = field[fi + 1, j0 + 1]
        return ((1 - tx) * (1 - tj) * p00 + tx * (1 - tj) * p10
                + (1 - tx) * tj * p01 + tx * tj * p11)

    def body_force_interpolated(self, sub: int = 4, offset: float | None = None):
        """Dynamic-pressure force and pitch moment on the section, smooth in ``xi``.

        The cut-cell face-fraction integral in :meth:`body_force` is exact in
        calm water but *discontinuous* in the body position: every time a face
        flips between fully solid and partly open the integrated force jumps by
        ``O(p dx)``.  In a time-domain run that is force noise, and feeding it
        back into the 2-DOF dynamics destabilises the body (the failure mode
        recorded in ``docs/wave-tank-td/README.md``: the heave force jumped
        ``412 -> 23018 N/m`` within one period at fixed geometry).

        Here the pressure is integrated on the *contour* instead: each wetted
        edge is split into ``sub`` segments and ``q`` is sampled a fraction of a
        cell outside the body with :meth:`pressure_at`.  The quadrature points
        move continuously with the body, so the force is continuous.  The
        hydrostatic part is added exactly from the displaced-area polygon
        (Archimedes), exactly as in :meth:`body_force`, so the calm-water limit
        is unchanged.
        """
        b = self.body
        if b is None:
            return 0.0, 0.0, 0.0
        g = self.cfg.gravity
        rho = self.cfg.density
        poly = self.body_polygon()
        ref = np.asarray(b["ref"], dtype=float)
        offset = 0.5 * self.dx if offset is None else float(offset)
        fx = fy = m5 = 0.0
        n = len(poly)
        for k in range(n):
            a = poly[k]
            c = poly[(k + 1) % n]
            d = c - a
            dl = float(math.hypot(d[0], d[1]))
            if dl < 1e-14:
                continue
            nx_, nz_ = d[1] / dl, -d[0] / dl          # outward normal (CCW polygon)
            for m in range(sub):
                t0 = (m + 0.5) / sub
                mid = a + t0 * d
                if mid[1] > 0.0:                     # dry part of the section
                    continue
                sx = mid[0] + offset * nx_
                sz = min(mid[1] + offset * nz_, -1e-6)
                q = float(self.pressure_at(sx, sz)[0])
                seg = dl / sub
                dfx = -q * nx_ * seg
                dfz = -q * nz_ * seg
                fx += dfx
                fy += dfz
                m5 += (mid[0] - ref[0]) * dfz - (mid[1] - ref[1]) * dfx
        wet = self.wetted_section()
        fx, fy, m5 = rho * fx, rho * fy, rho * m5
        if wet is not None and len(wet) >= 3:
            area, cx, _cz, _ = polygon_properties(wet, rho)
            fy += rho * g * area
            m5 += rho * g * area * (cx - ref[0])
        return fx, fy, m5

    def _cell_coverage(self):
        return 0.25 * ((1 - self.ox[:-1, :]) + (1 - self.ox[1:, :])
                       + (1 - self.os[:, :-1]) + (1 - self.os[:, 1:]))

    def body_force(self):
        """Force ``(Fx, Fy)`` [N/m] and pitch moment about ``ref`` [Nm/m].

        The pressure is split as ``p = g(eta - y) + q``.  Archimedes gives the
        hydrostatic part exactly from the *polygon* (``rho g nabla`` acting at the
        centre of buoyancy), so only the non-hydrostatic ``q`` is integrated over
        the stair-step wetted faces.  That keeps the buoyancy free of the O(dy)
        mask bias while the dynamic load keeps the cut-cell resolution.
        """
        b = self.body
        g, rho = self.cfg.gravity, self.cfg.density
        nx, ns = self.nx, self.ns
        zc = np.zeros((1, ns))
        zs = np.zeros((nx, 1))
        th_c = self._cell_coverage()                         # (nx, ns)
        p = self.p                                        # non-hydrostatic part
        ref = b["ref"]
        fx = fy = m5 = 0.0
        # --- x-faces: face i separates cell i-1 (left) and cell i (right)
        thx = 1.0 - self.ox                                  # (nx+1, ns)
        if thx.any():
            thl = np.vstack([zc, th_c])                      # cell i-1
            thr = np.vstack([th_c, zc])                      # cell i
            right = thr > thl                                # solid on the right
            pcell = np.where(right, np.vstack([zc, p]), np.vstack([p, zc]))
            ymid = -self.cfg.depth + np.outer(self.Hf, self.sigc)
            pface = pcell
            # only a face with exactly one fully solid neighbour is wetted
            use = (thr > 0.5) ^ (thl > 0.5)
            contrib = np.where(use, pface * thx * self.Ax[:, None], 0.0)
            sgn = np.where(right, 1.0, -1.0)
            fx += rho * float(np.sum(sgn * contrib))
            m5 += rho * float(np.sum(-(ymid - ref[1]) * sgn * contrib))
        # --- sigma-faces: face k separates cell k-1 (below) and cell k (above)
        thy = 1.0 - self.os                                  # (nx, ns+1)
        if thy.any():
            th_dn = np.hstack([zs, th_c])                    # cell k-1 (below)
            th_up = np.hstack([th_c, zs])                    # cell k   (above)
            up = th_up > th_dn                               # solid above -> fluid below
            pcell = np.where(up, np.hstack([zs, p]), np.hstack([p, zs]))
            pface = pcell
            use = (th_up > 0.5) ^ (th_dn > 0.5)
            contrib = np.where(use, pface * thy * self.dx, 0.0)
            sgn = np.where(up, 1.0, -1.0)
            fy += rho * float(np.sum(sgn * contrib))
            xmid = np.broadcast_to(self.x[:, None], (nx, ns + 1))
            m5 += rho * float(np.sum((xmid - ref[0]) * sgn * contrib))
        # --- exact hydrostatic part (Archimedes) from the section polygon
        wet = self.wetted_section()
        if wet is not None and len(wet) >= 3:
            area, cx, _cz, _ = polygon_properties(wet, rho)
            fy += rho * g * area
            m5 += rho * g * area * (cx - ref[0])
        return fx, fy, m5

    def wetted_section(self):
        """Polygon of the section below the still-water line ``y = 0``."""
        if self.body is None:
            return None
        poly = self.body_polygon()
        out = []
        n = len(poly)
        for i in range(n):
            a, bb = poly[i], poly[(i + 1) % n]
            ina, inb = a[1] <= 0.0, bb[1] <= 0.0
            if ina:
                out.append(a)
            if ina != inb:
                t = (0.0 - a[1]) / (bb[1] - a[1])
                out.append(np.array([a[0] + t * (bb[0] - a[0]), 0.0]))
        return np.array(out) if out else None

    def displaced_volume(self, sub: int = 6) -> float:
        """Sub-cell solid volume of the section per unit width [m^2].

        The step-time mask uses face fractions (cheap); for buoyancy bookkeeping
        the cell volume fraction is integrated properly, so a grid-aligned
        section reproduces ``length * draft`` to the sampling error.
        """
        if self.body is None:
            return 0.0
        poly = self.body_polygon()
        s = (np.arange(sub) + 0.5) / sub
        xs = self.xf[:-1, None] + s[None, :] * self.dx            # (nx, sub)
        frac = np.zeros(self.nx)
        for j in range(self.ns):
            ylo = self.yf[:, j][:, None]
            yhi = self.yf[:, j + 1][:, None]
            y = ylo + s[None, :] * (yhi - ylo)                    # (nx, sub)
            xx = np.broadcast_to(xs[:, :, None], (self.nx, sub, sub))
            yy = np.broadcast_to(y[:, None, :], (self.nx, sub, sub))
            ok = (self.x >= poly[:, 0].min() - self.dx) & (self.x <= poly[:, 0].max() + self.dx)
            if ok.any():
                idx = np.nonzero(ok)[0]
                cnt = points_in_polygon(xx[idx], yy[idx], poly).reshape(idx.size, sub, sub)
                frac[idx] += cnt.mean(axis=(1, 2))
        return float(np.sum(frac * self.V))

    def _prescribe(self, dt: float):
        """Prescribed kinematics for ``motion='fixed'`` / ``'forced'`` runs.

        A diffraction run holds the section still; a forced-oscillation run
        drives one DOF with ``xi = amp * cos(omega t + phase)``.  The load is
        measured after the projection (``step``), so the radiation force is the
        one of the field that the *prescribed* motion actually produced.
        """
        b = self.body
        mode = b.get("motion", "fixed")
        if mode == "fixed":
            b["vel"] = np.zeros(2)
            b["acc"] = np.zeros(2)
            return
        spec = b.get("forced") or {}
        dof = int(spec["dof"])
        amp = float(spec["amp"])
        omega = float(spec["omega"])
        phase = float(spec.get("phase", 0.0))
        t = self.t
        b["xi"][dof] = amp * math.cos(omega * t + phase)
        b["vel"][dof] = -amp * omega * math.sin(omega * t + phase)
        b["acc"][dof] = -amp * omega ** 2 * math.cos(omega * t + phase)

    def _body_dynamics(self, dt: float):
        """Kept for API compatibility: a free step is fully handled by ``step``."""
        if self.body is None or self.body.get("motion", "free") != "free":
            return
        self.body["force"], self.body["moment"] = self.body_loads()

    def _record_body(self, t):
        b = self.body
        ts = b["time_series"]
        ts["t"].append(float(t))
        ts["heave"].append(float(b["xi"][0]))
        ts["pitch"].append(float(b["xi"][1]))
        ts["fx"].append(float(b["force"][0]))
        ts["fy"].append(float(b["force"][1]))
        ts["m5"].append(float(b["moment"]))

    # --------------------------------------------------------------- probing
    def eta_at(self, xq) -> np.ndarray:
        return np.interp(np.asarray(xq, dtype=float), self.x, self.eta)

    def u_at(self, xq, yq):
        """Bilinear (column, sigma) interpolation of the horizontal velocity."""
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        yq = np.atleast_1d(np.asarray(yq, dtype=float))
        sig = np.clip((yq + self.cfg.depth) / (self.cfg.depth + self.eta_at(xq)), 0.0, 1.0)
        fi = np.clip(np.searchsorted(self.xf, xq) - 1, 0, self.nx - 1)
        fj = np.clip((sig * self.ns - 0.5), 0, self.ns - 1.001)
        j0 = np.floor(fj).astype(int)
        tj = fj - j0
        return (1 - tj) * self.u[fi + 1, j0] + tj * self.u[fi + 1, j0 + 1]

    def momentum_flux(self, i_face: int) -> float:
        """``int (rho u^2 + p) dy`` at x-face ``i_face`` -- control-surface check."""
        i = int(np.clip(i_face, 1, self.nx - 1))
        ptot = self.total_pressure()
        pface = 0.5 * (ptot[i - 1, :] + ptot[i, :])
        return float(self.cfg.density * self.Ax[i] * np.sum(self.u[i, :] ** 2 + pface))