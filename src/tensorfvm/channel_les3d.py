"""Three-dimensional wall-resolved LES of plane channel flow (MAC, spectral in x/z).

Why this module exists
----------------------
Every route the repository had for the anchor ``CASE-FVM-CHANNEL-KMM180`` was a
*closure measured against DNS statistics*: the 1-D equilibrium channel of
:mod:`tensorfvm.rans_validation.channel` imposes the total-stress identity, so it
can be graded but its momentum side can never be wrong.  ``channel_periodic`` is
two-dimensional (streamwise x wall-normal), ``periodic_mac`` is periodic in all
three directions, so neither can sustain turbulence.

This module supplies the missing route: a genuinely resolved, three-dimensional,
time-accreting simulation of the same physical case.

* ``x`` and ``z`` periodic (spectral -- FFT), ``y`` bounded by no-slip walls,
* staggered MAC placement, so the checkerboard pressure mode is absent by
  construction,
* constant streamwise body force ``G``; the wall shear and bulk velocity are
  *outputs*, not inputs (``tau_w = G*delta`` only in a statistically steady
  state, which is one of the checks),
* second-order central convection in flux (divergence) form, second-order
  viscous/SGS fluxes, and a pressure projection whose Poisson operator is built
  from the *same* divergence/gradient stencils as the momentum step -- so the
  velocity after the projection is divergence free to round-off,
* Smagorinsky sub-grid stress with van Driest wall damping.  It is the only
  modelling assumption and it is **not** tuned against the reference profile,
* plane/time statistics: ``u+``, the normal Reynolds stresses, ``-u'v'`` and the
  wall shear.

Discretisation, stated so the tests can check it
------------------------------------------------
Arrays are ``(ny, nx, nz)``.  Cell ``(i,j,k)`` spans ``[x_j, x_{j+1}] x
[y_i^f, y_{i+1}^f] x [z_k, z_{k+1}]`` with centre ``(x_j+dx/2, yc_i, z_k+dz/2)``.

* ``u[i,j,k]`` at ``(x_j, yc_i, zc_k)``, shape ``(ny, nx, nz)``;
* ``v[i,j,k]`` at ``(xc_j, yf_i, zc_k)`` with ``v = 0`` at ``i = 0, ny``,
  shape ``(ny+1, nx, nz)``;
* ``w[i,j,k]`` at ``(xc_j, yc_i, z_k)``, shape ``(ny, nx, nz)``;
* scalars at cell centres ``(xc_j, yc_i, zc_k)``.

The momentum-control-volume faces of a component are the positions where the
transverse components live, which is what makes the flux form conservative: the
``u`` control volume has cell centres as its x-faces (``(uc[j]^2 - uc[j-1]^2)/dx``
with ``uc`` the x-average of ``u``) and ``yf``/``z_k`` as its y/z-faces, while the
``v`` control volume has ``x_j``/``z_k`` as x/z-faces and cell centres as y-faces
(``(vc[i]^2 - vc[i-1]^2)/(yc[i]-yc[i-1])``).  Walls use the half-cell conductance
for the viscous flux, i.e. the wall-adjacent node sits ``dy[0]/2`` from the wall,
the standard LES wall treatment when ``y1+ ~ 1``.

The projection solves ``div(grad phi) = div(u*)/dt`` with ``dphi/dy = 0`` at the
walls: FFT in ``x`` and ``z``, one tridiagonal system in ``y`` per mode, all
pre-factored once.  The ``(0,0)`` mode is singular under pure Neumann conditions
and is pinned at ``phi[0] = 0``; only the gradient of ``phi`` enters the velocity,
so the pinning does not change the projected field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ['ChannelLESConfig', 'ChannelLES3D', 'van_driest_damping',
           'smagorinsky_viscosity']


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChannelLESConfig:
    """Physical and numerical parameters of the channel LES.

    Lengths are in units of the half channel height ``delta``; ``nu = 1/Re_tau``
    and ``G = 1`` are chosen so that ``u_tau = sqrt(G delta) = 1`` and
    ``tau_w = G delta = 1`` in a statistically steady state.
    """

    re_tau: float = 180.0
    nx: int = 64
    ny: int = 64
    nz: int = 32
    #: streamwise / spanwise box lengths in half heights
    lx: float = 2.0 * math.pi
    lz: float = math.pi
    #: wall clustering of ``y(xi) = 1 - tanh(gamma (1 - 2 xi))/tanh(gamma)``
    stretch: float = 1.9
    #: Smagorinsky constant and van Driest damping constant A+ (wall units)
    smagorinsky: float = 0.1
    van_driest_a_plus: float = 25.0
    #: time step in ``delta / u_tau`` units (explicit; wall-diffusion limited)
    dt: float = 2.5e-3

    def __post_init__(self):
        if self.nx < 8 or self.ny < 8 or self.nz < 8:
            raise ValueError('need at least 8 cells per direction')
        if not (self.re_tau > 0 and self.lx > 0 and self.lz > 0):
            raise ValueError('re_tau and box lengths must be positive')
        if not (0 < self.dt <= 0.05):
            raise ValueError('dt outside the stable band of the explicit scheme')
        if self.smagorinsky < 0:
            raise ValueError('smagorinsky constant must be non-negative')


# --------------------------------------------------------------------------- #
# Sub-grid model
# --------------------------------------------------------------------------- #
def van_driest_damping(y_plus, a_plus=25.0):
    """``1 - exp(-y+/A+)``: damps the SGS viscosity in the viscous sublayer."""
    y_plus = np.asarray(y_plus, dtype=float)
    if np.any(y_plus < 0):
        raise ValueError('y+ must be non-negative')
    return 1.0 - np.exp(-y_plus / a_plus)


def smagorinsky_viscosity(du_dy, du_dx, du_dz, dv_dy, dv_dx, dv_dz,
                          dw_dy, dw_dx, dw_dz, delta, y_plus, *, cs=0.1,
                          a_plus=25.0):
    """Smagorinsky eddy viscosity ``(Cs Delta)^2 |S|`` with van Driest damping.

    ``|S| = sqrt(2 S_ij S_ij)`` with the symmetric resolved strain tensor; the
    filter width is ``Delta = (dx dy dz)^(1/3)``.
    """
    s11, s22, s33 = du_dx, dv_dy, dw_dz
    s12 = 0.5 * (du_dy + dv_dx)
    s13 = 0.5 * (du_dz + dw_dx)
    s23 = 0.5 * (dv_dz + dw_dy)
    magnitude = np.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + s33 ** 2
                               + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)))
    return (cs * delta) ** 2 * magnitude * van_driest_damping(y_plus, a_plus)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class ChannelLES3D:
    """Staggered MAC channel LES; conventions are in the module docstring."""

    def __init__(self, config: ChannelLESConfig | None = None, seed: int = 0):
        self.config = config or ChannelLESConfig()
        cfg = self.config
        self.nu = 1.0 / cfg.re_tau
        #: body force per unit mass; ``u_tau = sqrt(G delta) = 1``
        self.body_force = 1.0

        self.yf = self._wall_normal_faces()
        self.yc = 0.5 * (self.yf[:-1] + self.yf[1:])
        self.dy = np.diff(self.yf)
        self.dy_cc = self.yc[1:] - self.yc[:-1]
        self.dx = cfg.lx / cfg.nx
        self.dz = cfg.lz / cfg.nz
        self.delta = (self.dx * self.dy * self.dz) ** (1.0 / 3.0)
        self.y_plus = self.yc * cfg.re_tau
        self.rng = np.random.default_rng(seed)

        shape_c = (cfg.ny, cfg.nx, cfg.nz)
        self.u = np.zeros(shape_c)
        self.v = np.zeros((cfg.ny + 1, cfg.nx, cfg.nz))
        self.w = np.zeros(shape_c)

        kx = 2.0 * np.pi * np.fft.fftfreq(cfg.nx, d=self.dx)
        kz = 2.0 * np.pi * np.fft.fftfreq(cfg.nz, d=self.dz)
        self._lambda = (2.0 * (np.cos(kx * self.dx) - 1.0) / self.dx ** 2)[None, :, None] \
            + (2.0 * (np.cos(kz * self.dz) - 1.0) / self.dz ** 2)[None, None, :]
        self._factor_poisson()

        self.steps = 0
        self.time = 0.0
        self._samples = 0
        self._accumulators = {}
        self.history = []

    # -- grid ---------------------------------------------------------- #
    def _wall_normal_faces(self):
        """Faces on ``[0, 2]`` clustered at both walls (tanh mapping)."""
        cfg = self.config
        xi = np.linspace(0.0, 1.0, cfg.ny + 1)
        if cfg.stretch <= 0:
            return 2.0 * xi
        return 1.0 - np.tanh(cfg.stretch * (1.0 - 2.0 * xi)) / np.tanh(cfg.stretch)

    # -- interpolation helpers ----------------------------------------- #
    @staticmethod
    def _center_x(a):
        """x-face array -> cell centre: ``(a[j] + a[j+1])/2``."""
        return 0.5 * (a + np.roll(a, -1, axis=1))

    @staticmethod
    def _face_x(a):
        """cell-centre array -> x-face: ``(a[j-1] + a[j])/2``."""
        return 0.5 * (a + np.roll(a, 1, axis=1))

    @staticmethod
    def _center_z(a):
        return 0.5 * (a + np.roll(a, -1, axis=2))

    @staticmethod
    def _face_z(a):
        return 0.5 * (a + np.roll(a, 1, axis=2))

    def _face_y(self, a):
        """cell-centre array ``(ny,...)`` -> y-faces ``(ny+1,...)`` (zero ends)."""
        out = np.zeros((self.config.ny + 1,) + a.shape[1:])
        out[1:-1] = 0.5 * (a[:-1] + a[1:])
        return out

    @staticmethod
    def _center_y(a):
        """y-face array ``(ny+1,...)`` -> cell centres ``(ny,...)``."""
        return 0.5 * (a[1:] + a[:-1])

    # -- discrete operators -------------------------------------------- #
    def divergence(self, u=None, v=None, w=None):
        """Discrete divergence at cell centres."""
        u = self.u if u is None else u
        v = self.v if v is None else v
        w = self.w if w is None else w
        div = (np.roll(u, -1, axis=1) - u) / self.dx
        div += (v[1:] - v[:-1]) / self.dy[:, None, None]
        div += (np.roll(w, -1, axis=2) - w) / self.dz
        return div

    def gradient(self, phi):
        """Gradient of a cell-centred scalar on the three staggered positions."""
        gu = (phi - np.roll(phi, 1, axis=1)) / self.dx
        gv = np.zeros((self.config.ny + 1,) + phi.shape[1:])
        gv[1:-1] = (phi[1:] - phi[:-1]) / self.dy_cc[:, None, None]
        gw = (phi - np.roll(phi, 1, axis=2)) / self.dz
        return gu, gv, gw

    def laplacian(self, phi):
        """``div(grad(phi))`` with exactly the stencils of :meth:`divergence`."""
        gu, gv, gw = self.gradient(phi)
        return self.divergence(gu, gv, gw)

    # -- convection ----------------------------------------------------- #
    def convection(self):
        """Second-order central convection in flux (divergence) form."""
        u, v, w = self.u, self.v, self.w
        uc = self._center_x(u)
        wc = self._center_z(w)
        vc = self._center_y(v)

        # u-momentum: x-faces at cell centres, y-faces at yf, z-faces at z_k
        flux = uc ** 2
        conv_u = (flux - np.roll(flux, 1, axis=1)) / self.dx
        flux = v * self._face_y(u)
        conv_u += (flux[1:] - flux[:-1]) / self.dy[:, None, None]
        flux = w * self._face_z(u)
        conv_u += (flux - np.roll(flux, 1, axis=2)) / self.dz

        # v-momentum: x-faces at x_j, y-faces at cell centres, z-faces at z_k
        vv = v[1:-1]
        flux = self._face_y(u)[1:-1] * self._face_x(vv)
        conv_v = (np.roll(flux, -1, axis=1) - flux) / self.dx
        flux = vc ** 2
        conv_v += (flux[1:] - flux[:-1]) / self.dy_cc[:, None, None]
        flux = self._face_y(w)[1:-1] * self._face_z(vv)
        conv_v += (np.roll(flux, -1, axis=2) - flux) / self.dz

        # w-momentum: x-faces at x_j, y-faces at yf, z-faces at cell centres
        flux = wc ** 2
        conv_w = (flux - np.roll(flux, 1, axis=2)) / self.dz
        flux = v * self._face_y(w)
        conv_w += (flux[1:] - flux[:-1]) / self.dy[:, None, None]
        flux = u * self._face_x(w)
        conv_w += (np.roll(flux, -1, axis=1) - flux) / self.dx
        return conv_u, conv_v, conv_w

    # -- sub-grid and diffusion ----------------------------------------- #
    def velocity_gradients(self):
        """Cell-centred velocity and its gradient tensor (input of the SGS model)."""
        def dy_center(a):
            out = np.empty_like(a)
            out[1:-1] = (a[2:] - a[:-2]) / self.dy_cc[:, None, None]
            out[0] = (a[1] - a[0]) / self.dy_cc[0]
            out[-1] = (a[-1] - a[-2]) / self.dy_cc[-1]
            return out

        return dict(
            du_dx=(np.roll(self.u, -1, axis=1) - self.u) / self.dx,
            du_dy=dy_center(self.u),
            du_dz=(np.roll(self.u, -1, axis=2) - self.u) / self.dz,
            dv_dx=(self.v[1:-1] - np.roll(self.v[1:-1], 1, axis=1)) / self.dx,
            dv_dy=dy_center(self.v),
            dv_dz=(self.v[1:-1] - np.roll(self.v[1:-1], 1, axis=2)) / self.dz,
            dw_dx=(self.w - np.roll(self.w, 1, axis=1)) / self.dx,
            dw_dy=dy_center(self.w),
            dw_dz=(np.roll(self.w, -1, axis=2) - self.w) / self.dz,
            uc=self._center_x(self.u), vc=self._center_y(self.v),
            wc=self._center_z(self.w))

    def eddy_viscosity(self):
        """Cell-centred SGS viscosity (Smagorinsky + van Driest)."""
        if self.config.smagorinsky == 0.0:
            return np.zeros((self.config.ny, self.config.nx, self.config.nz))
        g = self.velocity_gradients()
        return smagorinsky_viscosity(
            g['du_dy'], g['du_dx'], g['du_dz'],
            g['dv_dy'], g['dv_dx'], g['dv_dz'],
            g['dw_dy'], g['dw_dx'], g['dw_dz'],
            self.delta, self.y_plus,
            cs=self.config.smagorinsky, a_plus=self.config.van_driest_a_plus)

    def diffusion(self, nut=None):
        """Viscous + SGS diffusion of all three momentum components."""
        nu_eff = self.nu + (self.eddy_viscosity() if nut is None else nut)
        ny, nx, nz = self.config.ny, self.config.nx, self.config.nz
        dy = self.dy[:, None, None]

        def wall_normal(component):
            """y-diffusion with a half-cell conductance at both walls."""
            flux = np.zeros((ny + 1, nx, nz))
            face = 0.5 * (nu_eff[1:] + nu_eff[:-1])
            flux[1:-1] = face * (component[1:] - component[:-1]) / self.dy[:-1, None, None]
            flux[0] = 2.0 * nu_eff[0] * component[0] / self.dy[0]
            flux[-1] = -2.0 * nu_eff[-1] * component[-1] / self.dy[-1]
            return (flux[1:] - flux[:-1]) / dy

        nu_x = 0.5 * (nu_eff + np.roll(nu_eff, 1, axis=1))
        nu_z = 0.5 * (nu_eff + np.roll(nu_eff, 1, axis=2))

        # u: x- and z-faces at cell centres
        lap_u = wall_normal(self.u)
        q = nu_x * (self.u - np.roll(self.u, 1, axis=1)) / self.dx
        lap_u += (np.roll(q, -1, axis=1) - q) / self.dx
        q = nu_z * (self.u - np.roll(self.u, 1, axis=2)) / self.dz
        lap_u += (np.roll(q, -1, axis=2) - q) / self.dz

        # v: y-faces at cell centres, x- and z-faces at x_j / z_k
        v = self.v[1:-1]
        face = 0.5 * (nu_eff[1:] + nu_eff[:-1])
        q = face * (self.v[2:] - v) / self.dy_cc[:, None, None]
        qm = face * (v - self.v[:-2]) / self.dy_cc[:, None, None]
        lap_v = (q - qm) / dy
        q = nu_x * (v - np.roll(v, 1, axis=1)) / self.dx
        lap_v += (np.roll(q, -1, axis=1) - q) / self.dx
        q = nu_z * (v - np.roll(v, 1, axis=2)) / self.dz
        lap_v += (np.roll(q, -1, axis=2) - q) / self.dz

        # w: x-faces at x_j, z-faces at cell centres
        lap_w = wall_normal(self.w)
        q = nu_x * (self.w - np.roll(self.w, 1, axis=1)) / self.dx
        lap_w += (np.roll(q, -1, axis=1) - q) / self.dx
        q = nu_z * (self.w - np.roll(self.w, 1, axis=2)) / self.dz
        lap_w += (np.roll(q, -1, axis=2) - q) / self.dz
        return lap_u, lap_v, lap_w

    # -- pressure projection -------------------------------------------- #
    def _factor_poisson(self):
        """Pre-factor the y-operator by a symmetric (weighted) eigendecomposition.

        The finite-volume operator ``T`` is self-adjoint with respect to the
        cell volume ``dy``, so ``M = W T W^-1`` with ``W = diag(sqrt(dy))`` is
        symmetric.  Its eigendecomposition gives, for every Fourier mode
        ``(kx, kz)``, the exact inverse of ``T + lambda`` -- no Thomas sweep, no
        pivoting, and the singular ``(0,0)`` mode is handled by dropping the
        constant eigenvector (pseudo-inverse), which leaves the *gradient* of
        the pressure -- the only quantity that enters the velocity -- exact.
        """
        ny = self.config.ny
        dy = self.dy
        dc = self.dy_cc
        operator = np.zeros((ny, ny))
        operator[0, 0] = -1.0 / (dc[0] * dy[0])
        operator[0, 1] = 1.0 / (dc[0] * dy[0])
        for i in range(1, ny - 1):
            operator[i, i + 1] = 1.0 / (dc[i] * dy[i])
            operator[i, i - 1] = 1.0 / (dc[i - 1] * dy[i])
            operator[i, i] = -operator[i, i + 1] - operator[i, i - 1]
        operator[-1, -1] = -1.0 / (dc[-1] * dy[-1])
        operator[-1, -2] = 1.0 / (dc[-1] * dy[-1])

        weight = np.sqrt(dy)
        symmetric = weight[:, None] * operator / weight[None, :]
        assert np.allclose(symmetric, symmetric.T, atol=1e-9 * np.abs(symmetric).max())
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)

        self._po_weight = weight
        self._po_eigenvalues = eigenvalues
        self._po_eigenvectors = eigenvectors
        self._po_lambda_flat = np.asarray(self._lambda).reshape(-1)

    def solve_poisson(self, rhs):
        """Solve ``div(grad phi) = rhs`` (Neumann walls) by FFT + eigendecomposition."""
        ny = self.config.ny
        rhs_hat = np.fft.fft2(rhs, axes=(1, 2)).reshape(ny, -1)
        lam = self._po_lambda_flat[None, :]
        denominators = self._po_eigenvalues[:, None] + lam
        # the (0,0) Fourier mode meets the constant eigenvector: pseudo-inverse
        bad = np.abs(denominators) < 1e-12 * np.abs(denominators).max()
        safe = np.where(bad, 1.0, denominators)
        weighted = self._po_weight[:, None] * rhs_hat
        coefficients = self._po_eigenvectors.T @ weighted
        coefficients = np.where(bad, 0.0, coefficients / safe)
        solution = self._po_eigenvectors @ coefficients
        solution /= self._po_weight[:, None]
        solution = solution.reshape(ny, self.config.nx, self.config.nz)
        return np.real(np.fft.ifft2(solution, axes=(1, 2)))

    def project(self, dt=None):
        """Project the velocity onto the discretely divergence-free space."""
        dt = self.config.dt if dt is None else dt
        phi = self.solve_poisson(self.divergence() / dt)
        gu, gv, gw = self.gradient(phi)
        self.u -= dt * gu
        self.v[1:-1] -= dt * gv[1:-1]
        self.w -= dt * gw
        self.v[0] = 0.0
        self.v[-1] = 0.0
        return phi

    # -- time stepping --------------------------------------------------- #
    def momentum_rhs(self):
        """Convection + diffusion + body force, per unit mass."""
        conv_u, conv_v, conv_w = self.convection()
        lap_u, lap_v, lap_w = self.diffusion()
        return (-conv_u + lap_u + self.body_force,
                -conv_v + lap_v,
                -conv_w + lap_w)

    def step(self):
        """One explicit Euler step followed by an exact projection."""
        dt = self.config.dt
        rhs_u, rhs_v, rhs_w = self.momentum_rhs()
        self.u += dt * rhs_u
        self.v[1:-1] += dt * rhs_v
        self.w += dt * rhs_w
        self.project(dt)
        self.steps += 1
        self.time += dt
        if not np.isfinite(self.u).all():
            raise FloatingPointError('non-finite velocity at step %d' % self.steps)

    # -- initial condition and statistics -------------------------------- #
    def initialise(self, noise=0.1):
        """Parabolic laminar profile plus random noise, projected once."""
        cfg = self.config
        y = self.yc[:, None, None]
        profile = self.body_force * y * (2.0 - y) / (2.0 * self.nu)
        self.u = profile * np.ones((cfg.ny, cfg.nx, cfg.nz))
        self.v = np.zeros((cfg.ny + 1, cfg.nx, cfg.nz))
        self.w = np.zeros((cfg.ny, cfg.nx, cfg.nz))
        if noise:
            amplitude = noise * float(profile.max())
            self.u += amplitude * (self.rng.random(self.u.shape) - 0.5)
            self.v[1:-1] += amplitude * (self.rng.random(self.v[1:-1].shape) - 0.5)
            self.w += amplitude * (self.rng.random(self.w.shape) - 0.5)
            self.v[0] = 0.0
            self.v[-1] = 0.0
            self.project()
        self.time = 0.0
        self.steps = 0
        self._samples = 0
        self._accumulators = {}
        return self

    def sample(self):
        """Accumulate plane averages (over x and z) of the flow statistics."""
        u = self.u
        v = self._center_y(self.v)
        w = self._center_z(self.w)
        fields = dict(
            u=u.mean(axis=(1, 2)),
            uu=(u * u).mean(axis=(1, 2)),
            vv=(v * v).mean(axis=(1, 2)),
            ww=(w * w).mean(axis=(1, 2)),
            uv=(u * v).mean(axis=(1, 2)),
            tau_wall=float(self.nu * u[0].mean() / (0.5 * self.dy[0])),
        )
        if not self._accumulators:
            self._accumulators = {k: np.zeros_like(val) for k, val in fields.items()}
        for key, value in fields.items():
            self._accumulators[key] += value
        self._samples += 1
        return fields

    def statistics(self):
        """Time-averaged statistics in wall units."""
        if self._samples == 0:
            raise RuntimeError('no samples accumulated')
        mean = {k: val / self._samples for k, val in self._accumulators.items()}
        u_tau = math.sqrt(self.body_force * 1.0)     # tau_w = G*delta in equilibrium
        u_plus = mean['u'] / u_tau
        y = np.concatenate(([0.0], self.yc, [2.0]))
        u_padded = np.concatenate(([0.0], u_plus, [0.0]))
        return dict(
            y_plus=self.yc * self.config.re_tau,
            y_over_delta=self.yc,
            u_plus=u_plus,
            u_rms_plus=np.sqrt(np.maximum(mean['uu'] - mean['u'] ** 2, 0.0)) / u_tau,
            v_rms_plus=np.sqrt(np.maximum(mean['vv'], 0.0)) / u_tau,
            w_rms_plus=np.sqrt(np.maximum(mean['ww'], 0.0)) / u_tau,
            uv_plus=mean['uv'] / u_tau ** 2,
            tau_wall_plus=float(mean['tau_wall']),
            u_tau=u_tau,
            u_bulk_plus=float(np.trapezoid(u_padded, y) / (y[-1] - y[0])),
            samples=self._samples,
            time_averaged=self.time,
            steps=self.steps,
        )

    def bulk_velocity(self):
        """Instantaneous plane-averaged bulk velocity (half-height units)."""
        y = np.concatenate(([0.0], self.yc, [2.0]))
        u = np.concatenate(([0.0], self.u.mean(axis=(1, 2)), [0.0]))
        return float(np.trapezoid(u, y) / (y[-1] - y[0]))

    def run(self, steps, sample_every=10, check_every=2000, verbose=False):
        """Advance ``steps`` steps, sampling the statistics every ``sample_every``."""
        for _ in range(steps):
            self.step()
            if sample_every and (self.steps % sample_every == 0):
                self.sample()
            if check_every and (self.steps % check_every == 0):
                record = dict(steps=self.steps, time=self.time,
                              u_bulk=self.bulk_velocity(),
                              tau_wall=float(self.nu * self.u[0].mean() / (0.5 * self.dy[0])))
                self.history.append(record)
                if verbose:
                    print('step %6d  t=%7.3f  u_b=%8.5f  tau_w=%8.5f'
                          % (record['steps'], record['time'], record['u_bulk'],
                             record['tau_wall']))
        return self