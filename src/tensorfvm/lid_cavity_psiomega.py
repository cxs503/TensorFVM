"""Independent streamfunction-vorticity solver for the lid-driven cavity.

Why this module exists
----------------------
``CASE-FVM-CAVITY-RE1000`` is accepted on the production solver
(:mod:`tensorfvm.lid_cavity`): staggered MAC, primitive variables, van Leer
limited face reconstruction, pressure projection with a Neumann Poisson solve in
the DCT-II basis.  Its grid-convergence study
(:mod:`tensorfvm.cavity_uncertainty`) leaves one question open: the
Richardson-extrapolated limit sits ~1.8 % away from Ghia's tabulated values, and
a single code cannot say whether that offset is an artefact of extrapolating its
own truncation error or the discretisation error of the 129x129 reference
itself.

This module provides a *second, independent* numerical solution of the same PDE
so that the question becomes answerable by agreement/disagreement rather than by
reading one code's error model:

* primitive variables are eliminated -- ``u = dpsi/dy``, ``v = -dpsi/dx``;
* the vorticity transport equation is marched in **flux form** with QUICK face
  values (a scheme family different from the production van Leer harmonic
  limiter);
* the wall vorticity comes from the second-order Thom balance instead of a
  mirror ghost velocity;
* the Poisson equation for ``psi`` (Dirichlet ``psi = 0`` on every wall) is
  solved *directly* in the DST-I eigenbasis, in contrast to the production
  Neumann/DCT pressure solve.

What is deliberately shared with the production path is the *measurement*
(:func:`tensorfvm.cavity_uncertainty.functionals_from_fields`): a cross-code
comparison is only meaningful if both codes are scored by the same operator.

Grid layout
-----------
Nodes at ``x_i = i h``, ``y_j = j h`` with ``h = 1/n`` for ``i, j = 0..n``, so
the walls are nodes.  ``psi = 0`` on the walls, and the face velocities
``u = (psi[i,j+1] - psi[i,j]) / h``, ``v = -(psi[i+1,j] - psi[i,j]) / h`` are
discretely divergence-free to machine precision by construction -- a property
the projection solver only reaches to the tolerance of its pressure solve.

Boundary condition for the vorticity
------------------------------------
Expanding ``psi`` inwards from a wall along the inward normal ``s`` with
``psi_wall = 0`` and the flat-wall identity ``d2psi/dt2 = 0``::

    psi(h) = h psi_s + h^2/2 psi_ss   =>   psi_ss = 2 (psi_adj - h psi_s) / h^2
    omega_wall = -psi_ss = -2 psi_adj / h^2 + 2 psi_s / h

with ``psi_s = dpsi/ds`` at the wall (``0`` everywhere except the lid, where
``s = 1-y`` and ``psi_s = -U``).  This is the classical second-order treatment
(Thom 1933 / Jensen 1959 family) and it is what the wall-flux divergence below
needs: the boundary node carries a genuine vorticity value, so the convective
flux into the first interior node is the physical one.

References
----------
* U. Ghia, K. N. Ghia, C. T. Shin, *J. Comput. Phys.* 48 (1982) 387-411.
* B. P. Leonard, *Comput. Methods Appl. Mech. Engrg.* 19 (1979) 59-98 (QUICK).
* A. Thom, *Proc. R. Soc. A* 141 (1933) 651-669 (wall vorticity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["PsiOmegaConfig", "PsiOmegaSolver", "poisson_dirichlet"]


def _dst_matrix(m: int) -> np.ndarray:
    """Orthonormal DST-I matrix ``S[k, j] = sqrt(2/(m+1)) sin(pi (k+1)(j+1)/(m+1))``."""
    k = np.arange(m)[:, None]
    j = np.arange(m)[None, :]
    return np.sqrt(2.0 / (m + 1)) * np.sin(math.pi * (k + 1) * (j + 1) / (m + 1))


def poisson_dirichlet(rhs: np.ndarray, h: float) -> np.ndarray:
    """Solve ``lap(psi) = rhs`` on an interior block with ``psi = 0`` around it.

    ``rhs`` is the interior block (the walls are excluded).  The Laplacian is the
    standard five-point stencil, so the discrete operator is diagonalised exactly
    by the DST-I basis; the solve is direct, not iterative.  A fast transform is
    used when scipy is importable and an explicit matrix product otherwise, so
    the routine has no hard dependency beyond numpy.
    """
    rhs = np.asarray(rhs, dtype=float)
    m, k = rhs.shape
    if m != k:
        raise ValueError(f"poisson_dirichlet expects a square interior block, got {rhs.shape}")
    lam = (2.0 * np.cos(math.pi * np.arange(1, m + 1) / (m + 1)) - 2.0) / h ** 2
    denom = lam[:, None] + lam[None, :]
    try:  # pragma: no cover - the fast path is the one that runs in practice
        from scipy.fft import dstn, idstn

        coeff = dstn(rhs, type=1, norm="ortho")
        return idstn(coeff / denom, type=1, norm="ortho")
    except Exception:  # pragma: no cover - numpy-only fallback
        s = _dst_matrix(m)
        return s.T @ ((s @ rhs @ s.T) / denom) @ s


@dataclass
class PsiOmegaConfig:
    """Resolution and marching controls of the independent solver."""

    n: int = 64
    reynolds: float = 1000.0
    lid_velocity: float = 1.0
    cfl: float = 0.4
    max_steps: int = 400_000
    tolerance: float = 1e-7
    check_every: int = 250
    time_step: float | None = None

    def __post_init__(self) -> None:
        if self.n < 8:
            raise ValueError("n must be at least 8")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive")


@dataclass
class PsiOmegaSolver:
    """Node-centred streamfunction-vorticity solver (see the module docstring)."""

    config: PsiOmegaConfig = field(default_factory=PsiOmegaConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        self.n = int(cfg.n)
        self.h = 1.0 / self.n
        nodes = self.n + 1
        self.psi = np.zeros((nodes, nodes), dtype=float)
        self.omega = np.zeros((nodes, nodes), dtype=float)
        self.steps = 0
        self.residual = float("inf")
        self.dt_history: list[float] = []

    # ------------------------------------------------------------------ helpers
    @property
    def lid_velocity(self) -> float:
        return float(self.config.lid_velocity)

    @property
    def nu(self) -> float:
        return float(self.config.lid_velocity) / float(self.config.reynolds)

    def node_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """``(u, v)`` at the nodes; ``u = dpsi/dy``, ``v = -dpsi/dx`` (central)."""
        psi, h, n = self.psi, self.h, self.n
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        u[:, 1:n] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * h)
        v[1:n, :] = -(psi[2:, :] - psi[:-2, :]) / (2.0 * h)
        # no-slip / moving-wall values on the boundary rows and columns
        u[:, 0] = 0.0
        u[:, n] = self.lid_velocity
        v[0, :] = 0.0
        v[n, :] = 0.0
        return u, v

    def face_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Staggered face velocities ``(u, v)`` with shapes ``(n+1, n)`` / ``(n, n+1)``.

        These are exact differences of the streamfunction, hence exactly
        divergence-free, and they are laid out like the production solver's
        fields so the same measurement operator can be applied to both codes.
        """
        psi, h = self.psi, self.h
        u = (psi[:, 1:] - psi[:, :-1]) / h          # (n+1, n)   on x = i h
        v = -(psi[1:, :] - psi[:-1, :]) / h         # (n, n+1)   on y = j h
        return u, v

    def divergence(self) -> np.ndarray:
        """Discrete divergence of the staggered face velocities (interior cells)."""
        u, v = self.face_velocity()
        return (u[1:, :] - u[:-1, :]) / self.h + (v[:, 1:] - v[:, :-1]) / self.h

    # --------------------------------------------------------- elliptic solve
    def wall_vorticity(self) -> np.ndarray:
        """Vorticity on the wall nodes from the second-order Thom balance."""
        psi, h, n, u_lid = self.psi, self.h, self.n, self.lid_velocity
        omega = self.omega.copy()
        # axis 0 is x, axis 1 is y: the lid is the row j = n, the right wall i = n
        omega[0, :] = -2.0 * psi[1, :] / h ** 2                        # left  (i=0): psi_s = 0
        omega[n, :] = -2.0 * psi[n - 1, :] / h ** 2                    # right (i=n): psi_s = 0
        omega[:, 0] = -2.0 * psi[:, 1] / h ** 2                        # bottom (j=0): psi_s = 0
        omega[:, n] = -2.0 * psi[:, n - 1] / h ** 2 - 2.0 * u_lid / h  # lid (j=n): psi_s = -U
        # corners are the lid singularities; the two one-sided formulae disagree
        # by O(1/h^2) there, so average instead of picking one arbitrarily.
        for (i, j) in ((0, 0), (0, n), (n, 0), (n, n)):
            vals = [omega[i, j]]
            if i == 0:
                vals.append(-2.0 * psi[1, j] / h ** 2)
            if i == n:
                vals.append(-2.0 * psi[n - 1, j] / h ** 2)
            if j == 0:
                vals.append(-2.0 * psi[i, 1] / h ** 2)
            if j == n:
                vals.append(-2.0 * psi[i, n - 1] / h ** 2 - 2.0 * u_lid / h)
            omega[i, j] = 0.5 * (vals[0] + vals[-1])
        return omega

    def solve_streamfunction(self) -> np.ndarray:
        """Solve ``lap psi = -omega`` (Dirichlet) and refresh ``self.psi``."""
        interior = poisson_dirichlet(-self.omega[1:-1, 1:-1], self.h)
        self.psi = np.zeros_like(self.psi)
        self.psi[1:-1, 1:-1] = interior
        return self.psi

    def vorticity_from_streamfunction(self) -> np.ndarray:
        """``-lap psi`` at the interior nodes (diagnostic; the transport marches ``omega``)."""
        psi, h = self.psi, self.h
        lap = (psi[2:, 1:-1] + psi[:-2, 1:-1] + psi[1:-1, 2:] + psi[1:-1, :-2]
               - 4.0 * psi[1:-1, 1:-1]) / h ** 2
        return -lap

    # ------------------------------------------------------------- transport
    @staticmethod
    def _quick_face(w: np.ndarray, vel_face: np.ndarray, axis: int) -> np.ndarray:
        """QUICK face values of ``w``; face ``k`` sits between nodes ``k`` and ``k+1``."""
        m = w.shape[axis]
        pad = ((1, 1), (0, 0)) if axis == 0 else ((0, 0), (1, 1))
        wp = np.pad(w, pad, mode="edge")
        sl = (lambda lo, hi: (slice(lo, hi), slice(None))) if axis == 0 \
            else (lambda lo, hi: (slice(None), slice(lo, hi)))
        a, b = wp[sl(1, m)], wp[sl(2, m + 1)]      # w[k], w[k+1]
        c, d = wp[sl(0, m - 1)], wp[sl(3, m + 2)]  # w[k-1], w[k+2]
        up = vel_face > 0.0
        pos = 0.375 * (2.0 * a + b) - 0.125 * c          # (6a + 3b - c)/8
        neg = 0.375 * (2.0 * b + a) - 0.125 * d          # (6b + 3a - d)/8
        return np.where(up, pos, neg)

    def tendency(self) -> np.ndarray:
        """``domega/dt`` at the interior nodes (flux form + central diffusion)."""
        w, h, nu, n = self.omega, self.h, self.nu, self.n
        u_node, v_node = self.node_velocity()
        u_face = 0.5 * (u_node[:-1, :] + u_node[1:, :])   # (n, n+1)  x-faces
        v_face = 0.5 * (v_node[:, :-1] + v_node[:, 1:])   # (n+1, n)  y-faces

        w_x = self._quick_face(w, u_face, axis=0)         # (n, n+1)
        w_y = self._quick_face(w, v_face, axis=1)         # (n+1, n)
        flux_x = u_face * w_x
        flux_y = v_face * w_y

        # interior node i sees the faces (i-1) and i of both families
        div = ((flux_x[1:, 1:-1] - flux_x[:-1, 1:-1]) / h
               + (flux_y[1:-1, 1:] - flux_y[1:-1, :-1]) / h)
        lap = (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2]
               - 4.0 * w[1:-1, 1:-1]) / h ** 2
        return -div + nu * lap

    def stable_time_step(self) -> float:
        cfg = self.config
        if cfg.time_step is not None:
            return float(cfg.time_step)
        u_node, v_node = self.node_velocity()
        speed = max(float(np.abs(u_node).max()), float(np.abs(v_node).max()),
                    self.lid_velocity)
        conv = cfg.cfl * self.h / speed
        visc = 0.2 * self.h ** 2 / self.nu
        return min(conv, visc)

    def step(self, dt: float | None = None) -> float:
        """One explicit Euler update of the interior vorticity, then a Poisson solve."""
        dt = self.stable_time_step() if dt is None else float(dt)
        self.omega[1:-1, 1:-1] += dt * self.tendency()
        self.omega = self.wall_vorticity()
        self.solve_streamfunction()
        self.steps += 1
        self.dt_history.append(dt)
        return dt

    def run(self, *, max_steps: int | None = None, tolerance: float | None = None,
            check_every: int | None = None, on_check=None) -> dict[str, object]:
        """March to steady state; the residual is ``max|domega/dt|`` over the interior."""
        cfg = self.config
        max_steps = cfg.max_steps if max_steps is None else int(max_steps)
        tolerance = cfg.tolerance if tolerance is None else float(tolerance)
        check_every = cfg.check_every if check_every is None else int(check_every)

        dt = 0.0
        history: list[tuple[int, float]] = []
        for _ in range(max_steps):
            before = self.omega.copy()
            dt = self.step()
            if self.steps % check_every == 0:
                change = float(np.abs(self.omega - before).max()) / dt
                self.residual = change
                history.append((self.steps, change))
                if on_check is not None:
                    on_check(self, self.steps, change)
                if change <= tolerance:
                    break
        return {
            "steps": self.steps,
            "dt": dt,
            "residual": self.residual,
            "converged": bool(self.residual <= tolerance),
            "history": history,
            "max_divergence": float(np.abs(self.divergence()).max()),
            "vorticity_defect": float(np.abs(
                self.omega[1:-1, 1:-1] - self.vorticity_from_streamfunction()).max()),
        }

    # ------------------------------------------------------------ measurement
    def functionals(self) -> dict[str, float]:
        """The shared functional set (:func:`cavity_uncertainty.functionals_from_fields`)."""
        import torch

        from .cavity_uncertainty import functionals_from_fields

        u, v = self.face_velocity()
        return functionals_from_fields(torch.as_tensor(u), torch.as_tensor(v),
                                      reynolds=self.config.reynolds,
                                      lid_velocity=self.lid_velocity)

    def centreline_u(self) -> tuple[np.ndarray, np.ndarray]:
        """``(y, u)`` on ``x = L/2`` (node line), in the Ghia orientation."""
        j = self.n // 2
        y = (np.arange(self.n + 1) * self.h)
        u_node, _ = self.node_velocity()
        return y, u_node[j, :].copy()

    def centreline_v(self) -> tuple[np.ndarray, np.ndarray]:
        """``(x, v)`` on ``y = L/2`` (node line)."""
        i = self.n // 2
        x = (np.arange(self.n + 1) * self.h)
        _, v_node = self.node_velocity()
        return x, v_node[:, i].copy()

    def to_payload(self) -> dict[str, float]:
        """Convergence/consistency diagnostics worth storing next to the functionals."""
        u, v = self.face_velocity()
        lap_defect = float(np.abs(self.omega[1:-1, 1:-1]
                                  - self.vorticity_from_streamfunction()).max())
        return {
            "n": int(self.n),
            "reynolds": float(self.config.reynolds),
            "steps": int(self.steps),
            "dt": float(self.dt_history[-1]) if self.dt_history else float("nan"),
            "residual": float(self.residual),
            "max_divergence": float(np.abs(self.divergence()).max()),
            "vorticity_defect": lap_defect,
            "lid_velocity_error": float(np.abs(u[:, -1] - self.lid_velocity).max()),
            "psi_min": float(self.psi.min()),
            "omega_wall_lid": float(self.omega[:, -1].mean()),
        }