"""Body-fitted polar finite volumes around a circular cylinder (flow + CHT).

Grid
----
Node centred O-grid: radial nodes from ``r_node[0] = 0.5`` (the cylinder, lengths
normalised by the diameter) to ``r_node[-1] = r_out``, geometrically stretched
from ``wall_cell`` on the wall; azimuthal nodes are uniform and periodic.  Each
node owns a control volume bounded by the mid-points to its neighbours, so the
metric terms are those of a proper polar mesh.

Flow
----
Streamfunction-vorticity formulation::

    d_t w + u_r d_r w + (u_theta/r) d_theta w = (1/Re) lap(w)
    lap(psi) = -w,    u_r = (1/r) d_theta psi,   u_theta = -d_r psi

``psi = 0`` on the cylinder and ``psi = U r sin(theta)``, ``omega = 0`` on the
outer circle; the wall vorticity comes from a quadratic fit of ``psi`` (second
order).  The radial direction -- the stiff one, the mesh is clustered there -- is
implicit (periodic tridiagonal lines per azimuth), the azimuthal direction
(upwind convection + central diffusion) is explicit.  The Poisson operator is
factored once, so a step costs one sparse triangular solve plus one banded solve
per azimuth.

Heat
----
Two thermal domains (hot solid core, fluid annulus) coupled at the interface by a
Dirichlet-Neumann iteration -- deliberately *unlike*
:mod:`tensorfvm.cht.flatplate`, where one field plus a harmonic-mean face
conductance makes continuity automatic.  Here continuity has to be earned and is
then measured: the local jump ``max|q_fluid - q_solid|`` and the global jump of
the integral heat rate both go to the ledger.  The interface transfer uses an
exact arc-length overlap matrix, so a non-conformal azimuthal interface still
conserves the integral heat rate exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

__all__ = ["PolarVorticitySolver", "ConjugateCylinder", "CylinderCHTResult",
           "periodic_overlap_matrix", "one_sided_derivative"]


# --------------------------------------------------------------------------- grid
def _stretched_nodes(r_wall: float, n_cells: int, h0: float, r_out: float) -> np.ndarray:
    """Geometric stretching from ``h0`` at the wall out to ``r_out``."""
    total = r_out - r_wall
    lo, hi = 1.0 + 1e-12, 1.6
    for _ in range(400):
        q = 0.5 * (lo + hi)
        if h0 * (q ** n_cells - 1.0) / (q - 1.0) > total:
            hi = q
        else:
            lo = q
    q = 0.5 * (lo + hi)
    d = h0 * q ** np.arange(n_cells)
    return r_wall + np.concatenate([[0.0], np.cumsum(d)])


def periodic_overlap_matrix(n_from: int, n_to: int) -> np.ndarray:
    """Exact arc-length overlap of two uniform periodic partitions.

    Row ``j`` gives, for target cell ``j``, the fraction of its extent covered by
    each source cell.  Rows sum to one, and because both partitions tile the
    circle the same matrix conserves the integral of the transferred quantity --
    that is what makes the non-conformal interface conservative.
    """
    w = np.zeros((n_to, n_from))
    h_from, h_to = 1.0 / n_from, 1.0 / n_to
    for j in range(n_to):
        a, b = j * h_to, (j + 1) * h_to
        # enumerate the source cells the target cell overlaps instead of walking
        # ``pos`` forward: advancing by ``seg_end - pos`` stalls forever whenever
        # rounding makes ``floor(pos / h_from) * h_from`` land back on ``pos``
        # (which is what happened for every equal-partition case).
        i0 = int(math.floor(a / h_from))
        i1 = int(math.ceil(b / h_from))
        for i in range(i0, i1 + 1):
            low = max(a, i * h_from)
            high = min(b, (i + 1) * h_from)
            if high > low:
                w[j, i % n_from] += (high - low) / h_to
    return w


def one_sided_derivative(f0, f1, f2, s1: float, s2: float):
    """``df/dx`` at x=0 from values at 0, ``s1``, ``s2`` (quadratic, 2nd order).

    Lagrange coefficients differentiated at ``x = 0`` on the non-uniform stencil
    ``{0, s1, s2}``:

    ``a0 = -(s1+s2)/(s1 s2)``, ``a1 = s2/(s1 (s2-s1))``, ``a2 = -s1/(s2 (s2-s1))``.

    They reproduce a linear function exactly (``a0+a1+a2 = 0`` and
    ``a1 s1 + a2 s2 = 1``) -- the check that the previous coefficients failed,
    which biased every wall heat flux taken through this function by O(s2/s1-1).
    """
    if s1 <= 0.0 or s2 <= s1:
        raise ValueError("stencil must satisfy 0 < s1 < s2")
    a0 = -(s1 + s2) / (s1 * s2)
    a1 = s2 / (s1 * (s2 - s1))
    a2 = -s1 / (s2 * (s2 - s1))
    return a0 * f0 + a1 * f1 + a2 * f2


class _PolarGrid:
    """Node coordinates plus control-volume metrics for an annular O-grid."""

    def __init__(self, r_inner: float, r_outer: float, n_cells: int, h0: float,
                 n_theta: int):
        self.r_node = _stretched_nodes(r_inner, n_cells, h0, r_outer)
        self.n_node = self.r_node.size
        self.n_theta = n_theta
        self.dtheta = 2.0 * math.pi / n_theta
        self.theta = (np.arange(n_theta) + 0.5) * self.dtheta
        self.r_face = np.concatenate([[self.r_node[0]],
                                      0.5 * (self.r_node[:-1] + self.r_node[1:]),
                                      [self.r_node[-1]]])
        self.dr_cv = np.diff(self.r_face)
        self.dr_pp = np.diff(self.r_node)
        self.vol = self.r_node * self.dr_cv * self.dtheta


# ------------------------------------------------------------------ flow solver
class PolarVorticitySolver:
    """Streamfunction-vorticity FVM on a body-fitted polar O-grid."""

    def __init__(self, *, re: float, n_r: int = 120, n_theta: int = 180,
                 r_out: float = 12.0, wall_cell: float = 5.0e-3, u_inf: float = 1.0,
                 order: int = 1):
        if re <= 0:
            raise ValueError("Reynolds number must be positive")
        self.re = float(re)
        # ``order=1``: first-order upwind vorticity advection (legacy scheme; its
        # numerical viscosity ``U r dtheta/2`` is a large fraction of ``nu`` at
        # Re <= 40 and biases both the drag and the wall heat flux).
        # ``order=2``: van Leer limited flux-form reconstruction plus a deferred
        # correction of the radial flux; both remain conservative.
        self.order = int(order)
        self.nu = 1.0 / self.re
        self.u_inf = float(u_inf)
        self.grid = _PolarGrid(0.5, r_out, n_r - 1, wall_cell, n_theta)
        self.r_node = self.grid.r_node
        self.r_face = self.grid.r_face
        self.dr_pp = self.grid.dr_pp
        self.dr_cv = self.grid.dr_cv
        self.vol = self.grid.vol
        self.dtheta = self.grid.dtheta
        self.n_theta = self.grid.n_theta
        self.theta = self.grid.theta
        self.n_node = self.grid.n_node
        self.n_r = self.n_node
        self.n_int = self.n_node - 2
        self._build_poisson()

    # ------------------------------------------------------------- operators
    def _build_poisson(self):
        ni, nt = self.n_int, self.n_theta
        idx = np.arange(ni * nt).reshape(ni, nt)
        self._idx = idx
        dth = self.dtheta
        g_r = np.zeros(self.n_node)              # g_r[f]: face between nodes f-1, f
        for f in range(1, self.n_node):
            g_r[f] = self.r_face[f] * dth / self.dr_pp[f - 1]
        self._g_r = g_r
        g_w = g_r[1:ni + 1]                      # west face of interior node k
        g_e = g_r[2:ni + 2]                      # east face of interior node k
        g_t = self.dr_cv[1:-1] / (self.r_node[1:-1] * dth)
        self._g_t = g_t
        rows, cols, vals = [], [], []
        _d2 = np.zeros((ni, nt))
        _d2 -= (g_w + g_e)[:, None]
        _d2 -= 2.0 * g_t[:, None]
        diag = _d2.ravel().copy()
        for k in range(ni - 1):
            a, b = idx[k, :], idx[k + 1, :]
            g = g_e[k]
            rows += [a, b]
            cols += [b, a]
            vals += [np.full(nt, g), np.full(nt, g)]
        for j in range(nt):
            jn = (j + 1) % nt
            a, b = idx[:, j], idx[:, jn]
            rows += [a, b]
            cols += [b, a]
            vals += [g_t, g_t]
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(ni * nt, ni * nt)).tocsc()
        self._A_psi = (A + sp.diags(diag)).tocsc()
        self._lu_psi = spla.splu(self._A_psi)

    def _g_wall_face(self):
        return self.r_face[0] * self.dtheta / self.dr_pp[0]

    def _g_out_face(self):
        return self.r_face[-1] * self.dtheta / self.dr_pp[-1]

    # ------------------------------------------------------------ kinematics
    def face_fluxes(self, psi):
        """Radial volume flux ``r u_r dtheta`` and azimuthal flux ``u_theta dr``."""
        nt, nr = self.n_theta, self.n_node
        psi_ext = np.concatenate([psi, psi[:, :1]], axis=1)
        dpsi_t = psi_ext[:, 1:] - psi_ext[:, :-1]                # (nr, nt) dpsi/dtheta*dtheta
        urf = np.zeros((nr, nt))                                 # at r_face[f], theta_{j+1/2}
        urf[1:-1, :] = 0.5 * (dpsi_t[:-2, :] + dpsi_t[1:-1, :])
        # the flow *does* leave through the outer circle (psi varies along it);
        # leaving this face at zero made the outermost cell accumulate vorticity.
        urf[0, :] = 0.0
        urf[-1, :] = self.r_face[-1] * self.dtheta * self.u_inf * np.cos(self.theta)
        dpsi_r = np.zeros((nr, nt))
        denom = (self.r_node[2:] - self.r_node[:-2])[:, None]
        dpsi_r[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / denom
        ut = -dpsi_r                                             # u_theta at nodes
        return urf, ut

    def wall_vorticity(self, psi, order: int = 2):
        """``omega = -d^2 psi/dr^2`` at the wall.

        ``psi`` vanishes on the wall, so the local profile is fitted as
        ``psi = a s^2 + b s^3`` through the first two interior nodes.  For a
        uniform first step (``s2 = 2 s1``) this reduces to
        ``omega_w = -(8 psi_1 - psi_2) / (2 s1^2)``, which is second-order
        accurate; ``order=1`` keeps the older ``-2 psi_1/s1^2`` form, whose
        truncation error is ``O(s1)`` and which biases both the drag and the wall
        heat flux.
        """
        s1 = self.r_node[1] - self.r_node[0]
        if order == 1:
            return -2.0 * psi[1, :] / (s1 * s1)
        s2 = self.r_node[2] - self.r_node[0]
        a1 = psi[1, :] / (s1 * s1)
        a2 = psi[2, :] / (s2 * s2)
        b = (a2 - a1) / (s2 - s1)
        a = a1 - b * s1
        return -2.0 * a

    def solve_psi(self, omega, outer_psi):
        ni, nt = self.n_int, self.n_theta
        rhs = (-omega[1:-1, :] * self.vol[1:-1, None]).ravel()
        rhs[self._idx[-1, :]] -= (self.r_face[-1] * self.dtheta / self.dr_pp[-1]) * outer_psi
        psi = np.zeros((self.n_node, nt))
        psi[1:-1, :] = self._lu_psi.solve(rhs).reshape(ni, nt)
        psi[-1, :] = outer_psi
        return psi

    # ------------------------------------------------------------ time march
    @staticmethod
    def _harmonic_limiter(d_up, d_down):
        """van Leer / harmonic limiter on two consecutive differences.

        Returns ``2 d1 d2/(d1+d2)`` where the differences share a sign (smooth
        data: the second-order upwind slope) and zero where they do not (an
        extremum: fall back to the first-order upwind slope).  Bounded and TVD.
        """
        same = d_up * d_down > 0.0
        denom = np.where(same, d_up + d_down, 1.0)
        return np.where(same, 2.0 * d_up * d_down / denom, 0.0)

    def _azimuthal_explicit(self, omega, ut, high_order=False):
        """Explicit azimuthal advection, ``-(u_theta/r) d omega/d theta``.

        ``high_order`` replaces the first-order upwind difference by the limited
        second-order upwind slope.  First-order upwind adds a numerical viscosity
        ``u_theta r dtheta/2`` which at ``Re = 40`` with ``n_theta = 72`` is of the
        same order as the molecular viscosity -- that is the dominant error of the
        legacy scheme, not the wall boundary condition.
        """
        nt = self.n_theta
        r = self.r_node[1:-1]
        adv = np.zeros_like(omega)
        for j in range(nt):
            jp, jm = (j + 1) % nt, (j - 1) % nt
            u = ut[:, j]
            d_up = omega[:, j] - omega[:, jm]
            d_down = omega[:, jp] - omega[:, j]
            dT = np.where(u > 0.0, d_up, d_down) if not high_order else \
                self._harmonic_limiter(d_up, d_down)
            adv[:, j] = -u / r * dT / self.dtheta
        return adv

    def radial_velocity(self, psi):
        """Radial velocity at the nodes from a centred azimuthal difference."""
        nt = self.n_theta
        psi_ext = np.concatenate([psi[:, -1:], psi, psi[:, :1]], axis=1)
        ur = np.zeros_like(psi)
        ur[1:-1, :] = ((psi_ext[1:-1, 2:] - psi_ext[1:-1, :-2])
                       / (2.0 * self.dtheta * self.r_node[1:-1, None]))
        ur[0, :] = 0.0
        ur[-1, :] = self.u_inf * np.cos(self.theta)
        return ur

    def _radial_implicit(self, omega, dt, ur, wall_value, outer_value):
        """Backward Euler radial diffusion + upwind convection (banded lines).

        ``ur`` is the nodal radial velocity; the face value is the mean of the two
        bracketing nodes.  The convective assembly adds the *inflow* magnitudes to
        the diagonal, which keeps the line matrix an M-matrix (positive solution)
        even where the radial flux is strongly convergent -- assembling the exact
        ``sum face flux`` diagonal instead makes the line solve blow up in the
        stagnation region, where the radial inflow is carried away by the
        azimuthal flux.
        """
        ni, nt = self.n_int, self.n_theta
        nu = self.nu
        vol = self.vol[1:-1]
        gm = np.zeros(ni)
        gp = np.zeros(ni)
        fw = np.zeros((ni, nt))                 # volume flux at the west face (+r)
        fe = np.zeros((ni, nt))                 # volume flux at the east face (+r)
        for k in range(ni):
            i = k + 1
            gm[k] = nu * self.r_face[i] * self.dtheta / self.dr_pp[i - 1]
            gp[k] = nu * self.r_face[i + 1] * self.dtheta / self.dr_pp[i]
            u_w = 0.5 * (ur[k] + ur[k - 1]) if k > 0 else 0.0
            u_e = 0.5 * (ur[k] + ur[k + 1]) if k + 1 < ni else ur[-1]
            fw[k] = self.r_face[i] * self.dtheta * u_w
            fe[k] = self.r_face[i + 1] * self.dtheta * u_e
        diag = np.repeat((1.0 + dt * (gm + gp) / vol)[:, None], nt, axis=1)
        sub = np.repeat((-dt * gm / vol)[:, None], nt, axis=1)
        sup = np.repeat((-dt * gp / vol)[:, None], nt, axis=1)
        diag += dt * (np.where(fe > 0, fe, 0.0) - np.where(fw < 0, fw, 0.0)) / vol[:, None]
        sup += dt * np.where(fe < 0, fe, 0.0) / vol[:, None]
        sub += dt * np.where(fw > 0, -fw, 0.0) / vol[:, None]
        rhs = omega.copy()
        rhs[0, :] += dt * gm[0] / vol[0] * wall_value
        rhs[-1, :] += dt * gp[-1] / vol[-1] * outer_value
        ab = np.zeros((3, ni))
        out = np.empty_like(omega)
        for j in range(nt):
            ab[0, 1:] = sup[:-1, j]
            ab[1] = diag[:, j]
            ab[2, :-1] = sub[1:, j]
            out[:, j] = solve_banded((1, 1), ab, rhs[:, j])
        return out

    def march(self, *, t_end: float, dt: float, tol_steady: float = 1e-8,
              max_steps: int = 400000, sample_from: float | None = None,
              check_from: int = 100, t_ramp: float = 5.0,
              state: tuple | None = None):
        """March the vorticity-streamfunction system in time.

        ``state=(psi, omega)`` restarts a run from a saved pair of fields, which is
        what makes it possible to walk a long march forward in wall-clock-bounded
        segments (the flow solve of a cross-flow CHT case costs minutes).
        """
        nt = self.n_theta
        outer_psi = self.u_inf * self.r_node[-1] * np.sin(self.theta)
        if state is None:
            psi = np.zeros((self.n_node, nt))
            omega = np.zeros((self.n_node, nt))
        else:
            psi = np.array(state[0], dtype=float, copy=True)
            omega = np.array(state[1], dtype=float, copy=True)
        hist, samples, sample_t = [], [], []
        drag_history = []
        prev = np.zeros(self.n_theta)
        n_steps = min(int(round(t_end / dt)), max_steps)
        steady = False
        step = 0
        for step in range(n_steps):
            urf, ut = self.face_fluxes(psi)
            inner = omega[1:-1, :]
            w = inner + dt * self._azimuthal_explicit(inner, ut[1:-1, :],
                                                      high_order=self.order >= 2)
            w = self._radial_implicit(w, dt, self.radial_velocity(psi),
                                      omega[0, :], 0.0)
            omega[1:-1, :] = w
            omega[0, :] = 0.0
            omega[-1, :] = 0.0
            ramp = 1.0 if t_ramp <= 0 else min(1.0, step * dt / t_ramp)
            psi = self.solve_psi(omega, ramp * outer_psi)
            omega[0, :] = self.wall_vorticity(psi)
            prev = omega[0, :].copy() if step == 0 else prev
            hist.append(float(np.max(np.abs(omega[0, :] - prev))))
            prev = omega[0, :].copy()
            if (step % 200) == 0:
                cd, cl = self.drag_lift(omega[0, :])
                drag_history.append((float(step * dt), float(cd), float(cl)))
            if sample_from is not None and step * dt >= sample_from:
                samples.append(omega[0, :].copy())
                sample_t.append(step * dt)
            if step > check_from and len(hist) > 150:
                if max(hist[-100:]) < tol_steady:
                    steady = True
                    break
        return {
            "psi": psi, "omega": omega, "steps": step + 1, "dt": dt,
            "drag_history": drag_history,
            "time": (step + 1) * dt, "steady": steady,
            "wall_vorticity": omega[0, :].copy(),
            "wall_vorticity_history": hist,
            "wall_vorticity_mean": float(np.mean(omega[0, :])),
            "samples": samples, "sample_time": sample_t,
        }

    # ------------------------------------------------------------- diagnostics
    def drag_lift(self, wall_vorticity):
        """``(Cd, Cl)`` from the wall vorticity integral (2/Re convention)."""
        dth = self.dtheta
        w = wall_vorticity
        cd = -(2.0 / self.re) * np.sum(w * np.sin(self.theta)) * dth
        cl = -(2.0 / self.re) * np.sum(w * np.cos(self.theta)) * dth
        return float(cd), float(cl)


@dataclass
class CylinderCHTResult:
    nusselt_local: np.ndarray
    nusselt_mean: float
    nusselt_mean_flux_weighted: float
    wall_temperature: np.ndarray
    theta: np.ndarray
    q_fluid: np.ndarray
    q_solid: np.ndarray
    metrics: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)


class ConjugateCylinder:
    """Hot solid core plus fluid annulus, coupled through the interface flux."""

    def __init__(self, *, re: float, pr: float = 0.7, solid_ratio: float = 100.0,
                 n_theta_solid: int | None = None, n_r_solid: int = 40,
                 r_inner: float = 0.15, flow: PolarVorticitySolver | None = None):
        self.re = float(re)
        self.pr = float(pr)
        self.solid_ratio = float(solid_ratio)
        self.r_inner = float(r_inner)
        self.kappa_fluid = 1.0 / (self.re * self.pr)
        self.kappa_solid = self.solid_ratio * self.kappa_fluid
        self.flow = flow if flow is not None else PolarVorticitySolver(re=re)
        self.n_theta = self.flow.n_theta
        self.n_theta_solid = int(n_theta_solid) if n_theta_solid else self.n_theta
        self.solid_grid = _PolarGrid(self.r_inner, 0.5, n_r_solid, (0.5 - self.r_inner)
                                     / n_r_solid, self.n_theta_solid)
        self.overlap = periodic_overlap_matrix(self.n_theta, self.n_theta_solid)
        self._build_solid()

    # ------------------------------------------------------------ solid solve
    def _build_solid(self):
        g = self.solid_grid
        ni = g.n_node - 1                       # node 0 is the inner Dirichlet
        nt = g.n_theta
        idx = np.arange(ni * nt).reshape(ni, nt)
        self._solid_idx = idx
        kap = self.kappa_solid
        g_r = np.zeros(g.n_node)
        for f in range(1, g.n_node):
            g_r[f] = kap * g.r_face[f] * g.dtheta / g.dr_pp[f - 1]
        g_w = g_r[1:ni + 1]
        g_e = np.zeros(ni)
        g_e[:-1] = g_r[2:ni + 1]                # the interface face carries the Neumann flux
        g_t = kap * g.dr_cv[1:] / (g.r_node[1:] * g.dtheta)
        rows, cols, vals = [], [], []
        _d2 = np.zeros((ni, nt))
        _d2 -= (g_w + g_e)[:, None]
        _d2 -= 2.0 * g_t[:, None]
        diag = _d2.ravel().copy()
        for k in range(ni - 1):
            a, b = idx[k, :], idx[k + 1, :]
            gg = g_e[k]
            rows += [a, b]
            cols += [b, a]
            vals += [np.full(nt, gg), np.full(nt, gg)]
        for j in range(nt):
            jn = (j + 1) % nt
            a, b = idx[:, j], idx[:, jn]
            rows += [a, b]
            cols += [b, a]
            vals += [g_t, g_t]
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(ni * nt, ni * nt)).tocsc()
        self._A_solid = (A + sp.diags(diag)).tocsc()
        self._lu_solid = spla.splu(self._A_solid)
        self._g_inner = g_w[0]
        self._g_r_solid = g_r
        self._g_t_solid = g_t

    def solve_solid(self, q_interface):
        """Solid conduction with inner ``T=1`` and an interface Neumann flux."""
        g = self.solid_grid
        ni, nt = g.n_node - 1, g.n_theta
        b = np.zeros(ni * nt)
        # inner Dirichlet value contributes g_inner * T_inner
        b[self._solid_idx[0, :]] += self._g_inner * 1.0
        # Neumann flux entering the last node through its outer CV face.
        # ``_A_solid`` is assembled in the same negated-diagonal convention as the
        # fluid operator, so the right-hand side is negated on the way in
        # (``+q`` = heat added to the outer cell) -- without this the solid runs
        # the other way, e.g. an adiabatic interface leaves ``T = -1`` instead of
        # the prescribed core value ``T = +1``.
        area = g.r_face[-1] * g.dtheta
        b[self._solid_idx[-1, :]] += q_interface * area
        T = self._lu_solid.solve(-b).reshape(ni, nt)
        full = np.zeros((ni + 1, nt))
        full[1:, :] = T
        full[0, :] = 1.0
        return full

    def solid_interface_state(self, T_solid):
        """Interface temperature and the solid-side flux at the interface."""
        g = self.solid_grid
        s1 = g.r_node[-1] - g.r_node[-2]
        s2 = g.r_node[-1] - g.r_node[-3]
        # derivative w.r.t. r evaluated at the interface, looking inward
        dr = one_sided_derivative(T_solid[-1, :], T_solid[-2, :], T_solid[-3, :], s1, s2)
        q_solid = self.kappa_solid * dr
        return T_solid[-1, :], q_solid

    # ------------------------------------------------------------ fluid solve
    def _advective_net_out(self, T, urf, ut, wall_values, high_order: bool):
        """Net advective heat *leaving* each interior cell (per unit depth).

        ``T`` is the interior field ``(n_int, n_theta)``; the radial upstream
        value at the first interior row is the wall value and the azimuthal
        direction is periodic.  With ``high_order`` the face values are the
        van Leer limited linear reconstruction, otherwise plain upwind.  Both
        variants are *flux* differences over the same faces, so their cell sums
        telescope and the deferred correction below cannot create heat.
        """
        flow = self.flow
        ni, nt = flow.n_int, flow.n_theta
        net = np.zeros((ni, nt))
        kap = self.kappa_fluid

        def van_leer(up, down):
            r = np.where(np.abs(down) > 1e-300, up / np.where(down == 0.0, 1e-300, down), 0.0)
            psi = (r + np.abs(r)) / (1.0 + np.abs(r))
            return psi

        # --- radial faces ---------------------------------------------------
        for k in range(ni - 1):
            F = 0.5 * (urf[k + 1, :] + np.roll(urf[k + 1, :], 1))
            pos = F > 0
            if k == 0:
                up_p = np.full(nt, 0.0)
                up_n = np.full(nt, 0.0)
            else:
                up_p = T[k - 1, :]
                up_n = T[k + 1, :] if k + 1 < ni else np.zeros(nt)
            up_n2 = T[k + 2, :] if k + 2 < ni else T[k + 1, :]
            t_p, t_n = T[k, :], T[k + 1, :]
            if high_order:
                face = np.where(
                    pos,
                    t_p + 0.5 * van_leer(t_p - up_p, t_n - t_p) * (t_n - t_p),
                    t_n + 0.5 * van_leer(t_n - up_n2, t_p - t_n) * (t_p - t_n),
                )
                face = np.where(k == 0, np.where(pos, t_p, face), face)
            else:
                face = np.where(pos, t_p, t_n)
            net[k, :] += F * face
            net[k + 1, :] -= F * face

        # --- azimuthal faces (periodic) -------------------------------------
        area = flow.dr_cv[1:-1]
        for j in range(nt):
            jp = (j + 1) % nt
            F = 0.5 * (ut[:, j] + ut[:, jp]) * area
            pos = F > 0
            jm = (j - 1) % nt
            jpp = (jp + 1) % nt
            t_p, t_n = T[:, j], T[:, jp]
            if high_order:
                face = np.where(
                    pos,
                    t_p + 0.5 * van_leer(t_p - T[:, jm], t_n - t_p) * (t_n - t_p),
                    t_n + 0.5 * van_leer(t_n - T[:, jpp], t_p - t_n) * (t_p - t_n),
                )
            else:
                face = np.where(pos, t_p, t_n)
            net[:, j] += F * face
            net[:, jp] -= F * face
        return net

    def solve_fluid(self, t_wall, psi, outer_temperature=None, order: int = 1,
                    dc_max_iter: int = 200, dc_tol: float = 1e-13):
        """Steady fluid energy equation with a wall temperature ``t_wall(theta)``.

        ``outer_temperature`` (a scalar) turns the outer boundary into a cold
        Dirichlet wall; left as ``None`` the outer boundary is adiabatic, which is
        the convective cross-flow configuration.

        ``order=1`` uses first-order upwind convection (the original scheme).
        ``order=2`` keeps that matrix but adds a *conservative* deferred
        correction: the face values are replaced by van Leer limited linear
        reconstructions and the difference of the two flux divergences is lagged
        on the right-hand side until it stops changing.  Because both face-value
        sets are flux differences over the same faces, the converged solution is
        the second-order flux balance and the energy ledger still closes.
        """
        flow = self.flow
        ni, nt = flow.n_int, flow.n_theta
        idx = flow._idx
        urf, ut = flow.face_fluxes(psi)
        urf = urf[1:-1, :]
        ut = ut[1:-1, :]
        kap = self.kappa_fluid
        dth = flow.dtheta
        g_r = np.zeros(flow.n_node)
        for f in range(1, flow.n_node):
            g_r[f] = kap * flow.r_face[f] * dth / flow.dr_pp[f - 1]
        g_w = g_r[1:ni + 1]
        g_e = g_r[2:ni + 2]
        g_t = kap * flow.dr_cv[1:-1] / (flow.r_node[1:-1] * dth)
        rows, cols, vals = [], [], []
        _d2 = np.zeros((ni, nt))
        _d2 -= (g_w + g_e)[:, None]
        _d2 -= 2.0 * g_t[:, None]
        diag = _d2.ravel().copy()
        for k in range(ni - 1):
            a, b = idx[k, :], idx[k + 1, :]
            g = g_e[k]
            rows += [a, b]
            cols += [b, a]
            vals += [np.full(nt, g), np.full(nt, g)]
        for j in range(nt):
            jn = (j + 1) % nt
            a, b = idx[:, j], idx[:, jn]
            rows += [a, b]
            cols += [b, a]
            vals += [g_t, g_t]
        b = np.zeros(ni * nt)
        b[idx[0, :]] += g_w[0] * t_wall
        if outer_temperature is not None:
            b[idx[-1, :]] += g_e[-1] * outer_temperature
        # radial upwind convection between interior nodes
        for k in range(ni - 1):
            F = 0.5 * (urf[k + 1, :] + np.roll(urf[k + 1, :], 1))
            pos = F > 0
            P, N = idx[k, :], idx[k + 1, :]
            diag[P] += np.where(pos, -F, 0.0)
            diag[N] += np.where(pos, 0.0, F)
            rows += [P, N]
            cols += [N, P]
            vals += [np.where(pos, 0.0, -F), np.where(pos, F, 0.0)]
        # azimuthal upwind convection (outward flux through the +theta face)
        area = flow.dr_cv[1:-1]
        for j in range(nt):
            jp = (j + 1) % nt
            F = 0.5 * (ut[:, j] + ut[:, jp]) * area
            pos = F > 0
            P, N = idx[:, j], idx[:, jp]
            diag[P] += np.where(pos, -F, 0.0)
            diag[N] += np.where(pos, 0.0, F)
            rows += [P, N]
            cols += [N, P]
            vals += [np.where(pos, 0.0, -F), np.where(pos, F, 0.0)]
        # Sign convention: the assembly puts the (negative) sum of the face
        # conductances on the diagonal and the positive conductances off it, i.e.
        # it assembles ``-(M T) = b``.  The inhomogeneous Dirichlet terms above are
        # written for ``+M T = b``, so the right-hand side must be negated when the
        # system is solved -- otherwise a pure-conduction annulus comes out with the
        # correct shape but the opposite sign in the interior (the wall nodes are
        # overwritten afterwards, which is exactly why the error hides).
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(ni * nt, ni * nt)).tocsc()
        A = (A + sp.diags(diag)).tocsc()
        T = spla.spsolve(A, -b).reshape(ni, nt)
        self._dc_iterations = 1
        if order == 2:
            lu = spla.splu(A.tocsc())
            for dc_it in range(dc_max_iter):
                up = self._advective_net_out(T, urf, ut, t_wall, high_order=False)
                ho = self._advective_net_out(T, urf, ut, t_wall, high_order=True)
                dc = (ho - up).ravel()
                T_new = lu.solve(dc - b).reshape(ni, nt)
                delta = float(np.max(np.abs(T_new - T)))
                T = T_new
                self._dc_iterations += 1
                if delta < dc_tol:
                    break
            self._dc_residual = delta
        full = np.zeros((flow.n_node, nt))
        full[1:-1, :] = T
        full[0, :] = t_wall
        s1 = flow.r_node[1] - flow.r_node[0]
        s2 = flow.r_node[2] - flow.r_node[0]
        dr = one_sided_derivative(t_wall, T[0, :], T[1, :], s1, s2)
        q_wall = -kap * dr                              # positive into the fluid
        return full, q_wall

    # ------------------------------------------------------------- coupling
    def couple(self, psi, *, relax: float = 0.7, max_iter: int = 300,
               tol: float = 1e-9, t_wall_init: float = 1.0,
               outer_temperature=None):
        """Dirichlet-Neumann iteration graded on the interface flux jump."""
        t_wall = np.full(self.n_theta, t_wall_init)
        history = []
        qf = np.zeros(self.n_theta)
        for it in range(max_iter):
            T_fluid, qf = self.solve_fluid(t_wall, psi, outer_temperature)
            # ``qf`` is the heat flux *into the fluid*, i.e. the flux leaving the
            # solid: that is the quantity compared with the solid-side flux below.
            # What *enters* the solid across the interface is ``-qf``.
            q_to_solid = self.overlap @ qf
            T_solid = self.solve_solid(-q_to_solid)
            t_iface, q_solid_side = self.solid_interface_state(T_solid)
            t_wall_solid = self.overlap @ t_wall
            q_solid_at_wall = q_solid_side + self.kappa_solid * (t_iface - t_wall_solid) / (
                self.solid_grid.r_node[-1] - self.solid_grid.r_node[-2])
            q_scale = max(float(np.max(np.abs(q_to_solid))), 1e-30)
            local = float(np.max(np.abs(q_to_solid - q_solid_at_wall))) / q_scale
            glob = abs(float(np.sum(q_to_solid)) - float(np.sum(q_solid_at_wall))) / max(
                abs(float(np.sum(q_to_solid))), 1e-30)
            history.append({"iteration": it, "local_residual": local,
                            "global_residual": glob})
            t_wall = t_wall + relax * (t_iface - t_wall)
            if local < tol and glob < tol:
                break

        T_fluid, qf = self.solve_fluid(t_wall, psi, outer_temperature)
        q_to_solid = self.overlap @ qf
        T_solid = self.solve_solid(-q_to_solid)
        t_iface, q_solid_side = self.solid_interface_state(T_solid)
        t_wall_solid = self.overlap @ t_wall
        q_solid_at_wall = q_solid_side + self.kappa_solid * (t_iface - t_wall_solid) / (
            self.solid_grid.r_node[-1] - self.solid_grid.r_node[-2])
        q_scale = max(float(np.max(np.abs(q_to_solid))), 1e-30)
        nun = qf / (self.kappa_fluid * np.maximum(t_wall, 1e-30))
        metrics = {
            "reynolds": self.re,
            "prandtl": self.pr,
            "solid_ratio": self.solid_ratio,
            "interface_local_residual_rel": float(
                np.max(np.abs(q_to_solid - q_solid_at_wall)) / q_scale),
            "interface_global_residual_rel": abs(
                float(np.sum(q_to_solid)) - float(np.sum(q_solid_at_wall))) / max(
                    abs(float(np.sum(q_to_solid))), 1e-30),
            "interface_temperature_jump": float(np.max(np.abs(t_iface - t_wall_solid))),
            "coupling_iterations": len(history),
            "coupling_history_tail": history[-3:],
            "wall_temperature_mean": float(np.mean(t_wall)),
            "wall_temperature_ptp": float(np.ptp(t_wall)),
            "heat_rate_fluid": float(np.sum(qf) * (2.0 * math.pi / self.n_theta)),
            "heat_rate_solid": float(np.sum(q_to_solid)
                                     * (2.0 * math.pi / self.n_theta_solid)),
            "nusselt_mean": float(np.sum(qf)) / (self.n_theta * self.kappa_fluid
                                                 * float(np.mean(t_wall))),
            "nusselt_mean_flux_weighted": float(np.mean(nun)),
            "conformal": self.n_theta == self.n_theta_solid,
        }
        metrics["energy_rate_balance_rel"] = abs(
            metrics["heat_rate_fluid"] - metrics["heat_rate_solid"]) / max(
                abs(metrics["heat_rate_fluid"]), 1e-30)
        return CylinderCHTResult(
            nusselt_local=nun, nusselt_mean=metrics["nusselt_mean"],
            nusselt_mean_flux_weighted=metrics["nusselt_mean_flux_weighted"],
            wall_temperature=t_wall, theta=self.flow.theta, q_fluid=qf,
            q_solid=q_to_solid, metrics=metrics,
            fields={"T_fluid": T_fluid, "T_solid": T_solid, "psi": psi})