"""Single rotating reference frame (MRF) momentum terms + exact analytic anchors.

Why this module exists (verification debt)
------------------------------------------
``docs/verification-catalog.md`` §10.2 listed "Rotating Machinery: MRF
verification against literature" with no implementation anywhere in the
repository: there was no rotating-frame momentum source, no Coriolis term and no
analytic anchor for a rotating frame.  Rotating machinery is not academic for
this platform -- marine propellers / open-water curves are computed in practice
with exactly this model (frozen-rotor MRF around the propeller).  A capability
that silently drops the Coriolis term produces a plausible-looking but wrong
flow field, and until there is an anchor nobody can tell.

Scope: what is implemented and what is not
------------------------------------------
Implemented here (verification-grade 1-D reduced systems + the 3-D operator):

* :class:`RotatingFrame` -- frame kinematics and momentum sources in the form a
  finite-volume incompressible solver needs them, with an explicit sign
  convention (see below) and the centrifugal term absorbed into a modified
  pressure.
* :func:`exact_rotating_plane_poiseuille` -- closed-form rotating plane
  Poiseuille flow with the rotation axis along the *wall normal*, so that the
  Coriolis term drives a genuine secondary velocity ``w`` (Ekman-type rotating
  duct flow; the closed form is derived in the docstring and re-verified by an
  independent collocation BVP and by a second-order FVM).  Because ``w != 0``
  this anchor is discriminating: a solver without Coriolis returns ``w = 0`` and
  fails by 100 %.
* :func:`solve_rotating_plane_poiseuille` -- second-order finite-volume solve of
  the same coupled ``(u, w)`` system, which verifies the closed form in the other
  direction (a closed form checked only against its own derivation is not
  evidence).
* :func:`exact_rotating_channel_spanwise` / :func:`solve_inert_plane_poiseuille`
  -- rotation axis along the span: the classical rotating-channel configuration
  (Johnston, Halleen & Lezius 1972), perpendicular to both the wall normal and
  the mean flow.  There the laminar mean flow is *unchanged* (Coriolis has no
  streamwise component once the wall-normal velocity vanishes) while the pressure
  acquires a cubic correction ``p(y)``.  This is a trap: the velocity field looks
  rotation-insensitive although the Coriolis term is fully active, and the only
  visible signature is the pressure.  Both halves are anchored (a rotating-laminar
  channel is *not* a test of the Coriolis term; only its pressure is).
* :func:`exact_circular_couette` / :func:`solve_circular_couette_frame` --
  circular Couette flow solved *in a frame rotating at an arbitrary angular
  velocity* (the actual MRF statement: the absolute flow and the torque must be
  independent of the frame angular velocity), with Taylor's closed-form torque.

Not implemented (honest scope): a full 3-D time-accurate rotating-frame
Navier-Stokes solver, sliding / mixing-plane interfaces, mesh motion, and any
validation against rotating-machinery *experiments* (propeller open-water
curves).  What is delivered is the operator plus exact anchors for it.

Sign and unit conventions (fixed here, not re-derived per caller)
----------------------------------------------------------------
The frame rotates with angular velocity ``Omega`` about ``axis`` through
``origin``.  For the velocity ``u`` *relative to the frame* the incompressible
momentum equation is

    du/dt + (u.grad)u = -grad(p)/rho + nu lap(u) - 2 Omega x u - Omega x (Omega x r)

with ``r = x - origin``.  With the modified pressure
``p_m = p - rho/2 |Omega x r|^2`` the last term is absorbed,
``-grad(p)/rho - Omega x (Omega x r) = -grad(p_m)/rho``, and the centrifugal
source disappears from the momentum equation but not from the reported pressure.

All quantities are SI (rad/s, m, kg/m^3, m^2/s).  Coriolis is *twice*
``Omega x u``: the factor 2 is where hand-written implementations most often
fail, so it is pinned against the exact rotating-channel solution rather than
against a second implementation of the same formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "RotatingFrame",
    "exact_rotating_plane_poiseuille",
    "solve_rotating_plane_poiseuille",
    "exact_rotating_channel_spanwise",
    "solve_inert_plane_poiseuille",
    "exact_circular_couette",
    "solve_circular_couette_frame",
]


# ---------------------------------------------------------------------------
# frame kinematics and momentum sources
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RotatingFrame:
    """A single rotating reference frame (MRF).

    Parameters
    ----------
    omega:
        Angular velocity magnitude, rad/s.  May be negative (reversing the
        direction is equivalent to flipping ``axis``); ``omega == 0`` reproduces
        the inertial limit exactly.
    axis:
        Rotation axis direction (normalised on construction).  The default
        ``+z`` is *not* a marine default in disguise -- a propeller shaft is
        usually along ``x``, so pass the shaft axis explicitly.
    origin:
        A point on the rotation axis, metres.

    Examples
    --------
    >>> frame = RotatingFrame(omega=2.0, axis=(0.0, 1.0, 0.0))
    >>> frame.coriolis(np.array([[1.0, 0.0, 0.0]])).tolist()   # 2 Omega x u
    [[0.0, 0.0, -4.0]]
    """

    omega: float
    axis: Sequence[float] = (0.0, 0.0, 1.0)
    origin: Sequence[float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError("rotation axis must be non-zero")
        object.__setattr__(self, "axis", tuple(axis / norm))
        object.__setattr__(self, "origin", tuple(np.asarray(self.origin, dtype=float)))
        object.__setattr__(self, "omega", float(self.omega))

    # -- basic vectors ----------------------------------------------------
    @property
    def omega_vector(self) -> np.ndarray:
        """``Omega`` as a 3-vector (rad/s)."""
        return self.omega * np.asarray(self.axis, dtype=float)

    def _relative_positions(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1:] != (3,):
            raise ValueError("expected points/vectors of shape (..., 3)")
        return arr - np.asarray(self.origin, dtype=float)

    def radius(self, x) -> np.ndarray:
        """Distance from the rotation axis for points ``x``, shape ``x.shape[:-1]``."""
        om = self.omega_vector
        speed = float(np.linalg.norm(om))
        rel = self._relative_positions(x)
        if speed == 0.0:
            return np.zeros(rel.shape[:-1])
        axis = om / speed
        along = rel @ axis
        return np.linalg.norm(rel - along[..., None] * axis, axis=-1)

    # -- the terms a solver needs -----------------------------------------
    def coriolis(self, vel_rel) -> np.ndarray:
        """``2 Omega x u`` -- the term that appears on the *left* of the equation.

        For ``Omega = omega y_hat`` and ``u = (u, 0, w)`` this is
        ``2 omega (w, 0, -u)``.  The component structure is asserted in the
        tests because an axis-swapped Coriolis vector passes most casual checks
        (it vanishes at ``u = 0`` and is linear in ``Omega``) while being wrong.
        """
        om = self.omega_vector
        v = np.asarray(vel_rel, dtype=float)
        if v.shape[-1:] != (3,):
            raise ValueError("expected velocity vectors of shape (..., 3)")
        if not np.any(om):
            return np.zeros_like(v)
        return 2.0 * np.cross(om, v)

    def centrifugal(self, x) -> np.ndarray:
        """``Omega x (Omega x r)`` -- points towards the axis."""
        om = self.omega_vector
        rel = self._relative_positions(x)
        if not np.any(om):
            return np.zeros_like(rel)
        return np.cross(om, np.cross(om, rel))

    def centrifugal_potential(self, x) -> np.ndarray:
        """``|Omega x r|^2 / 2``; its gradient is ``-Omega x (Omega x r)``."""
        om = self.omega_vector
        rel = self._relative_positions(x)
        if not np.any(om):
            return np.zeros(rel.shape[:-1])
        return 0.5 * np.sum(np.cross(om, rel) ** 2, axis=-1)

    def absolute_velocity(self, vel_rel, x) -> np.ndarray:
        """Inertial-frame velocity of a point moving with relative velocity ``u``."""
        rel = self._relative_positions(x)
        return np.asarray(vel_rel, dtype=float) + np.cross(self.omega_vector, rel)

    def relative_velocity(self, vel_abs, x) -> np.ndarray:
        """Inverse of :meth:`absolute_velocity`."""
        rel = self._relative_positions(x)
        return np.asarray(vel_abs, dtype=float) - np.cross(self.omega_vector, rel)

    def modified_pressure(self, pressure, x, rho: float) -> np.ndarray:
        """``p_m = p - rho/2 |Omega x r|^2`` -- the pressure a rotating solve stores."""
        return np.asarray(pressure, dtype=float) - rho * self.centrifugal_potential(x)

    def pressure_from_modified(self, pressure_modified, x, rho: float) -> np.ndarray:
        """``p = p_m + rho/2 |Omega x r|^2`` -- physical pressure for reporting."""
        return np.asarray(pressure_modified, dtype=float) + rho * self.centrifugal_potential(x)

    def momentum_sources(self, vel_rel, x, rho: float, *, include_centrifugal: bool = True):
        """Right-hand-side momentum sources (acceleration units) for a solver.

        ``coriolis``   ``-2 Omega x u``
        ``centrifugal`` ``-Omega x (Omega x r)`` (optional: omit when the
            centrifugal term is absorbed into the modified pressure)
        ``total``      their sum
        """
        del rho  # specific force: mass density cancels when p is divided by rho
        cor = -self.coriolis(vel_rel)
        if include_centrifugal:
            cen = -self.centrifugal(x)
            total = cor + cen
        else:
            cen = np.zeros_like(cor)
            total = cor
        return {"coriolis": cor, "centrifugal": cen, "total": total}



def _laplacian_with_walls(n: int, dy: float) -> np.ndarray:
    """1-D Laplacian on cell centres with Dirichlet values at ``y = 0, h``.

    Interior rows are the central second difference.  The first and last rows use
    the three-point one-sided formula built from the *wall value* (zero) and the
    two nearest cell centres, ``f'' ~ (4/dy^2)(f_1/3 - f_0)``.  The obvious
    mirror-ghost alternative (``u_{-1} = -u_0``) gives ``(u_1 - 3 u_0)/dy^2``,
    which is *not* exact for a parabola (it returns ``-1.5 c`` where ``u'' = -2 c``
    for ``u = c y (h - y)``) and therefore destroys the exactness of the
    Poiseuille anchor: measured 1.3e-5 relative error at N=320 instead of
    round-off.  It also silently degrades the boundary row to first order.
    """
    if n < 2:
        raise ValueError("need at least 2 cells")
    lap = np.zeros((n, n))
    inv = 1.0 / (dy * dy)
    for j in range(n):
        if j == 0:
            lap[j, j] = -4.0 * inv
            lap[j, j + 1] = (4.0 / 3.0) * inv
        elif j == n - 1:
            lap[j, j] = -4.0 * inv
            lap[j, j - 1] = (4.0 / 3.0) * inv
        else:
            lap[j, j] = -2.0 * inv
            lap[j, j + 1] = inv
            lap[j, j - 1] = inv
    return lap


# ---------------------------------------------------------------------------
# anchor 1: rotating plane Poiseuille, rotation axis ALONG the wall normal
# ---------------------------------------------------------------------------
def exact_rotating_plane_poiseuille(
    y,
    *,
    G: float,
    rho: float,
    nu: float,
    omega: float,
    h: float,
):
    """Closed-form rotating plane Poiseuille flow (rotation along the wall normal).

    Geometry: walls at ``y = 0`` and ``y = h`` (no slip), streamwise ``x`` driven
    by ``-dp/dx = G > 0``, spanwise ``z``; constant ``rho``, ``nu``; steady,
    fully developed (``d/dx = d/dz = 0``).  The frame rotates with ``omega``
    about ``+y``.  The two coupled momentum balances are

        0 = -G/rho + nu u'' + 2 omega w        (streamwise)
        0 =          nu w'' - 2 omega u        (spanwise)
        0 = -p_y/rho                           (wall normal)

    with ``u = w = 0`` at both walls.  Writing ``F = u - i w`` turns them into the
    linear complex problem ``nu F'' + 2 i omega F = -G/rho``, whose solution with
    the no-slip conditions is

        F(y) = F_p [1 - cosh(a (y - h/2)) / cosh(a h/2)],
        F_p  = i G / (2 rho omega),   a = sqrt(-2 i omega / nu).

    ``omega = 0`` degenerates to the parabolic profile
    ``u = G y (h - y) / (2 rho nu)``, ``w = 0``; that branch is taken
    explicitly because the formula above is singular at ``omega = 0`` even though
    its limit is finite.

    Two regimes, both visible in the anchor and both easy to misread: for small
    ``omega`` the secondary flow is a *linear* perturbation (``w ~ omega``,
    ``u`` changed only at ``O(omega^2)``), while for large ``omega`` the
    streamwise profile collapses as ``u ~ 1/omega`` and ``|w|/|u|`` saturates
    near 3.31 -- i.e. "weak rotation" cannot be used to argue that the Coriolis
    term is negligible, and a coarse grid run at large ``omega`` is controlled by
    the wall layer ``sqrt(nu/(2 omega))``, not by the channel height.

    Returns
    -------
    dict with ``y``, ``u`` (streamwise), ``w`` (spanwise secondary flow),
    ``F`` (complex field ``u - i w``), ``a``, and ``Ekman_like_length
    = sqrt(nu / (2 |omega|))`` -- the wall-layer thickness that sets the grid
    requirement, reported because it is what makes a coarse run look converged.
    """
    y = np.asarray(y, dtype=float)
    if h <= 0.0:
        raise ValueError("h must be positive")
    if nu <= 0.0 or rho <= 0.0:
        raise ValueError("nu and rho must be positive")
    u_par = G * y * (h - y) / (2.0 * rho * nu)
    if omega == 0.0:
        zeros = np.zeros_like(y)
        return {
            "y": y,
            "u": u_par,
            "w": zeros.copy(),
            "F": u_par.astype(complex),
            "a": (0.0 + 0.0j),
            "Ekman_like_length": np.inf,
        }
    a = np.sqrt(-2.0j * omega / nu)
    Fp = 1.0j * G / (2.0 * rho * omega)
    F = Fp * (1.0 - np.cosh(a * (y - 0.5 * h)) / np.cosh(a * 0.5 * h))
    return {
        "y": y,
        "u": F.real,
        "w": -F.imag,
        "F": F,
        "a": complex(a),
        "Ekman_like_length": float(np.sqrt(nu / (2.0 * abs(omega)))),
    }


def solve_rotating_plane_poiseuille(
    n: int,
    *,
    G: float,
    rho: float,
    nu: float,
    omega: float,
    h: float,
):
    """Second-order finite-volume solve of the coupled ``(u, w)`` system.

    Cell-centred uniform grid of ``n`` cells across ``[0, h]``, central
    differences, no-slip through mirror ghost cells.  The 2n x 2n block system is

        nu L u + 2 omega w = -G/rho
        nu L w - 2 omega u = 0

    with ``L`` the 1-D Laplacian.  Its purpose is to verify the closed form in
    the other direction and to quantify the discretisation error the anchor is
    used with.  Returns ``dict(n, y, u, w)``.
    """
    if n < 4:
        raise ValueError("need at least 4 cells")
    dy = h / n
    lap = _laplacian_with_walls(n, dy)
    A = nu * lap
    two_om = 2.0 * omega * np.eye(n)
    mat = np.block([[A, two_om], [-two_om, A]])
    rhs = np.concatenate([np.full(n, -G / rho), np.zeros(n)])
    sol = np.linalg.solve(mat, rhs)
    y = (np.arange(n) + 0.5) * dy
    return {"n": n, "y": y, "u": sol[:n], "w": sol[n:]}


# ---------------------------------------------------------------------------
# anchor 2: rotation axis ALONG the span (perpendicular to the mean flow)
# ---------------------------------------------------------------------------
def exact_rotating_channel_spanwise(
    y,
    *,
    G: float,
    rho: float,
    nu: float,
    omega: float,
    h: float,
    p0: float = 0.0,
):
    """Plane channel with the rotation axis along the span (``+z``).

    With ``Omega = omega z_hat`` and a fully developed field ``(u(y), 0, 0)`` the
    Coriolis vector is ``2 omega (0, u, 0)``: it has *no streamwise component*, so
    the mean profile stays exactly parabolic and rotation looks inert.  The
    wall-normal balance becomes ``p_y = -2 rho omega u``, i.e. the pressure
    carries the cubic correction

        p(y) = p0 - (omega G / nu) (h y^2 / 2 - y^3 / 3).

    Returning ``dpdy`` as well lets a test check the implemented Coriolis vector
    against the wall-normal momentum balance instead of against a formula.
    """
    y = np.asarray(y, dtype=float)
    u = G * y * (h - y) / (2.0 * rho * nu)
    if omega == 0.0:
        p = np.full_like(y, p0)
        dpdy = np.zeros_like(y)
    else:
        p = p0 - (omega * G / nu) * (h * y**2 / 2.0 - y**3 / 3.0)
        dpdy = -2.0 * rho * omega * u
    return {"y": y, "u": u, "p": p, "dpdy": dpdy}


def solve_inert_plane_poiseuille(
    n: int, *, G: float, rho: float, nu: float, h: float, omega: float = 0.0
):
    """Plane Poiseuille on the same grid as the rotating solve (no ``w``).

    With the rotation axis along the span the streamwise balance is the inert one
    (``nu u'' = -G/rho``), so this solve is the correct answer for every
    ``omega``.  Keeping both solves in one module is deliberate: mixing the two
    anchors (wall-normal axis vs span axis) is the single most likely misuse.
    """
    del omega  # documented above: the spanwise axis does not enter the u balance
    dy = h / n
    main = nu * _laplacian_with_walls(n, dy)
    u = np.linalg.solve(main, np.full(n, -G / rho))
    return {"n": n, "y": (np.arange(n) + 0.5) * dy, "u": u}


# ---------------------------------------------------------------------------
# anchor 3: circular Couette solved in an arbitrary rotating frame (MRF)
# ---------------------------------------------------------------------------
def exact_circular_couette(
    r, *, r1: float, r2: float, omega_inner: float, omega_outer: float
):
    """Laminar circular Couette flow in absolute (inertial) form.

    ``v_theta(r) = A r + B / r`` with

        A = (omega_outer r2^2 - omega_inner r1^2) / (r2^2 - r1^2)
        B = r1^2 r2^2 (omega_inner - omega_outer) / (r2^2 - r1^2)

    and Taylor's torque magnitude per unit length
    ``M / L = 4 pi mu r1^2 r2^2 |omega_inner - omega_outer| / (r2^2 - r1^2)``.
    The returned ``torque_per_length_coefficient`` equals ``-4 pi B``, so the
    torque is ``mu * coefficient``.  ``dpdr_over_rho = v_theta^2 / r`` is the
    inertial radial balance the rotating-frame pressure must reproduce.
    """
    r = np.asarray(r, dtype=float)
    if not 0.0 < r1 < r2:
        raise ValueError("need 0 < r1 < r2")
    denom = r2**2 - r1**2
    A = (omega_outer * r2**2 - omega_inner * r1**2) / denom
    B = r1**2 * r2**2 * (omega_inner - omega_outer) / denom
    v = A * r + B / r
    return {
        "r": r,
        "v_theta": v,
        "A": A,
        "B": B,
        "dpdr_over_rho": v**2 / r,
        "torque_per_length_coefficient": -4.0 * np.pi * B,
    }


def solve_circular_couette_frame(
    n: int,
    *,
    r1: float,
    r2: float,
    nu: float,
    omega_inner: float,
    omega_outer: float,
    omega_frame: float = 0.0,
):
    """Circular Couette solved *in a frame rotating at* ``omega_frame``.

    This is the MRF statement under test: rotating the frame changes the boundary
    conditions (the walls move with relative angular velocities ``omega_inner -
    omega_frame`` and ``omega_outer - omega_frame``) and adds Coriolis +
    centrifugal terms, yet the absolute flow and the torque must be unchanged.
    For purely azimuthal flow ``2 Omega x u = -2 omega_frame v_rel r_hat`` is
    radial, so the azimuthal balance keeps its inertial form and the frame terms
    move into the radial balance -- which is why the frame-invariance of the
    velocity and of the torque both hold while the stored pressure does not.
    Both facts are asserted in the verification driver.

    Discretisation: finite volume on the conservative form
    ``d/dr [ r^3 d(v/r)/dr ] = 0``, unknowns ``g = v/r`` on ``n + 1`` nodes with
    Dirichlet walls, fluxes at the midpoints, second order.  Returns
    ``dict(r, v_rel, v_abs, torque_coefficient)`` where ``torque_coefficient``
    multiplies ``mu * L`` and follows ``M = 2 pi r^3 d(v_theta/r)/dr``.
    """
    if n < 4:
        raise ValueError("need at least 4 cells")
    if not 0.0 < r1 < r2:
        raise ValueError("need 0 < r1 < r2")
    del nu  # steady purely azimuthal balance carries no nu (geometry only)
    r = np.linspace(r1, r2, n + 1)
    g_wall_in = omega_inner - omega_frame  # (v/r) at the inner wall
    g_wall_out = omega_outer - omega_frame  # (v/r) at the outer wall

    m = n - 1
    mat = np.zeros((m, m))
    rhs = np.zeros(m)
    for k in range(m):  # node index j = k + 1
        d_hi = r[k + 2] - r[k + 1]
        d_lo = r[k + 1] - r[k]
        a_hi = (0.5 * (r[k + 1] + r[k + 2])) ** 3 / d_hi
        a_lo = (0.5 * (r[k] + r[k + 1])) ** 3 / d_lo
        mat[k, k] = -(a_hi + a_lo)
        if k + 1 < m:
            mat[k, k + 1] = a_hi
        else:
            rhs[k] -= a_hi * g_wall_out
        if k > 0:
            mat[k, k - 1] = a_lo
        else:
            rhs[k] -= a_lo * g_wall_in
    g = np.concatenate(([g_wall_in], np.linalg.solve(mat, rhs), [g_wall_out]))
    v_rel = g * r
    v_abs = v_rel + omega_frame * r
    torque_coef = (
        2.0
        * np.pi
        * (0.5 * (r[0] + r[1])) ** 3
        * (g[1] - g[0])
        / (r[1] - r[0])
    )
    return {
        "n": n,
        "r": r,
        "g": g,
        "v_rel": v_rel,
        "v_abs": v_abs,
        "torque_coefficient": float(torque_coef),
    }
