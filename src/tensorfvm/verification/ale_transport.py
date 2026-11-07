"""Analytic verification of the ALE passive-scalar transport step.

Why this module exists
----------------------
``tensorfvm.moving_mesh.ale_scalar_step`` is the transport half of the ALE
capability introduced for PMM / forced-oscillation CFD.  Its acceptance tests
(``tests/test_moving_mesh_ale.py``) checked geometry (space conservation law),
free-stream preservation of a *constant* field and Galilean invariance of a
non-diffusive Gaussian, but they never compared a **diffusing** field against an
exact solution, and the module docstring used to claim "second-order time
accuracy" that no test measured.

The three probes here close that gap with quantities that have closed forms:

* ``uniform_preservation`` - a constant field must be reproduced to roundoff on
  a moving mesh *with diffusivity switched on* (the diffusive conductance has to
  enter the numerator and the diagonal symmetrically).  A constant scalar field
  is the free-stream state of the transport operator, so any drift is a
  non-physical source/sink.
* ``frozen_translation`` - a mesh translating under a frozen linear field must
  reproduce the exact shift ``u_m dt dphi/dx``; this exercises the boundary
  patch closure (the exterior state must be the owner value, never another
  cell's value).
* ``gaussian_case`` - the exact solution of ``phi_t + U phi_x = D phi_xx`` on an
  infinite domain,
  ``phi = sqrt(s0^2/s^2) exp(-(x - x0 - U t)^2 / (2 s^2))`` with
  ``s^2 = s0^2 + 2 D t``, on a static mesh and on a mesh comoving with the flow.
  Used for the accuracy gate and for the measured temporal order.

All cases are small Cartesian blocks (a few thousand cells, tens of steps), so
the whole module runs in seconds - it is a unit-scale verification, not a
benchmark campaign.
"""
from __future__ import annotations

import numpy as np

from ..moving_mesh import StructuredCurvilinearMesh, ale_scalar_step

__all__ = [
    "build_cartesian_block",
    "gaussian_exact",
    "gaussian_case",
    "uniform_preservation",
    "frozen_translation",
    "relative_l2",
    "observed_order",
    "time_order_study",
]


# ---------------------------------------------------------------------------
# block builder
# ---------------------------------------------------------------------------
def build_cartesian_block(nx: int, ny: int, nz: int, *,
                          lx: float, ly: float, lz: float) -> StructuredCurvilinearMesh:
    """Single-block Cartesian mesh with labelled non-periodic patches."""
    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    z = np.linspace(0.0, lz, nz + 1)
    nodes = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    patches = {
        (0, "low"): "inlet", (0, "high"): "outlet",
        (1, "low"): "wall", (1, "high"): "wall",
        (2, "low"): "wall", (2, "high"): "wall",
    }
    return StructuredCurvilinearMesh(nodes, patches=patches)


# ---------------------------------------------------------------------------
# exact solution
# ---------------------------------------------------------------------------
def gaussian_exact(x, t: float, *, x0: float, sigma0: float, U: float,
                   D: float) -> np.ndarray:
    """Exact 1-D advection-diffusion Gaussian on an infinite domain."""
    s2 = sigma0 ** 2 + 2.0 * D * t
    amp = np.sqrt(sigma0 ** 2 / s2)
    return amp * np.exp(-((np.asarray(x, dtype=float) - x0 - U * t) ** 2) / (2.0 * s2))


def relative_l2(numeric: np.ndarray, exact: np.ndarray) -> float:
    """Volume-weighted L2 relative error (volumes assumed uniform)."""
    num = np.asarray(numeric, dtype=float)
    ex = np.asarray(exact, dtype=float)
    return float(np.linalg.norm(num - ex) / max(np.linalg.norm(ex), 1e-300))


def observed_order(errors, step_sizes) -> float:
    """Least-squares slope of log(error) versus log(dt) (positive = order)."""
    e = np.asarray(errors, dtype=float)
    h = np.asarray(step_sizes, dtype=float)
    if np.any(e <= 0.0) or np.any(h <= 0.0):
        raise ValueError("errors and step sizes must be positive")
    slope = np.polyfit(np.log(h), np.log(e), 1)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# probe 1: constant field under diffusion on a moving mesh
# ---------------------------------------------------------------------------
def uniform_preservation(*, value: float = 2.5, diffusivity: float = 1e-2,
                         mesh_shift=(1e-3, 0.0, 0.0), dt: float = 1e-3,
                         nx: int = 20, ny: int = 4, nz: int = 1,
                         lx: float = 0.2, ly: float = 0.05, lz: float = 0.05):
    """Return ``max |phi_new - value|`` after one diffusive ALE step."""
    mesh = build_cartesian_block(nx, ny, nz, lx=lx, ly=ly, lz=lz)
    nodes_new = mesh.nodes + np.asarray(mesh_shift, dtype=float)
    g_old, g_new = mesh.metrics(), mesh.metrics(nodes_new)
    sweep = mesh.swept_volumes(mesh.nodes, nodes_new)
    phi = np.full(mesh.n_cells, float(value))
    out = ale_scalar_step(g_old, g_new, field_old=phi,
                          face_velocity=np.zeros((mesh.n_faces, 3)), sweep=sweep,
                          dt=dt, diffusivity=diffusivity)
    return float(np.abs(out - value).max())


# ---------------------------------------------------------------------------
# probe 2: frozen linear field, pure mesh translation
# ---------------------------------------------------------------------------
def frozen_translation(*, mesh_speed: float = 1.0, dt: float = 1e-3,
                       nx: int = 20, ny: int = 6, nz: int = 4,
                       lx: float = 0.2, ly: float = 0.05, lz: float = 0.05):
    """Transport a frozen linear field ``phi = x`` with a mesh moving at ``mesh_speed``.

    The fluid is at rest in the fixed frame, so the cell value must move with the
    mesh: ``phi_new(centre_new) = centre_new_x`` exactly (upwind is exact for a
    linear field on a uniform translating mesh).  Returns the max deviation from
    that exact value and the exact solution array.
    """
    mesh = build_cartesian_block(nx, ny, nz, lx=lx, ly=ly, lz=lz)
    shift = np.array([mesh_speed * dt, 0.0, 0.0])
    nodes_new = mesh.nodes + shift
    g_old, g_new = mesh.metrics(), mesh.metrics(nodes_new)
    sweep = mesh.swept_volumes(mesh.nodes, nodes_new)
    phi = g_old.centers[:, 0].copy()
    out = ale_scalar_step(g_old, g_new, field_old=phi,
                          face_velocity=np.zeros((mesh.n_faces, 3)), sweep=sweep,
                          dt=dt, diffusivity=0.0)
    exact = g_new.centers[:, 0]
    # Cells touching a patch carry the (first-order) zero-gradient closure error,
    # so the interior of the block is the meaningful machine-precision probe.
    touches_boundary = np.bincount(mesh.owner[~mesh.interior], minlength=mesh.n_cells) > 0
    interior = ~touches_boundary
    return {
        "max_all": float(np.abs(out - exact).max()),
        "max_interior": float(np.abs(out - exact)[interior].max()),
        "n_interior": int(interior.sum()),
        "exact": exact,
        "numeric": out,
    }


# ---------------------------------------------------------------------------
# probe 3: analytic advection-diffusion Gaussian
# ---------------------------------------------------------------------------
def gaussian_case(*, nx: int = 200, ny: int = 4, nz: int = 1,
                  lx: float = 0.2, ly: float = 0.05, lz: float = 0.05,
                  x0: float = 0.06, sigma0: float = 0.02, U: float = 0.5,
                  D: float = 1e-4, dt: float = 1e-3, t_end: float = 0.04,
                  comoving: bool = False):
    """March the exact Gaussian and return ``(relative_l2, history)``.

    ``comoving=True`` translates the mesh with the flow (the pulse is then
    quasi-static in the mesh frame), ``False`` keeps the mesh fixed.  Both must
    reproduce the same fixed-frame analytic profile, which is the Galilean
    invariance statement extended to the diffusive operator.
    """
    steps = int(round(t_end / dt))
    if abs(steps * dt - t_end) > 1e-12:
        raise ValueError("t_end must be a multiple of dt")
    mesh = build_cartesian_block(nx, ny, nz, lx=lx, ly=ly, lz=lz)
    nodes = mesh.nodes.copy()
    g_old = mesh.metrics(nodes)
    phi = gaussian_exact(g_old.centers[:, 0], 0.0, x0=x0, sigma0=sigma0, U=U, D=D)
    speed = U if comoving else 0.0
    face_velocity = np.tile(np.array([U, 0.0, 0.0]), (mesh.n_faces, 1))
    history = []
    for _ in range(steps):
        nodes_new = nodes + np.array([speed * dt, 0.0, 0.0])
        g_new = mesh.metrics(nodes_new)
        sweep = mesh.swept_volumes(nodes, nodes_new)
        phi = ale_scalar_step(g_old, g_new, field_old=phi, face_velocity=face_velocity,
                              sweep=sweep, dt=dt, diffusivity=D)
        nodes, g_old = nodes_new, g_new
        history.append(phi.copy())
    exact = gaussian_exact(g_old.centers[:, 0], t_end, x0=x0, sigma0=sigma0, U=U, D=D)
    return relative_l2(phi, exact), history


def time_order_study(*, dts=(2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4), t_end: float = 0.04,
                     nx: int = 400, **kwargs):
    """Relative L2 error and observed temporal order against the exact Gaussian."""
    errors, used = [], []
    for dt in dts:
        err, _ = gaussian_case(nx=nx, dt=dt, t_end=t_end, **kwargs)
        errors.append(err)
        used.append(dt)
    order = observed_order(errors, used)
    return {"dts": list(used), "errors": errors, "observed_order": order}