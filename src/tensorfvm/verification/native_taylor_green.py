"""Frozen-time Taylor--Green vortex reference for the native hybrid Navier--Stokes solver.

The periodic MAC prototype is qualified against the exact Taylor--Green vortex in
:mod:`tensorfvm.verification.taylor_green_mac`, which integrates the decay in
time.  The native hybrid 3-D backend
(:class:`tensorfvm.native_hybrid_ns3d.NativeHybridNavierStokes3D`) is a *steady*
solver: it has no transient integrator, so a time-marched TGV decay ladder cannot
be run on it.  What can be qualified is the spatial operator, by freezing the
exact solution at one instant ``t`` and adding the time term as a manufactured
body force.

The exact solution on the doubly-periodic box ``[0, 2*pi]^3`` is

    u(x,y,z,t) =  A cos(x) sin(y) exp(-2 nu t)
    v(x,y,z,t) = -A sin(x) cos(y) exp(-2 nu t)
    w          =  0
    p(x,y,z,t) = -(A^2/4) (cos 2x + cos 2y) exp(-4 nu t)        (rho = 1)

with ``d u/d t = -2 nu u``.  Freezing the field at ``t`` and moving the time
term to the right-hand side gives the *steady* problem

    (u.grad) u = -grad(p) + nu lap(u) + f,      f = -d u/d t = 2 nu u,

which the frozen field satisfies identically.  ``f`` is the manufactured body
force returned by :func:`tgv_fields`; the boundary velocity is the exact
Dirichlet trace, sampled as face area means by
:func:`tensorfvm.native_manufactured3d.exact_boundary_velocity_means`.

Scope notes (measured, see ``scripts/verification-native-taylor-green``):

* The momentum/incompressibility content is qualified: the discrete velocity
  reproduces the frozen exact velocity at observed order ~1.9 with a discrete
  divergence at round-off.
* The cell-centred pressure of this *closed, all-Dirichlet* configuration
  carries a gauge-invariant odd--even mode that the discrete system does not
  determine, so the pressure error does not converge under refinement (it is
  reported by :func:`tgv_metrics` but never gated here).
* The steady frozen problem stiffens with decreasing viscosity; ``nu = 0.05``
  (the physical case of the MAC study) does not converge in the Newton
  iteration, therefore the published study uses ``nu = 1.0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..native_hybrid_ns3d import NativeHybridNavierStokes3D
from ..native_manufactured3d import exact_boundary_velocity_means, volume_weighted_error
from ..native_mesh3d import NativeMesh3D

BOX = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Closed-form reference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaylorGreenCase:
    """Physical parameters of the exact Taylor--Green vortex.

    ``viscosity`` sets the decay rate (``2 nu`` for velocity, ``4 nu`` for the
    pressure and the kinetic energy); ``amplitude`` sets the convective and the
    pressure scale.  Density is fixed to 1 because the closed-form pressure is
    written for ``rho = 1``.
    """

    amplitude: float = 1.0
    viscosity: float = 0.05
    density: float = 1.0

    def __post_init__(self) -> None:
        for name in ("amplitude", "viscosity", "density"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}: {value!r}")
        if self.density != 1.0:
            raise ValueError(
                "the closed-form pressure of this reference is written for rho = 1; "
                "generalise the case before using another density"
            )

    @property
    def decay_exponent(self) -> float:
        """Velocity decay rate: 2 * nu."""
        return 2.0 * self.viscosity

    @property
    def energy_decay_exponent(self) -> float:
        """Kinetic-energy decay rate: 4 * nu."""
        return 4.0 * self.viscosity


def tgv_fields(points: np.ndarray, case: TaylorGreenCase, time_s: float = 0.0) -> Dict:
    """Exact velocity, pressure, gradient and frozen-time force at ``points``.

    Parameters
    ----------
    points :
        ``(N, 3)`` array of Cartesian coordinates inside ``[0, 2*pi]^3``.
    case :
        Physical parameters.
    time_s :
        Instant of the exact solution that is frozen.

    Returns
    -------
    dict with keys ``velocity`` ``(N,3)``, ``pressure`` ``(N,)``,
    ``velocity_gradient`` ``(N,3,3)`` and ``force`` ``(N,3)``.  ``force`` is the
    manufactured source ``-d u/d t = 2 nu u`` that makes the frozen field an
    exact solution of the steady momentum equation.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if not np.isfinite(points).all():
        raise ValueError("finite points required")
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError(f"invalid time: {time_s!r}")
    x, y, _ = points.T
    amp = case.amplitude
    decay_u = math.exp(-case.decay_exponent * time_s)
    decay_p = math.exp(-case.energy_decay_exponent * time_s)

    velocity = np.column_stack(
        (
            amp * np.cos(x) * np.sin(y) * decay_u,
            -amp * np.sin(x) * np.cos(y) * decay_u,
            np.zeros_like(x),
        )
    )
    pressure = -0.25 * amp**2 * (np.cos(2 * x) + np.cos(2 * y)) * decay_p

    grad = np.zeros((len(x), 3, 3))
    grad[:, 0, 0] = -amp * np.sin(x) * np.sin(y) * decay_u
    grad[:, 0, 1] = amp * np.cos(x) * np.cos(y) * decay_u
    grad[:, 1, 0] = -amp * np.cos(x) * np.cos(y) * decay_u
    grad[:, 1, 1] = amp * np.sin(x) * np.sin(y) * decay_u

    # d u/dt = -2 nu u, so the frozen steady source is f = -du/dt = 2 nu u.
    force = case.decay_exponent * velocity

    return dict(
        velocity=velocity, pressure=pressure, velocity_gradient=grad, force=force
    )


# --------------------------------------------------------------------------- #
# Mesh and solver construction
# --------------------------------------------------------------------------- #
def tgv_mesh(n: int, *, triangulate_faces: bool = True) -> NativeMesh3D:
    """Uniform ``n x n x n`` hexahedral mesh of the periodic box ``[0, 2*pi]^3``.

    The mesh is a single period cell; the periodicity enters through the exact
    Dirichlet velocity trace on the six bounding faces (the box is a valid
    periodic cell, so those face means are mutually compatible and their total
    flux balance is zero).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("positive integer mesh subdivisions required")
    grid = [np.linspace(0.0, BOX, n + 1) for _ in range(3)]
    points = np.array([[x, y, z] for z in grid[2] for y in grid[1] for x in grid[0]])
    index = lambda x, y, z: z * (n + 1) * (n + 1) + y * (n + 1) + x  # noqa: E731
    cells = [
        [
            index(x, y, z),
            index(x + 1, y, z),
            index(x + 1, y + 1, z),
            index(x, y + 1, z),
            index(x, y, z + 1),
            index(x + 1, y, z + 1),
            index(x + 1, y + 1, z + 1),
            index(x, y + 1, z + 1),
        ]
        for z in range(n)
        for y in range(n)
        for x in range(n)
    ]
    return NativeMesh3D.from_cells(
        points, cells, ["hexahedron"] * len(cells), triangulate_faces=triangulate_faces
    )


def tgv_solver(
    n: int,
    case: Optional[TaylorGreenCase] = None,
    *,
    time_s: float = 0.5,
    convection_order: int = 2,
    coupled_linear_method: str = "direct",
    triangulate_faces: bool = True,
) -> tuple:
    """Build the frozen-time steady solver for the Taylor--Green vortex.

    Returns ``(mesh, solver, exact)`` where ``exact`` is the field dict of
    :func:`tgv_fields` at the cell centres, and ``solver`` carries the
    manufactured force ``exact['force']`` and the exact Dirichlet trace.

    ``amplitude``/``viscosity``/``time_s`` fully define the discrete problem; the
    solver is *not* time marched (this backend has no transient integrator).
    """
    if case is None:
        case = TaylorGreenCase()
    mesh = tgv_mesh(n, triangulate_faces=triangulate_faces)
    exact = tgv_fields(mesh.cell_centers, case, time_s)
    boundary = exact_boundary_velocity_means(
        mesh, lambda points: tgv_fields(points, case, time_s)
    )
    fixed = np.repeat((mesh.neighbor < 0)[:, None], 3, axis=1)
    solver = NativeHybridNavierStokes3D(
        mesh,
        fixed,
        boundary,
        viscosity=case.viscosity,
        body_force=exact["force"],
        gauge_value=float(np.mean(exact["pressure"])),
        convection_order=convection_order,
        coupled_linear_method=coupled_linear_method,
    )
    return mesh, solver, exact


def tgv_solve(
    n: int,
    case: Optional[TaylorGreenCase] = None,
    *,
    time_s: float = 0.5,
    max_iterations: int = 25,
    tolerance: float = 1e-10,
    **solver_options,
) -> tuple:
    """Solve one frozen-time Taylor--Green case and require nonlinear convergence.

    Returns ``(mesh, result, exact)``.  A failed Newton iteration raises instead
    of returning a partially converged field, so no metric can be computed from
    an unconverged state.
    """
    mesh, solver, exact = tgv_solver(n, case, time_s=time_s, **solver_options)
    result = solver.solve_newton(max_iterations=max_iterations, tolerance=tolerance)
    if not result["numerically_converged"]:
        raise RuntimeError(
            f"frozen Taylor--Green solve did not converge at n={n}, time_s={time_s}: "
            f"{result.get('newton_failure')}"
        )
    return mesh, result, exact


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def tgv_metrics(mesh, result, exact) -> Dict:
    """Volume-weighted verification metrics for one frozen Taylor--Green solve.

    ``velocity_relative_l2`` is the volume-weighted velocity error relative to
    the exact norm.  ``max_divergence_s_inv`` is the largest cell divergence of
    the accepted face flux.  ``pressure_relative_linf`` is reported for the
    record only: in this closed all-Dirichlet configuration the cell pressure
    retains an odd--even mode that the discrete system leaves undetermined, so
    the metric does not converge and must not be gated.
    """
    volume = np.asarray(mesh.cell_volumes, dtype=float)
    velocity = np.asarray(result["velocity"], dtype=float)
    if velocity.shape != np.asarray(exact["velocity"]).shape:
        raise ValueError("velocity shapes disagree")
    velocity_error = volume_weighted_error(velocity, exact["velocity"], volume)

    p_exact = np.asarray(exact["pressure"], dtype=float)
    p_computed = np.asarray(result["pressure"], dtype=float)
    p_scale = float(np.max(np.abs(p_exact)))
    pressure_error = float(np.max(np.abs(p_computed - p_exact)) / p_scale)

    history = result.get("history") or [{}]
    divergence = float(max(row.get("divergence_max", 0.0) for row in history))

    return dict(
        velocity_relative_l2=float(velocity_error),
        pressure_relative_linf=pressure_error,
        max_divergence_s_inv=divergence,
        velocity_z_max_abs_m_s=float(np.max(np.abs(velocity[:, 2]))),
    )


def observed_order(coarse: Dict, fine: Dict) -> float:
    """Observed order of ``velocity_relative_l2`` between two resolutions."""
    h_coarse, h_fine = coarse["spacing_m"], fine["spacing_m"]
    e_coarse, e_fine = coarse["velocity_relative_l2"], fine["velocity_relative_l2"]
    if h_coarse <= h_fine or e_fine <= 0.0 or e_coarse <= 0.0:
        raise ValueError("coarse/fine resolutions and nonzero errors required")
    return float(math.log(e_coarse / e_fine) / math.log(h_coarse / h_fine))


def run_study(
    n_list: List[int],
    case: Optional[TaylorGreenCase] = None,
    *,
    time_s: float = 0.5,
) -> List[Dict]:
    """Run the frozen-time spatial refinement study over ``n_list``.

    Returns one metric dict per resolution, augmented with ``n``,
    ``resolution``, ``spacing_m`` and ``time_s``.
    """
    if case is None:
        case = TaylorGreenCase()
    results = []
    for n in n_list:
        mesh, result, exact = tgv_solve(n, case, time_s=time_s)
        metrics = tgv_metrics(mesh, result, exact)
        metrics["n"] = int(n)
        metrics["resolution"] = f"{n}x{n}x{n}"
        metrics["spacing_m"] = float(BOX / n)
        metrics["time_s"] = float(time_s)
        results.append(metrics)
    return results
