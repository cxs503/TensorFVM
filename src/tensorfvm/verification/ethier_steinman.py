"""Ethier--Steinman 3-D manufactured verification for the native hybrid NS backend.

Reference
---------
D. L. Ethier and D. A. Steinman, "Exact fully 3-D Navier--Stokes solutions for
benchmarking", *International Journal for Numerical Methods in Fluids* **19**
(1994) 369--375, DOI ``10.1002/fld.1650190502``.

Exact field (rho = 1, ``a`` and ``d`` the two length-scale exponents)::

    u_i = -a [ e^{a x_i} sin(a (x_j + x_k)) + e^{a x_k} cos(a (x_i + x_j)) ]
          * e^{-d^2 t}       with (i, j, k) cyclic over (x, y, z)

The velocity is divergence-free identically, is an exact Laplacian
eigenfunction ``lap(u) = -a^2 u``, and satisfies the *unforced* incompressible
Navier--Stokes equations for

    nu = d^2 / a^2        (so Re = U L / nu = 1 / nu for U = a, L = 1/a),

which is the parameter relation the reference is built on.  All of it is
re-derived here from the closed form (``reference_audit``) instead of being taken
on trust: analytic divergence, the ``lap(u) = -a^2 u`` identity, the curl-free
property of the momentum residual (which holds *iff* ``nu = d^2/a^2``), and the
atom-algebra derivatives against finite differences.

Re is fixed by the viscosity alone and the exponent ``a`` only sets the
wavenumber, so a large-Re case can be posed at a wavenumber whose *cell*
Reynolds number ``a h / nu`` is still resolvable:
``Reynolds number = 1/nu`` regardless of ``a``.

Frozen-time formulation
-----------------------
:class:`tensorfvm.native_hybrid_ns3d.NativeHybridNavierStokes3D` is a *steady*
solver (the same restriction documented in
:mod:`tensorfvm.verification.native_taylor_green`).  Freezing the exact field at
one instant and moving the time term to the right-hand side gives the steady
problem with manufactured body force

    (u.grad) u = -grad(p) + nu lap(u) + f,
    f = du/dt + (u.grad) u - nu lap(u) = -d^2 u + (u.grad) u - nu lap(u).

``f`` is evaluated from closed-form derivatives only: every component of the
exact velocity is a sum of atoms ``exp(p.r) * sin(q.r)`` / ``exp(p.r) *
cos(q.r)``, whose gradient and Laplacian are exact:

    grad [e^{p.r} T(q.r)] = e^{p.r} (p T(q.r) + q T'(q.r))
    lap  [e^{p.r} T(q.r)] = e^{p.r} [ (|p|^2 - |q|^2) T(q.r) + 2 p.q T'(q.r) ]

No finite difference enters the reference field, the boundary trace or the
source term.  (The pressure is not used as an input: at ``nu = d^2/a^2`` the
manufactured force ``f`` is a pure gradient ``-grad(p)``, which is checked
numerically, so the closed-form pressure is never needed to state the problem.
The cell pressure of this closed all-Dirichlet configuration carries a gauge /
odd--even mode and is reported for the record only.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..native_hybrid_ns3d import NativeHybridNavierStokes3D
from ..native_manufactured3d import exact_boundary_velocity_means, volume_weighted_error
from ..native_mesh3d import NativeMesh3D

# --------------------------------------------------------------------------- #
# Exact exp/trig atom algebra (closed-form values, gradients and Laplacians)
# --------------------------------------------------------------------------- #
# One atom is ``coef * exp(p . r) * T(q . r)`` with ``T`` in {sin, cos}.
Atom = Tuple[float, Tuple[float, float, float], Tuple[float, float, float], str]


def _atom_value(coef, p, q, kind, points):
    argument = points @ q
    trig = np.sin(argument) if kind == "sin" else np.cos(argument)
    return coef * np.exp(points @ p) * trig


def _atom_gradient(coef, p, q, kind, points):
    exponent = np.exp(points @ p)
    argument = points @ q
    if kind == "sin":
        primary, secondary = np.sin(argument), np.cos(argument)
    else:
        primary, secondary = np.cos(argument), -np.sin(argument)
    return coef * exponent[:, None] * (primary[:, None] * p + secondary[:, None] * q)


def _atom_laplacian(coef, p, q, kind, points):
    exponent = np.exp(points @ p)
    argument = points @ q
    norm_p, norm_q, cross = float(p @ p), float(q @ q), float(p @ q)
    if kind == "sin":
        primary, secondary = np.sin(argument), np.cos(argument)
    else:
        primary, secondary = np.cos(argument), -np.sin(argument)
    return coef * exponent * ((norm_p - norm_q) * primary + 2.0 * cross * secondary)


def _as_points(points) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if not np.isfinite(array).all():
        raise ValueError("finite points required")
    return array


def field_value(atoms: Sequence[Atom], points) -> np.ndarray:
    """Evaluate ``sum atoms`` at ``points``."""
    points = _as_points(points)
    total = np.zeros(len(points))
    for coef, p, q, kind in atoms:
        total += _atom_value(coef, np.asarray(p, float), np.asarray(q, float), kind, points)
    return total


def field_gradient(atoms: Sequence[Atom], points) -> np.ndarray:
    """Exact gradient ``(N, 3)`` of ``sum atoms``."""
    points = _as_points(points)
    total = np.zeros((len(points), 3))
    for coef, p, q, kind in atoms:
        total += _atom_gradient(
            coef, np.asarray(p, float), np.asarray(q, float), kind, points
        )
    return total


def field_laplacian(atoms: Sequence[Atom], points) -> np.ndarray:
    """Exact Laplacian ``(N,)`` of ``sum atoms``."""
    points = _as_points(points)
    total = np.zeros(len(points))
    for coef, p, q, kind in atoms:
        total += _atom_laplacian(
            coef, np.asarray(p, float), np.asarray(q, float), kind, points
        )
    return total


# --------------------------------------------------------------------------- #
# Case definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EthierSteinmanCase:
    """Parameters of one Ethier--Steinman solution.

    ``a``/``d`` are the exponents of the closed form and ``viscosity`` the
    kinematic viscosity.  ``nu = d^2 / a^2`` is the *unforced* relation of the
    reference; when it does not hold the same velocity field is retained as a
    manufactured solution (the momentum residual then supplies a genuine body
    force instead of a pure pressure gradient), which is reported, not hidden.
    """

    a: float = 1.0
    d: float = 1.0
    viscosity: float = 1.0
    density: float = 1.0
    half_width: float = 1.0

    def __post_init__(self) -> None:
        for name in ("a", "d", "viscosity", "density", "half_width"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}: {value!r}")
        if self.density != 1.0:
            raise ValueError(
                "the closed form is written for rho = 1; generalise the case "
                "before using another density"
            )

    @property
    def unforced_viscosity(self) -> float:
        """``d^2 / a^2``: the viscosity at which the field is unforced NS."""
        return self.d**2 / self.a**2

    @property
    def is_unforced(self) -> bool:
        """True when the closed form satisfies the unforced NS equations."""
        return abs(self.viscosity - self.unforced_viscosity) <= 1e-12 * max(
            1.0, self.unforced_viscosity
        )

    @property
    def decay_rate(self) -> float:
        """Time-decay rate ``d^2`` of the velocity."""
        return self.d**2

    @property
    def reynolds_number(self) -> float:
        """``U L / nu`` with ``U = a`` and ``L = 1/a``, i.e. ``1 / nu``."""
        return 1.0 / self.viscosity

    @property
    def box(self) -> Tuple[float, float]:
        """Coordinate interval of the cubic domain on each axis."""
        return (-self.half_width, self.half_width)


def _velocity_atoms(case: EthierSteinmanCase, time_s: float) -> Tuple[List[Atom], ...]:
    """Atom decomposition of the three exact velocity components.

    ``u_i = -a [ e^{a x_i} sin(a (x_j + x_k)) + e^{a x_k} cos(a (x_i + x_j)) ]``
    with ``(i, j, k)`` cyclic.  The growth exponent equals the phase coefficient
    ``a``, which is exactly what makes ``div(u) = 0`` and ``lap(u) = -a^2 u``
    hold for every ``a``.
    """
    a, d = case.a, case.d
    decay = math.exp(-case.decay_rate * time_s)
    coef = -a * decay
    u = [
        (coef, (a, 0.0, 0.0), (0.0, a, a), "sin"),
        (coef, (0.0, 0.0, a), (a, a, 0.0), "cos"),
    ]
    v = [
        (coef, (0.0, a, 0.0), (a, 0.0, a), "sin"),
        (coef, (a, 0.0, 0.0), (0.0, a, a), "cos"),
    ]
    w = [
        (coef, (0.0, 0.0, a), (a, a, 0.0), "sin"),
        (coef, (0.0, a, 0.0), (a, 0.0, a), "cos"),
    ]
    return tuple([u, v, w])


def es_fields(points, case: EthierSteinmanCase, time_s: float = 0.0) -> Dict:
    """Exact velocity, its derivatives and the frozen-time force at ``points``.

    Returns a dict with

    ``velocity`` ``(N,3)``
        exact field at ``time_s``,
    ``velocity_gradient`` ``(N,3,3)`` with ``[..., j, i] = d u_i / d x_j``,
    ``velocity_laplacian`` ``(N,3)``,
    ``acceleration`` ``(N,3)`` -- the material acceleration
        ``du/dt + (u.grad)u``,
    ``force`` ``(N,3)`` -- the manufactured body force of the frozen steady
        problem ``du/dt + (u.grad)u - nu lap(u)`` (per unit mass, ``rho = 1``),
    ``decay`` -- ``exp(-d^2 t)``.
    """
    points = _as_points(points)
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError(f"invalid time: {time_s!r}")
    atoms = _velocity_atoms(case, time_s)
    velocity = np.column_stack([field_value(component, points) for component in atoms])
    gradient = np.stack([field_gradient(component, points) for component in atoms], axis=2)
    laplacian = np.column_stack([field_laplacian(component, points) for component in atoms])
    convection = np.einsum("nj,nji->ni", velocity, gradient)
    time_derivative = -case.decay_rate * velocity
    acceleration = time_derivative + convection
    force = acceleration - case.viscosity * laplacian
    return dict(
        velocity=velocity,
        velocity_gradient=gradient,
        velocity_laplacian=laplacian,
        acceleration=acceleration,
        force=force,
        decay=math.exp(-case.decay_rate * time_s),
    )


def es_field_function(case: EthierSteinmanCase, time_s: float = 0.0):
    """Callable ``points -> es_fields(points, case, time_s)`` for boundary traces."""

    def field(points):
        return es_fields(points, case, time_s)

    return field


# --------------------------------------------------------------------------- #
# Reference self-audit (the reference judges the solver only after it passes)
# --------------------------------------------------------------------------- #
def reference_audit(
    case: Optional[EthierSteinmanCase] = None,
    *,
    samples: int = 6,
    seed: int = 0,
    step: float = 1e-6,
) -> Dict[str, float]:
    """Check the closed form itself before it is allowed to judge a solver.

    * ``analytic_divergence_max`` -- closed-form ``div(u)`` at random points.
    * ``finite_difference_divergence_max`` -- independent central difference.
    * ``gradient_max_error`` / ``laplacian_max_error`` -- the atom algebra
      against central differences of the closed form.
    * ``unforced_momentum_curl_max`` -- curl of the momentum residual
      ``-rho (du/dt + (u.grad)u) + mu lap(u)``.  It vanishes iff the field is an
      unforced Navier--Stokes solution, i.e. iff ``nu = d^2/a^2``; the parameter
      relation of the reference is therefore re-derived, not assumed.
    * ``force_curl_max`` -- ``max|curl(f)|`` of the manufactured force; it
      vanishes for an unforced case because there ``f`` is a pure pressure
      gradient.
    """
    if case is None:
        case = EthierSteinmanCase()
    rng = np.random.default_rng(seed)
    span = case.half_width
    points = rng.uniform(-0.7 * span, 0.7 * span, (samples, 3))
    exact = es_fields(points, case, 0.0)
    gradient, laplacian = exact["velocity_gradient"], exact["velocity_laplacian"]
    atoms = _velocity_atoms(case, 0.0)
    offsets = [np.eye(3)[axis] * step for axis in range(3)]
    # A second difference needs a longer step: its round-off is eps/step^2.
    wide = [np.eye(3)[axis] * (100.0 * step) for axis in range(3)]

    def curl_of(vector_function):
        value = np.zeros((samples, 3))
        for axis in range(3):
            first, second = (axis + 1) % 3, (axis + 2) % 3
            value[:, axis] = (
                vector_function(points + offsets[second])[:, first]
                - vector_function(points - offsets[second])[:, first]
            ) / (2 * step) - (
                vector_function(points + offsets[first])[:, second]
                - vector_function(points - offsets[first])[:, second]
            ) / (2 * step)
        return value

    momentum_residual = lambda pts: (  # noqa: E731
        -case.density * es_fields(pts, case, 0.0)["acceleration"]
        + case.density * case.viscosity * es_fields(pts, case, 0.0)["velocity_laplacian"]
    )
    force = lambda pts: es_fields(pts, case, 0.0)["force"]  # noqa: E731

    gradient_error = laplacian_error = 0.0
    divergence_fd = np.zeros(samples)
    for index in range(3):
        component = lambda pts, i=index: field_value(atoms[i], pts)  # noqa: E731
        second = np.zeros(samples)
        for axis, (offset, broad) in enumerate(zip(offsets, wide)):
            first = (component(points + offset) - component(points - offset)) / (2 * step)
            gradient_error = max(
                gradient_error, float(np.abs(first - gradient[:, axis, index]).max())
            )
            second += (
                component(points + broad)
                - 2 * component(points)
                + component(points - broad)
            ) / (100.0 * step) ** 2
        laplacian_error = max(laplacian_error, float(np.abs(second - laplacian[:, index]).max()))
        divergence_fd += (
            (component(points + offsets[index]) - component(points - offsets[index]))
            / (2 * step)
        )

    return dict(
        analytic_divergence_max=float(
            np.abs(gradient[:, 0, 0] + gradient[:, 1, 1] + gradient[:, 2, 2]).max()
        ),
        finite_difference_divergence_max=float(np.abs(divergence_fd).max()),
        gradient_max_error=gradient_error,
        laplacian_max_error=laplacian_error,
        unforced_momentum_curl_max=float(np.abs(curl_of(momentum_residual)).max()),
        force_curl_max=float(np.abs(curl_of(force)).max()),
        unforced_viscosity=float(case.unforced_viscosity),
        viscosity=float(case.viscosity),
        velocity_max_abs=float(np.abs(exact["velocity"]).max()),
        force_max_abs=float(np.abs(exact["force"]).max()),
    )


# --------------------------------------------------------------------------- #
# Mesh and solver construction
# --------------------------------------------------------------------------- #
def es_mesh(n: int, *, half_width: float = 1.0, triangulate_faces: bool = True) -> NativeMesh3D:
    """Uniform ``n x n x n`` hexahedral mesh of ``[-half_width, half_width]^3``."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("positive integer mesh subdivisions required")
    if not math.isfinite(half_width) or half_width <= 0:
        raise ValueError("positive half width required")
    grid = [np.linspace(-half_width, half_width, n + 1) for _ in range(3)]
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


def es_solver(
    n: int,
    case: Optional[EthierSteinmanCase] = None,
    *,
    time_s: float = 0.0,
    convection_order: int = 2,
    coupled_linear_method: str = "direct",
    triangulate_faces: bool = True,
) -> Tuple[NativeMesh3D, NativeHybridNavierStokes3D, Dict]:
    """Build the frozen-time steady solver for one Ethier--Steinman case.

    Returns ``(mesh, solver, exact)``.  The manufactured force is the cell-sampled
    closed-form residual and the Dirichlet trace is the exact area-mean boundary
    velocity (order-8 Duffy quadrature in
    :func:`tensorfvm.native_manufactured3d.exact_boundary_velocity_means`).
    """
    if case is None:
        case = EthierSteinmanCase()
    mesh = es_mesh(n, half_width=case.half_width, triangulate_faces=triangulate_faces)
    field = es_field_function(case, time_s)
    exact = field(mesh.cell_centers)
    boundary = exact_boundary_velocity_means(mesh, field)
    fixed = np.repeat((mesh.neighbor < 0)[:, None], 3, axis=1)
    solver = NativeHybridNavierStokes3D(
        mesh,
        fixed,
        boundary,
        viscosity=case.viscosity,
        density=case.density,
        body_force=exact["force"],
        convection_order=convection_order,
        coupled_linear_method=coupled_linear_method,
    )
    return mesh, solver, exact


def es_solve(
    n: int,
    case: Optional[EthierSteinmanCase] = None,
    *,
    time_s: float = 0.0,
    max_iterations: int = 25,
    tolerance: float = 1e-10,
    **solver_options,
) -> Tuple[NativeMesh3D, Dict, Dict]:
    """Solve one frozen-time case and require nonlinear convergence.

    A failed Newton iteration raises instead of returning a partially converged
    field, so no metric can ever be computed from an unconverged state.
    """
    mesh, solver, exact = es_solver(n, case, time_s=time_s, **solver_options)
    result = solver.solve_newton(max_iterations=max_iterations, tolerance=tolerance)
    if not result["numerically_converged"]:
        raise RuntimeError(
            f"frozen Ethier--Steinman solve did not converge at n={n}, "
            f"Re={case.reynolds_number:.6g}: {result.get('newton_failure')}"
        )
    return mesh, result, exact


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def es_metrics(mesh, result, exact) -> Dict:
    """Verification metrics of one frozen-time solve.

    ``velocity_relative_linf`` is the gated metric: the largest component error
    over all cells divided by the largest exact component value, i.e.
    ``max|u_h - u| / max|u|`` with the maxima taken over every cell and all
    three components.  ``velocity_relative_l2`` is the volume-weighted L2
    counterpart.  ``pressure_relative_l2`` (mean removed) is reported for the
    record only -- the closed all-Dirichlet configuration leaves a cell-pressure
    gauge/odd--even mode undetermined, exactly as documented for the frozen
    Taylor--Green reference.
    """
    volume = np.asarray(mesh.cell_volumes, dtype=float)
    velocity = np.asarray(result["velocity"], dtype=float)
    exact_velocity = np.asarray(exact["velocity"], dtype=float)
    if velocity.shape != exact_velocity.shape:
        raise ValueError("velocity shapes disagree")
    scale = float(np.max(np.abs(exact_velocity)))
    if scale <= 0:
        raise ValueError("degenerate exact velocity scale")
    linf = float(np.max(np.abs(velocity - exact_velocity)) / scale)
    l2 = float(volume_weighted_error(velocity, exact_velocity, volume))
    pressure = np.asarray(result["pressure"], dtype=float)
    pressure_gauge = float(np.mean(pressure))
    pressure_scale = float(np.max(np.abs(exact["force"])))
    pressure_record = float(
        np.sqrt(np.sum(volume * (pressure - pressure_gauge) ** 2) / np.sum(volume))
        / pressure_scale
    )
    history = result.get("history") or [{}]
    return dict(
        velocity_relative_linf=linf,
        velocity_relative_l2=l2,
        exact_velocity_max_abs=scale,
        absolute_velocity_linf=float(np.max(np.abs(velocity - exact_velocity))),
        pressure_mean_removed_l2_over_force_scale=pressure_record,
        max_divergence_s_inv=float(max(row.get("divergence_max", 0.0) for row in history)),
    )


def observed_order(coarse: Dict, fine: Dict) -> float:
    """Observed order of ``velocity_relative_linf`` between two resolutions."""
    h_coarse, h_fine = coarse["spacing_m"], fine["spacing_m"]
    e_coarse, e_fine = coarse["velocity_relative_linf"], fine["velocity_relative_linf"]
    if h_coarse <= h_fine or e_fine <= 0.0 or e_coarse <= 0.0:
        raise ValueError("coarse/fine resolutions and nonzero errors required")
    return float(math.log(e_coarse / e_fine) / math.log(h_coarse / h_fine))


def run_study(
    n_list: Sequence[int],
    case: Optional[EthierSteinmanCase] = None,
    *,
    time_s: float = 0.0,
    **solve_options,
) -> List[Dict]:
    """Run the frozen-time spatial refinement ladder over ``n_list``."""
    if case is None:
        case = EthierSteinmanCase()
    results: List[Dict] = []
    for n in n_list:
        mesh, result, exact = es_solve(n, case, time_s=time_s, **solve_options)
        metrics = es_metrics(mesh, result, exact)
        metrics["n"] = int(n)
        metrics["resolution"] = f"{n}x{n}x{n}"
        metrics["cells"] = int(len(mesh.cells))
        metrics["spacing_m"] = float(2.0 * case.half_width / n)
        metrics["time_s"] = float(time_s)
        metrics["reynolds_number"] = float(case.reynolds_number)
        results.append(metrics)
    return results