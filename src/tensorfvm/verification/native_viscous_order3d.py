"""Re-audit of the ``NativeViscous3D`` Dirichlet wall closure with two metrics.

Background
----------
A previous audit (mission m2530, platform report "NativeViscous3D 精度阶审计")
concluded that the wall normal derivative of this closure is first order and
that a Dirichlet diffusion problem built from ``scalar_diffusion_matrix`` shows
no measurable convergence order.  That harness was never merged into this
repository, so the verdict carries no runnable evidence here.  This module
rebuilds the pieces needed to re-test both verdicts on orthogonal, anisotropic
and warped unit cubes with analytic manufactured fields, and deliberately
reports *two* metrics because the conclusion depends on the metric:

``wall_flux``
    ``||q_h - q_exact||_2 / ||q_exact||_2`` over the Dirichlet faces with
    ``q = mu grad(u).S``.  A norm metric cannot be inflated by a vanishing
    denominator.
``face``
    ``mean_f |dudn_h - dudn| / |dudn|`` on the same faces, i.e. the per-face
    mean relative error of the normal derivative.  This is the metric of a
    per-face table and it is dominated by faces whose exact normal derivative
    is near zero; ``face_floored`` repeats it with the denominator floored at
    10% of the peak normal derivative to expose that sensitivity.

``constant_mode``
    ``||A 1|| / (||A||_max sqrt(N))`` for the assembled fixed-value operator: the
    Dirichlet datum of this closure enters only through the right-hand side, so
    the matrix itself describes the zero-jump (mirror) operator and its constant
    mode is only weakly restrained.  This is the quantity that decides whether a
    fixed-value solve built from ``scalar_diffusion_matrix`` can be trusted.

Nothing here changes a solver; it is read-only verification code.
"""

from __future__ import annotations

import numpy as np

from ..native_mesh3d import NativeMesh3D
from ..native_viscous3d import NativeViscous3D


def cube_mesh(divisions, *, warp=0.0, triangulate_faces=True):
    """Unit cube structured hexahedral mesh, optionally warped in the interior.

    ``warp`` is the peak vertex displacement as a fraction of the local cell
    size; boundary vertices stay on the cube so the Dirichlet data remain exact.
    """
    divisions = tuple(int(v) for v in divisions)
    if len(divisions) != 3 or any(v < 1 for v in divisions):
        raise ValueError("three positive integer subdivisions required")
    nx, ny, nz = divisions
    gx, gy, gz = [np.linspace(0.0, 1.0, v + 1) for v in divisions]
    points = np.array([[x, y, z] for z in gz for y in gy for x in gx])
    if warp:
        h = 1.0 / max(divisions)
        interior = np.all((points > 0.0) & (points < 1.0), axis=1)
        phase = np.pi * points[:, 0] * points[:, 1] * points[:, 2]
        displacement = warp * h * np.stack(
            [np.sin(phase), 0.5 * np.sin(2 * phase), -0.5 * np.sin(phase)], axis=1
        )
        points = np.where(interior[:, None], points + displacement, points)
    def index(x, y, z):
        return z * (nx + 1) * (ny + 1) + y * (nx + 1) + x
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
        for z in range(nz)
        for y in range(ny)
        for x in range(nx)
    ]
    return NativeMesh3D.from_cells(
        points, cells, ["hexahedron"] * len(cells), triangulate_faces=triangulate_faces
    )


def scalar_field(points, mode="trig3d"):
    """Manufactured scalar field and its analytic gradient.

    ``trig3d``  ``sin(pi x) cos(pi y / 2) exp(z)``: every direction carries
                curvature, the shape used by the m2530 audit.
    ``trig1d``  ``sin(pi x)``: the one-dimensional negative control, exact in y
                and z, which isolates the x closure.
    """
    p = np.asarray(points, dtype=float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    if mode == "trig1d":
        return np.sin(np.pi * x), np.stack(
            [np.pi * np.cos(np.pi * x), np.zeros_like(x), np.zeros_like(x)], axis=1
        )
    if mode == "trig3d":
        values = np.sin(np.pi * x) * np.cos(np.pi * y / 2) * np.exp(z)
        gradient = np.stack(
            [
                np.pi * np.cos(np.pi * x) * np.cos(np.pi * y / 2) * np.exp(z),
                -(np.pi / 2) * np.sin(np.pi * x) * np.sin(np.pi * y / 2) * np.exp(z),
                np.sin(np.pi * x) * np.cos(np.pi * y / 2) * np.exp(z),
            ],
            axis=1,
        )
        return values, gradient
    raise ValueError("unknown manufactured field mode")


def wall_closure_metrics(mesh, viscosity=1.3, *, mode="trig3d"):
    """Closure metrics on the Dirichlet faces for one mesh and manufactured field."""
    boundary = mesh.neighbor < 0
    if not boundary.any():
        raise ValueError("mesh has no boundary faces")
    cell_values, _ = scalar_field(mesh.cell_centers, mode)
    boundary_values, boundary_gradient = scalar_field(mesh.face_centers, mode)
    op = NativeViscous3D(mesh, boundary, viscosity)
    exact_normal = np.einsum("ij,ij->i", boundary_gradient, op.normal)
    exact_flux = viscosity * exact_normal * op.area
    computed_flux = op.scalar_flux(cell_values, 0.0, boundary_values)
    residual = computed_flux[boundary] - exact_flux[boundary]
    wall_flux = float(np.linalg.norm(residual) / np.linalg.norm(exact_flux[boundary]))
    computed_normal = computed_flux[boundary] / (viscosity * op.area[boundary])
    exact_normal = exact_normal[boundary]
    magnitude = np.abs(exact_normal)
    live = magnitude > 0.0
    face = float(np.mean(np.abs(computed_normal[live] - exact_normal[live]) / magnitude[live]))
    floor = 0.1 * magnitude.max()
    face_floored = float(
        np.mean(np.abs(computed_normal[live] - exact_normal[live]) / np.maximum(magnitude[live], floor))
    )
    matrix = (op.divergence_matrix @ op.scalar_flux_matrix).tocsr()
    constant_mode = float(
        np.linalg.norm(matrix @ np.ones(op.N)) / (np.abs(matrix).max() * np.sqrt(op.N))
    )
    return dict(
        divisions=list(mesh.divisions if hasattr(mesh, "divisions") else ()),
        ncells=int(len(mesh.cells)),
        wall_flux=wall_flux,
        face=face,
        face_floored=face_floored,
        constant_mode=constant_mode,
        max_abs_flux_residual=float(np.abs(residual).max()),
        exact_peak_normal=float(magnitude.max()),
    )


def affine_exactness(mesh, viscosity=1.3):
    """Relative flux error of the closure on an affine field (must be machine zero)."""
    boundary = mesh.neighbor < 0
    op = NativeViscous3D(mesh, boundary, viscosity)
    gradient = np.array([0.7, -0.3, 0.2])
    base = 1.2
    cell_values = base + mesh.cell_centers @ gradient
    boundary_values = base + mesh.face_centers @ gradient
    exact_flux = viscosity * (mesh.area_vectors @ gradient)
    computed = op.scalar_flux(cell_values, 0.0, boundary_values)
    scale = np.linalg.norm(exact_flux[boundary])
    return float(np.abs(computed[boundary] - exact_flux[boundary]).max() / scale)


def family(divisions_list, *, warp=0.0, mode="trig3d", viscosity=1.3):
    """Metrics over a refinement family (ascending subdivisions)."""
    rows = []
    for divisions in divisions_list:
        mesh = cube_mesh(divisions, warp=warp)
        row = wall_closure_metrics(mesh, viscosity, mode=mode)
        row["divisions"] = list(divisions)
        row["warp"] = warp
        row["mode"] = mode
        row["affine_exactness"] = affine_exactness(mesh, viscosity)
        rows.append(row)
    return rows


def h_of(row):
    return 1.0 / max(row["divisions"])


def orders(rows, key):
    """Observed order of ``key`` under the refinement actually used."""
    return [
        float(np.log2(previous[key] / current[key]) / np.log2(h_of(previous) / h_of(current)))
        for previous, current in zip(rows, rows[1:])
    ]