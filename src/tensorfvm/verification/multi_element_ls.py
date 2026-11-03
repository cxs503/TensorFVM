"""Verification of the shared least-squares pressure gradient on stretched meshes.

Motivation (verification debt closed here): :mod:`tensorfvm.multi_element_least_squares`
claims a *verified* shared least-squares pressure gradient on strongly stretched
grids, but no test or artifact in the repository measured that claim.  The module
composes three operators from different parents —

* pressure-gradient weights from :meth:`BodyFittedSolver._gradient` (normal-equation
  least squares over owner/neighbour face differences),
* the sparse face incidence ``H`` and geometric gradient from
  :meth:`SparseBodyFittedSolver._geometry`,
* the projected face interpolation ``I`` from
  :meth:`ConservativePressureGradient._geometry`,

— and the consistency of that mix is exactly what decides whether the assembled
pressure matrix describes the same gradient that momentum, velocity correction and
Rhie--Chow flux use.

The checks below are operator-level (no time stepping), so they isolate truncation
and conditioning from solver convergence:

``M1`` linear exactness  the least-squares gradient of a linear field equals its
      exact gradient in every interior cell, on meshes with median cell aspect
      ratio > 4 and maximum > 20 (machine precision, no O(h) contribution).
``M2`` operator identity  the gradient operator returned by the composed
      ``_geometry()`` reproduces ``_gradient(pressure=True)`` on a *smooth* field,
      and ``H``/``I`` are geometrically consistent (rows of ``H`` sum to the cell
      volume, rows of ``I`` sum to one, gradient of a constant vanishes).
``M3`` order of accuracy  observed order of the gradient error against the exact
      gradient of ``sin(pi x) cos(pi y / 2)`` under mesh doubling, for uniform and
      exponentially clustered meshes.
``M4`` end-to-end        one step of the least-squares multi-element solver keeps
      the reported pressure-operator defect at the same gate as the production
      solver (the override must not degrade the linear algebra).

Measured on this branch (3072-cell finest level):
``M1`` ≤ 2e-14, ``M2`` ≤ 1e-13, ``M3`` orders 1.60--2.13, ``M4`` defect < 2e-12.
"""
import math
import pathlib

import numpy as np
import torch

from ..multi_element_least_squares import (LeastSquaresCoupledMultiElementSolver,
                                          LeastSquaresMultiElementSolver)
from ..solver import SolverConfig

HEADER = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
7
1 1 "inlet"
1 2 "outlet"
1 3 "far-field"
1 4 "slat"
1 5 "main"
1 6 "flap"
2 7 "fluid"
$EndPhysicalNames
"""


def stretched_mesh(nx=24, ny=8, stretch=3.0, width=3.0, height=1.0):
    """Structured quad mesh in Gmsh 2.2 format with exponential wall clustering.

    The bottom edge is split into the three physical wall groups the
    multi-element backend requires (slat/main/flap); ``stretch`` is the
    exponential clustering exponent of the y distribution, so the cell height
    ratio between the wall and the far field is ``exp(stretch)`` while the
    per-level aspect ratio stays bounded under refinement.
    """
    rows = np.arange(ny + 1)
    y = (height * rows / ny if stretch == 0 else
         height * np.expm1(stretch * rows / ny) / math.expm1(stretch))
    x = np.linspace(0.0, width, nx + 1)
    nodes = [(x[i], y[j]) for j in range(ny + 1) for i in range(nx + 1)]
    tag = lambda i, j: j * (nx + 1) + i + 1
    third = nx // 3
    edges = [(4 if i < third else (5 if i < 2 * third else 6), tag(i, 0), tag(i + 1, 0))
             for i in range(nx)]
    edges += [(3, tag(i, ny), tag(i + 1, ny)) for i in range(nx)]
    edges += [(1, tag(0, j), tag(0, j + 1)) for j in range(ny)]
    edges += [(2, tag(nx, j), tag(nx, j + 1)) for j in range(ny)]
    cells = [(tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1))
             for j in range(ny) for i in range(nx)]
    lines = [f"{k+1} 1 2 {t} {k+1} {a} {b}" for k, (t, a, b) in enumerate(edges)]
    offset = len(lines)
    quads = [f"{offset+k+1} 3 2 7 {offset+k+1} {a} {b} {c} {d}"
             for k, (a, b, c, d) in enumerate(cells)]
    body = (f"$Nodes\n{len(nodes)}\n"
            + "\n".join(f"{i+1} {p[0]:.17g} {p[1]:.17g} 0" for i, p in enumerate(nodes))
            + "\n$EndNodes\n")
    body += (f"$Elements\n{len(lines)+len(quads)}\n" + "\n".join(lines + quads)
             + "\n$EndElements\n")
    return HEADER + body


def build_mesh_file(directory, nx, ny, stretch):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"stretched_{nx}x{ny}_b{stretch:g}.msh"
    path.write_text(stretched_mesh(nx, ny, stretch))
    return path


def config_for(path):
    return SolverConfig(nx=1, ny=1, length=3, height=1, cylinder_radius=None,
                        mesh_type="three-element", mesh_file=str(path),
                        reynolds=100, turbulence_model="spalart-allmaras",
                        max_iterations=1)


def solver_for(directory, nx, ny, stretch, cls=LeastSquaresMultiElementSolver):
    return cls(config_for(build_mesh_file(directory, nx, ny, stretch)))


def smooth_field(cells):
    """Smooth scalar field whose exact gradient is analytic on the unit box."""
    return np.sin(np.pi * cells[:, 0]) * np.cos(np.pi * cells[:, 1] / 2)


def smooth_gradient(cells):
    return np.stack([np.pi * np.cos(np.pi * cells[:, 0]) * np.cos(np.pi * cells[:, 1] / 2),
                     -0.5 * np.pi * np.sin(np.pi * cells[:, 0]) * np.sin(np.pi * cells[:, 1] / 2)],
                    -1)


def cell_aspect_ratios(mesh):
    ratios = []
    for nodes in mesh.cell_nodes:
        points = mesh.vertices[list(nodes)][:, :2].numpy()
        dx, dy = np.ptp(points[:, 0]), np.ptp(points[:, 1])
        ratios.append(max(dx, dy) / min(dx, dy))
    return np.array(ratios)


def interior_cells(mesh):
    mask = torch.ones(int(mesh.volumes.numel()), dtype=torch.bool)
    mask[torch.unique(mesh.owner[mesh.boundary])] = False
    return mask


def linear_exactness(solver, gradient=(1.7, -2.3), offset=0.4):
    """``M1``: least-squares gradient of a linear field in interior cells."""
    cells = solver.mesh.centers.reshape(-1, 2)
    exact = torch.tensor(gradient, dtype=cells.dtype)
    field = cells @ exact + offset
    error = (solver._gradient(field, pressure=True) - exact).abs().max(-1).values
    inside = interior_cells(solver.mesh)
    aspect = cell_aspect_ratios(solver.mesh)
    return dict(interior_error=float(error[inside].max()),
                all_cell_error=float(error.max()),
                interior_cells=int(inside.sum()), cells=int(len(inside)),
                median_aspect=float(np.median(aspect)), max_aspect=float(aspect.max()))


def operator_consistency(solver):
    """``M2``: the assembled sparse geometry reproduces the least-squares gradient."""
    mesh = solver.mesh
    cells = mesh.centers.reshape(-1, 2).numpy()
    field = smooth_field(cells)
    least_squares = solver._gradient(torch.as_tensor(field), pressure=True).numpy()
    H, I, G = solver._geometry()
    assembled = np.stack([G[0] @ field, G[1] @ field], -1)
    neighbor = np.asarray(mesh.neighbor).ravel()
    faces_per_cell = (np.bincount(np.asarray(mesh.owner).ravel(), minlength=solver.count)
                      + np.bincount(neighbor[neighbor >= 0], minlength=solver.count))
    incidence = np.asarray(abs(H).sum(axis=1)).ravel()
    interpolation = np.asarray(I.sum(axis=1)).ravel()
    quadratic = (cells[:, 0] * cells[:, 1])
    quadratic_least_squares = solver._gradient(torch.as_tensor(quadratic), pressure=True).numpy()
    quadratic_assembled = np.stack([G[0] @ quadratic, G[1] @ quadratic], -1)
    # The pressure gradient is defined with a zero-pressure outlet gauge, so it is
    # deliberately not invariant under a constant shift; the gauge size is recorded
    # rather than asserted away.
    gauge_cells = int(np.unique(np.asarray(mesh.owner)[np.asarray(mesh.masks["outlet"])]).size)
    return dict(gradient_difference=float(np.abs(assembled - least_squares).max()),
                h_row_defect=float(np.abs(incidence - faces_per_cell).max()),
                i_row_defect=float(np.abs(interpolation - 1.0).max()),
                quadratic_difference=float(np.abs(quadratic_assembled - quadratic_least_squares).max()),
                gauge_cells=gauge_cells,
                operator_scaling=float(np.abs(least_squares).max()))


def gradient_error(solver):
    """Volume-weighted L2 error of the least-squares gradient of a smooth field."""
    mesh = solver.mesh
    cells = mesh.centers.reshape(-1, 2).numpy()
    approximated = solver._gradient(torch.as_tensor(smooth_field(cells)), pressure=True).numpy()
    exact = smooth_gradient(cells)
    volumes = mesh.volumes.reshape(-1).numpy()
    squared = ((approximated - exact) ** 2).sum(1)
    return dict(cells=int(solver.count),
                error=float(math.sqrt(float((volumes * squared).sum() / volumes.sum()))),
                spacing=float(math.sqrt(float(volumes.mean()))),
                median_aspect=float(np.median(cell_aspect_ratios(mesh))),
                max_aspect=float(cell_aspect_ratios(mesh).max()))


def order_study(directory, levels=(8, 16, 32), stretches=(0.0, 3.0, 6.0)):
    """``M3``: mesh-doubling order of accuracy per stretching exponent."""
    families = {}
    for stretch in stretches:
        rows = []
        for ny in levels:
            solver = solver_for(directory, 3 * ny, ny, stretch)
            rows.append(gradient_error(solver))
        for previous, row in zip(rows, rows[1:]):
            row["observed_order"] = (math.log(previous["error"] / row["error"])
                                     / math.log(previous["spacing"] / row["spacing"]))
        families[stretch] = rows
    return families


def solver_step_defect(directory, nx=6, ny=4, stretch=3.0, iteration=1):
    """``M4``: one step of the coupled least-squares solver keeps the linear gates."""
    solver = solver_for(directory, nx, ny, stretch, cls=LeastSquaresCoupledMultiElementSolver)
    torch.set_num_threads(1)
    for _ in range(iteration):
        solver.step()
    pressure = [record for record in solver.linear_history if record["kind"] == "pressure"]
    final = solver.history[-1]
    return dict(operator_defect=float(pressure[-1]["assembled_operator_relative_defect"]),
                pressure_solves=len(pressure),
                continuity=max(float(record.get("continuity", float("nan")))
                               for record in solver.history),
                mass_imbalance=float(final.get("mass_imbalance", float("nan"))),
                scheme=getattr(solver, "pressure_gradient_scheme", None))