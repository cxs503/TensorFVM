"""Verification of the ``gauss-skew-corrected`` conservative pressure reconstruction.

Motivation (verification debt closed here): :mod:`tensorfvm.multi_element_skew_pressure`
is a *selectable production* pressure-gradient scheme
(``scripts/run_three_element_steady.py --pressure-gradient gauss-skew-corrected``)
and its module docstring makes four quantitative claims that no test or artifact
in this repository measured —

1. *"least-squares gradients supply a geometric skewness correction"* -- the
   reconstructed face pressure must recover a **linear** pressure field exactly
   at the face centroid, and the correction must be non-trivial (a no-op scheme
   would satisfy the weaker claim but buy nothing);
2. *"the pressure force is integrated over faces, preserving equal/opposite
   internal forces"* -- the cell pressure force ``-V grad(p)`` must telescope to
   the boundary flux, i.e. internal faces cancel pairwise to machine precision;
3. *"Neumann boundary reconstruction removes its wall-normal gradient
   component"* -- only the wall-normal part of the skew displacement may be
   removed, so wall faces keep the tangential correction;
4. *"the assembled pressure response uses precisely the same reconstruction"* --
   the sparse operator built by ``_geometry()`` must reproduce the non-linear
   ``_gradient(pressure=True)`` used by momentum, velocity correction and
   Rhie--Chow flux.  This is the part that silently breaks a SIMPLE loop when
   the composed operator drifts from the residual.

All checks are operator level (no time stepping except one end-to-end step in
``S5``), and every reference value is analytic (the exact gradient / exact face
value of a prescribed polynomial), so the scheme is never graded against itself.

The baseline ``gauss-linear`` (:class:`tensorfvm.multi_element_flow.MultiElementFlowSolver`)
is measured on the *same* mesh, so the reported gain is the skew correction alone.

Mesh family: ``stretched_mesh`` from :mod:`.multi_element_ls` (structured quads,
exponential wall clustering) is transformed by ``x += shear * y`` and an optional
**deterministic interior-node jitter**.  This distinction matters and is a result
of this audit, not a setup detail: a sheared structured grid is still made of
*parallelograms*, whose shared-face centroid lies exactly on the segment joining
the two cell centroids, so the skew displacement -- and therefore the whole skew
correction -- is identically zero there (``max_interior_skew`` ~1e-16 on the
192-cell mesh).  Only non-parallelogram (jittered or curved) cells produce a
non-zero correction, which is what ``S1``/``S5`` therefore grade.

Measured on this branch (192-cell clustered meshes for ``S1``--``S4``; orders on
192/768/3072/12288 cells):

``S1`` linear-field face value, clean faces: skew error 1.8e-15 versus projected
      1.1e-2 (25 % jitter) -- machine exact versus second order; on a
      parallelogram mesh the displacement is 6.9e-18, so ``gauss-skew-corrected``
      is bit-identical to ``gauss-linear`` there.
``S2`` wall faces: normal displacement 0.0, tangential 6.2e-3, outlet 0.0.
``S3`` ``sum_cells V grad p`` minus boundary flux = 8.9e-16 (relative 6.2e-17).
``S4`` assembled versus non-linear gradient 9.9e-14 (smooth), 4.6e-15
      (quadratic), ``H``/``I`` row defects 0.0, 8 outlet gauge cells.
``S5`` jittered family: skew-corrected 9.6e-2 -> 3.9e-2 -> 1.9e-2 -> 9.3e-3
      (order 1.31, 1.07, 1.01) while the projected interpolation stagnates at
      ~2.9e-1 (order -0.17, -0.05, 0.07): first order versus no convergence.
      Parallelogram families keep order 1.8--2.2 for both schemes, because they
      are literally the same operator.
``S6`` one coupled step: pressure operator defect 1.5e-16, continuity 3.5e-3.

Caveat measured, not assumed: the outlet gauge forces ``p_f = 0`` on outlet faces,
so an outlet cell's least-squares gradient is not the gradient of a non-vanishing
linear field (44.4 against an exact 1.7); the correction multiplies the
*interpolated gradient* by ``delta`` and therefore imports that inconsistency into
adjacent interior face values (``leakage`` 1.1e-1), whereas the projected
interpolation only uses cell values.  Exactness is therefore asserted on faces
whose owner and neighbour are both interior cells.
"""
import math
import pathlib

import numpy as np
import torch

from ..body_fitted import BodyFittedSolver
from ..multi_element_flow import MultiElementFlowSolver
from ..multi_element_skew_pressure import (SkewPressureCoupledMultiElementSolver,
                                           SkewPressureMultiElementSolver)
from ..solver import SolverConfig
from .multi_element_ls import build_mesh_file, cell_aspect_ratios, interior_cells

__all__ = ["skewed_mesh_file", "solver_for", "face_reconstruction_exactness",
           "wall_normal_removal", "force_conservation", "operator_consistency",
           "gradient_error", "order_study", "solver_step_defect"]


def skewed_mesh_file(directory, nx, ny, stretch, shear=0.0, jitter=0.0, seed=0):
    """Three-element mesh, then ``x += shear*y`` and a deterministic node jitter.

    Reuses :func:`tensorfvm.verification.multi_element_ls.stretched_mesh` verbatim
    so physical groups and topology are unchanged; only node coordinates move.
    Boundary nodes (``i in {0,nx}`` or ``j in {0,ny}``) are never displaced, so the
    wall groups stay flat and every mesh remains geometrically admissible.
    ``jitter`` is the displacement amplitude as a fraction of the local spacing.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = build_mesh_file(directory, nx, ny, stretch)
    lines = path.read_text().splitlines()
    start = lines.index("$Nodes") + 2
    count = int(lines[start - 1])
    rows = np.arange(ny + 1)
    y_axis = (rows / ny if stretch == 0 else np.expm1(stretch * rows / ny) / math.expm1(stretch))
    spacing_x, spacing_y = 1.0 / nx, float(np.abs(np.diff(y_axis)).min())
    rng = np.random.RandomState(seed)
    for index in range(start, start + count):
        node, x, y, z = lines[index].split()
        x, y = float(x), float(y)
        j = (index - start) // (nx + 1)
        i = (index - start) % (nx + 1)
        dx = dy = 0.0
        if jitter and 0 < i < nx and 0 < j < ny:
            dx = jitter * spacing_x * rng.uniform(-1.0, 1.0)
            dy = jitter * spacing_y * rng.uniform(-1.0, 1.0)
        lines[index] = f"{node} {x + shear * y + dx:.17g} {y + dy:.17g} {z}"
    text = "\n".join(lines) + "\n"
    tag = f"skew_{nx}x{ny}_b{stretch:g}_s{shear:g}_j{jitter:g}_n{seed}"
    target = directory / f"{tag}.msh"
    target.write_text(text)
    return target


def config_for(path):
    return SolverConfig(nx=1, ny=1, length=3, height=1, cylinder_radius=None,
                        mesh_type="three-element", mesh_file=str(path),
                        reynolds=100, turbulence_model="spalart-allmaras",
                        max_iterations=1)


def solver_for(directory, nx, ny, stretch, shear=0.0, jitter=0.0, seed=0,
               cls=SkewPressureMultiElementSolver):
    return cls(config_for(skewed_mesh_file(directory, nx, ny, stretch, shear, jitter, seed)))


def _linear_field(solver, gradient=(1.7, -2.3), offset=0.4):
    cells = solver.mesh.centers.reshape(-1, 2)
    exact = torch.tensor(gradient, dtype=cells.dtype)
    return cells @ exact + offset, exact


def smooth_field(cells):
    """Scalar field used for the truncation-error study (identical to the LS audit)."""
    return np.sin(np.pi * cells[:, 0]) * np.cos(np.pi * cells[:, 1] / 2)


def smooth_gradient(cells):
    return np.stack([np.pi * np.cos(np.pi * cells[:, 0]) * np.cos(np.pi * cells[:, 1] / 2),
                     -0.5 * np.pi * np.sin(np.pi * cells[:, 0]) * np.sin(np.pi * cells[:, 1] / 2)],
                    -1)


def face_reconstruction_exactness(solver, gradient=(1.7, -2.3), offset=0.4):
    """``S1``: reconstructed face pressure of a linear field at face centroids.

    Two face sets are reported.  ``clean`` faces have an interior owner *and* an
    interior neighbour, so no cell contributing to the interpolated gradient is a
    boundary cell.  ``all_interior`` adds faces that touch a boundary cell and is
    reported separately because it exposes a genuine mechanism (``leakage``): the
    outlet gauge fixes ``p_f = 0`` on outlet faces, so an outlet cell's
    least-squares gradient does not see the outlet face and is *not* the exact
    gradient of a linear test field.  The projected interpolation only uses cell
    *values* and cannot amplify that, while the skew correction multiplies the
    interpolated *gradient* by ``delta`` and therefore imports the boundary-cell
    inconsistency into interior face values -- ``leakage`` measures exactly that
    term, ``delta . (I grad p - grad p)``.
    """
    mesh = solver.mesh
    field, exact = _linear_field(solver, gradient, offset)
    face_exact = mesh.face_centers.reshape(-1, 2) @ exact + offset
    boundary_cells = torch.zeros(solver.count, dtype=torch.bool)
    boundary_cells[torch.unique(mesh.owner[mesh.boundary])] = True
    owner = mesh.owner
    neighbor = torch.as_tensor(mesh.neighbor).clamp(min=0)
    interior = (~mesh.boundary) & (~mesh.masks["outlet"])
    clean = interior & (~boundary_cells[owner]) & (~boundary_cells[neighbor])
    corrected = solver.reconstructed_pressure_faces(field)
    projected = solver._interpolate(field)
    delta = solver._pressure_skew_displacement()
    gradient_error_interior = solver._interpolate(
        BodyFittedSolver._gradient(solver, field, pressure=True)) - exact
    volumes = mesh.volumes.reshape(-1)
    return dict(clean_faces=int(clean.sum()), interior_faces=int(interior.sum()),
                faces=int(len(interior)), boundary_faces=int(mesh.boundary.sum()),
                skew_error=float((corrected[clean] - face_exact[clean]).abs().max()),
                projected_error=float((projected[clean] - face_exact[clean]).abs().max()),
                all_interior_skew_error=float((corrected[interior] - face_exact[interior]).abs().max()),
                all_interior_projected_error=float(
                    (projected[interior] - face_exact[interior]).abs().max()),
                leakage=float((delta[interior] * gradient_error_interior[interior]).abs().sum(1).max()),
                max_skew=float(delta.norm(dim=1).max()),
                max_interior_skew=float(delta[interior].norm(dim=1).max()),
                mean_interior_skew=float(delta[interior].norm(dim=1).mean()),
                spacing=float(math.sqrt(float(volumes.mean()))), count=int(solver.count),
                median_aspect=float(np.median(cell_aspect_ratios(mesh))))


def wall_normal_removal(solver, gradient=(1.7, -2.3), offset=0.4):
    """``S2``: at wall faces the skew displacement keeps only its tangential part.

    A wall face is a real face of the physical domain; its centroid cannot be moved,
    so the *normal* part of ``x_f - x_interp`` must be dropped (it would otherwise
    double-count the boundary distance) while the tangential part is retained.
    """
    mesh = solver.mesh
    delta = solver._pressure_skew_displacement()
    wall = mesh.masks["wall"]
    if not bool(wall.any()):
        return dict(wall_faces=0)
    normal = mesh.face_normals[wall]
    normal_component = (delta[wall] * normal).sum(1)
    tangential = (delta[wall] - normal_component[:, None] * normal).norm(dim=1)
    return dict(wall_faces=int(wall.sum()),
                wall_normal_component=float(normal_component.abs().max()),
                wall_tangential_component=float(tangential.max()),
                boundary_skew=float(delta[mesh.boundary].norm(dim=1).max()),
                outlet_skew=float(delta[mesh.masks["outlet"]].norm(dim=1).max()),
                outlet_faces=int(mesh.masks["outlet"].sum()))


def force_conservation(solver, gradient=(1.7, -2.3), offset=0.4):
    """``S3``: internal pressure forces cancel, so ``sum_cells V grad p = boundary flux``.

    The left side is the pressure force the momentum equations subtract
    (``-V grad p``); the right side is the same face pressure integrated over the
    boundary only.  Equality to machine precision is the discrete statement of
    equal/opposite internal forces.
    """
    mesh = solver.mesh
    field, _ = _linear_field(solver, gradient, offset)
    faces = solver.reconstructed_pressure_faces(field)
    cell_force = (solver._gradient(field, pressure=True) * solver.volume[:, None]).sum(0)
    boundary = mesh.boundary
    boundary_flux = (faces[boundary][:, None] * solver.S[boundary]).sum(0)
    scale = float((faces[boundary][:, None] * solver.S[boundary]).abs().sum())
    internal = (~boundary) & (~mesh.masks["outlet"])
    internal_scale = float((faces[internal][:, None] * solver.S[internal]).abs().sum())
    defect = float((cell_force - boundary_flux).abs().max())
    return dict(identity_defect=defect,
                relative_defect=defect / max(scale, 1e-300),
                internal_flux_scale=internal_scale,
                cell_force=[float(value) for value in cell_force],
                boundary_flux=[float(value) for value in boundary_flux])


def operator_consistency(solver):
    """``S4``: the sparse pressure operators reproduce ``_gradient(pressure=True)``.

    Also records the outlet gauge (faces whose pressure value is fixed to zero,
    which makes the operator deliberately not invariant under a constant shift) and
    the row defects of the composed incidence/interpolation pair.
    """
    mesh = solver.mesh
    cells = mesh.centers.reshape(-1, 2).numpy()
    values = smooth_field(cells)
    non_linear = solver._gradient(torch.as_tensor(values), pressure=True).numpy()
    H, I, G = solver._geometry()
    assembled = np.stack([G[0] @ values, G[1] @ values], -1)
    neighbor = np.asarray(mesh.neighbor).ravel()
    owner = np.asarray(mesh.owner).ravel()
    faces_per_cell = (np.bincount(owner, minlength=solver.count)
                      + np.bincount(neighbor[neighbor >= 0], minlength=solver.count))
    quadratic = cells[:, 0] * cells[:, 1]
    quadratic_non_linear = solver._gradient(torch.as_tensor(quadratic), pressure=True).numpy()
    quadratic_assembled = np.stack([G[0] @ quadratic, G[1] @ quadratic], -1)
    outlet = np.asarray(mesh.masks["outlet"])
    return dict(gradient_difference=float(np.abs(assembled - non_linear).max()),
                quadratic_difference=float(np.abs(quadratic_assembled - quadratic_non_linear).max()),
                h_row_defect=float(np.abs(np.asarray(abs(H).sum(axis=1)).ravel() - faces_per_cell).max()),
                i_row_defect=float(np.abs(np.asarray(I.sum(axis=1)).ravel() - 1.0).max()),
                gauge_cells=int(np.unique(owner[outlet]).size),
                operator_scaling=float(np.abs(non_linear).max()))


def gradient_error(solver, interior_only=True):
    """Volume-weighted L2 error of the pressure gradient of a smooth field.

    ``interior_only`` drops cells that own a boundary face: those cells read an
    owner-extended Neumann face value, which is first-order by construction and is
    not what the skew correction claims to fix.
    """
    mesh = solver.mesh
    cells = mesh.centers.reshape(-1, 2).numpy()
    approximated = solver._gradient(torch.as_tensor(smooth_field(cells)), pressure=True).numpy()
    exact = smooth_gradient(cells)
    volumes = mesh.volumes.reshape(-1).numpy()
    keep = interior_cells(mesh).numpy() if interior_only else np.ones_like(volumes, dtype=bool)
    squared = ((approximated - exact) ** 2).sum(1)
    weight = volumes * keep
    return dict(cells=int(solver.count), graded_cells=int(keep.sum()),
                error=float(math.sqrt(float((weight * squared).sum() / weight.sum()))),
                max_error=float(np.sqrt(squared[keep].max())),
                spacing=float(math.sqrt(float(volumes.mean()))),
                median_aspect=float(np.median(cell_aspect_ratios(mesh))))


def order_study(directory, levels=(8, 16), stretches=(6.0,), shears=(0.0,), jitters=(0.0, 0.25)):
    """``S5``: mesh-doubling order of the smooth-field gradient error, per scheme."""
    families = {}
    for jitter in jitters:
        for shear in shears:
            for stretch in stretches:
                rows = []
                for ny in levels:
                    row = dict(ny=ny, shear=shear, stretch=stretch, jitter=jitter)
                    for name, cls in (("skew", SkewPressureMultiElementSolver),
                                      ("projected", MultiElementFlowSolver)):
                        solver = solver_for(directory, 3 * ny, ny, stretch, shear, jitter,
                                            cls=cls)
                        measured = gradient_error(solver)
                        row[name] = measured["error"]
                        row["cells"] = measured["cells"]
                        row["spacing"] = measured["spacing"]
                        row["median_aspect"] = measured["median_aspect"]
                    rows.append(row)
                for previous, row in zip(rows, rows[1:]):
                    for name in ("skew", "projected"):
                        row[name + "_order"] = (math.log(previous[name] / row[name])
                                                / math.log(previous["spacing"] / row["spacing"]))
                families[f"shear{shear:g}_stretch{stretch:g}_jitter{jitter:g}"] = rows
    return families


def solver_step_defect(directory, nx=6, ny=4, stretch=3.0, shear=0.0, jitter=0.25,
                       iteration=1):
    """``S6``: one end-to-end coupled step keeps the pressure linear gate the same."""
    try:
        solver = solver_for(directory, nx, ny, stretch, shear, jitter,
                            cls=SkewPressureCoupledMultiElementSolver)
    except ValueError as error:  # pragma: no cover - backend precondition
        return dict(skipped=str(error))
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
                force=[float(value) for value in solver._surface_force(solver.mesh.masks["wall"])],
                scheme=getattr(solver, "pressure_gradient_scheme", None))