"""Symmetry closure of the extruded Kovasznay order case: attribution and repair.

The debt being closed
---------------------
``CASE-FVM-KOVASZNAY-ORDER`` (feature ledger, open) asks for an observed L2 order
>= 1.8 on the Kovasznay (1948) analytic solution at ``Re = 40``.  The delivered
round measured 1.684 and honest-FAILed; the attribution round (m16396, ledger
D10) split that norm and found the in-plane budget already at 1.808 while the
*spanwise* budget -- whose exact value is identically zero on the ``n x n x 1``
extruded mesh -- was only 1.106, i.e. first order.  D10 left the mechanism open:
"*which* discrete term drives a cell-centre ``w`` that should be zero".

What is measured here
---------------------
The extruded case is a 2D solution in a single cell across the extrusion.  A
symmetry plane normal to ``z`` requires ``w = 0`` *and* ``du/dz = dv/dz = 0``.
``kovasznay_solver(..., symmetry_extrusion=True)`` implemented only the first
half: it un-fixed the tangential components on the two ``z``-planes, which
turns them into *free unknowns with their own momentum rows* (a traction-free
condition), instead of constraining their normal derivative to zero.  Two
closures are therefore compared on the same mesh:

``free``
    the delivered setup (tangential components of the ``z``-planes free);
``zero_gradient``
    the symmetry condition, i.e. the same faces handed to the solver's
    established ``zero_gradient_faces`` mechanism (face value tied to the
    owner cell value, so the discrete normal derivative vanishes).

The module reports, per level and closure,

* the three error budgets of the D10 split (total / in-plane / spanwise) and
  their observed orders,
* the *driving terms* of the spanwise momentum rows: the assembled operator is
  evaluated on the converged state with every ``w`` degree of freedom set to
  zero, and the residual is split by column block (``cell_u``, ``cell_v``,
  ``cell_w``, ``face_u``, ``face_v``, ``face_w``) -- that says through *which*
  unknown the spurious ``w`` is driven, and whether advection or viscosity is
  responsible,
* the analytic-state defect of the same rows (does the discrete spanwise
  momentum equation accept the exact solution at all?),
* the tangential values the free planes settle on, and the resulting spurious
  shear the cross-derivative term sees,
* uniqueness evidence: three different initial guesses must give the same
  converged ``w`` (a null space would not).

Pre-declared criteria live in :func:`criteria` (C1--C8).  Nothing here feeds a
reference into the solve: the exact field is used only for prescribed boundary
*area means* (the same helper the case already used) and for scoring.

Usage (one level per invocation keeps the cost of a level independent)::

    PYTHONPATH=src python -m tensorfvm.verification.kovasznay_symmetry \
        --output runs/kovasznay-symmetry/n32-zero_gradient --n 32 \
        --closure zero_gradient
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..native_hybrid_ns3d import NativeHybridNavierStokes3D
from ..native_manufactured3d import exact_boundary_velocity_means
from .native_hybrid_flow import channel_mesh
from .native_hybrid_ns import kovasznay_fields

CLOSURES = ('free', 'zero_gradient')


def build(n, *, closure='zero_gradient', reynolds=40., convection_order=2,
          triangulate_faces=True, divisions=None):
    """Extruded ``n x n x 1`` Kovasznay case with the requested symmetry closure."""
    if closure not in CLOSURES:
        raise ValueError(f'closure must be one of {CLOSURES}')
    divisions = (n, n, 1) if divisions is None else tuple(divisions)
    mesh = channel_mesh(divisions, triangulate_faces=triangulate_faces)
    fields = lambda points: kovasznay_fields(points, reynolds)
    exact = fields(mesh.cell_centers)
    boundary = exact_boundary_velocity_means(mesh, fields)
    fixed = np.repeat((mesh.neighbor < 0)[:, None], 3, axis=1)
    z = mesh.face_centers[:, 2]
    planes = (mesh.neighbor < 0) & ((z < 1e-10) | (z > 1 - 1e-10))
    fixed[planes, :2] = False
    zero_gradient = None
    if closure == 'zero_gradient':
        zero_gradient = np.zeros((len(mesh.owner), 3), bool)
        zero_gradient[planes, :2] = True
    solver = NativeHybridNavierStokes3D(
        mesh, fixed, boundary, viscosity=1. / reynolds, body_force=0.,
        gauge_value=exact['pressure'][0], convection_order=convection_order,
        coupled_linear_method='reused_lu', zero_gradient_faces=zero_gradient)
    return mesh, solver, exact, dict(planes=planes, fields=fields,
                                     boundary=boundary)


def _rel_l2(error, reference, volumes):
    weight = volumes.reshape((len(volumes),) + (1,) * (np.asarray(error).ndim - 1))
    return float(np.sqrt(np.sum(weight * np.square(error)) / np.sum(weight * np.square(reference))))


def budgets(velocity, exact, volumes):
    """D10's three budgets, computed with one volume-weighted norm."""
    velocity = np.asarray(velocity)
    exact_velocity = np.asarray(exact['velocity'])
    total = _rel_l2(velocity - exact_velocity, exact_velocity, volumes)
    inplane = _rel_l2(velocity[:, :2] - exact_velocity[:, :2], exact_velocity, volumes)
    span = _rel_l2(velocity[:, 2:] - exact_velocity[:, 2:], exact_velocity, volumes)
    return dict(velocity_l2=total, inplane_l2=inplane, spanwise_l2=span,
                quadrature_closure=abs(np.hypot(inplane, span) - total) / total,
                max_abs_spanwise=float(np.max(np.abs(velocity[:, 2]))))


def block_indices(n_cells, n_faces):
    cell = np.arange(3 * n_cells).reshape(n_cells, 3)
    face = 3 * n_cells + np.arange(3 * n_faces).reshape(n_faces, 3)
    return dict(cell_u=cell[:, 0], cell_v=cell[:, 1], cell_w=cell[:, 2],
                face_u=face[:, 0], face_v=face[:, 1], face_w=face[:, 2])


def spanwise_drive_terms(solver, exact):
    """Split the spanwise momentum residual into column blocks and operator families.

    The state is taken from the solve, but every ``w`` unknown is set to zero, so
    the reported numbers are exactly "what pushes on ``w``" rather than "what
    ``w`` already balances".  The frozen advection flux and gradient are the
    converged ones, i.e. the same linearisation the last outer iteration used.
    """
    N, F = solver.N, solver.F
    blocks = block_indices(N, F)
    state = np.asarray(solver.state)
    frozen = state.copy()
    frozen[blocks['cell_w']] = 0.
    frozen[blocks['face_w']] = 0.
    flux = solver.volume_flux(state)
    convection, convection_source = solver.convection.hybrid_system(
        flux, solver.face_mean.matrix, solver.velocity_gradient(state))
    convection = convection.tocsr()
    convection_source = np.asarray(convection_source).ravel()
    viscous = solver.viscous.full_matrix.tocsr()
    pressure = np.asarray(solver.divergence.T @ solver.p).ravel()
    load = np.asarray(solver.load).ravel()
    interior = np.flatnonzero(mesh_interior(solver))
    rows = blocks['face_w'][interior]

    per_block = {}
    for name, columns in blocks.items():
        per_block[name] = dict(
            advection=block_action(convection, rows, columns, frozen[columns]),
            viscous=block_action(viscous, rows, columns, frozen[columns]))
    residual = (convection @ frozen + convection_source + viscous @ frozen - pressure - load)[rows]
    return dict(
        faces_interior=int(len(interior)),
        residual_max=float(np.max(np.abs(residual))),
        blocks=per_block,
        convection_source=float(np.max(np.abs(convection_source[rows]))),
        pressure=float(np.max(np.abs(pressure[rows]))),
        # the cross-derivative family: tangential velocity of the *faces*
        viscous_cross=float(max(per_block['face_u']['viscous'], per_block['face_v']['viscous'])),
        viscous_spanwise=float(per_block['face_w']['viscous']),
        advection_total=float(max(per_block[name]['advection'] for name in blocks)),
    )


def block_action(matrix, rows, columns, values):
    """``max |(matrix[rows, columns] @ values)|`` without densifying."""
    action = matrix[rows, :][:, columns] @ values
    return float(np.max(np.abs(np.asarray(action).ravel()), initial=0.))


def mesh_interior(solver):
    return solver.mesh.neighbor >= 0


def analytic_defect(solver, exact, joints):
    """Residual of the spanwise momentum rows when the *analytic* state is plugged in."""
    N, F = solver.N, solver.F
    blocks = block_indices(N, F)
    state = np.zeros(3 * N + 3 * F)
    state[:3 * N] = np.asarray(exact['velocity']).ravel()
    face_values = joints['fields'](np.asarray(solver.mesh.face_centers))['velocity']
    open_faces = np.flatnonzero(solver.mesh.neighbor < 0)
    face_values[open_faces] = joints['boundary'][open_faces]
    state[3 * N:] = face_values.ravel()
    flux = np.sum(state[3 * N:].reshape(F, 3) * solver.mesh.area_vectors, axis=1)
    gradient = np.stack([solver.convection.D @ (state[3 * N:].reshape(F, 3)
                                               * solver.mesh.area_vectors[:, j, None])
                         / solver.mesh.cell_volumes[:, None] for j in range(3)], axis=2)
    convection, convection_source = solver.convection.hybrid_system(
        flux, solver.face_mean.matrix, gradient)
    viscous = solver.viscous.full_matrix
    pressure = solver.divergence.T @ np.asarray(exact['pressure'])
    residual = np.asarray(convection @ state).ravel() + np.asarray(convection_source).ravel() \
        + np.asarray(viscous @ state).ravel() - np.asarray(pressure).ravel() \
        - np.asarray(solver.load).ravel()
    rows = blocks['face_w'][np.flatnonzero(mesh_interior(solver))]
    return float(np.max(np.abs(residual[rows])))


def plane_shear(solver, exact):
    """Spurious normal shear the in-plane faces see: ``du_t/dz`` on the extrusion planes.

    The planes carry ``w = 0``; a symmetry-consistent closure forces the
    tangential plane data to equal the owner cell value, so the discrete
    two-point shear ``(u_plane - u_cell)/(dz/2)`` vanishes.  The reported number
    is the largest such shear, non-dimensionalised by ``U_max / (dz/2)``.
    """
    mesh = solver.mesh
    N, F = solver.N, solver.F
    blocks = block_indices(N, F)
    state = np.asarray(solver.state)
    cells = state[:3 * N].reshape(N, 3)
    faces = state[3 * N:].reshape(F, 3)
    z = mesh.face_centers[:, 2]
    planes = (mesh.neighbor < 0) & ((z < 1e-10) | (z > 1 - 1e-10))
    owner = mesh.owner[planes]
    dz = np.abs(mesh.face_centers[planes, 2] - mesh.cell_centers[owner, 2])
    shear = (faces[planes, :2] - cells[owner, :2]) / dz[:, None]
    scale = np.max(np.abs(np.asarray(exact['velocity']))) / np.mean(dz)
    return dict(max_abs_plane_shear=float(np.max(np.abs(shear))),
                dimensionless=float(np.max(np.abs(shear)) / scale),
                max_abs_plane_offset=float(np.max(np.abs(faces[planes, :2] - cells[owner, :2]))))


def uniqueness(n, *, closure='free', reynolds=40., perturbations=(0., 0.1, -0.1), seed=0):
    """Converged ``w`` must not depend on the initial guess (a null space would)."""
    out = []
    for amplitude in perturbations:
        mesh, solver, exact, joints = build(n, closure=closure, reynolds=reynolds)
        if amplitude:
            rng = np.random.default_rng(seed)
            solver.state = solver.state + amplitude * rng.standard_normal(solver.state.shape)
        result = solver.solve(algorithm='coupled', outer_relaxation=.7, tolerance=1e-8,
                              max_outer_iterations=200)
        velocity = np.asarray(result['velocity'])
        out.append(dict(amplitude=amplitude,
                        max_abs_spanwise=float(np.max(np.abs(velocity[:, 2]))),
                        spanwise_vector=velocity[:, 2].copy()))
    reference = out[0]
    spread = max(float(np.max(np.abs(row['spanwise_vector'] - reference['spanwise_vector'])))
                 for row in out[1:])
    return dict(levels=[dict(amplitude=r['amplitude'], max_abs_spanwise=r['max_abs_spanwise'])
                        for r in out],
                max_pairwise_spanwise_difference=spread,
                relative=spread / max(reference['max_abs_spanwise'], 1e-30))


def observed_order(coarse, fine, ratio):
    return float(np.log(coarse / fine) / np.log(ratio))


def criteria(level_rows, *, order_gate=1.8, machine=1e-10):
    """Pre-declared criteria (declared before the levels were run)."""
    fixed = sorted((row for row in level_rows if row['closure'] == 'zero_gradient'),
                   key=lambda row: row['n'])
    free = sorted((row for row in level_rows if row['closure'] == 'free'),
                  key=lambda row: row['n'])
    rows = []

    def add(name, description, value, passed, target):
        rows.append(dict(id=name, description=description, value=value,
                         target=target, passed=bool(passed)))

    if fixed:
        worst = max(row['max_abs_spanwise'] for row in fixed)
        add('C1', 'zero-gradient symmetry closure pins the spanwise velocity to zero',
            worst, worst < machine, f'< {machine:g}')
    if free:
        worst = max(row['max_abs_spanwise'] for row in free)
        add('C2', 'free tangential z-planes leave a spurious spanwise velocity',
            worst, worst > 1e-6, '> 1e-6 (negative control is not vacuous)')
        worst_shear = max(row.get('plane_shear', {}).get('max_abs_plane_shear', 0.) for row in free)
        add('C3', 'free planes settle on a tangential value that differs from their owner cell',
            worst_shear, worst_shear > 0., '> 0')
        orders = [row['spanwise_order'] for row in free if row.get('spanwise_order')]
        if orders:
            add('C4', 'spurious spanwise budget is first order in h',
                max(orders), max(orders) < 1.5, '< 1.5')
        drive = [row for row in free if row.get('drive')]
        if drive:
            row = drive[-1]
            blocks = row['drive']['blocks']
            ratio = row['drive']['viscous_cross'] / max(row['drive']['residual_max'], 1e-300)
            add('C5', 'the drive of the spanwise rows is the viscous cross-derivative block',
                ratio, ratio > 0.99, '> 0.99 of the residual')
            pure = max(blocks['cell_u']['advection'], blocks['cell_v']['advection'],
                       blocks['cell_w']['advection'], blocks['cell_w']['viscous'],
                       blocks['face_w']['viscous'], row['drive']['convection_source'],
                       row['drive']['pressure'])
            add('C6', 'advection, pressure and same-component blocks contribute nothing',
                pure, pure < 1e-9 * max(row['drive']['residual_max'], 1e-300), '< 1e-9 of the residual')
            add('C7', 'the assembled spanwise rows accept the analytic state (consistent equations)',
                row['analytic_defect'], row['analytic_defect'] < 1e-9, '< 1e-9')
    if len(fixed) >= 2:
        worst = max(row['velocity_order'] for row in fixed[1:])
        add('C8', 'velocity L2 order with the symmetry closure meets the anchor gate',
            worst, worst >= order_gate, f'>= {order_gate}')
    return rows


def run_level(n, *, closure, reynolds=40., convection_order=2, tolerance=1e-8,
              outer_relaxation=.7, probe=True, divisions=None):
    tick = time.perf_counter()
    mesh, solver, exact, joints = build(n, closure=closure, reynolds=reynolds,
                                        convection_order=convection_order,
                                        divisions=divisions)
    result = solver.solve(algorithm='coupled', outer_relaxation=outer_relaxation,
                          tolerance=tolerance, max_outer_iterations=200)
    seconds = time.perf_counter() - tick
    volumes = np.asarray(mesh.cell_volumes)
    velocity = np.asarray(result['velocity'])
    pressure = np.asarray(result['pressure'])
    closed = result['numerically_converged']
    outer = result['outer_history'][-1]
    row = dict(
        n=n, closure=closure, Reynolds=reynolds, convection_order=convection_order,
        divisions=list(divisions) if divisions else [n, n, 1],
        cells=int(solver.N), faces=int(solver.F), seconds=seconds,
        numerically_converged=bool(closed),
        boundary_velocity_error=float(result['boundary_velocity_error']),
        divergence_max=float(outer['divergence_max']),
        momentum_residual=float(outer['current_momentum_residual']),
        outer_iterations=len(result['outer_history']),
        pressure_l2=_rel_l2(pressure - exact['pressure'], exact['pressure'], volumes),
        **budgets(velocity, exact, volumes))
    if probe:
        row['drive'] = spanwise_drive_terms(solver, exact)
        row['analytic_defect'] = analytic_defect(solver, exact, joints)
        row['plane_shear'] = plane_shear(solver, exact)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--closure', choices=CLOSURES, default='zero_gradient')
    parser.add_argument('--reynolds', type=float, default=40.)
    parser.add_argument('--convection-order', type=int, default=2)
    parser.add_argument('--tolerance', type=float, default=1e-8)
    parser.add_argument('--no-probe', action='store_true')
    args = parser.parse_args()
    row = run_level(args.n, closure=args.closure, reynolds=args.reynolds,
                    convection_order=args.convection_order, tolerance=args.tolerance,
                    probe=not args.no_probe)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'level.json').write_text(json.dumps(row, indent=2) + '\n')
    print('LEVEL ' + json.dumps({k: v for k, v in row.items() if k != 'drive'}))
    if 'drive' in row:
        print('DRIVE ' + json.dumps(row['drive']))


if __name__ == '__main__':
    main()