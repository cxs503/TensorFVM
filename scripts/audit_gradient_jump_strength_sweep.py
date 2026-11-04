"""Two-knob admissibility audit of the hybrid viscous operator.

The hybrid cell/face viscous operator carries two independent stabilization
knobs (``src/tensorfvm/native_hybrid_viscous3d.py``):

* ``stabilization`` -- the per-cell face-offset residual penalty
  ``residual = jump - offset @ gradient`` weighted by ``mu*area/height``;
* ``gradient_jump_stabilization`` -- the tangential gradient-jump energy of
  ``NativeGradientJump3D`` (Hansbo & Larson 2003, eq. (26)).

``docs/verification-native-gradient-jump`` only audited the jump knob at
``s_j in {0, 1}`` and explicitly left the interaction with ``stabilization``
unstudied. This audit maps the 2-D knob space on a *simplex* (Kuhn 6-tetra)
family and a matching Cartesian hexahedron family, measuring

1. coercivity: ``lambda_min/lambda_max`` of the Dirichlet-eliminated
   symmetric operator and the dimension of its near-null space;
2. affine exactness (both penalties annihilate affine fields analytically);
3. accuracy on a divergence-free analytic viscous field with
   ``u = (1/2pi) curl A`` so that ``Delta u = -2 pi^2 u`` and
   ``f = -div(tau) = 2 pi^2 mu u`` is available in closed form;
4. the accuracy price of each knob, split into the gradient-jump energy and
   the face-residual penalty energy feeding the discrete energy form.

Nothing in the solver is modified: this is a verification-debt audit.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.native_mesh3d import NativeMesh3D

MU = .1
# 6-tetra Kuhn decomposition of the unit cube in the local node numbering used
# by ``_grid``: 0=000, 1=100, 2=110, 3=010, 4=001, 5=101, 6=111, 7=011
# (bit x=1, bit y=2, bit z=4).
_BITS = {'x': 1, 'y': 2, 'z': 4}
_LOCAL_OF_BITS = {0: 0, 1: 1, 3: 2, 2: 3, 4: 4, 5: 5, 7: 6, 6: 7}


def _kuhn_tets():
    tets = []
    for order in (('x', 'y', 'z'), ('x', 'z', 'y'), ('y', 'x', 'z'),
                  ('y', 'z', 'x'), ('z', 'x', 'y'), ('z', 'y', 'x')):
        bits, path = 0, [0]
        for axis in order:
            bits |= _BITS[axis]
            path.append(_LOCAL_OF_BITS[bits])
        tets.append(path)
    return tets


def _grid(n):
    axes = [np.linspace(0., 1., n + 1)] * 3
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    index = lambda x, y, z: (x * (n + 1) + y) * (n + 1) + z
    cells = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                cells.append([index(x, y, z), index(x + 1, y, z),
                              index(x + 1, y + 1, z), index(x, y + 1, z),
                              index(x, y, z + 1), index(x + 1, y, z + 1),
                              index(x + 1, y + 1, z + 1), index(x, y + 1, z + 1)])
    return points, cells


def hex_mesh(n):
    points, cells = _grid(n)
    return NativeMesh3D.from_cells(points, cells, ['hexahedron'] * len(cells))


def tet_mesh(n):
    points, cells = _grid(n)
    tets = [[cube[i] for i in tet] for cube in cells for tet in _kuhn_tets()]
    return NativeMesh3D.from_cells(points, tets, ['tetra'] * len(tets))


def analytic_velocity(x):
    """(1/2pi) curl A, A = (sin pi y sin pi z, sin pi x sin pi z, sin pi x sin pi y)."""
    x = np.asarray(x, float)
    px, py, pz = np.pi * x[:, 0], np.pi * x[:, 1], np.pi * x[:, 2]
    return .5 * np.column_stack([
        np.sin(px) * (np.cos(py) - np.cos(pz)),
        np.sin(py) * (np.cos(pz) - np.cos(px)),
        np.sin(pz) * (np.cos(px) - np.cos(py))])


def analytic_body_force(x):
    """f = -div tau = -mu Delta u = 2 pi^2 mu u for the field above."""
    return 2 * np.pi ** 2 * MU * analytic_velocity(x)


def affine_velocity(x, gradient, offset):
    return np.asarray(offset, float) + np.asarray(x, float) @ np.asarray(gradient, float).T


def volume_relative_error(computed, exact, weights):
    numerator = float(np.sum(weights[:, None] * (computed - exact) ** 2))
    denominator = float(np.sum(weights[:, None] * exact ** 2))
    return float(np.sqrt(numerator / denominator))


def smallest_eigenvalues(matrix, k):
    """Smallest algebraic eigenvalues; shift-invert first (exact near kernel)."""
    from scipy.linalg import eigvalsh
    from scipy.sparse.linalg import ArpackNoConvergence
    k = min(k, matrix.shape[0] - 1)
    scale = float(np.max(np.abs(matrix.diagonal())))
    for shift in (-1e-9, -1e-6, -1e-4):
        try:
            return np.sort(eigsh(matrix, k=k, sigma=shift * scale, which='LM',
                                 return_eigenvectors=False))
        except (ArpackNoConvergence, RuntimeError):
            continue
    try:
        return np.sort(eigsh(matrix, k=k, which='SA', return_eigenvectors=False,
                             maxiter=4000, tol=1e-9))
    except ArpackNoConvergence:
        pass
    if matrix.shape[0] <= 2500:
        return np.sort(eigvalsh(matrix.toarray()))[:k]
    raise RuntimeError('smallest eigenvalue extraction failed')


def operator_spectrum(op, k=6):
    matrix = op.matrix
    largest = float(eigsh(matrix, k=1, which='LA', return_eigenvectors=False)[0])
    smallest = smallest_eigenvalues(matrix, k)
    coarse = smallest <= 1e-9 * largest
    return dict(largest=largest, smallest=float(smallest[0]),
                smallest_beyond_null=float(smallest[~coarse][0])
                if np.any(~coarse) else None,
                ratio=float(smallest[0] / largest),
                near_null_coarse=int(np.sum(coarse)),
                near_null_tight=int(np.sum(smallest <= 1e-12 * largest)),
                smallest_spectrum=[float(v) for v in smallest])


def penalty_energy(op, state):
    """Energy stored in the face-offset residual penalty alone."""
    total = 0.
    for faces, sign, ids, local, gradient, residual, penalty in op.local:
        block = np.kron(residual.T @ (penalty[:, None] * residual), np.eye(op.components))
        local_state = state[ids]
        total += float(local_state @ block @ local_state)
    return total


def solve_case(mesh, stabilization, jump, *, affine=False, spectrum=True):
    """Assemble + solve one knob pair; failures are recorded, never raised."""
    started = time.perf_counter()
    op = NativeHybridViscous3D(mesh, MU, stabilization=stabilization,
                               gradient_jump_stabilization=jump)
    if affine:
        gradient = np.array([[.2, .35, .1], [-.3, .05, .25], [.1, .2, -.15]])
        offset = np.array([.3, -.2, .15])
        force = np.zeros((len(mesh.cells), 3))
        boundary = affine_velocity(mesh.face_centers, gradient, offset)
        exact = affine_velocity(mesh.cell_centers, gradient, offset)
        exact_faces = affine_velocity(mesh.face_centers, gradient, offset)
    else:
        force = analytic_body_force(mesh.cell_centers)
        boundary = analytic_velocity(mesh.face_centers)
        exact = analytic_velocity(mesh.cell_centers)
        exact_faces = analytic_velocity(mesh.face_centers)
    row = dict(affine=affine, solved=True, failure=None, polynomial=None)
    try:
        result = op.solve(body_force=force, boundary_values=boundary)
    except (RuntimeError, ValueError) as error:
        row.update(solved=False, failure=f'{type(error).__name__}: {error}')
        result = None
    if result is not None:
        internal = mesh.neighbor >= 0
        area = np.linalg.norm(mesh.area_vectors[internal], axis=1)
        energy = float(result['energy_form'])
        jump_energy = float(result['gradient_jump_energy'])
        state = np.zeros((op.N + op.F, 3))
        state[:op.N] = result['cell_values']
        state[op.N:] = result['face_values']
        resid_energy = penalty_energy(op, state.ravel())
        row.update(
            cell_relative_l2=volume_relative_error(result['cell_values'], exact, mesh.cell_volumes),
            face_relative_l2=volume_relative_error(result['face_values'][internal],
                                                 exact_faces[internal], area),
            cell_max_abs_error=float(np.max(np.abs(result['cell_values'] - exact))),
            energy_form=energy,
            gradient_jump_energy=jump_energy,
            residual_penalty_energy=resid_energy,
            physical_energy_share=float((energy - jump_energy - resid_energy) / max(abs(energy), 1e-300)),
            jump_energy_share=float(jump_energy / max(abs(energy), 1e-300)),
            residual_energy_share=float(resid_energy / max(abs(energy), 1e-300)),
            energy_balance_error=float(result['energy_balance_error']),
            linear_relative_residual=float(result['linear_relative_residual']))
    if spectrum:
        spectral = operator_spectrum(op)
        row.update({('spectrum_' + key if key != 'smallest_spectrum' else key): value
                    for key, value in spectral.items()})
    row['seconds'] = time.perf_counter() - started
    return row


def case(mesh, kind, n, stabilization, jump, *, affine=False, spectrum=True):
    row = dict(kind=kind, n=n, cells=len(mesh.cells), faces=len(mesh.owner),
               h=1. / n, stabilization=stabilization, gradient_jump=jump)
    row.update(solve_case(mesh, stabilization, jump, affine=affine, spectrum=spectrum))
    return row


def plot_map(rows, out, name='fig1_admissibility_map.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14., 4.1), constrained_layout=True)
    for axis, kind in zip(axes[:2], ('tet', 'hex')):
        subset = [r for r in rows if r['kind'] == kind and not r['affine'] and r['n'] == 3]
        stabs = sorted({r['stabilization'] for r in subset})
        jumps = sorted({r['gradient_jump'] for r in subset})
        grid = np.full((len(stabs), len(jumps)), np.nan)
        for row in subset:
            value = 100 * row['cell_relative_l2'] if row['solved'] else np.nan
            grid[stabs.index(row['stabilization']), jumps.index(row['gradient_jump'])] = value
        image = axis.imshow(np.log10(np.maximum(grid, 1e-8)), origin='lower',
                            aspect='auto', cmap='viridis', vmin=-1.3, vmax=1.7)
        for row in subset:
            i, j = stabs.index(row['stabilization']), jumps.index(row['gradient_jump'])
            if row['solved']:
                axis.plot(j, i, 'o', color='white', markersize=5, fillstyle='none')
            else:
                axis.plot(j, i, 'x', color='red', markersize=9, markeredgewidth=1.8)
            if row['solved'] and grid[i, j] is not None and np.isfinite(grid[i, j]):
                axis.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center',
                          fontsize=5.4, color='black')
        axis.set_xticks(range(len(jumps)), [f'{v:g}' for v in jumps])
        axis.set_yticks(range(len(stabs)), [f'{v:g}' for v in stabs])
        axis.set_xlabel('gradient-jump strength $s_j$')
        axis.set_ylabel('face-residual penalty $s_r$')
        axis.set_title(f'{kind} mesh (n=3): value = relative L2 [%], red x = unsolvable')
        figure.colorbar(image, ax=axis, label='log10 error [%]')
    axis = axes[2]
    for kind, color in (('tet', 'crimson'), ('hex', 'navy')):
        for stabilization, style in ((.1, ':'), (1., '-'), (10., '--'), (100., '-.')):
            subset = sorted([r for r in rows if r['kind'] == kind and not r['affine']
                             and r['n'] == 3 and r['stabilization'] == stabilization],
                            key=lambda r: r['gradient_jump'])
            good = [r for r in subset if r['solved']]
            if len(good) < 2:
                continue
            axis.loglog([max(r['gradient_jump'], 1e-3) for r in good],
                        [100 * r['cell_relative_l2'] for r in good],
                        style, color=color, marker='o', markersize=3,
                        label=f'{kind} $s_r$={stabilization:g}')
    axis.set_xlabel('gradient-jump strength $s_j$')
    axis.set_ylabel('relative L2 error [%]')
    axis.set_title('accuracy price vs the jump knob')
    axis.grid(True, which='both', alpha=.3)
    axis.legend(fontsize=6)
    figure.savefig(out / name, dpi=190)
    figure.savefig(out / name.replace('.png', '.pdf'))
    plt.close(figure)


def plot_convergence(rows, out, name='fig2_h_convergence.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.9, 4.2), constrained_layout=True)
    series = [(kind, jump) for kind in ('tet', 'hex') for jump in (.03, .1, 1.)]
    colors = {('tet', .03): 'crimson', ('tet', .1): 'darkorange', ('tet', 1.): 'goldenrod',
              ('hex', .03): 'navy', ('hex', .1): 'royalblue', ('hex', 1.): 'deepskyblue'}
    for kind, jump in series:
        subset = sorted([r for r in rows if r['kind'] == kind and r['gradient_jump'] == jump
                         and r['stabilization'] == 1. and not r['affine']
                         and r.get('cell_relative_l2') is not None], key=lambda r: r['n'])
        if len(subset) < 2:
            continue
        h = [r['h'] for r in subset]
        error = [100 * r['cell_relative_l2'] for r in subset]
        axis.loglog(h, error, 'o-', color=colors[(kind, jump)],
                    label=f'{kind} $s_j$={jump:g}')
        slopes = [np.log(error[i] / error[i + 1]) / np.log(h[i] / h[i + 1])
                  for i in range(len(h) - 1)]
        for index, slope in enumerate(slopes):
            axis.annotate(f'{slope:.2f}', (h[index], error[index]), fontsize=6,
                          textcoords='offset points', xytext=(3, 6),
                          color=colors[(kind, jump)])
    axis.set_xlabel('h = 1/n')
    axis.set_ylabel('relative cell L2 error [%]')
    axis.set_title('analytic divergence-free viscous field')
    axis.grid(True, which='both', alpha=.3)
    axis.legend(fontsize=6.5, ncol=2)
    figure.savefig(out / name, dpi=190)
    figure.savefig(out / name.replace('.png', '.pdf'))
    plt.close(figure)


def plot_coercivity(rows, out, name='fig3_coercivity.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.9, 4.2), constrained_layout=True)
    for kind, color in (('tet', 'crimson'), ('hex', 'navy')):
        for stabilization, style in ((.1, ':'), (1., '-'), (10., '--'), (100., '-.')):
            subset = sorted([r for r in rows if r['kind'] == kind and not r['affine']
                             and r['n'] == 3 and r['stabilization'] == stabilization
                             and r.get('spectrum_ratio') is not None],
                            key=lambda r: r['gradient_jump'])
            if len(subset) < 2:
                continue
            axis.loglog([max(r['gradient_jump'], 1e-3) for r in subset],
                        [max(abs(r['spectrum_ratio']), 1e-18) for r in subset],
                        style, color=color, marker='s', markersize=3,
                        label=f'{kind} $s_r$={stabilization:g}')
    axis.set_xlabel('gradient-jump strength $s_j$')
    axis.set_ylabel(r'$\lambda_{min}/\lambda_{max}$ of the free operator')
    axis.set_title('coercivity margin (bottom = singular)')
    axis.grid(True, which='both', alpha=.3)
    axis.legend(fontsize=6)
    figure.savefig(out / name, dpi=190)
    figure.savefig(out / name.replace('.png', '.pdf'))
    plt.close(figure)


GRID = [(s_r, s_j) for s_r in (.1, 1., 10., 100.)
        for s_j in (0., .01, .03, .1, .3, 1., 3., 10.)]
AFFINE_GRID = [(1., 0.), (1., 1.), (100., 1.), (100., 10.)]


STAGES = ('sweep-tet3', 'sweep-tet4', 'sweep-hex4', 'sweep-hex6',
          'affine', 'ladder', 'merge')


def _stage_rows(stage, out):
    """Run one bounded stage so every chunk fits an interactive time slice."""
    rows = []
    if stage.startswith('sweep-'):
        kind, n = stage.split('-')[1][:3], int(stage.split('-')[1][3:])
        mesh = tet_mesh(n) if kind == 'tet' else hex_mesh(n)
        for stabilization, jump in GRID:
            row = case(mesh, kind, n, stabilization, jump)
            rows.append(row)
            print('SWEEP', {k: (round(v, 12) if isinstance(v, float) else v)
                            for k, v in row.items()
                            if k in ('kind', 'n', 'stabilization', 'gradient_jump', 'solved',
                                     'cell_relative_l2', 'spectrum_ratio',
                                     'spectrum_near_null_tight', 'physical_energy_share')},
                  flush=True)
    elif stage == 'affine':
        for kind, n in (('tet', 3), ('hex', 4)):
            mesh = tet_mesh(n) if kind == 'tet' else hex_mesh(n)
            for stabilization, jump in AFFINE_GRID:
                row = case(mesh, kind, n, stabilization, jump, affine=True, spectrum=False)
                rows.append(row)
                print('AFFINE', {k: row.get(k) for k in ('kind', 'n', 'stabilization',
                                                         'gradient_jump', 'solved',
                                                         'cell_max_abs_error')}, flush=True)
    elif stage == 'ladder':
        for kind, n in (('tet', 2), ('tet', 3), ('tet', 4),
                        ('hex', 2), ('hex', 4), ('hex', 6), ('hex', 8)):
            mesh = tet_mesh(n) if kind == 'tet' else hex_mesh(n)
            for jump in ((.03, .1, 1.) if kind == 'tet' else (.1, 1.)):
                row = case(mesh, kind, n, 1., jump, spectrum=False)
                rows.append(row)
                print('LADDER', {k: row.get(k) for k in ('kind', 'n', 'gradient_jump', 'solved',
                                                         'cell_relative_l2',
                                                         'jump_energy_share')}, flush=True)
    (out / f'{stage}.json').write_text(json.dumps(rows, indent=1))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/gradient-jump-strength-sweep/run')
    parser.add_argument('--stage', choices=STAGES + ('all',), default='all')
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)
    if arguments.stage == 'merge':
        rows = []
        for stage in STAGES[:-1]:
            path = out / f'{stage}.json'
            if path.exists():
                rows.extend(json.loads(path.read_text()))
        (out / 'sweep.json').write_text(json.dumps(rows, indent=1))
        plot_map(rows, out); plot_convergence(rows, out); plot_coercivity(rows, out)
        print('MERGED', len(rows), flush=True)
        return 0
    if arguments.stage == 'all':
        closure = []
        for stage in STAGES[:-1]:
            closure.extend(_stage_rows(stage, out))
        rows = closure
        (out / 'sweep.json').write_text(json.dumps(rows, indent=1))
        plot_map(rows, out); plot_convergence(rows, out); plot_coercivity(rows, out)
        return 0
    _stage_rows(arguments.stage, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
