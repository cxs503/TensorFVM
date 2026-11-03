"""Measured truncation order of the NativeHybridConvection3D face flux.

Verification-debt closure: the merged convection operator
(``src/tensorfvm/native_hybrid_convection3d.py``) was covered by *algebraic*
identity tests only (affine exactness, global balance, energy ledger).  Its
asymptotic accuracy order -- the property behind the words "deferred linear
upwind" -- had never been measured.  This script measures it.

Reference: for the smooth analytic ABC field of ``native_manufactured3d``
(divergence free, k=0.7) the exact face-integrated momentum flux is

    F_f = \\oint_f (u . n) u dA ,

evaluated by Duffy-Gauss tensor quadrature on the actual planar facets, i.e.
independently of every solver code path.  The discrete flux is the operator's
own ``transport @ u + face_source``; the analytic velocity is supplied at cell
and face centres, so the difference is pure discretisation error.

Why the flux and not the cell residual: the cell-value form returns
sum_f q_f u_K = u_K sum_f q_f, i.e. it collapses onto the discrete divergence
defect of the volume flux.  With a discretely divergence-free q (the production
situation) that defect is zero and the cell-residual would compare an
identically vanishing operator against a nonzero continuous term.  The module
therefore transports the face mean ``R U``; the face flux is the object whose
truncation order is meaningful, and it is what ``diagnostics`` reports as
``momentum_flux``.

Run: PYTHONPATH=src python scripts/verify_native_hybrid_convection_order.py
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
from numpy.polynomial.legendre import leggauss

from tensorfvm.native_face_geometry3d import polygon_triangles
from tensorfvm.native_hybrid_convection3d import NativeHybridConvection3D
from tensorfvm.native_manufactured3d import abc_fields
from tensorfvm.native_mesh3d import NativeMesh3D

OUT = pathlib.Path(__file__).parent / 'verification-native-hybrid-convection-order'
LEVELS = (2, 4, 8)
QUADRATURE_ORDER = 6
AFFINE = np.array([[.2, .4, .1], [-.3, -.1, .2], [.1, .1, -.1]])


# ---------------------------------------------------------------- mesh family
def _cube_vertices(n: int) -> np.ndarray:
    return np.array([[x, y, z]
                     for z in np.linspace(0, 1, n + 1)
                     for y in np.linspace(0, 1, n + 1)
                     for x in np.linspace(0, 1, n + 1)])


def _hex_vertices(n: int) -> list[list[int]]:
    index = lambda x, y, z: z * (n + 1) ** 2 + y * (n + 1) + x
    return [[index(x, y, z), index(x + 1, y, z), index(x + 1, y + 1, z),
             index(x, y + 1, z), index(x, y, z + 1), index(x + 1, y, z + 1),
             index(x + 1, y + 1, z + 1), index(x, y + 1, z + 1)]
            for z in range(n) for y in range(n) for x in range(n)]


def hex_box(n: int) -> NativeMesh3D:
    cubes = _hex_vertices(n)
    return NativeMesh3D.from_cells(_cube_vertices(n), cubes,
                                   ['hexahedron'] * len(cubes),
                                   triangulate_faces=True)


def wedge_box(n: int) -> NativeMesh3D:
    """Every hex split into two prisms across its 0-2-4-6 diagonal plane.

    The cut plane contains only the two opposite corner edges, so all
    x- and y-normal faces stay whole quads: the split is conforming.
    """
    cells = []
    for c in _hex_vertices(n):
        cells.append([c[0], c[1], c[2], c[4], c[5], c[6]])
        cells.append([c[0], c[2], c[3], c[4], c[6], c[7]])
    return NativeMesh3D.from_cells(_cube_vertices(n), cells,
                                   ['wedge'] * len(cells),
                                   triangulate_faces=True)


def tet_box(n: int) -> NativeMesh3D:
    """Every hex split into the six tetrahedra around its main 0-6 diagonal.

    The induced triangulation of each cube face uses the same corner-to-corner
    diagonal, so neighbouring copies match.
    """
    pattern = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
               (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    cells = [[c[i] for i in p] for c in _hex_vertices(n) for p in pattern]
    return NativeMesh3D.from_cells(_cube_vertices(n), cells,
                                   ['tetra'] * len(cells),
                                   triangulate_faces=True)


FAMILIES = {'hex': hex_box, 'wedge': wedge_box, 'tetra': tet_box}


# ------------------------------------------------------- independent reference
def _duffy(order: int = QUADRATURE_ORDER):
    nodes, weights = leggauss(order)
    nodes = (nodes + 1.) / 2.
    weights = weights / 2.
    bary, wts = [], []
    for r, wr in zip(nodes, weights):
        for s, ws in zip(nodes, weights):
            # lambda = (1-r, r(1-s), r s); dA = 2*Area*r dr ds and the factor
            # 2*r below makes the weights sum to one over the triangle.
            bary.append((1. - r, r * (1. - s), r * s))
            wts.append(2. * r * wr * ws)
    return np.array(bary), np.array(wts)


def exact_face_flux(mesh: NativeMesh3D) -> np.ndarray:
    """Faces x 3 of \\oint (u.n) u dA for the ABC field, rho = 1."""
    bary, weights = _duffy()
    normals = mesh.area_vectors / np.linalg.norm(mesh.area_vectors, axis=1)[:, None]
    flux = np.zeros((len(mesh.owner), 3))
    for face, ids in enumerate(mesh.face_vertices):
        fan = polygon_triangles(mesh.vertices[list(ids)])
        local = np.zeros(3)
        for triangle in fan:
            nodes = np.einsum('qj,jk->qk', bary, triangle)
            velocity = abc_fields(nodes)['velocity']
            normal_flux = velocity @ normals[face]
            area = .5 * np.linalg.norm(np.cross(triangle[1] - triangle[0],
                                                triangle[2] - triangle[0]))
            local += area * np.einsum('q,q,qi->i', weights, normal_flux, velocity)
        flux[face] = local
    return flux


def exact_face_flux_affine(mesh: NativeMesh3D) -> np.ndarray:
    """Same integral for the affine, divergence-free field u = c + G x (exact)."""
    face_velocity = .4 + mesh.face_centers @ AFFINE.T
    area = np.linalg.norm(mesh.area_vectors, axis=1)
    normal_flux = np.einsum('fi,fi->f', face_velocity, mesh.area_vectors)
    return normal_flux[:, None] * face_velocity


# ------------------------------------------------------------- discrete paths
def gauss_velocity_gradient(mesh: NativeMesh3D, face_velocity: np.ndarray) -> np.ndarray:
    """Production-style Gauss gradient, gradient[c, i, j] = sum_f u_j S_i / V."""
    flux = np.einsum('fi,fj->fij', face_velocity, mesh.area_vectors)
    interior = mesh.neighbor >= 0
    gradient = np.zeros((len(mesh.cells), 3, 3))
    for i in range(3):
        accumulator = np.zeros((len(mesh.cells), 3))
        np.add.at(accumulator, mesh.owner, flux[:, i, :])
        np.add.at(accumulator, mesh.neighbor[interior], -flux[interior, i, :])
        gradient[:, i, :] = accumulator / mesh.cell_volumes[:, None]
    return gradient


def discrete(mesh, order, gradient_mode=None, affine=False):
    """Return the operator's discrete face flux plus its own diagnostics."""
    sample = (lambda p: .4 + p @ AFFINE.T) if affine else (
        lambda p: abc_fields(p)['velocity'])
    face_velocity = sample(mesh.face_centers)
    cell_velocity = sample(mesh.cell_centers)
    volume_flux = np.einsum('fi,fi->f', face_velocity, mesh.area_vectors)
    gradient = None
    if order == 2:
        if gradient_mode == 'exact':
            gradient = (np.broadcast_to(AFFINE, (len(cell_velocity), 3, 3)) if affine
                        else abc_fields(mesh.cell_centers)['velocity_gradient'])
        elif gradient_mode == 'gauss':
            gradient = gauss_velocity_gradient(mesh, face_velocity)
        else:
            raise ValueError('order two needs an explicit gradient source')
    operator = NativeHybridConvection3D(mesh, face_velocity, density=1., order=order)
    matrix, source, transport, face_source = operator.assemble(volume_flux, gradient)
    flux = transport @ cell_velocity + face_source
    diagnostics = operator.diagnostics(cell_velocity, volume_flux, gradient)
    return dict(flux=flux, operator=operator, volume_flux=volume_flux,
                cell_velocity=cell_velocity, gradient=gradient,
                matrix=matrix, source=source, diagnostics=diagnostics)


def face_error(mesh, order, gradient_mode=None) -> dict:
    exact = exact_face_flux(mesh)
    run = discrete(mesh, order, gradient_mode)
    interior = mesh.neighbor >= 0
    error = run['flux'] - exact
    relative = float(np.linalg.norm(error[interior]) / np.linalg.norm(exact[interior]))
    return dict(relative_error=relative,
                boundary_relative_error=float(
                    np.linalg.norm(error[~interior]) / np.linalg.norm(exact[~interior])),
                energy_identity_error=run['diagnostics']['energy_identity_error'],
                momentum_balance_error=run['diagnostics']['global_momentum_balance_error'],
                assembly_error=run['diagnostics']['assembly_error'],
                cells=len(mesh.cells), faces=len(mesh.owner),
                interior_faces=int(interior.sum()))


SCHEMES = {'first_order_upwind': (1, None),
           'second_order_exact_gradient': (2, 'exact'),
           'second_order_gauss_gradient': (2, 'gauss')}


def observed_order(errors) -> list[float]:
    return [math.log(errors[i] / errors[i + 1]) / math.log(LEVELS[i + 1] / LEVELS[i])
            for i in range(len(errors) - 1)]


def affine_anchor(family: str, n: int = 2) -> dict:
    """Both orders must be exact on an affine, divergence-free field."""
    mesh = FAMILIES[family](n)
    exact = exact_face_flux_affine(mesh)
    first = discrete(mesh, 1, None, affine=True)
    second = discrete(mesh, 2, 'exact', affine=True)
    # upwind uses the upstream *cell* value, which differs from the face value
    # by G (x_f - x_K): the affine residual is measured for the deferred form.
    return dict(first_order_affine_error=float(np.max(np.abs(first['flux'] - exact))),
                second_order_affine_error=float(np.max(np.abs(second['flux'] - exact))))


def cell_value_divergence_defect(family: str, n: int = 4) -> float:
    """Residual of the cell-value form = O(h^4) divergence defect of q."""
    mesh = FAMILIES[family](n)
    run = discrete(mesh, 1)
    return float(np.max(np.abs(run['matrix'] @ run['cell_velocity'].ravel()
                               + run['source'])))


def sweep() -> dict:
    results = {}
    for family, factory in FAMILIES.items():
        meshes = {n: factory(n) for n in LEVELS}
        for scheme, (order, mode) in SCHEMES.items():
            rows = []
            for n in LEVELS:
                row = face_error(meshes[n], order, mode)
                row['n'] = n
                row['h'] = 1. / n
                rows.append(row)
            results[f'{family}/{scheme}'] = dict(
                rows=rows,
                observed_order=observed_order([r['relative_error'] for r in rows]))
    return results


def volume_audit() -> dict:
    audit = {}
    for family, factory in FAMILIES.items():
        for n in LEVELS:
            mesh = factory(n)
            audit[f'{family}/n={n}'] = dict(
                volume_sum=float(np.sum(mesh.cell_volumes)),
                cells=len(mesh.cells), min_volume=float(np.min(mesh.cell_volumes)))
    return audit


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    payload = dict(levels=list(LEVELS), quadrature_order=QUADRATURE_ORDER,
                   field='ABC amplitude .5 wavenumber .7',
                   family_volume_audit=volume_audit(),
                   affine_anchor={f: affine_anchor(f) for f in FAMILIES},
                   cell_value_divergence_defect={f: cell_value_divergence_defect(f) for f in FAMILIES},
                   results=sweep())
    (OUT / 'evidence.json').write_text(json.dumps(payload, indent=2))
    plots(payload)
    print(report_table(payload))
    return payload


def plots(payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    markers = {'hex': 'o', 'wedge': 's', 'tetra': '^'}
    colors = {'first_order_upwind': 'tab:red',
              'second_order_exact_gradient': 'tab:blue',
              'second_order_gauss_gradient': 'tab:green'}
    for key, block in payload['results'].items():
        family, scheme = key.split('/')
        h = [row['h'] for row in block['rows']]
        e = [row['relative_error'] for row in block['rows']]
        axes[0].loglog(h, e, marker=markers[family], color=colors[scheme],
                       ls='-' if family == 'hex' else '--',
                       label=f'{family} {scheme.replace("_", " ")}')
    reference = np.array([1 / 8, 1 / 2])
    axes[0].loglog(reference, 4e-1 * reference, 'k:', label='slope 1')
    axes[0].loglog(reference, 4e-1 * reference ** 2, 'k-.', label='slope 2')
    axes[0].set_xlabel('h = 1/n'); axes[0].set_ylabel('interior-face flux error')
    axes[0].set_title('measured truncation order (reference = analytic integral)')
    axes[0].invert_xaxis(); axes[0].grid(True, which='both', alpha=.3)
    axes[0].legend(fontsize=6)

    mesh = FAMILIES['wedge'](4)
    exact = exact_face_flux(mesh)
    run = discrete(mesh, 2, 'exact')
    error = np.linalg.norm(run['flux'] - exact, axis=1)
    interior = mesh.neighbor >= 0
    artist = axes[1].scatter(mesh.face_centers[interior, 0],
                             mesh.face_centers[interior, 1],
                             c=error[interior], s=18, marker='s')
    axes[1].set_aspect('equal')
    axes[1].set_title('order-2 face-flux error, wedge n=4 (all z)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    figure.colorbar(artist, ax=axes[1], label='|discrete - exact| flux')
    figure.tight_layout()
    figure.savefig(OUT / 'convection-order.png', dpi=140)
    plt.close(figure)


def report_table(payload) -> str:
    lines = ['| family / scheme | cells n=2/4/8 | interior-face flux error | observed order |',
             '|---|---|---|---|']
    for key, block in payload['results'].items():
        rows = block['rows']
        errors = ' | '.join(f'{row["relative_error"]:.3e}' for row in rows)
        orders = ', '.join(f'{value:.2f}' for value in block['observed_order'])
        cells = '/'.join(str(row['cells']) for row in rows)
        lines.append(f'| {key} | {cells} | {errors} | {orders} |')
    worst_energy = max(row['energy_identity_error']
                       for block in payload['results'].values() for row in block['rows'])
    worst_balance = max(row['momentum_balance_error']
                        for block in payload['results'].values() for row in block['rows'])
    lines += ['', f'worst energy identity error: {worst_energy:.3e}',
              f'worst global momentum balance error: {worst_balance:.3e}']
    for family, block in payload['affine_anchor'].items():
        lines.append(f'{family} affine anchor: order1 {block["first_order_affine_error"]:.3e}, '
                     f'order2 {block["second_order_affine_error"]:.3e}')
    for family, value in payload['cell_value_divergence_defect'].items():
        lines.append(f'{family} cell-value form residual (h^4 divergence defect): {value:.3e}')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()