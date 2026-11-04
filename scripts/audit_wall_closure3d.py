"""Audit the ``wall_closure='three_point'`` Dirichlet closure.

Produces ``docs/verification-native-wall-closure/audit.json`` plus the figures
used by that report.  Every number quoted in the report is written here, so the
report is regenerable with::

    python scripts/audit_wall_closure3d.py

The four measured objects are

``invariants``
    upgraded-face count, affine traction error, quadratic face-normal
    derivative error, for both closures and both face topologies;
``truncation``
    cell-divergence error of the assembled traction against the analytic
    ``div(tau)`` of the manufactured field, split into wall / interior cells;
``solve``
    Dirichlet solve of the consistent manufactured problem (analytic stress
    divergence imposed as the cell source) against the analytic field;
``observation order``
    two-level ``log2`` slopes of the above.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'src'))

from tensorfvm.native_viscous3d import NativeViscous3D          # noqa: E402
from tensorfvm.verification import native_order3d as study     # noqa: E402

VISCOSITY = 1.3
CLOSURES = ('two_point', 'three_point')
QUAD_LEVELS = (4, 6, 8, 12)
TRI_LEVELS = (4, 8, 12)
OUT = pathlib.Path(__file__).resolve().parents[1]/'docs'/'verification-native-wall-closure'


def build(n, closure, triangulate=False):
    mesh = study.hex_box(n, 0.0, triangulate_faces=triangulate)
    return mesh, NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY, wall_closure=closure)


def invariants(n=6):
    records = {}
    matrix = np.array([[.3, .1, -.2], [.05, -.4, .2], [.1, .2, .6]])
    shift = np.array([.2, -.1, .3])

    def field(points):
        return np.einsum('ia,ab,ib->i', points, matrix, points) + points @ shift

    def gradient(points):
        return points @ (matrix + matrix.T) + shift

    affine = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
    base = np.array([1.2, .8, -.4])
    symmetric = affine + affine.T - 2/3*np.trace(affine)*np.eye(3)
    for closure in CLOSURES:
        for triangulate in (False, True):
            mesh, op = build(n, closure, triangulate)
            wall_values = np.zeros((op.F, 3))
            wall_values[op.dirichlet] = field(mesh.face_centers[op.dirichlet])[:, None]
            face_gradient = np.empty((op.F, 3))
            source = op.face_gradient_source(0.0, wall_values[:, 0])
            for k in range(3):
                face_gradient[:, k] = op.face_gradient_matrices[k] @ field(mesh.cell_centers) + source[:, k]
            normal = np.einsum('fb,fb->f', face_gradient, op.normal)
            exact = np.einsum('fb,fb->f', gradient(mesh.face_centers), op.normal)
            data = np.zeros((op.F, 3))
            data[op.dirichlet] = base + mesh.face_centers[op.dirichlet] @ affine.T
            flux = op.momentum_flux(base + mesh.cell_centers @ affine.T, dirichlet_values=data)
            records[f'{closure}/{"triangulated" if triangulate else "quad"}'] = dict(
                upgraded_faces=int(op.wall_second_order_faces),
                dirichlet_faces=int(op.dirichlet.sum()),
                quadratic_normal_derivative_error=float(np.abs(normal - exact)[op.dirichlet].max()),
                affine_traction_error=float(np.abs(flux - VISCOSITY*mesh.area_vectors @ symmetric.T).max()))
    return records


def truncation(n, closure, triangulate=False):
    mesh, op = build(n, closure, triangulate)
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = study.manufactured(mesh.face_centers[op.dirichlet])
    flux = op.momentum_flux(study.manufactured(mesh.cell_centers), dirichlet_values=data)
    discrete = op.op.integrate(flux)/mesh.cell_volumes[:, None]
    exact = study.divergence_of_stress(mesh.cell_centers, VISCOSITY)
    error = np.linalg.norm(discrete - exact, axis=1)
    wall = study._wall_mask(mesh, n)
    return dict(n=n, h=1.0/n, upgraded_faces=int(op.wall_second_order_faces),
                dirichlet_faces=int(op.dirichlet.sum()),
                l2_wall=float(np.sqrt(np.mean(error[wall]**2))),
                linf_wall=float(error[wall].max()),
                l2_interior=float(np.sqrt(np.mean(error[~wall]**2))),
                linf_interior=float(error[~wall].max()),
                field_scale=float(np.linalg.norm(exact, axis=1).max()))


def solve(n, closure, triangulate=False):
    from scipy.sparse.linalg import spsolve
    mesh, op = build(n, closure, triangulate)
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = study.manufactured(mesh.face_centers[op.dirichlet])
    rhs = -(op.vector_divergence_matrix @ op.momentum_flux_source(dirichlet_values=data).ravel())
    rhs = rhs + (study.divergence_of_stress(mesh.cell_centers, VISCOSITY)
                 * mesh.cell_volumes[:, None]).ravel()
    solved = spsolve(op.momentum_diffusion_matrix.tocsc(), rhs)
    residual = np.linalg.norm(op.momentum_diffusion_matrix @ solved - rhs)/max(np.linalg.norm(rhs), 1e-30)
    solved = solved.reshape(len(mesh.cells), 3)
    error = np.linalg.norm(solved - study.manufactured(mesh.cell_centers), axis=1)
    wall = study._wall_mask(mesh, n)
    return dict(n=n, h=1.0/n, l2=float(np.sqrt(np.mean(error**2))), linf=float(error.max()),
                l2_wall=float(np.sqrt(np.mean(error[wall]**2))),
                l2_interior=float(np.sqrt(np.mean(error[~wall]**2))),
                linear_residual=float(residual))


def slopes(rows, keys):
    return {key: [float(np.log2(rows[i][key]/rows[i+1][key])) for i in range(len(rows)-1)] for key in keys}


def figures(payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for closure, style in zip(CLOSURES, ('o-', 's--')):
        rows = payload['truncation'][f'quad/{closure}']
        h = [row['h'] for row in rows]
        axes[0].loglog(h, [row['l2_wall'] for row in rows], style, label=f'wall, {closure}')
        axes[0].loglog(h, [row['l2_interior'] for row in rows], style, alpha=.45,
                       label=f'interior, {closure}')
        rows = payload['solve'][f'quad/{closure}']
        axes[1].loglog(h, [row['l2'] for row in rows], style, label=closure)
        axes[1].loglog(h, [row['l2_wall'] for row in rows], style, alpha=.45, label=f'{closure} (wall cells)')
    h = np.array(payload['truncation']['quad/two_point'])[:, 'h'] if False else np.array([r['h'] for r in payload['truncation']['quad/two_point']])
    axes[0].loglog(h, 0.5*(h/h[0])**2*payload['truncation']['quad/two_point'][0]['l2_interior'],
                   'k:', label='slope 2 reference')
    axes[1].loglog(h, payload['solve']['quad/two_point'][0]['l2']*(h/h[0])**2, 'k:', label='slope 2 reference')
    axes[0].set(xlabel='h', ylabel='cell-divergence error', title='truncation error (quad faces)')
    axes[1].set(xlabel='h', ylabel='Dirichlet solve error', title='solution error (quad faces)')
    for axis in axes:
        axis.grid(True, which='both', alpha=.25)
        axis.legend(fontsize=7)
    figure.suptitle('three_point wall closure: repaired, guarded, and still not the plateau cure')
    figure.tight_layout()
    figure.savefig(OUT/'convergence.png', dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    labels = list(payload['invariants'])
    quad = [payload['invariants'][key]['quadratic_normal_derivative_error'] for key in labels]
    axis.barh(labels, np.maximum(quad, 1e-18))
    axis.set_xscale('log')
    axis.set_xlabel('max |d(u)/dn - exact| on Dirichlet faces (quadratic field)')
    axis.set_title('closure exactness, n=6 (repaired three_point vs two_point)')
    axis.grid(True, axis='x', alpha=.3)
    figure.tight_layout()
    figure.savefig(OUT/'exactness.png', dpi=150)
    plt.close(figure)


def main():
    payload = dict(
        invariants=invariants(),
        truncation={f'quad/{c}': [truncation(n, c) for n in QUAD_LEVELS] for c in CLOSURES},
        solve={f'quad/{c}': [solve(n, c) for n in QUAD_LEVELS] for c in CLOSURES},
        triangulated={f'tri/{c}': [truncation(n, c, True) for n in TRI_LEVELS] for c in CLOSURES},
        slopes={})
    for closure in CLOSURES:
        payload['slopes'][f'truncation/quad/{closure}'] = slopes(
            payload['truncation'][f'quad/{closure}'], ('l2_wall', 'linf_wall', 'l2_interior'))
        payload['slopes'][f'solve/quad/{closure}'] = slopes(
            payload['solve'][f'quad/{closure}'], ('l2', 'linf', 'l2_wall', 'l2_interior'))
    ratio = {f'n={row["n"]}': payload['solve'][f'quad/three_point'][i]['l2']/row['l2']
             for i, row in enumerate(payload['solve']['quad/two_point'])}
    payload['solve_error_ratio'] = ratio
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(payload, indent=1, sort_keys=True))
    figures(payload)
    print(json.dumps(dict(
        invariants=payload['invariants'],
        solve_error_ratio=ratio,
        slopes=payload['slopes']), indent=1, sort_keys=True))


if __name__ == '__main__':
    main()