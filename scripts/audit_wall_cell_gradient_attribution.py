"""Attribution of the O(1) wall-adjacent cell divergence of the 3-D viscous operator.

The merged operator (:class:`tensorfvm.native_viscous3d.NativeViscous3D`) leaves
``O(1)`` truncation error in the cells that touch a Dirichlet wall while the
interior refines at second order.  Two competing explanations were on record:

1. the wall-normal derivative is a half-cell central difference, so its ``O(h)``
   error produces a non-refining outflux (``native_order3d.wall_error_model``);
2. the tangential components of the wall-face gradient are not extrapolated to
   the face (``docs`` note of the wall-cell truncation audit).

This script tests both by oracle substitution.  The assembled flux is
reproduced exactly from the operator's own face-gradient tensor
(:func:`tensorfvm.verification.face_lsq3d.operator_face_gradients`, agreement
``2e-16``), so components of that tensor can be replaced one at a time and the
effect on the cell divergence measured without touching the operator.

Rows, per mesh level ``n``:

``as_built``                      operator flux, untouched
``wall_normal_exact``             wall faces: normal derivative replaced by the analytic one
``wall_full_exact``               wall faces: whole gradient replaced by the analytic one
``mirror_average``                wall faces: mean of the owner-cell gradient and the analytic
                                  gradient at the mirrored point across the wall
``all_faces_exact``               every face gradient replaced by the analytic one (ceiling)
``interp_exact_cell_gradients``   exact *cell-centre* gradients averaged to faces (as-built stencil)
``lsq_degree1`` / ``lsq_degree2`` first/second-order Taylor LSQ reconstruction at every face
                                  (:mod:`tensorfvm.verification.face_lsq3d`)
``amplification_law``             check of ``err_div ~ mu*|dg_face|/h``

Usage::

    PYTHONPATH=src python3 scripts/audit_wall_cell_gradient_attribution.py --ns 4 6 8
"""
import argparse
import json
import os

import numpy as np

from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification import face_lsq3d as fl
from tensorfvm.verification import native_order3d as order

VISCOSITY = 1.3


def _split_error(mesh, flux, n):
    divergence = fl.divergence(mesh, flux)
    exact = order.divergence_of_stress(mesh.cell_centers, VISCOSITY)
    error = np.linalg.norm(divergence - exact, axis=1)
    wall = order._wall_mask(mesh, n)
    return (float(np.sqrt(np.mean(error[wall]**2))),
            float(error[wall].max()),
            float(np.sqrt(np.mean(error[~wall]**2))),
            float(error[~wall].max()))


def study(n):
    mesh = order.hex_box(n, 0.0)
    wall = mesh.neighbor < 0
    velocity = order.manufactured(mesh.cell_centers)
    wall_values = order.manufactured(mesh.face_centers[wall])
    operator = NativeViscous3D(mesh, wall, VISCOSITY)
    data = np.zeros((len(mesh.face_centers), 3))
    data[wall] = wall_values
    assembled = operator.momentum_flux(velocity, dirichlet_values=data)
    built, gradients = fl.operator_face_gradients(operator, velocity, data)
    consistency = float(np.abs(built - assembled).max())
    exact_face = order.manufactured_gradient(mesh.face_centers)
    normal = operator.normal

    rows = {'as_built': _split_error(mesh, built, n)}

    def replace(mask, values):
        trial = gradients.copy()
        trial[mask] = values
        return fl.traction_flux(mesh, trial, VISCOSITY)

    normal_built = np.einsum('fab,fb->fa', gradients[wall], normal[wall])
    normal_exact = np.einsum('fab,fb->fa', exact_face[wall], normal[wall])
    rows['wall_normal_exact'] = _split_error(mesh, replace(
        wall, gradients[wall]
        + (normal_exact - normal_built)[:, :, None]*normal[wall][:, None, :]), n)
    rows['wall_full_exact'] = _split_error(mesh, replace(wall, exact_face[wall]), n)

    owner = mesh.owner[wall]
    mirror = 2*mesh.face_centers[wall] - mesh.cell_centers[owner]
    rows['mirror_average'] = _split_error(mesh, replace(
        wall, 0.5*(gradients[wall] + order.manufactured_gradient(mirror))), n)
    rows['all_faces_exact'] = _split_error(
        mesh, fl.traction_flux(mesh, exact_face, VISCOSITY), n)

    cell_exact = order.manufactured_gradient(mesh.cell_centers)
    averaged = np.zeros_like(exact_face)
    for b in range(3):
        averaged[:, :, b] = operator.interpolation @ cell_exact[:, :, b]
    rows['interp_exact_cell_gradients'] = _split_error(
        mesh, fl.traction_flux(mesh, averaged, VISCOSITY), n)

    diagnostics = {'consistency': consistency, 'n': n, 'h': 1.0/n,
                   'faces': int(len(mesh.face_centers)),
                   'cells': int(len(mesh.cells)),
                   'wall_faces': int(wall.sum())}
    for degree in (1, 2):
        reconstructed, info = fl.reconstruct(
            mesh, velocity, degree=degree, rings=2, dirichlet=wall,
            dirichlet_values=wall_values)
        rows['lsq_degree%d' % degree] = _split_error(
            mesh, fl.traction_flux(mesh, reconstructed, VISCOSITY), n)
        diagnostics['lsq_degree%d_wall_gradient_error' % degree] = float(np.sqrt(np.mean(
            np.sum((reconstructed[wall] - exact_face[wall])**2, axis=(1, 2)))))
        diagnostics['lsq_degree%d_condition_max' % degree] = float(info['condition'].max())
    diagnostics['as_built_wall_gradient_error'] = float(np.sqrt(np.mean(
        np.sum((gradients[wall] - exact_face[wall])**2, axis=(1, 2)))))
    diagnostics['as_built_internal_gradient_error'] = float(np.sqrt(np.mean(
        np.sum((gradients[~wall] - exact_face[~wall])**2, axis=(1, 2)))))
    # err_div ~ mu*|dg|/h at the wall cells, i.e. the flux is normalised by the
    # cell volume while the face area only scales with h**2.
    diagnostics['predicted_wall_error_from_wall_gradient'] = float(
        VISCOSITY*diagnostics['as_built_wall_gradient_error']/(1.0/n))
    diagnostics['predicted_wall_error_from_lsq2'] = float(
        VISCOSITY*diagnostics['lsq_degree2_wall_gradient_error']/(1.0/n))
    return rows, diagnostics


def orders(records, key, field=0):
    values = [record[0][key][field] for record in records]
    steps = [(records[i][1]['h'], records[i + 1][1]['h'], values[i], values[i + 1])
             for i in range(len(records) - 1)]
    return [float(np.log(a/b)/np.log(h1/h2)) for (h1, h2, a, b) in steps], values


def figure(records, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = ['as_built', 'wall_normal_exact', 'wall_full_exact', 'mirror_average',
            'interp_exact_cell_gradients', 'lsq_degree1', 'lsq_degree2', 'all_faces_exact']
    labels = {'as_built': 'as-built', 'wall_normal_exact': 'wall face: exact $\\partial_n u$ only',
              'wall_full_exact': 'wall face: exact gradient',
              'mirror_average': 'wall face: mirror-average (exact mirror)',
              'interp_exact_cell_gradients': 'exact cell gradients, averaged to faces',
              'lsq_degree1': 'Taylor LSQ degree 1 (all faces)',
              'lsq_degree2': 'Taylor LSQ degree 2 (all faces)',
              'all_faces_exact': 'all face gradients exact (ceiling)'}
    hs = [row['h'] for _, row in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for key in keys:
        values, _ = orders(records, key, 0)
        sequence = [record[0][key][0] for record in records]
        _, = axes[0].loglog(hs, sequence, marker='o', label=labels[key])
        axes[1].loglog(hs, [record[0][key][2] for record in records], marker='o',
                       linestyle='--', alpha=.7)
    axes[0].set_xlabel('h = 1/n'); axes[0].set_ylabel('L2 divergence error, wall cells')
    axes[0].set_title('wall-adjacent cells'); axes[0].invert_xaxis()
    axes[0].legend(fontsize=6.5, loc='lower right'); axes[0].grid(True, which='both', alpha=.3)
    axes[1].set_xlabel('h = 1/n'); axes[1].set_ylabel('L2 divergence error, interior cells')
    axes[1].set_title('interior cells (dashed)'); axes[1].invert_xaxis()
    axes[1].grid(True, which='both', alpha=.3)
    fig.suptitle('Where the O(1) wall-cell truncation error comes from '
                 '(manufactured field, $\\mu$=1.3)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns', type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument('--out', default='outputs/wall_cell_gradient_attribution')
    arguments = parser.parse_args()
    records = []
    for n in arguments.ns:
        rows, diagnostics = study(n)
        records.append((rows, diagnostics))
        print('n=%d h=%.4f faces=%d wall_faces=%d consistency=%.2e'
              % (n, diagnostics['h'], diagnostics['faces'],
                 diagnostics['wall_faces'], diagnostics['consistency']), flush=True)
        for key, value in rows.items():
            print('   %-30s wall l2=%.4e linf=%.4e | interior l2=%.4e linf=%.4e' % (
                key, value[0], value[1], value[2], value[3]), flush=True)
        print('   gradient error: as-built wall %.3e internal %.3e | LSQ1 %.3e | LSQ2 %.3e'
              % (diagnostics['as_built_wall_gradient_error'],
                 diagnostics['as_built_internal_gradient_error'],
                 diagnostics['lsq_degree1_wall_gradient_error'],
                 diagnostics['lsq_degree2_wall_gradient_error']), flush=True)
        print('   amplification law mu*|dg|/h: wall-gradient %.3e (measured as-built %.3e), '
              'LSQ2 %.3e (measured %.3e)' % (
                  diagnostics['predicted_wall_error_from_wall_gradient'],
                  rows['as_built'][0], diagnostics['predicted_wall_error_from_lsq2'],
                  rows['lsq_degree2'][0]), flush=True)

    summary = {}
    for key in records[0][0]:
        window_orders, values = orders(records, key, 0)
        interior_orders, _ = orders(records, key, 2)
        summary[key] = {'wall_l2': values, 'wall_orders': window_orders,
                        'interior_orders': interior_orders}
    result = {'ns': arguments.ns, 'viscosity': VISCOSITY,
              'diagnostics': [row for _, row in records], 'rows': summary}
    os.makedirs(os.path.dirname(arguments.out), exist_ok=True)
    with open(arguments.out + '.json', 'w') as stream:
        json.dump(result, stream, indent=1)
    figure(records, arguments.out + '.png')
    print('\nwall-cell orders:')
    for key, value in summary.items():
        print('  %-30s %s' % (key, ' '.join('%.2f' % order for order in value['wall_orders'])))
    print('written %s.json / %s.png' % (arguments.out, arguments.out))


if __name__ == '__main__':
    main()