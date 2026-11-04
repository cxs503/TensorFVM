"""Validation debt: is the native 3-D viscous Dirichlet solve actually convergent?

Runs the corrected order-of-accuracy protocol of
:mod:`tensorfvm.verification.native_order3d` and writes ``runs/`` JSON + figure.

The historical test ``test_dirichlet_diffusion_solution_shows_no_measurable_convergence``
pinned "no convergence" for the Dirichlet diffusion solve.  That protocol fed the
manufactured field, whose ``div(tau(u)) = mu*lap u`` is O(10) and never vanishes,
so the discrete solution converges to the solution of the homogeneous problem
instead and the error saturates by construction.  Here the same operator is
driven by :func:`~tensorfvm.verification.native_order3d.harmonic_velocity`
(``lap u = 0``, ``div u = 0`` => ``div tau = 0``), i.e. a field that really is the
solution, so the slope is a genuine order.

    python scripts/verify_native_viscous_dirichlet_order.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from tensorfvm.verification import native_order3d as study       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')
VISCOSITY = 1.3
HARMONIC_GRIDS = (4, 8, 12, 16)
DIAGNOSTIC_GRIDS = (4, 8, 12)


def figure(truncation, faces, harmonic, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    h = [r['h'] for r in harmonic]
    axes[0].loglog(h, [r['l2_global'] for r in harmonic], 'o-', label='harmonic (exact solution)')
    axes[0].loglog(h, [r['l2_interior'] for r in harmonic], 'o--', label='harmonic, interior cells')
    axes[0].loglog(h, [r['l2_wall'] for r in harmonic], 'o:', label='harmonic, wall cells')
    ref = harmonic[0]['l2_global']
    axes[0].loglog(h, [ref*(x/h[0]) for x in h], 'k-.', lw=.8, label='slope 1 reference')
    axes[0].set_xlabel('h'); axes[0].set_ylabel('L2 error'); axes[0].legend(fontsize=7)
    axes[0].set_title('Dirichlet solve vs an exact solution')
    axes[0].grid(True, which='both', alpha=.3)

    ht = [r['h'] for r in truncation]
    axes[1].loglog(ht, [r['l2_interior'] for r in truncation], 's-', label='interior cells')
    axes[1].loglog(ht, [r['l2_wall'] for r in truncation], 's--', label='wall cells (O(1))')
    axes[1].loglog(ht, [r['l2_wall_wall_faces_only'] for r in truncation], 's:',
                   label='wall cells, Dirichlet face deficit only')
    axes[1].loglog(ht, [truncation[0]['l2_interior']*(x/ht[0])**2 for x in ht], 'k-.', lw=.8, label='slope 2 reference')
    axes[1].set_xlabel('h'); axes[1].set_ylabel('L2 error of div(tau)_h')
    axes[1].set_title('Cell-divergence truncation'); axes[1].legend(fontsize=7)
    axes[1].grid(True, which='both', alpha=.3)

    hf = [r['h'] for r in faces]
    axes[2].loglog(hf, [r['rel_wall_normal_derivative'] for r in faces], 'd-', label='wall du/dn (relative)')
    axes[2].loglog(hf, [r['rel_internal_per_area'] for r in faces], 'd--', label='internal face traction (relative)')
    axes[2].loglog(hf, [faces[0]['rel_wall_normal_derivative']*(x/hf[0]) for x in hf], 'k-.', lw=.8, label='slope 1 reference')
    axes[2].set_xlabel('h'); axes[2].set_title('Wall closure vs interior faces'); axes[2].legend(fontsize=7)
    axes[2].grid(True, which='both', alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    harmonic = [study.harmonic_solution_study(n, 0.0, VISCOSITY) for n in HARMONIC_GRIDS]
    truncation = [study.truncation_study(n, 0.0, VISCOSITY) for n in DIAGNOSTIC_GRIDS]
    faces = [study.face_traction_study(n, 0.0, VISCOSITY) for n in DIAGNOSTIC_GRIDS]
    for row in harmonic:
        print('harmonic', json.dumps(row), flush=True)

    summary = dict(
        harmonic_l2_global=study.orders(harmonic, 'l2_global'),
        harmonic_l2_interior=study.orders(harmonic, 'l2_interior'),
        harmonic_l2_wall=study.orders(harmonic, 'l2_wall'),
        harmonic_linf_global=study.orders(harmonic, 'linf_global'),
        truncation_l2_interior=study.orders(truncation, 'l2_interior'),
        truncation_l2_wall=study.orders(truncation, 'l2_wall'),
        wall_flux_share=[r['l2_wall_wall_faces_only']/r['l2_wall'] for r in truncation],
        wall_normal_derivative=study.orders(faces, 'rel_wall_normal_derivative'),
        internal_face_traction=study.orders(faces, 'rel_internal_per_area'))
    png = os.path.join(OUT, 'native_viscous_dirichlet_order.png')
    payload = dict(operator='NativeViscous3D', viscosity=VISCOSITY,
                   harmonic_grids=list(HARMONIC_GRIDS),
                   diagnostic_grids=list(DIAGNOSTIC_GRIDS),
                   harmonic=harmonic, truncation=truncation, face_traction=faces,
                   orders=summary, figure=os.path.basename(png))
    json.dump(payload, open(os.path.join(OUT, 'native_viscous_dirichlet_order.json'), 'w'), indent=1)
    figure(truncation, faces, harmonic, png)
    print(json.dumps(summary, indent=1), flush=True)
    print('wrote', png, flush=True)


if __name__ == '__main__':
    main()
