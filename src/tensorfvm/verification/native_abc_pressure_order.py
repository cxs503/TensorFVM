"""Pressure-order attribution for the fixed-parameter 3D ABC manufactured case.

Open question (docs/verification-coverage-audit.md, high priority #1)
-----------------------------------------------------------------------
The committed ABC spatial verification (docs/verification-native-abc3d-spatial)
shows the velocity converging at ~1.88 order while the mean-removed pressure
converges at only ~1.45/1.64 order on n=3,6,12.  The audit asked whether that
sub-second-order pressure behaviour is *expected* for this discretisation or a
*fixable defect*.

Method
------
The same fixed-parameter ABC field (amplitude .5, wavenumber .7, nu .1, rho 1,
skewed logical cube) is solved in two limits on the identical meshes:

* **Stokes limit** -- ``convection=False`` and the independent body force
  ``grad p/rho + nu k^2 u`` (the ``(u.grad)u`` part of the acceleration is
  dropped, so the momentum equation is linear and the solve has no Picard
  iteration).
* **full NS** -- ``convection=True`` with the full analytic acceleration.

If the linear Stokes limit reproduces the same pressure order as the full
nonlinear case, the order reduction cannot be caused by the convection
discretisation or by the nonlinear iteration; it is a property of the
collocated pressure-continuity coupling (the pressure is a Lagrange
multiplier, whose L2 error is classically one order below the velocity).

The regional breakdown (interior cells vs boundary-adjacent cells) separates a
global consistency limit from a boundary-layer effect, and the discrete
continuity residual confirms the mass equation is enforced to machine
precision in every run.
"""

from __future__ import annotations

import math

import numpy as np

from ..native_collocated3d import NativeSteadySettings, NativeSteadySolver3D
from ..native_manufactured3d import (abc_fields, exact_boundary_flux,
                                     volume_weighted_error)
from ..native_mesh3d import NativeMesh3D

__all__ = ["SHEAR", "AMPLITUDE", "WAVENUMBER", "NU", "RHO", "skew_box",
           "run_case", "observed_order", "pressure_order_study"]

SHEAR = np.array([[1., .18, .12], [0., 1., .2], [0., 0., 1.]])
AMPLITUDE = .5
WAVENUMBER = .7
NU = .1
RHO = 1.


def skew_box(n: int):
    """Skewed hexahedral box, identical to scripts/verify_native_abc_convergence3d.py."""
    if not isinstance(n, int) or n < 2:
        raise ValueError('grid level must be an integer >= 2')
    points = np.array([[x, y, z] for z in np.linspace(0, 1, n + 1)
                       for y in np.linspace(0, 1, n + 1)
                       for x in np.linspace(0, 1, n + 1)]) @ SHEAR.T
    ix = lambda x, y, z: z * (n + 1) ** 2 + y * (n + 1) + x
    cells = []
    for z in range(n):
        for y in range(n):
            for x in range(n):
                cells.append([ix(x, y, z), ix(x + 1, y, z), ix(x + 1, y + 1, z),
                            ix(x, y + 1, z), ix(x, y, z + 1), ix(x + 1, y, z + 1),
                            ix(x + 1, y + 1, z + 1), ix(x, y + 1, z + 1)])
    return NativeMesh3D.from_cells(points, cells, ['hexahedron'] * len(cells),
                                  triangulate_faces=True)


def _settings(convection: bool) -> NativeSteadySettings:
    # Identical to the committed ABC spatial verification so the two studies
    # are directly comparable.
    return NativeSteadySettings(viscosity=NU, density=RHO, convection=convection,
                                convection_order=2, bounded_convection=False,
                                nonlinear_tolerance=1e-9, linear_tolerance=1e-11,
                                max_iterations=100, relaxation=.8)


def run_case(n: int, convection: bool) -> dict:
    """Solve one ABC case and return global + regional errors (percent)."""
    mesh = skew_box(n)
    exact = abc_fields(mesh.cell_centers, amplitude=AMPLITUDE, wavenumber=WAVENUMBER,
                         nu=NU, rho=RHO)
    boundary = abc_fields(mesh.face_centers, amplitude=AMPLITUDE, wavenumber=WAVENUMBER,
                          nu=NU, rho=RHO)['velocity']
    flux = exact_boundary_flux(mesh, lambda pts: abc_fields(
        pts, amplitude=AMPLITUDE, wavenumber=WAVENUMBER, nu=NU, rho=RHO))
    if convection:
        force = exact['acceleration']
    else:
        force = exact['pressure_gradient'] / RHO + NU * WAVENUMBER ** 2 * exact['velocity']
    solver = NativeSteadySolver3D(mesh, mesh.neighbor < 0, boundary,
                                 body_force=force, boundary_flux=flux,
                                 gauge_value=exact['pressure'][0],
                                 settings=_settings(convection))
    state = solver.solve()
    if not state['converged']:
        raise RuntimeError(f'ABC pressure-order case n={n} convection={convection} did not converge')

    own, nb = mesh.owner, mesh.neighbor
    adjacent = np.zeros(len(mesh.cell_volumes), bool)
    adjacent[np.unique(own[np.flatnonzero(nb < 0)])] = True
    interior = ~adjacent

    out = dict(n=n, cells=int(len(mesh.cell_volumes)), convection=bool(convection),
               converged=True,
               velocity=volume_weighted_error(state['velocity'], exact['velocity'],
                                              mesh.cell_volumes) * 100,
               pressure=volume_weighted_error(state['pressure'], exact['pressure'],
                                               mesh.cell_volumes, remove_mean=True) * 100,
               boundary_velocity=volume_weighted_error(state['velocity'][adjacent],
                                                        exact['velocity'][adjacent],
                                                        mesh.cell_volumes[adjacent]) * 100,
               boundary_pressure=volume_weighted_error(state['pressure'][adjacent],
                                                       exact['pressure'][adjacent],
                                                       mesh.cell_volumes[adjacent],
                                                       remove_mean=True) * 100)
    if interior.sum() >= 4:
        out['interior_velocity'] = volume_weighted_error(state['velocity'][interior],
                                                        exact['velocity'][interior],
                                                        mesh.cell_volumes[interior]) * 100
        out['interior_pressure'] = volume_weighted_error(state['pressure'][interior],
                                                         exact['pressure'][interior],
                                                         mesh.cell_volumes[interior],
                                                         remove_mean=True) * 100
    # Discrete continuity: divergence of the converged face flux (machine zero).
    out['continuity_residual'] = float(np.linalg.norm(
        solver.viscous.divergence_matrix @ state['flux']))
    return out


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Observed convergence order from two error levels (both must be positive)."""
    if coarse <= 0 or fine <= 0:
        raise ValueError('positive errors required for an observed order')
    return math.log(coarse / fine) / math.log(ratio)


def pressure_order_study(grids=(3, 6, 12)) -> dict:
    """Run the Stokes-limit and full-NS ABC cases and attribute the pressure order.

    Returns a dict with per-grid records for both limits plus the derived
    orders and the Stokes-vs-NS order agreement (the attribution evidence).
    """
    grids = tuple(grids)
    if len(grids) < 2 or any(g <= 1 for g in grids):
        raise ValueError('at least two grid levels >= 2 required')
    result = dict(grids=list(grids), amplitude=AMPLITUDE, wavenumber=WAVENUMBER,
                  nu=NU, rho=RHO, shear=SHEAR.tolist(),
                  stokes=[], full_ns=[])
    for convection, key in ((False, 'stokes'), (True, 'full_ns')):
        for n in grids:
            result[key].append(run_case(n, convection))
    for key in ('stokes', 'full_ns'):
        rows = result[key]
        for i in range(1, len(rows)):
            rows[i]['pressure_order'] = observed_order(rows[i - 1]['pressure'],
                                                      rows[i]['pressure'])
            rows[i]['velocity_order'] = observed_order(rows[i - 1]['velocity'],
                                                      rows[i]['velocity'])
        for i in range(1, len(rows)):
            if 'interior_pressure' in rows[i - 1] and 'interior_pressure' in rows[i]:
                rows[i]['interior_pressure_order'] = observed_order(
                    rows[i - 1]['interior_pressure'], rows[i]['interior_pressure'])
    # Attribution: the finest-grid pressure order in both limits must agree.
    s, f = result['stokes'][-1], result['full_ns'][-1]
    if 'pressure_order' in s and 'pressure_order' in f:
        result['stokes_pressure_order'] = s['pressure_order']
        result['full_ns_pressure_order'] = f['pressure_order']
        result['order_agreement'] = abs(s['pressure_order'] - f['pressure_order'])
    return result
