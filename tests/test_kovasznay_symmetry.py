"""Symmetry closure of the extruded Kovasznay case (ledger D10 mechanism, m16864).

The delivered order FAIL (1.684) of ``CASE-FVM-KOVASZNAY-ORDER`` is caused by the
extruded case closing its two ``z``-planes with a *traction-free* condition for
the tangential components instead of the symmetry condition ``du_t/dz = 0``.
These tests pin the mechanism on the coarse grids where it is cheap to reproduce.
"""
import numpy as np
import pytest

from tensorfvm.verification.kovasznay_symmetry import (
    analytic_defect, budgets, build, plane_shear, spanwise_drive_terms, uniqueness)
from tensorfvm.verification.native_hybrid_ns import kovasznay_solver

N = 6


def _solved(closure):
    mesh, solver, exact, joints = build(N, closure=closure)
    result = solver.solve(algorithm='coupled', outer_relaxation=.7, tolerance=1e-8,
                          max_outer_iterations=200)
    return mesh, solver, exact, joints, result


@pytest.fixture(scope='module')
def free_case():
    return _solved('free')


@pytest.fixture(scope='module')
def symmetry_case():
    return _solved('zero_gradient')


def test_symmetry_closure_pins_spanwise_velocity(symmetry_case):
    mesh, solver, exact, joints, result = symmetry_case
    value = budgets(np.asarray(result['velocity']), exact, np.asarray(mesh.cell_volumes))
    assert result['numerically_converged']
    assert value['max_abs_spanwise'] < 1e-12
    assert value['spanwise_l2'] < 1e-12
    assert value['quadrature_closure'] < 1e-9


def test_free_closure_leaves_spurious_spanwise_velocity(free_case):
    mesh, solver, exact, joints, result = free_case
    value = budgets(np.asarray(result['velocity']), exact, np.asarray(mesh.cell_volumes))
    # negative control: the pollution is real and far above round-off
    assert value['max_abs_spanwise'] > 1e-3


def test_spanwise_drive_is_the_viscous_cross_term(free_case):
    mesh, solver, exact, joints, result = free_case
    drive = spanwise_drive_terms(solver, exact)
    blocks = drive['blocks']
    assert drive['residual_max'] > 1e-6
    # advection, its source, pressure and every same-component block are exactly zero
    assert max(blocks[name]['advection'] for name in blocks) == 0.
    assert drive['convection_source'] == 0.
    assert drive['pressure'] == 0.
    assert max(blocks['cell_w']['viscous'], blocks['face_w']['viscous'],
               blocks['cell_u']['viscous'], blocks['cell_v']['viscous']) == 0.
    # ... so the whole residual is the cross-component (du_i/dz + dw/dx_i) family
    assert abs(drive['viscous_cross'] - drive['residual_max']) <= 1e-12 * drive['residual_max']


def test_spanwise_rows_accept_the_analytic_state(free_case):
    mesh, solver, exact, joints, result = free_case
    # the discrete spanwise momentum equation is consistent: it does not reject the
    # exact solution, so the pollution cannot be blamed on the equation itself
    assert analytic_defect(solver, exact, joints) < 1e-12


def test_symmetry_closure_has_no_drive_at_all(symmetry_case):
    mesh, solver, exact, joints, result = symmetry_case
    drive = spanwise_drive_terms(solver, exact)
    assert drive['residual_max'] < 1e-15
    assert drive['viscous_cross'] < 1e-15


def test_free_planes_settle_off_the_cell_value(symmetry_case, free_case):
    _, solver_free, exact_free, _, _ = free_case
    _, solver_sym, exact_sym, _, _ = symmetry_case
    shear_free = plane_shear(solver_free, exact_free)
    shear_sym = plane_shear(solver_sym, exact_sym)
    assert shear_free['max_abs_plane_shear'] > 1e-2      # the spurious du_t/dz
    assert shear_free['dimensionless'] > 1e-2
    assert shear_sym['max_abs_plane_shear'] == 0.        # tied to the cell value


def test_symmetry_closure_does_not_change_the_inplane_accuracy(free_case, symmetry_case):
    mesh, solver, exact, joints, free_result = free_case
    volumes = np.asarray(mesh.cell_volumes)
    free_value = budgets(np.asarray(free_result['velocity']), exact, volumes)
    symmetry_value = budgets(np.asarray(symmetry_case[4]['velocity']), exact, volumes)
    relative = abs(symmetry_value['inplane_l2'] - free_value['inplane_l2']) / free_value['inplane_l2']
    assert relative < 1e-2


def test_converged_spanwise_velocity_is_unique():
    # three initial guesses (two of them randomly perturbed) must give the same w,
    # which rules out a null-space / undetermined-mode explanation
    result = uniqueness(5, closure='free', perturbations=(0., .1, -.1))
    assert abs(result['relative']) < 1e-6


def test_case_driver_exposes_the_symmetry_closure():
    for closure in ('free', 'zero_gradient'):
        mesh, solver, exact = kovasznay_solver(5, 40., symmetry_extrusion=True,
                                               symmetry_closure=closure,
                                               coupled_linear_method='reused_lu')
        result = solver.solve(algorithm='coupled', outer_relaxation=.7, tolerance=1e-8,
                              max_outer_iterations=200)
        value = budgets(np.asarray(result['velocity']), exact, np.asarray(mesh.cell_volumes))
        assert result['numerically_converged']
        if closure == 'zero_gradient':
            assert value['max_abs_spanwise'] < 1e-12
        else:
            assert value['max_abs_spanwise'] > 1e-3


def test_invalid_closure_is_rejected():
    with pytest.raises(ValueError):
        kovasznay_solver(4, 40., symmetry_extrusion=True, symmetry_closure='slip')
    with pytest.raises(ValueError):
        kovasznay_solver(4, 40., symmetry_closure='zero_gradient')
