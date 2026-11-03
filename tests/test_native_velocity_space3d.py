import importlib.util
from pathlib import Path

import numpy as np
import pytest

script = Path(__file__).parents[1] / 'scripts/audit_native_velocity_space3d.py'
spec = importlib.util.spec_from_file_location('native_velocity_space_audit', script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


CASES = {(label, mesh.cells[0].__len__()): (mesh, mask, faces)
         for label, mesh, mask, faces in audit.configurations()[1]}


def build(label):
    mesh, mask, faces = next(value for name, value in CASES.items() if name[0] == label)
    return mesh, mask, faces


@pytest.mark.parametrize('label', ['tetra_normal_slanted', 'tetra_normal_plus_cartesian',
                                   'tetra_cartesian_one_component', 'channel_normal_axis_aligned'])
def test_columns_are_orthonormal_and_elimination_is_exact(label):
    mesh, mask, faces = build(label)
    space, record = audit.algebra_probe(mesh, mask, faces)
    assert record['rotated'] is (faces is not None)
    np.testing.assert_allclose(record['orthonormality_error'], 0., atol=1e-13)
    np.testing.assert_allclose(record['constraint_range_error'], 0., atol=1e-12)
    # exact elimination: reduced dimension equals the null space dimension of the
    # hard constraints, so no penalised pseudo-degree-of-freedom is introduced.
    assert record['nullity_gap'] == 0
    assert record['reduced_dofs'] < record['full_dofs']
    np.testing.assert_allclose(record['roundtrip_error'], 0., atol=1e-13)


def test_projection_removes_normal_violation_and_is_idempotent():
    mesh, mask, faces = build('tetra_normal_slanted')
    space, record = audit.algebra_probe(mesh, mask, faces)
    assert record['constraint_error_raw'] > 1e-2      # the raw vector really violates
    assert record['constraint_error_projected'] <= 1e-13
    assert record['idempotence_error'] <= 1e-13
    # the rotated reduced basis preserves tangential face components exactly
    assert record['tangential_normal_component'] <= 1e-13


def test_eliminated_solution_equals_hard_constrained_kkt_solution():
    mesh, mask, faces = build('tetra_normal_two_faces')
    _, record = audit.algebra_probe(mesh, mask, faces)
    assert record['kkt_solution_error'] <= 1e-11
    assert record['kkt_constraint_error'] <= 1e-13


def test_penalty_reference_is_first_order_so_no_penalty_is_used():
    mesh, mask, faces = build('tetra_normal_slanted')
    _, record = audit.algebra_probe(mesh, mask, faces)
    errors = [entry['solution_error'] for entry in record['penalty']]
    assert errors == sorted(errors, reverse=True)        # monotone towards elimination
    assert record['penalty_slope'] > .9                  # asymptotic first order in eps
    assert errors[-1] < 1e-4                             # far below the O(1) penalty error


def test_free_slip_elimination_leaves_no_zero_energy_mode():
    operator = audit.operator_probe()
    free_slip, no_slip = operator['free_slip_box'], operator['no_slip_box']
    assert free_slip['zero_modes'] == 0
    assert free_slip['min_eigenvalue'] > 0
    assert free_slip['symmetry_error'] <= 1e-12
    assert free_slip['reduced_dofs'] == 27 and no_slip['reduced_dofs'] == 3


def test_over_specified_input_is_rejected():
    rejections = audit.rejection_probe()
    assert all(entry['rejected'] for entry in rejections.values())