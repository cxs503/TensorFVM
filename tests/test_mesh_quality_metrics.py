"""Strict gates for the 2-D mesh-quality metrics and their 70-degree acceptance gate.

These tests need no mesh generator, so they run everywhere (the previous only
coverage lived inside ``gmsh``-gated tests).  Closed forms: a uniformly sheared
grid has ``nonorthogonality == arctan(gamma)`` and vanishing internal skewness.
"""
import math
import numpy as np
import pytest
from tensorfvm.verification.mesh_quality_audit import (
    GATE_DEGREES, NOISE_DEG, audit, curved_channel, gate_flip_angle,
    independent_metrics, inverted_cell_control, inverted_face_control,
    rotate, shear_closed_form)
from tensorfvm.mesh_quality import mesh_quality


@pytest.fixture(scope='module')
def evidence():
    return audit()


def test_orthogonal_reference_is_exactly_orthogonal(evidence):
    reference = evidence['orthogonal_reference']
    assert reference['nonorthogonality_max_deg'] < NOISE_DEG
    assert reference['internal_skewness_max'] < 1e-12
    # Boundary faces measure half a cell: height/(2*ny) is the tighter of the two.
    assert reference['minimum_face_projected_distance_m'] == pytest.approx(.02 / (2 * 4), rel=1e-12)
    assert reference['positive_cell_areas'] and reference['positive_projected_distances']
    assert reference['gate_passed']


@pytest.mark.parametrize('angle', [5., 20., 60., 69.])
def test_sheared_grid_matches_arctan_closed_form(angle):
    row = shear_closed_form([angle])[0]
    assert row['measured_nonorthogonality_deg'] == pytest.approx(angle, abs=NOISE_DEG)
    assert row['measured_skewness'] < 1e-10
    assert row['gate_passed']


def test_gate_flips_exactly_at_the_documented_angle(evidence):
    assert evidence['gate_flip']['absolute_error_deg'] < 1e-6
    assert evidence['gate_flip']['documented_gate_deg'] == GATE_DEGREES
    above = shear_closed_form([GATE_DEGREES + 1.])[0]
    assert not above['gate_passed']


def test_shipped_metrics_agree_with_independent_recomputation(evidence):
    curved = evidence['curved_channel']
    assert curved['max_absolute_difference'] < 1e-12
    assert curved['shipped']['quality_gate_passed']
    # The real curved channel is genuinely non-orthogonal and skewed, so the
    # audit is not passing trivially on an orthogonal grid.
    assert curved['shipped']['nonorthogonality_max_deg'] > 1.
    assert curved['shipped']['internal_skewness_max'] > 1e-3


def test_metrics_are_invariant_under_rigid_rotation(evidence):
    rows = evidence['rotation_invariance']
    assert len(rows) == 4
    for row in rows:
        assert row['nonorthogonality_max_deg'] == pytest.approx(
            rows[0]['nonorthogonality_max_deg'], abs=1e-9)
        assert row['internal_skewness_max'] == pytest.approx(
            rows[0]['internal_skewness_max'], abs=1e-12)
        assert row['minimum_face_projected_distance_m'] == pytest.approx(
            rows[0]['minimum_face_projected_distance_m'], rel=1e-12)


def test_single_flipped_face_orientation_is_rejected(evidence):
    control = evidence['inverted_face_control']
    assert not control['positive_projected_distances']
    assert not control['gate_passed']
    assert control['nonorthogonality_max_deg'] > 90.


def test_negative_cell_area_is_rejected(evidence):
    control = evidence['inverted_cell_control']
    assert control['inverted_cell_area_m2'] < 0
    assert not control['positive_cell_areas']
    assert not control['gate_passed']


def test_rotation_helper_preserves_topology_and_volume():
    mesh = curved_channel(6, 4)
    rotated = rotate(mesh, 37.)
    assert rotated.volumes.shape == mesh.volumes.shape
    np.testing.assert_allclose(rotated.volumes.numpy(), mesh.volumes.numpy(), rtol=0, atol=0)
    np.testing.assert_array_equal(rotated.owner.numpy(), mesh.owner.numpy())
    np.testing.assert_array_equal(rotated.neighbor.numpy(), mesh.neighbor.numpy())
    # The connector norm is rotation invariant, so the projected distances match.
    assert mesh_quality(rotated)['minimum_face_projected_distance_m'] == pytest.approx(
        mesh_quality(mesh)['minimum_face_projected_distance_m'], rel=1e-12)


def test_independent_metric_recomputation_matches_shipped_module():
    mesh = curved_channel(6, 4)
    shipped, independent = mesh_quality(mesh), independent_metrics(mesh)
    for key, value in independent.items():
        if isinstance(value, bool):
            assert shipped[key] is value
        else:
            assert shipped[key] == pytest.approx(value, rel=1e-12, abs=1e-15)


def test_whole_audit_verdict_passes(evidence):
    assert evidence['verdict'], evidence