"""Tests for the FVM ↔ anchor-criterion bridge (CASE-FVM-CHANNEL-KMM180).

Three layers, cheapest first:

* **synthetic** -- the reduction and the grading are checked on manufactured
  fields so that a wiring defect cannot masquerade as a physical finding;
* **real 1-D** -- the anchor's own grading function must reproduce the
  published 1-D equilibrium reading, so the bridge is provably applying the
  *same* criterion;
* **real FVM (smoke)** -- a deliberately truncated FVM solve (2 outer
  iterations) must come back *unconverged* and be graded ``WITHHELD``; a
  non-converged solution may never earn a verdict.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorfvm.verification.kmm180_fvm_bridge import (
    MIN_WINDOW_CELLS,
    developed_profiles,
    fvm_channel_profile,
    grade_profile,
)
from tensorfvm.verification.kmm1987 import (
    RE_TAU_REF,
    load_kmm1987_reference,
    validate_kmm1987,
)

REPO_ROOT_TOLERANCE = 0.05


@pytest.fixture(scope='module')
def dns():
    return load_kmm1987_reference()


def fake_channel(n_levels=24, nu=1.0 / RE_TAU_REF, tau_w=1.0, extent=(0.0, 1.0)):
    """A manufactured two-dimensional field with an exactly known profile.

    ``u`` is chosen so that the volume average has a closed form; the wall
    traction integral is set to ``-tau_w * A``.  Cells are extruded in x and z,
    as in the native developed-channel mesh.
    """
    levels = np.expm1(3.0 * (np.arange(n_levels) + 0.5) / n_levels) / np.expm1(3.0)
    levels = extent[0] + (extent[1] - extent[0]) * levels
    axial, span = 3, 2
    cells = []
    for j, y in enumerate(levels):
        for a in range(axial):
            for s in range(span):
                cells.append((a, y, s))
    centers = np.array(cells, float)
    dy = np.gradient(levels)
    volumes = np.array([dy[j] * (0.1 / axial) * (0.1 / span)
                        for j in range(n_levels) for _ in range(axial * span)])
    u_level = 1.0 + 0.5 * levels
    velocity = np.zeros((len(cells), 3))
    velocity[:, 0] = np.concatenate([np.full(axial * span, val) for val in u_level])
    wall_area = 1.0
    mesh = SimpleNamespace(
        cell_centers=centers,
        cell_volumes=volumes,
        face_centers=np.array([[0.0, 0.0, 0.0]]),
        area_vectors=np.array([[0.0, -wall_area, 0.0]]),
    )
    # The wall pulls the fluid backwards: the integrated axial traction on the
    # wall patch is -tau_w * A, matching the native channel solvers' convention.
    result = dict(velocity=velocity,
                  total_traction_integrated=np.array([[-tau_w * wall_area, 0.0, 0.0]]))
    info = dict(wall=np.array([True]), nu=nu)
    exact = dict(levels=levels, u=u_level,
                 bulk=float(volumes @ velocity[:, 0] / volumes.sum()),
                 u_tau=float(np.sqrt(tau_w)))
    return mesh, result, info, exact


def test_profiles_reduce_the_extrusion_and_use_measured_wall_shear(dns):
    mesh, result, info, exact = fake_channel()
    profile = developed_profiles(mesh, result, info)

    np.testing.assert_allclose(profile['y'], exact['levels'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(profile['u'], exact['u'], rtol=0, atol=1e-12)
    # Extruded directions must be identical; a non-zero spread means 3-D content.
    np.testing.assert_allclose(profile['u_spread'], 0.0, atol=1e-14)
    assert profile['tau_w'] == pytest.approx(1.0)
    assert profile['u_tau'] == pytest.approx(1.0)
    assert profile['bulk_u'] == pytest.approx(exact['bulk'])
    assert profile['Cf'] == pytest.approx(2.0 * 1.0 / exact['bulk'] ** 2)
    np.testing.assert_allclose(profile['y_plus'], exact['levels'] * RE_TAU_REF)
    np.testing.assert_allclose(profile['u_plus'], exact['u'])


def test_non_developed_and_malformed_solutions_are_rejected():
    mesh, result, info, _ = fake_channel()
    with pytest.raises(ValueError, match='non-positive wall shear'):
        developed_profiles(mesh, dict(result, total_traction_integrated=
                                      -result['total_traction_integrated']), info)
    with pytest.raises(ValueError, match='non-finite boundary traction'):
        developed_profiles(mesh, dict(result, total_traction_integrated=
                                      np.array([[np.nan, 0.0, 0.0]])), info)
    with pytest.raises(ValueError, match='positive viscosity'):
        developed_profiles(mesh, result, info, nu=0.0)


def test_grading_is_the_anchor_function_unchanged(dns):
    """Feeding the DNS table back through the bridge must be a perfect match."""
    profile = dict(solver='synthetic', n=0, converged=True,
                   y_plus=dns['y_plus'], u_plus=dns['u_plus'],
                   u_tau=1.0, re_tau=dns['re_tau'], Cf=dns['Cf'],
                   bulk_u_plus=dns['U_bulk_plus'], u_spread=[0.0])
    grade = grade_profile(profile, dns=dns)
    assert grade['legacy_full']['rms_relative'] < 1e-12
    assert grade['log_layer']['rms_relative'] < 1e-12
    assert grade['legacy_full_verdict'] == 'PASS'
    assert grade['log_layer_verdict'] == 'PASS'

    # A uniform 8 % over-prediction must read as 8 % in both windows: the
    # metric is relative, so the verdict may not depend on window dilution here.
    scaled = dict(profile, u_plus=dns['u_plus'] * 1.08)
    grade = grade_profile(scaled, dns=dns)
    assert grade['legacy_full']['rms_relative'] == pytest.approx(0.08, rel=1e-9)
    assert grade['log_layer']['rms_relative'] == pytest.approx(0.08, rel=1e-9)
    assert grade['legacy_full_verdict'] == 'FAIL'

    # 4 % offset: PASS under the 5 % criterion, in both windows.
    small = dict(profile, u_plus=dns['u_plus'] * 1.04)
    grade = grade_profile(small, dns=dns)
    assert grade['legacy_full_verdict'] == 'PASS'
    assert grade['log_layer_verdict'] == 'PASS'


def test_unconverged_or_sparse_profiles_are_withheld(dns):
    sparse = dict(solver='synthetic', n=0, converged=True,
                  y_plus=np.array([1.0, 2.0, 3.0]), u_plus=np.array([1.0, 2.0, 3.0]),
                  u_tau=1.0, re_tau=dns['re_tau'], Cf=dns['Cf'],
                  bulk_u_plus=dns['U_bulk_plus'], u_spread=[0.0])
    grade = grade_profile(sparse, dns=dns)
    assert grade['legacy_full']['gradeable'] is False
    assert grade['legacy_full_verdict'] == 'WITHHELD'
    assert grade['log_layer_verdict'] == 'WITHHELD'
    assert grade['verdict'] == 'WITHHELD'

    unconverged = dict(solver='synthetic', n=0, converged=False,
                       y_plus=dns['y_plus'], u_plus=dns['u_plus'],
                       u_tau=1.0, re_tau=dns['re_tau'], Cf=dns['Cf'],
                       bulk_u_plus=dns['U_bulk_plus'], u_spread=[0.0])
    grade = grade_profile(unconverged, dns=dns)
    assert grade['legacy_full_verdict'] == 'WITHHELD'
    assert grade['log_layer_verdict'] == 'WITHHELD'

    with pytest.raises(ValueError, match='monotonically'):
        grade_profile(dict(sparse, y_plus=np.array([3.0, 2.0, 1.0])), dns=dns)


def test_ode_grade_reproduces_the_published_anchor_reading(dns):
    """The bridge must not shift the 1-D verdict it inherits (sst, n=128)."""
    from tensorfvm.rans_validation.channel import solve_channel

    result = solve_channel('sst', 128, re_tau=180.0)
    assert result['converged']
    reference = validate_kmm1987(np.asarray(result['y_plus']),
                                 np.asarray(result['u_plus']),
                                 result['Cf'], result['bulk_u_plus'], dns=dns)
    bridged = grade_profile(dict(solver='1d_equilibrium_sst', n=128, converged=True,
                                 y_plus=np.asarray(result['y_plus']),
                                 u_plus=np.asarray(result['u_plus']),
                                 u_tau=1.0, re_tau=180.0,
                                 Cf=result['Cf'], bulk_u_plus=result['bulk_u_plus'],
                                 u_spread=[0.0]), dns=dns)
    assert bridged['legacy_full']['rms_relative'] == pytest.approx(
        reference['log_region']['rms_relative'], rel=1e-12)
    assert bridged['log_layer']['rms_relative'] == pytest.approx(
        reference['log_region_narrow']['rms_relative'], rel=1e-12)
    assert bridged['legacy_full_verdict'] == reference['verdict']
    assert bridged['log_layer_verdict'] == reference['verdict_log_layer']
    assert bridged['legacy_full']['cells'] >= MIN_WINDOW_CELLS


def test_truncated_fvm_solve_is_withheld_not_graded(dns):
    """A 2-iteration FVM solve must not be graded -- it cannot converge."""
    profile = fvm_channel_profile(16, re_tau=RE_TAU_REF, axial=2, length=0.1,
                                  span=0.1, max_iterations=2, tolerance=1e-8)
    assert profile['converged'] is False
    assert profile['solver'] == 'native_fvm_sst2003'
    assert np.all(np.isfinite(profile['u_plus']))
    assert np.all(np.diff(profile['y_plus']) > 0)
    grade = grade_profile(profile, dns=dns)
    assert grade['legacy_full_verdict'] == 'WITHHELD'
    assert grade['log_layer_verdict'] == 'WITHHELD'
    assert grade['tolerance'] == pytest.approx(REPO_ROOT_TOLERANCE)