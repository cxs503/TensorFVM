"""Tests for the double-body (waterplane mirror) added-mass verification (debt D2).

The fast subset of :mod:`tensorfvm.verification.mirror_added_mass`: the
mirror-pairing identity (machine precision), the analytic half-body anchor, the
frozen criteria of the reported study at a coarse mesh, and the documented
absence of a normal-direction anchor.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.panel_added_mass import lamb_spheroid_factors, sphere_added_mass
from tensorfvm.verification.mirror_added_mass import (
    half_body_added_mass,
    hemi_spheroid_panels,
    lamb_reference,
    pairing,
    rim_area,
    submerged_sphere,
    two_body_demihull,
    twin_hull_demihull,
    waterplane_residual,
)


def test_rim_is_the_equatorial_ellipse():
    """The open rim of the half body is exactly the ellipse of area ``pi a b``."""
    for a, b in ((1.0, 1.0), (2.0, 1.0)):
        panels = hemi_spheroid_panels(a, b, 12, 8)
        assert rim_area(panels) == pytest.approx(math.pi * a * b, rel=0.05)
        assert np.all(panels.centroids[:, 2] < 1e-12)


def test_hemisphere_half_body_is_half_of_the_sphere():
    """Mirror reassembles the sphere: ``2 M_half(single_hull) = M_full(body)``."""
    result = pairing(1.0, 1.0, 24, 8)
    assert result["gap_x"] < 1e-9
    assert result["gap_y"] < 1e-9
    # and the absolute value is the analytic half sphere (coarse mesh: 3-4 %
    # of first-order panel discretisation, 0.07 % at 48x16 in the study)
    assert result["half"]["M_xx"] == pytest.approx(0.5 * sphere_added_mass(1.0, 1.0), rel=0.05)


@pytest.mark.parametrize("axis_ratio", [2.0, 4.0])
def test_hemi_spheroid_matches_half_lamb(axis_ratio):
    """Wall-parallel translation of a hemi-spheroid vs ``1/2`` Lamb closed form."""
    got = half_body_added_mass(axis_ratio, 1.0, 24, 16)
    ref = lamb_reference(axis_ratio, 1.0)
    assert got["M_xx"] == pytest.approx(0.5 * ref["M_axial"], rel=0.03)
    assert got["M_yy"] == pytest.approx(0.5 * ref["M_transverse"], rel=0.03)
    # structural zero from the y -> -y symmetry
    assert abs(got["M_xy"]) / math.sqrt(got["M_xx"] * got["M_yy"]) < 1e-2


def test_normal_translation_has_no_half_body_anchor():
    """Heave is not half the full body: the naive anchor is falsified (> 20 %)."""
    got = half_body_added_mass(1.0, 1.0, 18, 12)
    naive = 0.5 * lamb_reference(1.0, 1.0)["M_transverse"]
    assert abs(got["M_zz"] - naive) / naive > 0.20


def test_waterplane_neumann_residual_is_round_off():
    """The double-body solve gives ``dphi/dz = 0`` on the waterplane."""
    residual = waterplane_residual(submerged_sphere(1.0, 2.0), order=3, radii=np.linspace(0.2, 3.0, 6))
    assert residual["max_abs_dphi_dz"] < 1e-10


def test_twin_hull_split_solve_matches_the_two_body_solve():
    """Partner images reproduce the independent two-body solve."""
    split = twin_hull_demihull(1.0, 1.0, 6.0, 18, 12)
    merged = two_body_demihull(1.0, 1.0, 6.0, 18, 12)
    assert split == pytest.approx(merged, rel=1e-3)


def test_twin_hull_partner_decays_with_separation():
    """Partner coupling vanishes as the demihulls separate."""
    single = half_body_added_mass(1.0, 1.0, 18, 12)["M_xx"]
    near = twin_hull_demihull(1.0, 1.0, 6.0, 18, 12)
    far = twin_hull_demihull(1.0, 1.0, 40.0, 18, 12)
    assert abs(near - single) > abs(far - single)
    assert abs(far - single) / single < 0.01


def test_lamb_factors_sphere_limit():
    """Sanity: the Lamb spheroid factors reduce to the sphere value at a = b."""
    k_axial, k_trans = lamb_spheroid_factors(1.0, 1.0)
    assert k_axial == pytest.approx(0.5, rel=1e-12)
    assert k_trans == pytest.approx(0.5, rel=1e-12)