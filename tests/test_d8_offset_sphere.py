"""D8 regression tests: parallel-axis anchor of the offset sphere (debt ledger D8).

m14901 flagged ``added_mass_matrix_6dof`` as inconsistent with the exact
parallel-axis anchors of the offset sphere (``docs/maneuvering-added-inertia/
REPORT.md``).  Re-check (this ticket, m16122) shows the current solver passes
both offset configurations against the exact values; the phantom came from the
anchor helper's silent ``distance = 1.0`` default.  These tests pin:

* configuration A (centre ``(0,0,d)``): ``I_zz = 0`` (axis through the centre),
  ``I_xx = I_yy = m_a d^2``;
* configuration B (centre ``(d,0,0)``): ``I_xx = 0``, ``I_yy = I_zz = m_a d^2``;
* the off-diagonal rotational entries stay at round-off level;
* the anchor helper now *requires* ``distance`` (the footgun is gone);
* ``yaw_added_inertia_panel``'s auto-origin semantics on an offset body
  (bounding-box midship -> the sphere's own centre -> zero), so the trap is
  documented and cannot silently change.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.maneuvering.added_inertia_panel import yaw_added_inertia_panel
from tensorfvm.panel_added_mass import (
    added_mass_matrix_6dof,
    sphere_added_mass,
    sphere_offset_axis_added_inertia,
    sphere_panels_offset,
)

A, D, RHO = 1.0, 1.5, 1.0


def _block(centre, n_theta=32, n_phi=18):
    panels = sphere_panels_offset(A, centre, n_theta, n_phi)
    return added_mass_matrix_6dof(panels, rho=RHO, directions=(3, 4, 5))


def test_config_a_offset_z():
    """Centre (0,0,d): z axis through the centre -> 0; x/y -> m_a d^2."""
    blk = _block((0.0, 0.0, D))
    exact = sphere_offset_axis_added_inertia(A, D, RHO)
    assert blk[2, 2] / (sphere_added_mass(A, RHO) * A ** 2) < 1.0e-3
    assert blk[0, 0] == pytest.approx(exact, rel=3.0e-2)
    assert blk[1, 1] == pytest.approx(exact, rel=3.0e-2)


def test_config_b_offset_x():
    """Centre (d,0,0): x axis through the centre -> 0; y/z -> m_a d^2."""
    blk = _block((D, 0.0, 0.0))
    exact = sphere_offset_axis_added_inertia(A, D, RHO)
    assert blk[0, 0] / (sphere_added_mass(A, RHO) * A ** 2) < 1.0e-3
    assert blk[1, 1] == pytest.approx(exact, rel=3.0e-2)
    assert blk[2, 2] == pytest.approx(exact, rel=3.0e-2)


def test_off_diagonal_roundoff():
    for centre in ((0.0, 0.0, D), (D, 0.0, 0.0)):
        blk = _block(centre)
        off = np.max(np.abs(blk - np.diag(np.diag(blk))))
        assert off / (sphere_added_mass(A, RHO) * D ** 2) < 1.0e-3


def test_anchor_distance_is_mandatory():
    """The silent distance=1.0 default (source of the m14901 phantom) is gone."""
    with pytest.raises(TypeError):
        sphere_offset_axis_added_inertia(A)
    with pytest.raises(TypeError):
        sphere_offset_axis_added_inertia(A, rho=RHO)
    # explicit calls keep working (all pre-existing call sites are positional)
    assert sphere_offset_axis_added_inertia(A, 0.0, RHO) == 0.0
    assert sphere_offset_axis_added_inertia(A, D, RHO) == pytest.approx(
        sphere_added_mass(A, RHO) * D ** 2, rel=1.0e-12)


def test_yaw_panel_auto_origin_on_offset_sphere():
    """Auto-origin = bounding-box midship: for an offset sphere that is the
    sphere's own centre, so the yaw inertia about it is zero (documented
    semantics, not a solver value for the coordinate-origin axis)."""
    for centre in ((0.0, 0.0, D), (D, 0.0, 0.0)):
        panels = sphere_panels_offset(A, centre, 24, 14)
        j = yaw_added_inertia_panel(panels, rho=RHO)
        assert abs(j) / (sphere_added_mass(A, RHO) * A ** 2) < 1.0e-3
    # explicit origin at the coordinate origin recovers the m_a d^2 anchor
    panels = sphere_panels_offset(A, (D, 0.0, 0.0), 32, 18)
    j = yaw_added_inertia_panel(panels, rho=RHO,
                                  origin=np.array([0.0, 0.0, 0.0]))
    assert j == pytest.approx(sphere_offset_axis_added_inertia(A, D, RHO),
                              rel=3.0e-2)
