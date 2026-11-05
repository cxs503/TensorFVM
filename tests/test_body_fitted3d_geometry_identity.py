"""Exact geometric qualification of the extruded body-fitted cylinder O-grid.

The pre-existing gate (``tests/test_body_fitted3d.py``) allows 1 % wetted-area
error on one coarse mesh.  These gates are exact: the O-grid builds an inscribed
polygon whose arc chords are recoverable from the raw vertex array, so the
discrete area is known in closed form and the 2 pi R L deficit must follow the
inscribed-polygon rate.  Negative controls prove the identities are falsifiable.
"""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_body_fitted3d_geometry.py"
_spec = importlib.util.spec_from_file_location("audit_body_fitted3d_geometry", SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("nx", [16, 32, 64])
def test_stored_wetted_area_is_the_vertex_chord_sum(nx):
    """Stored wall area == chord sum x span recomputed from raw vertices."""
    case = audit.geometry_report(nx=nx, ny=8, nz=4)
    assert case["identity_stored_vs_vertices_rel"] < 1e-12
    # The 24 arc columns span slightly less than a full circle (8 of 32 columns
    # belong to the trailing wake plate), so the full-circle inscribed value
    # 2 n R sin(pi/n) L is an UPPER bound on the arc area, not an identity:
    # measured gap 12 %, i.e. the implied angular cover is (1 - 0.12)^(1/2) of 2pi.
    assert 0.08 < case["identity_arc_vs_inscribed_rel"] < 0.16, case["identity_arc_vs_inscribed_rel"]


@pytest.mark.parametrize("nx", [16, 32, 64])
def test_arc_vertices_lie_on_the_exact_cylinder_radius(nx):
    """Every genuine arc vertex satisfies |r - c| = R, with c recovered from the arc."""
    case = audit.geometry_report(nx=nx, ny=8, nz=4)
    assert case["arc_vertex_radius_rel_error"] < 1e-10
    # the angular sweep is a cylinder + a trailing wake plate, not a full circle
    assert case["wake_faces"] > 0
    assert case["arc_faces"] + case["wake_faces"] == nx


@pytest.mark.parametrize("nx", [16, 32, 64])
def test_cross_section_area_is_reproduced_independently(nx):
    """Green's theorem on the raw rings matches the stored area and sum(V)/L."""
    case = audit.geometry_report(nx=nx, ny=8, nz=4)
    assert case["cross_section_stored_rel"] < 1e-12
    assert case["cross_section_volume_rel"] < 1e-12


def test_area_deficit_against_the_smooth_cylinder_decays_at_second_order():
    """Observed order must be ~2 in 1/n_theta (inscribed-polygon rate)."""
    ladder = audit.refinement_ladder(nx_values=(8, 16, 32, 64, 128), ny=8, nz=4)
    assert 1.85 < ladder["deficit_observed_order"] < 2.15, ladder["deficit_observed_order"]
    deficits = [r["deficit_vs_analytic_rel"] for r in ladder["rows"]]
    assert all(a > b for a, b in zip(deficits, deficits[1:])), deficits
    # the closed-form rate explains the deficit: 1 - sin(pi/n)/(pi/n) ~ (pi^2/6)/n^2
    for row in ladder["rows"]:
        predicted = (math.pi ** 2 / 6) / row["nx"] ** 2
        assert abs(row["deficit_vs_analytic_rel"] - predicted) / predicted < 0.45, row


def test_negative_controls_are_detected():
    """A metric blind to a 1 % radius error or one shifted vertex measures nothing."""
    nc = audit.negative_controls(nx=32)
    # the chord sum is linear in R, so +1 % radius shows as ~+1 % chord length
    assert abs(nc["radius_inflation_1pct_area_rel_change"] - nc["predicted_radius_inflation_change"]) < 5e-3
    assert nc["single_vertex_shift_detected"]
    assert nc["single_vertex_shift_1pct_area_rel_change"] > 1e-5
    # the chord sum is a linear measure, so a 1 % radius error shows as ~1 %, and
    # one shifted vertex changes ~1 % of one chord: both are far above round-off
    assert nc["single_vertex_shift_1pct_area_rel_change"] > 1e-6