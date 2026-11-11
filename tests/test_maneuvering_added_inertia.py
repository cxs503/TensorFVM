"""Tests for the panel-method yaw added inertia of the manoeuvring hull (D7)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.maneuvering import added_inertia_panel as A
from tensorfvm.maneuvering import dynamics as D
from tensorfvm.maneuvering.selfpropulsion import solve_self_propulsion


@pytest.fixture(scope="module")
def hull():
    return A.build_hull()


@pytest.fixture(scope="module")
def panels(hull):
    return A.hull_panels(hull, n_stations=41, n_girth=9)


def test_geometry_gate(panels, hull):
    """The panel body must reproduce Cb L B T and Cwp L B before any inertia."""
    gate = A.geometry_gate_panels(panels, hull)
    assert gate["passed"], gate["checks"]
    assert abs(gate["checks"]["volume"]["relative_error"]) < 0.02
    assert abs(gate["checks"]["lid_area"]["relative_error"]) < 0.01


def test_lid_carries_no_yaw_flux(panels, hull):
    """Structural: the waterline lid is the double-body symmetry plane for yaw."""
    from tensorfvm.panel_added_inertia import rigid_rotation_field

    lid = A.deck_panels(panels)
    assert lid.n_panels > 0
    V = rigid_rotation_field(lid.centroids, 2,
                             origin=np.array([0.5 * hull.L, 0.0, 0.0]))
    assert np.max(np.abs(V[:, 2])) < 1e-12


def test_yaw_inertia_positive_and_converging(hull, panels):
    """J_z > 0 and the two meshes agree to better than 10 % (monotone refinement)."""
    coarse = A.hull_panels(hull, n_stations=21, n_girth=5)
    j_fine = A.yaw_added_inertia_panel(panels)
    j_coarse = A.yaw_added_inertia_panel(coarse)
    assert j_fine > 0.0 and j_coarse > 0.0
    assert j_fine == pytest.approx(j_coarse, rel=0.10)
    # the slender-body placeholder is an upper bound: the 3-D value must be lower
    assert j_fine < A.slender_yaw_added_inertia()["J_z"]


def test_sphere_yaw_is_zero():
    """Closed-form anchor: yaw of a sphere about a diameter is exactly zero.

    The generator is fed a synthetic half-sphere + lid.  For a yaw field the lid
    carries no flux, so the body is equivalent to the full sphere, whose added
    rotational inertia about any diameter vanishes identically (D1 anchor).
    """

    class HalfSphere:
        def __init__(self, a=1.0):
            self.a = float(a)
            self.L = 2.0 * a
            self.T = a

        def draft_at(self, x):
            return np.sqrt(np.maximum(self.a ** 2 - (np.asarray(x) - self.a) ** 2, 0.0))

        def half_breadth(self, x, z):
            x = np.atleast_1d(np.asarray(x, float))
            z = np.atleast_1d(np.asarray(z, float))
            r = np.sqrt(np.maximum(self.a ** 2 - (x - self.a) ** 2, 0.0))
            return np.sqrt(np.maximum(r[:, None] ** 2 - z[None, :] ** 2, 0.0))

    s = HalfSphere(1.0)
    p = A.hull_panels(s, n_stations=24, n_girth=12)
    m_a = 2.0 / 3.0 * math.pi  # rho = 1
    j = A.yaw_added_inertia_panel(p, rho=1.0, origin=np.array([1.0, 0.0, 0.0]))
    assert abs(j) / (m_a * s.a ** 2) < 1e-4


def test_j_z_override_changes_only_the_yaw_inertia():
    sp = solve_self_propulsion()
    base = D.ManoeuvreModel(self_propulsion=sp, mode="direct")
    override = D.ManoeuvreModel(self_propulsion=sp, mode="direct", j_z_override=5000.0)
    assert base.masses["J_z"] != 5000.0
    assert base.masses == D.added_masses(base.hull_model)  # default untouched
    assert override.masses["J_z"] == pytest.approx(5000.0)
    for key in ("m_x", "m_y"):
        assert override.masses[key] == pytest.approx(base.masses[key])
    assert override.masses["J_z_over_mass_L2"] == pytest.approx(
        5000.0 / (base.conditions.mass * base.conditions.length ** 2))


def test_negative_j_z_override_rejected():
    sp = solve_self_propulsion()
    model = D.ManoeuvreModel(self_propulsion=sp, mode="direct", j_z_override=-1.0)
    with pytest.raises(ValueError):
        _ = model.masses


def test_manoeuvre_sensitivity_is_bounded():
    """Swapping J_z for the panel value must not destabilise the prediction."""
    sp = solve_self_propulsion()
    j_panel = A.yaw_added_inertia_panel(
        A.hull_panels(A.build_hull(), n_stations=21, n_girth=5))
    base = D.simulate_turning_circle(D.ManoeuvreModel(self_propulsion=sp,
                                                      mode="direct"),
                                     35.0, dt=1.0, t_end=400.0)
    alt = D.simulate_turning_circle(
        D.ManoeuvreModel(self_propulsion=sp, mode="direct", j_z_override=j_panel),
        35.0, dt=1.0, t_end=400.0)
    m0 = D.turning_circle_metrics(base)
    m1 = D.turning_circle_metrics(alt)
    assert np.isfinite(m1["tactical_diameter_over_L"])
    rel = abs(m1["tactical_diameter_over_L"] / m0["tactical_diameter_over_L"] - 1.0)
    assert rel < 0.30  # measured 10.9 % at dt = 0.5 s; coarse dt/kernel keeps margin