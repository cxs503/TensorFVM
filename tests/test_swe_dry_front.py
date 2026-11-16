"""Wet/dry front attribution judges: the invariants they claim are asserted here.

Every test that claims an *exact* property uses ``== 0.0`` (not a tolerance):
the discrete statement is algebraic, and a tolerance would hide a genuine
violation behind the grid resolution.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.shallow_water.swe2d import ShallowWater2D
from tensorfvm.verification import swe_dry_front as df
from tensorfvm.verification import thacker_basin as tb


def _bowl_solver(n=16, **kwargs):
    basin = tb.build_basin()
    L = 1.5 * basin.a
    X, Y = tb.grid(basin, n, 1.5)
    solver = ShallowWater2D(n, n, -L, L, -L, L, bed=basin.bed(X, Y), **kwargs)
    solver.set_state(*basin.initial_state(X, Y))
    return solver, basin, X, Y


# --------------------------------------------------------------- island judge
def test_enclosed_island_transmits_nothing():
    """A wet cell walled in by higher dry beds: mass *and* momentum frozen."""
    out = df.island_invariants()
    enclosed = out["enclosed"]
    assert enclosed["isolated"] is True
    assert enclosed["trapped"] is True
    assert enclosed["h_star_self"] == [0.0, 0.0, 0.0, 0.0]
    assert enclosed["dh"] == 0.0
    assert enclosed["dhu"] == 0.0
    assert enclosed["dhv"] == 0.0
    assert out["enclosed_invariants_hold"] is True


def test_open_island_does_transmit():
    """Control: with one sill removed the cell is not isolated and does drain."""
    out = df.island_invariants()["one_open"]
    assert out["isolated"] is False
    assert out["mass_frozen"] is False
    assert out["dh"] != 0.0


def test_island_geometry_is_not_a_lake_at_rest():
    """Negative control for the judge: drop the sills and the cell must move."""
    n = 5
    bed = np.zeros((n, n))
    h = np.zeros((n, n))
    h[n // 2, n // 2] = 0.01
    solver = ShallowWater2D(n, n, -2.5, 2.5, -2.5, 2.5, bed=bed)
    solver.set_state(h)
    budget = df.face_budget(solver, n // 2, n // 2)
    assert budget["isolated"] is False
    assert budget["dh"] != 0.0
    assert budget["consistency"]["dh"] == pytest.approx(0.0, abs=1e-12)


# -------------------------------------------------------------- face budget
def test_face_budget_reproduces_the_solver_rhs():
    solver, _, _, _ = _bowl_solver()
    solver.advance(0.02)
    for i, j in ((8, 8), (4, 12), (12, 3)):
        budget = df.face_budget(solver, i, j)
        for key in ("dh", "dhu", "dhv"):
            assert budget["consistency"][key] == pytest.approx(0.0, abs=1e-12)


def test_face_budget_needs_an_interior_cell():
    solver, _, _, _ = _bowl_solver()
    with pytest.raises(ValueError):
        df.face_budget(solver, 0, 5)


# NOTE: the two pressure-correction branches ("reconstructed" vs "cell") are
# compared *numerically* in docs/swe-dry-front-attribution/REPORT.md (on the
# 2-period run they differ by 4.4x in step count and 40x in peak velocity).
# No unit test pins that comparison: on the smooth 24x24 bowl snapshot the two
# branches are indistinguishable there, so such a test would assert nothing.


# ------------------------------------------------------- desingularisation
def test_desingularisation_is_the_identity_for_deep_water():
    eps = 1e-3
    solver, _, _, _ = _bowl_solver(desingularize=eps)
    h, u, v = solver.primitive()
    wet = h > 100 * eps
    naive = np.where(wet, solver.hu / np.where(wet, h, 1.0), 0.0)
    assert np.max(np.abs(u[wet] - naive[wet])) < 1e-12


def test_desingularisation_bounds_near_dry_velocity():
    eps = 1e-2
    n = 3
    solver = ShallowWater2D(n, n, -1.5, 1.5, -1.5, 1.5, desingularize=eps)
    solver.set_state(np.full((n, n), 1e-6), u=np.full((n, n), 0.0))
    solver.hu[1, 1] = 1e-4            # implies u = 100 m/s in the naive recovery
    _, u, _ = solver.primitive()
    assert abs(u[1, 1]) < 1.0
    naive = ShallowWater2D(n, n, -1.5, 1.5, -1.5, 1.5)
    naive.set_state(np.full((n, n), 1e-6))
    naive.hu[1, 1] = 1e-4
    _, u_naive, _ = naive.primitive()
    assert abs(u_naive[1, 1]) == pytest.approx(100.0)


def test_desingularisation_rejects_negative_eps():
    with pytest.raises(ValueError):
        ShallowWater2D(3, 3, 0.0, 1.0, 0.0, 1.0, desingularize=-1.0)


# ----------------------------------------------------------- option hygiene
def test_defaults_keep_the_merged_discretisation():
    solver = ShallowWater2D(4, 4, 0.0, 1.0, 0.0, 1.0)
    assert solver.desingularize == 0.0
    assert solver.correction_depth == "reconstructed"
    assert solver.gate_dry_face_correction is False
    with pytest.raises(ValueError):
        ShallowWater2D(4, 4, 0.0, 1.0, 0.0, 1.0, correction_depth="nonsense")


def _lake_at_rest_spurious(n=24, **kwargs):
    basin = tb.build_basin()
    L = 1.5 * basin.a
    X, Y = tb.grid(basin, n, 1.5)
    solver = ShallowWater2D(n, n, -L, L, -L, L, bed=basin.bed(X, Y), **kwargs)
    solver.lake_at_rest_state(surface=0.0)
    solver.advance(0.05)
    return solver.max_spurious_velocity()


def test_lake_at_rest_survives_the_shipped_option_settings():
    """Well-balancedness is a graded gate: it must survive the new switches."""
    for kwargs in ({}, {"desingularize": 1e-2}, {"correction_depth": "cell"},
                   {"desingularize": 1e-2, "correction_depth": "cell"}):
        assert _lake_at_rest_spurious(**kwargs) < 1e-11, kwargs


def test_gating_the_correction_breaks_well_balancedness():
    """Recorded refutation: ``gate_dry_face_correction`` is NOT well balanced.

    Kept as a test (not a comment) so that anyone who tries this cure again sees
    the measured damage: spurious velocity of order 0.2 m/s in a lake at rest.
    """
    assert _lake_at_rest_spurious(gate_dry_face_correction=True) > 1e-3


def test_mass_is_conserved_with_the_experimental_options():
    for kwargs in ({"desingularize": 1e-2}, {"correction_depth": "cell"},
                   {"gate_dry_face_correction": True}):
        solver, basin, _, _ = _bowl_solver(n=12, **kwargs)
        v0 = solver.total_volume()
        solver.advance(0.1)
        assert abs(solver.total_volume() - v0) / v0 < 1e-12, kwargs