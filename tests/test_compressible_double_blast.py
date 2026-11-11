"""Gates for the Woodward-Colella two interacting blast waves.

The grids here are deliberately small so the file stays inside a normal CI
budget; the production ladder (100 ... 3200 cells to t = 0.038) is run by
``python -m tensorfvm.compressible.double_blast`` and recorded under
``docs/woodward-colella-double-blast/artifacts/``.  What is asserted here is
that every anchor the report relies on actually measures something:

* the initial datum reproduces the three plateaus and the exact initial energy,
* a closed tube conserves mass and total energy to round-off,
* the early-time blast fronts travel at the exact-Riemann shock speed,
* the y-extruded 2D slab stays exactly y-uniform (the 2D path adds nothing).
"""

import math

import pytest
import torch

from tensorfvm.compressible.cases import (
    WC_BLAST_P_LEFT,
    WC_BLAST_P_MIDDLE,
    WC_BLAST_P_RIGHT,
    WC_BLAST_X1,
    WC_BLAST_X2,
    DoubleBlastConfig,
    blast_early_anchors,
    double_blast,
)
from tensorfvm.compressible.double_blast import (
    _run,
    leg_early_time,
    leg_initial_condition,
    leg_two_dimensional,
)


def test_initial_datum_and_energy():
    """Three plateaus plus the exact integral of p/(gamma-1)."""
    solver, meta = double_blast(DoubleBlastConfig(nx=400, t_end=1e-9))
    p0 = meta["p0"]
    x = meta["x"]
    assert float(p0[x < WC_BLAST_X1].min()) == pytest.approx(WC_BLAST_P_LEFT)
    mid = (x > WC_BLAST_X1) & (x < WC_BLAST_X2)
    assert float(p0[mid].max()) == pytest.approx(WC_BLAST_P_MIDDLE)
    assert float(p0[x > WC_BLAST_X2].min()) == pytest.approx(WC_BLAST_P_RIGHT)
    leg = leg_initial_condition()
    assert leg["energy_rel_error"] < 1e-14
    assert leg["mass_rel_error"] < 1e-14
    assert all(leg["verdicts"].values())


def test_closed_tube_conserves_mass_and_energy():
    """Reflective walls: mass and total energy are conserved identically."""
    run = _run(200, 0.01)
    assert abs(run["drift"]["mass"]) < 1e-13
    assert abs(run["drift"]["energy"]) < 1e-13
    assert run["positivity_fixes"] == 0


def test_early_fronts_move_at_the_exact_riemann_speed():
    """Each initial jump is an isolated Riemann problem up to t = 2.67e-3."""
    anchors = blast_early_anchors()
    assert anchors["left"]["shock_speed"] > 0 > anchors["left"]["rarefaction_head"]
    assert anchors["right"]["shock_speed"] < 0 < anchors["right"]["rarefaction_head"]
    leg = leg_early_time((400,))
    row = leg["rows"][0]
    for tag in ("left", "right"):
        assert row[tag]["measured_shock_speed"] == pytest.approx(
            row[tag]["exact_shock_speed"], rel=0.10), tag
        # the star pressure is the plateau value of the exact solution
        assert row[tag]["pressure_rel_error"] < 0.10, tag


def test_extruded_slab_stays_y_uniform():
    """The 2D path must not perturb a y-independent solution."""
    leg = leg_two_dimensional(nx=100, ny=3, t_end=0.004)
    assert leg["max_y_std"] < 1e-12
    assert all(leg["verdicts"].values())