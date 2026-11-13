"""Gates for :mod:`tensorfvm.verification.limiter_extremum_order`.

Two layers:

* **algebra** -- the fixed-count window, the exact crest/slope positions, and the
  order estimator on synthetic fields of known local order (so a later solver
  result cannot be an artefact of the window);
* **solver facts** -- the MC-limited scheme reads its design order 2 in the
  volume-weighted global ``L1`` norm while the same run reads ``< 1.7`` in a
  frozen 5-cell window at the crest, and the unlimited control reads 2 there.
"""
import math

import numpy as np
import pytest

from tensorfvm.verification import cellcentre_loglog as cc
from tensorfvm.verification import limiter_extremum_order as lx


# --------------------------------------------------------------- algebra layer
def test_window_has_a_grid_independent_count_and_wraps_periodically():
    assert len(lx.window(40, 0, 2)) == 5
    assert len(lx.window(320, 0, 2)) == 5
    assert lx.window(40, 0, 2) == [38, 39, 0, 1, 2]
    assert lx.window(40, 39, 1) == [38, 39, 0]
    with pytest.raises(ValueError):
        lx.window(40, 0, -1)


def test_crest_and_maxslope_positions_are_exact():
    t = 0.2
    assert lx.crest_x(t) == pytest.approx(0.25 + 0.5 * t)
    assert lx.maxslope_x(t) == pytest.approx(0.5 * t)
    # 0.35 is equidistant from the centres 0.3375 / 0.3625; argmin is deterministic
    assert lx.cell_index(40, lx.crest_x(t)) == 13
    assert lx.cell_index(320, lx.crest_x(t)) == 111
    # the unshifted crest of this case falls exactly on a cell face for every
    # grid of the ladder -- the degeneracy C8 removes with a phase shift
    for nx in lx.GRIDS:
        assert (lx.crest_x(t) * nx) % 1.0 == pytest.approx(0.0, abs=1e-12)
    # the shifted crest is inside a cell on every grid: never within 0.1 cell of a face
    for nx in lx.GRIDS:
        f = (lx.crest_x(t, phase=lx.MIDCELL_PHASE) * nx) % 1.0
        assert min(f, 1.0 - f) >= 0.1, (nx, f)


def test_local_norms_are_unweighted_over_the_fixed_cell_set():
    field = np.array([0.0, 3.0, -4.0, 0.0, 2.0])
    ref = np.zeros(5)
    norms = lx.local_norms(field, ref, [1, 2, 3])
    assert norms["cells"] == 3
    assert norms["linf"] == pytest.approx(4.0)
    assert norms["l1"] == pytest.approx((3.0 + 4.0 + 0.0) / 3.0)
    assert norms["l2"] == pytest.approx(math.sqrt((9.0 + 16.0) / 3.0))
    # volume weighting over *all* cells would be a different statistic (2.25/5)
    all_cells = lx.local_norms(field, ref, list(range(5)))
    assert all_cells["l1"] == pytest.approx(9.0 / 5.0)
    with pytest.raises(ValueError):
        lx.local_norms(field, ref, [])
    with pytest.raises(ValueError):
        lx.local_norms(field[:4], ref, [0])


def _synthetic_ladder(order, grids=(40, 80, 160, 320), centre_offset=0.0):
    """Fields whose error on a fixed 5-cell window is ``c h^order``."""
    rows = []
    for nx in grids:
        h = 1.0 / nx
        rho = np.full(nx, 1.0)
        i = lx.cell_index(nx, lx.crest_x(0.2) + centre_offset)
        idx = lx.window(nx, i)
        rho[idx] = 1.0 + 0.5 * h ** order
        rows.append({"nx": nx, "h": h, "t": 0.2, "crest_linf": float(0.5 * h ** order),
                     "global_l1": float(0.5 * h ** order * len(idx) / nx),
                     "crest_l1": float(0.5 * h ** order), "maxslope_linf": float(0.5 * h ** order)})
    return rows


@pytest.mark.parametrize("order", [1.0, 2.0])
def test_estimator_reads_the_local_order_it_is_given(order):
    study = lx.study(_synthetic_ladder(order))
    read = lx._read(study, "crest_linf", lx.TRIPLETS[1])
    assert read["p_loglog"] == pytest.approx(order, abs=1e-9)


def test_gate_flags_a_degradation_instead_of_hiding_it():
    studies = {"mc": lx.study(_synthetic_ladder(1.0)),
               "none": lx.study(_synthetic_ladder(2.0))}
    ratios = [{"nx": r["nx"], "mc": r["crest_linf"], "none": r["crest_linf"] / 4,
               "ratio": 4.0} for r in _synthetic_ladder(1.0)]
    gate = {rec["criterion"]: rec for rec in lx.criteria_gate(studies, ratios)}
    assert gate["C2_crest_degradation"]["passed"] is True
    assert gate["C3_unlimited_control"]["passed"] is True
    # C8 was not run: it must be reported as not-passed, never as a silent pass
    assert gate["C8_crest_placement"]["passed"] is False


# ---------------------------------------------------------- solver-fact layer
@pytest.fixture(scope="module")
def ladders():
    grids = (80, 160, 320)
    return {name: lx.ladder((name,), grids)[name] for name in ("mc", "none")}


def test_design_order_is_confirmed_in_the_global_volume_weighted_norm(ladders):
    study = lx.study(ladders["mc"], triplets=((80, 160, 320),))
    read = lx._read(study, "global_l1", (80, 160, 320))
    assert read["p_loglog"] == pytest.approx(2.0, rel=0.15), read
    assert read["loglog_pass_15"] is True


def test_the_crest_window_reads_a_degraded_order_for_the_limited_scheme(ladders):
    study = lx.study(ladders["mc"], triplets=((80, 160, 320),))
    read = lx._read(study, "crest_linf", (80, 160, 320))
    assert read["p_loglog"] <= lx.DEGRADED_MAX, read
    # ... and the loss is real, not a rounding artefact: >15 % below the design order
    assert read["loglog_rel_dev"] > 0.15
    # the module does not claim exactly 1 -- the sampled value is reported as is
    assert read["p_loglog"] > 1.0


def test_the_unlimited_control_recovers_the_design_order_at_the_crest(ladders):
    study = lx.study(ladders["none"], triplets=((80, 160, 320),))
    read = lx._read(study, "crest_linf", (80, 160, 320))
    assert read["p_loglog"] == pytest.approx(2.0, rel=0.15), read
    # the two schemes agree away from the crest: the loss is local
    triplets = ((80, 160, 320),)
    mc = lx._read(lx.study(ladders["mc"], triplets=triplets), "maxslope_linf", (80, 160, 320))
    none = lx._read(lx.study(ladders["none"], triplets=triplets), "maxslope_linf", (80, 160, 320))
    assert mc["p_loglog"] == pytest.approx(2.0, rel=0.15), mc
    assert none["p_loglog"] == pytest.approx(2.0, rel=0.15), none


def test_crest_error_magnitude_diverges_from_the_control(ladders):
    ratio = lx.crest_magnitude_ratio(
        {name: lx.study(rows, triplets=((80, 160, 320),))
         for name, rows in ladders.items()})
    finest = ratio[-1]
    assert finest["ratio"] >= 10.0, finest
    # growing with refinement ~ h^-1: the O(h) local error against the O(h^2) one
    assert finest["ratio"] / ratio[-2]["ratio"] >= 1.2


def test_run_grid_reports_the_solver_time_and_the_fields_it_used():
    row = lx.run_grid(80, "mc")
    assert row["t"] == pytest.approx(0.2)
    assert row["steps"] > 0
    assert len(row["error"]) == 80
    assert row["crest_cell"] == lx.cell_index(80, lx.crest_x(row["t"]))
    assert row["crest_linf"] == pytest.approx(
        float(np.abs(np.array(row["error"])[lx.window(80, row["crest_cell"])]).max()))