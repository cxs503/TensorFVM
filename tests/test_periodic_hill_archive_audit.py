"""Round-26 tests: artefact integrity gate and bubble-*object* audit.

Two things are pinned here.

1. **A zero-field dump is not a measurement.**  The archived refined rungs carry
   all-zero velocity *and* all-zero wall shear, so the certified crossing
   operator reports ``status = "no_recirculation"`` for them -- the same status a
   genuinely attached flow produces.  The integrity gate must classify them as
   ``zero_field`` before any ladder rule reads them.
2. **A wall-shear crossing pair need not bound a cell.**  The synthetic fixtures
   are built so the two routes must disagree in a controlled way: one connected
   reversed-flow cell whose wall signature is interrupted (the real laminar
   ``Re = 2800`` situation) versus two independent cells.  The audit must
   separate them, and it must not confuse a 2-cell field with a 1-cell one.

The stored-artefact tests bind the audit to numbers already published in this
tree (``1.254`` / ``8.264`` for the laminar anchor, ``7.682`` for SST round 18),
so a regression in the operator shows up as a test failure rather than as a
quietly different report.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_archive_audit import (
    ARTIFACT_NON_FINITE,
    ARTIFACT_USABLE,
    ARTIFACT_ZERO_FIELD,
    OBJECT_INCOMPLETE,
    OBJECT_SUPPORTED,
    artifact_health,
    bubble_object_audit,
    definition_band,
    definition_routes,
    ladder_summary,
    recirculation_cells,
    wall_spans,
)
from tensorfvm.verification.wall_shear_bubbles import separation_reattachment

CASE_DIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"

LAMINAR = CASE_DIR / "round20_laminar_Re2800_96x48.npz"
LAMINAR_DEGENERATE = CASE_DIR / "round20_laminar_Re2800_144x72.npz"
SST_R18 = CASE_DIR / "round11_sst_r18_96x48.npz"
SST_DEGENERATE = CASE_DIR / "round11_sst_r18_144x72.npz"


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------
def _field(nx=36, ny=8, period=9.0):
    dx = period / nx
    x = dx * (np.arange(nx) + 0.5)
    return dx, x, np.full((ny, nx), 0.5)


def _mask_from_spans(x, spans):
    m = np.zeros(x.size, dtype=bool)
    for lo, hi in spans:
        m |= (x >= lo) & (x <= hi)
    return m


def two_cells():
    """Two independent wall-attached recirculation cells."""
    _, x, u = _field()
    u[0:3, _mask_from_spans(x, [(1.0, 2.5)])] = -0.1
    u[0:3, _mask_from_spans(x, [(5.0, 7.0)])] = -0.1
    return u, x


def lifted_cell():
    """One cell that touches the wall in two spans (a lift-off in between)."""
    _, x, u = _field()
    lower = _mask_from_spans(x, [(1.0, 2.5), (4.0, 7.0)])
    upper = _mask_from_spans(x, [(1.0, 7.0)])
    u[0, lower] = -0.1
    u[1:4, upper] = -0.1
    u[0, _mask_from_spans(x, [(2.5, 4.0)])] = 0.02  # weak forward wall layer
    return u, x


def seam_cell():
    """One cell wrapping the periodic seam."""
    _, x, u = _field()
    u[0:3, _mask_from_spans(x, [(7.5, 9.0), (0.0, 1.0)])] = -0.1
    return u, x


# --------------------------------------------------------------------------
# artefact integrity
# --------------------------------------------------------------------------
def test_zero_field_rung_is_missing_not_measured():
    with np.load(LAMINAR_DEGENERATE) as d:
        health = artifact_health(d)
        u = np.asarray(d["mean_u"], float)
        tau = np.asarray(d["mean_tau_b"], float)
        x = np.asarray(d["x_over_h"], float)
    assert health["status"] == ARTIFACT_ZERO_FIELD
    assert health["usable"] is False
    assert health["max_abs_u"] == 0.0
    assert np.all(u == 0.0) and np.all(tau == 0.0)
    # the trap this gate exists for: the certified operator sees "no reversed
    # flow", i.e. a status that reads as a measurement of an attached flow
    ws = separation_reattachment(tau, x)
    assert ws["status"] == "no_recirculation"
    assert np.isnan(ws["x_r_over_H"])


def test_sst_zero_field_rung_is_missing():
    with np.load(SST_DEGENERATE) as d:
        assert artifact_health(d)["status"] == ARTIFACT_ZERO_FIELD


def test_non_finite_field_is_rejected():
    u, x = two_cells()
    u[2, 2] = np.nan
    health = artifact_health({"mean_u": u, "x_over_h": x, "mean_tau_b": u[0]})
    assert health["status"] == ARTIFACT_NON_FINITE
    assert health["usable"] is False


def test_missing_keys_is_rejected():
    health = artifact_health({"mean_u": np.zeros((3, 3))})
    assert health["usable"] is False
    assert "missing keys" in health["reason"]


def test_usable_artefact_health_matches_stored_laminar_anchor():
    with np.load(LAMINAR) as d:
        health = artifact_health(d)
        u = np.asarray(d["mean_u"], float)
    assert health["status"] == ARTIFACT_USABLE and health["usable"]
    assert (health["nx"], health["ny"]) == (96, 48)
    assert health["n_cells"] == 1
    assert health["wall_attached_columns"] == int(np.count_nonzero(u[0] < 0.0))


# --------------------------------------------------------------------------
# field topology route
# --------------------------------------------------------------------------
def test_two_cells_are_two_cells():
    u, x = two_cells()
    cells = recirculation_cells(u, x)
    assert len(cells) == 2
    assert all(c["area_cells"] > 0 for c in cells)
    assert sum(c["area_cells"] for c in cells) == int(np.count_nonzero(u < 0))
    assert all(len(c["wall_spans"]) == 1 for c in cells)


def test_lifted_cell_is_one_cell_with_two_wall_spans():
    u, x = lifted_cell()
    cells = recirculation_cells(u, x)
    assert len(cells) == 1
    spans = cells[0]["wall_spans"]
    assert len(spans) == 2
    assert spans[0]["separation"] == pytest.approx(1.0, abs=0.3)
    assert spans[1]["separation"] == pytest.approx(4.0, abs=0.3)


def test_wall_spans_wrap_the_periodic_seam():
    u, x = seam_cell()
    spans = wall_spans(u[0], x)
    assert len(spans) == 1
    assert spans[0]["spans_seam"] is True
    assert spans[0]["length"] == pytest.approx(2.5, abs=0.5)


def test_seam_cell_is_one_component():
    u, x = seam_cell()
    cells = recirculation_cells(u, x)
    assert len(cells) == 1
    assert cells[0]["wraps_seam"] is True


# --------------------------------------------------------------------------
# bubble-object audit
# --------------------------------------------------------------------------
def test_lifted_cell_makes_the_first_crossing_an_incomplete_object():
    u, x = lifted_cell()
    tau = u[0]  # the wall-adjacent row plays the role of the shear profile
    audit = bubble_object_audit(tau, u, x, primary="first")
    assert audit["n_field_cells"] == 1
    assert audit["verdict"] == OBJECT_INCOMPLETE
    assert audit["continued_downstream"] is True
    assert audit["wall_shear"]["x_r_over_H"] == pytest.approx(2.5, abs=0.3)
    assert audit["next_wall_touchdown"] == pytest.approx(4.0, abs=0.3)
    assert audit["lift_off_shear_ratio"] < 0.1


def test_two_cells_make_each_crossing_a_supported_object():
    u, x = two_cells()
    tau = u[0]
    audit = bubble_object_audit(tau, u, x, primary="first")
    assert audit["verdict"] == OBJECT_SUPPORTED
    assert audit["continued_downstream"] is False


def test_stored_laminar_anchor_bubble_is_one_cell_two_crossings():
    with np.load(LAMINAR) as d:
        u = np.asarray(d["mean_u"], float)
        x = np.asarray(d["x_over_h"], float)
        tau = np.asarray(d["mean_tau_b"], float)
    audit = bubble_object_audit(tau, u, x, primary="first")
    assert audit["n_field_cells"] == 1
    assert audit["bubble_cells"] == [0]
    assert audit["verdict"] == OBJECT_INCOMPLETE
    assert audit["wall_shear"]["x_r_over_H"] == pytest.approx(1.254, abs=2e-3)
    assert audit["next_wall_touchdown"] is not None
    # the "reattachment" at x/H = 1.25 sits on ~2 % of the peak wall shear
    assert audit["lift_off_shear_ratio"] < 0.05
    assert audit["field_reattachment_last_span"] == pytest.approx(8.258, abs=5e-3)


def test_stored_sst_bubble_is_a_closed_cell():
    with np.load(SST_R18) as d:
        u = np.asarray(d["mean_u"], float)
        x = np.asarray(d["x_over_h"], float)
        tau = np.asarray(d["mean_tau_b"], float)
    audit = bubble_object_audit(tau, u, x, primary="first")
    assert audit["n_field_cells"] == 1
    assert audit["verdict"] == OBJECT_SUPPORTED
    assert audit["wall_shear"]["x_r_over_H"] == pytest.approx(7.682, abs=2e-3)


# --------------------------------------------------------------------------
# definition band
# --------------------------------------------------------------------------
def test_definition_band_is_sensitive_on_stored_laminar_anchor():
    with np.load(LAMINAR) as d:
        u = np.asarray(d["mean_u"], float)
        x = np.asarray(d["x_over_h"], float)
        tau = np.asarray(d["mean_tau_b"], float)
    routes = definition_routes(tau, u, x)
    band = definition_band(routes)
    assert band["flag"] == "definition_sensitive"
    assert band["n"] == 4
    assert band["spread_rel"] > 1.0          # > 100 % of the reference
    assert routes["meta"]["object_verdict"] == OBJECT_INCOMPLETE
    # the cell count is metadata, not a length: it must not enter the band
    assert 1 not in band["values"]


def test_definition_band_is_insensitive_on_stored_sst_anchor():
    with np.load(SST_R18) as d:
        u = np.asarray(d["mean_u"], float)
        x = np.asarray(d["x_over_h"], float)
        tau = np.asarray(d["mean_tau_b"], float)
    band = definition_band(definition_routes(tau, u, x))
    assert band["flag"] == "definition_insensitive"
    assert band["spread_rel"] < 0.01


def test_definition_band_not_measured_when_every_route_is_nan():
    band = definition_band({"routes": {"wall_shear_first": float("nan"),
                                       "field_cell_last_wall_span": None}})
    assert band["flag"] == "not_measured"


# --------------------------------------------------------------------------
# ladder rules
# --------------------------------------------------------------------------
def test_ladder_refuses_a_verdict_on_two_usable_rungs():
    lad = ladder_summary([
        {"tag": "a", "nx": 72, "x_r_over_H": 7.600, "usable": True},
        {"tag": "b", "nx": 96, "x_r_over_H": 7.682, "usable": True},
        {"tag": "c", "nx": 120, "x_r_over_H": None, "usable": False},
        {"tag": "d", "nx": 144, "x_r_over_H": None, "usable": False},
    ])
    assert lad["verdict"] == "insufficient_rungs"
    assert lad["n_usable"] == 2 and lad["n_excluded"] == 2
    assert lad["excluded"] == ["c", "d"]
    assert lad["x_r_over_H"] is None
    assert lad["steps"][0]["delta_rel"] == pytest.approx(0.082 / 7.600, rel=1e-9)


def test_ladder_measures_when_three_rungs_are_usable():
    lad = ladder_summary([
        {"tag": "a", "nx": 32, "x_r_over_H": 7.90, "usable": True},
        {"tag": "b", "nx": 64, "x_r_over_H": 7.70, "usable": True},
        {"tag": "c", "nx": 128, "x_r_over_H": 7.62, "usable": True},
    ])
    assert lad["verdict"] == "measured"
    assert lad["x_r_over_H"] == pytest.approx(7.62)
    assert lad["error_rel"] == pytest.approx((7.62 - 5.41) / 5.41)
    assert lad["within_tolerance"] is False


def test_stored_archive_has_the_two_ladders_unanswered():
    """Pins the round-26 finding: both grid ladders are rung-starved."""
    payload = json.loads(
        (CASE_DIR / "round26_archive_audit.json").read_text())
    assert payload["n_artefacts"] == 14
    assert payload["n_degenerate"] == 6
    assert payload["n_usable"] == 8
    for name in ("sst_r18", "laminar_re2800"):
        assert payload["ladders"][name]["verdict"] == "insufficient_rungs"
    degenerate = {d["stem"] for d in payload["degenerate"]}
    assert "round11_sst_r18_144x72" in degenerate
    assert "round20_laminar_Re2800_192x96" in degenerate