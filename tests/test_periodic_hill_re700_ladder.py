"""Round 32: Re = 700 ladder completion + first y/H = 0.5 profile grading.

Pins what the round-32 table actually says so a future change cannot silently
move the numbers:

* the 96x48 rung (round-20 artefact) re-scores to ``x_r/H = 8.078`` with the
  certified operator, ``out_of_window`` (reattachment beyond the flat bottom);
* the first-ever ``y/H = 0.5`` profile grading at Re = 700 against the primary
  DNS files: 2/8 stations inside the 10 % band, and the computed profile is
  *negative* (reverse flow) at every station from ``x/H = 4`` on, where the
  DNS is strongly positive -- the 2-D bubble does not reattach inside the
  domain;
* the 144x72 / 192x96 rungs are recorded as ABSENT in this commit (cluster
  jobs cj4666 / cj4667 in flight); the pre-registered attribution is therefore
  ``None`` (fewer than two measured levels) and must not be quoted as a
  verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_re700_reference import (
    X_R_OVER_H,
    profile_at,
)
from tensorfvm.verification.wall_shear_bubbles import separation_reattachment

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
TABLE = OUTDIR / "round32_re700_ladder_completion.json"
WINDOW = (1.9, 7.1)
FLAT_BOTTOM_END = 7.071  # x/H where the hill surface returns to y=0


def _row(tag):
    table = json.loads(TABLE.read_text())
    return next(r for r in table["rows"] if r["tag"] == tag)


def test_table_present_and_protocol_pinned():
    table = json.loads(TABLE.read_text())
    assert table["reynolds"] == 700.0
    assert table["reference"]["x_r_over_H"] == pytest.approx(5.24)
    assert table["protocol"]["dt"] == 0.02
    assert table["protocol"]["jobs"] == {"144x72": "cj4666",
                                         "192x96": "cj4667"}


def test_96x48_rung_rescores_to_pinned_value():
    """The certified operator on the stored mean shear must reproduce the
    table's 96x48 number (the payload's derived value is never trusted)."""
    with np.load(OUTDIR / "round20_laminar_Re700_96x48.npz") as npz:
        tau = np.asarray(npz["mean_tau_b"], dtype=float)
        x = np.asarray(npz["x_over_h"], dtype=float)
    scored = separation_reattachment(tau, x, window=WINDOW)
    row = _row("96x48")
    assert row["x_r_over_H"] == pytest.approx(scored["x_r_over_H"], rel=1e-12)
    assert row["x_r_over_H"] == pytest.approx(8.0784, abs=5e-4)
    assert row["status"] == "out_of_window"
    assert row["x_r_over_H"] > FLAT_BOTTOM_END
    assert row["error"] == pytest.approx(0.5417, abs=5e-4)
    assert row["below_10"] is False
    assert row["below_3"] is False


def test_96x48_profile_against_primary_dns():
    row = _row("96x48")
    prof = {p["x_over_H"]: p for p in row["profile_stations"]}
    assert set(prof) == {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
    # every reference value in the table must equal the raw-file value
    for x, p in prof.items():
        assert p["reference"] == pytest.approx(profile_at(x, 0.5), rel=1e-12)
    # the qualitative headline: reverse flow where the DNS is forward flow
    for x in (4.0, 5.0, 6.0, 7.0, 8.0):
        assert prof[x]["computed"] < 0.0 < profile_at(x, 0.5)
    assert row["profile_pass_count"] == 2
    assert {x for x, p in prof.items() if p["pass"]} == {1.0, 3.0}


def test_missing_rungs_are_absent_not_faked():
    for tag in ("144x72", "192x96"):
        row = _row(tag)
        assert row["present"] is False
        assert row["status"] == "ABSENT"
        assert row["x_r_over_H"] is None
        assert row["error"] is None


def test_attribution_is_none_until_two_rungs_measured():
    table = json.loads(TABLE.read_text())
    assert table["n_measured"] == 1
    assert table["attribution"] is None
    # and the pre-registered rule really does refuse a one-level ladder
    from tensorfvm.verification.periodic_hill_grid_ladder import (
        INCONCLUSIVE,
        attribute,
    )
    verdict = attribute([(9.0 / 96, 8.0784)], X_R_OVER_H)
    assert verdict["verdict"] == INCONCLUSIVE
