"""Round 13 part C: the cj3108 artefact (fixed momentum operator, 96x48 SST)
is scored and its claims are bound to the artefact itself.

The original round-13 commit (PR #712 / mission 20987) left the post-fix
measurement open: job cj3108 was queued at commit time, so ROUND13.md carried
a WITHHELD row.  This module closes that loop:

* the fetched artefact ``round11_sst_r13-96x48.json`` must be internally
  consistent (gate values recomputed from the raw series, not read back from
  the gate block);
* the r11 -> r12 -> r13 ladder must be monotone (each fix moves the bubble
  downstream) and still short of the reference;
* the numbers published in ROUND13.md must match the artefact (doc-drift
  guard, the failure mode that bit this case before).

No solver call is made here: the artefact is a frozen measurement record and
the invariants below are arithmetic on its own fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
R13 = DOCS / "round11_sst_r13-96x48.json"
R12 = DOCS / "round11_sst_r12-96x48.json"
R11 = DOCS / "round11_sst_96x48.json"
DOC = DOCS / "ROUND13.md"

from tensorfvm.verification.periodic_hill_reference import X_R_OVER_H  # noqa: E402


def _load(name: Path) -> dict:
    assert name.exists(), f"missing artefact {name}"
    return json.loads(name.read_text())


def test_r13_artifact_present_and_well_formed():
    d = _load(R13)
    cfg = d["config"]
    assert (cfg["nx"], cfg["ny"]) == (96, 48)
    assert cfg["steps"] == 20000 and cfg["burn_in_steps"] == 8000
    assert cfg["time_step_s"] == 0.02 and cfg["end_time"] == 400.0
    assert cfg["reynolds"] == 2800.0
    assert d["closure"]["name"] == "sst"
    assert d["samples"] == 601
    assert len(d["series"]["t"]) == 601
    assert d["config"]["diverged_at_step"] is None
    # the npz twin must exist and carry fields
    npz = np.load(DOCS / "round11_sst_r13-96x48.npz")
    assert npz.files, "npz is empty"


def test_r13_gate_is_recomputable_from_the_raw_series():
    d = _load(R13)
    g = d["gate"]
    xr = np.asarray(d["series"]["x_r_over_H"], dtype=float)
    nu = np.asarray(d["series"]["max_nu_t_ratio"], dtype=float)
    # gate reattachment = time mean of the sampled series
    assert abs(g["x_r_over_H"] - xr.mean()) < 1e-9
    # gate peak eddy viscosity = max over the series
    assert abs(g["max_nu_t_ratio"] - nu.max()) < 1e-9
    # profile pass count agrees with the per-station table
    pc = d["profile_comparison"]
    assert pc["n_pass"] == g["profile_stations_pass"] == 0
    assert pc["n_stations"] == g["profile_stations"] == 7
    # the run is stationary: block means agree to < 1e-5 relative
    st = d["stationarity"]["x_r_over_H"]
    assert st["ok"] and st["spread_rel"] < 1e-5
    assert g["stationary"] is True


def test_ladder_monotone_and_still_short_of_reference():
    x11 = _load(R11)["gate"]["x_r_over_H"]
    x12 = _load(R12)["gate"]["x_r_over_H"]
    x13 = _load(R13)["gate"]["x_r_over_H"]
    # each fix moved the bubble downstream, none reached the reference
    assert x11 < x12 < x13 < X_R_OVER_H
    assert x12 / x11 > 1.4      # transport fix: +54 %
    assert x13 / x12 > 1.15     # operator fix: +19.8 %
    # and the eddy-viscosity level dropped with the operator fix
    assert _load(R13)["gate"]["max_nu_t_ratio"] < _load(R12)["gate"]["max_nu_t_ratio"]


def test_r13_gate_fails_with_the_measured_error():
    d = _load(R13)
    g = d["gate"]
    assert g["pass"] is False
    err = abs(g["x_r_over_H"] - X_R_OVER_H) / X_R_OVER_H
    assert 0.25 < err < 0.35    # the measured 30.4 %
    assert g["profile_stations_pass"] == 0
    assert g["stationary"] is True


def test_round13_doc_matches_artifact():
    """doc-drift guard: the numbers published in ROUND13.md part C must be
    the artefact's numbers (this case has been bitten by doc drift before)."""
    d = _load(R13)
    g = d["gate"]
    text = DOC.read_text()
    assert f"{g['x_r_over_H']:.3f}" in text                    # 3.767
    err = abs(g["x_r_over_H"] - X_R_OVER_H) / X_R_OVER_H
    assert f"{err * 100:.1f} %" in text                        # 30.4 %
    assert f"{g['max_nu_t_ratio']:.1f}" in text                # 202.5
    assert "cj3108" in text
    assert "WITHHELD (job cj3108 queued)" not in text          # row resolved


def test_round13_comparison_profile_table_is_populated():
    """The comparison JSON must carry the per-station profile table: the
    exact-string column matcher that shipped in rounds 12/13 matched no
    column on the 96x48 mesh (no cell centre sits on an integer x/H) and
    silently wrote an empty table.  The gate's own per-station block is the
    source of truth now."""
    d = json.loads((DOCS / "round13_comparison.json").read_text())
    rows = d["profile"]
    assert len(rows) == 7
    assert all(r["round13"] is not None and r["reference"] is not None for r in rows)
    assert all(r["pass_round13"] is False for r in rows)
    # the table agrees with the artefact's own gate verdict
    art = _load(R13)
    assert sum(r["pass_round13"] for r in rows) == art["gate"]["profile_stations_pass"]
    assert [r["x_over_H"] for r in rows] == [s["x_over_H"]
                                             for s in art["profile_comparison"]["stations"]]
