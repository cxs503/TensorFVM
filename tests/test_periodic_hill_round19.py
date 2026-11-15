"""Round 19: the round-18 ladder's two finest levels are *diverged*, not UNDEFINED.

``docs/case-periodic-hills/ROUND18.md`` states in section 5 that the missing
``120x60`` / ``144x72`` reattachment lengths were checked and are *not* a
numerical divergence (``diverged_at_step = null``).  These tests pin the
opposite, straight from the artefacts:

* both records carry ``diverged_at_step`` (281 / 159), ``samples = 0`` and an
  all-zero accumulated mean, while the stored instantaneous field of the same
  run has ``max|u|`` of order ``1e281`` / ``1e169``;
* therefore a level with a recorded divergence must never be reported as a
  physical "no zero crossing" result (``classify_level`` precedence);
* and the ladder verdict must list those levels as diverged and keep them out
  of the grid-law fit.

The round-19 reruns (``r19_<mesh>_dt0.01``) are asserted only when their
artefacts are present: a missing artefact is a *skip*, not a pass, so the test
can never manufacture a measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import round19_divergence_audit as r19  # noqa: E402

OUTDIR = REPO / "docs" / "case-periodic-hills"


def _audit() -> dict:
    return r19.analyse(figures=False)


# --------------------------------------------------------------------------- #
# 1. the errata: round-18's fine levels diverged (measured, from the artefacts)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag,nx,ny,step", [
    ("r18_120x60", 120, 60, 281),
    ("r18_144x72", 144, 72, 159),
])
def test_fine_levels_are_diverged(tag, nx, ny, step):
    lv = r19.classify_level(tag, nx, ny, 0.02, 18)
    assert lv["status"] == "DIVERGED", lv["reason"]
    assert lv["diverged_at_step"] == step
    assert lv["samples"] == 0
    assert lv["stored_field_max"] > r19.DIVERGED_FIELD
    assert lv["x_r_over_H"] is None
    assert "divergence recorded" in lv["reason"]


def test_diverged_level_has_all_zero_accumulated_mean():
    """The zero mean shear of the two fine levels is an artefact of ``samples=0``."""
    for tag in ("r18_120x60", "r18_144x72"):
        data = np.load(OUTDIR / f"round11_sst_{tag}.npz", allow_pickle=True)
        assert np.all(np.asarray(data["mean_tau_b"]) == 0.0)
        assert np.all(np.asarray(data["mean_u"]) == 0.0)
        # ... while the stored instantaneous field is not finite in magnitude
        assert np.max(np.abs(np.asarray(data["u"]))) > r19.DIVERGED_FIELD


def test_divergence_takes_precedence_over_no_crossing():
    """A diverged run may not be re-labelled as a physical 'no bubble' result."""
    diverged = r19.classify_level("r18_120x60", 120, 60, 0.02, 18)
    assert diverged["status"] != "NO_CROSSING"
    # a converged level is still scored normally
    ok = r19.classify_level("r18_96x48", 96, 48, 0.02, 18)
    assert ok["status"] == "MEASURED"
    assert ok["x_r_over_H"] == pytest.approx(7.6816, abs=5e-3)


# --------------------------------------------------------------------------- #
# 2. the verdict must expose the divergence and exclude it from the fit
# --------------------------------------------------------------------------- #
def test_verdict_lists_diverged_levels_and_does_not_fit_them():
    out = _audit()
    v = out["verdict"]
    assert v["n_diverged_levels"] == 2
    assert set(v["diverged_tags"]) == {"r18_120x60", "r18_144x72"}
    assert v["n_measured_levels"] == len(
        [lv for lv in out["levels"] if lv["status"] == "MEASURED"])
    assert "diverged" in v["honest_summary"]
    # the grid law may only use measured levels; with < 3 measured it must refuse
    fitted = v["grid_law"]["levels_coarse_to_fine"] or []
    assert all(entry["x_r_over_H"] is not None for entry in fitted)
    if v["n_measured_levels"] < 3:
        assert v["grid_law_admissible"] is False


def test_stability_map_flags_protocol_dt_from_120_up():
    stab = _audit()["stability"]
    assert 120 in stab["protocol_dt_unstable_at"]
    assert 144 in stab["protocol_dt_unstable_at"]
    assert 72 not in stab["protocol_dt_unstable_at"]
    assert 96 not in stab["protocol_dt_unstable_at"]
    outcomes = {e["outcome"] for e in stab["entries"]}
    assert {"MEASURED", "DIVERGED"} <= outcomes


# --------------------------------------------------------------------------- #
# 3. round-19 reruns (skip when the artefact is absent -- never a silent pass)
# --------------------------------------------------------------------------- #
def _level_or_skip(tag, nx, ny, dt):
    lv = r19.classify_level(tag, nx, ny, dt, 19)
    if lv["status"] == "NO_ARTEFACT":
        pytest.skip(f"{tag} artefact not produced yet")
    return lv


def test_fine_level_stable_step_is_measured():
    """120x60 at dt = 0.01 must be a converged, sampled run (not a divergence)."""
    lv = _level_or_skip("r19_120x60_dt0.01", 120, 60, 0.01)
    assert lv["status"] == "MEASURED", lv["reason"]
    assert lv["diverged_at_step"] is None
    assert lv["samples"] > 0
    assert lv["x_r_over_H"] is not None and np.isfinite(lv["x_r_over_H"])
    # the run must have kept the ERCOFTAC bulk velocity, i.e. be a real solution
    assert lv["stored_field_max"] < r19.DIVERGED_FIELD


def test_dt_control_at_96x48():
    """Halving dt at fixed mesh measures the time-step sensitivity of x_r/H."""
    ctrl = _audit()["dt_control"]
    if not ctrl.get("available"):
        pytest.skip("round-19 dt-control artefact not produced yet")
    assert ctrl["mesh"] == [96, 48]
    assert len(ctrl["x_r_over_H"]) == 2
    assert ctrl["relative_change"] == pytest.approx(
        abs(ctrl["x_r_over_H"]["0.02"] - ctrl["x_r_over_H"]["0.01"])
        / abs(ctrl["x_r_over_H"]["0.01"]))
    assert ctrl["time_step_converged_5pct"] == (ctrl["relative_change"] <= 0.05)