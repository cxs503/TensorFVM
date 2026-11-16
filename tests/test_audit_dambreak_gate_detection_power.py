"""Drift gate for the dam-break gate detection-power audit.

The audit ``scripts/audit_dambreak_gate_detection_power.py`` answers a question
no other artifact in the anchor answers: **can the acceptance gate fail?**  A
"PASS with 100x margin" is only evidence if the graded criteria move when the
solution is wrong, so the audit perturbs the accepted-grid profile with declared
transforms and re-grades through the production gate path.

This test keeps the committed evidence honest:

1. the committed ``detection_power.json`` must still state the properties the
   report quotes -- a gate that has been observed to fail, a bit-exact
   reproduction of the pool-pinned artifact, unit-responsiveness thresholds for
   the wave-position criteria, a finite effective monotonicity threshold, and
   the declared boundary-cell blind spot;
2. the perturbation harness itself is re-exercised live on a small grid, so the
   audit cannot silently rot into "the script still runs but perturbs nothing":
   an 8 % displacement must fail the wave-position gate, a 10x-floor spike on a
   flat cell must be counted, a 0.3x-floor spike must not.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "audit_dambreak_gate_detection_power.py")
ARTIFACT = os.path.join(
    REPO_ROOT, "src", "tensorfvm", "verification", "artifacts",
    "shallow-water-dam-break-gate-power", "detection_power.json")


def _load_script():
    spec = importlib.util.spec_from_file_location("_audit_gate_power", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_script()


@pytest.fixture(scope="module")
def committed():
    with open(ARTIFACT, encoding="utf-8") as handle:
        return json.load(handle)


def test_committed_evidence_states_gate_detection_power(committed):
    verdict = committed["verdict"]
    assert committed["meta"]["tolerance"] == 0.05
    # a gate that has never been observed to fail proves nothing
    assert verdict["gate_can_fail"] is True
    # the case pool is pinned to the artifact this audit re-integrated
    assert verdict["artifact_reproduced"] is True
    assert committed["reproduction"]["max_rel_delta"] <= committed["meta"]["reproduction_rel_tol"]
    # every wave-position criterion must fire at <= 5 % injected displacement
    assert all(verdict["translation_detected_at_5pct"].values()), verdict[
        "translation_detected_at_5pct"]
    # every monotonicity probe position must have a finite effective threshold
    thr = verdict["monotone_effective_threshold_floor_multiples"]
    assert thr and all(value is not None for value in thr.values()), thr
    assert all(value >= 1.0 for value in thr.values()), thr
    assert verdict["all_sound"] is True


def test_committed_evidence_records_the_boundary_blind_spot(committed):
    """The detector mask excludes the two boundary cells; that must stay stated."""
    for case_key, block in committed["battery"].items():
        probe = block["boundary_probe"]
        assert probe["floor_multiple"] >= 1.0
        assert probe["detected"] is False, case_key
        assert all(p["extrema"] == 0 for p in probe["probes"]), case_key


def test_committed_thresholds_are_ordered_flat_below_sloped(committed):
    """A trend-reversal criterion must be *harder* to trip on a sloped cell.

    The one-cell spike reverses the local trend only when its height exceeds the
    local cell-to-cell slope, so the effective threshold on the fan kink cannot
    be below the flat-region threshold.  If that ordering ever inverts, the
    probe positions (or the detector) moved and the report's scope statement is
    stale.
    """
    thr = committed["verdict"]["monotone_effective_threshold_floor_multiples"]
    for case_key in ("stoker-wet", "ritter-dry"):
        flat = thr[f"{case_key}@flat"]
        kink = thr[f"{case_key}@kink"]
        assert flat <= kink, (case_key, flat, kink)


def test_harness_still_perturbs_and_grades(audit):
    """Live re-exercise on a small grid (fast): the harness must bite."""
    from tensorfvm.shallow_water.dam_break import STOKER_WET, run_case

    n = 400
    base = run_case(STOKER_WET, n=n)
    h = base["profile"]["h_numeric"]
    x = base["profile"]["x"]
    ref = audit._case_reference(STOKER_WET, base)

    # 8 % displacement of the bore must fail the wave-position criterion
    delta = 0.08 * abs(ref["wave_x"])
    graded = audit._regrade(STOKER_WET, audit._translate(
        __import__("numpy").asarray(x), __import__("numpy").asarray(h), delta), n, base)
    assert graded["graded"][ref["crit"]]["pass"] is False
    assert graded["graded"][ref["crit"]]["value"] > 0.05

    # monotonicity: 10x the declared floor is counted, 0.3x is not
    import numpy as np
    floor = audit.EXTREMUM_FRACTION * STOKER_WET.h0
    k = int(np.argmin(np.abs(np.asarray(x) - ref["flat_x"])))
    for mult, expect in ((10.0, True), (0.3, False)):
        h_p = np.asarray(h).copy()
        h_p[k] += mult * floor
        out = audit._regrade(STOKER_WET, h_p, n, base)
        assert (out["metrics"]["local_extrema"] > 0) is expect, (mult, out["metrics"]["local_extrema"])