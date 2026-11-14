"""Gate for the dry-bed ``x_f(t)`` *series* metric resolution floor.

The case acceptance (``CASE-FVM-DAMBREAK-STOKER``) grades the whole transient
``x_f(t)`` against Ritter's closed form.  That grading used to admit samples
whose fan window held only **three** cells.  On those samples the
shape-consistent estimator degenerates to a cell-centre locator and reads
``0.493 dx`` -- a half cell -- so the relative error is ``0.49 dx / x_f``, i.e. a
*resolution-invariant* number: measured ``6.613 %`` at both ``n = 200`` and
``n = 400`` (both carry 3 fan cells / 7.46 travelled cells).  Consequences:
the graded series error was non-monotone (4.70 % -> 6.61 % -> 6.61 % -> 1.40 %
-> 0.43 % for n = 100 ... 1600), the coarse grids reported a spurious
``front_series<5%`` failure, and the pool entry carried
``accepted_grid_min = 800`` on the strength of that artefact.

``SERIES_MIN_FAN_CELLS`` declares the fan-window resolution below which a
dry-bed sample is not graded (the samples stay visible under
``unresolved_reasons["fan_cells"]``).  This test file locks:

1. the identity is *exact* on the exact profile at any cell count -- so the
   floor is about the numerical tip being resolved, not about estimator bias;
2. the floor is declared in the module and applied by the sampler;
3. excluded samples keep their error and their reason (no silent dropping);
4. the committed artefacts carry the fixed grading and no dry grid reports a
   series failure;
5. the headline number the pool entry quotes at ``n = 1600`` did not move.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from tensorfvm.shallow_water import CASES, run_case
from tensorfvm.shallow_water.dam_break import (
    SERIES_MIN_CELLS,
    SERIES_MIN_FAN_CELLS,
    fan_front_estimate,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                        "artifacts", "shallow-water-dam-break")
POOL = os.path.join(REPO_ROOT, "docs", "case-fvm-dambreak-stoker",
                    "case_pool_entry.json")
G, H0 = 9.81, 1.0


def test_estimator_identity_is_exact_on_the_exact_profile_at_any_cell_count():
    """F1-style fixed point, including the *low* cell counts now excluded.

    If the 3-cell reading were estimator bias this would fail; it does not, so
    the defect has to come from the numerical tip, which is exactly why the
    sample must not be graded.
    """
    import numpy as np

    case = CASES["ritter-dry"]
    c0 = math.sqrt(G * H0)
    counts = []
    for n in (100, 200, 400, 800):
        dx = 10.0 / n  # the solver's 10 m channel
        x = (0.5 + np.arange(n)) * dx
        for t_star in (0.02, 0.05, 0.10, 0.25, 0.5, 1.0):
            t = t_star * math.sqrt(H0 / G)
            h, _ = case.exact(x, t)
            est = fan_front_estimate(x, h, t, G, H0)
            if est["cells"] == 0:
                continue
            exact = 2.0 * c0 * t
            assert est["front"] == pytest.approx(exact, rel=1e-12), (n, t_star)
            counts.append(est["cells"])
    # the identity holds even below the grading floor: the smallest window seen
    # is at or under the floor, so the floor really is exercised
    assert min(counts) <= SERIES_MIN_FAN_CELLS, counts


def test_floor_is_declared_and_non_vacuous():
    assert isinstance(SERIES_MIN_FAN_CELLS, int) and SERIES_MIN_FAN_CELLS >= 3
    assert SERIES_MIN_CELLS >= 1.0


def test_excluded_samples_are_reported_with_reason_and_error():
    series = run_case(CASES["ritter-dry"], n=400)["metrics"]["front_series"]
    assert series["min_fan_cells"] == SERIES_MIN_FAN_CELLS
    reasons = series["unresolved_reasons"]
    assert reasons.get("fan_cells", 0) >= 1, reasons
    # nothing is dropped: graded + unresolved == sampled
    assert series["samples_graded"] + series["unresolved_samples"] == series["samples"]
    # the excluded sample's error is still on the record
    assert series["unresolved_max_error"] is not None
    assert series["unresolved_max_error"] > series["max_error"]


def test_committed_artifact_has_no_dry_series_failure():
    conv = json.load(open(os.path.join(ARTIFACT, "convergence.json"),
                          encoding="utf-8"))
    for row in conv["ritter-dry"]["rows"]:
        assert row["front_series"] < 0.05, row["n"]
        assert "front_series<5%" not in row["verdict_failing"], row["n"]
    metrics = json.load(open(os.path.join(ARTIFACT, "metrics.json"),
                             encoding="utf-8"))
    dry = metrics["cases"]["ritter-dry"]["front_series_summary"]
    assert dry["pass"] is True
    assert dry["min_fan_cells"] == SERIES_MIN_FAN_CELLS


def test_pool_headline_number_is_unchanged_by_the_floor():
    """The floor must not buy the coarse grids at the price of the headline.

    The accepted grid is n = 1600; its maximum graded sample already carried
    >= SERIES_MIN_FAN_CELLS cells, so the quoted 0.425 % must be bit-stable.
    """
    metrics = json.load(open(os.path.join(ARTIFACT, "metrics.json"),
                             encoding="utf-8"))
    pool = json.load(open(POOL, encoding="utf-8"))
    headline = metrics["cases"]["ritter-dry"]["front_series_summary"]["max_error"]
    assert headline == pytest.approx(0.0042472950845003216, rel=1e-9)
    row = [r for r in pool["grid_study"]["rows"] if r["n"] == 1600][0]
    assert row["cases"]["ritter-dry"]["front_series_error_percent"] \
        == pytest.approx(100.0 * headline, rel=1e-9)
    assert pool["accepted_grid_min"] <= 1600
    for crit in pool["cases"]:
        for item in crit["criteria"]:
            if item.get("primary"):
                assert item["pass"] is True, (crit["key"], item["criterion"])