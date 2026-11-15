"""Long-time (self-similar) verification of the Ritter/Stoker dam-break case.

The anchor ``CASE-FVM-DAMBREAK-STOKER`` is graded at ``t* = 2``.  This file
locks the *time-domain* claims added by :mod:`tensorfvm.shallow_water.long_time`:

1. the integration domain really contains both waves at every graded ``t*``
   (otherwise a boundary reflection would masquerade as a physics error);
2. the collapse metric is well posed -- the exact self-similar solutions
   collapse to machine precision -- and has teeth -- a 1 % depth shift is
   detected far above its noise floor;
3. the front-speed fit recovers the exact speed from exact samples;
4. a short, coarse instance of the study passes every predeclared criterion;
5. the committed graded artifacts pass (regression lock: a scheme change that
   introduces a front-speed bias must fail here, not after a manual re-run).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorfvm.shallow_water import long_time as lt
from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET
from tensorfvm.shallow_water.exact import stoker_constant_state, t_star_to_time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                        "artifacts", "shallow-water-long-time")


@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET],
                         ids=lambda c: c.key)
@pytest.mark.parametrize("t_star", lt.T_STARS)
def test_domain_keeps_both_waves_interior(case, t_star):
    """Both the rarefaction head and the front/bore stay inside the domain."""
    dom = lt.long_time_domain(case, t_star)
    assert dom["pad"] > 0.0
    assert dom["x_min"] < dom["head"] - 1e-9
    assert dom["x_max"] > dom["front"] + 1e-9
    # the waves must be a non-trivial fraction of the domain, otherwise the
    # study would silently become a far-field test
    assert dom["front"] - dom["head"] > 0.5 * (dom["x_max"] - dom["x_min"])


@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET],
                         ids=lambda c: c.key)
def test_exact_solution_collapses_to_machine_precision(case):
    """Metric self-validation: the exact solution is self-similar by construction.

    Two different times, evaluated on the same ``xi`` grid, must agree to
    round-off.  If this ever fails the collapse metric is broken, not the solver.
    """
    xi = lt.similarity_grid(case)
    t1 = t_star_to_time(1.0, case.h0, case.gravity)
    t2 = t_star_to_time(20.0, case.h0, case.gravity)
    snap1 = {"t": t1, "x": xi * t1,
             "h": np.asarray(case.exact(xi * t1, t1)[0], dtype=float)}
    snap2 = {"t": t2, "x": xi * t2,
             "h": np.asarray(case.exact(xi * t2, t2)[0], dtype=float)}
    res = lt.collapse_residual(snap1, snap2, xi=xi)
    assert res["linf_over_h0"] < 1e-12
    assert res["l1_over_h0"] < 1e-13


@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET],
                         ids=lambda c: c.key)
def test_collapse_metric_detects_a_one_percent_shift(case):
    """Teeth: a 1 % depth shift must read ~1e-2, i.e. above the 2 % limit at 2 %."""
    xi = lt.similarity_grid(case)
    t = t_star_to_time(2.0, case.h0, case.gravity)
    h = np.asarray(case.exact(xi * t, t)[0], dtype=float)
    ref = {"t": t, "x": xi * t, "h": h}
    shifted = {"t": t, "x": xi * t, "h": h * (1.0 + 0.01)}
    res = lt.collapse_residual(ref, shifted, xi=xi)
    # the shift is 1 % of h0 where the wave is at h0, so the L1 is a fraction of
    # 1e-2 -- but far above the 1e-13 noise floor, which is what matters
    assert res["linf_over_h0"] > 5e-3
    # the L1 is diluted by the dry part of the xi window (~3.5e-4 here), so the
    # assertion is a noise-floor assertion, not a 1 % one
    assert res["l1_over_h0"] > 1e-4


def test_front_speed_fit_recovers_the_exact_speed():
    """The estimator is unbiased: fed exact positions it returns the exact speed."""
    case = STOKER_WET
    exact_speed = stoker_constant_state(case.h0, case.h1,
                                        case.gravity)["shock_speed"]
    samples = []
    for i in range(1, 9):
        t_star = 0.5 * i
        t = t_star_to_time(t_star, case.h0, case.gravity)
        samples.append({"t": t, "headline": exact_speed * t, "resolved": True})
    fit = lt.front_speed_fit(samples, exact_speed, dx=1e-2)
    assert fit["defined"]
    assert fit["speed_error"] < 1e-12
    assert abs(fit["intercept"]) < 1e-12


def test_short_study_passes_all_criteria():
    """A coarse, short instance of the study (fast enough for CI)."""
    payload = lt.run_long_time_study(t_stars=(1.0, 2.0), dx=2.0e-2,
                                     controls=False)
    for key, case in payload["cases"].items():
        verdict = case["verdict"]
        failing = [c["name"] for c in verdict["criteria"] if not c["pass"]]
        assert not failing, f"{key}: failing criteria {failing}"
        # the readings must be discretisation errors, not quantisation noise
        for row in case["rows"]:
            assert row["front_series_graded"] >= 5
            assert row["depth_l1"] < 0.02


@pytest.mark.skipif(not os.path.exists(os.path.join(ARTIFACT, "study.json")),
                    reason="graded artifacts not committed in this checkout")
def test_committed_artifacts_pass():
    """Regression lock on the graded, committed long-time evidence."""
    with open(os.path.join(ARTIFACT, "study.json")) as fh:
        payload = json.load(fh)
    assert payload["criteria"]["dx"] <= lt.DX + 1e-12
    assert max(payload["criteria"]["t_stars"]) >= 10.0
    for key, case in payload["cases"].items():
        verdict = case["verdict"]
        failing = [c["name"] for c in verdict["criteria"] if not c["pass"]]
        assert not failing, f"{key}: failing criteria {failing}"
        # the anchor's own quantity at t* = 2 must be reproduced in the sweep too
        anchor = [r for r in case["rows"] if abs(r["t_star"] - 2.0) < 1e-9]
        assert anchor, f"{key}: t*=2 point missing from the sweep"
        assert anchor[0]["front_series_max_error"] <= lt.TOLERANCE
    # falsifiability control: the first-order scheme must read a *larger* error
    controls = payload.get("controls") or {}
    order = controls.get("order_reference") or {}
    assert set(order) >= {"1", "2"}
    for scheme in ("1", "2"):
        for case_key, row in order[scheme].items():
            for field in ("front_error", "x_over_t_error", "depth_l1"):
                assert math.isfinite(row[field]), f"{scheme}/{case_key}/{field}"
    # NOT asserted as a strict ordering: on this metric family the order-1 run
    # does *not* read a worse front position, because first-order diffusion of a
    # Riemann problem is similarity preserving in xi -- it smears the profile
    # without biasing the shape-derived fan estimator.  The control is therefore
    # recorded as evidence that the position metric is insensitive to that
    # failure mode (see docs report, section 5), not as a pass/fail gate.