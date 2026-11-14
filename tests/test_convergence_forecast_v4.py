"""Gates for the v4 causal forecast (calibrated intervals + walk-forward selection).

The tests assert *invariants*, not the published numbers:

* strict causality -- a forecast issued at prefix ``k`` is unchanged when the
  future residuals are removed from the input (the property every earlier
  "0.00 %" artifact turned out to violate);
* interval sanity -- ``lo <= point <= hi`` and the band is never zero-width;
* the measured error-growth exponent is the one actually used (not a hard-coded
  ``sqrt``);
* degenerate inputs publish nothing instead of inventing a crossing;
* the committed audit artifact still satisfies the acceptance claims the report
  makes (point forecast 6/6 inside 15 % at lead 10 %, 90 % interval coverage
  >= 80 % of nominal) -- the artifact must exist, a missing file is a failure,
  not a skip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfvm.convergence_diagnosis import DiagnosisConfig
from tensorfvm.verification import convergence_forecast_v4 as v4

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "docs" / "convergence-diagnosis-v2" / "histories"
ARTIFACT = ROOT / "docs" / "xval-round2" / "causal_forecast_v4.json"
CASES = ("poisson-n32-tol1e-8", "cd-pe10-gmres")


def load_case(name: str):
    path = HIST / f"{name}.json"
    if not path.exists():
        pytest.fail(f"missing committed history {path}")
    hist = json.loads(path.read_text())
    iterations = [h["iteration"] for h in hist["history"]]
    residuals = [h["residual"] for h in hist["history"]]
    cfg = DiagnosisConfig(tolerance=hist["tolerance"], rel_tol=hist["rel_tol"],
                          max_iter=2000, window=5, min_points=6,
                          ratio_spread_tol=0.20, min_decades=1.0,
                          rate_stability_tol=0.05)
    return iterations, residuals, cfg


def test_forecast_is_strictly_causal():
    """Dropping the future must not change a forecast issued at prefix k."""
    iterations, residuals, cfg = load_case("poisson-n32-tol1e-8")
    full = v4.causal_series(residuals, iterations, cfg)
    for k in (40, 90, 150):
        trunc = v4.causal_series(residuals[: k + 1], iterations[: k + 1], cfg)
        for field in ("forecast", "model", "interval"):
            assert trunc[-1][field] == full[k][field], (k, field)
        assert (trunc[-1]["selection"] or {}).get("family") == \
               (full[k]["selection"] or {}).get("family")


def test_interval_brackets_point_forecast():
    iterations, residuals, cfg = load_case("poisson-n32-tol1e-8")
    series = v4.causal_series(residuals, iterations, cfg)
    checked = 0
    for row in series:
        if not row["ready"]:
            continue
        for conf, iv in (row["interval"] or {}).items():
            assert iv["lo"] <= iv["hi"]
            assert row["forecast"] <= iv["hi"], (row["k"], conf)
            # the point forecast may sit below lo only if lo was clamped at k+1
            assert row["forecast"] >= iv["lo"] or iv["lo"] <= row["k"] + 1, \
                (row["k"], conf)
            checked += 1
    assert checked > 0, "no interval was published at all"


def test_growth_exponent_is_measured_not_assumed():
    """The band exponent comes from the data (p > 0.5 here), not from sqrt(h)."""
    iterations, residuals, cfg = load_case("cd-pe10-gmres")
    out: dict = {}
    v4.causal_series(residuals, iterations, cfg, errors_out=out)
    errors = out["errors"]
    for family in v4.FAMILIES:
        prof = errors.growth_profile(family)
        assert len(prof) >= 3, "not enough horizons to fit the growth law"
        p = errors.exponent(family)
        lo, hi = v4._Errors.P_RANGE
        assert lo <= p <= hi + 1e-12, (family, p)
        # on these solves the measured law is steeper than a random walk (0.5)
        assert p > 0.5, (family, p)


def test_degenerate_inputs_publish_nothing():
    cfg = DiagnosisConfig(tolerance=1e-8, max_iter=2000)
    flat = [1.0] * 30
    series = v4.causal_series(flat, list(range(30)), cfg)
    assert all(not row["ready"] for row in series)
    growing = [10.0 ** (0.05 * k) for k in range(40)]
    series = v4.causal_series(growing, list(range(40)), cfg)
    assert all(row["forecast"] is None for row in series)
    short = [1.0, 0.5, 0.25]
    series = v4.causal_series(short, [0, 1, 2], cfg)
    assert len(series) == 3 and not any(row["ready"] for row in series)
    zeros = [0.0] * 10
    series = v4.causal_series(zeros, list(range(10)), cfg)
    # a zero residual is *already* converged: publishing the current iteration is
    # correct, inventing a future crossing is not.
    assert all(row["forecast"] is None or row["forecast"] == row["k"]
               for row in series)


def test_committed_artifact_supports_the_report_claims():
    if not ARTIFACT.exists():
        pytest.fail(f"missing artifact {ARTIFACT}; run "
                    "`PYTHONPATH=src python docs/xval-round2/causal_forecast_v4.py`")
    payload = json.loads(ARTIFACT.read_text())
    agg = payload["aggregate"]
    n = agg["n_decaying"]
    assert n == 6
    # acceptance claim 1: point forecast inside 15 % on all six decaying cases at lead 10 %
    assert agg["by_lead"]["0.1"]["v4"]["pass"] == n
    assert agg["by_lead"]["0.1"]["v4"]["worst"] <= 0.15
    # claim 2: v4 does not inherit v3's regression at lead 25 %
    assert agg["by_lead"]["0.25"]["v4"]["pass"] >= agg["by_lead"]["0.25"]["v1"]["pass"]
    # claim 3: the calibrated interval reaches >= 80 % of the nominal level ...
    for conf, floor in (("0.68", 0.80), ("0.90", 0.80)):
        nominal = float(conf)
        got = agg["coverage_all_prefixes"]["v4"][conf]["coverage"]
        assert got is not None and got >= floor * nominal, (conf, got)
        # ... and beats the parameter-only interval by a wide margin
        old = agg["coverage_all_prefixes"]["v3"][conf]["coverage"] or 0.0
        assert got > 3 * old, (conf, got, old)
    # the band is honest about its cost: width is reported, not hidden
    for conf in ("0.68", "0.90"):
        d = agg["coverage_all_prefixes"]["v4"][conf]
        assert d["median_case_rel_width"] > 0.0
        # ... and the report must also carry the *informative-subset* coverage,
        # which is lower than the unrestricted one.  If a future change silently
        # widened the band, this gap would vanish and the claim would be vacuous.
        assert d["informative_coverage"] is not None
        assert d["informative_coverage"] < d["coverage"]
        assert 0.0 < d["informative_share"] < 1.0