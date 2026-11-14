"""Tests for the causal stopping-iteration forecast, v3.

Two kinds of assertions:

* **invariants** on synthetic histories where the true crossing is known in
  closed form (a geometric series must be reproduced exactly, an accelerating
  series must be reproduced inside the acceptance band, and the forecast must be
  a pure function of the prefix);
* **claims** made in ``docs/xval-round2/FORECAST-V3.md``, checked against the
  committed artifact ``docs/xval-round2/causal_forecast_v3.json`` -- the test
  fails if the artifact is missing (the earlier round-2 suite silently skipped
  because its path was wrong).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfvm.convergence_diagnosis import DiagnosisConfig
from tensorfvm.verification.convergence_forecast_v3 import (CONFIDENCES,
                                                            T_CRIT, forecast,
                                                            causal_series,
                                                            fit_quadratic,
                                                            select_model)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "xval-round2" / "causal_forecast_v3.json"
HISTORIES = ROOT / "docs" / "convergence-diagnosis-v2" / "histories"

CFG = DiagnosisConfig(tolerance=1e-8, rel_tol=0.0, max_iter=2000, window=5,
                      min_points=6, ratio_spread_tol=0.20, min_decades=1.0,
                      rate_stability_tol=0.05)


def _series(values):
    residuals = [float(v) for v in values]
    iterations = list(range(len(residuals)))
    return residuals, iterations


def _geometric(rho: float, n: int = 260, r0: float = 1.0):
    return _series([r0 * rho ** k for k in range(n)])


def _accelerating(n: int = 220, a: float = 1e-3):
    """``log r = -a k^2``: a strictly accelerating (superlinear) decay."""
    return _series([math.exp(-a * k * k) for k in range(n)])


def _true_crossing(residuals, threshold):
    for k, r in enumerate(residuals):
        if r < threshold:
            return k
    return None


def test_geometric_series_is_reproduced_exactly():
    residuals, iterations = _geometric(0.9)
    stop = _true_crossing(residuals, CFG.tolerance)
    res = forecast(residuals[:150], iterations[:150], CFG)
    assert res["model"] == "constant"
    assert res["forecast_iterations"] == pytest.approx(stop, abs=1)


def test_accelerating_series_selects_quadratic_and_hits_the_crossing():
    residuals, iterations = _accelerating()
    stop = _true_crossing(residuals, CFG.tolerance)
    fit = fit_quadratic(residuals[:120], iterations[:120])
    assert fit is not None and fit.c < 0.0 and fit.t_c <= -T_CRIT
    res = forecast(residuals[:120], iterations[:120], CFG)
    assert res["model"] == "quadratic"
    assert abs(res["forecast_iterations"] - stop) / stop <= 0.15


def test_forecast_depends_only_on_the_prefix():
    """Appending a different future must not change any earlier forecast."""
    residuals, iterations = _accelerating()
    other = residuals[:120] + [1e-30, 1e-30, 1e-30]
    a = causal_series(residuals[:120], iterations[:120], CFG)
    b = causal_series(other[:120], list(range(120)), CFG)
    assert [s["forecast"] for s in a] == [s["forecast"] for s in b]
    # and every recorded entry equals a standalone prefix forecast
    for k in (40, 90, 119):
        solo = forecast(residuals[:k + 1], iterations[:k + 1], CFG)
        assert solo["forecast_iterations"] == a[k]["forecast"]


def test_non_decaying_series_publishes_nothing():
    residuals, iterations = _series([1.0 + 0.01 * k for k in range(80)])
    res = forecast(residuals, iterations, CFG)
    assert res["forecast_iterations"] is None
    assert res["model"] in ("none", "constant")


def test_interval_brackets_the_point_forecast():
    residuals, iterations = _geometric(0.95, n=400)
    res = forecast(residuals[:120], iterations[:120], CFG)
    point = res["forecast_iterations"]
    assert point is not None and res["interval"]
    for conf in (f"{c:.2f}" for c in CONFIDENCES):
        iv = res["interval"].get(conf)
        if iv is None:
            continue
        assert iv["lo"] <= point <= iv["hi"]
    iv68, iv90 = res["interval"].get("0.68"), res["interval"].get("0.90")
    if iv68 and iv90:
        assert iv90["hi"] - iv90["lo"] >= iv68["hi"] - iv68["lo"]


def test_curvature_gate_rejects_a_late_kink():
    """A bend present only in the last few points must not become a quadratic fit."""
    residuals, iterations = _geometric(0.9, n=200)
    kinked = list(residuals[:150])
    for i in range(5):                     # flatten the tail -> spurious curvature
        kinked[145 + i] = kinked[144] * (0.999 ** i)
    model = select_model(kinked, iterations[:150])
    assert model.kind == "constant"


def test_committed_histories_artifact():
    if not ARTIFACT.exists():
        pytest.fail(f"missing artifact {ARTIFACT}: regenerate with "
                    "PYTHONPATH=src python docs/xval-round2/causal_forecast_v3.py")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    decaying = [c for c in payload["cases"] if c["decaying"]]
    assert len(decaying) == 6
    assert sorted(payload["meta"]["arms"]) == ["v1", "v2", "v2g", "v3"]
    # the capability claim: at a lead of 10 % v3 is inside 15 % on every case
    for case in decaying:
        entry = case["arms"]["v3"]["leads"]["0.1"]
        assert entry["published"], case["case"]
        assert entry["error"] <= 0.15, (case["case"], entry["error"])
    # ... and the long-lead claim is honestly recorded as *not* met
    long_lead = payload["meta"]["arms"]["v3"]["leads"]["0.25"]
    assert long_lead["passed_15pct"] < long_lead["n_published"]


def test_committed_histories_are_reproducible():
    """Re-running v3 on the committed histories reproduces the recorded lead-10 % errors."""
    if not ARTIFACT.exists():
        pytest.fail(f"missing artifact {ARTIFACT}")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    checked = 0
    for case in payload["cases"]:
        if not case["decaying"]:
            continue
        hist = json.loads((HISTORIES / f"{case['case']}.json").read_text())
        residuals = [h["residual"] for h in hist["history"]]
        iterations = [h["iteration"] for h in hist["history"]]
        overrides = {k: v for k, v in CFG.__dict__.items()
                     if k not in ("tolerance", "rel_tol")}
        cfg = DiagnosisConfig(tolerance=hist["tolerance"],
                              rel_tol=hist["rel_tol"], **overrides)
        series = causal_series(residuals, iterations, cfg)
        stop = case["stopping_iteration"]
        k = int(round(stop * 0.9))
        recorded = case["arms"]["v3"]["leads"]["0.1"]["error"]
        if recorded is None:
            assert series[k]["forecast"] is None
            continue
        recomputed = abs(series[k]["forecast"] - stop) / stop
        assert recomputed == pytest.approx(recorded, abs=1e-9)
        checked += 1
    assert checked == 6