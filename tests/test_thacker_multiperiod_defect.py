"""Regression guards for the open multi-period defect of CASE-FVM-THACKER-BASIN.

``docs/case-fvm-thacker-basin/REPORT.md`` §4.4 records that the Thacker basin
run cannot be extended past ``t = 0.5T``: the order-2 scheme develops a
non-physical velocity runaway, the CFL step collapses and the history departs
from the closed form.  ``docs/case-fvm-thacker-basin/multiperiod/`` diagnoses
that defect; these tests keep the diagnosis honest from both sides:

* the defect is *real and still open* -- the order-2 run over ``2T`` must still
  exceed the analytic velocity envelope by a large factor, and the centre-depth
  history must still be far outside the 3 % acceptance.  If a future fix makes
  these pass, ``test_two_period_history_is_within_tolerance`` stops failing and
  the strict xfail below turns into a failure, which is the intended signal to
  retire the debt entry instead of silently overwriting it;
* the **control** is not the culprit -- the order-1 scheme must stay inside the
  analytic envelope over the same window, so the runaway cannot be blamed on the
  wet/dry model itself;
* the velocity runaway is **not** sufficient to explain the history error -- the
  order-1 control, whose velocity is bounded, misses the 2T history as badly as
  order 2.  That is the falsified hypothesis recorded in the diagnosis and it is
  what stops a "momentum cut-off" from being sold as a fix.

Everything runs at ``n = 40`` and ``2T = 1.42 s`` (a few seconds per test), so
the guards are cheap enough for the default suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.verification import thacker_basin as tb

N = 40
PERIODS = 2.0


def _envelope(basin, n_grid: int = 60, samples: int = 120) -> float:
    """Analytic ``max|u|`` over one oscillation period (closed form)."""
    X, Y = tb.grid(basin, n_grid, tb.DEFAULTS["domain_factor"])
    times = np.linspace(0.0, basin.period, samples)
    peak = 0.0
    for t in times:
        u, v = basin.velocity(X, Y, t)[:2]
        peak = max(peak, float(np.nanmax(np.hypot(np.asarray(u), np.asarray(v)))))
    return peak


@pytest.fixture(scope="module")
def basin():
    d = tb.DEFAULTS
    return tb.build_basin(**{k: d[k] for k in ("H", "a", "amplitude", "gravity")})


@pytest.fixture(scope="module")
def order2_run(basin):
    return tb.run_case(n=N, periods=PERIODS, order=2)


@pytest.fixture(scope="module")
def order1_run(basin):
    return tb.run_case(n=N, periods=PERIODS, order=1)


def test_order2_velocity_runaway_is_still_present(basin, order2_run):
    """The order-2 run over 2T must still leave the analytic velocity envelope.

    Measured baseline (``multiperiod/artifacts/summary.json``): the order-2 run
    reaches ``71.2 m/s`` against an analytic envelope of ``2.46 m/s`` (factor
    ~29).  The assertion is deliberately loose (factor 3) so that a *partial*
    improvement does not hide the fact that the defect is unfixed.
    """
    env = _envelope(basin)
    # reconstruct the velocity history from the recorded series is not possible,
    # so re-integrate a short trajectory: the runaway is already visible by 1.5T
    d = tb.DEFAULTS
    params = dict(d)
    solver = tb._solver(basin, N, params)
    X, Y = tb.grid(basin, N, d["domain_factor"])
    solver.set_state(*basin.initial_state(X, Y))
    peak = {"value": 0.0}

    def cb(sol):
        peak["value"] = max(peak["value"], sol.max_spurious_velocity())

    solver.advance(1.5 * basin.period, max_steps=200000, callback=cb)
    assert peak["value"] > 3.0 * env, (
        "order-2 velocity runaway over 1.5T no longer reproduces "
        "(peak %.3f m/s vs envelope %.3f m/s) -- re-grade the debt entry "
        "docs/case-fvm-thacker-basin/multiperiod/README.md"
        % (peak["value"], env))


def test_order1_control_stays_inside_the_envelope(basin, order1_run):
    """The first-order control must stay physically bounded over 2T.

    Measured running max over ``2T``: ``2.181 m/s`` against the ``2.461 m/s``
    analytic envelope (ratio 0.886); the order-2 run on the same window reaches
    ``71.16 m/s`` (ratio 28.9).  If this fails, the runaway is in the
    wet/dry model rather than in the second-order reconstruction and the
    diagnosis in the README has to be re-derived.
    """
    env = _envelope(basin)
    d = tb.DEFAULTS
    params = dict(d)
    params["order"] = 1
    solver = tb._solver(basin, N, params)
    X, Y = tb.grid(basin, N, d["domain_factor"])
    solver.set_state(*basin.initial_state(X, Y))
    peak = {"value": 0.0}

    def cb(sol):
        peak["value"] = max(peak["value"], sol.max_spurious_velocity())

    solver.advance(2.0 * basin.period, max_steps=200000, callback=cb)
    assert peak["value"] <= env, (
        "order-1 control left the analytic envelope (%.3f > %.3f m/s)"
        % (peak["value"], env))


@pytest.mark.xfail(strict=True, reason=(
    "open defect: CASE-FVM-THACKER-BASIN is verified only on 0 <= t <= 0.5T. "
    "Over 2T the centre-depth history misses the closed form by 29-35 % even "
    "with bounded velocity (order-1 control: 49 %), so the failure is phase/"
    "amplitude drift, not the momentum artefact.  Strict xfail: a fix must flip "
    "this to XPASS, see multiperiod/README.md §4."))
def test_two_period_history_is_within_tolerance(order2_run):
    """``2T`` centre-depth history must be inside the 3 % acceptance."""
    assert order2_run["centre_depth"]["l2_relative"] <= 0.03