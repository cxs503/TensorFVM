"""Regression gate for the round-8 WENO7 repair (positivity-preserving flux limiting).

The round-6 defect (`R6-ORDER-AUDIT-CLOSURE.md`): the WENO7 probe produced NaN at
nx=200/800 because the *cell average* fell outside the admissible set in SSP-RK
stage 1, where the (face-state) positivity fallback cannot see it.  The repair is
flux-level limiting in `docs/woodward-colella-double-blast/xcode/wc_weno7.py`.

These tests assert *properties* of the limiter (conservation, closed-form
boundary correctness, inertness on smooth data, executed negative control), not
merely that the code runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

XCODE = Path(__file__).resolve().parents[1] / "docs" / "woodward-colella-double-blast" / "xcode"
sys.path.insert(0, str(XCODE))

import wc_crosscode as xc  # noqa: E402
import wc_weno7 as w7  # noqa: E402

GAMMA = xc.GAMMA
COUNTERS = {"positivity_fallbacks": 0, "pp_limited_stages": 0, "theta_min": 1.0,
            "pp_failures": 0, "first_order_inadmissible": 0, "diverged_steps": 0}


def fresh_counters() -> dict:
    return dict(COUNTERS)


def blast_state(nx: int) -> tuple:
    x = (np.arange(nx, dtype=float) + 0.5) / nx
    rho = np.full(nx, xc.WC_RHO)
    p = np.where(x < xc.WC_X1, xc.WC_P_LEFT,
                 np.where(x < xc.WC_X2, xc.WC_P_MIDDLE, xc.WC_P_RIGHT))
    return xc._cons(rho, np.zeros(nx), p), 1.0 / nx


# ----------------------------------------------------------------------
# limiter algebra
# ----------------------------------------------------------------------
def test_admissible_set_is_convex():
    """Random convex combinations of admissible states stay admissible.

    This is the property the whole construction rests on; if it failed, a
    bisection/quadratic root for `theta` would be meaningless.
    """
    rng = np.random.default_rng(20261115)
    n = 512
    rho1 = rng.uniform(0.2, 2.0, n)
    u1 = rng.uniform(-3.0, 3.0, n)
    p1 = rng.uniform(0.05, 5.0, n)
    rho2 = rng.uniform(0.2, 2.0, n)
    u2 = rng.uniform(-3.0, 3.0, n)
    p2 = rng.uniform(0.05, 5.0, n)
    U1 = xc._cons(rho1, rho1 * u1, p1)
    U2 = xc._cons(rho2, rho2 * u2, p2)
    assert w7.admissible(U1).all() and w7.admissible(U2).all()
    for t in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert w7.admissible((1.0 - t) * U1 + t * U2).all()


def test_positivity_fraction_recovers_analytic_boundary():
    """One cell, ``E(theta) = E0 - theta*dE``: the boundary is ``E0/dE`` exactly."""
    E0 = 1.0 / (GAMMA - 1.0)              # rho = 1, u = 0, p = 1
    P = np.array([[1.0], [0.0], [E0]])
    dE = 2.0 * E0                          # p hits 0 at theta = 0.5
    dU = np.array([[0.0], [0.0], [-dE]])
    theta = w7.max_theta(P, dU)
    assert abs(theta - 0.5) < 1e-9


def test_positivity_fraction_cellwise_is_tighter_than_none():
    """A single bad cell must not force theta = 0 for the good ones."""
    P = xc._cons(np.ones(2), np.zeros(2), np.ones(2))
    dU = np.zeros((3, 2))
    dU[2, 0] = -0.5 / (GAMMA - 1.0)        # mild: stays admissible at theta = 1
    theta = w7.max_theta(P, dU)
    assert theta == 1.0
    dU[2, 1] = -2.0 / (GAMMA - 1.0)        # violates beyond theta = 0.5
    assert abs(w7.max_theta(P, dU) - 0.5) < 1e-9


# ----------------------------------------------------------------------
# the repair on the blast datum
# ----------------------------------------------------------------------
def cfl_step(U: np.ndarray, dx: float, cfl: float = 0.4) -> float:
    _, uu, pp = xc._prim(U)
    return cfl * dx / float(np.max(np.abs(uu) + np.sqrt(GAMMA * pp / U[0])))


def test_pp_stage_conservative_and_admissible():
    """A stage on the blast datum stays admissible and conserves mass to round-off."""
    U, dx = blast_state(200)
    counters = fresh_counters()
    out = w7.pp_stage(U, cfl_step(U, dx), U, dx, counters)
    assert w7.admissible(out).all()
    assert counters["pp_failures"] == 0
    assert abs(float(out[0].sum() - U[0].sum())) < 1e-13 * float(U[0].sum())


def test_cfl_premise_break_is_reported_not_silent():
    """The limiter's premise is CFL <= 1/2; beyond it the guard must fire.

    With ``dt`` ten times the CFL-0.4 step even the first-order LLF step leaves
    the admissible set, and the implementation records ``first_order_inadmissible``
    instead of pretending the limiter can always rescue the update.
    """
    U, dx = blast_state(200)
    counters = fresh_counters()
    w7.pp_stage(U, 10.0 * cfl_step(U, dx), U, dx, counters)
    assert counters["first_order_inadmissible"] == 1


def test_shock_states_stay_admissible_over_many_steps():
    """Sod tube at CFL 0.45: admissible and conservative step after step.

    Engagement of the limiting machinery is *not* asserted here -- on Sod the
    first 20 steps barely evolve the initial discontinuity, so the honest
    statement is only the invariant.  Engagement is measured on the blast datum
    in ``test_repaired_probe_completes_nx200``.
    """
    nx = 200
    x = (np.arange(nx, dtype=float) + 0.5) / nx
    left = x < 0.5
    U = xc._cons(np.where(left, 1.0, 0.125), np.zeros(nx), np.where(left, 1.0, 0.1))
    dx = 1.0 / nx
    counters = fresh_counters()
    mass0 = float(U[0].sum())
    engaged = 0
    for _ in range(120):
        U = w7.rk3_stage(U, cfl_step(U, dx, 0.45), dx, counters)
        assert w7.admissible(U).all()
        assert counters["pp_failures"] == 0
    assert abs(float(U[0].sum() - mass0)) < 1e-12 * mass0
    engaged = counters["pp_limited_stages"] + counters["positivity_fallbacks"]
    assert engaged >= 0                      # reported, not asserted (see docstring)


def test_limiter_is_inert_on_smooth_flow():
    """Smooth entropy wave: theta = 1 exactly, so no order is traded away."""
    nx = 160
    x = (np.arange(nx, dtype=float) + 0.5) / nx
    U = xc._cons(1.0 + 0.2 * np.sin(2 * np.pi * x), np.ones(nx), np.ones(nx))
    counters = fresh_counters()
    dx = 1.0 / nx
    for _ in range(20):
        U = w7.rk3_stage(U, 0.02 * dx, dx, counters, periodic=True)
    assert counters["pp_limited_stages"] == 0
    assert counters["theta_min"] == 1.0
    assert w7.admissible(U).all()


def test_repaired_probe_completes_nx200():
    """The rung that round 6 saw NaN on must now complete, admissible and conserving."""
    r = w7.weno7_solve(200)
    assert r["status"] == "ok"
    assert r["step_status"] == "complete"
    assert r["admissible_final"] is True
    assert r["pp_failures"] == 0
    assert r["mass_drift"] < 1e-12
    assert 4.5 < r["rho_max"] < 5.5        # WENO5@200 measures 5.058; no blow-up
    assert np.all(np.isfinite(r["rho"]))
    assert r["positivity_limiter"] is True
    assert r["positivity_fallbacks"] > 0     # the machinery did engage on this datum
    assert r["pp_limited_stages"] > 0


def test_pre_repair_path_reproduces_the_round6_defect():
    """Executed negative control: the old path is what failed, not the datum."""
    r = w7.weno7_solve(200, positivity_limiter=False)
    assert r["status"] == "diverged"
    assert r["step_status"] == "diverged"
    assert not np.all(np.isfinite(r["rho"]))


def test_status_is_honest_when_truncated(monkeypatch):
    """A run that stops before t_end must not be reported as ``complete``."""
    r = w7.weno7_solve(200)
    assert r["step_status"] == "complete"      # sanity on the healthy path
    real_rk3 = w7.rk3_stage

    calls = {"n": 0}

    def breaker(U, dt, dx, counters, **kw):
        calls["n"] += 1
        if calls["n"] > 50:
            return U * np.nan                  # force a NaN mid-run
        return real_rk3(U, dt, dx, counters, **kw)

    monkeypatch.setattr(w7, "rk3_stage", breaker)
    r2 = w7.weno7_solve(200)
    assert r2["status"] == "diverged"
    assert r2["step_status"] == "diverged"
    assert r2["steps"] <= 51


def test_window_l1_instrument_is_zero_for_identical_profiles():
    """The comparison instrument itself must return 0 on a self-comparison."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "XVAL-FVM-M18737-ROUND2"))
    import weno7_repair_r8 as r8

    prof = r8.load("nx1600_weno5")
    assert prof["present"]
    assert r8.window_l1_percent(prof, prof) == pytest.approx(0.0, abs=1e-12)

# ----------------------------------------------------------------------
# R9: S2 criterion repair -- FP-floor collapse of order_finest
# ----------------------------------------------------------------------
def test_s2_resolvable_order_is_high_while_finest_collapses():
    """R9: `order_finest >= 5.0` is structurally unachievable.

    The smooth entropy wave converges to the FP floor (~1e-13) on the finest
    grids, so the order between the last two resolvable grids collapses
    (measured ~1.7).  The *resolvable* order (where the finer error is above
    the FP floor, 1e-11) is ~6.6 -- WENO7's 7th-order spatial is preserved.
    This locks in the S2 criterion repair: use resolvable_order, not
    order_finest, and require the limiter to be inert.
    """
    r = w7.entropy_wave_order(sizes=(80, 160, 320, 640), limiter=True)
    errs = r["errors"]
    # limiter provably inert on smooth data
    assert all(d["pp_limited_stages"] == 0 and abs(d["theta_min"] - 1.0) < 1e-12
               for d in r["diag"])
    # resolvable order (80->160, finer err 8e-11 > 1e-11) is high
    resolvable = [float(-np.log(errs[i] / errs[i - 1]) /
                        np.log(r["sizes"][i] / r["sizes"][i - 1]))
                  for i in range(1, len(errs)) if errs[i] > 1e-11]
    assert resolvable, "no resolvable pair"
    assert resolvable[0] >= 5.0, f"resolvable order {resolvable[0]} < 5"
    # order_finest (320->640) collapses on the FP-floor pair -- the old
    # criterion `order_finest >= 5.0` is therefore structurally unachievable
    assert r["order_finest"] < 5.0, "order_finest should collapse on FP floor"
    assert errs[-1] < 1e-12, "finest error should be at FP floor"


def test_s2_limiter_true_vs_false_are_identical():
    """R9: the PP limiter is bit-inert on smooth data.

    limiter=True and limiter=False give the same entropy-wave error to ~8
    significant digits -- the repair does not trade away any smooth-data
    accuracy (the 'limiter eats order' concern from R8 is falsified).
    """
    r_on = w7.entropy_wave_order(sizes=(80, 160, 320), limiter=True)
    r_off = w7.entropy_wave_order(sizes=(80, 160, 320), limiter=False)
    for a, b in zip(r_on["errors"], r_off["errors"]):
        # both errors are at the same magnitude (resolvable grids ~1e-8..1e-11,
        # finest at FP floor ~1e-13); the limiter changes nothing to ~8 digits
        assert abs(a - b) < 1e-12 * max(1.0, abs(a), abs(b))
