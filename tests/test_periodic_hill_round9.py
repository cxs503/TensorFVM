"""Round 9 of CASE-FVM-PERIODIC-HILLS: time-averaged gate post-processing.

Every helper is pinned to an *independent* check rather than to itself:

* the wall-shear crossing search is cross-checked against the merged
  ``PeriodicHillSolver.reattachment`` on the same shear profile;
* ``horizontal_profile`` is checked against the analytic plane-Poiseuille
  solution on the hill-free mesh (``hill_amplitude = 0``), and against the
  "inside the hill" rejection rule on the real geometry;
* the stationarity diagnostics are checked on series whose answer is known by
  construction (constant, drifting, non-finite);
* the profile comparison is checked against the round-6 single-sourced table,
  including the near-zero band switch.

None of this asserts that the case *passes*: rounds 1--8 already record that
the 2-D URANS answer is over-diffusive.  What is asserted here is that the
time-averaged metrics are computed correctly, so the round-9 verdict (in
``docs/case-periodic-hills/ROUND9.md``) is about physics and not about
post-processing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.periodic_hill import (
    PeriodicHillConfig,
    PeriodicHillSolver,
    Statistics,
    analytic_flat_channel,
)
from tensorfvm.periodic_hill_turbulence import SpalartAllmaras, SpalartAllmarasConfig
from tensorfvm.verification.periodic_hill_round6 import (
    BREUER_PROFILE_Y05,
    REFERENCE,
    REFERENCE_TOLERANCE,
)
from tensorfvm.verification.periodic_hill_round9 import (
    averaging_window,
    block_means,
    compare_profile,
    horizontal_profile,
    stationarity_report,
    time_averaged_reattachment,
    zero_crossings_periodic,
)


# --------------------------------------------------------------------------- #
# wall-shear crossings
# --------------------------------------------------------------------------- #
def _step_shear(n, x0=-4.5, x1=4.5, reversed_intervals=((-3.0, 0.5), (1.0, 2.5))):
    """Uniform periodic grid whose shear is ``-1`` on given intervals, else ``+1``.

    Only ``tau < 0 -> tau >= 0`` transitions are crossings (the merged
    production convention: the reversed-flow region ends where the wall shear
    returns to positive), so each interval contributes **one** crossing at its
    right end.
    """
    x = np.linspace(x0, x1, n, endpoint=False)
    tau = np.ones_like(x)
    for lo, hi in reversed_intervals:
        tau[(x > lo) & (x < hi)] = -1.0
    return tau, x


def test_zero_crossings_recovers_known_zeros():
    tau, x = _step_shear(901)
    out = zero_crossings_periodic(tau, x)
    assert out["finite"] is True
    # reversed intervals end at x/H = 0.5 and 2.5 -> separation / reattachment
    assert out["separation"] == pytest.approx(0.5, abs=0.02)
    assert out["reattachment"] == pytest.approx(2.5, abs=0.02)
    assert out["sign_changes"] == 2


def test_zero_crossings_returns_nan_when_no_crossing_is_admissible():
    # shear negative everywhere on the period: reversed flow never reattaches
    tau, x = _step_shear(901, reversed_intervals=((-5.0, 5.0),))
    out = zero_crossings_periodic(tau, x)
    assert not np.isfinite(out["separation"])
    assert not np.isfinite(out["reattachment"])
    assert out["crossings"] == []
    assert out["sign_changes"] == 0


def test_zero_crossings_matches_solver_reattachment():
    """The new search must reproduce the merged production path exactly."""
    solver = PeriodicHillSolver(PeriodicHillConfig(nx=96, ny=48, reynolds=2800.0))
    solver.turbulence = SpalartAllmaras(solver, SpalartAllmarasConfig())
    solver.advance(120)
    tau_b, _ = solver.wall_shear()
    x = solver.mesh.x_n[0] / solver.config.height_m
    sep_solver, rea_solver = solver.reattachment()
    out = zero_crossings_periodic(tau_b, x)
    if not np.isfinite(sep_solver):
        assert not np.isfinite(out["separation"])
    else:
        assert out["separation"] == pytest.approx(sep_solver, abs=1e-9)
    if not np.isfinite(rea_solver):
        assert not np.isfinite(out["reattachment"])
    else:
        assert out["reattachment"] == pytest.approx(rea_solver, abs=1e-9)


def test_zero_crossings_flags_nonfinite_shear():
    tau, x = _step_shear(101)
    tau[7] = np.nan
    out = zero_crossings_periodic(tau, x)
    assert out["finite"] is False
    assert not np.isfinite(out["reattachment"])


def test_zero_crossings_rejects_bad_shapes():
    with pytest.raises(ValueError):
        zero_crossings_periodic(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError):
        zero_crossings_periodic(np.zeros(2), np.zeros(2))


def test_time_averaged_reattachment_reports_both_lengths():
    tau, x = _step_shear(901)
    out = time_averaged_reattachment(tau, x)
    assert out["x_s_over_H"] == pytest.approx(0.5, abs=0.02)
    assert out["x_r_over_H"] == pytest.approx(2.5, abs=0.02)


# --------------------------------------------------------------------------- #
# horizontal profile
# --------------------------------------------------------------------------- #
def test_horizontal_profile_recovers_analytic_poiseuille():
    """On the hill-free mesh the sampled line must return u(y) exactly."""
    cfg = PeriodicHillConfig(nx=24, ny=32, reynolds=100.0, hill_amplitude=0.0)
    solver = PeriodicHillSolver(cfg)
    G, nu, H = 0.05, cfg.nu, cfg.channel_top_m
    yc = solver.mesh.yc
    u_exact = analytic_flat_channel(yc, H, G, nu)
    x_over_h = solver.mesh.xc[0] / cfg.height_m
    out = horizontal_profile(u_exact, yc, x_over_h, y_over_h=0.5, height=cfg.height_m)
    assert out["inadmissible"] == []
    expected = analytic_flat_channel(0.5 * cfg.height_m, H, G, nu)
    # linear interpolation of a quadratic between cell centres: the residual is
    # O(dy^2 * |u''| / 8) with dy = 3.036 / 32, i.e. ~3e-3
    assert np.max(np.abs(out["u"] - expected)) < 4e-3
    # only one distinct value: the field does not vary in x
    assert np.ptp(out["u"]) < 1e-12


def test_horizontal_profile_rejects_columns_inside_the_hill():
    """At ``y/H = 0.5`` the crest region is solid: those columns are dropped."""
    cfg = PeriodicHillConfig(nx=48, ny=24, reynolds=2800.0)
    solver = PeriodicHillSolver(cfg)
    x_over_h = solver.mesh.xc[0] / cfg.height_m
    out = horizontal_profile(np.ones_like(solver.u), solver.mesh.yc, x_over_h,
                             y_over_h=0.5, height=cfg.height_m)
    assert len(out["inadmissible"]) > 0
    assert np.all(out["x_over_h"][np.abs(out["x_over_h"]) > 1.0])
    # the mask is consistent with the returned columns
    assert out["mask"].sum() == len(out["indices"])


# --------------------------------------------------------------------------- #
# stationarity diagnostics
# --------------------------------------------------------------------------- #
def test_block_means_are_contiguous():
    values = np.arange(16.0)
    blocks = block_means(values, 4)
    assert blocks.tolist() == [1.5, 5.5, 9.5, 13.5]


def test_stationarity_accepts_constant_series():
    t = np.linspace(0.0, 100.0, 400)
    rep = stationarity_report(t, np.full_like(t, 4.2), n_blocks=4)
    assert rep["ok"] is True
    assert rep["spread_rel"] == pytest.approx(0.0, abs=1e-12)
    assert rep["mean"] == pytest.approx(4.2)


def test_stationarity_rejects_a_drifting_series():
    t = np.linspace(0.0, 100.0, 400)
    rep = stationarity_report(t, 1.0 + 0.05 * t, n_blocks=4)
    assert rep["ok"] is False
    # the spread is reported, not just the verdict
    assert rep["spread_rel"] > 0.05
    assert rep["drift_rel"] > 0.05


def test_stationarity_rejects_nonfinite_samples():
    t = np.linspace(0.0, 100.0, 40)
    v = np.full_like(t, 3.0)
    v[5] = np.nan
    rep = stationarity_report(t, v, n_blocks=4)
    assert rep["ok"] is False
    assert rep["finite_samples"] == 39


def test_averaging_window_drops_leading_transient():
    t = np.linspace(0.0, 200.0, 800)
    v = np.where(t < 100.0, 1.0 + 0.5 * t / 100.0, 2.0)
    # interleave a small noise so the block spread is not exactly zero
    v = v + 1e-6 * np.sin(t)
    rep = averaging_window(t, v, n_blocks=8, tolerance=1e-3, drift_tolerance=1e-3)
    assert rep["ok"] is True
    assert rep["blocks_dropped"] >= 1
    assert rep["mean"] == pytest.approx(2.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# profile comparison against the single-sourced reference
# --------------------------------------------------------------------------- #
def test_compare_profile_accepts_the_reference_itself():
    x = np.array(sorted(BREUER_PROFILE_Y05))
    u = np.array([BREUER_PROFILE_Y05[k] for k in x])
    out = compare_profile(x, u)
    assert out["n_stations"] >= 3
    assert out["ok"] is True
    assert out["worst"]["error"] == pytest.approx(0.0, abs=1e-12)


def test_compare_profile_rejects_a_shifted_profile():
    x = np.array(sorted(BREUER_PROFILE_Y05))
    u = np.array([BREUER_PROFILE_Y05[k] for k in x]) - 0.5
    out = compare_profile(x, u)
    assert out["ok"] is False
    assert out["n_pass"] == 0


def test_compare_profile_uses_absolute_band_near_zero():
    # reference at x/H = 1.5 is 0.05 -> absolute band: a 0.06 miss is 0.06,
    # not 120 % (which is what a naive relative criterion would report)
    out = compare_profile([1.5], [0.20])
    assert out["n_stations"] == 1
    row = out["stations"][0]
    assert row["band"] == "absolute"
    assert row["error"] == pytest.approx(0.15, abs=1e-12)   # |0.20 - 0.05|
    assert row["pass"] is False
    # ... while a 0.06 miss stays inside the 0.10 absolute band
    assert compare_profile([1.5], [0.11])["stations"][0]["pass"] is True


def test_compare_profile_reports_unmatched_columns():
    out = compare_profile([0.0, 4.0], [0.0, 0.85])
    # x/H = 0 has no reference station within the attribution window
    assert 0.0 in out["unmatched_columns"]
    assert out["n_stations"] == 1


def test_reference_tolerance_is_the_ledger_tolerance():
    assert REFERENCE_TOLERANCE == pytest.approx(0.10)
    assert REFERENCE["x_r_over_H"] == pytest.approx(5.34)


# --------------------------------------------------------------------------- #
# Statistics extension
# --------------------------------------------------------------------------- #
def test_statistics_accumulates_wall_shear():
    cfg = PeriodicHillConfig(nx=24, ny=16, reynolds=200.0)
    solver = PeriodicHillSolver(cfg)
    stats = Statistics(solver)
    tau_samples, u_samples = [], []
    for _ in range(3):
        solver.advance(1)
        tau_samples.append(solver.wall_shear()[0].copy())
        u_samples.append(solver.u.copy())
        stats.accumulate(solver)
    mean_tau_b, mean_tau_t = stats.mean_wall_shear()
    assert mean_tau_b == pytest.approx(np.mean(tau_samples, axis=0))
    assert mean_tau_t.shape == stats.tau_t.shape
    # the merged API contract is unchanged
    mean_u, mean_v, mean_G = stats.mean()
    assert mean_u.shape == solver.u.shape
    assert mean_u == pytest.approx(np.mean(u_samples, axis=0))


def test_statistics_mean_is_time_average_of_samples():
    cfg = PeriodicHillConfig(nx=16, ny=12, hill_amplitude=0.0, reynolds=50.0)
    solver = PeriodicHillSolver(cfg)
    stats = Statistics(solver)
    us = []
    for _ in range(2):
        solver.advance(3)
        us.append(solver.u.copy())
        stats.accumulate(solver)
    mean_u, _, _ = stats.mean()
    assert mean_u == pytest.approx(0.5 * (us[0] + us[1]))