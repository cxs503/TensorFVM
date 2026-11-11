"""Tests for the round-2 Sod (1978) convergence diagnosis.

The study itself is a measurement, so these tests do not assert a *value* of the
observed order (that is physics and lands in the artifact).  They assert the
invariants that make the measurement trustworthy:

* the new ``superbee`` limiter sits inside the TVD region touched by the other
  limiters, is odd in its arguments, and is never more diffusive than ``mc``;
* the smooth windows of the exact solution are non-empty at every grid the study
  uses and never contain a wave;
* the restricted L1 norm is zero on an exact field and the wave positions are
  recovered from an exact-sampled field;
* the rate estimators return the rate they are given (``h**p`` synthetic data);
* one real measurement and one tiny study run end-to-end and write ``study.json``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from tensorfvm.compressible.reconstruct import (
    LIMITERS,
    mc,
    minmod,
    superbee,
    vanleer,
)
from tensorfvm.verification.sod1978 import (
    GAMMA,
    LEFT,
    RIGHT,
    T_TARGET,
    X0,
    exact_sample,
    exact_star_state,
)
from tensorfvm.verification.sod1978_convergence import (
    StudyConfig,
    fitted_exponent,
    measure,
    restricted_l1,
    run_study,
    smooth_windows,
    successive_orders,
)


# ----------------------------------------------------------------------
# limiter
# ----------------------------------------------------------------------
def test_superbee_registered():
    assert "superbee" in LIMITERS and LIMITERS["superbee"] is superbee


def test_superbee_is_tvd_and_odd():
    a = torch.linspace(-3.0, 3.0, 61, dtype=torch.float64)
    b = torch.linspace(3.0, -3.0, 61, dtype=torch.float64)
    phi = superbee(a, b)
    # odd in both arguments
    assert torch.allclose(superbee(-a, -b), -phi)
    # symmetric in the two slopes (Roe's maxmod form is)
    assert torch.allclose(superbee(b, a), superbee(a, b))
    same_sign = a * b > 0
    # TVD region: 0 <= phi <= 2 min(|a|, |b|)
    assert torch.all(phi.abs()[same_sign] <= 2.0 * torch.minimum(a.abs(), b.abs())[same_sign] + 1e-12)
    # opposite slopes are clipped to zero
    assert torch.all(phi[~same_sign] == 0.0)


def test_superbee_is_less_diffusive_than_mc():
    g = torch.Generator().manual_seed(1978)
    a = torch.rand(400, dtype=torch.float64, generator=g) * 2.0 + 0.1
    b = torch.rand(400, dtype=torch.float64, generator=g) * 2.0 + 0.1
    order = [minmod(a, b).abs().sum(), vanleer(a, b).abs().sum(),
             mc(a, b).abs().sum(), superbee(a, b).abs().sum()]
    assert order[0] <= order[1] + 1e-12
    assert order[1] <= order[2] + 1e-12
    assert order[2] <= order[3] + 1e-12


# ----------------------------------------------------------------------
# windows and norms
# ----------------------------------------------------------------------
@pytest.mark.parametrize("nx", [100, 200, 400, 800, 1600, 3200, 6400])
def test_smooth_windows_non_empty_and_wave_free(nx):
    star = exact_star_state()
    x = (np.arange(nx) + 0.5) / nx
    windows = smooth_windows(x, T_TARGET, star, margin_cells=16)
    assert [w[0] for w in windows] == ["left_state", "fan_interior", "star_left",
                                       "star_right", "right_state"]
    for label, lo, hi in windows:
        assert lo < hi, f"{label} window collapsed at nx={nx}"
        sel = (x >= lo) & (x <= hi)
        assert sel.any(), f"{label} window has no cells at nx={nx}"

    # the trimmed windows must stay away from every wave by at least one cell
    waves = [X0 + star[k] * T_TARGET for k in
             ("rarefaction_head_speed", "rarefaction_tail_speed",
              "contact_speed", "shock_speed")]
    h = 1.0 / nx
    for label, lo, hi in windows:
        if label in ("fan_interior",):
            # the fan interior legitimately sits between head and tail
            assert lo > waves[0] and hi < waves[1]
            continue
        for w in waves:
            assert not (lo < w < hi), f"{label} window contains wave at x={w}"


def test_restricted_l1_zero_on_exact_field():
    star = exact_star_state()
    nx = 200
    x = (np.arange(nx) + 0.5) / nx
    exact = exact_sample((x - X0) / T_TARGET, star)["rho"]
    windows = smooth_windows(x, T_TARGET, star, margin_cells=16)
    out = restricted_l1(x, exact, exact, windows)
    assert out["all"] == pytest.approx(0.0, abs=1e-15)
    assert out["plateaus"] == pytest.approx(0.0, abs=1e-15)
    assert out["cells"] > 0 and out["plateau_cells"] > 0
    # a constant offset must be recovered exactly by the norm
    out = restricted_l1(x, exact + 0.5, exact, windows)
    assert out["all"] == pytest.approx(0.5, rel=1e-12)


# ----------------------------------------------------------------------
# rate estimators
# ----------------------------------------------------------------------
def test_successive_orders_and_fit_on_synthetic_power_law():
    rows = [{"nx": nx, "e": 1.0 / nx ** 1.5} for nx in (100, 200, 400, 800)]
    rates = successive_orders(rows, "e")
    assert [round(r["rate"], 6) for r in rates] == [1.5, 1.5, 1.5]
    assert fitted_exponent(rows, "e") == pytest.approx(1.5, rel=1e-9)
    # a non-power-law sequence gives a decreasing rate, which is the signature the
    # study looks for on the whole-field Sod error
    rows = [{"nx": nx, "e": 1.0 / nx ** 0.9} for nx in (100, 200, 400)]
    rows.append({"nx": 800, "e": 1.0 / 800 ** 0.9 * 1.05})
    rates = successive_orders(rows, "e")
    assert rates[-1]["rate"] < rates[0]["rate"]


# ----------------------------------------------------------------------
# one real measurement
# ----------------------------------------------------------------------
def test_measure_recovers_waves_and_positive_error():
    row = measure(100, order=1)
    assert row["l1_rho_full"] > 0.0
    assert row["l1_rho_plateaus"] >= 0.0
    assert 0 < row["restricted_cells"] < row["nx"]
    # first-order scheme: wave positions good to a couple of cells
    h = row["dx"]
    for name in ("rarefaction_head_x", "rarefaction_tail_x", "contact_x", "shock_x"):
        assert row["wave_error"][name] < 6.0 * h, (name, row["wave_error"][name])

    # limiter does change the answer (it is a real knob, not a placeholder)
    row_mc = measure(100, order=2, limiter="mc")
    row_sb = measure(100, order=2, limiter="superbee")
    assert row_sb["l1_rho_full"] < row_mc["l1_rho_full"], "superbee must be sharper"


# ----------------------------------------------------------------------
# end-to-end tiny study
# ----------------------------------------------------------------------
def test_tiny_study_writes_artifact(tmp_path):
    cfg = StudyConfig(
        grids=(50, 100), orders=(1,), limiter="mc", cfl=0.4,
        limiter_grids=(50,), limiters=("mc", "superbee"),
        cfl_probe_nx=50, cfl_values=(0.4,), smooth_grids=(50,), figures=False,
    )
    artifact = run_study(cfg, tmp_path, log=lambda *a, **k: None)
    out = tmp_path / "study.json"
    assert out.exists()
    reloaded = json.loads(out.read_text())
    assert reloaded["anchor"] == "REPRO-FVM-SOD1978"
    assert reloaded["round"] == 2
    assert reloaded["thresholds_declared_before_run"]["c3_full_l1_rate"] == 1.0
    verdicts = artifact["verdicts"]
    for key in ("c3_full_l1_rate_finest_pair_order2",
                "c3_full_exponent_finest3_order2",
                "plateau_rate_finest_pair_order2",
                "contact_position_rate_order2",
                "shock_position_rate_order2",
                "c3_literal_met", "restricted_design_order_met"):
        assert key in verdicts
    # the round-1 criterion is not silently turned into a pass
    assert isinstance(verdicts["c3_literal_met"], bool)
    row = artifact["tables"]["order1"][0]
    assert math.isfinite(row["l1_rho_full"]) and row["l1_rho_full"] > 0