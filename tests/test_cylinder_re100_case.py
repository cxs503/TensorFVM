"""Tests for the m21018 rework of CASE-CYLINDER-RE100 on the immersed backend.

The physics test (``test_force_is_control_volume_independent``) is the one that
matters: the momentum budget is an identity, so two nested control volumes must
return the same force on the same field.  Everything else pins the honest gates
(no St without shedding) and the reference binding.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfvm.immersed.forces import (ForceRecorder, budget_terms, control_box)
from tensorfvm.immersed.geometry import circular_cylinder, cylinder_mask_area_error
from tensorfvm.verification import cylinder_re100 as case

REPO = Path(__file__).resolve().parents[1]
REFERENCE_FILE = REPO / "artifacts" / "case-cylinder-re100" / "reference" / "reference.json"


def _stub_flow(grid, u, v, p, nu=0.01, rho=1.0):
    return SimpleNamespace(u=u, v=v, p=p,
                           config=SimpleNamespace(grid=grid, viscosity=nu,
                                                  density=rho))


def test_uniform_flow_has_zero_force():
    """d'Alembert: a uniform stream exerts no net force on a body (budget = 0)."""
    grid = circular_cylinder(8, diameter=1.0, upstream_lengths=4.0,
                             downstream_lengths=6.0, half_height=4.0)
    u = torch.ones((grid.ny, grid.nx + 1), dtype=torch.float64) * 1.3
    v = torch.zeros((grid.ny + 1, grid.nx), dtype=torch.float64)
    p = torch.zeros((grid.ny, grid.nx), dtype=torch.float64)
    flow = _stub_flow(grid, u, v, p)
    box = control_box(grid, x_min=-2.0, x_max=3.0, y_min=-1.5, y_max=1.5)
    terms = budget_terms(flow, box)
    for key, value in terms.items():
        assert abs(value) < 1e-10, (key, value)


def test_control_box_rejects_a_box_that_clips_the_body():
    grid = circular_cylinder(8, diameter=1.0)
    with pytest.raises(ValueError):
        control_box(grid, x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2)


def test_cylinder_mask_area_converges_to_the_circle():
    errors = {cpd: abs(cylinder_mask_area_error(circular_cylinder(cpd)))
              for cpd in (8, 16, 32)}
    assert errors[32] < 0.02, errors
    assert errors[32] <= errors[8], errors


def test_force_is_control_volume_independent():
    """Two nested control volumes around the same body must agree.

    The momentum budget ``F = -d/dt int rho u - oint [rho u (u.n) + p n - tau n]``
    holds for *any* box that contains the body, so disagreement would mean a
    wrong flux, a wrong normal or a wrong sign.  The residual difference is the
    discretisation error of the two surface integrals.
    """
    spec = case.ImmersedSpec(cells_per_diameter=8, dt=0.02, t_end=2.4,
                             upstream_lengths=6.0, downstream_lengths=10.0,
                             half_height=6.0)
    cfg, flow, recorder, _box = case.build_immersed(spec)
    for _ in range(120):
        flow.step()
        recorder.sample(flow.ledger.time)
    small = control_box(cfg.grid, x_min=-1.5, x_max=2.0, y_min=-1.2, y_max=1.2)
    large = control_box(cfg.grid, x_min=-2.5, x_max=3.5, y_min=-2.0, y_max=2.0)
    rec_small = ForceRecorder(flow, small)
    rec_large = ForceRecorder(flow, large)
    # The unsteady term needs two consecutive samples at the same instants.
    rec_small.sample(flow.ledger.time)
    rec_large.sample(flow.ledger.time)
    flow.step()
    sample_small = rec_small.sample(flow.ledger.time)
    sample_large = rec_large.sample(flow.ledger.time)
    assert sample_small.fx > 0.0, "drag must be positive on an accelerating start-up"
    rel = abs(sample_small.fx - sample_large.fx) / abs(sample_small.fx)
    assert rel < 0.10, (sample_small.fx, sample_large.fx, rel)
    rel_y = abs(sample_small.fy - sample_large.fy) / max(abs(sample_small.fx), 1e-9)
    assert rel_y < 0.10, (sample_small.fy, sample_large.fy, rel_y)


def _synthetic_history(st=0.164, amp=0.32, periods=12.0, n=4000, noise=0.0,
                       cd=1.34, cl_floor=0.0):
    dt = periods / st / n
    t = np.arange(n) * dt
    phase = 2 * np.pi * st * t
    cl = cl_floor + amp * np.sin(phase)
    if noise:
        rng = np.random.default_rng(0)
        cl = cl + noise * rng.standard_normal(n)
    return t, np.full(n, cd), cl


def test_statistics_recover_a_known_strouhal_number():
    t, cd, cl = _synthetic_history()
    stats = case.shedding_statistics(t, cd, cl)
    assert stats["shedding_detected"] is True
    assert stats["st"] == pytest.approx(0.164, rel=1e-3)
    assert stats["st_fft"] == pytest.approx(0.164, rel=2e-2)
    assert stats["cd_mean"] == pytest.approx(1.34, rel=1e-9)
    assert stats["cl_amplitude"] == pytest.approx(0.32, rel=2e-2)
    assert stats["window_ok"] is True
    assert not stats["gate_reasons"]


def test_statistics_survive_noise():
    t, cd, cl = _synthetic_history(noise=0.02)
    stats = case.shedding_statistics(t, cd, cl)
    assert stats["st"] == pytest.approx(0.164, rel=5e-3)
    assert stats["freq_cross_check"] < case.CROSS_CHECK_TOL


def test_statistics_withhold_st_when_nothing_sheds():
    t, cd, cl = _synthetic_history(cl_floor=0.0, amp=1e-7, noise=1e-9)
    stats = case.shedding_statistics(t, cd, cl)
    assert stats["shedding_detected"] is False
    assert stats["st"] is None and stats["st_fft"] is None
    assert stats["cl_amplitude"] < case.CL_AMPLITUDE_GATE
    assert stats["gate_reasons"]
    errors = case.compute_errors(dict(stats))
    assert errors["st"]["computed"] is None
    assert errors["gates"]["passed"] is False
    assert case.verdict(errors)[0] is False


def test_statistics_flag_a_window_with_too_few_periods():
    t, cd, cl = _synthetic_history(periods=3.0)
    stats = case.shedding_statistics(t, cd, cl)
    assert stats["window_ok"] is False
    assert any("periods" in reason for reason in stats["gate_reasons"])


def test_reference_file_matches_module_constants():
    reference = json.loads(REFERENCE_FILE.read_text())
    for key, value in case.REFERENCE.items():
        assert reference["values"][key] == pytest.approx(value)
    for key, value in case.REFERENCE_BAND.items():
        assert reference["tolerance_band"][key] == pytest.approx(value)
    assert case.load_reference()["file"].endswith("reference.json")


def test_publish_emits_a_referee_payload(tmp_path):
    t, cd, cl = _synthetic_history()
    stats = case.shedding_statistics(t, cd, cl)
    result = dict(stats)
    result.update(backend="immersed", force_history=[
        {"time": float(ti), "drag": float(ci), "lift": float(li)}
        for ti, ci, li in zip(t, cd, cl)])
    errors = case.compute_errors(result)
    passed = case.publish(tmp_path, result, errors)
    assert passed is True
    payload = json.loads((tmp_path / "benchmark.json").read_text())
    assert payload["anchor_slug"] == "GAP-DIFFLBM-EXT-LIT-CYL"
    assert payload["reference"]["values"]["cd_mean"] == pytest.approx(1.34)
    assert payload["got"]["cd_mean"] == pytest.approx(1.34, rel=1e-9)
    assert payload["got"]["st"] == pytest.approx(0.164, rel=1e-3)
    assert payload["passed"] is True
    assert set(payload["errors"]) == {"cd_mean", "st"}
    for name in ("result.json", "errors.json", "force_history.csv", "report.md",
                 "force_time_series.png", "lift_spectrum.png", "error_bars.png",
                 "phase_plot.png"):
        assert (tmp_path / name).is_file(), name