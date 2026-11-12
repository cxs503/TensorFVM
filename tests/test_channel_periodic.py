"""Periodic plane-channel kernel: operators, exact solution, wall treatments, wiring.

Covers ``src/tensorfvm/channel_periodic.py`` and
``src/tensorfvm/benchmark_channel_periodic.py`` as well as the suite caller
``tensorfvm.verification.cases.channel_periodic``.
"""
import json

import pytest
import torch

from tensorfvm.benchmark_channel_periodic import (
    main as benchmark_main,
    run_case,
    run_perturbation_case,
    summarise,
)
from tensorfvm.channel_periodic import (
    PeriodicChannelConfig,
    PeriodicChannelSolver,
    analytic_reference,
    profile_at,
    symbolic_checks,
)


def _metric(case, name):
    return next(m for m in case["metrics"] if m["name"] == name)


def test_symbolic_checks_reproduce_every_quoted_target():
    checks = symbolic_checks(PeriodicChannelConfig())
    assert checks, "symbolic_checks must return its findings"
    failed = {k: v for k, v in checks.items() if not v}
    assert not failed, f"symbolic derivation disagrees with the quoted target: {failed}"


def test_analytic_reference_is_the_plane_channel_relation():
    config = PeriodicChannelConfig(nx=8, ny=8).with_re_h(100.0)
    ref = analytic_reference(config)
    assert ref["bulk_velocity_m_s"] == pytest.approx(1.0, rel=1e-12)
    assert ref["peak_over_bulk"] == pytest.approx(1.5, rel=1e-12)
    assert ref["tau_w_pa"] == pytest.approx(0.06, rel=1e-12)
    assert ref["dpdx_pa_m"] == pytest.approx(-0.12, rel=1e-12)
    assert ref["poiseuille"] == pytest.approx(96.0, rel=1e-12)
    y = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    assert profile_at(config, y).tolist() == pytest.approx([1.125, 1.5, 1.125], rel=1e-12)


def test_projection_keeps_divergence_at_round_off_with_streamwise_waves():
    """The x-mean pressure block must be solvable, or an x-mean divergence survives."""
    config = PeriodicChannelConfig(nx=16, ny=16, time_step_s=0.02)
    solver = PeriodicChannelSolver(config)
    solver.set_time_step(20.0)
    solver.solve_steady()
    solver.set_time_step(0.02)
    solver.add_streamwise_wave(0.05)
    assert solver.max_divergence() < 1e-12
    for _ in range(3):
        solver.step()
        assert solver.max_divergence() < 1e-12
        assert float(solver.divergence().mean(dim=-1).abs().max()) < 1e-12


def test_steady_solution_reproduces_the_exact_parabola_and_the_momentum_balance():
    case = run_case(ny=16)
    assert case["converged"]
    assert _metric(case, "velocity_l2")["relative_error"] < 2e-2
    assert _metric(case, "bulk_velocity")["relative_error"] < 2e-2
    # the scheme-consistent wall flux closes the discrete momentum balance exactly
    assert _metric(case, "momentum_balance")["relative_error"] < 1e-8
    assert _metric(case, "wall_shear_flux")["relative_error"] < 1e-8
    # the convective term vanishes at the x-uniform solution
    assert _metric(case, "convection_at_solution")["computed"] < 1e-12


def test_quadratic_wall_flux_is_exact_for_a_parabolic_solution():
    case = run_case(ny=16, wall_flux="quadratic")
    assert _metric(case, "velocity_l2")["relative_error"] < 1e-9
    assert _metric(case, "wall_shear_quadratic")["relative_error"] < 1e-9
    assert _metric(case, "momentum_balance")["relative_error"] < 1e-8


def test_grid_refinement_orders_match_the_discretisation():
    coarse, fine = run_case(ny=8), run_case(ny=16)
    def order(name):
        a = _metric(coarse, name)["relative_error"]
        b = _metric(fine, name)["relative_error"]
        return torch.log(torch.tensor(a / b)) / torch.log(torch.tensor(2.0))
    assert order("velocity_l2") == pytest.approx(2.0, abs=0.05)
    # first-order: the half-cell wall flux leaves an O(dy) slip that the
    # three-point shear fit inherits
    assert order("wall_shear_quadratic") == pytest.approx(1.0, abs=0.05)


def test_steady_state_does_not_depend_on_the_pseudo_time_step():
    coarse = run_case(ny=16, time_step_s=2.0)
    fast = run_case(ny=16, time_step_s=20.0)
    assert fast["steps"] < coarse["steps"]
    assert max(abs(a - b) for a, b in zip(coarse["fields"]["u_m_s"], fast["fields"]["u_m_s"])) < 1e-9


def test_normalised_error_is_reynolds_independent():
    low = run_case(ny=16, re_h=10.0)
    high = run_case(ny=16, re_h=1000.0)
    assert _metric(low, "velocity_l2")["relative_error"] == pytest.approx(
        _metric(high, "velocity_l2")["relative_error"], rel=1e-6)
    assert _metric(high, "bulk_velocity")["relative_error"] < 2e-2


def test_perturbation_decays_and_the_profile_returns_to_the_parabola():
    case = run_perturbation_case(ny=16, nx=16, steps=80, time_step_s=0.02)
    assert case["passed"], case["metrics"]
    amplitude = case["decay"]["amplitude"]
    assert amplitude[-1] < 5e-2 * amplitude[0]


def test_verification_suite_caller_runs_the_case():
    from tensorfvm.verification.cases import channel_periodic
    result = channel_periodic(nx=8, ny=64)
    assert result["case"] == "channel-periodic"
    assert result["passed"], [r for r in result["metrics"] if not r["passed"]]


def test_benchmark_cli_writes_a_complete_artifact_set(tmp_path):
    out = tmp_path / "case"
    assert benchmark_main(["--out", str(out), "--quick"]) in (0, 1)
    for name in ("benchmark.json", "metrics.csv", "profiles.csv", "manifest.json", "REPORT.md"):
        assert (out / name).stat().st_size > 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["repro_cmd"].startswith("PYTHONPATH=src python -m")
    assert all(a["sha256"] for a in manifest["artifacts"])
    summary = json.loads((out / "benchmark.json").read_text())
    assert summary["schema"] == "tensorfvm.case-channel-periodic/1"
    assert summary["symbolic_checks"]
    # the registered plan must be the one that ran
    assert {c["role"] for c in summary["cases"]} >= {"reference", "spatial", "physics"}


def test_summarise_keeps_coarse_grids_out_of_the_gate():
    cases = [run_case(ny=8, gate=False), run_case(ny=16, gate=True)]
    summary = summarise(cases)
    assert summary["all_passed"] == summary["all_gated_passed"]
    assert summary["gated_cases"] == ["spatial:16x16:re100:dt2.0:L2.0:half-cell"]