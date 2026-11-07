"""Verification-face tests: the artifacts under ``benchmarks/verified``.

These tests do not re-run the solvers (that takes hours); they read the
published ledgers and check that

* every case carries three mesh levels and a converged Richardson order,
* every solver run reports a stationary state with a clean discrete divergence,
* every criterion that fails is accompanied by a written reason (an honest
  FAIL is allowed, an unexplained one is not),
* the reported figures and the report exist and are non-empty.

They also pin the two discretisation fixes that the fine-mesh study depended on
(exact inlet flux of the sudden expansion, second-order wall heat flux in the
cavity), so the earlier defect cannot silently come back.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import torch

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "benchmarks" / "verified"

CASES = {
    "fvm_backward_step": {"levels": 3, "metric": "reattachment"},
    "fvm_sudden_expansion": {"levels": 3, "metric": "loss_coefficient"},
    "fvm_natural_convection": {"levels": 3, "metric": "convergence"},
}


def _summary(case: str) -> dict:
    path = VERIFIED / case / "summary.json"
    if not path.exists():
        pytest.skip(f"{path} has not been produced yet")
    return json.loads(path.read_text())


@pytest.mark.parametrize("case", sorted(CASES))
def test_summary_is_present_and_audited(case: str) -> None:
    """Each case publishes a summary with an audit trail of the runs behind it."""
    summary = _summary(case)
    audit = summary["audit"]
    assert audit["ledgers"], "no raw ledgers recorded"
    assert audit["all_stationary"], "a run did not reach a stationary state"
    assert audit["max_divergence"] < 1e-8
    assert summary["figures"], "no figures published"
    for figure in summary["figures"]:
        path = VERIFIED / case / "figures" / figure
        assert path.exists() and path.stat().st_size > 10_000
    assert (VERIFIED / case / "REPORT.md").exists()
    assert (VERIFIED / case / "summary.json").stat().st_size > 1_000


@pytest.mark.parametrize("case", sorted(CASES))
def test_three_mesh_levels(case: str) -> None:
    """Three levels, or an explicit written statement of the coverage gap.

    The campaign node asks for three meshes per case. A study that could not
    finish one of them must say so in ``fail_reasons`` instead of publishing a
    two-level "convergence study" as if nothing were missing.
    """
    summary = _summary(case)
    reasons = summary.get("fail_reasons") or {}
    families = ([summary["meshes"]] if case != "fvm_natural_convection"
                else list(summary["meshes"].values()))
    short = [len(family) for family in families if len(family) < 3]
    if short:
        assert "third_level" in reasons or "order" in reasons, (
            f"{case} has a level missing and no written reason")


@pytest.mark.parametrize("case", sorted(CASES))
def test_convergence_order(case: str) -> None:
    """A published grid-convergence order must be at least first-and-a-half."""
    summary = _summary(case)
    criteria = summary["criteria"]
    if case == "fvm_backward_step":
        orders = [criteria["reattachment"].get("order")]
    elif case == "fvm_sudden_expansion":
        orders = [criteria["reattachment"].get("order"),
                  criteria["loss_coefficient"].get("order")]
    else:
        orders = [entry["convergence"].get("order")
                  for entry in criteria["per_rayleigh"].values()]
    for order in orders:
        if order is None:
            pytest.skip("order not available for every metric")
        assert order >= 1.5, f"observed order {order} below the 1.5 floor"


@pytest.mark.parametrize("case", sorted(CASES))
def test_failures_are_explained(case: str) -> None:
    """Every criterion that does not pass must carry a written reason."""
    summary = _summary(case)
    verdicts = list(summary["criteria"].values())
    if case == "fvm_natural_convection":
        verdicts += list(summary["criteria"]["per_rayleigh"].values())
    failed = [verdict for verdict in verdicts
              if isinstance(verdict, dict) and verdict.get("pass") is False]
    failures = {key: value for key, value in summary["criteria"].items()
                if key.endswith("_pass") and value is False}
    if case == "fvm_natural_convection":
        failures = {key: value for key, value in summary["criteria"].items()
                    if key.endswith("_pass") and value is False}
    if failed or failures:
        reasons = summary.get("fail_reasons") or {}
        assert reasons, f"{case} reports FAIL verdicts without a written reason"
        for key in reasons:
            assert isinstance(reasons[key], str) and len(reasons[key]) > 40


def test_backward_step_matches_the_reference_band() -> None:
    summary = _summary("fvm_backward_step")
    criteria = summary["criteria"]
    assert criteria["reattachment_relative_error"] < 0.10
    # the campaign asks for the C_f trend (sign inside the bubble, zero crossing
    # at reattachment, positive and recovering downstream), not a magnitude band
    assert criteria["cf_pass"], criteria["cf_trend"]
    # the campaign target is quoted in downstream channel heights
    assert summary["geometry"]["downstream_channel_height"] == 2.0
    assert abs(criteria["reattachment_computed"] - 6.0) <= 0.6


def test_sudden_expansion_is_flux_exact_and_symmetric() -> None:
    """The inlet opening must carry exactly ``U * H_in`` on every mesh."""
    from tensorfvm.internal_flow.sudden_expansion import (
        SuddenExpansionConfig, inlet_profile)

    for n in (20, 30, 40, 80, 160):
        config = SuddenExpansionConfig(cells_per_downstream_height=n)
        profile = inlet_profile(config)
        flux = sum(profile) * config.downstream_height / n
        assert math.isclose(flux, config.inlet_height, rel_tol=1e-12)
        assert max(profile) == pytest.approx(config.inlet_velocity)
        # symmetric about the centre line
        for lower, upper in zip(profile, reversed(profile)):
            assert lower == pytest.approx(upper)

    summary = _summary("fvm_sudden_expansion")
    criteria = summary["criteria"]
    assert criteria["reattachment_symmetry"] < 0.02
    assert criteria["k_borda_carnot"] == pytest.approx(0.5625)


def test_cavity_nusselt_against_de_vahl_davis() -> None:
    summary = _summary("fvm_natural_convection")
    per_rayleigh = summary["criteria"]["per_rayleigh"]
    assert {float(key) for key in per_rayleigh} == {1e4, 1e5, 1e6}
    for rayleigh, entry in per_rayleigh.items():
        assert entry["nu_reference"] in (2.243, 4.519, 8.8)
        # the finest mesh must be inside the de Vahl Davis band
        assert entry["nu_relative_error"] < 0.03, rayleigh
        assert entry["convergence"]["monotone"] is True


def test_second_order_wall_gradient_is_exact_for_a_quadratic() -> None:
    """The Nusselt-number wall flux must be second-order, not half-cell."""
    from tensorfvm.internal_flow.natural_convection import (
        NaturalConvectionCase, NaturalConvectionConfig)

    n = 8
    case = NaturalConvectionCase(NaturalConvectionConfig(n=n, rayleigh=1e4))
    solver = case.solver
    h = float(solver.hx)
    wall = 0.5 * case.config.temperature_drop
    slope, curvature = 1.7, 3.1
    x = ((torch.arange(n, dtype=torch.float64) + 0.5) * h)[:, None]
    # theta varies with x (the wall-normal direction of the hot wall), not y
    solver.theta = (wall - slope * x + curvature * x ** 2).repeat(1, n)
    _, nu = case.nusselt_profile()
    assert torch.allclose(nu, torch.full_like(nu, slope), atol=1e-10)
    first_order = (wall - solver.theta[0, :]) / (0.5 * h)
    assert not torch.allclose(first_order, torch.full_like(first_order, slope), atol=1e-3)


def test_explicit_solver_records_reference_values() -> None:
    """The published references are the ones the campaign quotes."""
    from tensorfvm.internal_flow.reference import (BACKWARD_STEP_REFERENCE,
                                                   DE_VAHL_DAVIS_NU,
                                                   borda_carnot_loss)

    assert borda_carnot_loss(4.0) == pytest.approx(0.5625)
    assert DE_VAHL_DAVIS_NU[1.0e4] == 2.243
    assert DE_VAHL_DAVIS_NU[1.0e5] == 4.519
    assert DE_VAHL_DAVIS_NU[1.0e6] == 8.8
    assert BACKWARD_STEP_REFERENCE["campaign_target"] == 6.0
    assert BACKWARD_STEP_REFERENCE["campaign_band"] == 0.2