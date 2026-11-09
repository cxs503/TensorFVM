"""Gates for the sudden-expansion momentum closure (:mod:`tensorfvm.internal_flow.expansion_closure`).

The closure makes a strong claim -- that a measured expansion loss equals the
Borda-Carnot value plus four independently measurable terms -- so the tests pin
it from three sides:

* the **algebra** (ideal limit recovers Borda-Carnot exactly; every term enters
  linearly with the coefficient the derivation gives);
* the **discrete momentum balance** (the identity holds on a solved field to the
  solver's steady-state tolerance, and the residual is *reported*, not assumed);
* the **negative controls** (a mesh whose inlet rows are only partially open must
  show the momentum-flux deficit the theory predicts, and an unknown right-hand
  closure must be rejected) -- a metric that cannot report the defect cannot
  certify the fix either.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.internal_flow.expansion_closure import (
    EXIT_CLOSURES,
    ClosureTerms,
    closure_loss,
    closure_terms,
    run_case,
)
from tensorfvm.internal_flow.mac import MACConfig
from tensorfvm.internal_flow.reference import borda_carnot_loss
from tensorfvm.internal_flow.sudden_expansion import SuddenExpansionCase, SuddenExpansionConfig

#: ``A1 / A2`` of the test cases (``AR = 4``); n = 16 makes the inlet opening four
#: whole rows, n = 20 makes its two edge rows half open.
RATIO = 0.25
SHORT = dict(reynolds=100.0, length=8.0, tolerance=1e-6)


def _solve(ratio: float = RATIO, right: str = "outlet", **kwargs) -> tuple[ClosureTerms, object]:
    config = SuddenExpansionConfig(
        area_ratio=1.0 / ratio,
        reynolds=kwargs.pop("reynolds", 100.0),
        cells_per_downstream_height=kwargs.pop("cells", 16),
        length_in_downstream_heights=kwargs.pop("length", 8.0),
        right=right,
        tolerance=kwargs.pop("tolerance", 1e-6),
        max_steps=kwargs.pop("max_steps", 200_000),
    )
    case = SuddenExpansionCase(config)
    ledger = case.run()
    return closure_terms(case), (case, ledger)


# ------------------------------------------------------------------ algebra
@pytest.mark.parametrize("ratio", [1.25, 2.0, 4.0, 10.0])
def test_ideal_limit_is_borda_carnot(ratio: float) -> None:
    """Uniform profiles, no base suction, no friction -> the classical value."""
    assert closure_loss(ratio) == pytest.approx(borda_carnot_loss(ratio), rel=0, abs=1e-15)


@pytest.mark.parametrize("area_ratio", [1.25, 2.0, 4.0, 10.0])
def test_each_term_enters_with_its_derived_coefficient(area_ratio: float) -> None:
    """The closure is linear in every term, with the coefficients of the derivation."""
    base = borda_carnot_loss(area_ratio)
    step = 1e-4
    assert closure_loss(area_ratio, beta_in=1.0 - step) - base == pytest.approx(
        2.0 * step / area_ratio, rel=1e-9)
    assert closure_loss(area_ratio, beta_out=1.0 + step) - base == pytest.approx(
        2.0 * step / area_ratio ** 2, rel=1e-9)
    assert closure_loss(area_ratio, cp_step=step) - base == pytest.approx(
        step * (1.0 - 1.0 / area_ratio), rel=1e-9)
    assert closure_loss(area_ratio, wall_friction=step) - base == pytest.approx(step, rel=1e-12)
    assert closure_loss(area_ratio, outlet_viscous=step) - base == pytest.approx(-step, rel=1e-12)


def test_invalid_area_ratio_is_rejected() -> None:
    for bad in (1.0, 0.5, float("nan")):
        with pytest.raises(ValueError):
            closure_loss(bad)


# ---------------------------------------------------- inlet momentum coefficient
def test_whole_row_inlet_carries_unit_momentum_coefficient() -> None:
    """Four whole inlet rows: the discrete inlet profile is uniform, ``beta_in = 1``."""
    terms, (case, _) = _solve(cells=16)
    assert case.config.inlet_height * case.config.cells_per_downstream_height == pytest.approx(4.0)
    assert terms.beta_in == pytest.approx(1.0, abs=1e-12)
    assert terms.k_inlet == pytest.approx(0.0, abs=1e-12)


def test_half_open_inlet_rows_report_the_predicted_momentum_deficit() -> None:
    """Negative control: two half-open rows move ``beta`` by exactly 10 % at AR = 4."""
    terms, (case, _) = _solve(cells=20)
    rows = case.config.inlet_height * case.config.cells_per_downstream_height
    assert rows == pytest.approx(5.0)
    assert terms.beta_in == pytest.approx(0.9, abs=1e-12)
    assert terms.k_inlet == pytest.approx(2.0 * 0.1 / 4.0, rel=1e-12)


# ------------------------------------------------------------- the solved field
@pytest.fixture(scope="module")
def solved() -> tuple[ClosureTerms, object]:
    return _solve(cells=16)


def test_closure_reproduces_the_measured_loss(solved) -> None:
    """Everything the measurement contains is accounted for by a named term."""
    terms, _ = solved
    assembled = (terms.k_borda_carnot + terms.k_inlet + terms.k_profile + terms.k_base
                 + terms.k_friction)
    assert terms.k_closed == pytest.approx(assembled, rel=0, abs=1e-12)
    assert terms.error_of_closure == pytest.approx(terms.closure_residual / terms.k_measure,
                                                   rel=0, abs=1e-12)
    # the residual is a *reported* discretisation indicator (0.06 % on the meshes of
    # the published sweep), not a fitted tolerance; this test mesh is short and coarse
    assert abs(terms.error_of_closure) < 1.5e-2


def test_discrete_momentum_balance_of_the_control_volume(solved) -> None:
    """The solver's own momentum operator telescopes to the closure's boundary fluxes."""
    terms, (case, ledger) = solved
    assert ledger["converged"]
    assert abs(terms.discrete_residual) < 1e-6
    assert float(case.solver.divergence().abs().max()) < 1e-9


def test_borda_carnot_deviation_is_attributed_term_by_term(solved) -> None:
    """The anchor is not the answer: every bit of the gap has a measured cause."""
    terms, _ = solved
    gap = terms.k_measure - terms.k_borda_carnot
    attributed = terms.k_inlet + terms.k_profile + terms.k_base + terms.k_friction
    assert gap > 0.0, "a separated expansion loses more than the inviscid anchor"
    # k_measure - k_closed = -(control-volume balance residual); see closure_terms
    assert gap - attributed == pytest.approx(-terms.closure_residual, abs=1e-12)
    assert terms.error_vs_borda_carnot < 0.0


# ---------------------------------------------------------------- exit closures
@pytest.fixture(scope="module")
def exit_study() -> dict[str, ClosureTerms]:
    return {name: _solve(right=name, cells=16)[0] for name in EXIT_CLOSURES}


def test_every_outlet_closure_is_mass_consistent(exit_study) -> None:
    for name, terms in exit_study.items():
        assert math.isfinite(terms.k_measure), name
        assert abs(terms.closure_residual) < 1e-2, name


def test_developed_outlet_imposes_the_parabolic_profile() -> None:
    """``u = 6 U eta (1 - eta)`` carries ``beta = 1.2`` in the continuum.

    On a ``ny``-row mesh the sampled parabola carries ``beta = ny sum(u^2) / (sum u)^2``,
    which is 1.1953 at ``ny = 16``: the closure must report the *discrete* value,
    and that value has to sit within 1 % of the analytic 6/5.
    """
    terms, (case, _) = _solve(right="developed", cells=16)
    profile = case.solver.developed_profile()
    assert float(profile.sum()) == pytest.approx(float(case.solver._inlet.sum()), rel=1e-12)
    sampled = (16.0 * float((profile ** 2).sum()) / float(profile.sum()) ** 2)
    assert terms.beta_out == pytest.approx(sampled, rel=1e-9)
    assert terms.beta_out == pytest.approx(1.2, rel=1e-2)
    assert terms.k_profile == pytest.approx(2.0 * (terms.beta_out - 1.0) / 16.0, rel=1e-12)


def test_step_local_loss_is_outlet_insensitive(exit_study) -> None:
    """The loss the *step* injects moves less than 2 % across the outflow closures."""
    values = [terms.k_step for terms in exit_study.values()]
    reference = exit_study["outlet"].k_step
    assert max(abs(value - reference) for value in values) / reference < 0.02


def test_unknown_outlet_closure_is_rejected() -> None:
    with pytest.raises(ValueError):
        MACConfig(nx=8, ny=8, length=1.0, height=1.0, viscosity=0.1, right="sponge")
    with pytest.raises(ValueError):
        SuddenExpansionConfig(right="sponge")
    with pytest.raises(ValueError):
        # a developed profile needs a flux to impose
        MACConfig(nx=8, ny=8, length=1.0, height=1.0, viscosity=0.1, right="developed")


def test_run_case_helper_returns_a_closure() -> None:
    terms, meta = run_case(4.0, 100.0, 8, length=6.0, tolerance=1e-6, max_steps=100_000)
    assert torch.isfinite(torch.tensor(terms.k_measure))
    assert meta["ledger"]["steps"] > 0