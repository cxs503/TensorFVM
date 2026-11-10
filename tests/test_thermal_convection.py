"""Tests of the thermal-convection verification face.

The three claims that must not silently rot:

* the linear onset solver reproduces the two textbook Rayleigh numbers
  (closed-form stress-free 657.511, rigid 1707.762) and changes sign there,
* the nonlinear layer keeps the discrete conduction state as an equilibrium and
  conserves mass and energy as it convects,
* the power-law fit and the acceptance ledger report what they measure.

Full-resolution sweeps are cluster work (``tensorfvm.thermal.study sweep``); the
tests use small meshes and shortened marches so they stay usable as a gate.
"""

from __future__ import annotations

import json
import math
import os

import pytest

torch = pytest.importorskip("torch")

from tensorfvm.thermal.rayleigh_benard import (  # noqa: E402
    RayleighBenardCase,
    RayleighBenardConfig,
)
from tensorfvm.thermal.reference import (  # noqa: E402
    NU_EXPONENT_CLASSICAL,
    RA_CRITICAL_FREE,
    RA_CRITICAL_RIGID,
)
from tensorfvm.thermal.scaling import (  # noqa: E402
    criterion,
    evaluate_criteria,
    fit_power_law,
)
from tensorfvm.thermal.stability import (  # noqa: E402
    StabilityConfig,
    critical_rayleigh,
    growth_rate,
)


# --------------------------------------------------------------------- onset
def test_stress_free_threshold_matches_closed_form():
    config = StabilityConfig(nodes=32, boundary="free")
    result = critical_rayleigh(0.71, config, k_min=1.5, k_max=3.2, k_points=13,
                              rel_tol=1e-6)
    assert result["rayleigh_critical"] == pytest.approx(RA_CRITICAL_FREE, rel=0.01)
    assert result["wavenumber_critical"] == pytest.approx(2.22144, rel=0.01)


def test_rigid_threshold_matches_textbook_value():
    config = StabilityConfig(nodes=32, boundary="rigid")
    result = critical_rayleigh(0.71, config, k_min=2.3, k_max=4.2, k_points=13,
                              rel_tol=1e-6)
    assert result["rayleigh_critical"] == pytest.approx(RA_CRITICAL_RIGID, rel=0.02)
    assert result["wavenumber_critical"] == pytest.approx(3.11632, rel=0.02)


def test_wall_condition_changes_the_threshold_by_the_textbook_factor():
    """A rigid wall is felt: ``Ra_c`` must move by the textbook factor ~2.6."""
    free = critical_rayleigh(0.71, StabilityConfig(nodes=24, boundary="free"),
                             k_min=1.5, k_max=3.2, k_points=11, rel_tol=1e-5)
    rigid = critical_rayleigh(0.71, StabilityConfig(nodes=24, boundary="rigid"),
                              k_min=2.3, k_max=4.2, k_points=11, rel_tol=1e-5)
    ratio = rigid["rayleigh_critical"] / free["rayleigh_critical"]
    assert ratio == pytest.approx(RA_CRITICAL_RIGID / RA_CRITICAL_FREE, rel=0.03)


def test_growth_rate_changes_sign_at_the_threshold():
    config = StabilityConfig(nodes=32, boundary="rigid")
    k = 3.11632
    below = growth_rate(k, 0.97 * RA_CRITICAL_RIGID, 0.71, config)
    above = growth_rate(k, 1.03 * RA_CRITICAL_RIGID, 0.71, config)
    assert below < 0.0 < above
    assert abs(below) < 1.0 and abs(above) < 1.0      # not a spurious branch


def test_marginal_curve_is_a_minimum_at_the_critical_wavenumber():
    config = StabilityConfig(nodes=24, boundary="rigid")
    result = critical_rayleigh(0.71, config, k_min=2.3, k_max=4.2, k_points=11,
                              rel_tol=1e-5)
    curve = result["curve"]
    index = curve["k"].index(min(curve["k"], key=lambda k: abs(k - result["wavenumber_critical"])))
    neighbours = [curve["rayleigh"][i] for i in (index - 1, index + 1)]
    assert all(value > curve["rayleigh"][index] for value in neighbours)


# ---------------------------------------------------------------- identities
@pytest.mark.parametrize("rayleigh", [1.0e4, 3.0e5])
def test_discrete_identities_reproduce_rayleigh_and_prandtl(rayleigh):
    case = RayleighBenardCase(RayleighBenardConfig(cells=12, aspect=2, rayleigh=rayleigh,
                                                   prandtl=0.71))
    assert case.rayleigh_check() == pytest.approx(rayleigh, rel=1e-12)
    assert case.prandtl_check() == pytest.approx(0.71, rel=1e-12)


def test_plate_temperature_boundary_condition_is_active():
    """The plate rows must impose the Dirichlet values, not the side ones."""
    case = RayleighBenardCase(RayleighBenardConfig(cells=8, aspect=1, rayleigh=1.0e4))
    case.initialise(mode="conduction")
    theta = case.solver.theta
    assert float(theta[:, 0].mean()) > 0.4      # hot cell row next to the lower plate
    assert float(theta[:, -1].mean()) < -0.4    # cold cell row next to the upper plate
    bottom, top = case.nusselt()
    assert float(bottom.mean()) == pytest.approx(1.0, rel=1e-10)
    assert float(top.mean()) == pytest.approx(1.0, rel=1e-10)
    flux = case.flux_profile()
    assert flux["spread"] < 1.0e-10             # conduction: one single flux


def test_conduction_state_is_a_discrete_equilibrium():
    """No perturbation -> no convection, to machine precision, forever."""
    case = RayleighBenardCase(RayleighBenardConfig(cells=8, aspect=1, rayleigh=1.0e5))
    case.initialise(mode="conduction")
    for _ in range(40):
        case.solver.step()
    flow = case.flow_summary()
    assert flow["u_max"] < 1.0e-10 and flow["v_max"] < 1.0e-10
    bottom, top = case.nusselt()
    assert float(bottom.mean()) == pytest.approx(1.0, rel=1e-9)
    assert float(top.mean()) == pytest.approx(1.0, rel=1e-9)


def test_side_walls_stay_adiabatic_after_the_plate_option_was_added():
    """The default (side-heated) cavity must be unchanged by the plates option."""
    from tensorfvm.internal_flow.mac import MACConfig, StaggeredMAC

    solver = StaggeredMAC(MACConfig(nx=8, ny=8, length=1.0, height=1.0, viscosity=1.0e-2,
                                    energy=True, diffusivity=1.0e-2,
                                    temperature_left=0.5, temperature_right=-0.5))
    for _ in range(20):
        solver.step()
    theta = solver.theta
    # conduction across x: a steep wall-normal gradient at the x walls
    wall_flux_x = float((theta[0, :] - theta[1, :]).abs().mean())
    # adiabatic y walls: the two cell rows next to a plate carry the same value
    wall_flux_y = float((theta[:, 0] - theta[:, 1]).abs().mean())
    assert wall_flux_x > 100.0 * max(wall_flux_y, 1e-12)
    assert float(theta[0, :].mean()) > 0.0 > float(theta[-1, :].mean())


# ------------------------------------------------------------- nonlinear runs
def _short_march(cells: int, rayleigh: float, time_budget: float) -> RayleighBenardCase:
    case = RayleighBenardCase(RayleighBenardConfig(cells=cells, aspect=2, rayleigh=rayleigh))
    case.initialise(mode="roll", rolls=1, amplitude=2.0e-3)
    solver = case.solver
    elapsed = 0.0
    while elapsed < time_budget:
        elapsed += solver.step()
    case.elapsed = elapsed
    return case


def test_layer_conserves_mass_and_drives_heat_upward():
    case = _short_march(cells=12, rayleigh=1.0e4, time_budget=45.0)
    flow = case.flow_summary()
    assert flow["divergence_max"] < 1.0e-12
    bottom, top = case.nusselt()
    assert float(bottom.mean()) > 1.2           # convection transports more than conduction
    assert float(top.mean()) > 1.2
    assert float(bottom.mean()) == pytest.approx(float(top.mean()), rel=0.05)
    budget = case.nusselt_budget()
    assert budget["imbalance"] < 1.0e-3         # still converging, must be small
    assert budget["nusselt_bottom"] > 1.2


def test_heat_transport_increases_with_rayleigh():
    quiet = _short_march(cells=12, rayleigh=3.0e3, time_budget=45.0)
    loud = _short_march(cells=12, rayleigh=3.0e4, time_budget=45.0)
    nu_quiet = float(quiet.nusselt()[0].mean())
    nu_loud = float(loud.nusselt()[0].mean())
    assert nu_quiet > 1.0
    assert nu_loud > nu_quiet


def test_energy_imbalance_shrinks_towards_steady_state():
    case = RayleighBenardCase(RayleighBenardConfig(cells=12, aspect=2, rayleigh=1.0e4))
    case.initialise(mode="roll", rolls=1, amplitude=2.0e-3)
    solver = case.solver
    history = []
    elapsed = 0.0
    while elapsed < 90.0:
        elapsed += solver.step()
        if solver.steps % 200 == 0:
            history.append(case.nusselt_budget()["imbalance"])
    assert history[-1] < history[len(history) // 3]
    assert history[-1] < 1.0e-5


# -------------------------------------------------------------------- scaling
def test_power_law_fit_recovers_a_synthetic_exponent():
    rayleigh = [1.0e4, 1.0e5, 1.0e6, 1.0e7]
    nusselt = [0.25 * ra ** NU_EXPONENT_CLASSICAL for ra in rayleigh]
    fit = fit_power_law(rayleigh, nusselt)
    assert fit["exponent"] == pytest.approx(NU_EXPONENT_CLASSICAL, rel=1e-9)
    assert fit["coefficient"] == pytest.approx(0.25, rel=1e-9)
    assert fit["stderr"] < 1e-9
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_power_law_fit_reports_the_error_of_a_noisy_series():
    rayleigh = [1.0e4, 1.0e5, 1.0e6, 1.0e7]
    nusselt = [3.0, 6.0, 12.0, 24.0]               # exponent 0.30103
    fit = fit_power_law(rayleigh, nusselt)
    assert 0.28 < fit["exponent"] < 0.32
    assert math.isfinite(fit["stderr"])


def test_power_law_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_power_law([1.0e4, 1.0e5], [2.0, 3.0])
    with pytest.raises(ValueError):
        fit_power_law([1.0e4, 1.0e5, 1.0e6], [2.0, -3.0, 4.0])


def test_criterion_ledger_marks_failures_and_missing_values():
    rows = [
        criterion("a", "inside tolerance", 1.02, 1.0, 0.05),
        criterion("b", "outside tolerance", 1.20, 1.0, 0.05),
        criterion("c", "missing value", None, 1.0, 0.05),
    ]
    ledger = evaluate_criteria(rows)
    assert ledger["passed_count"] == 1
    assert ledger["all_passed"] is False
    assert rows[1]["error"] == pytest.approx(0.2)


def test_verified_cavity_artifact_still_resolves():
    """Criterion (c) of the campaign is answered from the repository's evidence."""
    path = os.path.join("docs", "benchmarks", "verified", "fvm_natural_convection",
                        "results.json")
    if not os.path.exists(path):
        pytest.skip("verified cavity artifact not checked out")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    finest = [row for row in payload["criteria"]
              if row["id"].startswith("cavity.Nu.") and row["id"].endswith(".finest")]
    assert finest, "the verified cavity artifact has no finest-mesh Nusselt entry"
    for row in finest:
        assert row["error"] is not None and row["error"] <= 0.03
        assert row["passed"] is True