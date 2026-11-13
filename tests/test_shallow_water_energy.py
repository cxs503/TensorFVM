"""Tests for the shallow-water dam-break *energy budget* verification.

Node: CASE-FVM-DAMBREAK-STOKER, energy leg (see
``docs/verification-shallow-water-energy/REPORT.md`` and
``src/tensorfvm/shallow_water/energy.py``).

The invariants asserted here are the ones a scheme cannot satisfy by accident:

* the closed-form Ritter solution conserves mechanical energy exactly, so the
  exact-energy integral is a *reference*, not a fudge;
* the classical hydraulic-jump head loss ``(h2-h1)^3/(4 h1 h2)`` really is the
  dissipation rate of the Stoker bore — checked against ``dE/dt`` of the
  closed-form solution, so the formula is certified and not inherited;
* the FV scheme conserves mass to machine precision while it *dissipates*
  energy, and the dissipated amount converges to the physical bore loss at
  first order (a scheme that conserved energy across a bore would be wrong).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET, TOLERANCE
from tensorfvm.shallow_water.energy import (
    bore_dissipation_rate,
    certify_bore_dissipation,
    energy_budget,
    energy_convergence_study,
    exact_energy,
    total_energy,
)
from tensorfvm.shallow_water.exact import stoker_constant_state, t_star_to_time


# --------------------------------------------------------------------------- #
# closed-form reference leg
# --------------------------------------------------------------------------- #
def test_ritter_exact_energy_is_time_independent():
    """No shock in the dry-bed solution => E(t) is constant to quadrature error."""
    case = RITTER_DRY
    values = [exact_energy(case, t_star_to_time(ts, case.h0, case.gravity),
                           n_quad=40_000)["total"]
              for ts in (0.0, 0.5, 1.0, 1.5, 2.0)]
    e0 = values[0]
    drift = max(abs(v - e0) / abs(e0) for v in values)
    assert drift < 1e-8, f"Ritter exact energy drifted by {drift:.2e}"


def test_exact_initial_energy_matches_quadrature():
    """E(0) of both cases equals the analytic rectangular-block value."""
    g = RITTER_DRY.gravity
    left = abs(RITTER_DRY.x_dam - RITTER_DRY.x_min)
    right = abs(RITTER_DRY.x_max - RITTER_DRY.x_dam)
    e0_dry = exact_energy(RITTER_DRY, 0.0, n_quad=20_000)["total"]
    expected_dry = 0.5 * g * RITTER_DRY.h0 ** 2 * left
    assert abs(e0_dry - expected_dry) / expected_dry < 1e-9

    e0_wet = exact_energy(STOKER_WET, 0.0, n_quad=20_000)["total"]
    expected_wet = 0.5 * g * (STOKER_WET.h0 ** 2 * left + STOKER_WET.h1 ** 2 * right)
    assert abs(e0_wet - expected_wet) / expected_wet < 1e-9


def test_bore_head_loss_is_the_classical_expression():
    """``D / (g q_rel)`` must equal ``(h2-h1)^3/(4 h1 h2)`` (Chow §3-4)."""
    case = STOKER_WET
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    h1, h2 = case.h1, sol["h2"]
    q_rel = h1 * sol["shock_speed"]
    expected = (h2 - h1) ** 3 / (4.0 * h1 * h2)
    got = bore_dissipation_rate(case) / (case.gravity * q_rel)
    assert abs(got - expected) / expected < 1e-14


def test_bore_dissipation_formula_certified_against_closed_form():
    """``dE/dt`` of the exact Stoker solution equals the formula to <1e-6.

    The three step sizes give the *same* derivative (spread < 1e-6), which is the
    stronger statement: ``E(t)`` of the closed form is exactly linear because the
    bore states are time independent, so a finite difference has no truncation
    term to hide a wrong formula behind.
    """
    cert = certify_bore_dissipation(STOKER_WET, n_quad=40_000)
    assert cert["worst_relative_error"] < 1e-6, cert["rows"]
    derivatives = [r["dE_dt_measured"] for r in cert["rows"]]
    spread = (max(derivatives) - min(derivatives)) / abs(derivatives[0])
    assert spread < 1e-6, derivatives
    # the measured derivative must be a *loss* (dissipation), not a gain
    assert cert["rows"][0]["dE_dt_measured"] < 0.0


def test_bore_dissipation_rejects_dry_case():
    with pytest.raises(ValueError):
        bore_dissipation_rate(RITTER_DRY)


# --------------------------------------------------------------------------- #
# numerical leg
# --------------------------------------------------------------------------- #
def test_mass_conserved_but_energy_dissipated():
    """The two conservation legs are independent: mass exact, energy not."""
    for case in (RITTER_DRY, STOKER_WET):
        budget = energy_budget(case, n=400, samples=1, n_quad=40_000)
        assert budget["mass_drift"] < 1e-12, (case.key, budget["mass_drift"])
        assert 1e-4 < budget["scheme_dissipation"] < 0.02, (case.key, budget)


def test_no_energy_is_created():
    """A TVD scheme may only lose energy (no anti-dissipation)."""
    for case in (RITTER_DRY, STOKER_WET):
        budget = energy_budget(case, n=400, samples=1, n_quad=40_000)
        assert budget["energy_numeric"] <= budget["energy_initial"], (case.key, budget)


def test_headline_energy_error_within_tolerance():
    """Energy error at n=400 is far inside the task's 5 % acceptance line."""
    for case in (RITTER_DRY, STOKER_WET):
        budget = energy_budget(case, n=400, samples=1, n_quad=40_000)
        assert budget["energy_error"] < 0.005, (case.key, budget)
        assert budget["energy_error"] < TOLERANCE


def test_wet_scheme_reproduces_physical_bore_loss():
    """At n=1600 the FVM burnt energy must match the bore loss to <1 % of E0."""
    budget = energy_budget(STOKER_WET, n=1600, samples=1, n_quad=40_000)
    excess = budget["scheme_dissipation"] - budget["exact_dissipation_fraction"]
    assert 0.0 < excess < 0.01, budget
    # and the physical sink is at the level the closed form predicts
    assert abs(budget["exact_dissipation_fraction"] - 0.0027601) < 1e-6, budget


def test_energy_error_is_first_order():
    """Energy error and excess dissipation converge at ~first order (TVD/shock)."""
    for case, key in ((RITTER_DRY, "energy_error"), (STOKER_WET, "energy_error")):
        study = energy_convergence_study(case, resolutions=(200, 400, 800),
                                         n_quad=40_000)
        orders = [o for o in study["observed_order"] if math.isfinite(o)]
        assert orders, study
        for o in orders:
            assert 0.7 < o < 1.4, (case.key, orders)
        # monotone decrease in the error itself
        errs = [r["energy_error"] for r in study["rows"]]
        assert errs[0] > errs[1] > errs[2], (case.key, errs)


def test_excess_dissipation_decays_with_refinement():
    """Wet case: the *extra* (numerical) burning above the physical bore loss decays."""
    study = energy_convergence_study(STOKER_WET, resolutions=(200, 400, 800),
                                     n_quad=40_000)
    excess = [abs(r["excess_dissipation"]) for r in study["rows"]]
    assert excess[0] > 1.6 * excess[1] > 2.5 * excess[2], excess
    orders = [o for o in study["observed_order_excess"] if math.isfinite(o)]
    for o in orders:
        assert 0.6 < o < 1.5, orders


def test_kinetic_fraction_is_physical():
    """Sanity: the release converts only a few per cent into kinetic energy."""
    for case in (RITTER_DRY, STOKER_WET):
        budget = energy_budget(case, n=400, samples=1, n_quad=40_000)
        frac = budget["kinetic_fraction_numeric"]
        assert 0.0 < frac < 0.30, (case.key, frac)


def test_total_energy_helper_matches_manual_integral():
    h = np.array([1.0, 2.0, 0.5])
    hu = np.array([0.0, 2.0, 0.0])
    g, dx = 9.81, 0.25
    manual = float(((0.5 * g * h ** 2 + 0.5 * hu ** 2 / h) * dx).sum())
    got = total_energy(h, hu, g, dx)["total"]
    assert abs(got - manual) < 1e-12