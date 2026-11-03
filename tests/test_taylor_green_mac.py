"""Taylor--Green vortex: does the exact reference really solve Navier--Stokes?

Two layers are tested here.

1. *The reference itself.*  A verification case is only as trustworthy as its
   closed form, so the analytic velocity/pressure are pushed through the
   momentum and continuity equations symbolically (autograd) before they are
   used to judge the solver.
2. *The solver against the reference.*  Discrete divergence-free sampling, the
   grid-convergence ladder, and -- most importantly -- the discrimination
   controls that show which gate can see which broken ingredient.
"""
import json
import math
from pathlib import Path

import pytest
import torch

from tensorfvm.periodic_mac import PeriodicMACConfig, PeriodicMACSolver
from tensorfvm.verification.taylor_green_mac import (
    TaylorGreenCase,
    compare,
    energy_decay_rate_error,
    exact_fields,
    initialise,
    relative_l2,
    run_to,
)

BOX = 2 * math.pi
STUDY = Path(__file__).resolve().parents[1] / "scripts" / "verification-taylor-green-mac" / "study.json"


def build(nx=8, ny=None, nz=4, dt=0.02, viscosity=0.05):
    return PeriodicMACSolver(PeriodicMACConfig(
        nx=nx, ny=ny or nx, nz=nz, lengths_m=(BOX, BOX, BOX),
        viscosity_pa_s=viscosity, time_step_s=dt,
        nonlinear_tolerance=1e-12, nonlinear_iterations=100))


class NoConvection(PeriodicMACSolver):
    """Control kernel: the nonlinear momentum flux is switched off."""

    def convective_fluxes(self, velocity):
        return torch.zeros((3, 3, *self.shape), dtype=velocity.dtype, device=velocity.device)


# --------------------------------------------------------------------------- #
# 1. the reference verifies itself
# --------------------------------------------------------------------------- #
def test_analytic_fields_are_an_exact_navier_stokes_solution():
    """Autograd the closed form: momentum and continuity residuals must vanish."""
    case = TaylorGreenCase(amplitude=1.3, viscosity=0.07)
    for x0, y0, t0 in ((0.3, 1.1, 0.0), (2.0, 0.4, 0.75), (5.1, 3.3, 1.9)):
        x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(y0, dtype=torch.float64, requires_grad=True)
        t = torch.tensor(t0, dtype=torch.float64, requires_grad=True)
        decay_u = torch.exp(-case.decay_exponent * t)
        decay_p = torch.exp(-case.energy_decay_exponent * t)
        u = case.amplitude * torch.cos(x) * torch.sin(y) * decay_u
        v = -case.amplitude * torch.sin(x) * torch.cos(y) * decay_u
        p = -case.amplitude**2 / 4 * (torch.cos(2 * x) + torch.cos(2 * y)) * decay_p

        def d(field, var):
            return torch.autograd.grad(field, var, create_graph=True)[0]

        continuity = d(u, x) + d(v, y)
        momentum_x = d(u, t) + u * d(u, x) + v * d(u, y) + d(p, x) \
            - case.viscosity * (d(d(u, x), x) + d(d(u, y), y))
        momentum_y = d(v, t) + u * d(v, x) + v * d(v, y) + d(p, y) \
            - case.viscosity * (d(d(v, x), x) + d(d(v, y), y))
        assert abs(float(continuity)) < 1e-13
        assert abs(float(momentum_x)) < 1e-13
        assert abs(float(momentum_y)) < 1e-13


def test_sampled_field_is_discretely_divergence_free_only_when_dx_equals_dy():
    """The face samples cancel *identically* for dx == dy, and only then."""
    case = TaylorGreenCase()
    square = build()
    init = initialise(square, case)
    assert init["spacing_equal_xy"]
    assert init["sampled_max_divergence_s_inv"] < 1e-13

    stretched = build(nx=8, ny=12)
    initialise(stretched, case)
    assert float(stretched.divergence(stretched.velocity).abs().max()) > 1e-2


def test_sampled_kinetic_energy_matches_the_closed_form():
    case = TaylorGreenCase(amplitude=1.1)
    s = build(nx=16)
    init = initialise(s, case)
    assert init["energy_sampling_relative_error"] < 1e-12
    exact = exact_fields(s, case, 0.0)
    assert init["kinetic_energy_J"] == pytest.approx(exact["kinetic_energy_J"], rel=1e-12)


def test_case_rejects_unscaled_density_and_invalid_parameters():
    with pytest.raises(ValueError):
        TaylorGreenCase(density=2.0)
    with pytest.raises(ValueError):
        TaylorGreenCase(viscosity=0.0)


def test_relative_l2_refuses_a_zero_reference():
    with pytest.raises(ValueError):
        relative_l2(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))


def test_run_to_requires_whole_steps():
    s = build(nx=8, dt=0.03)
    initialise(s, TaylorGreenCase())
    with pytest.raises(ValueError):
        run_to(s, 0.5)


# --------------------------------------------------------------------------- #
# 2. the solver against the reference
# --------------------------------------------------------------------------- #
def test_second_order_grid_convergence_and_three_percent_accuracy():
    case = TaylorGreenCase()
    coarse, fine = build(nx=8), build(nx=16)
    results = []
    for s in (coarse, fine):
        init = initialise(s, case)
        run_to(s, 1.0)
        results.append(compare(s, case, init["kinetic_energy_J"]))
    coarse_err, fine_err = (r["velocity_relative_l2"] for r in results)
    observed = math.log(coarse_err / fine_err, 2)
    assert 1.8 < observed < 2.2, observed
    assert fine_err < 0.03
    for r in results:
        assert r["max_divergence_s_inv"] < 1e-12
        assert r["velocity_z_max_abs_m_s"] == 0.0
    # The 3% decay-rate gate is a *resolution-qualified* claim: on n=8 it fails
    # honestly (5.1%), so it is only asserted where the spatial error supports it.
    coarse_rate = results[0]["energy"]["decay_rate_relative_error"]
    fine_rate = results[1]["energy"]["decay_rate_relative_error"]
    assert coarse_rate > fine_rate
    assert fine_rate < 0.03
    # pressure converges too, but from a larger constant: 3.5% at n=16.
    assert results[1]["pressure_relative_l2"] < results[0]["pressure_relative_l2"] / 3


def test_no_convection_control_is_blind_to_velocity_but_blind_to_nothing_on_pressure():
    """The decisive gate-informativeness test.

    ``(u.grad)u`` of the Taylor--Green field is a pure gradient, so the exact
    solution of the *inviscid-free* (pure diffusion) problem is the same
    velocity field.  Dropping convection therefore leaves the velocity gate
    untouched while the pressure collapses to zero -- and only the pressure
    comparison notices.
    """
    case = TaylorGreenCase()
    honest, broken = build(nx=16), NoConvection(build(nx=16).config)
    errors = {}
    for label, s in (("honest", honest), ("broken", broken)):
        init = initialise(s, case)
        run_to(s, 1.0)
        errors[label] = compare(s, case, init["kinetic_energy_J"])
    assert errors["broken"]["velocity_relative_l2"] == pytest.approx(
        errors["honest"]["velocity_relative_l2"], rel=1e-6)
    assert errors["broken"]["velocity_relative_l2"] < 0.03
    assert errors["broken"]["pressure_relative_l2"] > 0.99
    assert float(broken.pressure.abs().max()) < 1e-12
    assert errors["honest"]["pressure_relative_l2"] < 0.05


def test_energy_decay_gate_sees_a_viscosity_error_the_velocity_gate_misses():
    case = TaylorGreenCase()
    s = build(nx=16)
    init = initialise(s, case)
    run_to(s, 1.0)
    wrong = TaylorGreenCase(viscosity=1.2 * case.viscosity)
    # Judged against a reference whose viscosity is 20% off:
    assert compare(s, wrong, init["kinetic_energy_J"])["velocity_relative_l2"] < 0.03
    decay = energy_decay_rate_error(s, wrong, init["kinetic_energy_J"])
    assert decay["decay_rate_relative_error"] > 0.03
    # ...while the honest reference passes both.  (The rate error is the energy
    # error divided by 4*nu*t = 0.2, so it amplifies by five over a t=1 s run.)
    assert energy_decay_rate_error(s, case, init["kinetic_energy_J"])["decay_rate_relative_error"] < 0.02


# --------------------------------------------------------------------------- #
# 3. the published study
# --------------------------------------------------------------------------- #
def test_published_study_qualifies_physical_accuracy_against_exact_solution():
    if not STUDY.exists():  # pragma: no cover - sparse checkout
        pytest.skip(f"published study absent from this checkout: {STUDY}")
    study = json.loads(STUDY.read_text())
    assert study["schema"] == "tensorfvm.taylor-green-mac-study/1"
    summary = study["gate_summary"]
    assert summary["physical_accuracy_qualified"] is True
    gates = study["gates"]
    for name in ("velocity", "pressure", "energy_decay_rate", "energy_at_final_time",
                 "discrete_divergence_free_initial_condition",
                 "observed_order_velocity_in_1p8_2p2", "observed_order_pressure_in_1p8_2p2"):
        assert gates[name]["passed"] is True, name
    assert gates["velocity"]["value"] < gates["pressure"]["value"]
    # Honest negatives are part of the contract, not an embarrassment.
    assert study["temporal_ladder"]["temporal_order_resolved"] is False
    assert summary["informative_gates"]["velocity_gate_blind_to_convection"] is True
    assert summary["informative_gates"]["velocity_gate_blind_to_20_percent_viscosity_error"] is True
    assert summary["informative_gates"]["decay_rate_gate_sees_20_percent_viscosity_error"] is True
    assert study["discrimination"]["no_convection"]["pressure_relative_l2"] > 0.99
    # The ladder must actually converge, not merely pass at the end.
    errors = [row["velocity_relative_l2"] for row in study["spatial_ladder"]]
    assert errors == sorted(errors, reverse=True)
    assert study["spatial_ladder"][-1]["velocity_observed_order"] == pytest.approx(2.0, abs=0.2)