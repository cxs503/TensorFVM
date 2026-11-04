"""Physical-accuracy qualification of the periodic MAC kernel on the exact
two-dimensional Taylor-Green vortex.

The reference solution is verified independently here (PDE residual by finite
differences), then used to score the merged ``periodic_mac`` kernel on velocity,
pressure, vorticity and the energy/dissipation closure.  The previously
published evidence only qualified refinement order, which is why
``docs/periodic-mac/study.json`` reports ``physical_accuracy_qualified = False``.
"""
import functools
import json
import math
from pathlib import Path

import pytest
import torch

from tensorfvm.periodic_mac import PeriodicMACConfig, PeriodicMACSolver
from tensorfvm.verification import periodic_tgv as tgv

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'docs/periodic-tgv/study.json'


def build(n=16, viscosity=.05, time_step=.02):
    return PeriodicMACSolver(PeriodicMACConfig(nx=n, ny=n, nz=4,
                                               viscosity_pa_s=viscosity,
                                               time_step_s=time_step))


def _field(x, y, t, nu):
    """Closed-form velocity/pressure of the case, evaluated anywhere."""
    decay, decay2 = math.exp(-2 * nu * t), math.exp(-4 * nu * t)
    u = torch.sin(x) * torch.cos(y) * decay
    v = -torch.cos(x) * torch.sin(y) * decay
    p = .25 * (torch.cos(2 * x) + torch.cos(2 * y)) * decay2
    return u, v, p


def _derivative(component, axis, x, y, t, nu, step=1e-4):
    shift = [0., 0.]
    shift[axis] = step
    plus = _field(x + shift[0], y + shift[1], t, nu)[component]
    minus = _field(x - shift[0], y - shift[1], t, nu)[component]
    return (plus - minus) / (2 * step)


def _time_derivative(component, x, y, t, nu, step=1e-4):
    return (_field(x, y, t + step, nu)[component]
            - _field(x, y, t - step, nu)[component]) / (2 * step)


def _laplacian(component, x, y, t, nu, step=1e-4):
    total = 0.
    for axis in (0, 1):
        shift = [0., 0.]
        shift[axis] = step
        total = total + (_field(x + shift[0], y + shift[1], t, nu)[component]
                         - 2 * _field(x, y, t, nu)[component]
                         + _field(x - shift[0], y - shift[1], t, nu)[component]) / step ** 2
    return total


def test_reference_solution_satisfies_continuous_navier_stokes():
    """Independent check that the reference is a real unforced NS solution."""
    generator = torch.Generator().manual_seed(17)
    nu, t = .05, .37
    x = torch.rand(16, generator=generator, dtype=torch.float64) * 2 * math.pi + .11
    y = torch.rand(16, generator=generator, dtype=torch.float64) * 2 * math.pi + .23
    u0, v0, _ = _field(x, y, t, nu)
    assert float(u0.abs().max()) > .2                       # field is not trivial
    assert float((_derivative(0, 0, x, y, t, nu)
                  + _derivative(1, 1, x, y, t, nu)).abs().max()) < 1e-10  # incompressible
    residual_u = (_time_derivative(0, x, y, t, nu)            # du/dt
                  + u0 * _derivative(0, 0, x, y, t, nu)       # u du/dx
                  + v0 * _derivative(0, 1, x, y, t, nu)       # v du/dy
                  + _derivative(2, 0, x, y, t, nu)            # dp/dx   (rho = 1)
                  - nu * _laplacian(0, x, y, t, nu))
    residual_v = (_time_derivative(1, x, y, t, nu)
                  + u0 * _derivative(1, 0, x, y, t, nu)
                  + v0 * _derivative(1, 1, x, y, t, nu)
                  + _derivative(2, 1, x, y, t, nu)
                  - nu * _laplacian(1, x, y, t, nu))
    assert float(residual_u.abs().max()) < 3e-6
    assert float(residual_v.abs().max()) < 3e-6


def test_energy_reference_matches_box_volume_and_converges():
    """The analytic energy must use the box volume, not one cell volume."""
    errors = []
    for n in (8, 16):
        solver = build(n)
        reference = tgv.TaylorGreenReference(solver)
        analytic = reference.analytic_kinetic_energy(0.)
        assert abs(analytic - solver.config.density_kg_m3 * (2 * math.pi) ** 3 / 4) < 1e-9 * analytic
        errors.append(abs(reference.discrete_kinetic_energy(0.) - analytic) / analytic)
    assert errors[0] < 1e-4 and errors[1] < errors[0]


def test_sampled_reference_is_discretely_divergence_free():
    for n in (8, 16, 32):
        solver = build(n)
        divergence = solver.divergence(tgv.exact_face_velocity(solver, 0.)).abs().max()
        assert float(divergence) < 1e-13


@functools.lru_cache(maxsize=None)
def case(n):
    steps = 50 if n == 16 else 100
    return tgv.run_case(n, viscosity_pa_s=.05, time_end_s=1., steps_per_unit_time=steps)


def test_merged_kernel_matches_closed_form_fields_within_three_percent():
    result = case(32)
    assert result.nonlinear_all_converged
    assert result.max_abs_divergence_s_inv < 1e-12
    assert result.velocity_error < .03
    assert result.pressure_error < .03
    assert result.vorticity_error < .03
    assert result.energy_error < .03
    assert result.dissipation_error < .03
    assert result.viscous_power_error < .03
    assert result.initial_energy_error < 1e-9


def test_observed_spatial_order_is_second_order():
    coarse, fine = case(16), case(32)
    ratio = fine.n / coarse.n
    for key in ('velocity_error', 'pressure_error', 'vorticity_error'):
        order = math.log(getattr(coarse, key) / getattr(fine, key)) / math.log(ratio)
        assert 1.8 < order < 2.2, (key, order)


def test_convective_work_vanishes_while_viscous_work_closes():
    result = case(32)
    assert result.max_abs_convective_power_W < 1e-10 * result.power_scale_W
    last = result.trace[-1]
    assert last['viscous_power_numeric_W'] < 0
    assert abs(-last['viscous_power_numeric_W']
               - last['dissipation_exact_W']) / last['dissipation_exact_W'] < .03
    energies = [row['energy_numeric_J'] for row in result.trace]
    assert all(b < a for a, b in zip(energies, energies[1:]))


def test_published_study_gates_are_consistent():
    study = json.loads(STUDY.read_text())
    assert study['schema'] == 'tensorfvm.periodic-tgv-study/1'
    assert study['physical_accuracy_qualified'] is True
    assert study['observed_order_passed'] is True
    assert study['convective_budget_passed'] is True
    assert study['nonlinear_convergence_passed'] is True
    finest = study['refinement'][-1]
    for key in ('velocity_3_percent_passed', 'pressure_3_percent_passed',
                'vorticity_3_percent_passed', 'energy_3_percent_passed',
                'dissipation_3_percent_passed', 'viscous_power_3_percent_passed'):
        assert finest[key] is True, key
    # The gate must be able to fail: the coarsest published grid does.
    assert study['refinement'][0]['pressure_3_percent_passed'] is False
    # Honest limit: the published record states whether the high-Re probe failed.
    limit = study['limit_case']
    assert limit['expected_failure_observed'] == (not limit['all_quantities_passed'])
    assert study['negative_control_qualified'] == limit['expected_failure_observed']
    assert study['distributed_supported'] is False


def test_study_numbers_are_reproducible_from_the_module():
    study = json.loads(STUDY.read_text())
    smallest = study['refinement'][0]
    rerun = case(smallest['n'])
    assert smallest['n'] == rerun.n
    assert smallest['velocity_error'] == pytest.approx(rerun.velocity_error, rel=1e-9)
    assert smallest['pressure_error'] == pytest.approx(rerun.pressure_error, rel=1e-9)

def test_negative_control_is_qualified():
    """The negative control must fail on at least one physical quantity.
    
    This test verifies that the verification gate is discriminating: a challenging
    case (high Re on coarse grid) must fail on pressure or vorticity, proving the
    3% gate is not vacuous. Velocity alone is nearly exact on the staggered grid,
    so we check multiple quantities.
    """
    # Use a quick study with coarse grids to test the negative control
    study = tgv.run_study(
        grids=(8, 16),
        viscosity_pa_s=0.05,
        time_end_s=0.5,
        base_steps_per_unit_time=50,
        limit_viscosity_pa_s=0.002,
        limit_steps_per_unit_time=100,
    )
    
    # The negative control must be qualified (i.e., the limit case must fail)
    assert study['negative_control_qualified'] is True, (
        "Negative control not qualified: the limit case passed all gates, "
        "making the verification vacuous"
    )
    
    # The limit case should fail on pressure or vorticity (not just velocity)
    limit = study['limit_case']
    assert limit['all_quantities_passed'] is False, (
        "Limit case passed all quantities; negative control is not discriminating"
    )
    
    # At least one of pressure/vorticity should fail
    pressure_failed = not limit['pressure_3_percent_passed']
    vorticity_failed = not limit['vorticity_3_percent_passed']
    assert pressure_failed or vorticity_failed, (
        "Neither pressure nor vorticity failed in the limit case; "
        "the negative control is not challenging enough"
    )
    
    # The expected failure should be observed
    assert limit['expected_failure_observed'] is True
