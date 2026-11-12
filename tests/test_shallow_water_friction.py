"""Tests for the shallow-water bottom-friction / slope source terms.

The gates here are the cheap subset of
``tensorfvm.verification.shallow_water_friction`` (which carries the full
pre-declared criteria set); they are written to fail if the source terms are
plausible-but-wrong -- a factor ``h``, a missing ``|u|``, a sign, or a silent
change of the inviscid solver.
"""

import numpy as np
import pytest

from tensorfvm.shallow_water import ShallowWater1D
from tensorfvm.verification.shallow_water_friction import (
    decay_coefficient,
    decay_velocity,
    normal_velocity,
    relaxation_velocity,
    uniform_solver,
)

G = 9.81


# ------------------------------------------------------------------ closed form
def test_decay_closed_form_satisfies_its_own_ode():
    """u(t) = u0/(1+C u0 t) must satisfy du/dt = -C u|u| (finite differences)."""
    c = decay_coefficient(0.03, 1.0, G)
    u0 = 1.5
    t = np.linspace(0.0, 50.0, 5001)
    u = decay_velocity(t, u0=u0, manning_n=0.03)
    dudt = np.gradient(u, t)
    residual = dudt + c * u * np.abs(u)
    assert np.max(np.abs(residual)) / np.max(np.abs(c * u * u)) < 1e-3


def test_normal_velocity_matches_manning_formula():
    # u_n = h^(2/3) sqrt(S0)/n ; h=1, S0=1e-3, n=0.03 -> 1.05409 m/s
    assert normal_velocity(1.0, 0.03, 1e-3) == pytest.approx(1.0540925533894598, rel=1e-12)
    # depth exponent must be 2/3 (a linear-in-h bug would give 2.0)
    assert normal_velocity(8.0, 0.03, 1e-3) / normal_velocity(1.0, 0.03, 1e-3) == \
        pytest.approx(4.0, rel=1e-12)


def test_relaxation_closed_form_satisfies_its_own_ode():
    h, n, s0 = 1.0, 0.03, 1e-3
    un = normal_velocity(h, n, s0)
    t = np.linspace(0.0, 300.0, 20001)
    u = relaxation_velocity(t, h, n, s0)
    dudt = np.gradient(u, t)
    residual = dudt - G * s0 * (1.0 - (u / un) ** 2)
    assert np.max(np.abs(residual)) / (G * s0) < 1e-3


# ------------------------------------------------------------- solver: source
def test_friction_only_decay_matches_closed_form():
    solver = uniform_solver(manning_n=0.03)
    solver.advance(60.0)
    u = solver.primitive()[1]
    exact = float(decay_velocity(60.0))
    assert np.max(np.abs(u - exact)) / exact < 1e-3
    # uniform state stays uniform: no spurious wave from the source term
    assert np.max(u) - np.min(u) == pytest.approx(0.0, abs=1e-12)


def test_normal_flow_relaxation_approaches_manning_velocity():
    solver = uniform_solver(u=0.0, manning_n=0.03, slope=1e-3)
    solver.advance(600.0)
    u = solver.primitive()[1]
    un = normal_velocity()
    assert abs(float(u[0]) / un - 1.0) < 1e-3
    # the tanh closed form is the tighter statement
    assert abs(float(u[0]) / float(relaxation_velocity(600.0)) - 1.0) < 1e-3
    # depth must not drift: friction is momentum-only
    assert np.max(np.abs(solver.h - 1.0)) < 1e-12


def test_friction_slope_helper_uses_manning_law():
    solver = uniform_solver(manning_n=0.03)
    sf = solver.friction_slope()
    expected = 0.03 ** 2 * 1.5 * 1.5 / 1.0 ** (4.0 / 3.0)
    assert np.allclose(sf, expected, rtol=1e-12)
    # momentum sink per unit volume is -g h S_f
    sink = solver._source_momentum(solver.h, solver.hu)
    assert np.allclose(sink, -G * 1.0 * expected, rtol=1e-12)


# ------------------------------------------------------- regression/negatives
def test_sources_off_is_bit_identical_to_the_inviscid_residual():
    solver = uniform_solver()                      # no friction, no slope
    hL, uL, hR, uR = solver._face_states()
    fh, fhu = solver._hll_flux(hL, uL, hR, uR)
    ref_dh = -(fh[1:] - fh[:-1]) / solver.dx
    ref_dhu = -(fhu[1:] - fhu[:-1]) / solver.dx
    dh, dhu = solver.rhs(solver.h, solver.hu)
    assert np.array_equal(dh, ref_dh)
    assert np.array_equal(dhu, ref_dhu)


def test_dry_cells_carry_no_friction():
    solver = uniform_solver(n=16, manning_n=0.03)
    h = np.zeros(16)
    h[:8] = 1.0
    solver.set_state(h, np.full(16, 1.0))
    assert np.all(solver.friction_slope()[8:] == 0.0)
    sink = solver._source_momentum(solver.h, solver.hu)
    assert np.all(sink[8:] == 0.0)


def test_stiff_drag_never_reverses_the_flow():
    """Drag impulse larger than the cell momentum must be capped, not applied."""
    solver = ShallowWater1D(16, 0.0, 200.0, gravity=G, manning_n=0.15)
    solver.set_state(np.full(16, 1.0), np.full(16, 3.0))
    for _ in range(3):
        solver.step(10.0)                       # dt far beyond any stable CFL
        assert np.min(solver.primitive()[1]) >= 0.0
        assert np.all(np.isfinite(solver.h))
    assert solver.friction_clipped > 0


def test_lake_at_rest_stays_at_rest_with_friction():
    """Friction must be exactly zero at u = 0 (no spurious force at rest)."""
    from tensorfvm.shallow_water import ShallowWaterBed1D

    n, length, s0, surface = 100, 200.0, 1e-3, 5.0
    x = np.linspace(0.0, length, n) + 0.5 * length / n
    bed = -s0 * x
    solver = ShallowWaterBed1D(n, 0.0, length, bed=bed, gravity=G, manning_n=0.03)
    solver.set_state(np.maximum(surface - bed, 0.0))
    solver.advance(30.0)
    assert np.max(np.abs(solver.primitive()[1])) < 1e-12
    assert solver.surface_drift() < 1e-12


def test_energy_decreases_only_through_friction():
    solver = uniform_solver(manning_n=0.03)
    energies = [solver.kinetic_energy()]
    for _ in range(200):
        solver.step(0.1)
        energies.append(solver.kinetic_energy())
    diffs = np.diff(energies)
    assert np.max(diffs) <= 0.0


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        ShallowWater1D(10, 0.0, 1.0, manning_n=-0.1)
    with pytest.raises(ValueError):
        ShallowWater1D(10, 0.0, 1.0, slope=float("nan"))
    with pytest.raises(ValueError):
        ShallowWater1D(10, 0.0, 1.0).normal_velocity()   # n = 0 -> undefined