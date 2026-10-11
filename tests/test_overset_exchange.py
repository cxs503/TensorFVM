import numpy as np
import pytest
from tensorfvm.overset import annular_grid, cartesian_grid, circle_polygon, build_overset
from tensorfvm.overset_exchange import (
    MacroscopicState, LatticeUnits, exchange_macroscopic, interpolate_time,
)


def units(policy='incompressible_reference', **overrides):
    config = dict(dx=.01, dt=.0001, density_scale=1000.,
                  reference_density=998., reference_pressure=101325.,
                  cs2=1/3, density_policy=policy)
    config.update(overrides)
    return LatticeUnits(**config)


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('policy', ['incompressible_reference', 'isothermal_eos'])
def test_si_roundtrip_pressure_gauge_and_eos(dimension, policy):
    scale = units(policy)
    p = np.array([100325., 101325., 102325.])
    rho = np.full(3, 998.)
    if policy == 'isothermal_eos':
        rho += (p-101325.) / ((.01/.0001)**2/3)
    u = np.tile(np.arange(1, dimension+1)*.2, (3, 1))
    state = MacroscopicState(rho, u, p, .2)
    lrho, lu = scale.encode(state)
    assert lrho[1] == pytest.approx(.998)
    assert lrho[2]-lrho[1] == pytest.approx(.0003)
    decoded = scale.decode(lrho, lu, time=.2)
    np.testing.assert_allclose(decoded.density, rho)
    np.testing.assert_allclose(decoded.velocity, u)
    np.testing.assert_allclose(decoded.pressure, p)


def test_distinct_lattice_scales_exchange_through_si():
    state = MacroscopicState([998.], [[.3, -.2, .1]], [101825.], 1.)
    a, b = units(), units(dx=.02, dt=.0004, density_scale=500.)
    rho_a, u_a = a.encode(state)
    rho_b, u_b = b.encode(a.decode(rho_a, u_a, time=1.))
    np.testing.assert_allclose(u_b, 2*u_a)
    assert rho_b[0] != pytest.approx(rho_a[0])
    np.testing.assert_allclose(b.decode(rho_b, u_b, time=1.).pressure, state.pressure)


def test_viscosity_scale_preserves_reynolds_number():
    scale = units()
    nu, speed, length = 1e-6, .5, .4
    lattice_nu = scale.lattice_viscosity(nu)
    assert lattice_nu == pytest.approx(1e-6)
    assert (speed*scale.dt/scale.dx)*(length/scale.dx)/lattice_nu == pytest.approx(
        speed*length/nu)
    with pytest.raises(ValueError, match='viscosity'):
        scale.lattice_viscosity(-1.)


def test_reject_density_conflict_negative_density_and_high_mach():
    with pytest.raises(ValueError, match='conflicts'):
        units().encode(MacroscopicState([1000.], [[0., 0.]], [101325.], 0.))
    with pytest.raises(ValueError, match='conflicts'):
        units('isothermal_eos').encode(
            MacroscopicState([998.], [[0., 0.]], [111325.], 0.))
    with pytest.raises(ValueError, match='Invalid lattice'):
        units().decode([-1.], [[0., 0.]], time=0.)
    with pytest.raises(ValueError, match='Mach'):
        units().encode(MacroscopicState([998.], [[20., 0.]], [101325.], 0.))
    with pytest.raises(ValueError, match='supported'):
        units('multiphase')


def test_state_copies_inputs_and_rejects_invalid_shapes():
    rho = np.array([998.])
    state = MacroscopicState(rho, [[0., 0.]], [0.], 0.)
    rho[0] = -1.
    assert state.density[0] == 998.
    with pytest.raises(ValueError):
        state.velocity[0, 0] = 1.
    with pytest.raises(ValueError):
        MacroscopicState([998.], [[0., 0., 0., 0.]], [0.], 0.)


def test_subcycling_interpolation_no_extrapolation():
    a = MacroscopicState([998.], [[1., 2.]], [10.], .1)
    b = MacroscopicState([998.], [[3., 4.]], [30.], .3)
    mid = interpolate_time(a, b, .2)
    np.testing.assert_allclose(mid.velocity, [[2., 3.]])
    np.testing.assert_allclose(mid.pressure, [20.])
    with pytest.raises(ValueError, match='bracketed'):
        interpolate_time(a, b, .4)


def test_real_cartesian_body_fitted_exchange_preserves_affine_si_fields():
    bg = cartesian_grid((-2, 2, -2, 2), 32, 32)
    fitted = annular_grid((0, 0), .3, 1., 96, 16)
    conn = build_overset(bg, fitted, circle_polygon((0, 0), .3),
                         circle_polygon((0, 0), .65))
    states = []
    for grid in conn.grids:
        x, y = grid.centers.T
        states.append(MacroscopicState(998.+.1*x, np.column_stack((x+2*y, 2*x-y)),
                                       101325.+100*x-20*y, .5))
    exchanged = exchange_macroscopic(conn, states)
    for original, result in zip(states, exchanged):
        for name in ('density', 'velocity', 'pressure'):
            np.testing.assert_allclose(getattr(original, name), getattr(result, name),
                                       rtol=1e-13, atol=1e-11)
    later = MacroscopicState(states[1].density, states[1].velocity,
                             states[1].pressure, .6)
    with pytest.raises(ValueError, match='Synchronize'):
        exchange_macroscopic(conn, [states[0], later])
