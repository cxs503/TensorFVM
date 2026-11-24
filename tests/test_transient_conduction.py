"""Transient conduction against exact storage and two-volume exchange balances."""
import numpy as np
import pytest

from tensorfvm.cht.energy import solve_transient_conduction


@pytest.mark.parametrize("theta", [.5, 1.])
def test_heat_storage_matches_integrated_boundary_and_source(theta):
    dx, dy = np.array([.1, .2, .7]), np.array([.3, .7])
    volume = np.outer(dy, dx)
    capacity = np.array([[2., 3., 4.], [1., 2., 1.]])
    time = np.array([0., .003, .01, .1])
    source = np.array([[1., -3., 0.], [4., 7., -2.]])
    influx = np.array([2., 5.])
    bc = {"east": {"type": "neumann", "values": influx}}
    result = solve_transient_conduction(dx, dy, 3., capacity, bc, time, 258., source=source,
                                        contact_resistance_y=.3, theta=theta)
    stored = np.sum((result - result[0]) * capacity * volume, axis=(1, 2))
    expected = time * (np.dot(influx, dy) + np.sum(source * volume))
    np.testing.assert_allclose(stored, expected, atol=1e-12, rtol=0)


def test_insulated_two_cell_contact_recovers_exact_relaxation_rate():
    # Widths .4/.6, k=2/3, capacities 5/10. The face conductance is
    # 1/(.2/2+.3+.3/3)=2, masses are 2 and 6, hence decay rate 4/3.
    errors = []
    for steps in (10, 20, 40):
        times = np.linspace(0, 1., steps+1)
        history = solve_transient_conduction([.4, .6], [1.], [2., 3.], [5., 10.],
                                             {}, times, [3., -1.], contact_resistance_x=.3)
        exact = np.exp(-4/3 * times[:, None]) * [3., -1.]
        errors.append(abs(history[:, 0] - exact).max())
    assert errors[0] / errors[1] > 3.9
    assert errors[1] / errors[2] > 3.9
    assert errors[-1] < 1.1e-4


def test_vertical_layered_conduction_matches_rotated_horizontal_solution():
    widths = [.2, .3, .5]
    k = np.array([1., 4., 2.])
    initial = np.array([2., 0., 1.])
    hot = {"type": "robin", "values": 3., "film_coefficient": 2.}
    cold = {"type": "dirichlet", "values": -1.}
    times = [0., .01, .1]
    horizontal = solve_transient_conduction(widths, [1.], k, 1., {"west": hot, "east": cold},
                                            times, initial, contact_resistance_x=.1)
    vertical = solve_transient_conduction([1.], widths, 9., 1., {"south": hot, "north": cold},
                                          times, initial[:, None], conductivity_y=k[:, None],
                                          contact_resistance_y=.1)
    np.testing.assert_allclose(vertical[:, :, 0], horizontal[:, 0, :], atol=1e-13, rtol=0)


@pytest.mark.parametrize("ky", [0., -1., np.nan, np.inf])
def test_invalid_directional_conductivity(ky):
    with pytest.raises(ValueError, match="conductivity_y"):
        solve_transient_conduction([1.], [1.], 1., 1., {}, [0., 1.], 0., conductivity_y=ky)


@pytest.mark.parametrize("theta", [.5, .75, 1.])
def test_mixed_wall_power_history_closes_every_nonuniform_step(theta):
    dx, dy = np.array([.2, .8]), np.array([.4, .6])
    bc = {"west": {"types": ["dirichlet", "adiabatic"], "values": [2., 0.]},
          "north": {"types": ["robin", "neumann"], "values": [3., -2.],
                    "film_coefficient": 4.}}
    times = np.array([0., .001, .05, .1])
    temp, walls = solve_transient_conduction(dx, dy, 2., 3., bc, times, 1., source=.7,
                                             conductivity_y=5., theta=theta, return_wall_power=True)
    np.testing.assert_allclose(walls["west"][0], [8., 0.])
    np.testing.assert_allclose(walls["north"][0], [2.*dx[0]/(.3/5.+1/4.), -2.*dx[1]])
    for side in ("south", "east"):
        np.testing.assert_array_equal(walls[side], np.zeros((len(times), 2)))
    net = sum(power.sum(axis=1) for power in walls.values()) + .7
    stored = (np.diff(temp, axis=0)*3.*np.outer(dy, dx)).sum(axis=(1, 2))
    np.testing.assert_allclose(stored, np.diff(times)*((1-theta)*net[:-1]+theta*net[1:]), atol=1e-13)
