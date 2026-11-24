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
