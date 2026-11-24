import numpy as np
import pytest

from tensorfvm.cht.energy import solve_harmonic_conduction


def test_harmonic_single_cell_complex_wall_and_source_power():
    omega, west, north, east, source = 3., 2.-1j, -.5+.8j, -.4j, .7+1.5j
    bc = {"west": {"type": "dirichlet", "values": west},
          "north": {"type": "robin", "values": north, "film_coefficient": 8.},
          "east": {"type": "neumann", "values": east}}
    t, wall = solve_harmonic_conduction([2.], [1.], 2., 3., bc, omega,
                                        source=source, conductivity_y=4.)
    # West conductance = 2, north film in series with half cell gives 8.
    exact = (2*west + 8*north + east + 2*source)/(10 + 6j*omega)
    np.testing.assert_allclose(t, exact)
    for side, power in dict(west=2*(west-exact), north=8*(north-exact), east=east, south=0.).items():
        np.testing.assert_allclose(wall[side], power)


def test_insulated_pair_contact_with_opposing_heat_inputs():
    t, walls = solve_harmonic_conduction([1.,1.], [1.], 2., 1., {}, 4., source=[1j,-1j],
                                         contact_resistance_x=.5)
    np.testing.assert_allclose(t[0], np.array([1j,-1j])/(2+4j))
    for power in walls.values():
        np.testing.assert_array_equal(power, 0.)
    with pytest.raises(ValueError, match="angular_frequency"):
        solve_harmonic_conduction([1.], [1.], 1., 1., {}, 0.)


def test_harmonic_cell_storage_matches_every_oriented_face_and_source():
    dx, dy = np.array([.2,.3,.5]), np.array([.3,.7])
    volume = np.outer(dy, dx)
    source = np.array([[1j,2.,.5],[-.6,1.2j,3.]])
    bc = {"west": {"type":"robin", "values":2.-1j, "film_coefficient":3.},
          "north": {"type":"neumann", "values":[1.,-1j,.4j]}}
    t, walls, power = solve_harmonic_conduction(dx, dy, [1.,2.,.7], 3., bc, 2.,
                      conductivity_y=4., contact_resistance_x=.2, contact_resistance_y=.3,
                      source=source, return_interior_power=True)
    x = np.column_stack((walls["west"],power["x"],-walls["east"]))
    y = np.vstack((walls["south"],power["y"],-walls["north"]))
    stored = 2j*3.*volume*t
    np.testing.assert_allclose(stored, source*volume-np.diff(x,axis=1)-np.diff(y,axis=0), atol=1e-13)
