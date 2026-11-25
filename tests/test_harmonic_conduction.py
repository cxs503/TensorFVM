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


@pytest.mark.parametrize("dy", [[.7], [.3, .7], [.2, .3, .5]])
def test_periodic_y_seam_material_resistance_and_cell_balance(dy):
    nx, ny = 2, len(dy)
    dx = np.array([.2, .8])
    ky = 1+np.arange(ny*nx).reshape(ny, nx)
    volume = np.outer(dy, dx)
    source = 1j*ky+.3
    t, wall, interior = solve_harmonic_conduction(dx, dy, 3., 2., {}, 4., conductivity_y=ky,
                            source=source, return_interior_power=True, periodic_y=True)
    expected = dx*(t[-1]-t[0])/(.5*dy[-1]/ky[-1]+.5*dy[0]/ky[0])
    np.testing.assert_allclose(wall["south"], expected)
    np.testing.assert_array_equal(wall["south"], -wall["north"])
    qy = np.vstack((wall["south"], interior["y"], -wall["north"]))
    qx = np.column_stack((wall["west"], interior["x"], -wall["east"]))
    np.testing.assert_allclose(8j*volume*t, source*volume-np.diff(qy,axis=0)-np.diff(qx,axis=1))
    with pytest.raises(ValueError, match="periodic_y"):
        solve_harmonic_conduction(dx, dy, 1., 1., {"north": {"type":"neumann", "values":0.}},
                                  4., periodic_y=True)


@pytest.mark.parametrize("h", [0., 3.])
def test_dynamic_wall_matches_explicit_two_temperature_system(h):
    omega, cw, ambient, src = 4., .7, .3+.2j, 1.-.1j
    # Unit cube, unit depth: G_half=4, cell capacity=3.
    matrix = [[4+3j*omega,-4],[-4,4+h+1j*omega*cw]]
    cell, wall = np.linalg.solve(matrix,[src,h*ambient])
    data = dict(type="robin",values=ambient,film_coefficient=h,wall_areal_heat_capacity=cw)
    u, powers = solve_harmonic_conduction([1.],[1.],2.,3.,{"west":data},omega,source=src)
    np.testing.assert_allclose(u, cell)
    np.testing.assert_allclose(powers["west"], 4*(wall-cell))
    np.testing.assert_allclose(powers["west"], h*(ambient-wall)-1j*omega*cw*wall)


def test_mixed_face_storage_closes_individual_wall_balances():
    h, cw = np.array([2., 4., 0.]), np.array([1., 0., 0.])
    ambient = np.array([.5+.8j, -.2, .7j])
    dy = np.array([.2,.3,.5])
    west = dict(types=["robin","robin","neumann"], values=ambient,
                film_coefficient=h, wall_areal_heat_capacity=cw)
    u, powers = solve_harmonic_conduction([.4],dy,3.,2.,{"west":west},5.)
    flux = powers["west"]/dy
    surface = u[:,0]+flux*(.2/3.)
    np.testing.assert_allclose(flux[:2],(h*(ambient-surface)-5j*cw*surface)[:2])
    np.testing.assert_allclose(flux[2],ambient[2])
    np.testing.assert_allclose(5j*2.*.4*(dy @ u[:,0]),powers["west"].sum())
    west["wall_areal_heat_capacity"] = [0., 0., 1.]
    with pytest.raises(ValueError,match="Robin faces"):
        solve_harmonic_conduction([.4],dy,3.,2.,{"west":west},5.)
