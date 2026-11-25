"""Check both components of the conductive flux on a nonuniform rectangle."""
import numpy as np

from tensorfvm.cht.energy import solve_steady_energy


def test_affine_anisotropic_temperature_and_four_boundary_fluxes():
    dx, dy = np.array([.3, .2, .5]), np.array([.3, .8, .9])
    x, y = dx.cumsum() - dx/2, dy.cumsum() - dy/2
    # T = 1 + 2x - 3y, so the outward Fourier flux vector is (-10, 1.5).
    values = {"west": 1-3*y, "east": 3-3*y, "south": 1+2*x, "north": -5+2*x}
    bc = {side: {"type": "dirichlet", "values": face} for side, face in values.items()}
    temperature, info = solve_steady_energy(dx, dy, np.full((3, 3), 5.),
                                            np.zeros((3, 4)), np.zeros((4, 3)), bc,
                                            conductivity_y=.5)
    np.testing.assert_allclose(temperature, 1+2*x-3*y[:, None], atol=1e-13, rtol=0)
    inward = {"west": -20., "east": 20., "south": 1.5, "north": -1.5}
    for side, expected in inward.items():
        np.testing.assert_allclose(info["ledger"].boundary[side], expected, atol=1e-13, rtol=0)
