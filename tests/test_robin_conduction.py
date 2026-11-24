"""Robin conduction with finite-volume half-cell and external film resistances."""
import numpy as np
import pytest

from tensorfvm.cht.energy import solve_steady_energy, solve_transient_conduction


@pytest.mark.parametrize("vertical", [False, True])
def test_film_series_flux_and_contact_on_stretched_slab(vertical):
    widths = np.array([.1, .4, .2, .3])
    k = np.array([[4., 4., 1., 1.]])
    contacts = np.array([[0., .2, 0.]])
    sides = ("south", "north") if vertical else ("west", "east")
    dx, dy = ([1.], widths) if vertical else (widths, [1.])
    if vertical:
        k = k.T
    dx, dy = np.asarray(dx), np.asarray(dy)
    ny, nx = k.shape
    bc = {s: {"type": "robin", "values": t, "film_coefficient": h}
          for s, t, h in zip(sides, (310., 290.), (12., 3.))}
    contact = {"contact_resistance_y": contacts.T} if vertical else {"contact_resistance_x": contacts}
    temp, info = solve_steady_energy(dx, dy, k, np.zeros((ny,nx+1)), np.zeros((ny+1,nx)),
                                     bc, high_order=False, **contact)
    rate = 20 / (1/12 + .5/4 + .2 + .5 + 1/3)
    bulk = widths/k.ravel()
    expected = 310 - rate*(1/12 + bulk.cumsum() - bulk/2 + [0.,0.,.2,.2])
    np.testing.assert_allclose(temp.ravel(), expected, rtol=0, atol=2e-12)
    for side, sign in zip(sides, (1., -1.)):
        assert abs(info["ledger"].boundary[side] - sign*rate) < 1e-12


def cooling_error(n):
    # The Robin eigenmode cos(2*(x-1/2)) decays at rate 4 for k=C=1.
    dx = np.full(n, 1/n)
    x = (np.arange(n)+.5)/n
    h = 2 * np.tan(1.)
    bc = {side: {"type": "robin", "values": 0., "film_coefficient": h} for side in ("west","east")}
    initial = np.cos(2*(x-.5))
    time = np.linspace(0, .1, n+1)
    result = solve_transient_conduction(dx, [1.], 1., 1., bc, time, initial)
    return np.max(np.abs(result[-1,0] - np.exp(-.4)*initial))


def test_robin_cooling_refinement():
    errors = [cooling_error(n) for n in (16, 32, 64)]
    assert all(a > 3.8*b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-4


def test_mixed_per_face_films_and_zero_film_balance():
    # T=x+2y on a unit square; exact face heat rates integrate the gradient.
    dx = dy = np.full(3, 1/3)
    x = y = (np.arange(3)+.5)/3
    h = np.array([3., 5., 9.])
    bc = {"west": {"types": ["robin", "dirichlet", "robin"],
                   "values": 2*y - [1/h[0], 0., 1/h[2]], "film_coefficient": h},
          "east": {"type": "dirichlet", "values": 1+2*y},
          "north": {"type": "robin", "values": x+2+2/h, "film_coefficient": h},
          "south": {"type": "neumann", "values": -2.}}
    result, info = solve_steady_energy(dx, dy, np.ones((3,3)), np.zeros((3,4)), np.zeros((4,3)), bc)
    np.testing.assert_allclose(result, x + 2*y[:,None], atol=1e-13)
    assert abs(info["ledger"].boundary["west"] + 1.) < 1e-13
    bc = {"west": {"type": "robin", "values": 25., "film_coefficient": 0.}}
    result = solve_transient_conduction(dx, dy, 1., 1., bc, [0., 1.], 7.)
    np.testing.assert_allclose(result, 7., atol=1e-13)
