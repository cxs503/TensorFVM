import numpy as np
import pytest

from tensorfvm.cht.annulus import solve_annular_conduction


def test_steady_series_resistances_on_graded_two_material_annulus():
    r = np.array([.25,.27,.41,.6,.83,1.])
    k = np.array([2.,2.,2.,1.3,1.3])
    contacts = np.array([0.,0.,.15,0.])
    t, rates = solve_annular_conduction(r, k, 1., [2.,-.5], 0., contact_resistance=contacts)
    expected = 2.5*2*np.pi/(np.log(.6/.25)/2+.15/.6+np.log(1/.6)/1.3)
    np.testing.assert_allclose(rates, expected, rtol=2e-14)
    centers = .5*(r[1:]+r[:-1])
    exact = np.where(centers < .6, 2.-expected*np.log(centers/.25)/(4*np.pi),
                     -.5+expected*np.log(1./centers)/(2*np.pi*1.3))
    np.testing.assert_allclose(t, exact, atol=1e-14)


@pytest.mark.parametrize("m", [0,2])
def test_complex_single_volume_balances_both_walls_storage_source_and_angular_loss(m):
    walls = np.array([1.+2j, -.3j])
    t, rates = solve_annular_conduction([.5,1.], 1.5, 2., walls, 3., source=.7j, azimuthal_mode=m)
    volume = .75*np.pi
    lateral = 2*np.pi*1.5*m**2*np.log(2)
    np.testing.assert_allclose(rates[0]-rates[1]+.7j*volume, (6j*volume+lateral)*t[0])


def test_steady_film_annulus_matches_series_resistance_and_insulation():
    r = np.array([.25,.4,.6,1.])
    h = (4., 3.)
    _, rate = solve_annular_conduction(r, 2., 1., [3.,0.], 0., film_coefficients=h)
    resistance = 1/(r[0]*h[0]) + np.log(r[-1]/r[0])/2 + 1/(r[-1]*h[1])
    np.testing.assert_allclose(rate, 6*np.pi/resistance, atol=1e-13)
    field, rate = solve_annular_conduction(r, 2., 1., [3j,2.], 2., source=4j, film_coefficients=(0.,0.))
    np.testing.assert_allclose(field, 2., atol=1e-13)
    np.testing.assert_allclose(rate, 0., atol=1e-13)
    with pytest.raises(ValueError, match="reference"):
        solve_annular_conduction(r, 2., 1., [0.,0.], 0., film_coefficients=(0.,0.))


def test_volume_centroid_recovers_steady_log_profile_on_graded_cells():
    radii = np.array([.1,.15,.6,.68,1.])
    field, rate = solve_annular_conduction(radii, 2., 1., np.log(radii[[0,-1]]), 0.,
                                           cell_location="volume_centroid")
    centroid = (2./3)*np.diff(radii**3)/np.diff(radii**2)
    np.testing.assert_allclose(field, np.log(centroid), rtol=0., atol=1e-14)
    np.testing.assert_allclose(rate, -4*np.pi, atol=1e-13, rtol=0.)
    with pytest.raises(ValueError, match="cell_location"):
        solve_annular_conduction(radii, 2., 1., [0.,1.], 0., cell_location="unknown")
