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


def test_two_flux_phasors_drive_single_shell_storage():
    r = [.25, 1.]
    q = np.array([3.+2j, -.7j])
    t, rates = solve_annular_conduction(r, 2., 3., (None,None), 4., wall_fluxes=q)
    inward = 2*np.pi*np.array(r)*q
    np.testing.assert_allclose(rates, inward*[1,-1])
    np.testing.assert_allclose(t[0], inward.sum()/(12j*np.pi*(1.-.25**2)))
    with pytest.raises(ValueError, match="wall flux"):
        solve_annular_conduction(r, 2., 1., (1.,0.), 2., wall_fluxes=(1.,None))


@pytest.mark.parametrize("driven", [0,1])
def test_prescribed_wall_flux_on_either_end_matches_steady_log_profile(driven):
    radii = np.array([.25,.4,.7,1.])
    walls = list(np.log(radii[[0,-1]]))
    walls[driven] = None
    fluxes = [None,None]
    fluxes[driven] = (-1 if driven == 0 else 1)*2./radii[[0,-1]][driven]
    field, rates = solve_annular_conduction(radii, 2., 1., walls, 0., wall_fluxes=fluxes)
    np.testing.assert_allclose(field, np.log(.5*(radii[:-1]+radii[1:])), atol=1e-14)
    np.testing.assert_allclose(rates, -4*np.pi, atol=1e-13)


@pytest.mark.parametrize("omega", [0., 7.])
def test_source_driven_angular_mode_with_zero_radial_flux(omega):
    # T=(1+2i)*cos(3 angle) is continuous through the material interface.
    # Its forcing contains both harmonic storage and k*m²*T/r². Integrate
    # the latter over each cylindrical shell before passing its cell average.
    r = np.array([.2,.34,.5,.83,1.])
    k, capacity = np.array([1.,1.,3.,3.]), np.array([2.,2.,1.5,1.5])
    exact, mode = 1.+2j, 3
    inverse_r2_average = 2*np.log(r[1:]/r[:-1])/np.diff(r**2)
    source = exact*(1j*omega*capacity + mode**2*k*inverse_r2_average)
    t, rate = solve_annular_conduction(r, k, capacity, (None,None), omega, azimuthal_mode=mode,
                                       source=source, contact_resistance=[0.,.3,0.],
                                       wall_fluxes=(0.,0.), cell_location="volume_centroid")
    np.testing.assert_allclose(t, exact, rtol=0., atol=3e-14)
    np.testing.assert_allclose(rate, 0., rtol=0., atol=1e-12)


@pytest.mark.parametrize("location", ["midpoint", "volume_centroid"])
@pytest.mark.parametrize("omega", [0., 5.])
def test_distributed_balance_recovers_forced_log_quadratic_field(location, omega):
    r = np.array([.2,.23,.31,.55,.67,.9,1.])
    k, cv = 2.5, 1.7
    center = .5*(r[:-1]+r[1:])
    if location == "volume_centroid":
        center = (2./3)*np.diff(r**3)/np.diff(r**2)
    quadratic, logarithmic = 1.-.2j, -.3+.4j
    def field(x):
        return 2.+quadratic*x**2+logarithmic*np.log(x)
    # The discrete storage term is collocated at the declared cell radius.
    generation = -4*k*quadratic + 1j*omega*cv*field(center)
    exact_rate = -2*np.pi*k*(2*quadratic*r**2+logarithmic)
    temp, rate = solve_annular_conduction(r, k, cv, field(r[[0,-1]]), omega, source=generation,
                                          cell_location=location, distributed_source=True)
    np.testing.assert_allclose(temp, field(center), rtol=0., atol=1e-13)
    np.testing.assert_allclose(rate, exact_rate, rtol=0., atol=2e-12)
