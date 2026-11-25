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


def test_steady_flux_driven_annulus_sets_mean_and_rejects_net_heating():
    r = np.array([.2,.3,.56,.65,.95,1.2])
    cv, k = 2., 3.
    centers = .5*(r[:-1]+r[1:])
    volume = np.pi*np.diff(r**2)
    # T=r² + 0.4 log(r) + constant. Both walls impose the exact flux.
    reference = centers**2 + .4*np.log(centers)
    exact_rate = -2*np.pi*k*(2*r**2+.4)
    flux = exact_rate[[0,-1]]*[1,-1]/(2*np.pi*r[[0,-1]])
    def solve(mean, source):
        return solve_annular_conduction(r, k, cv, [None,None], 0., wall_fluxes=flux, source=source,
                                        distributed_source=True, mean_temperature=mean)
    for mean in (0., 17.-.5j):
        t, rate = solve(mean, -4*k)
        np.testing.assert_allclose(t, reference + mean - np.average(reference,weights=volume), atol=1e-13)
        np.testing.assert_allclose(rate, exact_rate, rtol=0., atol=1e-12)
    with pytest.raises(ValueError, match="incompatible"):
        solve(0., -4*k+.01)
    t, rate = solve_annular_conduction([1.,2.], 1., 1., (None,None), 0.,
                                       wall_fluxes=(0.,0.), mean_temperature=3.)
    np.testing.assert_allclose(t, 3.)
    np.testing.assert_array_equal(rate, [0.,0.])


@pytest.mark.parametrize("fraction",[0.,.37,1.])
def test_storing_sheet_uniform_harmonic_temperature(fraction):
    from tensorfvm.cht.annulus import solve_annular_sheet
    edges = np.array([.3,.42,.58,.73,1.])
    omega,temperature = 4.7,.6+.3j
    cv = np.array([1.,1.,2.,2.])
    ambient,h,capacity = .2-.1j,1.1,.4
    sheet_load = 1j*omega*capacity*temperature+h*(temperature-ambient)
    field,inner,outer,sheet = solve_annular_sheet(edges,[1.,1.,3.,3.],cv,(None,None),omega,2,
               wall_fluxes=(0.,0.),resistance=.3,inner_resistance_fraction=fraction,sheet_capacity=capacity,
               sheet_film=h,ambient=ambient,sheet_source=sheet_load,source=1j*omega*cv*temperature,
               distributed_source=True)
    np.testing.assert_allclose(field,temperature,atol=2e-14,rtol=0.)
    np.testing.assert_allclose(sheet,temperature,atol=2e-14,rtol=0.)
    np.testing.assert_allclose(np.r_[inner,outer],0.,atol=5e-14)


def test_floating_annular_sheet_generation_and_compatibility():
    from tensorfvm.cht.annulus import solve_annular_sheet
    edges = np.array([.3,.4,.6,.8,1.])
    opts = dict(wall_fluxes=(0.,-.6),mean_temperature=1.8,sheet_source=1.,resistance=.3)
    u,qi,qo,s = solve_annular_sheet(edges,1.,1.,(None,None),0.,2,**opts)
    np.testing.assert_allclose(qi,0.,atol=1e-14)
    np.testing.assert_allclose(qo,2*np.pi*.6,atol=1e-14)
    np.testing.assert_allclose(s,u[0],atol=1e-14)
    np.testing.assert_allclose(np.average(u,weights=np.diff(edges**2)),1.8)
    opts["sheet_source"] = .9
    with pytest.raises(ValueError,match="incompatible"):
        solve_annular_sheet(edges,1.,1.,(None,None),0.,2,**opts)


def test_low_frequency_insulated_sheet_total_capacity_and_radial_flux():
    from tensorfvm.cht.annulus import solve_annular_sheet
    edges = np.linspace(.4,1.1,31)
    j = 13
    circum = 2*np.pi*edges[j]
    c = np.linspace(1.,2.,30)*np.pi*np.diff(edges**2)
    sheet_c = .35
    omega = 1e-10
    field,inner,outer,sheet = solve_annular_sheet(edges,2.,np.linspace(1.,2.,30),(None,None),omega,j,
                       wall_fluxes=(0.,0.),resistance=.3,sheet_capacity=sheet_c,sheet_source=1.)
    rise = circum/(c.sum()+circum*sheet_c)
    np.testing.assert_allclose(1j*omega*np.r_[field,sheet],rise,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(inner,-rise*np.r_[0.,np.cumsum(c[:j])],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(outer,rise*np.r_[np.cumsum(c[j:][::-1])[::-1],0.],atol=1e-10,rtol=1e-10)


def test_transient_annular_sheet_switched_pulse_matches_three_store_exponential():
    from scipy.linalg import expm
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    edges = np.array([.4,.7,1.1])
    centers = (2/3)*np.diff(edges**3)/np.diff(edges**2)
    k = [1.2,2.3]
    circum = 2*np.pi*.7
    capacity = np.array([1.5*np.pi*(.7**2-.4**2),circum*.31,2.*np.pi*(1.1**2-.7**2)])
    g = 2*np.pi/np.array([np.log(centers[0]/.4)/k[0],np.log(.7/centers[0])/k[0]+.09/.7,
                          np.log(centers[1]/.7)/k[1]+.21/.7,np.log(1.1/centers[1])/k[1]])
    operator = np.diag(g[:-1]+g[1:]+[0.,circum*.8,0.])-np.diag(g[1:-1],1)-np.diag(g[1:-1],-1)
    previous = None
    for n in (8,16,32):
        times = np.r_[np.linspace(0,.07,n+1),np.linspace(.07,.21,n+1)[1:]]
        drive = np.repeat([1.4,-.3],n)
        u,s,qi,qo,ext = solve_transient_annular_sheet(edges,k,[1.5,2.],(.3,-.1),times,[.4,.7],1,
                 initial_sheet_temperature=.1,sheet_capacity=.31,resistance=.3,inner_resistance_fraction=.3,
                 sheet_film=.8,ambient=.23,sheet_source_intervals=np.repeat(drive[:,None],2,axis=1))
        state = np.column_stack((u[:,0],s,u[:,1]))
        heat = np.column_stack((qi,qo))
        dt = np.diff(times)
        np.testing.assert_allclose(np.diff(state,axis=0)*capacity+np.diff(heat,axis=1),
                                   np.column_stack((np.zeros_like(dt),circum*drive*dt-ext,np.zeros_like(dt))),atol=2e-14)
        ref = state[0]
        for power,span in ((1.4,.07),(-.3,.14)):
            steady = np.linalg.solve(operator,[g[0]*.3,circum*(power+.8*.23),-g[-1]*.1])
            ref = steady+expm(-span*operator/capacity[:,None])@(ref-steady)
        error = max(abs(state[-1]-ref))
        if previous is not None:
            assert error < previous/3.8
        previous = error


@pytest.mark.parametrize("theta",[.5,1.])
def test_annular_sheet_flux_wall_global_energy(theta):
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    edges = np.array([.3,.5,.64,.92])
    t = np.array([0.,.01,.13,.3])
    west,east = 2.-t,.3+2*t
    drive = np.array([[.7,.8],[-1.,-.2],[.1,.3]])
    h,cs = .3,.5
    u,s,hi,ho,external = solve_transient_annular_sheet(edges,[1.,2.,2.],1.3,(None,None),
               t,[.3,.1,-.2],1,initial_sheet_temperature=.2,sheet_capacity=cs,sheet_film=h,
               ambient=.6,sheet_source_intervals=drive,wall_fluxes=(west,east),theta=theta)
    dt = np.diff(t)
    def stage(x):
        return (1-theta)*x[:-1]+theta*x[1:]
    np.testing.assert_allclose(hi[:,0],dt*2*np.pi*edges[0]*stage(west))
    np.testing.assert_allclose(ho[:,-1],-dt*2*np.pi*edges[-1]*stage(east))
    total = u@(1.3*np.pi*np.diff(edges**2))+2*np.pi*edges[1]*cs*s
    generation = ((1-theta)*drive[:,0]+theta*drive[:,1])*dt*2*np.pi*edges[1]
    np.testing.assert_allclose(np.diff(total),hi[:,0]-ho[:,-1]+generation-external,atol=2e-14)
