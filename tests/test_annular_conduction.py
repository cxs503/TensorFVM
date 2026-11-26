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


@pytest.mark.parametrize("theta",[.5,.75,1.])
def test_massless_sheet_jump_right_trace_and_integrated_heat(theta):
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    edges = np.array([.4,.7,1.2])
    times = np.array([0.,.1,.28,.36])
    drive = np.array([[0.,0.],[1.4,.2],[-.8,-.3]])
    k = np.array([1.5,2.1])
    fluid,h = -.4,.7
    opts = dict(sheet_capacity=0.,resistance=.3,inner_resistance_fraction=.4,theta=theta,
                ambient=fluid,sheet_film=h,wall_fluxes=(0.,0.))
    u,s,qi,qo,external = solve_transient_annular_sheet(edges,k,1.,(None,None),times,[.2,.5],1,
                                                         sheet_source_intervals=drive,**opts)
    center = (2/3)*np.diff(edges**3)/np.diff(edges**2)
    gi,go = 2*np.pi/np.array([np.log(.7/center[0])/k[0]+.12/.7,np.log(center[1]/.7)/k[1]+.18/.7])
    circum = 2*np.pi*.7
    target = (u[:,0]*gi+u[:,1]*go+circum*(np.r_[drive[:,0],drive[-1,1]]+h*fluid))/(gi+go+circum*h)
    np.testing.assert_allclose(s,target,atol=2e-14,rtol=0.)
    applied = np.diff(times)*circum*((1-theta)*drive[:,0]+theta*drive[:,1])
    np.testing.assert_allclose(qo[:,0]-qi[:,-1]+external,applied,atol=2e-14)
    np.testing.assert_allclose(np.diff(u,axis=0)*np.pi*np.diff(edges**2),
                               np.column_stack((-qi[:,-1],qo[:,0])),atol=2e-14)


def test_transient_volume_heating_one_sided_pulse_and_uniform_storage():
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    edges = np.array([.4,.52,.7,.9,1.1])
    cv = np.array([1.3,1.3,2.1,2.1])
    t = np.array([0.,.04,.19,.26])
    rate = np.array([[.3,1.1],[-.4,-.4],[1.,.2]])
    expected = .6+np.r_[0.,np.cumsum(np.diff(t)*rate.mean(1))]
    u,s,qi,qo,external = solve_transient_annular_sheet(edges,[2.,2.,3.,3.],cv,(None,None),t,.6,2,
             wall_fluxes=(0.,0.),sheet_capacity=.41,initial_sheet_temperature=.6,resistance=.2,
             sheet_source_intervals=.41*rate,source_intervals=rate[:,:,None]*cv)
    np.testing.assert_allclose(u,np.broadcast_to(expected[:,None],u.shape),rtol=0.,atol=3e-14)
    np.testing.assert_allclose(s,expected,rtol=0.,atol=3e-14)
    for heat in (qi,qo,external):
        np.testing.assert_allclose(heat,0.,atol=3e-14)


def test_transient_annular_films_preserve_logarithmic_contact_solution():
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    edges = np.array([.4,.48,.7,.92,1.1])
    k = np.array([1.3,1.3,2.1,2.1])
    rate = 1.4
    b = edges[2]
    sheet = .55
    face = np.array([sheet+.4*.2*rate/(2*np.pi*b),sheet-.6*.2*rate/(2*np.pi*b)])
    center = (2/3)*np.diff(edges**3)/np.diff(edges**2)
    initial = np.repeat(face,2)-rate*np.log(center/b)/(2*np.pi*k)
    fluid = face-rate*np.log(edges[[0,-1]]/b)/(2*np.pi*k[[0,-1]])+rate/(2*np.pi*edges[[0,-1]]*np.array([1.4,-2.3]))
    t = np.array([0.,.02,.14,.39])
    u,s,qi,qo,_ = solve_transient_annular_sheet(edges,k,1.,fluid,t,initial,2,sheet_capacity=.3,
             initial_sheet_temperature=sheet,resistance=.2,inner_resistance_fraction=.4,film_coefficients=(1.4,2.3))
    np.testing.assert_allclose(u,np.broadcast_to(initial,u.shape),rtol=0.,atol=1e-14)
    np.testing.assert_allclose(s,sheet,atol=1e-14,rtol=0.)
    for heat in (qi,qo):
        np.testing.assert_allclose(heat,np.broadcast_to(rate*np.diff(t)[:,None],heat.shape),rtol=0.,atol=1e-14)


@pytest.mark.parametrize("bc",["temperature","flux","film"])
def test_annulus_surface_temperature_at_each_output(bc):
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    edges = np.array([.35,.6,.79,1.])
    t = np.array([0.,.014,.087])
    walls = np.stack((.2+t,.7-.5*t),axis=1)
    fluxes,films = (None,None),(None,None)
    if bc=="flux":
        fluxes = walls.T
    if bc=="film":
        films = (0.,2.3)
    u,_,qi,qo,_,surface = solve_transient_annular_sheet(edges,1.3,2.1,(None,None) if bc=="flux" else walls.T,
              t,[.1,.5,.4],1,sheet_capacity=.4,initial_sheet_temperature=.3,wall_fluxes=fluxes,
              film_coefficients=films,return_wall_temperatures=True)
    centers = (2/3)*np.diff(edges**3)/np.diff(edges**2)
    half = abs(np.log(centers[[0,-1]]/edges[[0,-1]]))/(2*np.pi*1.3)
    inward = (surface-u[:,[0,-1]])/half
    np.testing.assert_allclose(np.column_stack((qi[:,0],-qo[:,-1])),np.diff(t)[:,None]*(inward[:-1]+inward[1:])/2,atol=2e-14)
    if bc=="temperature":
        np.testing.assert_array_equal(surface,walls)
    elif bc=="flux":
        np.testing.assert_allclose(inward,2*np.pi*edges[[0,-1]]*walls)
    else:
        np.testing.assert_allclose(inward,2*np.pi*edges[[0,-1]]*films*(walls-surface),atol=1e-14)


@pytest.mark.parametrize("fraction",[0.,.4,1.])
def test_contact_traces_recover_logarithmic_steady_state(fraction):
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    edges = np.array([.4,.49,.7,.85,1.2])
    b = edges[2]
    k = np.array([1.3,1.3,2.1,2.1])
    rates = np.array([1.6,-.8])
    s,res = .6,.25
    traces = s + np.array([fraction,-(1-fraction)])*res*rates/(2*np.pi*b)
    center = (2/3)*np.diff(edges**3)/np.diff(edges**2)
    u0 = np.repeat(traces,2)-np.repeat(rates,2)*np.log(center/b)/(2*np.pi*k)
    walls = traces-rates*np.log(edges[[0,-1]]/b)/(2*np.pi*k[[0,-1]])
    times = np.array([0.,.02,.13])
    u,sheet,hi,ho,_,observed = solve_transient_annular_sheet(edges,k,1.4,walls,times,u0,2,
             initial_sheet_temperature=s,sheet_capacity=.4,resistance=res,inner_resistance_fraction=fraction,
             sheet_source_intervals=np.diff(rates).item()/(2*np.pi*b),return_contact_temperatures=True)
    np.testing.assert_allclose(observed,np.broadcast_to(traces,observed.shape),atol=1e-14,rtol=0.)
    np.testing.assert_allclose(hi[:,-1],np.diff(times)*rates[0],atol=1e-14)
    np.testing.assert_allclose(ho[:,0],np.diff(times)*rates[1],atol=1e-14)


def test_sdirk2_annulus_stiff_decay_and_uniform_insulated_limit():
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    # Uniform cooling mode: all masses receive linear losses proportional
    # to capacity using m=1. Choose cell capacities to make that rate 1.
    r = np.array([.4,.55,.7,1.])
    c = 2*np.log(r[1:]/r[:-1])/np.diff(r**2)
    opts = dict(azimuthal_mode=1,wall_fluxes=(0.,0.),initial_sheet_temperature=1.,
                sheet_capacity=.23,sheet_film=.23,resistance=.3,method="sdirk2")
    for dt in (.02,.1,1000.):
        u,s,_,_,_ = solve_transient_annular_sheet(r,1.,c,(None,None),[0.,dt],1.,1,**opts)
        gamma = 1-1/np.sqrt(2)
        exact_step = (1-(1-2*gamma)*dt)/(1+gamma*dt)**2
        np.testing.assert_allclose(u[-1],exact_step,rtol=0.,atol=2e-14)
        np.testing.assert_allclose(s[-1],exact_step,rtol=0.,atol=2e-14)
        if dt==1000.:
            assert abs(s[-1])<.005


def test_sdirk2_damps_fast_mode_of_a_light_contact_sheet():
    from scipy.linalg import eigh
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    r = np.array([.4,.7,1.])
    center = (2/3)*np.diff(r**3)/np.diff(r**2)
    g = 2*np.pi/np.array([np.log(r[1]/center[0]),np.log(center[1]/r[1])])
    mass = np.diag([np.pi*(r[1]**2-r[0]**2),2*np.pi*r[1]*1e-4,np.pi*(r[2]**2-r[1]**2)])
    incidence = np.diff(np.eye(3),axis=0)
    rates,modes = eigh(incidence.T@np.diag(g)@incidence,mass)
    initial = modes[:,-1]/modes[1,-1]
    dt = 1000/rates[-1]
    u,s,qi,qo,_ = solve_transient_annular_sheet(r,1.,1.,(None,None),[0.,dt],initial[[0,2]],1,
                 initial_sheet_temperature=1.,sheet_capacity=1e-4,wall_fluxes=(0.,0.),method="sdirk2")
    assert max(abs(np.r_[u[-1],s[-1]]))<.005
    np.testing.assert_allclose(np.diag(mass)*(np.r_[u[-1,0],s[-1],u[-1,1]]-initial),
                               -np.diff(np.r_[qi[0],qo[0]]),atol=1e-15)


@pytest.mark.parametrize("method",["theta","sdirk2"])
@pytest.mark.parametrize("capacity",[0.,.27])
def test_interval_temperatures_close_each_modal_balance_at_switch(method,capacity):
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    r = np.array([.4,.52,.7,.96])
    k = np.array([1.2,1.6,2.3])
    t = np.array([0.,.033,.12,.17])
    source = np.array([[0.,1.],[2.,2.],[-1.,.3]])
    ambient = .3+.6*t
    mode,h,ks,res,frac = 2,.9,.3,.27,.4
    u,s,qi,qo,ext,avg = solve_transient_annular_sheet(r,k,1.,(None,None),t,[.3,.5,-.2],2,
                 wall_fluxes=(0.,0.),sheet_capacity=capacity,initial_sheet_temperature=.7,
                 sheet_film=h,ambient=ambient,sheet_conductivity=ks,azimuthal_mode=mode,resistance=res,
                 inner_resistance_fraction=frac,sheet_source_intervals=source,method=method,return_interval_temperatures=True)
    dt = np.diff(t)
    circum = 2*np.pi*r[2]
    angular = 2*np.pi*k*mode**2*np.log(r[1:]/r[:-1])
    np.testing.assert_allclose(np.diff(u,axis=0)*np.pi*np.diff(r**2)+dt[:,None]*avg["cell"]*angular,
                               -np.column_stack((np.diff(qi),np.diff(qo))),atol=3e-14)
    np.testing.assert_allclose(ext,dt*circum*h*(avg["sheet"]-(ambient[:-1]+ambient[1:])/2),atol=3e-14)
    np.testing.assert_allclose(circum*(capacity*np.diff(s)+dt*(ks*(mode/r[2])**2*avg["sheet"]-source.mean(1))),
                               qi[:,-1]-qo[:,0]-ext,atol=3e-14)
    np.testing.assert_allclose(avg["contact"][:,0]-avg["contact"][:,1],
                               res*(frac*qi[:,-1]+(1-frac)*qo[:,0])/(dt*circum),atol=2e-14)


@pytest.mark.parametrize("method",["theta","sdirk2"])
@pytest.mark.parametrize("cs",[0.,.43])
def test_independent_annular_source_clock_keeps_one_sided_powers(method,cs):
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    times = np.array([0.,.1,.2,.4])
    clock = np.array([0.,.045,.1,.32,.4])
    full = np.union1d(times,clock)
    source = np.array([[.5,.7],[-.3,.1],[1.,.2],[-.4,-.6]])
    body = source[:,:,None]*np.array([1.,-2.,.3])
    options = dict(method=method,sheet_capacity=cs,initial_sheet_temperature=.6,return_contact_temperatures=True,
                   return_interval_temperatures=True,return_wall_temperatures=True,resistance=.25,sheet_film=.7)
    def run(t,drive,bulk,grid=None):
        return solve_transient_annular_sheet([.4,.5,.7,.93],[1.,3.,2.],1.2,(.4+t,-.3*t),t,.5,2,
                    sheet_source_intervals=drive,source_intervals=bulk,ambient=.3+t,
                    source_times=grid,**options)
    result = run(times,source,body,clock)
    explicit = []
    for left,right in zip(full,full[1:]):
        i = np.searchsorted(clock,(left+right)/2)-1
        explicit.append(np.interp([left,right],clock[i:i+2],source[i]))
    explicit = np.array(explicit)
    reference = run(full,explicit,explicit[:,:,None]*np.array([1.,-2.,.3]))
    keep = np.searchsorted(full,times)
    for i in (0,1,5,6):
        np.testing.assert_allclose(result[i],reference[i][keep],rtol=0.,atol=2e-14)
    for i in (2,3,4):
        np.testing.assert_allclose(result[i],np.add.reduceat(reference[i],keep[:-1],axis=0),atol=2e-14)
    for key,field in reference[-1].items():
        means = [np.average(field[i:j],axis=0,weights=np.diff(full)[i:j]) for i,j in zip(keep,keep[1:])]
        np.testing.assert_allclose(result[-1][key],means,atol=2e-14)


def test_wall_power_pulse_inside_output_interval_and_right_limit_surface():
    from tensorfvm.cht.annulus import solve_transient_annular_sheet
    edges = np.array([.4,.6,.8,1.1])
    times = np.array([0.,.12,.3])
    clock = np.array([0.,.035,.12,.2,.3])
    powers = np.array([0.,1.3,-.7,0.])
    cap = .45
    u,s,qi,qo,external,surface = solve_transient_annular_sheet(edges,1.7,1.2,(None,None),times,.2,1,
        wall_fluxes=(np.repeat(powers[:,None],2,axis=1),0.),wall_times=clock,
        initial_sheet_temperature=.2,sheet_capacity=cap,resistance=.2,return_wall_temperatures=True,
        method="sdirk2")
    exact_heat = 2*np.pi*.4*np.array([1.3*(.12-.035),-.7*(.2-.12)])
    total = u@(1.2*np.pi*np.diff(edges**2))+s*2*np.pi*.6*cap
    np.testing.assert_allclose(qi[:,0],exact_heat,atol=1e-14)
    np.testing.assert_allclose(np.diff(total),exact_heat,atol=2e-14)
    np.testing.assert_array_equal(qo[:,-1],0.)
    np.testing.assert_array_equal(external,0.)
    center = (2/3)*(.6**3-.4**3)/(.6**2-.4**2)
    np.testing.assert_allclose(surface[:,0]-u[:,0],.4*np.log(center/.4)/1.7*np.array([0.,-.7,0.]),atol=2e-14)


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("flux_side", [0, 1])
def test_harmonic_flux_and_film_annulus_against_bessel_solution(mode, flux_side, record_property):
    from scipy.special import iv, ivp, kv, kvp

    k, capacity, omega, film = 1.6, 2.1, 3.2, 2.7
    wave = np.sqrt(1j * omega * capacity / k)
    # Both independent radial modes are regular on the hollow annulus.
    def temperature(r):
        return (1 + .3j)*iv(mode, wave*r) + (.2 - .1j)*kv(mode, wave*r)

    def radial_flux(r):
        return -k*wave*((1 + .3j)*ivp(mode, wave*r) + (.2 - .1j)*kvp(mode, wave*r))

    ends = np.array([0.5, 1.2])
    inward = radial_flux(ends) * [1, -1]
    other = 1 - flux_side
    fluxes, walls, films = [None, None], [None, None], [None, None]
    fluxes[flux_side] = inward[flux_side]
    films[other] = film
    walls[other] = temperature(ends[other]) + inward[other]/film
    errors = []
    for n in (16, 32, 64):
        r = np.geomspace(*ends, n + 1)
        centers = (2/3)*np.diff(r**3)/np.diff(r**2)
        field, rates = solve_annular_conduction(
            r, k, capacity, walls, omega, wall_fluxes=fluxes, azimuthal_mode=mode,
            film_coefficients=films, cell_location="volume_centroid", distributed_source=True)
        field_error = np.max(np.abs(field-temperature(centers)))
        rate_error = np.max(np.abs(rates-2*np.pi*r*radial_flux(r)))
        sink = 1j*omega*capacity*np.pi*np.diff(r**2) + 2*np.pi*k*mode**2*np.diff(np.log(r))
        residual = np.max(np.abs(np.diff(rates) + sink*field))
        record_property(f"max_temperature_error_{n}_K", float(field_error))
        record_property(f"max_radial_rate_error_{n}_W_per_m", float(rate_error))
        record_property(f"max_shell_balance_error_{n}_W_per_m", float(residual))
        assert residual < 1e-11
        errors.append((field_error, rate_error))
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 3.7 for a, b in zip(coarse, fine))


@pytest.mark.parametrize("flux_side", [None, 0, 1], ids=["fixed-walls", "inner-flux", "outer-flux"])
@pytest.mark.parametrize("axial,sheet_k,axial_k,orthotropic,reservoir_film,fraction", [
    (0., 0., None, False, 0., .3), (-1.25, .35, None, False, 0., .3),
    (-1.25, .35, 1.2, False, 0., .3), (-1.25, .35, 0., False, 0., .3),
    (-1.25, .35, 1.2, True, 0., .3),
    pytest.param(-1.25, .35, 1.2, True, 2.4, .3, id="reservoir-driven"),
    pytest.param(-1.25, .35, 1.2, True, 2.4, 0., id="bonded-inner"),
    pytest.param(-1.25, .35, 1.2, True, 2.4, 1., id="bonded-outer"),
])
def test_harmonic_two_material_sheet_traces_against_bessel_modes(
        axial, sheet_k, axial_k, orthotropic, reservoir_film, fraction, flux_side, record_property):
    from scipy.special import iv, ivp, kv, kvp
    from tensorfvm.cht.annulus import solve_annular_sheet

    radius, omega, mode = 0.8, 3.2, 2
    k = np.array([1.4, 2.3])
    capacity = np.array([1.8, 1.1])
    kt = np.array([.75, 3.8]) if orthotropic else k
    kz = np.array([2.9, 1.6]) if orthotropic else k
    order = mode*np.sqrt(kt/k)
    wave = np.sqrt((kz*axial**2 + 1j*omega*capacity)/k)
    sheet = 0.65 + 0.2j
    q = np.array([1.1-0.4j, -0.25+0.5j])  # Increasing-r heat flux on each side.
    resistance, surface_capacity = 0.25, 0.4
    traces = sheet + resistance*np.array([fraction, -(1-fraction)])*q
    surface_response = 1j*omega*surface_capacity + sheet_k*(mode/radius)**2 + (sheet_k if axial_k is None else axial_k)*axial**2
    required_load = surface_response*sheet + q[1] - q[0]
    # Drive the same manufactured field entirely through reservoir exchange.
    ambient = sheet + required_load/reservoir_film if reservoir_film else 0.
    generation = 0. if reservoir_film else required_load

    def basis(side, r):
        z = wave[side]*r
        return np.stack((iv(order[side], z), kv(order[side], z)), axis=-1)

    def flux_basis(side, r):
        z = wave[side]*r
        return -k[side]*wave[side]*np.stack((ivp(order[side], z), kvp(order[side], z)), axis=-1)

    coeff = [np.linalg.solve([basis(i, radius), flux_basis(i, radius)], [traces[i], q[i]])
             for i in range(2)]
    ends = (0.5, 1.2)
    walls = [basis(i, r) @ coeff[i] for i, r in enumerate(ends)]
    fluxes, films = [None, None], [None, None]
    if flux_side is not None:
        inward = [(1-2*i)*(flux_basis(i, r) @ coeff[i]) for i, r in enumerate(ends)]
        film_side = 1-flux_side
        fluxes[flux_side] = inward[flux_side]
        walls[flux_side] = None
        films[film_side] = 2.7
        walls[film_side] += inward[film_side]/films[film_side]
    errors = []
    for n in (12, 24, 48):
        parts = (np.geomspace(0.5, radius, n+1), np.geomspace(radius, 1.2, n+1))
        edges = np.r_[parts[0], parts[1][1:]]
        field, inner, outer, bulk_sheet, observed = solve_annular_sheet(
            edges, np.repeat(k, n), np.repeat(capacity, n), walls, omega, n,
            wall_fluxes=fluxes, film_coefficients=films,
            circumferential_conductivity=np.repeat(kt, n) if orthotropic else None,
            axial_conductivity=np.repeat(kz, n) if orthotropic else None,
            azimuthal_mode=mode, resistance=resistance, inner_resistance_fraction=fraction,
            axial_wavenumber=axial, sheet_conductivity=sheet_k, sheet_axial_conductivity=axial_k,
            sheet_capacity=surface_capacity, sheet_source=generation, sheet_film=reservoir_film,
            ambient=ambient, distributed_source=True,
            cell_location="volume_centroid", return_contact_temperatures=True)
        exact_field, exact_rates = [], []
        for side, r in enumerate(parts):
            centers = (2/3)*np.diff(r**3)/np.diff(r**2)
            exact_field.extend(basis(side, centers) @ coeff[side])
            exact_rates.extend(2*np.pi*r*(flux_basis(side, r) @ coeff[side]))
        row = [max(abs(field-exact_field)), max(abs(np.r_[inner, outer]-exact_rates)),
               max(abs(observed-traces)), abs(bulk_sheet-sheet)]
        for name, error in zip(("field_K", "radial_rate_W_per_m", "trace_K", "sheet_K"), row):
            record_property(f"max_error_{n}_{name}", float(error))
        interface_error = abs(np.array([inner[-1], outer[0]])/(2*np.pi*radius)-q)
        for side, error in zip(("inner", "outer"), interface_error):
            record_property(f"contact_{side}_{n}_W_per_m2", float(error))
        row.extend(interface_error)
        if fraction in (0., 1.):
            bonded = int(fraction)
            continuity_error = abs(observed[bonded]-bulk_sheet)
            jump_error = abs((observed[1-bonded]-bulk_sheet)-(traces[1-bonded]-sheet))
            record_property(f"bonded_continuity_{n}_K", float(continuity_error))
            record_property(f"resistive_jump_error_{n}_K", float(jump_error))
            assert continuity_error < 1e-13
            row.append(jump_error)
        exchange = reservoir_film*(ambient-bulk_sheet)
        if reservoir_film:
            exchange_error = abs(exchange-required_load)
            record_property(f"reservoir_{n}_W_per_m2", float(exchange_error))
            row.append(exchange_error)
        residual = inner[-1]-outer[0]+2*np.pi*radius*(generation+exchange-surface_response*bulk_sheet)
        record_property(f"balance_{n}_W_per_m", float(abs(residual)))
        assert abs(residual) < 1e-11
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 3.7 for a, b in zip(coarse, fine))


def test_massless_insulated_shell_requires_datum_at_nonzero_frequency():
    from tensorfvm.cht.annulus import solve_annular_sheet

    r = [.3, .5, .7, .9]
    walls = (None, None)
    opts = dict(wall_fluxes=(0., 0.))
    with pytest.raises(ValueError, match="reference"):
        solve_annular_conduction(r, 1., 0., walls, 4., **opts)
    with pytest.raises(ValueError, match="mean temperature"):
        solve_annular_sheet(r, 1., 0., walls, 4., 1, **opts)
    field, rates = solve_annular_conduction(r, 1., 0., walls, 4., mean_temperature=.4j, **opts)
    np.testing.assert_allclose(field, .4j)
    np.testing.assert_allclose(rates, 0., atol=1e-13)
    # One storing sheet anchors the harmonic zero mode even with all bulk cells massless.
    field, _, _, sheet = solve_annular_sheet(r, 1., 0., walls, 4., 1, sheet_capacity=.5, sheet_source=1., **opts)
    np.testing.assert_allclose(field, -.5j, atol=1e-13)
    np.testing.assert_allclose(sheet, -.5j, atol=1e-13)


def test_mixed_cell_storage_in_unanchored_material():
    edges = np.geomspace(.4, 1.1, 9)
    cv = np.array([0., .7, 0., 1.4, 0., .6, 1.2, 0.])
    exact = .3+.6j
    field, rates = solve_annular_conduction(edges, 2., cv, (None, None), 8.,
        wall_fluxes=(0., 0.), source=8j*cv*exact, distributed_source=True)
    np.testing.assert_allclose(field, exact, atol=1e-13)
    np.testing.assert_allclose(rates, 0., atol=1e-13)


@pytest.mark.parametrize("wave,mode,omega", [(1.75, 0, 0.), (-1.75, 2, 3.)])
def test_axial_fourier_mode_matches_bessel_field_and_face_rates(wave, mode, omega, record_property):
    from scipy.special import iv, ivp

    k, capacity = 2., 1.5
    decay = np.sqrt(wave**2 + 1j * omega * capacity / k)
    previous = None
    for cells in (16, 32, 64):
        r = np.linspace(.4, 1., cells + 1)
        centers = .5 * (r[1:] + r[:-1])
        expected_rate = -2 * np.pi * r * k * decay * ivp(mode, decay * r)
        # Prescribe the inner inward flux and outer temperature independently.
        field, rate = solve_annular_conduction(
            r, k, capacity, (None, iv(mode, decay * r[-1])), omega,
            wall_fluxes=(expected_rate[0] / (2 * np.pi * r[0]), None),
            axial_wavenumber=wave, azimuthal_mode=mode, distributed_source=True)
        errors = np.array([max(abs(field - iv(mode, decay * centers))), max(abs(rate - expected_rate))])
        for quantity, error in zip(("temperature_K", "rate_W_per_m"), errors):
            record_property(f"n{cells}_{quantity}", float(error))
        if previous is not None:
            assert np.all(np.log2(previous / errors) > 1.8)
        previous = errors


def test_axial_mode_with_insulated_walls_balances_source_without_reference():
    r = [.3, .45, .8, 1.]
    k = np.array([1., 3., 2.])
    field, rate = solve_annular_conduction(
        r, k, 0., (None, None), 0., wall_fluxes=(0., 0.), axial_wavenumber=2.,
        source=k * 4 * (1 + .5j), contact_resistance=.2)
    np.testing.assert_allclose(field, 1 + .5j, atol=1e-14, rtol=0)
    np.testing.assert_allclose(rate, 0., atol=1e-13)
    for invalid in (np.inf, 1j, [1.]):
        with pytest.raises(ValueError, match="axial_wavenumber"):
            solve_annular_conduction(r, k, 1., (0., 0.), 0., axial_wavenumber=invalid)


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
@pytest.mark.parametrize("circumferential, axial_k", [(.3, None), (1.8, .3)])
def test_axial_decay_across_storing_sheet(method, circumferential, axial_k, record_property):
    from tensorfvm.cht.annulus import solve_transient_annular_sheet

    # Equal diffusivities give the same exponential decay throughout both
    # materials and the sheet, with no radial heat crossing either contact.
    k = np.array([.8, .8, 1.7, 1.7])
    axial, diffusivity = 2., .6
    previous = None
    for steps in (10, 20, 40):
        times = np.linspace(0., .5, steps + 1)
        field, sheet, inner, outer, _ = solve_transient_annular_sheet(
            [.3, .38, .5, .73, 1.], k, k/diffusivity, (None, None), times, 1., 2,
            sheet_capacity=.5, sheet_conductivity=circumferential, sheet_axial_conductivity=axial_k,
            axial_wavenumber=axial,
            initial_sheet_temperature=1., resistance=.4, wall_fluxes=(0., 0.), method=method)
        exact = np.exp(-diffusivity*axial**2*times)
        error = max(np.max(abs(field - exact[:, None])), np.max(abs(sheet - exact)))
        record_property(f"steps{steps}_temperature_K", float(error))
        assert max(np.max(abs(inner)), np.max(abs(outer))) < 1e-13
        if previous is not None:
            assert previous/error > 3.7
        previous = error


def test_axial_sheet_steady_mode_between_insulated_walls():
    from tensorfvm.cht.annulus import solve_annular_sheet

    k = np.array([1., 3., 2.])
    axial, amplitude, sheet_k = 1.5, 2., .7
    field, inner, outer, sheet = solve_annular_sheet(
        [.3, .45, .8, 1.], k, 0., (None, None), 0., 1, axial_wavenumber=axial,
        wall_fluxes=(0., 0.), source=k*axial**2*amplitude, sheet_conductivity=sheet_k,
        sheet_source=sheet_k*axial**2*amplitude)
    np.testing.assert_allclose(np.r_[field, sheet], amplitude, atol=1e-13, rtol=0)
    np.testing.assert_allclose(np.r_[inner, outer], 0., atol=1e-13)


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_transient_orthotropic_annulus_mixed_mode_decay(method, record_property):
    from tensorfvm.cht.annulus import solve_transient_annular_sheet

    # Choose per-cell heat capacities so the shell-integrated angular and
    # axial losses give the same decay rate everywhere, including the sheet.
    edges = np.array([.3, .42, .55, .8, 1.])
    radius = edges[2]
    kt, kz = np.array([.4, .8, .6, 1.]), np.array([1.7, 1.2, 2., .9])
    mode, axial, decay = 2, -1.5, 3.
    mean_inverse_r2 = 2*np.log(edges[1:]/edges[:-1])/np.diff(edges**2)
    capacity = (kt*mode**2*mean_inverse_r2 + kz*axial**2)/decay
    sheet_capacity = (.35*(mode/radius)**2 + .8*axial**2)/decay
    previous = None
    for steps in (10, 20, 40):
        times = np.linspace(0., .5, steps+1)
        field, sheet, inner, outer, _ = solve_transient_annular_sheet(
            edges, [1., 1., 3., 3.], capacity, (None, None), times, 1., 2,
            circumferential_conductivity=kt, axial_conductivity=kz,
            sheet_conductivity=.35, sheet_axial_conductivity=.8, sheet_capacity=sheet_capacity,
            azimuthal_mode=mode, axial_wavenumber=axial, initial_sheet_temperature=1.,
            wall_fluxes=(0., 0.), method=method)
        exact = np.exp(-decay*times)
        error = max(np.max(abs(field-exact[:, None])), np.max(abs(sheet-exact)))
        record_property(f"steps{steps}_max_temperature_error_K", float(error))
        assert max(np.max(abs(inner)), np.max(abs(outer))) < 1e-13
        if previous is not None:
            assert previous/error > 3.7
        previous = error


@pytest.mark.parametrize("insulated", [None, 0, 1], ids=["two-films", "inner-insulated", "outer-insulated"])
@pytest.mark.parametrize("location", ["midpoint", "volume_centroid"])
def test_generated_heat_partition_in_two_material_annulus(location, insulated, record_property):
    from tensorfvm.cht.annulus import solve_annular_sheet

    # Both materials and the contact sheet generate heat.
    radius, sheet_temperature, resistance, fraction = .7, 1.2, .3, .35
    k = np.array([1.4, 2.3])
    source = np.array([1.9, .8])
    edges = np.array([.3, .39, .58, radius, .73, .91, 1.15])
    ends = edges[[0, -1]]
    generation = .8
    contact_flux = np.array([-.2, .6])
    if insulated is not None:
        # Integrate from the insulated wall to the sheet, then apply its source jump.
        q = source[insulated]*(radius**2-ends[insulated]**2)/(2*radius)
        contact_flux = q + generation*(np.arange(2)-insulated)
    traces = sheet_temperature + resistance*np.array([fraction, fraction-1])*contact_flux
    face = 3
    films = np.array([2.1, 3.4])
    # Integrate d(r*q)/dr = r*S, then integrate dT/dr = -q/k from the sheet.
    def exact(side, r):
        constant = radius*contact_flux[side] - source[side]*radius**2/2
        flux = source[side]*r/2 + constant/r
        temperature = traces[side] - (source[side]*(r**2-radius**2)/4
                                      + constant*np.log(r/radius))/k[side]
        return temperature, flux

    surfaces, flux = exact(np.arange(2), ends)
    ambient = surfaces + np.array([1., -1.])*flux/films
    if insulated is not None:
        films[insulated] = 0.
    centers = (edges[:-1]+edges[1:])/2
    if location == "volume_centroid":
        centers = (2/3)*np.diff(edges**3)/np.diff(edges**2)
    sides = (centers > radius).astype(int)
    field, inner, outer, sheet, observed = solve_annular_sheet(
        edges, k[sides], 1., ambient, 0., face, resistance=resistance,
        inner_resistance_fraction=fraction, sheet_source=generation, source=source[sides],
        film_coefficients=films, cell_location=location, distributed_source=True,
        return_contact_temperatures=True)
    field_error = np.max(abs(field-exact(sides, centers)[0]))
    wall_heat = np.array([-inner[0], outer[-1]])
    target = 2*np.pi*ends*flux*[-1, 1]
    supplied = np.pi*np.diff(edges**2) @ source[sides] + 2*np.pi*radius*generation
    if insulated is not None:
        np.testing.assert_allclose(wall_heat[insulated], 0., atol=1e-13)
        np.testing.assert_allclose(wall_heat[1-insulated], supplied, rtol=0, atol=2e-12)
    balance = abs(wall_heat.sum()-supplied)
    np.testing.assert_allclose(np.r_[sheet, observed], np.r_[sheet_temperature, traces], rtol=0, atol=2e-13)
    np.testing.assert_allclose(wall_heat, target, rtol=0, atol=2e-12)
    assert field_error < 2e-13 and balance < 2e-12
    record_property("field_error_K", float(field_error))
    record_property("wall_rate_error_W_per_m", float(max(abs(wall_heat-target))))
    record_property("balance_W_per_m", float(balance))
    for label, value in zip(("inner", "outer"), wall_heat):
        record_property(f"{label}_outward_W_per_m", float(value.real))


@pytest.mark.parametrize("film", [.4, 3.5])
def test_insulated_annulus_rejects_all_generated_heat_at_sheet(film, record_property):
    from tensorfvm.cht.annulus import solve_annular_sheet

    radius, ambient, resistance, fraction = .7, .35, .3, .35
    edges = np.array([.3, .39, .58, radius, .73, .91, 1.15])
    face = 3
    source = np.array([1.9, .8])
    sheet_source = .8
    circumference = 2*np.pi*radius
    # Each insulated layer must deliver its entire generation to the sheet.
    layer_heat = np.pi*np.diff(edges[[0, face, -1]]**2)*source
    expected_rate = layer_heat*np.array([1., -1.])
    total = layer_heat.sum() + circumference*sheet_source
    expected_sheet = ambient + total/(circumference*film)
    expected_traces = expected_sheet + resistance*np.array([fraction, fraction-1])*expected_rate/circumference
    _, inner, outer, sheet, traces = solve_annular_sheet(
        edges, np.repeat([1.4, 2.3], 3), 1., (None, None), 0., face,
        wall_fluxes=(0., 0.), resistance=resistance, inner_resistance_fraction=fraction,
        sheet_source=sheet_source, source=np.repeat(source, 3), sheet_film=film,
        ambient=ambient, distributed_source=True, return_contact_temperatures=True)
    exchange = circumference*film*(sheet-ambient)
    rate_error = max(abs(np.array([inner[-1], outer[0]])-expected_rate))
    np.testing.assert_allclose([inner[0], outer[-1]], 0., rtol=0, atol=1e-13)
    np.testing.assert_allclose(traces, expected_traces, rtol=0, atol=2e-13)
    assert rate_error < 2e-12 and abs(exchange-total) < 2e-12 and abs(sheet-expected_sheet) < 2e-13
    record_property("contact_rate_error_W_per_m", float(rate_error))
    record_property("reservoir_rate_W_per_m", float(exchange.real))
    record_property("balance_error_W_per_m", float(abs(exchange-total)))
    record_property("sheet_temperature_K", float(sheet.real))
    record_property("sheet_temperature_error_K", float(abs(sheet-expected_sheet)))


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_insulated_annular_sheet_cooldown_after_steady_heating(method, record_property):
    from scipy.linalg import expm
    from tensorfvm.cht.annulus import solve_transient_annular_sheet

    edges = np.array([.3, .7, 1.15])
    radius = edges[1]
    area = 2*np.pi*radius
    volumes = np.pi*np.diff(edges**2)
    centers = (2/3)*np.diff(edges**3)/np.diff(edges**2)
    # Two cell stores couple to the sheet through radial conduction and contact resistance.
    conductance = 2*np.pi/(abs(np.log(centers/radius))/[1.4, 2.3] + .3*np.array([.35, .65])/radius)
    film = 2.4
    mass = np.r_[volumes*np.array([1.5, 2.]), area*.4]
    operator = np.diag(np.r_[conductance, conductance.sum()+area*film])
    operator[:2, 2] = operator[2, :2] = -conductance
    # Start from the discrete steady state for two bulk heaters and a sheet heater.
    # After shutoff the homogeneous three-store system decays toward ambient.
    excess = np.linalg.solve(operator, np.r_[volumes*[1.9, .8], area*.8])
    ambient, duration = .35, .6
    exact = expm(-duration*operator/mass[:, None]) @ excess
    reference_heat = mass @ (excess-exact)
    previous = None
    for steps in (12, 24, 48):
        times = np.linspace(0., duration, steps+1)
        field, sheet, inner, outer, reservoir = solve_transient_annular_sheet(
            edges, [1.4, 2.3], [1.5, 2.], (None, None), times, ambient+excess[:2], 1,
            initial_sheet_temperature=ambient+excess[2], sheet_capacity=.4,
            resistance=.3, inner_resistance_fraction=.35, wall_fluxes=(0., 0.),
            sheet_film=film, ambient=ambient, method=method)
        state = np.column_stack((field, sheet))
        stored = np.diff(state, axis=0)*mass
        into_stores = np.column_stack((-inner[:, -1], outer[:, 0], inner[:, -1]-outer[:, 0]-reservoir))
        residual = np.max(abs(stored-into_stores))
        errors = np.array([max(abs(state[-1]-ambient-exact)), abs(reservoir.sum()-reference_heat)])
        record_property(f"steps{steps}_temperature_error_K", float(errors[0]))
        record_property(f"steps{steps}_reservoir_heat_error_J_per_m", float(errors[1]))
        record_property(f"steps{steps}_local_balance_J_per_m", float(residual))
        assert residual < 2e-13 and np.all(reservoir > 0)
        if previous is not None:
            assert np.all(previous/errors > 3.7)
        previous = errors


def test_insulated_annular_cooldown_against_continuous_radial_mode(record_property):
    from scipy.special import jv, yv
    from tensorfvm.cht.annulus import solve_transient_annular_sheet

    ends = np.array([.3, 1.15])
    radius, decay, sheet_capacity = .7, .8, .4
    conductivity, capacity = np.array([1.4, 2.3]), np.array([1.5, 2.])
    wave = np.sqrt(decay*capacity/conductivity)
    contact = .3*np.array([.35, -.65])

    # Each radial eigenfunction has zero derivative at its insulated wall.
    def mode(side, r):
        z = wave[side]*r
        wall = wave[side]*ends[side]
        a, b = yv(1, wall), -jv(1, wall)
        value = a*jv(0, z) + b*yv(0, z)
        flux = conductivity[side]*wave[side]*(a*jv(1, z) + b*yv(1, z))
        return value, flux

    sides = np.arange(2)
    values, flux = mode(sides, radius)
    # Normalize both fields to a unit sheet temperature via their contact jumps.
    amplitude = 1/(values-contact*flux)
    contact_flux = amplitude*flux
    film = contact_flux[0]-contact_flux[1]+decay*sheet_capacity
    assert film > 0
    duration = .6
    integrated = -np.expm1(-decay*duration)/decay
    exact_heat = 2*np.pi*radius*integrated*np.r_[contact_flux, film]
    previous = None
    for n in (8, 16, 32):
        edges = np.r_[np.geomspace(ends[0], radius, n+1), np.geomspace(radius, ends[1], n+1)[1:]]
        centers = (2/3)*np.diff(edges**3)/np.diff(edges**2)
        side = np.repeat(sides, n)
        initial = amplitude[side]*mode(side, centers)[0]
        times = np.linspace(0., duration, 8*n+1)
        field, sheet, inner, outer, reservoir = solve_transient_annular_sheet(
            edges, conductivity[side], capacity[side], (None, None), times, initial, n,
            initial_sheet_temperature=1., sheet_capacity=sheet_capacity, resistance=.3,
            inner_resistance_fraction=.35, wall_fluxes=(0., 0.), sheet_film=film, method="sdirk2")
        final = np.r_[field[-1], sheet[-1]]
        temperature_error = max(abs(final-np.r_[initial, 1.]*np.exp(-decay*duration)))
        heat = np.array([inner[:, -1].sum(), outer[:, 0].sum(), reservoir.sum()])
        errors = np.r_[temperature_error, abs(heat-exact_heat)]
        for label, error in zip(("temperature_K", "inner_J_per_m", "outer_J_per_m", "reservoir_J_per_m"), errors):
            record_property(f"n{n}_{label}", float(error))
        if previous is not None:
            assert np.all(previous/errors > 3.5)
        previous = errors
