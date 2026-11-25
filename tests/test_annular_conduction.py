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
