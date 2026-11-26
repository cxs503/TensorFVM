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


def test_vertical_layered_conduction_matches_rotated_horizontal_solution():
    widths = [.2, .3, .5]
    k = np.array([1., 4., 2.])
    initial = np.array([2., 0., 1.])
    hot = {"type": "robin", "values": 3., "film_coefficient": 2.}
    cold = {"type": "dirichlet", "values": -1.}
    times = [0., .01, .1]
    horizontal = solve_transient_conduction(widths, [1.], k, 1., {"west": hot, "east": cold},
                                            times, initial, contact_resistance_x=.1)
    vertical = solve_transient_conduction([1.], widths, 9., 1., {"south": hot, "north": cold},
                                          times, initial[:, None], conductivity_y=k[:, None],
                                          contact_resistance_y=.1)
    np.testing.assert_allclose(vertical[:, :, 0], horizontal[:, 0, :], atol=1e-13, rtol=0)


@pytest.mark.parametrize("ky", [0., -1., np.nan, np.inf])
def test_invalid_directional_conductivity(ky):
    with pytest.raises(ValueError, match="conductivity_y"):
        solve_transient_conduction([1.], [1.], 1., 1., {}, [0., 1.], 0., conductivity_y=ky)


@pytest.mark.parametrize("theta", [.5, .75, 1.])
def test_mixed_wall_power_history_closes_every_nonuniform_step(theta):
    dx, dy = np.array([.2, .8]), np.array([.4, .6])
    bc = {"west": {"types": ["dirichlet", "adiabatic"], "values": [2., 0.]},
          "north": {"types": ["robin", "neumann"], "values": [3., -2.],
                    "film_coefficient": 4.}}
    times = np.array([0., .001, .05, .1])
    temp, walls = solve_transient_conduction(dx, dy, 2., 3., bc, times, 1., source=.7,
                                             conductivity_y=5., theta=theta, return_wall_power=True)
    np.testing.assert_allclose(walls["west"][0], [8., 0.])
    np.testing.assert_allclose(walls["north"][0], [2.*dx[0]/(.3/5.+1/4.), -2.*dx[1]])
    for side in ("south", "east"):
        np.testing.assert_array_equal(walls[side], np.zeros((len(times), 2)))
    net = sum(power.sum(axis=1) for power in walls.values()) + .7
    stored = (np.diff(temp, axis=0)*3.*np.outer(dy, dx)).sum(axis=(1, 2))
    np.testing.assert_allclose(stored, np.diff(times)*((1-theta)*net[:-1]+theta*net[1:]), atol=1e-13)


@pytest.mark.parametrize("theta", [.5, .75, 1.])
def test_time_dependent_face_signals_match_each_step_storage(theta):
    time = np.array([.1, .105, .13, .2])
    calls = []
    def north(t):
        calls.append(t)
        return [3.*t, 1.-t]
    bc = {"north": {"types": ["neumann", "robin"], "values": north, "film_coefficient": 2.},
          "south": {"type": "dirichlet", "values": lambda t: t*t}}
    t, power = solve_transient_conduction([.4, .6], [1.], 1., 2., bc, time, 0.,
                                          theta=theta, return_wall_power=True)
    assert calls == time.tolist()
    np.testing.assert_allclose(power["north"][:,0], .4*3.*time)
    np.testing.assert_allclose(power["north"][:,1], .6*(1.-time-t[:,0,1]))
    np.testing.assert_allclose(power["south"], 2*np.array([.4,.6])*(time[:,None]**2-t[:,0]))
    net = sum(side.sum(axis=1) for side in power.values())
    stored = 2*np.diff(t[:,0], axis=0) @ np.array([.4,.6])
    np.testing.assert_allclose(stored, np.diff(time)*((1-theta)*net[:-1]+theta*net[1:]), atol=1e-14)


@pytest.mark.parametrize("theta", [.5, 1.])
def test_periodic_two_rows_integrate_seam_and_interior_separately(theta):
    time = np.array([0., .01, .03, .1])
    dy = np.array([.3, .7])
    ky = np.array([[1.], [4.]])
    capacity = np.array([[2.], [3.]])
    source = np.array([[2.], [-1.]])
    t, wall = solve_transient_conduction([1.], dy, 1., capacity, {}, time, [[1.], [3.]],
                    theta=theta, conductivity_y=ky, contact_resistance_y=.4,
                    source=source, periodic_y=True, return_wall_power=True)
    resistance = .15+.35/4
    delta = t[:, 1, 0]-t[:, 0, 0]
    np.testing.assert_allclose(wall["south"][:, 0], delta/resistance)
    np.testing.assert_array_equal(wall["north"], -wall["south"])
    into_lower = wall["south"][:, 0]+delta/(resistance+.4)
    power = np.column_stack((into_lower, -into_lower))+source.ravel()*dy
    stored = np.diff(t[:, :, 0], axis=0)*(dy*capacity.ravel())
    np.testing.assert_allclose(stored, np.diff(time)[:, None]*((1-theta)*power[:-1]+theta*power[1:]))


@pytest.mark.parametrize("theta", [.5, 1.])
def test_varying_generation_integrates_over_nonuniform_steps(theta):
    time = np.array([.1, .11, .15, .4])
    volume = np.outer([.3, .7], [.25, .75])
    capacity = np.array([[3., 1.], [2., 4.]])
    pattern = np.array([[1., 3.], [-4., 2.]])
    samples = []
    def heating(t):
        samples.append(t)
        return (1+t**2)*pattern

    history = solve_transient_conduction([.25,.75], [.3,.7], 2., capacity, {}, time, 0.,
                        source=heating, periodic_y=True, theta=theta)
    np.testing.assert_array_equal(samples, time)
    stored = np.sum(np.diff(history, axis=0)*capacity*volume, axis=(1,2))
    power = (1+time**2)*(pattern*volume).sum()
    np.testing.assert_allclose(stored, np.diff(time)*((1-theta)*power[:-1]+theta*power[1:]))
    with pytest.raises(ValueError, match="source must be finite"):
        solve_transient_conduction([1.], [1.], 1., 1., {}, time, 0., source=lambda t: np.nan)


def test_initial_wall_heat_pulse_with_insulated_ambient():
    # Cell capacity 3, wall capacity 1, half-cell conductance 4.
    time = np.linspace(0., .6, 601)
    bc = {"west":dict(type="robin", values=0., film_coefficient=0.,
                      wall_areal_heat_capacity=1., initial_wall_temperature=4.)}
    u, powers = solve_transient_conduction([1.],[1.],2.,3.,bc,time,0.,return_wall_power=True)
    cell = u[:,0,0]
    flux = powers["west"][:,0]
    wall = cell + flux/4
    decay = np.exp(-16*time/3)
    np.testing.assert_allclose(cell, 1-decay, atol=1e-6)
    np.testing.assert_allclose(wall, 1+3*decay, atol=3e-6)
    np.testing.assert_allclose(3*cell+wall, 4., atol=1e-12)
    np.testing.assert_allclose(3*np.diff(cell), np.diff(time)*(flux[:-1]+flux[1:])/2, atol=1e-13)


@pytest.mark.parametrize("theta", [.5,1.])
def test_mixed_wall_storage_tracks_nonuniform_steps_and_forcing(theta):
    dx, dy = np.array([.25,.75]), np.array([.4,.6])
    time = np.array([0.,.01,.03,.08,.11])
    volume = np.outer(dy, dx)
    ambient = lambda t: np.array([2.+t, 4.-t])
    source = lambda t: 1.-2*t
    spec = dict(types=["robin","robin"], values=ambient, film_coefficient=[2.,5.],
                wall_areal_heat_capacity=[3.,0.], initial_wall_temperature=[4.,0.])
    u, power = solve_transient_conduction(dx,dy,2.,3.,{"west":spec},time,0.,source=source,
                                          theta=theta,return_wall_power=True,periodic_y=True)
    q = power["west"]
    wall = u[:,:,0]+q*dx[0]/(4*dy)
    air = np.array([ambient(t) for t in time])
    incoming = (air-wall)*np.array([2.,5.])*dy
    def integral(rate):
        return np.diff(time)*((1-theta)*rate[:-1]+theta*rate[1:])
    np.testing.assert_allclose(3*dy[0]*np.diff(wall[:,0]),integral((incoming-q)[:,0]),atol=1e-12)
    np.testing.assert_allclose(incoming[:,1],q[:,1],atol=1e-12)
    energy = 3*(u*volume).sum(axis=(1,2))
    np.testing.assert_allclose(np.diff(energy),integral(q.sum(axis=1)+source(time)),atol=1e-12)


@pytest.mark.parametrize("side", ["west","east","south","north"])
@pytest.mark.parametrize("wall_type",["robin","dirichlet"])
def test_second_order_wall_with_adjacent_material_contact(side,wall_type):
    # Prescribe an initial linear inward flux across two unequal, dissimilar cells.
    d1,d2,k1,k2,contact = .2,.63,1.7,4.3,.28
    wall, q0, slope = 2.,1.8,-.7
    mean1 = wall-(q0*d1/2+slope*d1**2/6)/k1
    interface = wall-(q0*d1+slope*d1**2/2)/k1-contact*(q0+slope*d1)
    mean2 = interface-((q0+slope*d1)*d2/2+slope*d2**2/6)/k2
    widths,k,u = np.array([d1,d2]), np.array([k1,k2]), np.array([mean1,mean2])
    if side in ("east","north"):
        widths,k,u = widths[::-1],k[::-1],u[::-1]
    area = .8
    if side in ("west","east"):
        dx,dy = widths,[area]
        opts = dict(contact_resistance_x=contact)
    else:
        dx,dy = [area],widths
        u = u[:,None]
        opts = dict(contact_resistance_y=contact,conductivity_y=k[:,None])
        k = 1.
    times = np.array([0., .002, .008, .02])
    bc = {side:dict(type="robin",values=0.,film_coefficient=0.,wall_areal_heat_capacity=3.,
                    initial_wall_temperature=wall)}
    if wall_type == "dirichlet":
        bc = {side:dict(type="dirichlet",values=wall)}
        opts["dirichlet_flux_order"] = 2
    t,power = solve_transient_conduction(dx,dy,k,1.,bc,times,u,wall_flux_order=2,
                                         return_wall_power=True,**opts)
    q = power[side][:,0]
    np.testing.assert_allclose(q[0],area*q0,atol=1e-13)
    net = (t*np.outer(dy,dx)).sum(axis=(1,2))
    np.testing.assert_allclose(np.diff(net),np.diff(times)*(q[:-1]+q[1:])/2,atol=1e-13)


@pytest.mark.parametrize("theta",[.5,1.])
def test_nonuniform_bulk_sink_balances_wall_storage(theta):
    dx,dy = np.array([.2,.8]),np.array([.4,.6])
    volume = np.outer(dy,dx)
    c = np.array([[1.,1.5],[2.,1.2]])
    loss = np.array([[0.,3.],[.4,1.7]])
    bc = {"west":dict(type="robin",values=2.,film_coefficient=1.3,wall_areal_heat_capacity=.7)}
    time = np.array([0.,.01,.05,.15])
    fields,powers = solve_transient_conduction(dx,dy,1.,c,bc,time,0.,source=.8,theta=theta,
                                                heat_loss_coefficient=loss,return_wall_power=True)
    stage = (1-theta)*fields[:-1]+theta*fields[1:]
    west = powers["west"].sum(axis=1)
    expected = (1-theta)*west[:-1]+theta*west[1:]+((.8-loss*stage)*volume).sum(axis=(1,2))
    np.testing.assert_allclose((np.diff(fields,axis=0)*volume*c).sum(axis=(1,2)),np.diff(time)*expected,atol=1e-13)


@pytest.mark.parametrize("loss",[-1.,float("inf"),float("nan"),1j])
def test_invalid_bulk_loss(loss):
    from tensorfvm.cht.energy import solve_harmonic_conduction
    with pytest.raises(ValueError,match="heat_loss_coefficient"):
        solve_transient_conduction([1.],[1.],1.,1.,{},[0.,1.],0.,heat_loss_coefficient=loss)
    with pytest.raises(ValueError,match="heat_loss_coefficient"):
        solve_harmonic_conduction([1.],[1.],1.,1.,{},3.,heat_loss_coefficient=loss)


@pytest.mark.parametrize("theta",[.5,1.])
def test_interior_power_closes_each_graded_periodic_cell(theta):
    dx,dy = np.array([.2,.3,.5]),np.array([.35,.65])
    volume = np.outer(dy,dx)
    c = np.array([[1.,2.,1.5],[2.,.5,1.]])
    times = np.array([0.,.01,.05,.12])
    initial = np.array([[1.,0.,2.],[-.4,1.3,.7]])
    source = lambda t: np.array([1.,2.,-1.])*t
    fields,walls,power = solve_transient_conduction(dx,dy,2.,c,
            {"west":dict(type="neumann",values=.3)},times,initial,source=source,
            contact_resistance_x=.17,contact_resistance_y=.12,theta=theta,conductivity_y=3.,
            periodic_y=True,heat_loss_coefficient=.4,return_wall_power=True,return_interior_power=True)
    qx = np.concatenate((walls["west"][:,:,None],power["x"],-walls["east"][:,:,None]),axis=2)
    qy = np.concatenate((walls["south"][:,None,:],power["y"],-walls["north"][:,None,:]),axis=1)
    net = -np.diff(qx,axis=2)-np.diff(qy,axis=1)+volume*(np.stack([source(t) for t in times])[:,None,:]-.4*fields)
    np.testing.assert_allclose(c*volume*np.diff(fields,axis=0),np.diff(times)[:,None,None]*
                               ((1-theta)*net[:-1]+theta*net[1:]),rtol=0.,atol=1e-14)


def test_single_cell_optional_interior_output():
    u,power = solve_transient_conduction([.3],[.8],1.,1.,{},[0.,1.],2.,return_interior_power=True)
    np.testing.assert_allclose(u,2.)
    assert power["x"].shape==(2,1,0) and power["y"].shape==(2,0,1)


@pytest.mark.parametrize("theta",[.5,1.])
def test_offgrid_source_and_flux_jumps(theta):
    times = np.array([0.,.04,.09,.13])
    jumps = [.031,.09]
    flux = lambda t: 1.4 if t<jumps[0] else -.3
    source = lambda t: .5 if t<jumps[1] else 2.
    field,walls = solve_transient_conduction([1.],[1.],1.,2.,{"east":dict(type="neumann",values=flux)},
            times,.3,theta=theta,source=source,load_jump_times=jumps,return_wall_power=True)
    heat = (1.4*np.minimum(times,jumps[0])-.3*np.maximum(times-jumps[0],0.)
            +.5*np.minimum(times,jumps[1])+2*np.maximum(times-jumps[1],0.))
    np.testing.assert_allclose(field[:,0,0],.3+.5*heat,atol=1e-14)
    np.testing.assert_allclose(walls["east"][:,0],[flux(t) for t in times])


@pytest.mark.parametrize("jumps",[[0.],[1.],[.3,.3],[np.nan]])
def test_invalid_conduction_jump_times(jumps):
    with pytest.raises(ValueError,match="load_jump_times"):
        solve_transient_conduction([1.],[1.],1.,1.,{},[0.,1.],0.,load_jump_times=jumps)


@pytest.mark.parametrize("theta",[.5,1.])
def test_interval_face_heat_balances_cells_across_unreported_jumps(theta):
    dx,dy = np.array([.2,.3,.5]),np.array([.35,.65])
    c = np.array([1.,2.,.7])
    times = np.array([0.,.05,.12])
    jumps = [.018,.037,.09]
    flux = lambda t: 1.2 if t<jumps[0] else -.3
    source = lambda t: .8 if t<jumps[1] else -.2
    bc = {"west":dict(type="robin",values=lambda t:1. if t<jumps[2] else 2.,
                      film_coefficient=[1.,2.],wall_areal_heat_capacity=[0.,.6]),
          "east":dict(type="neumann",values=flux)}
    field,heat = solve_transient_conduction(dx,dy,1.,c,bc,times,.4,theta=theta,periodic_y=True,
            source=source,load_jump_times=jumps,return_face_heat=True)
    w,q = heat["wall"],heat["interior"]
    x = np.concatenate((w["west"][:,:,None],q["x"],-w["east"][:,:,None]),axis=2)
    y = np.concatenate((w["south"][:,None,:],q["y"],-w["north"][:,None,:]),axis=1)
    generated = .8*np.minimum(times,jumps[1])-.2*np.maximum(times-jumps[1],0.)
    volume = np.outer(dy,dx)
    np.testing.assert_allclose(np.diff(field,axis=0)*volume*c+np.diff(x,axis=2)+np.diff(y,axis=1),
               np.diff(generated)[:,None,None]*volume,atol=1e-14)
    np.testing.assert_allclose(w["east"],np.diff(1.2*np.minimum(times,jumps[0])-.3*np.maximum(times-jumps[0],0.))[:,None]*dy)


def test_rannacher_after_jump_matches_explicit_restarts_and_heat():
    times = [0.,.06,.12]
    jump = .025
    opts = dict(source=lambda t:1+t,heat_loss_coefficient=.8,return_face_heat=True)
    def solve(grid,init,west,**kw):
        return solve_transient_conduction([.4,.6],[1.],2.,1.3,
            {"west":dict(type="dirichlet",values=west)},grid,init,**opts,**kw)
    fields,heat = solve(times,0.,lambda t:1. if t<jump else -.3,rannacher=True,load_jump_times=[jump])
    a,qa = solve([0.,jump/2,jump],0.,1.,theta=1.)
    b,qb = solve([jump,(jump+.06)/2,.06],a[-1],-.3,theta=1.)
    c,qc = solve([.06,.12],b[-1],-.3)
    np.testing.assert_allclose(fields[1:],c,atol=1e-14)
    for group in heat:
        for key,v in heat[group].items():
            np.testing.assert_allclose(v[0],qa[group][key].sum(0)+qb[group][key].sum(0),atol=1e-14)
            np.testing.assert_allclose(v[1],qc[group][key][0],atol=1e-14)


def test_rannacher_rejects_incompatible_theta_and_collapsed_midpoint():
    for times,theta,reason in (([0.,1.],1.,"theta"),([1.,np.nextafter(1.,2.)],.5,"midpoint")):
        with pytest.raises(ValueError,match=reason):
            solve_transient_conduction([1.],[1.],1.,1.,{},times,0.,theta=theta,rannacher=True)


def test_monotone_theta_single_cell_stiff_cooling_uses_consistent_heat():
    opts = dict(dx=[1.],dy=[1.],conductivity=5.,volumetric_heat_capacity=1.,
                bc={"west":dict(type="dirichlet",values=0.)},time=[0.,.01,1.],initial_temperature=1.,
                return_face_heat=True)
    raw,_ = solve_transient_conduction(**opts)
    safe,heat = solve_transient_conduction(**opts,monotone_theta=True)
    assert raw[-1,0,0]<0.
    np.testing.assert_allclose(safe[:,0,0],[1.,.95/1.05,0.],atol=2e-15)
    np.testing.assert_allclose(safe[1:,0,0]-safe[:-1,0,0],heat["wall"]["west"][:,0],atol=2e-15)


def test_monotone_theta_rejects_negative_storing_wall_coupling():
    bc = {"west":dict(type="robin",values=0.,film_coefficient=2.,wall_areal_heat_capacity=1.)}
    with pytest.raises(ValueError,match="off-diagonal"):
        solve_transient_conduction([.5,.5],[1.],1.,1.,bc,[0.,.01],1.,wall_flux_order=2,monotone_theta=True)


def test_fixed_temperature_quadratic_flux_uses_graded_cell_means_on_each_side():
    dx, dy = np.array([.1,.3,.6]), np.array([.2,.45,.35])
    x, y = np.cumsum(dx)-dx/2, np.cumsum(dy)-dy/2
    mean_x = x**2+dx**2/12
    mean_y = y**2+dy**2/12
    initial = mean_x+mean_y[:,None]+x+2*y[:,None]
    west, south = mean_y+2*y, mean_x+x
    bc = {side:dict(type="dirichlet",values=values) for side,values in
          (("west",west),("east",west+2),("south",south),("north",south+3))}
    _,power = solve_transient_conduction(dx,dy,2.,1.,bc,[0.,.001],initial,
        conductivity_y=3.,dirichlet_flux_order=2,return_wall_power=True)
    for side,exact in (("west",-2*dy),("east",6*dy),("south",-6*dx),("north",12*dx)):
        np.testing.assert_allclose(power[side][0],exact,atol=2e-14)


@pytest.mark.parametrize("rannacher", [False, True])
def test_layered_plate_cell_heat_at_switched_heating_rate(rannacher, record_property):
    dx, dy = np.array([.25,.75]), np.array([.25,.5])
    k, capacity, loss = np.array([1.,3.]), np.array([2.,4.]), np.array([.5,2.])
    volume = np.outer(dy,dx)
    jump, stop = .125, .5
    # A unit negative flux requires a positive temperature jump at the contact.
    initial = np.array([.125,.5])
    def rise(t):
        return min(t,jump)-2*max(t-jump,0.)
    def source(t):
        return capacity*(1. if t < jump else -2.)+loss*(initial+rise(t))
    bc = {"west":dict(type="dirichlet",values=rise),
          "east":dict(type="dirichlet",values=lambda t: .625+rise(t))}
    fields, faces, cell = solve_transient_conduction(
        dx,dy,k,capacity,bc,[0.,stop],initial,source=source,contact_resistance_x=.125,
        heat_loss_coefficient=loss,load_jump_times=[jump],rannacher=rannacher,
        return_face_heat=True,return_cell_heat=True)
    # Crank-Nicolson integrates piecewise linear T exactly. Rannacher uses
    # two right-endpoint half steps on each segment of this output interval.
    area = jump**2/2 + jump*(stop-jump)-(stop-jump)**2
    if rannacher:
        area += (jump**2-2*(stop-jump)**2)/4
    expected_bulk = -volume*loss*(stop*initial+area)
    storage = volume*capacity*rise(stop)
    cut = faces["interior"]["x"][0,:,0]
    face_input = np.column_stack((faces["wall"]["west"][0]-cut,cut+faces["wall"]["east"][0]))
    errors = dict(final_K=np.max(abs(fields[-1]-initial-rise(stop))),
                  interface_J=np.max(abs(cut+dy*stop)),
                  bulk_J=np.max(abs(cell["bulk"][0]-expected_bulk)),
                  source_J=np.max(abs(cell["source"][0]-(storage-expected_bulk))),
                  budget_J=np.max(abs(volume*capacity*(fields[-1]-fields[0])
                                      -face_input-cell["source"][0]-cell["bulk"][0])))
    for name,error in errors.items():
        record_property(name,float(error))
        assert error < 1e-13


def test_cell_heat_static_source_and_monotone_sink():
    fields, heat = solve_transient_conduction([.5],[.25],1.,1.,{},[0.,.01,1.],2.,
        source=1.,heat_loss_coefficient=10.,monotone_theta=True,return_cell_heat=True)
    np.testing.assert_allclose(heat["source"][:,0,0], .125*np.array([.01,.99]))
    np.testing.assert_allclose(.125*np.diff(fields,axis=0),heat["source"]+heat["bulk"],atol=1e-15)


@pytest.mark.parametrize("rannacher", [False, True])
def test_switched_bulk_ambient_keeps_plate_heater_separate(rannacher, record_property):
    dx, dy = np.array([.25,.75]), np.array([.5,1.])
    volume = np.outer(dy,dx)
    x = np.cumsum(dx)-dx/2
    initial = 305.+x
    capacity, loss = np.array([2.,3.]), np.array([1.,2.])
    pattern = np.array([[1.,-.5],[-1.,.5]])
    def excess(t):
        return pattern*(1. if t < .125 else -2.)
    fields, faces, cell = solve_transient_conduction(
        dx,dy,1.,capacity,{"west":dict(type="dirichlet",values=lambda t: 305.+t),
                            "east":dict(type="dirichlet",values=lambda t: 306.+t)},
        [0.,.5],initial,heat_loss_coefficient=loss,source=lambda t: capacity-loss*excess(t),
        bulk_ambient_temperature=lambda t: initial+t+excess(t),load_jump_times=[.125],
        rannacher=rannacher,return_face_heat=True,return_cell_heat=True)
    bulk = -.625*pattern*loss*volume
    stored = .5*capacity*volume
    west = faces["wall"]["west"][0]
    east = faces["wall"]["east"][0]
    cut = faces["interior"]["x"][0,:,0]
    inward = np.column_stack((west-cut,cut+east))
    errors = dict(final_K=np.max(abs(fields[-1]-initial-.5)),
                  bulk_J=np.max(abs(cell["bulk"][0]-bulk)),
                  source_J=np.max(abs(cell["source"][0]-(stored-bulk))),
                  budget_J=np.max(abs(capacity*volume*(fields[-1]-fields[0])
                                      -inward-cell["source"][0]-cell["bulk"][0])))
    for name,error in errors.items():
        record_property(name,float(error))
        assert error < 2e-13


def test_sub_ulp_heater_drives_wall_and_bulk_heat(record_property):
    # One cell with unit capacity, west conductance 2, ambient conductance 3.
    # A single CN step gives the exact discrete mean excess dt*P/(2+5*dt).
    power, dt = 2.**-50, .25
    _, face, cell = solve_transient_conduction(
        [1.],[1.],1.,1.,{"west":dict(type="dirichlet",values=305.)},[0.,dt],305.,
        source=power,heat_loss_coefficient=3.,bulk_ambient_temperature=305.,
        return_face_heat=True,return_cell_heat=True)
    mean = dt/(2+5*dt)
    scaled = np.array([face["wall"]["west"].item(),cell["bulk"].item(),cell["source"].item()])/power
    error = max(abs(scaled-dt*np.array([-2*mean,-3*mean,1.])))
    record_property("heat_error_divided_by_heater_power_s",float(error))
    assert error < 1e-15


def test_bulk_reference_preserves_mixed_storing_wall_and_flux_values():
    def run(base):
        wall = dict(types=["robin","neumann"],values=lambda t: [base+t, .75],
                    film_coefficient=2.,wall_areal_heat_capacity=[1.,0.],
                    initial_wall_temperature=base+.25)
        return solve_transient_conduction([.5,.5],[.5,.5],1.,1.,{"west":wall},[0.,.125,1.],base,
            heat_loss_coefficient=1.,bulk_ambient_temperature=base+.5,monotone_theta=True,
            return_wall_power=True,return_cell_heat=True)
    absolute,wall,cell = run(305.)
    relative,reference,expected = run(0.)
    np.testing.assert_allclose(absolute-305.,relative,atol=3e-14,rtol=0)
    for key in reference:
        np.testing.assert_allclose(wall[key],reference[key],atol=1e-14)
    for key in expected:
        np.testing.assert_allclose(cell[key],expected[key],atol=1e-14)
