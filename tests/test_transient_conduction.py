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
def test_second_order_dynamic_wall_with_adjacent_material_contact(side):
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
