import numpy as np
import pytest

from tensorfvm.cht.energy import solve_harmonic_conduction
from tensorfvm.cht.rod import solve_harmonic_stepped_rod


@pytest.mark.parametrize("area", [[.8], [.7,2.,2.,.6]])
def test_harmonic_rod_matches_cartesian_reduction_and_cell_balance(area):
    n = len(area)
    area = np.asarray(area)
    dx = np.linspace(.2,.6,n)
    k = np.linspace(2.,3.,n)
    c = np.linspace(.8,2.,n)
    ends,omega,source = [.2+.4j,-.3j],4.,.6-.3j
    cell,face,left,right = solve_harmonic_stepped_rod(dx,area,k,c,ends,omega,source=source)
    bc = {s:dict(type="dirichlet",values=t) for s,t in zip(("west","east"),ends)}
    ref, _ = solve_harmonic_conduction(dx,[1.],k*area,c*area,bc,omega,source=source*area)
    np.testing.assert_allclose(cell,ref[0],atol=1e-12)
    np.testing.assert_allclose(face[[0,-1]],ends)
    np.testing.assert_allclose(left-right,(1j*omega*c*cell-source)*dx*area,atol=1e-12)
    np.testing.assert_allclose(right[:-1],left[1:],atol=1e-12)


def test_opposite_shoulders_close_independently():
    a = np.array([.3,.8,.4])
    h, cw, ambient = np.array([2.,5.]), np.array([.3,.6]), np.array([1j,-.5j])
    _,faces,left,right = solve_harmonic_stepped_rod([.2,.4,.3],a,2.,1.,[1.,0.],3.,
                       film_coefficient=h,ambient_temperature=ambient,shoulder_areal_heat_capacity=cw)
    exposed = abs(np.diff(a))
    np.testing.assert_allclose(left[1:]-right[:-1],exposed*(h*(ambient-faces[1:-1])-3j*cw*faces[1:-1]))


@pytest.mark.parametrize("theta", [.5,1.])
@pytest.mark.parametrize("order", [1,2])
def test_transient_mixed_shoulders_close_with_initial_storage_and_varying_load(theta, order):
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    dx = np.array([.2,.3,.1,.4])
    a = np.array([.3,.8,.8,.4])
    exposed = abs(np.diff(a))
    c = np.array([1.,2.,3.,4.])
    times = np.array([0.,.002,.01,.07,.1])
    h,cw = np.array([3.,.2,1.]),np.array([.6,.5,0.])
    cells,face,left,right = solve_transient_stepped_rod(dx,a,1.2,c,lambda t:[t,2*t],times,1.,
                       film_coefficient=h,ambient_temperature=lambda t:t+.5,source=lambda t:1.-t,
                       shoulder_areal_heat_capacity=cw,initial_shoulder_temperature=3.,theta=theta,flux_order=order)
    assert face[0,1] == 3.
    if order == 1:
        assert face[0,2] == pytest.approx(1.)
    def integrate(v):
        return np.diff(times)[:,None]*((1-theta)*v[:-1]+theta*v[1:])
    np.testing.assert_allclose(np.diff(cells,axis=0)*(dx*a*c),integrate(left-right+(1-times[:,None])*dx*a),
                               atol=1e-13)
    flux = right[:,:-1]-left[:,1:] + h*exposed*(.5+times[:,None]-face[:,1:-1])
    np.testing.assert_allclose(np.diff(face[:,1:-1],axis=0)*(cw*exposed),integrate(flux),atol=1e-13)
    np.testing.assert_allclose(flux[:,1:],0.,atol=1e-12)


def test_hot_shoulder_pulse_matches_matrix_exponential():
    from scipy.linalg import expm
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    # Two unit-wide cells with areas 1 and 2 and k=c=1. The excess area
    # carries 1 J/K, initially hot. Both ends are held at zero.
    capacity = np.diag([1.,1.,2.])
    conductance = np.array([[4.,-2.,0.],[-2.,6.,-4.],[0.,-4.,8.]])
    exact = expm(-.5*np.linalg.solve(capacity,conductance)) @ [0.,1.,0.]
    previous = None
    for steps in (16,32,64):
        cell,face,_,_ = solve_transient_stepped_rod([1.,1.],[1.,2.],1.,1.,[0.,0.],
                             np.linspace(0.,.5,steps+1),0.,shoulder_areal_heat_capacity=1.,
                             initial_shoulder_temperature=1.)
        error = np.max(abs(np.r_[cell[-1,0],face[-1,1],cell[-1,1]]-exact))
        if previous is not None:
            assert previous/error > 3.9
        previous = error
    assert error < 2e-5


@pytest.mark.parametrize("n", [1,4])
def test_quadratic_rod_reproduces_exact_means_and_both_powers(n):
    dx = np.array([.13,.4,.27,.2])[:n]
    a = np.array([1.,2.,.8,1.2])[:n]
    k = np.array([2.,.8,3.,1.5])[:n]
    c = np.arange(n)+1.
    slope = (.3-.6j)*(1+np.arange(n))
    incoming = (.7+.1j)/(1+np.arange(n))
    outgoing = incoming+slope*dx
    drops = dx*(incoming+.5*slope*dx)/(k*a)
    faces = 1.+.4j-np.r_[0.,np.cumsum(drops)]
    means = faces[:-1]-dx*(incoming/2+slope*dx/6)/(k*a)
    omega, h, cw = 2.,3.,.7
    source = 1j*omega*c*means+slope/a
    ambient = faces[1:-1]*(1+1j*omega*cw/h)+(incoming[1:]-outgoing[:-1])/(abs(np.diff(a))*h)
    observed = solve_harmonic_stepped_rod(dx,a,k,c,faces[[0,-1]],omega,source=source,
                 film_coefficient=h,ambient_temperature=ambient,shoulder_areal_heat_capacity=cw,flux_order=2)
    for result, exact in zip(observed,(means,faces,incoming,outgoing)):
        np.testing.assert_allclose(result,exact,atol=2e-13,rtol=0)


@pytest.mark.parametrize("order",[1,2])
def test_resistive_shoulder_transient_temperature_and_power_paths(order):
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    area = np.array([.8,1.3,.6,.6])
    resistance = np.array([.3,.2,.15])
    times = np.array([0.,.003,.013,.1])
    cell,faces,enter,leave,(up,down) = solve_transient_stepped_rod(
        [.2,.3,.4,.1],area,2.,1.,[.3,1.],times,0.,film_coefficient=2.,
        shoulder_areal_heat_capacity=.5,ambient_temperature=1.,initial_shoulder_temperature=3.,
        theta=.6,contact_resistance=resistance,return_traces=True,flux_order=order)
    bridge = (up-down)*np.minimum(area[:-1],area[1:])/resistance
    np.testing.assert_allclose(faces[:,1:-1],np.column_stack((down[:,0],up[:,1],up[:,2])))
    np.testing.assert_allclose(faces[0,1:3],3.)
    np.testing.assert_allclose(bridge[:,0],leave[:,0],atol=1e-12)
    np.testing.assert_allclose(bridge[:,1:],enter[:,2:],atol=1e-12)
    np.testing.assert_allclose(bridge[:,2],leave[:,2],atol=1e-12)
    exposed = abs(np.diff(area))
    net = leave[:,:-1]-enter[:,1:]+2*exposed*(1.-faces[:,1:-1])
    np.testing.assert_allclose(.5*exposed*np.diff(faces[:,1:-1],axis=0),
                               np.diff(times)[:,None]*(.4*net[:-1]+.6*net[1:]),atol=1e-12)


@pytest.mark.parametrize("method",["theta","sdirk2"])
def test_reported_step_heats_track_storage_on_nonuniform_intervals(method):
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    dx,area = np.array([.2,.4,.3]),np.array([1.,2.,.5])
    exposed = abs(np.diff(area))
    times = np.array([.2,.207,.24,.31,.46])
    cell,face,_,_,heat = solve_transient_stepped_rod(dx,area,2.,3.,lambda t:[1.+t,t*t],times,0.,
                   source=lambda t:np.array([1.,2.,3.])*t*t,film_coefficient=2.,
                   ambient_temperature=lambda t:[t,1.-2*t],shoulder_areal_heat_capacity=.7,
                   initial_shoulder_temperature=1.,contact_resistance=[.13,.22],method=method,
                   return_step_heat=True,flux_order=2)
    np.testing.assert_allclose(3*area*dx*np.diff(cell,axis=0),
                               heat["left"]-heat["right"]+heat["source"],atol=1e-12)
    np.testing.assert_allclose(.7*exposed*np.diff(face[:,1:-1],axis=0),
                               heat["right"][:,:-1]-heat["left"][:,1:]+heat["ambient"],atol=1e-12)
    np.testing.assert_allclose(heat["contact"],np.column_stack((heat["right"][:,0],heat["left"][:,2])),atol=1e-12)
    gamma = 1.-1./np.sqrt(2.) if method=="sdirk2" else 0.
    last_weight = gamma if method=="sdirk2" else .5
    sample = times[:-1]+gamma*np.diff(times)
    integral = np.diff(times)*((1-last_weight)*sample**2+last_weight*times[1:]**2)
    np.testing.assert_allclose(heat["source"],integral[:,None]*area*dx*[1.,2.,3.])


def test_sdirk2_stiff_mode_damping_and_temporal_order():
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    # One unit cell between zero-temperature ends obeys dT/dt=-4T.
    def run(times):
        return solve_transient_stepped_rod([1.],1.,1.,1.,[0.,0.],times,1.,method="sdirk2")[0][-1,0]
    errors = [abs(run(np.linspace(0.,.25,n+1))-np.exp(-1)) for n in (8,16,32)]
    assert all(e/f>3.9 for e,f in zip(errors,errors[1:]))
    assert abs(run([0.,1e6])) < 2e-6


def test_rannacher_start_matches_two_implicit_half_steps_including_wall_heat():
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    dx,area = [.2,.3,.4],[.5,2.,.8]
    opts = dict(source=lambda t:1.-3*t,film_coefficient=1.2,ambient_temperature=lambda t:3*t,
                contact_resistance=[.2,.15],shoulder_areal_heat_capacity=.4,
                return_step_heat=True,return_traces=True,flux_order=2)
    def advance(time,initial,wall,**kwargs):
        return solve_transient_stepped_rod(dx,area,2.,1.,lambda t:[t,t*t],
                       time,initial,initial_shoulder_temperature=wall,**kwargs,**opts)
    combined = advance([.2,.26,.3],0.,[1.,.5],method="rannacher")
    first = advance([.2,.23,.26],0.,[1.,.5],theta=1.)
    second = advance([.26,.3],first[0][-1],first[1][-1,1:-1],theta=.5)
    for i in range(4):
        np.testing.assert_allclose(combined[i][1],first[i][-1],atol=2e-12)
        np.testing.assert_allclose(combined[i][2],second[i][-1],atol=2e-12)
    for key,heat in combined[-1].items():
        np.testing.assert_allclose(heat[0],first[-1][key].sum(axis=0),atol=1e-13)
        np.testing.assert_allclose(heat[1],second[-1][key][0],atol=1e-13)


def test_positive_cn_substeps_are_nonnegative_and_aggregate_interval_heat():
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    sampled = []
    times = [.1,.13,.2]
    def ends(t):
        sampled.append(t)
        return [.02*t,0.]
    opts = dict(contact_resistance=[.15,.2],film_coefficient=1.5,
                shoulder_areal_heat_capacity=.015,initial_shoulder_temperature=[1.,.4],
                ambient_temperature=lambda t:t*t,source=lambda t:[t,0.,.1],
                return_step_heat=True)
    def run(method,boundary,grid):
        return solve_transient_stepped_rod([.2,.3,.4],[.4,1.6,.8],2.,1.,boundary,grid,0.,
                                           method=method,**opts)
    coarse = run("positive_cn",ends,times)
    assert len(sampled)>len(times)
    fine = run("theta",lambda t:[.02*t,0.],sampled)
    indices = [sampled.index(t) for t in times]
    for i in range(4):
        np.testing.assert_allclose(coarse[i],fine[i][indices],atol=2e-13)
    assert min(fine[0].min(),fine[1].min())>=-1e-14
    for i,(a,b) in enumerate(zip(indices,indices[1:])):
        for key,val in coarse[-1].items():
            np.testing.assert_allclose(val[i],fine[-1][key][a:b].sum(axis=0),atol=1e-13)
    with pytest.raises(ValueError,match="flux_order=1"):
        solve_transient_stepped_rod([1.],1.,1.,1.,[0.,0.],[0.,1.],1.,method="positive_cn",flux_order=2)


@pytest.mark.parametrize("order",[1,2])
@pytest.mark.parametrize("n",[1,5])
def test_harmonic_power_ends_match_discrete_temperature_solution(order,n):
    from tensorfvm.cht.rod import solve_harmonic_stepped_rod
    dx = np.linspace(.15,.3,n)
    area = 1.+np.arange(n)%2
    opts = dict(flux_order=order,source=.3-.2j,film_coefficient=3.,
                shoulder_areal_heat_capacity=.8,contact_resistance=.2,return_traces=True)
    ends = [1.+.3j,-.5+.1j]
    ref = solve_harmonic_stepped_rod(dx,area,2.,3.,ends,4.,**opts)
    power = [ref[2][0],-ref[3][-1]]
    for kinds in (("temperature","power"),("power","temperature"),("power","power")):
        values = [p if kind=="power" else t for kind,t,p in zip(kinds,ends,power)]
        obs = solve_harmonic_stepped_rod(dx,area,2.,3.,values,4.,end_types=kinds,**opts)
        for a,b in zip(obs,ref):
            np.testing.assert_allclose(a,b,rtol=1e-11,atol=2e-12)


@pytest.mark.parametrize("method",["theta","sdirk2","rannacher","positive_cn"])
def test_power_ends_account_for_each_joule_of_heating(method):
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    times = np.array([0.,.01,.05,.18])
    dx,area = np.array([.25,.5]),np.array([1.,3.])
    cell,face,left,right,heat = solve_transient_stepped_rod(
        dx,area,2.,4.,lambda t:[1+t,.3],times,1.,shoulder_areal_heat_capacity=.2,
        contact_resistance=.12,return_step_heat=True,method=method,end_types=("power","power"))
    np.testing.assert_allclose(left[:,0],1.+times,atol=1e-12)
    np.testing.assert_allclose(right[:,-1],-.3,atol=1e-12)
    stored = cell @ (4*dx*area)+.4*face[:,1]
    np.testing.assert_allclose(np.diff(stored),heat["left"][:,0]-heat["right"][:,-1],atol=1e-12)


@pytest.mark.parametrize("order",[1,2])
def test_steady_balanced_end_powers_preserve_contact_jump_and_energy_gauge(order):
    from tensorfvm.cht.rod import solve_steady_stepped_rod
    dx,a,k,c = np.array([.2,.5,.3]),np.array([1.,3.,.8]),np.array([2.,3.,1.]),np.array([2.,1.,4.])
    resistance = np.array([.13,.2])
    opts = dict(end_types=("power","power"),contact_resistance=resistance,flux_order=order,
                shoulder_areal_heat_capacity=.4,return_traces=True)
    cell,faces,left,right,(up,down) = solve_steady_stepped_rod(dx,a,k,c,[.8,-.8],mean_temperature=5.,**opts)
    np.testing.assert_allclose(left,.8,atol=1e-12)
    np.testing.assert_allclose(right,.8,atol=1e-12)
    np.testing.assert_allclose(up-down,.8*resistance/np.minimum(a[1:],a[:-1]),atol=1e-13)
    cm = dx*a*c
    wm = .4*abs(np.diff(a))
    assert (cell@cm+faces[1:-1]@wm)/(cm.sum()+wm.sum()) == pytest.approx(5.)
    with pytest.raises(ValueError,match="incompatible"):
        solve_steady_stepped_rod(dx,a,k,c,[.8,.8],mean_temperature=5.,**opts)
    with pytest.raises(ValueError,match="mean_temperature"):
        solve_steady_stepped_rod(dx,a,k,c,[0.,0.],**opts)
    _,_,qin,qout,_ = solve_steady_stepped_rod(dx,a,k,c,[.8,.8],film_coefficient=2.,**opts)
    np.testing.assert_allclose([qin[0],qout[-1]],[.8,-.8])


@pytest.mark.parametrize("order",[1,2])
def test_robin_ends_match_series_resistance(order):
    from tensorfvm.cht.rod import solve_steady_stepped_rod
    dx,a,k = np.array([.25,.75]),np.array([.7,1.4]),np.array([1.6,.8])
    h,air = np.array([2.,5.]),np.array([315.,283.])
    resistance = .13
    q = (air[0]-air[1])/(sum(1/(h*a))+sum(dx/(k*a))+resistance/min(a))
    _,faces,left,right,(up,down) = solve_steady_stepped_rod(
        dx,a,k,1.,air,end_types=("robin","robin"),end_film_coefficient=h,
        contact_resistance=resistance,flux_order=order,return_traces=True)
    np.testing.assert_allclose(left,q)
    np.testing.assert_allclose(right,q)
    np.testing.assert_allclose(h*a*(air-faces[[0,-1]]),[q,-q])
    np.testing.assert_allclose(up-down,q*resistance/min(a))


def test_insulating_robin_end_needs_gauge_with_power_end():
    from tensorfvm.cht.rod import solve_steady_stepped_rod
    opts = dict(end_types=("power","robin"))
    args = ([.4],2.,3.,4.,[0.,380.])
    with pytest.raises(ValueError,match="mean_temperature"):
        solve_steady_stepped_rod(*args,**opts)
    cell,faces,*_ = solve_steady_stepped_rod(*args,mean_temperature=290.,**opts)
    np.testing.assert_allclose(cell,290.)
    np.testing.assert_allclose(faces,290.)


@pytest.mark.parametrize("method",["theta","rannacher","sdirk2","positive_cn"])
def test_storing_end_heat_matches_time_dependent_inputs(method):
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    area = 2.
    wall_mass = np.array([.2,.5])*area
    times = np.array([0.,.03,.07,.2])
    cell,wall,_,_,heat = solve_transient_stepped_rod(
        [.4],area,1.5,3.,lambda t:[.3+t,2.-t],times,.8,
        initial_end_temperature=[1.5,.4],end_types=("power","robin"),
        end_film_coefficient=[0.,4.],end_areal_heat_capacity=wall_mass/area,
        method=method,return_step_heat=True)
    np.testing.assert_allclose(wall[0],[1.5,.4])
    inward = np.column_stack((heat["left"][:,0],-heat["right"][:,-1]))
    np.testing.assert_allclose(np.diff(wall,axis=0)*wall_mass,heat["end_input"]-inward,atol=1e-13)
    total = cell[:,0]*(3.*.4*area)+wall@wall_mass
    np.testing.assert_allclose(np.diff(total),heat["end_input"].sum(axis=1),atol=1e-13)
    # Rannacher starts with two backward Euler half steps.
    expected = np.diff(times)*(.3+(times[:-1]+times[1:])/2)
    if method=="rannacher":
        expected[0] += np.diff(times)[0]**2/4
    np.testing.assert_allclose(heat["end_input"][:,0],expected,atol=1e-13)


def test_steady_gauge_includes_end_storage():
    from tensorfvm.cht.rod import solve_steady_stepped_rod
    cell,wall,*_ = solve_steady_stepped_rod(
        [.4],2.,1.5,3.,[2.,-2.],end_types=("power","power"),
        end_areal_heat_capacity=[.3,.8],mean_temperature=5.)
    assert (cell[0]*2.4+wall@np.array([.6,1.6]))/4.6 == pytest.approx(5.)
    with pytest.raises(ValueError,match="fixed-temperature"):
        solve_steady_stepped_rod([.4],2.,1.5,3.,[2.,3.],end_areal_heat_capacity=1.)


@pytest.mark.parametrize("method",["theta","sdirk2","rannacher","positive_cn"])
def test_mixed_power_jump_matches_segment_restarts_and_aggregates_heat(method):
    from tensorfvm.cht.rod import solve_transient_stepped_rod
    times = [0.,.03,.1]
    jumps = [.013,.03,.047]
    levels = [.4,1.,-.2,.7]
    def run(wall,grid,initial,**opts):
        return solve_transient_stepped_rod([.4,.6],[1.,2.],2.,1.,wall,grid,initial,
                 contact_resistance=.1,method=method,end_types=("power","temperature"),
                 return_step_heat=True,**opts)
    # Rannacher restarts in each segment, and the endpoint with no storage
    # must immediately satisfy the new imposed power at the output-time jump.
    obs = run(lambda t:[levels[np.searchsorted(jumps,t,side="right")],0.],times,0.,load_jump_times=jumps)
    initial = 0.
    integrated = {key:np.zeros_like(val) for key,val in obs[-1].items()}
    for a,b,level in zip((0.,*jumps),(*jumps,.1),levels):
        part = run([level,0.],[a,b],initial)
        initial = part[0][-1]
        for key,val in part[-1].items():
            integrated[key][int(a>=.03)] += val[0]
    np.testing.assert_allclose(obs[2][1,0],-.2,atol=1e-13)
    np.testing.assert_allclose(obs[0][-1],initial,atol=1e-13)
    for key,val in integrated.items():
        np.testing.assert_allclose(obs[-1][key],val,atol=1e-13)


@pytest.mark.parametrize("area", [[2.,1.],[1.,2.],[1.,1.]])
def test_interface_heater_splits_power_through_two_series_resistances(area):
    from tensorfvm.cht.rod import solve_steady_stepped_rod

    dx,k = np.array([.4,.6]),np.array([1.5,2.])
    resistance,power = .3,4.
    paths = dx/(k*area)
    other_side = int(area[0]>=area[1])
    paths[other_side] += resistance/min(area)
    sheet = power/(1/paths).sum()
    cell,face,left,right = solve_steady_stepped_rod(dx,area,k,1.,[0.,0.],
                                         contact_resistance=resistance,interface_power=power)
    np.testing.assert_allclose(face[1],sheet,atol=1e-13)
    np.testing.assert_allclose([left[0],right[-1]],[-sheet/paths[0],sheet/paths[1]],atol=1e-13)


@pytest.mark.parametrize("method",["theta","sdirk2","positive_cn","rannacher"])
def test_pulsed_interface_power_conserves_energy(method):
    from tensorfvm.cht.rod import solve_transient_stepped_rod

    time = [0.,.05,.12]
    def pulse(t):
        return 5.*(.021<=t<.083)
    u,f,_,_,heat = solve_transient_stepped_rod([.4,.6],[2.,1.],1.,2.,[0.,0.],time,0.,
                   end_types=("power","power"),interface_power=pulse,load_jump_times=(.021,.083),
                   shoulder_areal_heat_capacity=.3,initial_shoulder_temperature=0.,
                   contact_resistance=.2,method=method,return_step_heat=True)
    supplied = 5*np.array([.05-.021,.083-.05])
    np.testing.assert_allclose(heat["interface"][:,0],supplied,atol=1e-14)
    energy = u @ np.array([1.6,1.2])+.3*f[:,1]
    np.testing.assert_allclose(np.diff(energy),supplied,atol=1e-13)
    np.testing.assert_allclose(.3*np.diff(f[:,1]),heat["right"][:,0]-heat["left"][:,1]+supplied,atol=1e-13)


def test_lumped_interface_storage_sets_steady_energy_gauge():
    from tensorfvm.cht.rod import solve_steady_stepped_rod

    opts = dict(end_types=("power","power"),mean_temperature=2.,flux_order=2)
    cell,face,left,right = solve_steady_stepped_rod([.2,.8],1.,1.,1.,[1.,-1.],
                                    interface_heat_capacity=.6,**opts)
    np.testing.assert_allclose((cell @ [.2,.8]+.6*face[1])/1.6,2.,atol=1e-14)
    np.testing.assert_allclose([left,right],1.,atol=1e-14)
    with pytest.raises(ValueError, match="interface heat capacity"):
        solve_steady_stepped_rod([.2,.8],1.,1.,1.,[1.,-1.],interface_heat_capacity=-.1,**opts)


def test_equal_area_harmonic_interface_can_store_heat():
    capacity,omega,power = .6,2.,3.+1j
    _,face,left,right = solve_harmonic_stepped_rod([.3,.7],1.,2.,1.,[0.,0.],omega,
                                    interface_power=power,interface_heat_capacity=capacity)
    np.testing.assert_allclose(left[1]-right[0]+1j*omega*capacity*face[1],power,atol=1e-13)


@pytest.mark.parametrize("area",[[1.,2.],[2.,1.]])
@pytest.mark.parametrize("resistance",[0.,.24])
def test_partitioned_interface_heating_closes_both_material_traces(area,resistance):
    from tensorfvm.cht.rod import solve_transient_stepped_rod

    frac,power = .35,3.
    times = np.linspace(0.,.4,17)
    _,_,_,_,heat = solve_transient_stepped_rod([.4,.6],area,2.,1.,[0.,0.],times,0.,
            interface_power=power,contact_resistance=resistance,interface_left_fraction=frac,
            return_step_heat=True)
    q = heat["contact"][:,0]
    np.testing.assert_allclose(q-heat["right"][:,0],frac*power*np.diff(times),atol=1e-14)
    np.testing.assert_allclose(heat["left"][:,1]-q,(1-frac)*power*np.diff(times),atol=1e-14)


@pytest.mark.parametrize("fraction",[-.01,1.01,np.nan])
def test_invalid_interface_power_partition(fraction):
    with pytest.raises(ValueError,match="interface left fraction"):
        solve_harmonic_stepped_rod([.4,.6],1.,2.,1.,[0.,0.],2.,interface_left_fraction=fraction)


def test_steady_power_partition_matches_two_trace_network():
    from tensorfvm.cht.rod import solve_steady_stepped_rod

    # Two branch conductances (k*A/L) and one overlap conductance (A/R).
    power,frac = 3.,.35
    gw,ge,g = 2/.4,4/.6,1/.24
    exact = np.linalg.solve([[gw+g,-g],[-g,ge+g]],power*np.array([frac,1-frac]))
    _,_,left,right,(west,east) = solve_steady_stepped_rod([.4,.6],[1.,2.],2.,1.,[0.,0.],
                contact_resistance=.24,interface_power=power,interface_left_fraction=frac,return_traces=True)
    np.testing.assert_allclose([west[0],east[0]],exact,atol=1e-14)
    np.testing.assert_allclose([-left[0],right[-1]],exact*np.array([gw,ge]),atol=1e-14)


@pytest.mark.parametrize("area", [[2.,1.],[1.,2.]])
@pytest.mark.parametrize("resistance", [0.,.24])
@pytest.mark.parametrize("method", ["theta","sdirk2","rannacher","positive_cn"])
def test_interface_heat_with_storage_on_each_material(area,resistance,method):
    from tensorfvm.cht.rod import solve_transient_stepped_rod

    temps = (1.,.4 if resistance else 1.)
    _,_,_,_,(left,right),heat = solve_transient_stepped_rod(
        [.4,.6],area,2.,1.,[0.,0.],[0.,.03,.08],0.,method=method,
        interface_heat_capacity=2.,interface_capacity_left_fraction=.25,
        interface_power=lambda t: 3.*(t<.045),interface_left_fraction=.7,load_jump_times=[.045],
        contact_resistance=resistance,initial_trace_temperatures=temps,return_step_heat=True,return_traces=True)
    np.testing.assert_allclose([left[0,0],right[0,0]],temps)
    q,p = heat["contact"][:,0],heat["interface"][:,0]
    np.testing.assert_allclose(.5*np.diff(left[:,0]),heat["right"][:,0]+.7*p-q,atol=1e-13)
    np.testing.assert_allclose(1.5*np.diff(right[:,0]),q+.3*p-heat["left"][:,1],atol=1e-13)


def test_shared_storing_trace_requires_consistent_initial_value():
    from tensorfvm.cht.rod import solve_transient_stepped_rod

    with pytest.raises(ValueError,match="conflicting"):
        solve_transient_stepped_rod([.4,.6],1.,2.,1.,[0.,0.],[0.,.1],0.,
                    interface_heat_capacity=1.,initial_trace_temperatures=(.2,.8))
    with pytest.raises(ValueError,match="interface capacity left fraction"):
        solve_harmonic_stepped_rod([.4,.6],1.,2.,1.,[0.,0.],2.,interface_capacity_left_fraction=-.1)


@pytest.mark.parametrize("resistance",[0.,.18])
def test_opposed_pulses_transfer_heat_between_contact_traces(resistance):
    from tensorfvm.cht.rod import solve_transient_stepped_rod

    time = np.array([0.,.035,.08])
    def drive(t):
        p = 3.*(.015<=t<.052)
        return p,-p
    _,_,_,_,(west,east),heat = solve_transient_stepped_rod(
        [.4,.6],[1.,2.],2.,1.,[0.,0.],time,0.,method="sdirk2",trace_power=drive,
        load_jump_times=(.015,.052),contact_resistance=resistance,interface_power=1.,
        interface_left_fraction=.6,interface_heat_capacity=.8,interface_capacity_left_fraction=.25,
        initial_trace_temperatures=(0.,0.),return_traces=True,return_step_heat=True)
    pulse = 3*np.array([.02,.017])
    dt = np.diff(time)
    np.testing.assert_allclose(heat["trace_left"][:,0],.6*dt+pulse,atol=1e-14)
    np.testing.assert_allclose(heat["trace_right"][:,0],.4*dt-pulse,atol=1e-14)
    np.testing.assert_allclose(heat["interface"][:,0],dt,atol=1e-14)
    q = heat["contact"][:,0]
    np.testing.assert_allclose(.2*np.diff(west[:,0]),heat["right"][:,0]+.6*dt+pulse-q,atol=1e-14)
    np.testing.assert_allclose(.6*np.diff(east[:,0]),q+.4*dt-pulse-heat["left"][:,1],atol=1e-14)


@pytest.mark.parametrize("types", [("power","power"), ("power","robin"), ("robin","temperature")])
def test_steady_quartic_means_and_face_powers(types):
    from tensorfvm.cht.rod import solve_steady_stepped_rod

    x = np.array([-.7,-.54,-.1,.31,.85])
    dx = np.diff(x)
    mid = x[:-1]+dx/2
    area, k, h = 1.4, 2.1, 3.2
    temp = 1.+.2*x-.4*x**3+x**4
    power = -k*area*(.2-1.2*x**2+4*x**3)
    primitive = x+.1*x**2-.1*x**4+x**5/5
    means = np.diff(primitive)/dx
    source = np.diff(power)/(area*dx)
    ends = []
    for kind, t, inward in zip(types, temp[[0,-1]], [power[0],-power[-1]]):
        ends.append(inward if kind=="power" else t+inward/(h*area) if kind=="robin" else t)
    gauge = dx@means/dx.sum() if types==("power","power") else None
    cell,face,left,right = solve_steady_stepped_rod(dx,area,k,1.,ends,source=source,
        source_gradient=-k*(24*mid-2.4),source_curvature=-24*k,flux_order=2,
        end_types=types,end_film_coefficient=h,mean_temperature=gauge)
    for actual, exact in ((cell,means),(face,temp),(left,power[:-1]),(right,power[1:]),
                          (right-left,source*area*dx)):
        np.testing.assert_allclose(actual,exact,atol=1e-12,rtol=0.)


def test_steady_field_reconstruction_keeps_contact_traces():
    from tensorfvm.cht.rod import reconstruct_steady_rod_field, solve_steady_stepped_rod

    x = np.array([0.,.2,.45,1.])
    dx = np.diff(x)
    mid = x[:-1]+dx/2
    resistance = .3
    # T = 2-x⁴ in each region, with an additional jump R*4*x³ at contact.
    jump = resistance*4*x[1]**3
    args = dict(source_gradient=24*mid,source_curvature=24.)
    cell,_,west,east,(hot,cold) = solve_steady_stepped_rod(dx,1.,1.,1.,[2.,1.-jump],
        source=4*np.diff(x**3)/dx,contact_resistance=[resistance,0.],return_traces=True,flux_order=2,**args)
    samples = np.array([0., .17, .64, 1.])
    t,p = reconstruct_steady_rod_field(dx,1.,1.,cell,west,east,samples,**args)
    target = x[:-1,None]+dx[:,None]*samples
    np.testing.assert_allclose(t,2-target**4-np.array([0.,jump,jump])[:,None],atol=1e-13,rtol=0.)
    np.testing.assert_allclose(p,4*target**3,atol=1e-13,rtol=0.)
    np.testing.assert_allclose([t[0,-1],t[1,0]],[hot[0],cold[0]],atol=1e-13,rtol=0.)


def test_harmonic_graded_reservoir_and_stepped_area_conserve_material_power():
    # Choose a linear temperature per section, with analytic constant power,
    # then balance volumetric exchange and storage with an independent source.
    x = np.array([0.,.2,.5,.63,1.])
    dx = np.diff(x)
    a,k,c = np.array([1.,1.,2.,2.]),np.array([3.,3.,2.,2.]),np.array([2.,2.,.8,.8])
    current, omega = .7-.4j, 2.3
    resistance = .3
    left_temp = 1+.5j
    west = []
    for i in range(len(dx)):
        west.append(left_temp)
        left_temp -= current*dx[i]/(k[i]*a[i])
        if i == 1:
            left_temp -= current*resistance
    mean = np.asarray(west)-current*dx/(2*k*a)
    coeff = 1.+np.arange(4)
    reservoir = np.array([1j,.4+.2j,-.1+.3j,.2])
    source = 1j*omega*c*mean+coeff*(mean-reservoir)
    field,_,incoming,outgoing = solve_harmonic_stepped_rod(dx,a,k,c,[west[0],left_temp],omega,
        contact_resistance=[0.,resistance,0.],flux_order=2,source=source,
        heat_loss_coefficient=coeff,bulk_ambient_temperature=reservoir)
    for actual,exact in ((field,mean),(incoming,current),(outgoing,current)):
        np.testing.assert_allclose(actual,exact,atol=2e-13,rtol=0.)
    defect = outgoing-incoming+a*dx*(1j*omega*c*field-source-coeff*(reservoir-field))
    assert abs(defect).max()<1e-13


@pytest.mark.parametrize("method", ["theta","sdirk2","rannacher","positive_cn"])
def test_transient_bulk_reservoir_switch_balances_each_stepped_cell(method):
    from tensorfvm.cht.rod import solve_transient_stepped_rod

    dx,area,c = np.array([.2,.35,.45]), np.array([1.,.8,1.7]), np.array([2.,1.,3.])
    jump, rate = .083, 4.
    time = np.linspace(0.,.2,21)
    reservoir = lambda t: 2. if t<jump else -.5
    u,_,_,_,heat = solve_transient_stepped_rod(dx,area,[1.,2.,1.],c,[0.,0.],time,np.zeros(3),
        method=method, end_types=("power","power"), heat_loss_coefficient=rate*c,
        bulk_ambient_temperature=reservoir, load_jump_times=[jump],return_step_heat=True)
    first = -2*np.expm1(-rate*np.minimum(time,jump))
    exact = np.where(time<=jump,first,-.5+(first+.5)*np.exp(-rate*(time-jump)))
    np.testing.assert_allclose(u,np.broadcast_to(exact[:,None],u.shape),atol=2e-3,rtol=0.)
    defect = np.diff(u,axis=0)*(dx*area*c)-heat["left"]+heat["right"]-heat["bulk_exchange"]
    np.testing.assert_allclose(defect,0.,atol=1e-14,rtol=0.)
    assert heat["bulk_exchange"][0].min()>0 and heat["bulk_exchange"][-1].max()<0


def test_local_steady_exchange_anchors_insulated_rod_with_remote_heating():
    from tensorfvm.cht.rod import solve_steady_stepped_rod

    dx = np.array([.25,.25,.5])
    area,k = np.array([1.,1.,2.]),np.array([2.,2.,1.])
    source = np.array([4.,4.,0.])
    loss = [0.,0.,3.]
    cells,_,left,right = solve_steady_stepped_rod(dx,area,k,1.,[0.,0.],source=source,
        end_types=("power","power"),flux_order=2,heat_loss_coefficient=loss,bulk_ambient_temperature=5.)
    np.testing.assert_allclose(left,[0.,1.,2.],rtol=0.,atol=1e-12)
    np.testing.assert_allclose(right,[1.,2.,0.],rtol=0.,atol=1e-12)
    np.testing.assert_allclose(cells[-1],5+2/3,rtol=0.,atol=1e-12)
    with pytest.raises(ValueError,match="mean_temperature"):
        solve_steady_stepped_rod(dx,area,k,1.,[0.,0.],end_types=("power","power"),
                                 heat_loss_coefficient=loss,mean_temperature=1.)
