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
