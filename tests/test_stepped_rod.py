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
