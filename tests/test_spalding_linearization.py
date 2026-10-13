import numpy as np
import pytest
from tensorfvm.wall_functions import wall_traction,wall_traction_linearization,u_plus_at_y_plus


@pytest.mark.parametrize('yp',[.001,1.,30.,300.,3000.])
def test_spalding_tangent_matches_independent_traction_differences(yp):
    rho=998.2;nu=1e-6;ut=.07;y=yp*nu/ut
    speed=float(u_plus_at_y_plus(yp))*ut
    direction=np.array([2.,-1.,3.]);direction/=np.linalg.norm(direction);v=speed*direction
    r=wall_traction_linearization(v,y,rho,rho*nu)
    np.testing.assert_allclose(r['friction_velocity_m_s'],ut,rtol=2e-12)
    h=speed*1e-5;fd=np.empty((3,3))
    for j in range(3):
        dv=np.eye(3)[j]*h
        fd[:,j]=(wall_traction(v+dv,y,rho,rho*nu)['traction_pa']-wall_traction(v-dv,y,rho,rho*nu)['traction_pa'])/(2*h)
    np.testing.assert_allclose(r['traction_jacobian'],fd,rtol=3e-7,atol=1e-8*np.linalg.norm(fd))
    eigen=np.linalg.eigvalsh(r['traction_jacobian'])
    assert np.max(eigen)<0 and v@r['traction_pa']<0
    np.testing.assert_allclose(r['traction_jacobian'],r['traction_jacobian'].T,atol=0)


def test_zero_speed_viscous_limit_and_rotation_covariance():
    rho=1000.;mu=.001;y=np.array([.001,.01]);v=np.array([[0.,0.,0.],[1.,2.,-.5]])
    a=wall_traction_linearization(v,y,rho,mu)
    np.testing.assert_array_equal(a['traction_pa'][0],0.)
    np.testing.assert_allclose(a['traction_jacobian'][0],-mu/y[0]*np.eye(3),atol=0)
    q,_=np.linalg.qr(np.array([[1.,2.,4.],[-1.,3.,2.],[3.,2.,1.]]))
    b=wall_traction_linearization(v@q.T,y,rho,mu)
    np.testing.assert_allclose(b['traction_pa'],a['traction_pa']@q.T,atol=1e-13)
    np.testing.assert_allclose(b['traction_jacobian'],q@a['traction_jacobian']@q.T,atol=1e-13)
    batch=wall_traction_linearization(v[1],y,rho,mu)
    for i,distance in enumerate(y):
        one=wall_traction_linearization(v[1],distance,rho,mu)
        np.testing.assert_allclose(batch['traction_jacobian'][i],one['traction_jacobian'],atol=0)
