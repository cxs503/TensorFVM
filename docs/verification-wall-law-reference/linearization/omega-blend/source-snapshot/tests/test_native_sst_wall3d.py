from decimal import Decimal,localcontext
import numpy as np
import pytest
from tensorfvm.native_sst_wall3d import foundation12_blended_omega_wall,area_average_wall_cells


@pytest.mark.parametrize('k',['0.00000001','0.0004','0.04','4'])
def test_blended_wall_against_independent_decimal(k):
    with localcontext() as ctx:
        ctx.prec=60
        d=Decimal;K=d(k);y=d('.002');nu=d('.000001');grad=d('80');G=d('.025')
        beta=d('.075');cmu=d('.09');kap=d('.41');E=d('9.8')
        rey=y*K.sqrt()/nu;yp=cmu.sqrt().sqrt()*rey
        lam=(-rey/11).exp();turb=1-lam
        us=(lam*nu*grad+turb*cmu.sqrt()*K).sqrt()
        omega=lam*6*nu/(beta*y*y)+turb*us/(cmu.sqrt()*kap*y)
        production=lam*G+turb*(us*grad*y/((E*yp).ln()/kap))**2/(nu*kap*yp)
    result=foundation12_blended_omega_wall(float(k),.002,1e-6,80.,.025)
    assert result['omega']==pytest.approx(float(omega),rel=2e-14)
    assert result['production']==pytest.approx(float(production),rel=2e-14)


def test_viscous_limit_zero_gradient_and_broadcasting_without_clipping():
    r=foundation12_blended_omega_wall([0.,0.],.002,1e-6,[0.,80.],[.025,-.01])
    np.testing.assert_allclose(r['omega'],20.,rtol=1e-15)
    np.testing.assert_array_equal(r['production'],[.025,-.01])
    r=foundation12_blended_omega_wall(np.array([.001,.04])[:,None],.002,1e-6,0.,[.025,.05])
    np.testing.assert_allclose(r['production'],r['viscous_fraction']*[.025,.05])
    assert r['omega'].shape==(2,2)


def test_area_weighting_multiple_walls_is_invariant_to_face_splitting():
    original=area_average_wall_cells([7,2,7],[3.,2.,1.],[10.,20.,30.],[1.,2.,9.])
    split=area_average_wall_cells([7,2,7,7],[1.,2.,2.,1.],[10.,20.,10.,30.],[1.,2.,1.,9.])
    for key in original:np.testing.assert_array_equal(original[key],split[key])
    np.testing.assert_array_equal(original['cells'],[2,7])
    np.testing.assert_array_equal(original['omega'],[20.,15.])
    np.testing.assert_array_equal(original['production'],[2.,3.])


def test_wall_values_reject_invalid_inputs_and_log_pole():
    for args in [(-1.,.1,.1,1.,0.),(1.,0.,.1,1.,0.),(1.,.1,0.,1.,0.),(1.,.1,.1,-1.,0.),(1.,.1,.1,1.,np.nan)]:
        with pytest.raises(ValueError):foundation12_blended_omega_wall(*args)
    with pytest.raises(ValueError,match='singular'):
        foundation12_blended_omega_wall(1.,1.,1.,1.,0.,cmu=1.,E=1.)
    with pytest.raises(ValueError):area_average_wall_cells([0.],[1.],[1.],[1.])
    with pytest.raises(ValueError):area_average_wall_cells([0],[0.],[1.],[1.])


def test_computed_wall_omega_enters_native_transport_with_constraint_reaction():
    from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
    from tensorfvm.native_hybrid_sst3d import NativeHybridSSTTransport3D
    mesh=pressure_mesh(2);boundary=mesh.neighbor<0
    wall=boundary & (abs(mesh.face_centers[:,2])<1e-12)
    assert wall.any()
    distance=abs(mesh.cell_centers[mesh.owner[wall],2])
    values=foundation12_blended_omega_wall(.01,distance,.1,.3,0.)
    cells=area_average_wall_cells(mesh.owner[wall],np.linalg.norm(mesh.area_vectors[wall],axis=1),values['omega'],values['production'])
    op=NativeHybridSSTTransport3D(mesh,k_dirichlet_mask=boundary&~wall,omega_dirichlet_mask=boundary&~wall,
        scalar_correction='nonlinear_mmp',scalar_precision='longdouble')
    n,f=op.geometry.N,op.geometry.F
    r=op.step(np.full(n,.01),np.full(n,2.),np.full(f,.01),np.full(f,2.),np.zeros(f),.1,.1,
        np.zeros((n,3,3)),.01,k_boundary=.01,omega_boundary=2.,
        omega_constraints=(cells['cells'],cells['omega']))
    omega=r['transport']['omega']
    np.testing.assert_array_equal(r['omega'][cells['cells']],cells['omega'])
    assert np.any(abs(omega['cell_constraint_reaction'])>1e-3)
    assert abs(omega['global_balance'])<1e-8
    assert omega['cell_balance_max']<1e-8 and abs(omega['work_balance'])<1e-8
    # This tests the actual constraint route, not complete wall-model coupling:
    # returned wall G is intentionally NOT claimed to be installed here.
