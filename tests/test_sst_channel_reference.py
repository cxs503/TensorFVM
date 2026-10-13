import numpy as np
import pytest
from tensorfvm.verification.sst_channel_reference import closure,scalar_step,scalar_residual,solve_reference


def test_independent_parallel_shear_closure_matches_production_and_total_stress():
    from tensorfvm.native_sst3d import sst2003_closure
    y=np.array([.0001,.03,.4,.9]);k=np.array([0.,.01,.7,.8]);w=np.array([9000.,100.,10.,2.])
    dk=np.array([1.,2.,-.1,-.2]);dw=np.array([-10000.,-300.,-2.,-.1]);nu=1/182.088
    independent=closure(y,k,w,dk,dw,nu)
    G=np.zeros((4,3,3));G[:,0,1]=independent['shear']
    gk=np.zeros((4,3));gk[:,1]=dk;gw=np.zeros((4,3));gw[:,1]=dw
    production=sst2003_closure(k,w,y,nu,G,gk,gw)
    np.testing.assert_allclose((nu+independent['nut'])*independent['shear'],1-y,rtol=1e-13)
    for a,b in [('nut','nu_t'),('F1','F1'),('F2','F2'),('k_production','production_limited')]:
        np.testing.assert_allclose(independent[a],production[b],rtol=1e-13)
    np.testing.assert_allclose(independent['omega_production'],production['omega_production']+production['cross_diffusion_source'],rtol=1e-13)
    np.testing.assert_allclose(independent['k_diffusion'],nu+production['sigma_k']*production['nu_t'],rtol=1e-13)
    np.testing.assert_allclose(independent['omega_sink']*w,production['omega_destruction'],rtol=1e-13)


def test_independent_diffusion_refines_quadratic_and_conserves_sources():
    errors=[]
    for n in (16,32,64):
        faces=np.expm1(3*np.linspace(0,1,n+1))/np.expm1(3);y=(faces[:-1]+faces[1:])/2
        value=scalar_step(faces,y,np.zeros(n),np.ones(n),np.ones(n),np.zeros(n),1.,1e30)
        error=scalar_residual(faces,y,value,np.ones(n),np.ones(n),np.zeros(n),1.)
        assert error['relative_local_max']<1e-11
        assert abs(error['global_balance'])<1e-11
        errors.append(np.max(abs(value-(1+y-.5*y*y))))
    assert errors[0]/errors[1]>3.5 and errors[1]/errors[2]>3.5


def test_independent_reference_requires_current_equations_and_is_time_step_independent():
    a=solve_reference(16,tolerance=1e-8,dt=.01)
    b=solve_reference(16,tolerance=1e-8,dt=.02)
    assert a['converged'] and b['converged']
    for result in (a,b):
        assert all(max(row['current_relative_local_residual'],row['unrelaxed_change'])<1e-8 for row in result['history'][-2:])
        assert not result['physical_suboff_benchmark_passed']
    np.testing.assert_allclose(a['Cf'],b['Cf'],rtol=1e-8)
    assert not solve_reference(16,max_iterations=2)['converged']
    for options in ({'dt':0},{'re_tau':np.nan},{'relaxation':2}):
        with pytest.raises(ValueError):solve_reference(16,**options)
