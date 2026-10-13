import numpy as np
import pytest
from tensorfvm.native_hybrid_rans3d import NativeHybridRans3D
from tensorfvm.native_sst_wall3d import NativeSSTWall3D
from tensorfvm.native_sst3d import sst2003_closure
from tensorfvm.verification.native_spalding import spalding_couette_solver


def setup(n=2):
    m,flow,*_=spalding_couette_solver(n,mu=.02,nu=.01)
    wall=flow.wall_model.mask;fixed=(m.neighbor<0)&~wall
    r=NativeHybridRans3D(flow,np.minimum(m.cell_centers[:,1],1-m.cell_centers[:,1]),.01,
        k_dirichlet_mask=fixed,omega_dirichlet_mask=fixed,k_boundary=.01,omega_boundary=2.,
        k_inflow=.01,omega_inflow=2.,scalar_correction='nonlinear_mmp',scalar_warm_start=True,
        scalar_precision='longdouble',wall_treatment='spalding_blended')
    flow.initialize_stokes();r.initialize(.01,2.,face_initialization='cell_average')
    return r


def test_wall_G_replaces_only_wall_k_production_and_keeps_omega_expression():
    s=setup();n=s.flow.N;grad=np.zeros((n,3,3));grad[:,0,0]=20.
    closure=sst2003_closure(s.k,s.omega,s.wall_distance,s.nu,grad,np.zeros((n,3)),np.zeros((n,3)))
    before={key:value.copy() for key,value in closure.items()}
    model=NativeSSTWall3D(s.flow.wall_model)
    value=model.apply(closure,s.k,np.tile([1.,0.,0.],(n,1)));wall=value['wall_model'];ids=wall['cells']
    outside=np.ones(n,bool);outside[ids]=False
    assert np.any(abs(value['k_source'][ids]-closure['k_source'][ids])>1e-4)
    np.testing.assert_array_equal(value['k_source'][outside],closure['k_source'][outside])
    np.testing.assert_array_equal(value['omega_source'],closure['omega_source'])
    np.testing.assert_allclose(value['production_limited'][ids],
        np.minimum(wall['production'],closure['production_limit'][ids])-2/3*s.k[ids]*20.)
    for key in closure:np.testing.assert_array_equal(closure[key],before[key])


def test_actual_step_preserves_prescribed_omega_rows_and_reaudits_current_target(monkeypatch):
    s=setup();step=s.transport.step;captured={}
    def record(*args,**kwargs):
        result=step(*args,**kwargs);captured.update(result);return result
    monkeypatch.setattr(s.transport,'step',record)
    r=s.solve(max_iterations=1,dt=.05,relaxation=.3,tolerance=1e-8,require_steady_ledgers=True)
    constrained=captured['transport']['omega']['constrained_cells']
    target=captured['transport']['omega']['prescribed_cell_values']
    np.testing.assert_array_equal(s.omega[constrained],target)
    assert np.any(abs(captured['transport']['omega']['cell_constraint_reaction'])>1e-5)
    current=r['current_rans_residual'];newtarget=current['transport']['closure']['wall_model']['omega']
    assert current['omega_constraint']==pytest.approx(float(np.max(abs(s.omega[constrained]-newtarget))))
    assert current['omega_constraint']>1e-8  # old subsolve target is not the current nonlinear constraint
    assert not r['numerically_converged']
    assert np.any(abs(captured['closure']['wall_model']['production'])>0)
    fresh=setup();fresh.restore_accepted_fields(r)
    np.testing.assert_array_equal(fresh.k,s.k);np.testing.assert_array_equal(fresh.omega,s.omega)
    np.testing.assert_array_equal(fresh._closure()['wall_model']['omega'],newtarget)
    assert fresh.current_residual()['omega_constraint']==current['omega_constraint']


def test_missing_wall_mode_or_conflicting_scalar_conditions_are_rejected():
    s=setup();flow=s.flow;kwargs=dict(k_boundary=.01,omega_boundary=2.)
    with pytest.raises(ValueError,match='explicit'):NativeHybridRans3D(flow,s.wall_distance,.01,**kwargs)
    with pytest.raises(ValueError,match='free scalar faces'):
        NativeHybridRans3D(flow,s.wall_distance,.01,wall_treatment='spalding_blended',**kwargs)
    mask=s.transport.k_mask
    with pytest.raises(ValueError,match='zero k/omega'):
        NativeHybridRans3D(flow,s.wall_distance,.01,k_dirichlet_mask=mask,omega_dirichlet_mask=mask,
            k_neumann=1.,wall_treatment='spalding_blended',**kwargs)


def test_failed_momentum_trial_rolls_back_complete_coupled_state(monkeypatch):
    s=setup();before=s.result();old_history=list(s.history)
    def failed(**kwargs):
        s.flow.state[:3*s.flow.N]=1e100
        s.flow.p[:]=1e100
        raise RuntimeError('injected subsolve failure after velocity mutation')
    monkeypatch.setattr(s.flow,'solve_newton',failed)
    with pytest.raises(RuntimeError,match='injected'):s.solve(max_iterations=1)
    after=s.result()
    for key in ('velocity','face_velocity','modified_pressure','k','omega','k_faces','omega_faces'):
        np.testing.assert_array_equal(after[key],before[key])
    assert s.history==old_history and not after['numerically_converged']
    assert s.last_failure['rolled_back']
    assert np.max(s.last_failed_state['velocity'])==1e100
    assert np.isfinite(s.current_residual()['omega_constraint'])
