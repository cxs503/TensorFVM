import numpy as np
import pytest
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_scalar_mmp3d import ScalarMMPConvergenceError
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh


def setup():
    mesh=pressure_mesh(2)
    op=NativeHybridScalar3D(mesh,.2,correction='nonlinear_mmp')
    boundary=1.+.2*mesh.face_centers[:,0]
    q=mesh.area_vectors@np.array([.3,-.1,.2])
    kwargs=dict(boundary_values=boundary,inflow_values=boundary,
                source=.1,sink=.3,old_values=1.,dt=.2)
    return op,q,kwargs


def test_warm_original_equations_match_cold_after_changed_source():
    op,q,kwargs=setup()
    previous=op.solve(q,**kwargs)
    state=np.r_[previous['cell_values'],previous['face_values']];saved=state.copy()
    kwargs['source']=.1001
    warm=op.solve(q,initial_state=state,**kwargs)
    cold=op.solve(q,**kwargs)
    assert warm['warm_start']['accepted'] and not warm['warm_start']['fallback_to_continuation']
    assert all(row['smoothing']==0 for row in warm['nonlinear_history'])
    assert warm['current_nonlinear_residual']<1e-12 and op.ledger_acceptable(warm)
    assert warm['nonlinear_history'][-1]['change']<1e-12
    for key in ('cell_values','face_values','total_flux','neighbor_total_flux'):
        np.testing.assert_allclose(warm[key],cold[key],rtol=1e-10,atol=1e-11)
    np.testing.assert_array_equal(state,saved)
    assert len(warm['nonlinear_history'])<len(cold['nonlinear_history'])


def test_rejected_warm_attempt_falls_back_without_changing_input(monkeypatch):
    import tensorfvm.native_scalar_mmp_newton3d as newton
    op,q,kwargs=setup();expected=op.solve(q,**kwargs)
    state=np.r_[expected['cell_values'],expected['face_values']];saved=state.copy()
    original=newton._solve_mmp_newton;calls=[]
    def reject_warm(scalar,flux,*,initial_state=None,**options):
        calls.append(initial_state is not None)
        if initial_state is not None:
            failed=dict(expected,converged=False,failure='injected original ledger rejection')
            raise ScalarMMPConvergenceError(failed)
        return original(scalar,flux,**options)
    monkeypatch.setattr(newton,'_solve_mmp_newton',reject_warm)
    actual=op.solve(q,initial_state=state,**kwargs)
    assert calls==[True,False]
    assert actual['warm_start']['fallback_to_continuation'] and not actual['warm_start']['accepted']
    assert op.ledger_acceptable(actual)
    for key in ('cell_values','face_values','total_flux'):
        np.testing.assert_array_equal(actual[key],expected[key])
    np.testing.assert_array_equal(state,saved)


def test_invalid_warm_state_and_unsupported_solver_are_not_silently_ignored():
    op,q,kwargs=setup();r=op.solve(q,**kwargs)
    state=np.r_[r['cell_values'],r['face_values']]
    bad=state.copy();bad[op.N+np.flatnonzero(op.boundary)[0]]+=.1
    for initial in (state[:-1],np.full_like(state,np.nan),bad):
        with pytest.raises(ValueError):op.solve(q,initial_state=initial,**kwargs)
    for options in ({},dict(correction='nonlinear_mmp',mmp_solver='picard')):
        plain=NativeHybridScalar3D(op.mesh,**options)
        with pytest.raises(ValueError,match='requires nonlinear_mmp'):
            plain.solve(q,initial_state=state,**kwargs)


def test_coupled_sst_warm_start_matches_cold_steps_with_updated_wall_targets():
    from tensorfvm.native_hybrid_sst3d import NativeHybridSSTTransport3D
    from tensorfvm.verification.native_hybrid_flow import channel_mesh
    mesh=channel_mesh((2,4,1));N=len(mesh.cells);F=len(mesh.owner)
    gradient=np.zeros((N,3,3));gradient[:,0,1]=.3
    state=dict(k=np.full(N,.01),omega=np.full(N,2.),
               k_faces=np.full(F,.01),omega_faces=np.full(F,2.))
    args=dict(k_boundary=.01,omega_boundary=2.,k_inflow=.01,omega_inflow=2.,omega_constraints=([0],[2.1]))
    warm=NativeHybridSSTTransport3D(mesh,scalar_correction='nonlinear_mmp',scalar_warm_start=True)
    cold=NativeHybridSSTTransport3D(mesh,scalar_correction='nonlinear_mmp')
    for step in range(2):
        a=warm.step(**state,volume_flux=np.zeros(F),wall_distance=.1,nu=.1,velocity_gradient=gradient,dt=.2,**args)
        b=cold.step(**state,volume_flux=np.zeros(F),wall_distance=.1,nu=.1,velocity_gradient=gradient,dt=.2,**args)
        for key in ('k','omega','k_faces','omega_faces'):
            np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-11)
        for name in ('k','omega'):
            assert a['transport'][name]['warm_start']['attempted']
            assert a['transport'][name]['current_nonlinear_residual']<1e-12
            assert abs(a['transport'][name]['global_balance'])<1e-8
        assert a['omega'][0]==2.1
        assert not a['nonlinear_rans_converged']
        state={key:a[key] for key in state}
