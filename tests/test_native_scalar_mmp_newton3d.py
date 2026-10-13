import numpy as np
import pytest
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tensorfvm.verification.native_hybrid_rans import coupled_shear_solver
from tensorfvm.suboff_hybrid_mesh import structured_layers
from tensorfvm.native_mesh3d import NativeMesh3D


@pytest.mark.parametrize('smoothing',[0.,1e-4])
def test_graph_jacobian_matches_independent_directional_difference(smoothing):
    op=NativeHybridScalar3D(pressure_mesh(2),correction='nonlinear_mmp')
    rng=np.random.default_rng(45);u=1+rng.uniform(-.4,.4,op.N+op.F);v=rng.normal(size=len(u))
    matrix,jac=op.mmp.nonlinear_system(u,smoothing=smoothing,state_scale=2.,jacobian=True)
    np.testing.assert_allclose(op.mmp.nonlinear_reaction(u,smoothing=smoothing,state_scale=2.),matrix@u,atol=2e-14)
    eps=1e-7
    residual=lambda z:op.mmp.nonlinear_system(z,smoothing=smoothing,state_scale=2.)@z
    finite=(residual(u+eps*v)-residual(u-eps*v))/(2*eps)
    np.testing.assert_allclose(jac@v,finite,rtol=2e-6,atol=2e-7)
    if smoothing==0:
        original,_=op.mmp.operator(u)
        np.testing.assert_allclose(matrix.toarray(),original.full_matrix.toarray(),atol=2e-14)
    scaled=op.mmp.nonlinear_system(3*u,smoothing=smoothing,state_scale=6.)
    np.testing.assert_allclose(scaled.toarray(),matrix.toarray(),atol=2e-14)


def test_continuation_matches_picard_positive_harmonic_solution():
    m=pressure_mesh(2);bd=np.zeros(len(m.owner));bd[[38,52,60,90]]=1.
    a=NativeHybridScalar3D(m,correction='nonlinear_mmp',mmp_solver='picard').solve(boundary_values=bd)
    b=NativeHybridScalar3D(m,correction='nonlinear_mmp').solve(boundary_values=bd)
    for key in ('cell_values','face_values','diffusive_flux'):
        np.testing.assert_allclose(a[key],b[key],atol=3e-10)
    assert b['final_smoothing']==0
    assert b['current_nonlinear_residual']<1e-12
    for row in b['nonlinear_history']:
        if row['accepted']:
            assert row['merit_after']<=(1-1e-4*row['step_length'])*row['smoothed_relative_residual']


def test_actual_suboff_thin_layer_original_equations_and_ledger_converge():
    points,cells,types,_,_,_=structured_layers(axial=12,azimuthal=8,layers=2,first_height=.001,growth=1.2)
    m=NativeMesh3D.from_cells(points,cells,types,triangulate_faces=True);g=np.array([.1,-.2,.3])
    op=NativeHybridScalar3D(m,correction='nonlinear_mmp')
    r=op.solve(boundary_values=1+m.face_centers@g)
    assert r['final_smoothing']==0 and r['current_nonlinear_residual']<1e-12
    assert max(r['cell_balance_max'],r['interface_flux_defect_max'],abs(r['global_balance']),abs(r['work_balance']))<1e-8
    assert np.max(np.abs(r['cell_values']-1-m.cell_centers@g))<1e-4
    assert r['cell_min']>0 and r['face_min']>0


def test_fine_triangle_wall_coupling_finishes_without_clipping():
    _,s=coupled_shear_solver(8,'low_re',triangulate_faces=True,scalar_correction='nonlinear_mmp')
    r=s.solve(max_iterations=60,dt=1.,tolerance=1e-8)
    assert r['numerically_converged'],r['rans_history'][-1]
    assert r['k_faces'].min()==0 and r['omega_faces'].min()>0
    assert max(r['current_rans_residual'][name] for name in ('momentum','divergence','k','omega'))<1e-8
    assert not r['physical_benchmark_passed']


def test_corrected_residual_broadcasts_uniform_scalar_inputs():
    op=NativeHybridScalar3D(pressure_mesh(1),correction='nonlinear_mmp')
    r=op.steady_residual(1.,1.,boundary_values=1.)
    assert r['relative_residual']<1e-12


def test_stable_power_ledger_preserves_identity_and_detects_perturbation():
    op=NativeHybridScalar3D(pressure_mesh(2))
    r=op.solve(boundary_values=1e4)
    np.testing.assert_allclose(r['work_balance'],
        r['work_balance_cell_terms'].sum()+r['work_balance_interface_terms'].sum(),atol=1e-16)
    state=np.r_[r['cell_values'],r['face_values']].astype(np.longdouble)
    state[0]+=1e-3
    perturbed=op._solution_result(state,np.zeros(op.F),np.zeros(op.N),np.zeros(op.N),np.zeros(op.N),0.,None,0.)
    assert abs(perturbed['work_balance'])>1e-3
    np.testing.assert_allclose(perturbed['work_balance'],
        perturbed['work_balance_cell_terms'].sum()+perturbed['work_balance_interface_terms'].sum(),rtol=1e-12)


def test_numerical_restart_rechecks_original_equations_and_unrelaxed_updates():
    m=pressure_mesh(2);bd=np.zeros(len(m.owner));bd[[38,52,60,90]]=1.
    op=NativeHybridScalar3D(m,correction='nonlinear_mmp')
    original=op.solve(boundary_values=bd)
    state=np.r_[original['cell_values'],original['face_values']]
    saved=state.copy()
    restarted=op.solve(boundary_values=bd,nonlinear_initial_state=state)
    assert restarted['numerical_state_restart'] and restarted['final_smoothing']==0
    assert restarted['current_nonlinear_residual']<1e-12
    history=restarted['nonlinear_history']
    assert len(history)>=2
    assert all(row['fixed_point_polish'] and row['unrelaxed'] and row['change']<1e-12 for row in history[-2:])
    np.testing.assert_array_equal(state,saved)
    np.testing.assert_allclose(restarted['cell_values'],original['cell_values'],atol=1e-10)


def test_numerical_restart_rejects_invalid_states_and_boundary_changes():
    m=pressure_mesh(1);op=NativeHybridScalar3D(m,correction='nonlinear_mmp')
    for state in (np.ones(op.N),np.full(op.N+op.F,np.nan),np.zeros(op.N+op.F)):
        with pytest.raises(ValueError):op.solve(boundary_values=1.,nonlinear_initial_state=state)
    for correction,method in [('none','continuation_newton'),('nonlinear_mmp','picard')]:
        s=NativeHybridScalar3D(m,correction=correction,mmp_solver=method)
        with pytest.raises(ValueError):s.solve(boundary_values=1.,nonlinear_initial_state=np.ones(s.N+s.F))
