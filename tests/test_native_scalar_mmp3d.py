import numpy as np
import pytest
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tensorfvm.verification.native_hybrid_rans import coupled_shear_solver
from tensorfvm.native_hybrid_sst3d import HybridSSTAdmissibilityError


def test_symmetric_graph_correction_preserves_constants_and_energy():
    m=pressure_mesh(2);op=NativeHybridScalar3D(m,2.,correction='nonlinear_mmp')
    state=np.random.default_rng(8).normal(size=op.N+op.F)
    corrected,info=op.mmp.operator(state);B=(corrected.full_matrix-op.diffusion.full_matrix).toarray()
    np.testing.assert_allclose(B,B.T,atol=1e-13)
    np.testing.assert_allclose(B.sum(axis=1),0.,atol=1e-13)
    assert np.linalg.eigvalsh(B)[0]>-1e-12
    np.testing.assert_allclose(state@B@state,info['correction_energy'],rtol=1e-12)
    _,constant=op.mmp.operator(np.ones(op.N+op.F)*3.)
    assert constant['correction_energy']==0


def test_negative_wall_candidate_repaired_by_conservative_nonlinear_equations():
    m,s=coupled_shear_solver(2,'low_re',triangulate_faces=True)
    args=(s.k,s.omega,s.k_faces,s.omega_faces,s.flow.volume_flux(),s.wall_distance,s.nu,s.flow.velocity_gradient(),1.)
    with pytest.raises(HybridSSTAdmissibilityError):s.transport.step(*args,**s.boundary)
    s.transport.scalar_correction='nonlinear_mmp'
    r=s.transport.step(*args,**s.boundary)
    for name in ('k','omega'):
        t=r['transport'][name]
        assert min(t['cell_values'].min(),t['face_values'].min())>=0
        assert t['current_nonlinear_residual']<1e-10
        assert t['interface_flux_defect_max']<1e-8
        assert abs(t['global_relative_balance'])<1e-8
        assert t['correction_energy']>0
    # This is a changed conservative equation, not clipping the old solution.
    assert r['transport']['k']['cell_balance_max']<1e-8


def test_triangle_wall_rans_uses_current_corrected_residual():
    _,s=coupled_shear_solver(2,'low_re',triangulate_faces=True,scalar_correction='nonlinear_mmp')
    r=s.solve(max_iterations=40,dt=1.,tolerance=1e-8)
    assert r['numerically_converged'],r['rans_history'][-1]
    assert r['k_faces'].min()==0 and r['omega_faces'].min()>0
    s.transport.scalar_correction='none'
    original=s.current_residual()
    assert max(original['k'],original['omega'])>1e-3
    assert not s.result()['numerically_converged']


def test_nonzero_neumann_is_explicitly_unsupported_by_correction():
    m=pressure_mesh(1);mask=np.zeros(len(m.owner),bool)
    op=NativeHybridScalar3D(m,dirichlet_mask=mask,correction='nonlinear_mmp')
    with pytest.raises(ValueError,match='homogeneous Neumann'):
        op.solve(old_values=1.,dt=.1,boundary_diffusive_flux=.1)


def test_strict_correction_affine_regularization_error_decreases():
    errors=[]
    for n in (2,4):
        m=pressure_mesh(n);g=np.array([.1,-.2,.3]);bd=1+m.face_centers@g
        r=NativeHybridScalar3D(m,2.,correction='nonlinear_mmp').solve(boundary_values=bd)
        errors.append(np.max(np.abs(r['cell_values']-(1+m.cell_centers@g))))
        assert r['current_nonlinear_residual']<1e-12
        assert abs(r['work_balance'])<1e-8
    # Strict correction is no longer affine-exact; quantify the diminishing bias.
    assert errors[1]<errors[0]/2 and errors[0]<.001,errors


def test_zero_source_harmonic_negative_solution_is_not_accepted_as_converged():
    m=pressure_mesh(2);bd=np.zeros(len(m.owner));bd[[38,52,60,90]]=1.
    base=NativeHybridScalar3D(m).solve(boundary_values=bd)
    assert base['face_min']<-.03 and base['cell_min']<0
    result=NativeHybridScalar3D(m,correction='nonlinear_mmp').solve(boundary_values=bd)
    assert result['cell_min']>0 and result['face_min']>=0
    assert result['cell_values'].max()<=1 and result['face_values'].max()<=1
    assert result['current_nonlinear_residual']<1e-12
    assert abs(result['global_balance'])<1e-8
