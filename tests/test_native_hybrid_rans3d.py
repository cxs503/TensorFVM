import numpy as np
import pytest
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_hybrid_rans3d import NativeHybridRans3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh
from tensorfvm.verification.native_hybrid_ns import potential_solver


@pytest.mark.parametrize('method',['reused_lu','block_amg'])
def test_viscosity_refresh_invalidates_cache_and_matches_fresh_solver(method):
    if method=='block_amg':pytest.importorskip('pyamg')
    _,s,_=potential_solver(2,.1);s.coupled_linear_method=method
    s.initialize_stokes();s.solve_newton(tolerance=1e-9)
    state=s.state.copy();p=s.p.copy()
    s.update_dynamic_viscosity(.2)
    assert s._coupled_factor is None and s._block_preconditioner is None
    assert not s.converged and not s.nonlinear_converged
    np.testing.assert_array_equal(state,s.state);np.testing.assert_array_equal(p,s.p)
    r=s.solve_newton(tolerance=1e-9)
    _,fresh,_=potential_solver(2,.2);fresh.initialize_stokes();reference=fresh.solve_newton(tolerance=1e-9)
    assert r['numerically_converged']
    for key in ('velocity','pressure','face_velocity','total_traction_integrated'):
        np.testing.assert_allclose(r[key],reference[key],atol=2e-8)
    matrix=s.viscous
    with pytest.raises(ValueError):s.update_dynamic_viscosity(-1.)
    assert s.viscous is matrix


def test_current_scalar_residual_detects_transient_and_free_face_imbalance():
    m=channel_mesh(2);mask=np.zeros(len(m.owner),bool)
    op=NativeHybridScalar3D(m,.1,dirichlet_mask=mask)
    r=op.solve(old_values=1.,dt=.1,sink=2.)
    steady=op.steady_residual(r['cell_values'],r['face_values'],sink=2.)
    assert r['linear_relative_residual']<1e-12
    assert steady['relative_residual']>.9
    np.testing.assert_allclose(steady['cell_equation'],2*m.cell_volumes/1.2,atol=1e-13)
    faces=np.ones(len(m.owner));faces[np.flatnonzero(m.neighbor>=0)[0]]+=.1
    defect=op.steady_residual(np.ones(len(m.cells)),faces)
    assert defect['free_face_equation_max']>.01
    # A shared owner-only ledger is insufficient to qualify the face equations.
    assert defect['interface_flux_defect_max']>.01


def setup(shear=0.,k=.01,omega=2.,nu=.1):
    m=channel_mesh((2,4,1));b=m.neighbor<0
    boundary=np.zeros((len(m.owner),3));boundary[:,0]=shear*m.face_centers[:,1]
    flow=NativeHybridNavierStokes3D(m,b,boundary,viscosity=nu,convection_order=2)
    flow.initialize_stokes()
    rans=NativeHybridRans3D(flow,.1,nu,k_boundary=k,omega_boundary=omega,
                          k_inflow=k,omega_inflow=omega)
    rans.initialize(k,omega)
    return rans


def test_current_sst_source_residual_is_not_frozen_step_residual():
    s=setup();r=s.solve(max_iterations=1,dt=.01,relaxation=.7)
    assert not r['numerically_converged']
    row=r['rans_history'][-1]
    assert max(row['frozen_linear_residuals'].values())<1e-11
    assert row['omega']>1e-5 and row['k']>1e-5
    np.testing.assert_allclose(r['physical_pressure'],r['modified_pressure']-2/3*r['k'],atol=1e-15)
    np.testing.assert_allclose(r['total_traction_integrated'],r['pressure_traction_integrated']+
        r['isotropic_turbulent_traction_integrated']+r['deviatoric_traction_integrated']+
        r['numerical_traction_correction'],atol=1e-13)
    assert not r['physical_benchmark_passed']


def test_laminar_limit_couette_preserves_exact_velocity_force_and_sst_equilibrium():
    # k=0 implies F1=0, gamma=.44, beta=.0828. Constant omega equilibrium.
    shear=.3;omega=np.sqrt(.44/.0828)*shear
    s=setup(shear=shear,k=0.,omega=omega)
    r=s.solve(max_iterations=3,dt=.1,relaxation=1.,tolerance=1e-8)
    assert r['numerically_converged'],r['rans_history']
    np.testing.assert_allclose(r['velocity'][:,0],shear*s.mesh.cell_centers[:,1],atol=1e-10)
    np.testing.assert_allclose(r['omega'],omega,atol=1e-11)
    np.testing.assert_array_equal(r['k'],0.)
    upper=(s.mesh.neighbor<0)&(s.mesh.face_centers[:,1]>1-1e-10)
    np.testing.assert_allclose(r['total_traction_integrated'][upper,0].sum(),.1*shear,atol=1e-10)
    assert r['energy_balance_error']<1e-10


def test_positive_coupling_converges_current_equations_and_rejects_stale_claim():
    s=setup(shear=.3);r=s.solve(max_iterations=60,dt=1.,tolerance=1e-8)
    assert r['numerically_converged'],r['rans_history'][-1]
    assert np.ptp(r['nu_t'])>1e-6
    assert r['k'].min()>0 and r['omega'].min()>0
    assert max(r['current_rans_residual'][k] for k in ('momentum','divergence','k','omega'))<1e-8
    assert r['energy_balance_error']<1e-8
    np.testing.assert_allclose(s.flow.viscous.mu,s.nu+r['nu_t'],atol=1e-15)
    # External changes must not inherit the previous convergence label.
    s.omega*=1.1
    assert not s.result()['numerically_converged']


def test_low_re_wall_positive_on_quad_faces_and_triangle_counterexample_retained():
    from tensorfvm.verification.native_hybrid_rans import coupled_shear_solver
    from tensorfvm.native_hybrid_sst3d import HybridSSTAdmissibilityError
    from tensorfvm.native_sst3d import wall_omega
    m,s=coupled_shear_solver(4,'low_re',density=1.7)
    r=s.solve(max_iterations=60,dt=1.,tolerance=1e-8)
    assert r['numerically_converged'],r['rans_history'][-1]
    wall=(m.neighbor<0)&((m.face_centers[:,1]<1e-10)|(m.face_centers[:,1]>1-1e-10))
    np.testing.assert_array_equal(r['k_faces'][wall],0.)
    np.testing.assert_allclose(r['omega_faces'][wall],wall_omega(.1,s.wall_distance[m.owner[wall]]))
    np.testing.assert_allclose(r['physical_pressure'],r['modified_pressure']-2/3*1.7*r['k'])
    np.testing.assert_allclose(s.flow.viscous.mu,1.7*(s.nu+r['nu_t']))
    _,bad=coupled_shear_solver(2,'low_re',triangulate_faces=True)
    old=bad.k.copy()
    with pytest.raises(HybridSSTAdmissibilityError) as failure:bad.solve(max_iterations=1,dt=1.)
    assert failure.value.scalar_result['face_min']<0
    np.testing.assert_array_equal(bad.k,old)
