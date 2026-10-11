import numpy as np
import pytest
from tensorfvm.native_collocated3d import NativeSteadySolver3D,NativeSteadySettings
from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
from tensorfvm.native_quadratic_system3d import current_matrix
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means,volume_weighted_error
from tests.test_native_quadratic_maps3d import actual_mesh
from tests.test_native_face_geometry3d import mesh as planar_mesh
from tensorfvm.native_face_geometry3d import merge_coplanar_faces


def test_reject_pressure_dirichlet_with_all_fixed_boundary_fluxes():
    m=actual_mesh();boundary=m.neighbor<0
    pressure_boundary=np.zeros(len(m.owner),bool)
    pressure_boundary[np.flatnonzero(boundary)[0]]=True
    with pytest.raises(ValueError,match='requires a pressure gauge'):
        NativeQuadraticSteadySolver3D(m,boundary,np.zeros((len(m.owner),3)),
            pressure_dirichlet=pressure_boundary,quadratic_linear_solver='assembled')


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('prescribed_mean',[False,True])
@pytest.mark.parametrize('merged',[False,True])
def test_assembled_full_equations_match_original_residual_with_frozen_flux(mixed,prescribed_mean,merged):
    m=merge_coplanar_faces(planar_mesh(mixed))[0] if merged else actual_mesh(mixed)
    exact=abc_fields(m.cell_centers);boundary=abc_fields(m.face_centers)
    solver=NativeQuadraticSteadySolver3D(m,m.neighbor<0,boundary['velocity'],quadratic_linear_solver='assembled',
        boundary_velocity_means=exact_boundary_velocity_means(m,abc_fields) if prescribed_mean else None,
        boundary_velocity_gradients=boundary['velocity_gradient'],boundary_flux=exact_boundary_flux(m,abc_fields),
        body_force=exact['acceleration'],gauge_value=exact['pressure'][0],settings=NativeSteadySettings(viscosity=.1))
    rng=np.random.default_rng(3931);solver.u=rng.normal(size=solver.u.shape);solver.p=rng.normal(size=solver.p.shape)
    q=solver.velocity_flux_matrix@solver.u.ravel()+solver.velocity_flux_source;solver._linear_flux=q
    solver._frozen_covariance_divergence=solver._covariance_divergence()
    _,_,a,_,_=solver._system(q);matrix=current_matrix(solver,q,a)
    solver.u=np.zeros_like(solver.u);solver.p=np.zeros_like(solver.p)
    _,source,_,_,_=solver._system(q)
    for amplitude in (1.,1e5):
        x=rng.normal(size=4*solver.N)*amplitude;solver.u=x[:3*solver.N].reshape(solver.N,3);solver.p=x[3*solver.N:]
        base,rhs,_,_,_=solver._system(q)
        actual=base@x-rhs;assembled=matrix@x-source
        assert np.linalg.norm(actual-assembled)/max(np.linalg.norm(actual),1)<3e-13
    del solver._frozen_covariance_divergence


@pytest.mark.parametrize('merged',[False,True])
@pytest.mark.parametrize('ordering',['COLAMD','MMD_AT_PLUS_A'])
def test_assembled_nonlinear_mixed_mesh_converges_from_numerical_seed(merged,ordering):
    m=merge_coplanar_faces(planar_mesh(True))[0] if merged else actual_mesh(True)
    exact=abc_fields(m.cell_centers);boundary=abc_fields(m.face_centers)
    settings=NativeSteadySettings(viscosity=.1,nonlinear_tolerance=1e-9,linear_tolerance=1e-11,max_iterations=100,relaxation=.8,lu_ordering=ordering)
    kwargs=dict(settings=settings,body_force=exact['acceleration'],boundary_flux=exact_boundary_flux(m,abc_fields),gauge_value=exact['pressure'][0])
    seed=NativeSteadySolver3D(m,m.neighbor<0,boundary['velocity'],**kwargs).solve();assert seed['converged']
    solver=NativeQuadraticSteadySolver3D(m,m.neighbor<0,boundary['velocity'],**kwargs,quadratic_linear_solver='assembled',boundary_velocity_means=exact_boundary_velocity_means(m,abc_fields),boundary_velocity_gradients=boundary['velocity_gradient'])
    solver.u=seed['velocity'].copy();solver.p=seed['pressure'].copy()
    result=solver.solve();assert result['converged'],result['history'][-1]
    assert volume_weighted_error(result['velocity'],exact['velocity'],m.cell_volumes)<.03
    assert volume_weighted_error(result['pressure'],exact['pressure'],m.cell_volumes,remove_mean=True)<.03
