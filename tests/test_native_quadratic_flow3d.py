import numpy as np
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
from tensorfvm.native_mesh3d import NativeMesh3D


def skew_mesh(n=3):
    points=np.array([[x,y,z] for z in np.linspace(0,1,n+1) for y in np.linspace(0,1,n+1) for x in np.linspace(0,1,n+1)])
    points=points@np.array([[1,.18,.12],[0,1,.2],[0,0,1]]).T
    def ix(x,y,z): return z*(n+1)**2+y*(n+1)+x
    cells=[[ix(x,y,z),ix(x+1,y,z),ix(x+1,y+1,z),ix(x,y+1,z),ix(x,y,z+1),ix(x+1,y,z+1),ix(x+1,y+1,z+1),ix(x,y+1,z+1)] for z in range(n) for y in range(n) for x in range(n)]
    return NativeMesh3D.from_cells(points,cells,['hexahedron']*len(cells),triangulate_faces=True)


def fields(points):
    x,y,z=points.T
    return np.column_stack((x*x,-2*x*y,np.zeros(len(x)))), x*x+.3*y*z


def boundary_means(mesh):
    # Independent polynomial triangle rule, exact degree two.
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    vertices=mesh.vertices[np.array(mesh.face_vertices)]
    points=np.einsum('qj,fjk->fqk',bary,vertices)
    u,p=fields(points.reshape(-1,3))
    return u.reshape(len(vertices),3,3).mean(1),p.reshape(len(vertices),3).mean(1)


def test_quadratic_shared_pressure_stress_and_rhie_chow_are_consistent():
    m=skew_mesh();boundary=m.neighbor<0
    u,p=fields(m.cell_centers);ub,pb=fields(m.face_centers);means,pm=boundary_means(m)
    solver=NativeQuadraticSteadySolver3D(m,boundary,ub,boundary_velocity_means=means,
        settings=NativeSteadySettings(viscosity=.5,convection=False),gauge_value=p[0])
    solver.u=u;solver.p=p
    np.testing.assert_allclose(solver.quadratic_pressure_faces(),pm,atol=3e-13)
    np.testing.assert_allclose(solver.quadratic_pressure_defect(pm),0,atol=3e-13)
    np.testing.assert_allclose(solver.viscous.op.integrate(solver.quadratic_traction())/m.cell_volumes[:,None],np.tile([1,0,0],(len(u),1)),atol=2e-12)
    f=solver.forces('boundary')
    np.testing.assert_allclose(f['viscous_force'],[-m.cell_volumes.sum(),0,0],atol=3e-13)


def test_actual_stokes_quadratic_solution_from_zero_with_continuum_force():
    m=skew_mesh();boundary=m.neighbor<0
    u,p=fields(m.cell_centers);ub,pb=fields(m.face_centers);means,pm=boundary_means(m)
    x,y,z=m.cell_centers.T
    force=np.column_stack((2*x-1,.3*z,.3*y))
    flux=np.einsum('fi,fi->f',means,m.area_vectors)
    solver=NativeQuadraticSteadySolver3D(m,boundary,ub,boundary_velocity_means=means,
        boundary_flux=flux,body_force=force,gauge_value=p[0],settings=NativeSteadySettings(
            viscosity=.5,convection=False,max_iterations=100,nonlinear_tolerance=1e-9,relaxation=.7))
    assert np.all(solver.u==0)
    result=solver.solve()
    assert result['converged'],result['history'][-1]
    np.testing.assert_allclose(result['velocity'],u,atol=1e-8)
    np.testing.assert_allclose(result['pressure'],p,atol=1e-8)
    assert result['history'][-1]['current_reconstructed_divergence_max']<1e-9


def test_nonlinear_affine_velocity_quadratic_pressure_from_zero():
    m=skew_mesh();boundary=m.neighbor<0
    G=np.array([[.2,.4,.1],[-.3,-.1,.2],[.1,.1,-.1]])
    u=np.array([.3,-.1,.2])+m.cell_centers@G.T
    ub=np.array([.3,-.1,.2])+m.face_centers@G.T
    x,y,z=m.cell_centers.T
    p=x*x+.3*y*z
    force=u@G.T+np.column_stack((2*x,.3*z,.3*y))
    solver=NativeQuadraticSteadySolver3D(m,boundary,ub,
        boundary_velocity_means=ub,boundary_velocity_gradients=np.tile(G,(len(m.owner),1,1)),
        boundary_flux=np.einsum('fi,fi->f',ub,m.area_vectors),body_force=force,
        gauge_value=p[0],settings=NativeSteadySettings(viscosity=.5,
            max_iterations=100,nonlinear_tolerance=1e-9,relaxation=.7))
    result=solver.solve()
    assert result['converged'],result['history'][-1]
    np.testing.assert_allclose(result['velocity'],u,atol=1e-8)
    np.testing.assert_allclose(result['pressure'],p,atol=1e-8)


def test_mixed_outlet_is_rejected_until_its_quadratic_boundary_is_validated():
    import pytest
    m=skew_mesh();fixed=m.neighbor<0;fixed[np.flatnonzero(fixed)[0]]=False
    with pytest.raises(ValueError,match='mixed outlet'):
        NativeQuadraticSteadySolver3D(m,fixed,np.ones((len(m.owner),3)),
            pressure_dirichlet=(m.neighbor<0)&~fixed)


def test_coarse_nonlinear_abc_uses_recorded_numerical_warmstart_and_backtracking():
    from tensorfvm.native_collocated3d import NativeSteadySolver3D
    from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means,volume_weighted_error
    m=skew_mesh();boundary=m.neighbor<0;exact=abc_fields(m.cell_centers)
    settings=NativeSteadySettings(viscosity=.1,nonlinear_tolerance=1e-9,linear_tolerance=1e-11,max_iterations=100,relaxation=.8)
    data=dict(settings=settings,body_force=exact['acceleration'],boundary_flux=exact_boundary_flux(m,abc_fields),gauge_value=exact['pressure'][0])
    ub=abc_fields(m.face_centers)['velocity']
    seed=NativeSteadySolver3D(m,boundary,ub,**data).solve()
    assert seed['converged']
    solver=NativeQuadraticSteadySolver3D(m,boundary,ub,**data,
        boundary_velocity_means=exact_boundary_velocity_means(m,abc_fields),
        boundary_velocity_gradients=abc_fields(m.face_centers)['velocity_gradient'])
    solver.u=seed['velocity'].copy();solver.p=seed['pressure'].copy()
    result=solver.solve()
    assert result['converged'],result['history'][-1]
    assert any(row['accepted_relaxation']<settings.relaxation for row in result['history'])
    assert volume_weighted_error(result['velocity'],exact['velocity'],m.cell_volumes)<.03
    assert volume_weighted_error(result['pressure'],exact['pressure'],m.cell_volumes,remove_mean=True)<.03


def test_unvalidated_quadratic_linearizations_are_rejected():
    import pytest
    m=skew_mesh();boundary=m.neighbor<0
    for changed in (dict(bounded_convection=True),dict(convection_order=1),dict(linear_method='schur')):
        with pytest.raises(ValueError,match='Quadratic'):
            NativeQuadraticSteadySolver3D(m,boundary,np.zeros((len(m.owner),3)),
                settings=NativeSteadySettings(**changed))
