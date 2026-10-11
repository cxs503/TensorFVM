import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import FreePressureReconstruction3D,NativeSteadySolver3D,NativeSteadySettings
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux


def skew_mesh(n=3):
    points=np.array([[x,y,z] for z in np.linspace(0,1,n+1) for y in np.linspace(0,1,n+1) for x in np.linspace(0,1,n+1)])
    points=points@np.array([[1,.18,.12],[0,1,.2],[0,0,1]]).T
    def ix(x,y,z):return z*(n+1)**2+y*(n+1)+x
    cells=[]
    for z in range(n):
        for y in range(n):
            for x in range(n):cells.append([ix(x,y,z),ix(x+1,y,z),ix(x+1,y+1,z),ix(x,y+1,z),ix(x,y,z+1),ix(x+1,y,z+1),ix(x+1,y+1,z+1),ix(x,y+1,z+1)])
    return NativeMesh3D.from_cells(points,cells,['hexahedron']*len(cells),triangulate_faces=True)


def test_pressure_wall_extrapolation_does_not_impose_zero_normal_gradient():
    m=skew_mesh();pd=(m.neighbor<0)&(m.face_centers[:,0]>1)
    op=FreePressureReconstruction3D(m,pd);g=np.array([.7,-.3,.2]);p=1.2+m.cell_centers@g;pf=1.2+m.face_centers@g
    source,defect,force,_=op.sources(pf)
    np.testing.assert_allclose(op.face_matrix@p+source,pf,atol=2e-14)
    np.testing.assert_allclose(op.pressure_flux_defect@p+defect,0,atol=2e-14)
    np.testing.assert_allclose((op.pressure_force_matrix@p+force).reshape(-1,3),m.cell_volumes[:,None]*g,atol=2e-14)


@pytest.mark.parametrize('linear_method',['direct','block','schur'])
def test_complete_stokes_and_nonlinear_ns_solve_affine_field_with_physical_force(linear_method):
    m=skew_mesh();G=np.array([[.2,.4,.1],[-.3,-.1,.2],[.1,.1,-.1]])
    u=np.array([.3,-.1,.2])+m.cell_centers@G.T;bu=np.array([.3,-.1,.2])+m.face_centers@G.T;g=np.array([.7,-.3,.2]);p=m.cell_centers@g
    for convection in [False,True]:
        forcing=np.tile(g,(len(u),1))+(u@G.T if convection else 0)
        solver=NativeSteadySolver3D(m,m.neighbor<0,bu,body_force=forcing,gauge_value=p[0],settings=NativeSteadySettings(viscosity=.5,convection=convection,nonlinear_tolerance=1e-10,linear_method=linear_method))
        r=solver.solve();assert r['converged']
        np.testing.assert_allclose(r['velocity'],u,atol=2e-10)
        np.testing.assert_allclose(r['pressure'],p,atol=2e-9)
        assert r['history'][-1]['native_divergence_max']<1e-10
        f=solver.forces('boundary');np.testing.assert_allclose(f['pressure_force'],g*m.cell_volumes.sum(),atol=1e-9);np.testing.assert_allclose(f['viscous_force'],0,atol=1e-9)


def test_abc_force_independent_differential_identity_and_closed_boundary_quadrature():
    m=skew_mesh();points=m.cell_centers;f=abc_fields(points);eps=1e-5
    du=np.stack([(abc_fields(points+np.eye(3)[axis]*eps)['velocity']-abc_fields(points-np.eye(3)[axis]*eps)['velocity'])/(2*eps) for axis in range(3)],axis=2)
    np.testing.assert_allclose(du,f['velocity_gradient'],atol=1e-10)
    lap=sum((abc_fields(points+np.eye(3)[axis]*eps)['velocity']-2*f['velocity']+abc_fields(points-np.eye(3)[axis]*eps)['velocity'])/eps**2 for axis in range(3))
    reconstructed=np.einsum('cij,cj->ci',du,f['velocity'])+f['pressure_gradient']-.1*lap
    np.testing.assert_allclose(reconstructed,f['acceleration'],atol=2e-6)
    q=exact_boundary_flux(m,abc_fields);assert abs(q.sum())<1e-13
    with pytest.raises(ValueError,match='unbalanced'):NativeSteadySolver3D(m,m.neighbor<0,f['velocity'][m.owner],boundary_flux=q+1)


def test_nonaffine_rhie_chow_uses_actual_conservative_momentum_gradient():
    m=skew_mesh();op=FreePressureReconstruction3D(m,m.neighbor<0)
    p=np.sin(m.cell_centers[:,0]*2)+m.cell_centers[:,1]**2
    boundary=np.cos(m.face_centers[:,0])
    face_source,defect,force,_=op.sources(boundary)
    gauss=(op.pressure_force_matrix@p+force).reshape(-1,3)/m.cell_volumes[:,None]
    expected=np.einsum('ij,ij->i',op.I@gauss,m.area_vectors)
    compact=op.compact_flux@p
    # Recover the independently known compact boundary source from LS data.
    _,_,_,ls_source=op.sources(boundary)
    tangent=m.area_vectors-op.coefficient[:,None]*op.base.op.d
    compact_source=np.einsum('ij,ij->i',op.I@ls_source,tangent)
    compact_source[op.dirichlet]+=op.coefficient[op.dirichlet]*boundary[op.dirichlet]
    np.testing.assert_allclose(op.pressure_flux_defect@p+defect,expected-compact-compact_source,atol=2e-14)
