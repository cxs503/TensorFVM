from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_rans3d import NativeSSTRansSolver3D,NativeRansSettings,true_triangle_wall_distance,_triangle_distance
from tensorfvm.native_sst3d import wall_omega


def box(n=2):
    vertices=np.array([[x,y,z] for z in np.linspace(0,1,n+1) for y in np.linspace(0,1,n+1) for x in np.linspace(0,1,n+1)])
    ix=lambda x,y,z:z*(n+1)**2+y*(n+1)+x
    cells=[]
    for z in range(n):
        for y in range(n):
            for x in range(n):cells.append([ix(x,y,z),ix(x+1,y,z),ix(x+1,y+1,z),ix(x,y+1,z),ix(x,y,z+1),ix(x+1,y,z+1),ix(x+1,y+1,z+1),ix(x,y+1,z+1)])
    m=NativeMesh3D.from_cells(vertices,cells,['hexahedron']*len(cells),triangulate_faces=True)
    return replace(m,patch_labels=np.where(m.neighbor<0,'wall','internal'))


def solver(m,outer=1):
    return NativeSSTRansSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),settings=NativeSteadySettings(viscosity=.1,convection=False,max_iterations=10),rans_settings=NativeRansSettings(max_outer_iterations=outer,pseudo_time_step=1e-5,turbulence_relaxation=.4))


def test_true_triangle_distance_not_closest_centroid_and_edge_vertex_cases():
    vertices=np.array([[0,0,0],[10,0,0],[0,10,0],[0,0,1],[.6,0,1],[0,.6,1]])
    mesh=SimpleNamespace(vertices=vertices,face_vertices=((0,1,2),(3,4,5)),patch_labels=np.array(['wall','wall']),cell_centers=np.array([[.1,.1,.01]]))
    np.testing.assert_allclose(true_triangle_wall_distance(mesh),[.01],atol=1e-14)
    triangle=np.array([[[0,0,0],[1,0,0],[0,1,0]]]*3)
    points=np.array([[.25,.25,.3],[.5,-.2,0],[-.3,-.4,0]])
    np.testing.assert_allclose(_triangle_distance(points,triangle),[.3,.2,.5],atol=1e-14)


def test_box_nearest_surface_distance_independent_analytic_geometry():
    m=box();actual=true_triangle_wall_distance(m)
    exact=np.minimum(m.cell_centers,1-m.cell_centers).min(1)
    np.testing.assert_allclose(actual,exact,atol=2e-15)


def test_integrated_resting_box_update_preserves_force_and_reports_true_residual():
    m=box();s=solver(m)
    r=s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.)
    np.testing.assert_allclose(r['velocity'],0,atol=1e-12)
    np.testing.assert_allclose(r['flux'],0,atol=1e-12)
    assert np.all(r['k']>=0) and np.all(r['omega']>0)
    np.testing.assert_allclose(s.omega_boundary[s.wall],wall_omega(.1,s.wall_distance[m.owner[s.wall]]))
    np.testing.assert_allclose(s.viscous.face_viscosity[s.wall],.1,atol=0)
    last=r['history'][-1]
    assert last['stored_native_divergence_max']<1e-11
    assert last['frozen_transport_ledger']['k']['max_local_integrated_residual']<1e-11
    # A conservative linear solve is not convergence of original nonlinear SST.
    assert last['k_current_nonlinear_residual']>1e-3
    assert last['omega_current_nonlinear_residual']>1e-3
    assert r['numerically_converged'] is False
    assert r['physical_benchmark_passed'] is False
    f=s.forces();np.testing.assert_allclose(f['total_force'],0,atol=1e-11)
    np.testing.assert_allclose(f['total_force'],f['physical_pressure_force']+f['anisotropic_viscous_force']+f['isotropic_turbulent_force'],atol=1e-12)
    np.testing.assert_allclose(r['physical_pressure'],r['modified_pressure']-2/3*r['k'],atol=0)


def test_updated_viscosity_sources_and_gauss_velocity_gradient_are_current():
    m=box();s=solver(m);mu=np.linspace(.1,.2,s.F);s.update_dynamic_viscosity(mu)
    np.testing.assert_allclose(s.viscous.face_viscosity,mu)
    np.testing.assert_allclose(s._diffusion.toarray(),-s.viscous.momentum_diffusion_matrix.toarray())
    np.testing.assert_allclose(s.diffusion_source,s.viscous.vector_divergence_matrix@s.viscous.momentum_flux_source(dirichlet_values=s.velocity_boundary).ravel())
    # Independent affine gradient, with manufactured boundary temporarily set;
    # this tests gradient assembly, not physical turbulence in the resting box.
    G=np.array([[.1,.2,.3],[-.2,-.1,.4],[.5,.2,0.]])
    s.u=m.cell_centers@G.T;s.velocity_boundary=m.face_centers@G.T
    np.testing.assert_allclose(s.velocity_gradient(),np.broadcast_to(G,(s.N,3,3)),atol=1e-14)


def test_negative_initial_state_is_rejected_without_clipping():
    s=solver(box())
    with pytest.raises(ValueError):s.solve_rans(-.001,1.,inlet_k=.001,inlet_omega=1.)


def channel(n=3):
    m=box(n);b=m.neighbor<0;fc=m.face_centers
    labels=np.full(len(m.owner),'internal',dtype='<U10')
    labels[b]='farfield'
    labels[b&((np.abs(fc[:,1])<1e-12)|(np.abs(fc[:,1]-1)<1e-12))]='wall'
    labels[b&(np.abs(fc[:,0])<1e-12)]='inlet'
    labels[b&(np.abs(fc[:,0]-1)<1e-12)]='outlet'
    return replace(m,patch_labels=labels)


def test_nonzero_channel_inlet_no_slip_sst_coupled_pilot_is_not_false_convergence():
    m=channel();labels=m.patch_labels;fixed=(m.neighbor<0)&(labels!='outlet')
    bd=np.zeros((len(m.owner),3));bd[fixed&(labels!='wall'),0]=1.
    s=NativeSSTRansSolver3D(m,fixed,bd,labels=='outlet',settings=NativeSteadySettings(viscosity=.05,max_iterations=60,convection_order=1,nonlinear_tolerance=1e-8),rans_settings=NativeRansSettings(max_outer_iterations=2,pseudo_time_step=1e-4,turbulence_relaxation=.4))
    r=s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.)
    assert np.max(r['velocity'][:,0])>.5
    assert np.all(np.isfinite(r['physical_pressure']))
    assert r['history'][-1]['stored_native_divergence_max']<1e-7
    assert np.max(np.abs(r['flux'][labels=='wall']))<1e-12
    assert np.all(r['k']>=0) and np.all(r['omega']>0)
    assert r['history'][-1]['k_current_nonlinear_residual']>1e-3
    assert r['numerically_converged'] is False
    f=s.forces();assert abs(f['total_force'][0])>1e-3
    np.testing.assert_allclose(f['total_force'],f['physical_pressure_force']+f['anisotropic_viscous_force']+f['isotropic_turbulent_force'],atol=1e-12)
    expected=s.physical_pressure_boundary+2/3*s._scalar_faces(s.k,s.k_boundary)
    np.testing.assert_allclose(s.pressure_boundary[labels=='outlet'],expected[labels=='outlet'])
    face,defect,force,_=s.pressure.sources(expected)
    np.testing.assert_allclose(s.p_face_source,face)
    np.testing.assert_allclose(s.p_defect_source,defect)
    np.testing.assert_allclose(s.p_force_source,force)


def test_rejected_turbulence_update_retries_smaller_dt_without_clipping(monkeypatch):
    s=solver(box());original=s.transport.step;steps=[]
    def fail_then_real(*args,**kwargs):
        steps.append(args[6])
        if len(steps)<3:raise RuntimeError('invalid SST update; no clipping applied')
        return original(*args,**kwargs)
    monkeypatch.setattr(s.transport,'step',fail_then_real)
    r=s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.)
    np.testing.assert_allclose(steps,[1e-5,5e-6,2.5e-6])
    assert r['history'][0]['step_retries']==2


def test_nonzero_channel_converges_original_sst_equations_with_unchanged_gates():
    m=channel();labels=m.patch_labels;fixed=(m.neighbor<0)&(labels!='outlet')
    bd=np.zeros((len(m.owner),3));bd[fixed&(labels!='wall'),0]=1.
    s=NativeSSTRansSolver3D(m,fixed,bd,labels=='outlet',settings=NativeSteadySettings(viscosity=.05,max_iterations=60,convection_order=1,nonlinear_tolerance=1e-8),rans_settings=NativeRansSettings(max_outer_iterations=60,pseudo_time_step=1.,turbulence_relaxation=.7,nonlinear_tolerance=1e-7))
    r=s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.)
    assert r['numerically_converged'] is True
    last=r['history'][-1]
    keys=['momentum_current_residual','current_reconstructed_divergence_max','stored_native_divergence_max','face_flux_consistency_max','velocity_change','modified_pressure_change','flux_change','k_change','omega_change','k_current_nonlinear_residual','omega_current_nonlinear_residual']
    assert max(last[k] for k in keys)<1e-7
    # Current original nonlinear equations are independently recomputed here;
    # frozen linearized step ledgers are not the tested residual.
    original=s.turbulence_residual()
    assert original['k']['relative_residual']<1e-7
    assert original['omega']['relative_residual']<1e-7
    assert r['physical_benchmark_passed'] is False
