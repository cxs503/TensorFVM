"""Pressure-driven fully developed half-channel on the native hybrid backend.

Independent imposed total shear is tau_xy=1-y and p*=1-x. Velocity and scalar
inflow are extrapolated from current cells, never supplied from DNS profiles.
Symmetry extrusion is deliberate: this verifies a 1D physical channel profile,
not a genuinely three-dimensional turbulent simulation.
"""
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_boundary_constraints3d import require_rigid_motion_constraints
from tensorfvm.native_hybrid_rans3d import NativeHybridRans3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_sst3d import wall_omega
from .native_hybrid_flow import channel_mesh


def developed_channel_flow(n=32,*,nu=1.,stretch=3.,triangulate_faces=False,axial=2,spanwise=2,length=1.,span=1.):
    if n<2 or not np.isfinite(stretch) or stretch<0 or not np.isfinite(nu) or nu<=0:
        raise ValueError('n>=2, positive viscosity and nonnegative stretch required')
    if not np.isfinite([length,span]).all() or min(length,span)<=0:raise ValueError('positive domain lengths required')
    original=channel_mesh((axial,n,spanwise),triangulate_faces=False)
    points=original.vertices.copy()
    if stretch:points[:,1]=np.expm1(stretch*points[:,1])/np.expm1(stretch)
    points[:,0]*=length;points[:,2]*=span
    mesh=NativeMesh3D.from_cells(points,original.cells,original.cell_types,triangulate_faces=triangulate_faces)
    b=mesh.neighbor<0;x,y,z=mesh.face_centers.T
    wall=b & (y<1e-12);center=b & (y>1-1e-12)
    sides=b & ((z<1e-12)|(z>span-1e-12));ends=b & ((x<1e-12)|(x>length-1e-12))
    mask=np.zeros((len(b),3),bool);mask[wall]=True;mask[center,1]=True;mask[sides,2]=True
    normal=mesh.area_vectors/np.linalg.norm(mesh.area_vectors,axis=1)[:,None]
    traction=-(length-x[:,None])*normal
    traction[:,0]+=(1-y)*normal[:,1]
    traction[:,1]+=(1-y)*normal[:,0]
    require_rigid_motion_constraints(mesh,mask)
    flow=NativeHybridNavierStokes3D(mesh,mask,0.,viscosity=nu,boundary_traction=traction,
        backflow_velocity=0.,convection_order=2)
    flow.initialize_stokes()
    update_developed_inflow(flow)
    return mesh,flow,dict(wall=wall,center=center,sides=sides,ends=ends,nu=float(nu),stretch=float(stretch),length=float(length),span=float(span),axial=axial,spanwise=spanwise)


def update_developed_inflow(flow):
    """Zero axial profile derivative; explicit fixed-point boundary update."""
    previous=flow.convection.inflow.copy()
    cells=flow.state[:3*flow.N].reshape(flow.N,3)
    ends=flow.open_normal_boundary
    flow.convection.inflow[ends]=cells[flow.mesh.owner[ends]]
    return float(np.linalg.norm(flow.convection.inflow-previous)/max(np.linalg.norm(flow.convection.inflow),1.))


def developed_channel_rans(n=32,*,re_tau=182.088,stretch=3.,scalar_correction='none',axial=2,spanwise=2,length=1.,span=1.):
    nu=1./re_tau
    mesh,flow,info=developed_channel_flow(n,nu=nu,stretch=stretch,axial=axial,spanwise=spanwise,length=length,span=span)
    wall=info['wall'];distance=mesh.cell_centers[:,1]
    omega_boundary=np.ones(flow.F)
    omega_boundary[wall]=wall_omega(nu,distance[mesh.owner[wall]])
    # Positive reaction-diffusion initial guess, not DNS/target interior values.
    seed=6*nu/(.075*distance**2)+1/(.3*.41*distance)
    omega=NativeHybridScalar3D(mesh,nu,dirichlet_mask=wall).solve(
        sink=.075*seed,boundary_values=omega_boundary)['cell_values']
    solver=NativeHybridRans3D(flow,distance,nu,k_dirichlet_mask=wall,omega_dirichlet_mask=wall,
        k_boundary=0.,omega_boundary=omega_boundary,k_inflow=.1,omega_inflow=omega[mesh.owner],
        scalar_correction=scalar_correction)
    solver.initialize(.1,omega)
    update_developed_rans_inflow(solver)
    info.update(re_tau=float(re_tau))
    return mesh,solver,info


def update_developed_rans_inflow(solver):
    changes={'velocity':update_developed_inflow(solver.flow)}
    for name in ('k','omega'):
        old=np.broadcast_to(np.asarray(solver.boundary[name+'_inflow'],float),(solver.flow.F,))
        new=np.asarray(getattr(solver,name),float)[solver.mesh.owner].copy()
        changes[name]=float(np.linalg.norm(new-old)/max(np.linalg.norm(new),1e-30))
        solver.boundary[name+'_inflow']=new
    return changes


def solve_developed_rans(solver,*,max_iterations=400,dt=1.,relaxation=.5,tolerance=1e-8,callback=None,momentum_method='coupled'):
    """Developed-flow predictor, accepted using the FULL native RANS equations.

    Convection vanishes analytically for this channel. The predictor omits it;
    acceptance always uses actual nonzero face volume flux, updated extrapolated
    inflow, current viscosity, and both scalar free-face equations. This is a
    specialized iteration for the physical channel, not a general NS algorithm.
    """
    if not np.isfinite([dt,relaxation,tolerance]).all() or dt<=0 or not 0<relaxation<=1 or tolerance<=0 or max_iterations<1:
        raise ValueError('positive iteration controls and relaxation in (0,1] required')
    if momentum_method not in ('coupled','simplec','simple'):raise ValueError('unknown momentum predictor')
    from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
    s=solver;s.converged=False;s.tolerance=tolerance;failure=None
    for iteration in range(max_iterations):
        old={k:getattr(s,k).copy() for k in ('k','omega','k_faces','omega_faces')}
        old_u=s.flow.state.copy();old_p=s.flow.p.copy()
        try:
            s._refresh_viscosity()
            if momentum_method=='coupled':s.flow.initialize_stokes(tolerance=tolerance)
            else:
                f=s.flow;f.A=f.viscous.matrix
                f.rhs=(f.load-f.viscous.full_matrix@f.prescribed)[f.viscous.free]
                # The predictor is linear viscous flow; no nonlinear inertia damping.
                # Keep the general NS solver's relaxation basis unchanged.
                basis=f.momentum_relaxation_basis
                try:
                    f.momentum_relaxation_basis='diagonal'
                    NativeHybridStokes3D.solve(f,algorithm=momentum_method,relaxation=1.,
                        pressure_relaxation=1.,tolerance=tolerance*.1)
                finally:f.momentum_relaxation_basis=basis
                if not f.converged:raise RuntimeError('developed momentum predictor did not converge')
            update_developed_rans_inflow(s)
            step=s.transport.step(s.k,s.omega,s.k_faces,s.omega_faces,0.,s.wall_distance,s.nu,
                s.flow.velocity_gradient(),dt,rho=s.rho,**s.boundary)
            for k in old:setattr(s,k,old[k]+relaxation*(step[k]-old[k]))
        except RuntimeError as exc:
            failure=f'{type(exc).__name__}: {exc}'
        boundary=update_developed_rans_inflow(s);current=s.current_residual()
        changes={k:float(np.linalg.norm(getattr(s,k)-old[k])/max(np.linalg.norm(getattr(s,k)),1e-30)) for k in old}
        changes['velocity']=float(np.linalg.norm(s.flow.state-old_u)/max(np.linalg.norm(s.flow.state),1.))
        changes['modified_pressure']=float(np.linalg.norm(s.flow.p-old_p)/max(np.linalg.norm(s.flow.p),1.))
        ledgers={name:max(current['transport'][name]['cell_equation_max'],
            current['transport'][name]['free_face_equation_max'],current['transport'][name]['interface_flux_defect_max'],
            abs(current['transport'][name]['global_balance'])) for name in ('k','omega')}
        row=dict(iteration=len(s.history)+1,**{k:v for k,v in current.items() if k!='transport'},
            changes=changes,inflow_changes=boundary,scalar_absolute_ledger=ledgers,failure=failure,
            minimum_k=float(min(s.k.min(),s.k_faces.min())),minimum_omega=float(min(s.omega.min(),s.omega_faces.min())))
        s.history.append(row)
        s.converged=bool(failure is None and max(row[k] for k in ('momentum','divergence','gauge','k','omega','k_boundary','omega_boundary'))<tolerance
            and max(changes.values())<tolerance and max(boundary.values())<tolerance and max(ledgers.values())<1e-8)
        if callback:callback(row)
        if s.converged or failure is not None:break
    s.flow.nonlinear_converged=s.converged
    result=s.result()
    result.update(nonlinear_method='fully_developed_predictor_with_full_native_rans_acceptance',
        momentum_predictor=momentum_method,failure=failure,
        momentum_predictor_settings={'relaxation':1.,'pressure_relaxation':1.,'basis':'diagonal'} if momentum_method!='coupled' else {},
        predictor_convection='zero by developed-flow ansatz; actual flux used for final residuals',
        scalar_absolute_ledger=ledgers,inflow_changes=boundary)
    return result
