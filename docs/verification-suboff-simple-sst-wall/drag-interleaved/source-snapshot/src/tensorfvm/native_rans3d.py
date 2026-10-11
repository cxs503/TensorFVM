"""Opt-in native 3D steady SST-2003 coupling; no physical qualification.

Momentum solves modified pressure pi=p+2*rho*k/3 and total anisotropic
molecular/eddy stress. Physical pressure and isotropic Reynolds stress are
converted together, so reported total boundary force preserves the same flux.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .native_collocated3d import NativeSteadySolver3D
from .native_viscous3d import NativeViscous3D
from .native_sst3d import NativeSSTTransport3D,sst2003_closure,wall_omega,VARIANT,BETA1,BETA_STAR
from .wall_functions import wall_traction,KAPPA
from .native_simple3d import NativeSimpleMixin


def _triangle_distance(points,triangles):
    """Pairwise point/triangle Euclidean distances, including edges and vertices."""
    p=np.asarray(points);t=np.asarray(triangles);a,b,c=t[:,0],t[:,1],t[:,2]
    ab=b-a;ac=c-a;normal=np.cross(ab,ac);norm2=np.sum(normal*normal,axis=1)
    if np.any(norm2<=0):raise ValueError('degenerate wall triangle')
    ap=p-a;height=np.sum(ap*normal,axis=1)/norm2
    projected=ap-height[:,None]*normal
    # Cross-product barycentrics avoid subtracting nearly equal Gram terms
    # on thin but nondegenerate boundary-layer triangles.
    u=np.sum(np.cross(projected,ac)*normal,axis=1)/norm2
    v=np.sum(np.cross(ab,projected)*normal,axis=1)/norm2
    inside=(u>=0)&(v>=0)&(u+v<=1)
    distance=np.where(inside,height*height*norm2,np.inf)
    for left,right in [(a,b),(b,c),(c,a)]:
        edge=right-left;s=np.clip(np.sum((p-left)*edge,axis=1)/np.sum(edge*edge,axis=1),0,1)
        distance=np.minimum(distance,np.sum((p-left-s[:,None]*edge)**2,axis=1))
    return np.sqrt(distance)


def true_triangle_wall_distance(mesh,points=None,*,patch='wall',block_size=128):
    """Exact nearest distance to actual planar wall facets, accelerated by a KD tree.

    A closest-centroid candidate establishes an upper bound d0. Every triangle
    that can improve it has centroid distance <=d0+its bounding radius; querying
    d0+global maximum radius therefore cannot miss the true nearest facet.
    """
    face_ids=np.flatnonzero(mesh.patch_labels==patch)
    if not len(face_ids):raise ValueError('a nonempty physical wall patch is required')
    if any(len(mesh.face_vertices[f])!=3 for f in face_ids):raise ValueError('wall distance requires actual native triangle facets')
    triangles=mesh.vertices[np.asarray([mesh.face_vertices[f] for f in face_ids])]
    centers=triangles.mean(1);radius=np.linalg.norm(triangles-centers[:,None,:],axis=2).max()
    tree=cKDTree(centers);p=mesh.cell_centers if points is None else np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or block_size<1:raise ValueError('finite (n,3) points and positive block size required')
    result=np.empty(len(p))
    for start in range(0,len(p),block_size):
        block=p[start:start+block_size];_,nearest=tree.query(block)
        upper=_triangle_distance(block,triangles[nearest])
        candidates=tree.query_ball_point(block,np.nextafter(upper+radius,np.inf))
        counts=np.array([len(c) for c in candidates]);point_ids=np.repeat(np.arange(len(block)),counts)
        triangle_ids=np.concatenate(candidates)
        distances=_triangle_distance(block[point_ids],triangles[triangle_ids])
        best=np.full(len(block),np.inf);np.minimum.at(best,point_ids,distances);result[start:start+len(block)]=best
    return result


@dataclass(frozen=True)
class NativeRansSettings:
    max_outer_iterations: int=50
    pseudo_time_step: float=1e-3
    turbulence_relaxation: float=.5
    nonlinear_tolerance: float=1e-7
    maximum_step_retries: int=8
    wall_treatment: str="resolved"


class NativeSSTRansSolver3D(NativeSteadySolver3D):
    def __init__(self,*args,rans_settings=None,wall_patch='wall',**kwargs):
        super().__init__(*args,**kwargs)
        self.rans_settings=rans_settings or NativeRansSettings();r=self.rans_settings
        if r.max_outer_iterations<1 or r.maximum_step_retries<0 or not np.isfinite([r.pseudo_time_step,r.turbulence_relaxation,r.nonlinear_tolerance]).all() or r.pseudo_time_step<=0 or not 0<r.turbulence_relaxation<=1 or r.nonlinear_tolerance<=0:raise ValueError('positive finite RANS iteration settings required')
        if r.wall_treatment not in ("resolved","spalding"):raise ValueError("wall_treatment must be resolved or spalding")
        self.wall=self.mesh.patch_labels==wall_patch
        if not np.all(self.fixed[self.wall]):raise ValueError('SST wall requires prescribed no-slip velocity')
        if not np.allclose(self.velocity_boundary[self.wall],0,rtol=0,atol=1e-14):raise ValueError('stationary SST wall velocity must be zero')
        self.wall_distance=true_triangle_wall_distance(self.mesh,patch=wall_patch)
        if np.any(self.wall_distance<=0):raise ValueError('positive cell-to-wall distances required')
        self.nu=self.settings.viscosity/self.settings.density
        self.scalar_fixed=self.fixed & ~self.wall if r.wall_treatment=="spalding" else self.fixed
        self.transport=NativeSSTTransport3D(self.mesh,self.scalar_fixed)
        self.physical_pressure_boundary=np.broadcast_to(np.asarray(self.pressure_boundary,dtype=float),(self.F,)).copy()
        self.physical_gauge_value=self.gauge_value
        self.k=None;self.omega=None;self.rans_history=[]

    def update_dynamic_viscosity(self,face_viscosity):
        """Update stress, its boundary source and RC mobility without changing topology."""
        self.viscous=NativeViscous3D(self.mesh,self.fixed,face_viscosity)
        self._diffusion=(-self.viscous.momentum_diffusion_matrix).tocsr()
        self.diffusion_source=self.viscous.vector_divergence_matrix@self.viscous.momentum_flux_source(dirichlet_values=self.velocity_boundary).ravel()
        coefficient=self.viscous.face_viscosity*np.sum(self.mesh.area_vectors**2,axis=1)/np.sum(self.mesh.area_vectors*self.viscous.op.d,axis=1)
        self._mobility_reference=(-self.viscous.divergence_matrix@sparse.diags(coefficient)@self.viscous.delta_matrix).diagonal()*10/9

    def _scalar_gradient(self,field,boundary):
        b=self.transport.base
        return np.column_stack([g@field for g in b.gradient_matrices])+b.gradient_source(dirichlet_values=boundary)

    def _scalar_faces(self,field,boundary):
        b=self.transport.base;m=self.mesh;gradient=self._scalar_gradient(field,boundary)
        offset=m.face_centers-m.cell_centers[m.owner]-b.weight[:,None]*b.op.d
        nb=b.neumann;offset[nb]-=b.normal[nb]*np.sum(offset[nb]*b.normal[nb],axis=1)[:,None]
        result=np.asarray(b.interpolation@field)+np.sum(offset*(b.interpolation@gradient),axis=1)
        result[self.scalar_fixed]=boundary[self.scalar_fixed];return result

    def velocity_gradient(self):
        return np.stack([np.column_stack([g@self.u[:,k] for g in self.viscous.gradient_matrices])+self.viscous.gradient_source(dirichlet_values=self.velocity_boundary[:,k]) for k in range(3)],axis=1)

    def current_closure(self):
        closure=sst2003_closure(self.k,self.omega,self.wall_distance,self.nu,self.velocity_gradient(),self._scalar_gradient(self.k,self.k_boundary),self._scalar_gradient(self.omega,self.omega_boundary),rho=self.settings.density)
        if self.rans_settings.wall_treatment=='spalding':
            ids,omega,production=self.wall_model()
            closure['production_limited'][ids]=np.minimum(production,closure['production_limit'][ids])
            closure['k_source']=closure['production_limited']-closure['k_destruction']
            # omega equation is constrained in these cells, with its reaction
            # reported separately from conservative unconstrained transport.
        return closure

    def wall_model(self):
        m=self.mesh;faces=np.flatnonzero(self.wall);owner=m.owner[faces]
        area=np.linalg.norm(m.area_vectors[faces],axis=1);n=m.area_vectors[faces]/area[:,None]
        tangent=self.u[owner]-n*np.sum(self.u[owner]*n,axis=1)[:,None]
        y=self.wall_distance[owner]
        law=wall_traction(tangent,y,self.settings.density,self.settings.viscosity)
        ut=law['friction_velocity_m_s'];yp=law['y_plus']
        omega_vis=6*self.nu/(BETA1*y*y)
        omega_log=np.sqrt(self.k[owner])/(BETA_STAR**.25*KAPPA*y)
        omega=np.hypot(omega_vis,omega_log)
        # Equilibrium log-layer production; smoothly vanish at the viscous wall.
        blend=1-np.exp(-(yp/11.)**2)
        production=blend*ut**3/(KAPPA*y)
        weights=np.bincount(owner,weights=area,minlength=self.N);ids=np.flatnonzero(weights)
        average=lambda value:np.bincount(owner,weights=area*value,minlength=self.N)[ids]/weights[ids]
        return ids,average(omega),average(production)

    def _apply_wall_traction(self):
        m=self.mesh;v=self.viscous;faces=np.flatnonzero(self.wall);owner=m.owner[faces]
        area=np.linalg.norm(m.area_vectors[faces],axis=1);n=m.area_vectors[faces]/area[:,None]
        tangent=self.u[owner]-n*np.sum(self.u[owner]*n,axis=1)[:,None]
        y=self.wall_distance[owner];speed=np.linalg.norm(tangent,axis=1)
        law=wall_traction(tangent,y,self.settings.density,self.settings.viscosity)
        drag=np.divide(self.settings.density*law['friction_velocity_m_s']**2,speed,
            out=self.settings.viscosity/y,where=speed>0)
        normal=n[:,:,None]*n[:,None,:];project=np.eye(3)-normal
        # Retain normal viscous traction; replace only tangential stress by
        # the same implicit Robin law used in the reported force.
        rows=(3*faces[:,None,None]+np.arange(3)[None,:,None]+np.zeros((1,1,3),int)).ravel()
        cols=(3*faces[:,None,None]+np.arange(3)[None,None,:]+np.zeros((1,3,1),int)).ravel()
        selector=sparse.eye(3*self.F,format='csr')-sparse.coo_matrix((project.ravel(),(rows,cols)),shape=(3*self.F,3*self.F)).tocsr()
        cellcols=(3*owner[:,None,None]+np.arange(3)[None,None,:]+np.zeros((1,3,1),int)).ravel()
        robin=sparse.coo_matrix(((-area[:,None,None]*drag[:,None,None]*project).ravel(),(rows,cellcols)),shape=(3*self.F,3*self.N)).tocsr()
        orth_coefficient=v.face_viscosity[faces]*area**2/np.sum(m.area_vectors[faces]*v.op.d[faces],axis=1)
        implicit_difference=(area*drag-orth_coefficient)[:,None,None]*project
        cellrows=(3*owner[:,None,None]+np.arange(3)[None,:,None]+np.zeros((1,1,3),int)).ravel()
        self.wall_implicit_correction=sparse.coo_matrix((implicit_difference.ravel(),(cellrows,cellcols)),shape=(3*self.N,3*self.N)).tocsr()
        self.wall_flux_matrix=selector@v.momentum_flux_matrix+robin
        self.wall_flux_source=selector@v.momentum_flux_source(dirichlet_values=self.velocity_boundary).ravel()
        self._diffusion=-v.vector_divergence_matrix@self.wall_flux_matrix
        self.diffusion_source=v.vector_divergence_matrix@self.wall_flux_source

    def _update_momentum_coefficients(self,closure):
        rho=self.settings.density
        face_nut=np.asarray(self.viscous.interpolation@closure['nu_t'])
        if not np.isfinite(face_nut).all() or np.any(face_nut<0):raise RuntimeError('invalid face eddy viscosity; no clipping applied')
        mu=rho*(self.nu+face_nut)
        # Low-Re SST resolves the wall: eddy viscosity is exactly zero on wall faces.
        mu[self.wall]=self.settings.viscosity
        self.update_dynamic_viscosity(mu)
        if self.rans_settings.wall_treatment=="spalding":self._apply_wall_traction()
        kface=self._scalar_faces(self.k,self.k_boundary)
        self.pressure_boundary=self.physical_pressure_boundary+2/3*rho*kface
        self.p_face_source,self.p_defect_source,self.p_force_source,_=self.pressure.sources(self.pressure_boundary)
        if self.gauge:self.gauge_value=self.physical_gauge_value+2/3*rho*self.k[0]

    def turbulence_residual(self,closure=None):
        """Recompute the actual current nonlinear steady sources and fluxes."""
        closure=self.current_closure() if closure is None else closure
        result={};m=self.mesh;b=self.transport.base
        for name,value,boundary,sigma,source in [
            ('k',self.k,self.k_boundary,closure['sigma_k'],closure['k_source']),
            ('omega',self.omega,self.omega_boundary,closure['sigma_omega'],closure['omega_source'])]:
            coefficient=self.nu+sigma*closure['nu_t']
            diffusion=NativeViscous3D(m,self.scalar_fixed,coefficient)
            adv,adv_source=self.transport._upwind(self.flux,boundary)
            af=adv@value+adv_source;df=diffusion.scalar_flux(value,dirichlet_values=boundary)
            integrated=b.divergence_matrix@(af-df)-m.cell_volumes*source
            scale=np.linalg.norm(b.divergence_matrix@af)+np.linalg.norm(b.divergence_matrix@df)+np.linalg.norm(m.cell_volumes*source)
            reaction=0.;constraint_residual=0.
            if name=='omega' and self.rans_settings.wall_treatment=='spalding':
                ids,values,_=self.wall_model();reaction=float(integrated[ids].sum())
                integrated[ids]=0.
                constraint_residual=float(np.max(np.abs(value[ids]-values)/values))
            result[name]=dict(wall_constraint_reaction=reaction,wall_constraint_relative_residual=constraint_residual,relative_residual=max(constraint_residual,float(np.linalg.norm(integrated)/max(scale,np.finfo(float).tiny))),maximum_specific_residual=float(np.max(np.abs(integrated/m.cell_volumes))),global_integrated_residual=float(integrated.sum()),boundary_advective_flux=float(af[b.boundary].sum()),boundary_diffusive_flux=float(df[b.boundary].sum()),integrated_current_source=float(np.sum(m.cell_volumes*source)))
        return result

    def solve_rans(self,initial_k,initial_omega,*,inlet_k,inlet_omega,callback=None,momentum_callback=None):
        self.k=np.broadcast_to(np.asarray(initial_k,dtype=float),(self.N,)).copy();self.omega=np.broadcast_to(np.asarray(initial_omega,dtype=float),(self.N,)).copy()
        self.k_boundary=np.broadcast_to(np.asarray(inlet_k,dtype=float),(self.F,)).copy();self.omega_boundary=np.broadcast_to(np.asarray(inlet_omega,dtype=float),(self.F,)).copy()
        if self.rans_settings.wall_treatment=='resolved':
            self.k_boundary[self.wall]=0.
            self.omega_boundary[self.wall]=wall_omega(self.nu,self.wall_distance[self.mesh.owner[self.wall]])
        if not np.isfinite(self.k_boundary).all() or not np.isfinite(self.omega_boundary).all() or np.any(self.k_boundary<0) or np.any(self.omega_boundary<=0):raise ValueError('nonnegative k and positive omega boundary values required')
        r=self.rans_settings;converged=False
        for iteration in range(r.max_outer_iterations):
            old_u=self.u.copy();old_p=self.p.copy();old_flux=self.flux.copy();old_k=self.k.copy();old_w=self.omega.copy()
            closure=self.current_closure();self._update_momentum_coefficients(closure)
            def inner_callback(row):
                if momentum_callback:momentum_callback(dict(outer_iteration=iteration+1,**row))
            momentum=self.solve(callback=inner_callback)
            # If an implicit turbulence solve proposes a negative state, reduce
            # pseudo dt and retry from the same old k/omega; never clip values.
            dt=r.pseudo_time_step;update=None
            extra={}
            if r.wall_treatment=='spalding':
                ids,values,_=self.wall_model()
                extra=dict(closure_override=self.current_closure(),omega_constraints=(ids,values))
            for retry in range(r.maximum_step_retries+1):
                try:
                    update=self.transport.step(self.k,self.omega,self.flux,self.wall_distance,self.nu,self.velocity_gradient(),dt,k_boundary=self.k_boundary,omega_boundary=self.omega_boundary,rho=self.settings.density,**extra)
                    break
                except RuntimeError:
                    if retry==r.maximum_step_retries:raise
                    dt*=.5
            self.k=(1-r.turbulence_relaxation)*self.k+r.turbulence_relaxation*update['k']
            self.omega=(1-r.turbulence_relaxation)*self.omega+r.turbulence_relaxation*update['omega']
            closure=self.current_closure();self._update_momentum_coefficients(closure)
            current,target,_,correction,qsource=self._system(self.flux)
            residual=current@np.r_[self.u.ravel(),self.p]-target
            momentum_residual=float(np.linalg.norm(residual[:3*self.N])/max(np.linalg.norm(self.force_integrated)+np.linalg.norm(self.diffusion_source)+np.linalg.norm(self.p_force_source),1.))
            consistent=self.velocity_flux_matrix@self.u.ravel()+correction@self.p+qsource
            continuum=float(np.max(np.abs(self.viscous.op.integrate(consistent)/self.mesh.cell_volumes)))
            coupling=float(np.max(np.abs(consistent-self.flux)/np.linalg.norm(self.mesh.area_vectors,axis=1)))
            change=lambda new,old:float(np.linalg.norm(new-old)/max(np.linalg.norm(new),np.finfo(float).tiny))
            turbulence=self.turbulence_residual(closure)
            gauge_residual=abs(self.p[0]-self.gauge_value) if self.gauge else 0.
            row=dict(iteration=iteration+1,momentum_inner_converged=momentum['converged'],momentum_current_residual=momentum_residual,modified_pressure_gauge_residual=float(gauge_residual),current_reconstructed_divergence_max=continuum,stored_native_divergence_max=float(np.max(np.abs(self.viscous.op.integrate(self.flux)/self.mesh.cell_volumes))),face_flux_consistency_max=coupling,velocity_change=change(self.u,old_u),modified_pressure_change=float(np.linalg.norm(self.p-old_p)/max(np.linalg.norm(self.p),1.)),flux_change=change(self.flux,old_flux),k_change=change(self.k,old_k),omega_change=change(self.omega,old_w),k_current_nonlinear_residual=turbulence['k']['relative_residual'],omega_current_nonlinear_residual=turbulence['omega']['relative_residual'],pseudo_time_step_used=dt,step_retries=retry,turbulence_current_ledger=turbulence,frozen_transport_ledger=update['ledger'])
            self.rans_history.append(row)
            if callback:callback(row)
            checks=[momentum_residual,gauge_residual,continuum,coupling,row['stored_native_divergence_max'],row['velocity_change'],row['modified_pressure_change'],row['flux_change'],row['k_change'],row['omega_change'],row['k_current_nonlinear_residual'],row['omega_current_nonlinear_residual']]
            if momentum['converged'] and max(checks)<r.nonlinear_tolerance:converged=True;break
        self.rans_converged=converged
        return dict(numerically_converged=converged,physical_benchmark_passed=False,variant=VARIANT,velocity=self.u.copy(),modified_pressure=self.p.copy(),physical_pressure=self.p-2/3*self.settings.density*self.k,k=self.k.copy(),omega=self.omega.copy(),flux=self.flux.copy(),wall_distance=self.wall_distance.copy(),history=self.rans_history)

    def forces(self,patch='wall'):
        if self.k is None:raise RuntimeError('initialize RANS state before evaluating RANS forces')
        base=super().forces(patch);mask=self.mesh.patch_labels==patch
        if self.rans_settings.wall_treatment=='spalding':
            traction=(self.wall_flux_matrix@self.u.ravel()+self.wall_flux_source).reshape(self.F,3)
            base['viscous_traction_integrated']=traction
            base['viscous_force']=-traction[mask].sum(0)
            base['total_force']=base['pressure_force']+base['viscous_force']
        kfaces=self._scalar_faces(self.k,self.k_boundary);isotropic=2/3*self.settings.density*kfaces[:,None]*self.mesh.area_vectors
        physical=base['pressure_faces']-2/3*self.settings.density*kfaces
        pressure_force=(physical[mask,None]*self.mesh.area_vectors[mask]).sum(0)
        isotropic_force=isotropic[mask].sum(0)
        return dict(total_force=base['total_force'],physical_pressure_force=pressure_force,anisotropic_viscous_force=base['viscous_force'],isotropic_turbulent_force=isotropic_force,modified_pressure_force=base['pressure_force'],physical_pressure_faces=physical,modified_pressure_faces=base['pressure_faces'],isotropic_reynolds_stress_traction=-isotropic,anisotropic_stress_traction=base['viscous_traction_integrated'])


class NativeSimpleSSTRansSolver3D(NativeSimpleMixin,NativeSSTRansSolver3D):
    """SIMPLE pressure/velocity coupling with shared SST transport and wall laws."""
