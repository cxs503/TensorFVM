"""Conforming annular Mesh2D and fully developed axial laminar FV backend.

The three-dimensional physical flow has only axial velocity, independent of z.
Its exact reduction is a two-dimensional scalar cross-section problem. The full
cross-section sparse matrix is assembled and solved; no ring averaging or
analytic initial velocity is used. Diffusion uses the production fitted kernel.
"""
from dataclasses import dataclass,asdict
import math
import time
import numpy as np
import torch
from .backend_registry import SolverBackend,register_backend
from .mesh_api import validate_mesh2d
from .solver import SolverConfig
from .body_fitted import BodyFittedSolver


@dataclass(frozen=True)
class AnnularConfig:
    ntheta:int=256
    nr:int=64
    inner_radius_m:float=.05
    outer_radius_m:float=.1
    viscosity_pa_s:float=.001
    density_kg_m3:float=1.
    pressure_gradient_pa_m:float=.01
    linear_tolerance:float=1e-11
    max_linear_iterations:int=5000
    def __post_init__(self):
        if self.ntheta<8 or self.nr<2:raise ValueError('annular grid is too small')
        if not 0<self.inner_radius_m<self.outer_radius_m:raise ValueError('invalid annular radii')
        if any(not math.isfinite(v) or v<=0 for v in (self.viscosity_pa_s,self.density_kg_m3,self.pressure_gradient_pa_m,self.linear_tolerance)):raise ValueError('positive finite SI parameters required')


class AnnularMesh:
    """Polygonal inner/outer no-slip walls; periodic conforming angular faces."""
    def __init__(self,c):
        # The registered factory reads explicitly attached annular parameters.
        ac=c.annular_config;n,nr=ac.ntheta,ac.nr;self.field_shape=(nr,n)
        theta=torch.linspace(0,2*math.pi,n+1,dtype=torch.float64);radius=torch.linspace(ac.inner_radius_m,ac.outer_radius_m,nr+1,dtype=torch.float64)
        rr,tt=torch.meshgrid(radius,theta,indexing='ij');self.vertices=torch.stack((rr*torch.cos(tt),rr*torch.sin(tt)),-1)
        v=self.vertices;poly=torch.stack((v[:-1,:-1],v[1:,:-1],v[1:,1:],v[:-1,1:]),-2);nxt=poly.roll(-1,-2)
        cross=poly[...,0]*nxt[...,1]-nxt[...,0]*poly[...,1];self.volumes=.5*cross.sum(-1);self.centers=((poly+nxt)*cross[...,None]).sum(-2)/(6*self.volumes[...,None])
        j,i=torch.meshgrid(torch.arange(nr+1),torch.arange(n),indexing='ij');ro=((j-1).clamp_min(0)*n+i).flatten();rn=torch.where((j>0)&(j<nr),j*n+i,-1).flatten()
        ra=v[:,:-1].reshape(-1,2);rb=v[:,1:].reshape(-1,2);inner=(j==0).flatten();first=torch.where(inner[:,None],rb,ra);second=torch.where(inner[:,None],ra,rb)
        radial_end=torch.stack((first,second),1)
        j,i=torch.meshgrid(torch.arange(nr),torch.arange(n),indexing='ij');ao=(j*n+(i-1)%n).flatten();an=(j*n+i).flatten();angular_end=torch.stack((v[1:,:-1].reshape(-1,2),v[:-1,:-1].reshape(-1,2)),1)
        self.owner=torch.cat((ro,ao));self.neighbor=torch.cat((rn,an));self.face_vertices=torch.cat((radial_end,angular_end));self.face_centers=self.face_vertices.mean(1)
        edge=self.face_vertices[:,1]-self.face_vertices[:,0];self.face_area_vectors=torch.stack((edge[:,1],-edge[:,0]),1);self.face_lengths=torch.linalg.vector_norm(edge,dim=1);self.face_normals=self.face_area_vectors/self.face_lengths[:,None]
        self.interior=self.neighbor>=0;self.boundary=~self.interior
        self.masks={name:torch.zeros_like(self.boundary) for name in ('inlet','outlet','far-field','wall','cylinder')};self.masks['wall']=self.boundary.clone()
        self.boundary_labels=tuple('interior' if v else 'wall' for v in self.interior.tolist())
        self.inner_wall=torch.zeros_like(self.boundary);self.inner_wall[:n]=True
        self.outer_wall=torch.zeros_like(self.boundary);self.outer_wall[nr*n:(nr+1)*n]=True
        validate_mesh2d(self,required_masks=('inlet','outlet','far-field','wall'))


def _factory(c):return AnnularMesh(c)


# Registration is local to importing this backend; built-in implementations are unchanged.
register_backend(SolverBackend(name='annular',solver_factory=BodyFittedSolver,mesh_factory=_factory,structured=True,min_nx=8,min_ny=2,forbids_cylinder=True,reference_length='height'))


def analytic_velocity(r,c):
    ri,ro=c.inner_radius_m,c.outer_radius_m;K=(ro*ro-ri*ri)/math.log(ro/ri)
    return c.pressure_gradient_pa_m/(4*c.viscosity_pa_s)*(ro*ro-r*r-K*np.log(ro/r))


def analytic_flow(c):
    ri,ro=c.inner_radius_m,c.outer_radius_m
    return math.pi*c.pressure_gradient_pa_m/(8*c.viscosity_pa_s)*(ro**4-ri**4-(ro*ro-ri*ri)**2/math.log(ro/ri))


def solve(c):
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.linalg import cg
    start=time.perf_counter();cfg=SolverConfig(mesh_type='annular',nx=c.ntheta,ny=c.nr,length=2*c.outer_radius_m,height=2*c.outer_radius_m,cylinder_radius=None,inlet_velocity=.01,density=c.density_kg_m3,reynolds=c.density_kg_m3*.01*2*c.outer_radius_m/c.viscosity_pa_s,max_iterations=1)
    cfg.annular_config=c
    s=BodyFittedSolver(cfg);s.velocity.zero_();s.boundary_velocity.zero_();s.mass_flux.zero_();s.p.zero_();history=[];total_iterations=0
    source=c.pressure_gradient_pa_m*s.volume.numpy();old=np.zeros(s.count);linear_start=time.perf_counter()
    for outer in range(1,6):
        diagonal,ao,an,rhs=s._momentum(s.velocity,s.p,s.mass_flux)
        oi=s.oi.numpy();ni=s.ni.numpy();dd=diagonal.numpy();indices=np.arange(s.count)
        A=coo_matrix((np.r_[dd,-ao.numpy(),-an.numpy()],(np.r_[indices,oi,ni],np.r_[indices,ni,oi])),shape=(s.count,s.count)).tocsr();b=source+rhs[:,0].numpy()
        count=[0]
        def callback(x):count[0]+=1
        w,info=cg(A,b,x0=old,M=diags(1/dd),rtol=c.linear_tolerance,atol=0.,maxiter=c.max_linear_iterations,callback=callback)
        total_iterations+=count[0];res=float(np.linalg.norm(A@w-b)/np.linalg.norm(b))
        if info!=0 or not np.isfinite(w).all() or res>2*c.linear_tolerance:raise RuntimeError(f'annular full sparse CG failed: info={info}, true relative residual={res}')
        change=float(np.max(abs(w-old)));s.velocity[:,0]=torch.from_numpy(w);old=w.copy()
        history.append(dict(iteration=outer,linear_iterations=count[0],relative_linear_residual=res,velocity_change_m_s=change))
        # On this orthogonal polar topology the deferred nonorthogonal correction vanishes.
        # Verify the actual final production momentum operator, not just the first system.
        df,aof,anf,sf=s._momentum(s.velocity,s.p,s.mass_flux)
        residual=s._matvec(s.velocity[:,0],df,aof,anf)-sf[:,0]-torch.from_numpy(source)
        final=float(torch.linalg.vector_norm(residual)/torch.linalg.vector_norm(torch.from_numpy(source)))
        if final<2*c.linear_tolerance:break
    else:raise RuntimeError('annular production diffusion did not reach its true residual gate')
    faces_mu_k=(c.viscosity_pa_s*s.k).numpy();o=s.o.numpy();ne=s.n.numpy();inter=s.f.numpy();flux=faces_mu_k*w[o];flux[inter]=faces_mu_k[inter]*(w[o[inter]]-w[ne[inter]])
    return dict(config=asdict(c),elapsed_s=time.perf_counter()-start,linear_solve_elapsed_s=time.perf_counter()-linear_start,linear_iterations=total_iterations,history=history,
      fields=dict(vertices_m=s.mesh.vertices.numpy(),cell_centers_m=s.mesh.centers.numpy(),cell_volumes_m2=s.volume.numpy(),axial_velocity_m_s=w.reshape(s.field_shape),face_owner=o,face_neighbor=ne,interior_face_mask=inter,wall_face_mask=s.mesh.boundary.numpy(),inner_wall_face_mask=s.mesh.inner_wall.numpy(),outer_wall_face_mask=s.mesh.outer_wall.numpy(),face_area_vectors_m=s.S.numpy(),face_centers_m=s.mesh.face_centers.numpy(),face_lengths_m=s.mesh.face_lengths.numpy(),diffusion_coefficient_pa_s=c.viscosity_pa_s*s.k.numpy(),axial_momentum_flux_n_m=flux,source_force_n_m=source,reference_axial_velocity_m_s=analytic_velocity(np.linalg.norm(s.mesh.centers.numpy(),axis=-1),c)),
      diagnostics=dict(true_production_relative_residual=final,maximum_nonorthogonal_area_vector_m=float(s.T.abs().max()),full_matrix_unknowns=s.count,full_matrix_nnz=A.nnz,initial_velocity='zero; no analytic initialization',scope='fully developed 3D axial annular flow reduced exactly to scalar 2D cross-section; full production diffusion matrix, not general 3D Navier-Stokes',linear_backend='SciPy CSR conjugate gradients with Jacobi; CPU float64; full angular/radial unknown vector'))
