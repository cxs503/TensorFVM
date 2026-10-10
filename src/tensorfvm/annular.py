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
        theta=torch.linspace(0,2*math.pi,n+1,dtype=torch.float64,device=c.device);radius=torch.linspace(ac.inner_radius_m,ac.outer_radius_m,nr+1,dtype=torch.float64,device=c.device)
        rr,tt=torch.meshgrid(radius,theta,indexing='ij');self.vertices=torch.stack((rr*torch.cos(tt),rr*torch.sin(tt)),-1)
        v=self.vertices;poly=torch.stack((v[:-1,:-1],v[1:,:-1],v[1:,1:],v[:-1,1:]),-2);nxt=poly.roll(-1,-2)
        cross=poly[...,0]*nxt[...,1]-nxt[...,0]*poly[...,1];self.volumes=.5*cross.sum(-1);self.centers=((poly+nxt)*cross[...,None]).sum(-2)/(6*self.volumes[...,None])
        j,i=torch.meshgrid(torch.arange(nr+1,device=c.device),torch.arange(n,device=c.device),indexing='ij');ro=((j-1).clamp_min(0)*n+i).flatten();rn=torch.where((j>0)&(j<nr),j*n+i,-1).flatten()
        ra=v[:,:-1].reshape(-1,2);rb=v[:,1:].reshape(-1,2);inner=(j==0).flatten();first=torch.where(inner[:,None],rb,ra);second=torch.where(inner[:,None],ra,rb)
        radial_end=torch.stack((first,second),1)
        j,i=torch.meshgrid(torch.arange(nr,device=c.device),torch.arange(n,device=c.device),indexing='ij');ao=(j*n+(i-1)%n).flatten();an=(j*n+i).flatten();angular_end=torch.stack((v[1:,:-1].reshape(-1,2),v[:-1,:-1].reshape(-1,2)),1)
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


def _array(x):return x.detach().cpu().numpy()


def _torch_cg(A,b,diagonal,x0,tolerance,iterations):
    """Identical full CSR/Jacobi CG on CPU and CUDA; report true residual."""
    x=x0.clone();count=0;target=tolerance*torch.linalg.vector_norm(b)
    mv=lambda v:torch.sparse.mm(A,v[:,None])[:,0]
    # Restart using the true residual if a recursively updated residual drifts.
    for restart in range(4):
        r=b-mv(x)
        if float(torch.linalg.vector_norm(r))<=float(target):break
        z=r/diagonal;p=z.clone();rz=torch.dot(r,z)
        for _ in range(iterations-count):
            ap=mv(p);alpha=rz/torch.dot(p,ap);x=x+alpha*p;r=r-alpha*ap;count+=1
            if float(torch.linalg.vector_norm(r))<=float(target):break
            z=r/diagonal;new=torch.dot(r,z);p=z+(new/rz)*p;rz=new
    true=float(torch.linalg.vector_norm(b-mv(x))/torch.linalg.vector_norm(b))
    if not torch.isfinite(x).all() or true>2*tolerance:raise RuntimeError(f'torch full CSR CG failed: {true}')
    return x,count,true


def solve(c,device='cpu',linear_backend='scipy'):
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.linalg import cg
    if device!='cpu' and linear_backend!='torch-cg':raise ValueError('CUDA requires torch-cg; no silent CPU linear fallback')
    if device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();cfg=SolverConfig(mesh_type='annular',nx=c.ntheta,ny=c.nr,length=2*c.outer_radius_m,height=2*c.outer_radius_m,cylinder_radius=None,inlet_velocity=.01,density=c.density_kg_m3,reynolds=c.density_kg_m3*.01*2*c.outer_radius_m/c.viscosity_pa_s,max_iterations=1,device=device)
    cfg.annular_config=c;s=BodyFittedSolver(cfg);s.velocity.zero_();s.boundary_velocity.zero_();s.mass_flux.zero_();s.p.zero_();history=[];total_iterations=0
    source=c.pressure_gradient_pa_m*s.volume;old=torch.zeros_like(s.p);linear_start=time.perf_counter();matrix_nnz=0
    for outer in range(1,6):
        diagonal,ao,an,rhs=s._momentum(s.velocity,s.p,s.mass_flux);b=source+rhs[:,0]
        if linear_backend=='scipy':
            oi,ni,dd=_array(s.oi),_array(s.ni),_array(diagonal);indices=np.arange(s.count)
            A=coo_matrix((np.r_[dd,-_array(ao),-_array(an)],(np.r_[indices,oi,ni],np.r_[indices,ni,oi])),shape=(s.count,s.count)).tocsr();matrix_nnz=A.nnz;ba=_array(b);count=[0]
            def callback(x):count[0]+=1
            wa,info=cg(A,ba,x0=_array(old),M=diags(1/dd),rtol=c.linear_tolerance,atol=0.,maxiter=c.max_linear_iterations,callback=callback)
            res=float(np.linalg.norm(A@wa-ba)/np.linalg.norm(ba));niter=count[0];w=torch.from_numpy(wa)
            if info!=0 or not np.isfinite(wa).all() or res>2*c.linear_tolerance:raise RuntimeError(f'annular full sparse CG failed: info={info}, true residual={res}')
        elif linear_backend=='torch-cg':
            indices=torch.arange(s.count,device=s.p.device);ij=torch.stack((torch.cat((indices,s.oi,s.ni)),torch.cat((indices,s.ni,s.oi))));values=torch.cat((diagonal,-ao,-an))
            A=torch.sparse_coo_tensor(ij,values,(s.count,s.count),device=s.p.device,check_invariants=True).coalesce().to_sparse_csr();matrix_nnz=A.values().numel()
            w,niter,res=_torch_cg(A,b,diagonal,old,c.linear_tolerance,c.max_linear_iterations)
        else:raise ValueError('linear backend must be scipy or torch-cg')
        total_iterations+=niter;change=float((w-old).abs().max());s.velocity[:,0]=w;old=w.clone();history.append(dict(iteration=outer,linear_iterations=niter,relative_linear_residual=res,velocity_change_m_s=change))
        df,aof,anf,sf=s._momentum(s.velocity,s.p,s.mass_flux);residual=s._matvec(s.velocity[:,0],df,aof,anf)-sf[:,0]-source;final=float(torch.linalg.vector_norm(residual)/torch.linalg.vector_norm(source))
        if final<2*c.linear_tolerance:break
    else:raise RuntimeError('annular production diffusion did not reach its true residual gate')
    gradient=s._gradient(s.velocity)[:,:,0];face_gradient=s._interpolate(gradient);correction=c.viscosity_pa_s*(s.T*face_gradient).sum(-1)
    coefficients=c.viscosity_pa_s*s.k;flux=coefficients*w[s.o];flux[s.f]=coefficients[s.f]*(w[s.oi]-w[s.ni]);flux-=correction
    if device.startswith('cuda'):torch.cuda.synchronize()
    compute_elapsed=time.perf_counter()-start;linear_elapsed=time.perf_counter()-linear_start;peak=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None
    fields=dict(vertices_m=_array(s.mesh.vertices),cell_centers_m=_array(s.mesh.centers),cell_volumes_m2=_array(s.volume),axial_velocity_m_s=_array(w).reshape(s.field_shape),face_owner=_array(s.o),face_neighbor=_array(s.n),interior_face_mask=_array(s.f),wall_face_mask=_array(s.mesh.boundary),inner_wall_face_mask=_array(s.mesh.inner_wall),outer_wall_face_mask=_array(s.mesh.outer_wall),face_area_vectors_m=_array(s.S),face_centers_m=_array(s.mesh.face_centers),face_lengths_m=_array(s.mesh.face_lengths),diffusion_coefficient_pa_s=_array(coefficients),nonorthogonal_area_vectors_m=_array(s.T),cell_axial_velocity_gradient_s_inv=_array(gradient),nonorthogonal_momentum_flux_n_m=_array(correction),axial_momentum_flux_n_m=_array(flux),source_force_n_m=_array(source),reference_axial_velocity_m_s=analytic_velocity(np.linalg.norm(_array(s.mesh.centers),axis=-1),c))
    config=asdict(c);config.update(device=device,linear_backend=linear_backend)
    return dict(config=config,elapsed_s=time.perf_counter()-start,compute_elapsed_s=compute_elapsed,linear_solve_elapsed_s=linear_elapsed,linear_iterations=total_iterations,peak_cuda_allocated_bytes=peak,history=history,fields=fields,
      diagnostics=dict(true_production_relative_residual=final,maximum_nonorthogonal_area_vector_m=float(s.T.abs().max()),full_matrix_unknowns=s.count,full_matrix_nnz=matrix_nnz,initial_velocity='zero; no analytic initialization',scope='fully developed 3D axial annular reduction; full production diffusion matrix, not general 3D Navier-Stokes',linear_backend=linear_backend,device=str(s.p.device),timing_scope='synchronized mesh/operator/full solve before raw host export; elapsed_s additionally includes export; no plots',silent_cpu_solver_fallback=False))
