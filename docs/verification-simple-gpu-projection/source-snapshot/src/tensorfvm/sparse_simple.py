"""Opt-in full SIMPLE with assembled sparse Krylov systems and line preconditioning.

Mesh and finite-volume operators remain BodyFittedSolver's production operators.
SciPy constructs CSR on the host; all iterative solves run on the requested Torch
CPU/CUDA device. Host assembly and transfer are included in reported timings.
"""
import time
import math
import numpy as np
import torch
from scipy import sparse
from .body_fitted import BodyFittedSolver


def host(x):return x.detach().cpu().numpy()


class SparseBodyFittedSolver(BodyFittedSolver):
    def __init__(self,config):
        super().__init__(config);self.linear_history=[];self._pressure_csr=None;self._geometry_operators=None

    def _momentum(self,velocity,pressure,flux):
        diagonal,ao,an,source=super()._momentum(velocity,pressure,flux)
        force=getattr(self,"momentum_body_force",None)
        if force is not None:source=source+self.volume[:,None]*force
        return diagonal,ao,an,source

    def _matrix(self,diagonal,ao,an):
        n=self.count;oi,ni=host(self.oi),host(self.ni);indices=np.arange(n)
        return sparse.coo_matrix((np.r_[host(diagonal),-host(ao),-host(an)],(np.r_[indices,oi,ni],np.r_[indices,ni,oi])),shape=(n,n)).tocsr()

    def _geometry(self):
        if self._geometry_operators is not None:return self._geometry_operators
        self._gradient(self.p,pressure=True);go,gn=self._gradient_weights['pressure'];go,gn=host(go),host(gn)
        o,ni,f=host(self.o),host(self.ni),host(self.f);oi=o[f];n=self.count;nf=len(o);idx=np.arange(nf);out=host(self.mesh.masks['outlet']);active=f|out
        H=sparse.coo_matrix((np.r_[np.ones(nf),-np.ones(len(ni))],(np.r_[o,ni],np.r_[idx,idx[f]])),shape=(n,nf)).tocsr()
        I=sparse.coo_matrix((np.r_[np.where(f,.5,1.),np.full(len(ni),.5)],(np.r_[idx,idx[f]],np.r_[o,ni])),shape=(nf,n)).tocsr()
        gradients=[]
        for a in range(2):
            rows=np.r_[o[active],oi,ni,ni];cols=np.r_[o[active],ni,oi,ni];vals=np.r_[-go[active,a],go[f,a],-gn[:,a],gn[:,a]]
            gradients.append(sparse.coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr())
        self._geometry_operators=H,I,gradients;return self._geometry_operators

    def pressure_matrix(self,coefficient,D,steady_D=None):
        H,I,G=self._geometry();co=host(coefficient);diag=np.asarray(H.multiply(co).sum(axis=1)).ravel()
        # Owner and neighbour both contribute positively to the diagonal.
        diag=np.bincount(host(self.o),weights=co,minlength=self.count)+np.bincount(host(self.ni),weights=co[host(self.f)],minlength=self.count)
        A=self._matrix(torch.as_tensor(diag,device=self.p.device),coefficient[self.f],coefficient[self.f]);T,S=host(self.T),host(self.S);k=host(self.k)
        F=sum((sparse.diags(-co/k*T[:,a])@I@G[a] for a in range(2)),start=sparse.csr_matrix((len(co),self.count)))
        if steady_D is not None and self.config.pseudo_time_step is not None:
            d,sd=host(D),host(steady_D);beta=np.clip(1-(I@d)/(I@sd),0,1);fixed=co==0
            for a in range(2):
                response=(sparse.diags(1-beta)@I@sparse.diags(sd)-I@sparse.diags(d))@G[a]
                F+=sparse.diags(self.config.density*S[:,a]*(~fixed))@response
        return (A+H@F).tocsr()

    def _correct(self,predicted_flux,coefficient,D,steady_D=None):
        self._pressure_csr=self.pressure_matrix(coefficient,D,steady_D)
        try:return super()._correct(predicted_flux,coefficient,D,steady_D)
        finally:self._pressure_csr=None

    def _linear(self,diagonal,ao,an,rhs,initial,symmetric=False,operator=None,
                threshold_floor=1e-10,relative_tolerance=1e-9,allow_inexact=False):
        started=time.perf_counter();pressure=operator is not None
        A=self._pressure_csr if pressure else self._matrix(diagonal,ao,an)
        if A is None:raise RuntimeError('pressure CSR not assembled')
        A.sort_indices();device=rhs.device;opts=dict(device=device,dtype=rhs.dtype)
        M=torch.sparse_csr_tensor(torch.as_tensor(A.indptr,device=device),torch.as_tensor(A.indices,device=device),torch.as_tensor(A.data,**opts),size=A.shape,check_invariants=True)
        mv=lambda q:torch.sparse.mm(M,q[:,None])[:,0]
        # Check the entire assembled operator against the production matrix-free
        # response for a deterministic vector on every pressure construction.
        if pressure:
            probe=torch.sin(torch.arange(self.count,**opts)*.371)
            defect=float(torch.linalg.vector_norm(mv(probe)-operator(probe))/torch.linalg.vector_norm(operator(probe)))
            if defect>2e-12:raise RuntimeError(f'assembled pressure operator mismatch {defect}')
            ny,nx=self.field_shape;blocks=np.zeros((nx,ny,ny))
            rr=np.arange(ny)[:,None]*nx+np.arange(nx)[None,:]
            for i in range(nx):blocks[i]=A[rr[:,i]][:,rr[:,i]].toarray()
            lu,piv=torch.linalg.lu_factor(torch.as_tensor(blocks,**opts))
            factor=2
            while math.ceil(nx/factor)*math.ceil(ny/factor)>600:factor*=2
            cx,cy=math.ceil(nx/factor),math.ceil(ny/factor)
            group=(np.arange(ny)[:,None]//factor*cx+np.arange(nx)[None,:]//factor).ravel()
            P=sparse.coo_matrix((np.ones(self.count),(np.arange(self.count),group)),shape=(self.count,cx*cy)).tocsr()
            coarse=(P.T@A@P).toarray();clu,cpiv=torch.linalg.lu_factor(torch.as_tensor(coarse,**opts));groups=torch.as_tensor(group,device=device)
            def line(q):return torch.linalg.lu_solve(lu,piv,q.reshape(ny,nx).T[:,:,None])[:,:,0].T.reshape(-1)
            def pre(q):
                z=.7*line(q);res=q-mv(z);restricted=torch.zeros(cx*cy,**opts);restricted.index_add_(0,groups,res)
                correction=torch.linalg.lu_solve(clu,cpiv,restricted[:,None])[:,0];z+=correction[groups]
                return z+.7*line(q-mv(z))
        else:
            dd=torch.as_tensor(A.diagonal(),**opts)
            pre=lambda q:q/dd
        x=initial.clone();bn=float(torch.linalg.vector_norm(rhs));force_scale=max(self.config.density*self.config.inlet_velocity**2*self.config.residual_length,self.config.viscosity*self.config.inlet_velocity);floor=1e-12*min(1.,512/self.count) if pressure else 1e-12;target=max(floor,min(bn*1e-11,self.config.tolerance*force_scale/(10*math.sqrt(self.count))));iterations=0;restart=32
        # Restarted left-preconditioned GMRES; identical operations on CPU/CUDA.
        for cycle in range(80):
            true=rhs-mv(x)
            if float(torch.linalg.vector_norm(true))<=target:break
            r=pre(true);beta=torch.linalg.vector_norm(r);V=[r/beta];H=torch.zeros((restart+1,restart),**opts);base=x.clone();e=torch.zeros(restart+1,**opts);e[0]=beta
            for j in range(restart):
                w=pre(mv(V[j]))
                # Double modified Gram-Schmidt prevents loss of orthogonality.
                basis=torch.stack(V,1)
                for _ in range(2):
                    values=basis.T@w;H[:j+1,j]+=values;w-=basis@values
                H[j+1,j]=torch.linalg.vector_norm(w)
                y=torch.linalg.lstsq(H[:j+2,:j+1],e[:j+2],driver='gels').solution
                x=base+torch.stack(V,1)@y;iterations+=1
                if float(torch.linalg.vector_norm(rhs-mv(x)))<=target:break
                if float(H[j+1,j])<1e-25:break
                V.append(w/H[j+1,j])
            else:continue
            if float(torch.linalg.vector_norm(rhs-mv(x)))<=target:break
        residual=float(torch.linalg.vector_norm(rhs-mv(x)))
        self.linear_history.append(dict(pressure=pressure,iterations=iterations,true_residual=residual,target=target,relative_residual=residual/max(bn,1e-30),actual_device=str(device),host_csr_assembly=True,matrix_nnz=int(A.nnz),assembled_operator_relative_defect=defect if pressure else None,elapsed_s=time.perf_counter()-started))
        if not torch.isfinite(x).all() or residual>target*1.05:raise RuntimeError(f'sparse SIMPLE GMRES failed: {residual} > {target}')
        return x

    def step(self):
        if getattr(self,'coupled_coarse_shape',None):
            if self.config.time_step is not None:raise ValueError('coupled coarse correction currently supports steady solves only')
            metrics=super().step()
            if not self.converged:
                from .coupled_coarse import correct
                correct(self)
            return metrics
        depth=getattr(self,'anderson_depth',0)
        if not depth:return super().step()
        c=self.config;u_scale=c.inlet_velocity;p_scale=max(c.density*u_scale**2,12*c.viscosity*u_scale*c.length/c.height**2);flux_scale=c.density*u_scale*self.mesh.face_lengths
        def pack():
            chunks=[self.velocity.reshape(-1)/u_scale,self.p/p_scale,self.mass_flux/flux_scale]
            if c.turbulence_model=='spalart-allmaras':chunks.append(self.nu_tilde/(10*c.sa_freestream_ratio*self.kinematic_viscosity))
            return torch.cat(chunks)
        before=pack();metrics=super().step();after=pack()
        if self.converged:return metrics
        histories=getattr(self,'_anderson_history',[]);histories.append((after,after-before));histories=histories[-depth:];self._anderson_history=histories
        if len(histories)<2:return metrics
        residual=torch.stack([q[1] for q in histories],1);gram=residual.T@residual;regularization=max(float(torch.trace(gram))*1e-12,1e-30);gram+=regularization*torch.eye(len(histories),device=self.p.device,dtype=self.p.dtype)
        alpha=torch.linalg.solve(gram,torch.ones(len(histories),device=self.p.device,dtype=self.p.dtype));alpha/=alpha.sum()
        if not torch.isfinite(alpha).all() or float(alpha.abs().max())>100:return metrics
        mixed=torch.stack([q[0] for q in histories],1)@alpha;n=self.count;nf=len(self.o)
        self.velocity.copy_(mixed[:2*n].reshape(n,2)*u_scale);self.p.copy_(mixed[2*n:3*n]*p_scale);self.mass_flux.copy_(mixed[3*n:3*n+nf]*flux_scale)
        if c.turbulence_model=='spalart-allmaras':
            self.nu_tilde.copy_(mixed[3*n+nf:].clamp_min(0)*(10*c.sa_freestream_ratio*self.kinematic_viscosity));self.turbulent_kinematic_viscosity=self._sa_eddy_viscosity(self.nu_tilde)
        return metrics

    def _turbulence_step(self,physical_old=None):
        count=getattr(self,'sa_inner_iterations',1)
        if self.config.turbulence_model!='spalart-allmaras' or count==1:return super()._turbulence_step(physical_old)
        old=self.nu_tilde.clone()
        for _ in range(count):
            residual,_=super()._turbulence_step(physical_old)
            if residual<self.config.tolerance*.1:break
        change=float((self.nu_tilde-old).abs().max()/self.kinematic_viscosity)
        return residual,change
