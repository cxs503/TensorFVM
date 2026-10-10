"""Optional coupled coarse residual correction for production fine-grid SIMPLE.

The coarse Picard Jacobian retains scalar momentum nonorthogonality, pressure
LS gradients and the complete steady Rhie--Chow continuity response. It is a
preconditioner, not an analytic solution or a replacement acceptance gate.
"""
from dataclasses import replace
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse


def array(x):return x.detach().cpu().numpy()


def operators(s):
    H,I,Gp=s._geometry();s._gradient(s.velocity);go,gn=s._gradient_weights['non-outlet-dirichlet'];go,gn=array(go),array(gn);o,ni,f=array(s.o),array(s.ni),array(s.f);oi=o[f];out=array(s.mesh.masks['outlet']);active=f|((~f)&(~out));Gv=[]
    for a in range(2):
        Gv.append(sparse.coo_matrix((np.r_[-go[active,a],go[f,a],-gn[:,a],gn[:,a]],(np.r_[o[active],oi,ni,ni],np.r_[o[active],ni,oi,ni])),shape=(s.count,s.count)).tocsr())
    nf=len(o);idx=np.arange(nf);B=sparse.coo_matrix((np.r_[-np.ones(np.sum(f)),np.ones(np.sum(f)),-np.ones(np.sum(out))],(np.r_[idx[f],idx[f],idx[out]],np.r_[oi,ni,o[out]])),shape=(nf,s.count)).tocsr()
    return H,I,Gp,Gv,B


def jacobian(s):
    H,I,Gp,Gv,B=operators(s);diagonal,ao,an,_=s._momentum(s.velocity,s.p,s.mass_flux);D=s.volume/diagonal;d=array(D);df=I@d;S,T,k=array(s.S),array(s.T),array(s.k);out=array(s.mesh.masks['outlet']);fixed=array(s.mesh.boundary)&(~out);mask=(~fixed).astype(float);visc=array(s._interpolate(s._dynamic_viscosity));visc[array(s._no_slip_faces())]=s.config.viscosity;visc[out]=0
    A=s._matrix(diagonal,ao,an)-H@sum((sparse.diags(visc*T[:,a])@I@Gv[a] for a in range(2)),start=sparse.csr_matrix((len(df),s.count)))
    C=[s.config.density*H@sparse.diags(mask*S[:,a])@I for a in range(2)]
    flux_p=sum((sparse.diags(S[:,a])@I@sparse.diags(d)@Gp[a]-sparse.diags(df*T[:,a])@I@Gp[a] for a in range(2)),start=sparse.csr_matrix((len(df),s.count)))-sparse.diags(df*k)@B
    P=s.config.density*H@sparse.diags(mask)@flux_p;V=sparse.diags(array(s.volume));Z=sparse.csr_matrix(A.shape)
    J=sparse.bmat([[A,Z,V@Gp[0]],[Z,A,V@Gp[1]],[C[0],C[1],P]],format='csr')
    # Independent production response: momentum at frozen mass flux and
    # continuity at frozen D. Both responses are linear in the tested increment.
    q=torch.sin(torch.arange(3*s.count,device=s.p.device,dtype=s.p.dtype)*.371);du=torch.stack((q[:s.count],q[s.count:2*s.count]),1);dp=q[2*s.count:]
    def residual(u,p):
        dd,aa,bb,source=s._momentum(u,p,s.mass_flux);mom=torch.stack([s._matvec(u[:,a],dd,aa,bb)-source[:,a] for a in range(2)],1);flux,_=s._rhie_chow(u,p,D);return torch.cat((mom[:,0],mom[:,1],s._sum(flux)))
    response=array(residual(s.velocity+du,s.p+dp)-residual(s.velocity,s.p));defect=np.linalg.norm(J@array(q)-response)/np.linalg.norm(response)
    if defect>2e-11:raise RuntimeError(f'coupled coarse Jacobian mismatch {defect}')
    return J,defect


def correct(s):
    from .sparse_simple import SparseBodyFittedSolver
    cx,cy=s.coupled_coarse_shape;ny,nx=s.field_shape;device=s.p.device;dtype=s.p.dtype
    if nx%cx or ny%cy:raise ValueError('coarse shape must divide fine shape')
    cache=getattr(s,'_coupled_coarse_cache',None)
    if cache is None:
        coarse=SparseBodyFittedSolver(replace(s.config,nx=cx,ny=cy));fine_volume=s.volume.reshape(ny,nx);groups=(torch.arange(ny,device=device)[:,None]//(ny//cy)*cx+torch.arange(nx,device=device)[None,:]//(nx//cx)).reshape(-1);den=torch.zeros(cx*cy,device=device,dtype=dtype);den.index_add_(0,groups,s.volume)
        x=(torch.arange(nx,device=device,dtype=dtype)+.5)/nx;y=(torch.arange(ny,device=device,dtype=dtype)+.5)/ny;yy,xx=torch.meshgrid(y,x,indexing='ij');grid=torch.stack((2*(xx+.5/cx)/(1+1/cx)-1,2*(yy+.5/cy)/(1+1/cy)-1),-1)[None]
        cache=dict(coarse=coarse,groups=groups,den=den,grid=grid,calls=0);s._coupled_coarse_cache=cache
    coarse,groups,den,grid=cache['coarse'],cache['groups'],cache['den'],cache['grid']
    def restrict(q):
        scalar=q.ndim==1
        if scalar:q=q[:,None]
        result=torch.zeros((cx*cy,q.shape[1]),device=device,dtype=dtype);result.index_add_(0,groups,q*s.volume[:,None]);result/=den[:,None];return result[:,0] if scalar else result
    if cache['calls']%20==0:
        coarse.velocity.copy_(restrict(s.velocity));coarse.p.copy_(restrict(s.p));coarse.nu_tilde.copy_(restrict(s.nu_tilde));coarse.turbulent_kinematic_viscosity=coarse._sa_eddy_viscosity(coarse.nu_tilde);dd,_,_,_=coarse._momentum(coarse.velocity,coarse.p,coarse.mass_flux);coarse.mass_flux=coarse._rhie_chow(coarse.velocity,coarse.p,coarse.volume/dd)[0];J,defect=jacobian(coarse);cache['matrix']=torch.as_tensor(J.toarray(),device=device,dtype=dtype);cache['lu'],cache['piv']=torch.linalg.lu_factor(cache['matrix']);cache['operator_defect']=defect
    dd,ao,an,source=s._momentum(s.velocity,s.p,s.mass_flux);mom=torch.stack([s._matvec(s.velocity[:,a],dd,ao,an)-source[:,a] for a in range(2)],1);density=restrict(mom/s.volume[:,None])*coarse.volume[:,None];mass=restrict(s._sum(s.mass_flux)/s.volume)*coarse.volume;rhs=-torch.cat((density[:,0],density[:,1],mass));q=torch.linalg.lu_solve(cache['lu'],cache['piv'],rhs[:,None])[:,0];linear_residual=float(torch.linalg.vector_norm(cache['matrix']@q-rhs)/torch.linalg.vector_norm(rhs).clamp_min(1e-30))
    if not torch.isfinite(q).all() or linear_residual>1e-9:raise RuntimeError(f'coarse coupled true residual failed {linear_residual}')
    cache['last_q']=q;cache['last_rhs']=rhs
    def prolong(q,pressure=False):
        z=q.reshape(1,-1,cy,cx)
        top=2*z[:,:,:1]-z[:,:,1:2] if pressure else -z[:,:,:1];bottom=2*z[:,:,-1:]-z[:,:,-2:-1] if pressure else -z[:,:,-1:];z=torch.cat((top,z,bottom),2)
        left=2*z[:,:,:,:1]-z[:,:,:,1:2] if pressure else -z[:,:,:,:1];right=-z[:,:,:,-1:] if pressure else z[:,:,:,-1:];z=torch.cat((left,z,right),3)
        return F.grid_sample(z,grid,mode='bilinear',align_corners=True)[0].reshape(-1,ny*nx).T
    du=prolong(torch.stack((q[:cx*cy],q[cx*cy:2*cx*cy]),0));dp=prolong(q[2*cx*cy:][None],True)[:,0];s.velocity+=.7*du;s.p+=.7*dp
    diagonal,_,_,_=s._momentum(s.velocity,s.p,s.mass_flux);D=s.volume/diagonal;flux,coef=s._rhie_chow(s.velocity,s.p,D);s._correct(flux,coef,D)
    cache['calls']+=1
    s.coarse_history=getattr(s,'coarse_history',[]);s.coarse_history.append(dict(cells=cx*cy,actual_device=str(device),operator_relative_defect=cache['operator_defect'],linear_true_residual=linear_residual,scope='coupled coarse Picard residual correction, followed by production fine continuity correction'))
