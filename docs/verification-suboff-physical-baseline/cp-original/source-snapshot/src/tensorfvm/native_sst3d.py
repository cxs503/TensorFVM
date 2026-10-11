"""NASA TMR SST-2003 local closure and opt-in conservative 3D transport.

Exact production variant (not SST-2003m): isotropic Reynolds stress must be
retained in momentum, or absorbed into a documented modified pressure.
Reference: https://tmbwg.github.io/turbmodels/sst.html#sst-2003.
This module does not establish an integrated RANS solver or physical accuracy.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from .native_viscous3d import NativeViscous3D

BETA_STAR=.09
BETA1=.075
BETA2=.0828
SIGMA_K1=.85
SIGMA_K2=1.
SIGMA_W1=.5
SIGMA_W2=.856
GAMMA1=5/9
GAMMA2=.44
A1=.31
CD_FLOOR=1e-10
VARIANT='NASA TMR SST-2003 (exact production, isotropic Reynolds stress retained)'


def _finite(name,value):
    value=np.asarray(value,dtype=float)
    if not np.isfinite(value).all():raise ValueError(f'{name} must be finite')
    return value


def sst2003_closure(k,omega,wall_distance,nu,velocity_gradient,grad_k,grad_omega,*,rho=1.):
    """Pointwise 3D SST-2003 coefficients and specific (per mass) sources.

    grad_u[...,i,j]=du_i/dx_j. Wall distance is positive nearest-wall distance;
    k>=0, omega>0, nu>0 and rho>0 are required without silent state clipping.
    CD has the density convention and dimensional numerical floor of TMR.
    Outputs production/source are per unit mass; tau_t is dynamic stress.
    """
    k,omega,d,nu,rho=np.broadcast_arrays(*[_finite(name,x) for name,x in [('k',k),('omega',omega),('wall distance',wall_distance),('nu',nu),('rho',rho)]])
    if np.any(k<0) or np.any(omega<=0) or np.any(d<=0) or np.any(nu<=0) or np.any(rho<=0):raise ValueError('k>=0 and positive omega, wall distance, nu, rho required')
    shape=k.shape;G=np.broadcast_to(_finite('velocity gradient',velocity_gradient),shape+(3,3));gk=np.broadcast_to(_finite('k gradient',grad_k),shape+(3,));gw=np.broadcast_to(_finite('omega gradient',grad_omega),shape+(3,))
    S=.5*(G+G.swapaxes(-1,-2));W=.5*(G-G.swapaxes(-1,-2))
    S2=2*np.sum(S*S,axis=(-2,-1));W2=2*np.sum(W*W,axis=(-2,-1));div=np.trace(G,axis1=-2,axis2=-1)
    cross=np.sum(gk*gw,axis=-1)
    CD=np.maximum(2*rho*SIGMA_W2*cross/omega,CD_FLOOR)
    wall_term=500*nu/(d*d*omega);turb_term=np.sqrt(k)/(BETA_STAR*omega*d)
    arg1=np.minimum(np.maximum(turb_term,wall_term),4*rho*SIGMA_W2*k/(CD*d*d))
    arg2=np.maximum(2*turb_term,wall_term)
    # tanh saturates far below these caps; cap avoids powers overflowing.
    F1=np.tanh(np.minimum(arg1,10.)**4);F2=np.tanh(np.minimum(arg2,20.)**2)
    blend=lambda inner,outer:F1*inner+(1-F1)*outer
    beta=blend(BETA1,BETA2);gamma=blend(GAMMA1,GAMMA2)
    sigma_k=blend(SIGMA_K1,SIGMA_K2);sigma_w=blend(SIGMA_W1,SIGMA_W2)
    denominator=np.maximum(A1*omega,np.sqrt(S2)*F2)
    nut=A1*k/denominator
    # Algebraic P/nut remains finite at k=0 and retains exact div(u) terms.
    production_ratio=S2-2/3*div*div-2/3*(denominator/A1)*div
    raw=nut*production_ratio
    limit=10*BETA_STAR*omega*k
    limited=np.minimum(raw,limit)
    limited_ratio=np.minimum(production_ratio,10*BETA_STAR*omega*denominator/A1)
    cross_source=2*(1-F1)*SIGMA_W2*cross/omega
    k_sink=BETA_STAR*omega*k;omega_sink=beta*omega*omega
    omega_production=gamma*limited_ratio
    stress=rho[...,None,None]*(nut[...,None,None]*(2*S-2/3*div[...,None,None]*np.eye(3))-2/3*k[...,None,None]*np.eye(3))
    return dict(F1=F1,F2=F2,CD_komega=CD,strain_squared=S2,vorticity_squared=W2,velocity_divergence=div,nu_t=nut,mu_t=rho*nut,beta=beta,gamma=gamma,sigma_k=sigma_k,sigma_omega=sigma_w,production_raw=raw,production_limited=limited,production_limit=limit,k_destruction=k_sink,omega_production=omega_production,omega_destruction=omega_sink,cross_diffusion_source=cross_source,k_source=limited-k_sink,omega_source=omega_production-omega_sink+cross_source,turbulent_stress=stress)


def wall_omega(nu,first_cell_distance):
    """Original SST low-Re wall condition: 10*6nu/(beta1*d1^2), not a wall function."""
    nu,d=np.broadcast_arrays(_finite('nu',nu),_finite('first cell distance',first_cell_distance))
    if np.any(nu<=0) or np.any(d<=0):raise ValueError('positive nu and first-cell distance required')
    return 60*nu/(BETA1*d*d)


def inflow_from_intensity(speed,intensity,length_scale):
    """Isotropic inlet parameterization, not an experimental turbulence measurement.

    k=3/2(UI)^2; omega=sqrt(k)/(beta_star^(1/4)*ell).
    I is a fraction, ell has mesh length units. Callers must document provenance.
    """
    u,I,ell=np.broadcast_arrays(_finite('speed',speed),_finite('intensity',intensity),_finite('length scale',length_scale))
    if np.any(u<=0) or np.any(I<=0) or np.any(ell<=0):raise ValueError('positive speed, intensity, length scale required')
    k=1.5*(u*I)**2
    return k,np.sqrt(k)/(BETA_STAR**.25*ell)


class NativeSSTTransport3D:
    """One frozen-coefficient implicit pseudo-time update of k and omega.

    Face flux is volume flux (u dot S); density must be constant. Matrix/source
    use specific conservation form, so shared advective/diffusive fluxes cancel
    exactly and volume-integrated sources are reported in the ledger.
    No positivity clipping: an invalid update is rejected. Full nonorthogonal
    diffusion is not necessarily an M matrix. A step is not convergence.
    """
    def __init__(self,mesh,scalar_dirichlet_mask):
        self.mesh=mesh;self.mask=np.asarray(scalar_dirichlet_mask,dtype=bool)
        self.base=NativeViscous3D(mesh,self.mask,1.)

    def _upwind(self,flux,boundary_values):
        m=self.mesh;op=self.base;q=_finite('face flux',flux);bd=np.broadcast_to(_finite('boundary values',boundary_values),(op.F,))
        if q.shape!=(op.F,):raise ValueError('one volume flux per native face required')
        f=np.arange(op.F);internal=op.internal;up=np.where(q>=0,m.owner,m.neighbor.clip(min=0))
        # Physical inflow always requires explicit scalar data, even if diffusion
        # uses Neumann. Outflow carries the owner value.
        active=internal|(q>=0);matrix=coo_matrix((q[active],(f[active],up[active])),shape=(op.F,op.N)).tocsr()
        source=np.zeros(op.F);incoming=op.boundary&(q<0);source[incoming]=q[incoming]*bd[incoming]
        return matrix,source

    def step(self,k,omega,face_flux,wall_distance,nu,velocity_gradient,dt,*,k_boundary,omega_boundary,k_neumann=0.,omega_neumann=0.,rho=1.):
        if np.ndim(rho)!=0 or not np.isfinite(rho) or rho<=0:raise ValueError('transport requires positive constant density')
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive pseudo time step required')
        m=self.mesh;b=self.base;N=b.N;k=_finite('k',k);omega=_finite('omega',omega)
        if k.shape!=(N,) or omega.shape!=(N,):raise ValueError('one k and omega per cell required')
        gk=np.column_stack([g@k for g in b.gradient_matrices])+b.gradient_source(k_neumann,k_boundary)
        gw=np.column_stack([g@omega for g in b.gradient_matrices])+b.gradient_source(omega_neumann,omega_boundary)
        closure=sst2003_closure(k,omega,wall_distance,nu,velocity_gradient,gk,gw,rho=rho)
        result={};ledgers={};matrices={};volume=m.cell_volumes
        for name,old,sigma,production,destruction,bd,gn in [
            ('k',k,closure['sigma_k'],closure['production_limited'],BETA_STAR*omega,k_boundary,k_neumann),
            ('omega',omega,closure['sigma_omega'],closure['omega_production']+closure['cross_diffusion_source'],closure['beta']*omega,omega_boundary,omega_neumann)]:
            coefficient=np.broadcast_to(np.asarray(nu,float),(N,))+sigma*closure['nu_t']
            diffusion=NativeViscous3D(m,self.mask,coefficient)
            adv,adv_source=self._upwind(face_flux,bd)
            diff_source=diffusion.scalar_flux_source(gn,bd)
            # Patankar split preserves a negative source without silently dropping
            # it. It is linearized as sink coefficient at the old positive state.
            negative=np.minimum(production,0)
            if np.any((old<=0)&(negative<0)):raise ValueError('negative source cannot be linearized at zero state')
            extra_sink=np.divide(-negative,old,out=np.zeros_like(old),where=old>0)
            positive=np.maximum(production,0)
            A=diags(volume/dt+volume*(destruction+extra_sink))+b.divergence_matrix@adv-diffusion.scalar_diffusion_matrix
            rhs=volume*old/dt+volume*positive+b.divergence_matrix@(diff_source-adv_source)
            new=spsolve(A.tocsc(),rhs)
            if not np.isfinite(new).all() or np.any(new<0) or (name=='omega' and np.any(new<=0)):raise RuntimeError('invalid SST update; no clipping applied')
            adv_flux=adv@new+adv_source;diff_flux=diffusion.scalar_flux(new,gn,bd)
            local_source=positive-(destruction+extra_sink)*new
            transient=volume*(new-old)/dt
            residual=transient+b.divergence_matrix@(adv_flux-diff_flux)-volume*local_source
            ledgers[name]=dict(max_local_integrated_residual=float(np.max(np.abs(residual))),global_residual=float(transient.sum()+(adv_flux-diff_flux)[b.boundary].sum()-(volume*local_source).sum()),boundary_advective_flux=float(adv_flux[b.boundary].sum()),boundary_diffusive_flux=float(diff_flux[b.boundary].sum()),integrated_source=float((volume*local_source).sum()),relative_update=float(np.linalg.norm(new-old)/max(np.linalg.norm(old),np.finfo(float).tiny)))
            result[name]=new;matrices[name]=A.tocsr()
        return dict(**result,closure=closure,ledger=ledgers,matrices=matrices,variant=VARIANT,integrated_rans_validated=False,physical_benchmark_passed=False)
