"""Independent 1D SST-2003 half-channel diagnostic, not a production CFD solver.

Equations/constants: https://tmbwg.github.io/turbmodels/sst.html#sst-2003
No production closure, mesh, momentum, scalar or RANS module is imported.
Total shear is integrated analytically: (nu+nu_t)*dU/dy = 1-y.
The k/omega equations use independent conservative tridiagonal diffusion.
No DNS values enter the solve. Grid/near-wall convergence must be checked.
"""
import numpy as np
from scipy.linalg import solve_banded


def closure(y,k,w,dk,dw,nu):
    """Specialization to incompressible parallel shear, independently coded."""
    if np.any(k<0) or np.any(w<=0):raise ValueError('inadmissible turbulence state')
    a=np.sqrt(k)/(.09*w*y);b=500*nu/(y*y*w)
    cd=np.maximum(2*.856*dk*dw/w,1e-10)
    f1=np.tanh(np.minimum(np.minimum(np.maximum(a,b),4*.856*k/(cd*y*y)),10)**4)
    f2=np.tanh(np.minimum(np.maximum(2*a,b),20)**2)
    beta=.075*f1+.0828*(1-f1);gamma=(5/9)*f1+.44*(1-f1)
    sk=.85*f1+1*(1-f1);sw=.5*f1+.856*(1-f1)
    # Solve the eddy-viscosity limiter and total-shear relation exactly.
    shear=(1-y)/(nu+k/w)
    limited=shear*f2>.31*w
    shear[limited]=((1-y[limited])-.31*k[limited]/f2[limited])/nu
    denominator=np.maximum(.31*w,shear*f2)
    nut=.31*k/denominator
    production=np.minimum(nut*shear**2,10*.09*w*k)
    omega_production=gamma*np.minimum(shear**2,10*.09*w*denominator/.31)
    cross=2*(1-f1)*.856*dk*dw/w
    return dict(nut=nut,shear=shear,F1=f1,F2=f2,k_diffusion=nu+sk*nut,
        omega_diffusion=nu+sw*nut,k_production=production,k_sink=.09*w,
        omega_production=omega_production+cross,omega_sink=beta*w)


def conductance(faces,y,coefficient):
    g=np.zeros(len(faces));g[0]=coefficient[0]/y[0]
    g[1:-1]=1/((faces[1:-1]-y[:-1])/coefficient[:-1]+(y[1:]-faces[1:-1])/coefficient[1:])
    return g


def scalar_step(faces,y,old,coefficient,production,sink,boundary,dt):
    volume=np.diff(faces);g=conductance(faces,y,coefficient)
    negative=np.minimum(production,0.)
    destruction=sink+np.divide(-negative,old,out=np.zeros_like(old),where=old>0)
    band=np.zeros((3,len(y)));band[0,1:]=-g[1:-1];band[2,:-1]=-g[1:-1]
    band[1]=g[:-1]+g[1:]+volume*(destruction+1/dt)
    rhs=volume*(np.maximum(production,0.)+old/dt);rhs[0]+=g[0]*boundary
    candidate=solve_banded((1,1),band,rhs)
    if not np.isfinite(candidate).all() or np.any(candidate<0):raise RuntimeError('invalid scalar candidate; no clipping')
    return candidate


def scalar_residual(faces,y,value,coefficient,production,sink,boundary):
    g=conductance(faces,y,coefficient);flux=np.zeros(len(faces))
    flux[0]=g[0]*(value[0]-boundary);flux[1:-1]=g[1:-1]*np.diff(value)
    source=np.diff(faces)*(production-sink*value)
    defect=-np.diff(flux)-source
    scale=abs(flux[:-1])+abs(flux[1:])+np.diff(faces)*(abs(production)+abs(sink*value))
    return dict(relative_local_max=float(np.max(abs(defect)/np.maximum(scale,1e-30))),
        absolute_local_max=float(np.max(abs(defect))),global_balance=float(np.sum(defect)),
        flux=flux,integrated_source=source,defect=defect)


def solve_reference(n,*,re_tau=182.088,stretch=3.,max_iterations=8000,tolerance=1e-10,dt=.01,relaxation=.5,callback=None):
    if (not isinstance(n,(int,np.integer)) or n<4 or not np.isfinite([re_tau,stretch,tolerance,dt,relaxation]).all()
            or re_tau<=0 or stretch<0 or max_iterations<1 or tolerance<=0 or dt<=0 or not 0<relaxation<=1):
        raise ValueError('invalid reference controls')
    faces=np.linspace(0,1,n+1)
    if stretch:faces=np.expm1(stretch*faces)/np.expm1(stretch)
    y=(faces[:-1]+faces[1:])/2;nu=1/re_tau;wall_w=60*nu/(.075*y[0]**2)
    k=np.full(n,.1);w=6*nu/(.075*y**2)+1/(.3*.41*y)
    def gradient(value,bd):
        return np.gradient(np.r_[bd,value,value[-1]],np.r_[0.,y,2-y[-1]],edge_order=2)[1:-1]
    def evaluate(k,w):
        c=closure(y,k,w,gradient(k,0),gradient(w,wall_w),nu)
        return c,{name:scalar_residual(faces,y,val,c[name+'_diffusion'],c[name+'_production'],c[name+'_sink'],bd)
                  for name,val,bd in [('k',k,0.),('omega',w,wall_w)]}
    history=[];stable=0
    for iteration in range(max_iterations):
        c,_=evaluate(k,w)
        kn=scalar_step(faces,y,k,c['k_diffusion'],c['k_production'],c['k_sink'],0.,dt)
        wn=scalar_step(faces,y,w,c['omega_diffusion'],c['omega_production'],c['omega_sink'],wall_w,dt)
        change=max(float(np.max(abs(kn-k)/np.maximum(abs(kn),1e-30))),float(np.max(abs(wn-w)/wn)))
        k=(1-relaxation)*k+relaxation*kn;w=(1-relaxation)*w+relaxation*wn
        c,r=evaluate(k,w);error=max(v['relative_local_max'] for v in r.values())
        row=dict(iteration=iteration+1,current_relative_local_residual=error,unrelaxed_change=change)
        history.append(row);stable=stable+1 if max(error,change)<tolerance else 0
        if callback:callback(row)
        if stable>=2:break
    uf=np.r_[0,np.cumsum(np.diff(faces)*c['shear'])];u=(uf[:-1]+uf[1:])/2
    bulk=float(np.diff(faces)@u)
    return dict(converged=stable>=2,n=n,re_tau=re_tau,faces=faces,y=y,k=k,omega=w,
        velocity=u,face_velocity=uf,nu_t=c['nut'],shear=c['shear'],F1=c['F1'],F2=c['F2'],
        Cf=2/bulk**2,bulk_velocity=bulk,wall_omega=wall_w,first_cell_y_plus=y[0]*re_tau,
        history=history,scalar_residuals=r,physical_suboff_benchmark_passed=False)
