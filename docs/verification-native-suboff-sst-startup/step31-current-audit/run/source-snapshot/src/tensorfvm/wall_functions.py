"""Spalding all-y-plus equilibrium wall law and dimensional traction.

This boundary constitutive module is independently verified. It is not yet
coupled to the existing wall-resolved SA transport boundary conditions.
"""
import numpy as np

KAPPA=.41
E=9.8


def spalding_y_plus(u_plus,kappa=KAPPA,E=E):
    u=np.asarray(u_plus,dtype=float)
    if not np.isfinite(u).all() or np.any(u<0) or not np.isfinite([kappa,E]).all() or kappa<=0 or E<=0:raise ValueError('finite nonnegative u+ and positive wall constants required')
    z=kappa*u
    with np.errstate(over='ignore',invalid='ignore'):
        remainder=np.expm1(np.minimum(z,700))-z-.5*z*z-z**3/6
    small=z<1e-3
    remainder=np.where(small,z**4/24*(1+z/5+z*z/30+z**3/210),remainder)
    return u+np.maximum(remainder,0)/E


def u_plus_at_y_plus(y_plus,kappa=KAPPA,E=E):
    yp=np.asarray(y_plus,dtype=float)
    if not np.isfinite(yp).all() or np.any(yp<0):raise ValueError('nonnegative finite y+ required')
    low=np.zeros_like(yp);high=np.maximum(yp,1.)
    for _ in range(90):
        mid=.5*(low+high);small=spalding_y_plus(mid,kappa,E)<yp;low=np.where(small,mid,low);high=np.where(small,high,mid)
    return .5*(low+high)


def friction_velocity(speed_m_s,distance_m,kinematic_viscosity_m2_s,kappa=KAPPA,E=E):
    speed,y,nu=np.broadcast_arrays(np.asarray(speed_m_s,dtype=float),np.asarray(distance_m,dtype=float),np.asarray(kinematic_viscosity_m2_s,dtype=float))
    if not all(np.isfinite(a).all() for a in (speed,y,nu)) or np.any(speed<0) or np.any(y<=0) or np.any(nu<=0):raise ValueError('finite speed>=0, positive distance and viscosity required')
    lower=np.sqrt(nu*speed/y);upper=2*np.maximum(lower,speed)
    # Bracket the unique positive root without evaluating exp(infinity) at zero.
    for _ in range(90):
        mid=np.maximum(.5*(lower+upper),np.finfo(float).tiny)
        positive=y*mid/nu>=spalding_y_plus(speed/mid,kappa,E)
        upper=np.where(positive,mid,upper);lower=np.where(positive,lower,mid)
    return np.where(speed==0,0.,.5*(lower+upper))


def wall_traction(relative_tangential_velocity_m_s,distance_m,density_kg_m3,viscosity_pa_s):
    velocity=np.asarray(relative_tangential_velocity_m_s,dtype=float)
    if velocity.shape[-1] not in (2,3) or not np.isfinite(velocity).all():raise ValueError('finite 2D/3D tangential vector required')
    if not np.isfinite([density_kg_m3,viscosity_pa_s]).all() or density_kg_m3<=0 or viscosity_pa_s<=0:raise ValueError('positive finite density and viscosity required')
    speed=np.linalg.norm(velocity,axis=-1);ut=friction_velocity(speed,distance_m,viscosity_pa_s/density_kg_m3)
    direction=np.divide(velocity,speed[...,None],out=np.zeros_like(velocity),where=speed[...,None]>0)
    # Fluid-side resisting traction. Moving-wall velocity is already subtracted.
    traction=-density_kg_m3*ut[...,None]**2*direction
    return dict(traction_pa=traction,friction_velocity_m_s=ut,y_plus=np.asarray(distance_m)*ut/(viscosity_pa_s/density_kg_m3))


def wall_traction_linearization(relative_tangential_velocity_m_s,distance_m,density_kg_m3,viscosity_pa_s):
    """Spalding traction and its exact velocity Jacobian, in Pa s/m.

    Differentiate the implicit friction-velocity equation at fixed molecular
    viscosity and sampling distance. At zero speed the limiting Jacobian is
    -mu/y times the identity, so no artificial speed floor is required.

    Input is already tangential to the wall. A Cartesian boundary assembly
    must compose this Jacobian with its tangential projection and its actual
    velocity sampling map. This function alone is not a coupled wall model.
    """
    result=wall_traction(relative_tangential_velocity_m_s,distance_m,density_kg_m3,viscosity_pa_s)
    v=np.asarray(relative_tangential_velocity_m_s,float);speed=np.linalg.norm(v,axis=-1)
    ut=result['friction_velocity_m_s'];yp=result['y_plus']
    v=np.broadcast_to(v,ut.shape+(v.shape[-1],));speed=np.broadcast_to(speed,ut.shape);moving=speed>0
    up=np.divide(speed,ut,out=np.zeros_like(speed),where=moving);z=KAPPA*up
    with np.errstate(over='ignore'):
        rem=np.expm1(np.minimum(z,700))-z-.5*z*z
    rem=np.where(z<1e-3,z**3/6*(1+z/4+z*z/20+z**3/120),rem)
    derivative=1+KAPPA/E*rem
    # du_tau/dU = f'(u+) / (y+ + u+ f'(u+)).
    dut=np.divide(1.,up+np.divide(yp,derivative),out=np.zeros_like(up),where=moving)
    beta=np.broadcast_to(viscosity_pa_s/np.asarray(distance_m,float),speed.shape).copy()
    np.divide(density_kg_m3*ut**2,speed,out=beta,where=moving)
    radial=np.where(moving,2*density_kg_m3*ut*dut,beta)
    direction=np.divide(v,speed[...,None],out=np.zeros_like(v),where=moving[...,None])
    jac=-beta[...,None,None]*np.eye(v.shape[-1])-(radial-beta)[...,None,None]*direction[..., :,None]*direction[...,None,:]
    result.update(traction_jacobian=jac,tangential_secant_coefficient=beta,
                  radial_tangent_coefficient=radial)
    return result


def torch_wall_traction(relative_tangential_velocity_m_s,distance_m,density_kg_m3,viscosity_pa_s):
    """Same constitutive inversion on real CPU/CUDA torch tensors, no host solve."""
    import torch
    v=relative_tangential_velocity_m_s
    if not isinstance(v,torch.Tensor) or v.dtype!=torch.float64 or v.shape[-1] not in (2,3):raise ValueError('float64 torch tangential vectors required')
    y=torch.as_tensor(distance_m,device=v.device,dtype=v.dtype)
    if not bool(torch.isfinite(v).all()) or not bool(torch.isfinite(y).all()) or bool(torch.any(y<=0)) or density_kg_m3<=0 or viscosity_pa_s<=0:raise ValueError('finite vectors and positive SI parameters required')
    speed=torch.linalg.vector_norm(v,dim=-1);nu=viscosity_pa_s/density_kg_m3;lower=torch.sqrt(nu*speed/y);upper=2*torch.maximum(lower,speed)
    tiny=torch.finfo(v.dtype).tiny
    for _ in range(90):
        mid=torch.clamp(.5*(lower+upper),min=tiny);u=speed/mid;z=KAPPA*u
        rem=torch.expm1(torch.clamp(z,max=700))-z-.5*z*z-z**3/6
        series=z**4/24*(1+z/5+z*z/30+z**3/210);rem=torch.where(z<1e-3,series,rem)
        law=u+torch.clamp(rem,min=0)/E;positive=y*mid/nu>=law;upper=torch.where(positive,mid,upper);lower=torch.where(positive,lower,mid)
    ut=torch.where(speed==0,torch.zeros_like(speed),.5*(lower+upper));direction=v/torch.clamp(speed[...,None],min=tiny)
    return dict(traction_pa=-density_kg_m3*ut[...,None]**2*direction,friction_velocity_m_s=ut,y_plus=y*ut/nu)
