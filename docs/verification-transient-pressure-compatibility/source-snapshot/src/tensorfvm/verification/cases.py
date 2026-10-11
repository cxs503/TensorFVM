"""Physical cases provide solvers and references; export/gates/reporting are shared."""
from dataclasses import asdict
import math
import time
import numpy as np
import torch
from .core import Metric, ERROR_LIMIT, evaluate, relative_error
from ..solver import SimpleSolver, SolverConfig
from ..periodic_mac import PeriodicMACConfig, PeriodicMACSolver


def array(t):
    return t.detach().cpu().numpy()


def channel(ny=12, max_iterations=1000):
    started=time.perf_counter()
    c=SolverConfig(nx=3*ny,ny=ny,length=.12,height=.02,
                   inlet_velocity=.0005,density=1000.,reynolds=1.,
                   cylinder_radius=None,tolerance=1e-6,max_iterations=max_iterations)
    s=SimpleSolver(c);r=s.solve()
    u,v=r.cell_center_velocity();u,v,p=array(u),array(v),array(r.p)
    x,y=array(r.x),array(r.y);dx,dy=c.length/c.nx,c.height/c.ny
    region=(x>=3*c.height)&(x<=5*c.height)
    section=int(np.argmin(abs(x-5*c.height)))
    exact=6*c.inlet_velocity*y/c.height*(1-y/c.height)
    px=p[:,region].mean(0);xr=x[region];xx=xr-xr.mean()
    slope=float(np.dot(xx,px-px.mean())/np.dot(xx,xx))
    reference_slope=-12*c.viscosity*c.inlet_velocity/c.height**2
    # Pressure gauge is aligned by its mean in the declared downstream window.
    pressure_ref=px.mean()+reference_slope*(xr-xr.mean())
    wall=-c.viscosity*np.stack((u[0],u[-1]))/(.5*dy)
    shear=6*c.viscosity*c.inlet_velocity/c.height
    uf,vf=array(r.u),array(r.v)
    div=(uf[:,1:]-uf[:,:-1])/dx+(vf[1:]-vf[:-1])/dy
    inlet=c.density*dy*uf[:,0].sum();outlet=c.density*dy*uf[:,-1].sum()
    last=r.history[-1];length=region.sum()*dx
    wallforce=float(wall[:,region].sum()*dx);pressureforce=-slope*c.height*length
    metrics=[
      Metric("velocity_l2",relative_error(u[:,section],exact),ERROR_LIMIT,"u profile at x nearest 5H"),
      Metric("velocity_linf",relative_error(u[:,section],exact,"linf"),ERROR_LIMIT,"u profile peak-normalized maximum error"),
      Metric("pressure_gradient",abs(slope-reference_slope)/abs(reference_slope),ERROR_LIMIT,"least-squares dp/dx in 3H<=x<=5H"),
      Metric("pressure_curve_l2",relative_error(px-px.mean(),pressure_ref-px.mean()),ERROR_LIMIT,"mean-gauge aligned downstream pressure curve"),
      Metric("wall_shear_linf",float(np.max(abs(wall[:,region]+shear))/shear),ERROR_LIMIT,"half-cell wall shear, downstream only"),
      Metric("force_balance",abs(wallforce+pressureforce)/abs(pressureforce),ERROR_LIMIT,"developed pressure/wall-force balance"),
      Metric("relative_mass_imbalance",float(abs(outlet-inlet)/abs(inlet)),1e-6,"global inlet/outlet mass balance"),
      Metric("scaled_divergence",float(np.max(abs(div))*c.height/c.inlet_velocity),1e-6,"authoritative face divergence H/Ub"),
      Metric("solver_residual",max(float(last[k]) for k in ("continuity","momentum","mass_imbalance")),c.tolerance,"final SIMPLE residual gate"),
      Metric("convergence_failure",0. if r.converged else 1.,.5,"solver explicitly converged"),
    ]
    rows,passed=evaluate(metrics)
    return dict(case="poiseuille",role="spatial",resolution=f"{c.nx}x{c.ny}",
      spacing_m=dy,config=asdict(c),elapsed_s=time.perf_counter()-started,
      iterations=len(r.history),converged=bool(r.converged),time_s=None,pressure_time_s=None,metrics=rows,passed=passed,
      reference=dict(velocity_formula="u=6 Ub (y/H)(1-y/H)",pressure_gradient_pa_m=reference_slope,
                     wall_shear_pa=shear,comparison_region_m=[3*c.height,5*c.height],
                     pressure_gauge="mean aligned only in comparison region",section_x_m=float(x[section])),
      history=r.history,fields=dict(x_m=x,y_m=y,u_m_s=u,v_m_s=v,p_pa=p,
       u_faces_m_s=uf,v_faces_m_s=vf,reference_u_profile_m_s=exact,profile_y_m=y,
       profile_u_m_s=u[:,section],pressure_x_m=xr,pressure_curve_pa=px,
       reference_pressure_curve_pa=pressure_ref,lower_wall_traction_pa=wall[0],
       upper_wall_traction_pa=wall[1],divergence_s_inv=div))


def taylor_green(n=32,dt=.0125,end_time=.5,role="spatial"):
    started=time.perf_counter()
    steps=round(end_time/dt)
    if abs(steps*dt-end_time)>1e-13:raise ValueError("end time must be an integer multiple of dt")
    c=PeriodicMACConfig(nx=n,ny=n,nz=4,time_step_s=dt,viscosity_pa_s=.1)
    s=PeriodicMACSolver(c)
    for a in (0,1):
        x,y,z=s.face_coordinates(a)
        s.velocity[a]=torch.sin(x)*torch.cos(y) if a==0 else -torch.cos(x)*torch.sin(y)
    initial=s.velocity.clone();states=[array(initial).copy()]
    for _ in range(steps):
        s.step();states.append(array(s.velocity).copy())
    nu=c.viscosity_pa_s/c.density_kg_m3
    exact=initial*math.exp(-2*nu*end_time)
    x,y,z=s.centers();pressure_time=end_time-dt/2
    exactp=.25*c.density_kg_m3*(torch.cos(2*x)+torch.cos(2*y))*math.exp(-4*nu*pressure_time)
    exactp-=exactp.mean()
    # Derived fixed-grid eigenvalue isolates temporal truncation from spatial error.
    eigen=sum(4*math.sin(s.spacing[a]/2)**2/s.spacing[a]**2 for a in (0,1))
    semiexact=initial*math.exp(-nu*eigen*end_time)
    metrics=[
      Metric("velocity_l2",relative_error(array(s.velocity),array(exact)),ERROR_LIMIT,"continuous solution at actual face locations"),
      Metric("velocity_linf",relative_error(array(s.velocity),array(exact),"linf"),ERROR_LIMIT,"continuous face velocity peak-normalized maximum error"),
      Metric("pressure_l2",relative_error(array(s.pressure),array(exactp)),ERROR_LIMIT,"continuous cell pressure at midpoint time, mean-zero gauge"),
      Metric("pressure_linf",relative_error(array(s.pressure),array(exactp),"linf"),ERROR_LIMIT,"cell pressure peak-normalized maximum error"),
      Metric("temporal_velocity_l2",relative_error(array(s.velocity),array(semiexact)),ERROR_LIMIT,"fixed-grid exact exponential; temporal error only"),
      Metric("scaled_divergence",max(h["max_divergence_s_inv"] for h in s.history),1e-10,"all-step face divergence for U0=1 m/s and k=1 /m"),
      Metric("nonlinear_residual",max(h["nonlinear_residual_m_s"] for h in s.history),1.01e-12,"all-step true nonlinear residual","m/s"),
      Metric("energy_defect",max(abs(h["energy_balance_defect_J"]) for h in s.history),1e-10,"all-step discrete work/KE defect","J"),
      Metric("energy_increase",max(0.,max(h["kinetic_after_J"]-h["kinetic_before_J"] for h in s.history)),1e-12,"no step energy increase","J"),
      Metric("momentum_change",max(max(abs(v) for v in h["momentum_change_Ns"]) for h in s.history),1e-10,"all-step total momentum change","Ns"),
      Metric("convective_power",max(abs(h["convective_power_W"]) for h in s.history),1e-10,"all-step zero convective work","W"),
      Metric("positive_viscous_power",max(0.,max(h["viscous_power_W"] for h in s.history)),1e-12,"viscous power nonpositive","W"),
      Metric("pressure_failure",0. if all(h["projection"]["pressure_converged"] for h in s.history) else 1.,.5,"all-step pressure success"),
      Metric("projection_energy_increase",max(0.,max(h["projection"]["projected_energy_J"]-h["projection"]["tentative_energy_J"] for h in s.history)),1e-12,"orthogonal pressure projection does not add KE","J"),
    ]
    rows,passed=evaluate(metrics)
    faces=array(s.velocity);center=.5*(faces+np.stack([np.roll(faces[a],1,axis=-(a+1)) for a in range(3)]))
    return dict(case="taylor-green",role=role,resolution=f"{n}x{n}x4",
      spacing_m=s.spacing[0] if role!="temporal" else dt,
      config=asdict(c),elapsed_s=time.perf_counter()-started,iterations=steps,
      time_s=end_time,pressure_time_s=pressure_time,metrics=rows,passed=passed,
      reference=dict(velocity_formula="u=sin(x)cos(y)exp(-2 nu t); v=-cos(x)sin(y)exp(-2 nu t); w=0",
       pressure_formula="p=rho/4*(cos(2x)+cos(2y))*exp(-4 nu t_mid)",
       pressure_gauge="mean zero",semidiscrete_decay_rate_s_inv=nu*eigen,
       scope="2D analytic vortex embedded in a 3D periodic grid; not 3D turbulence"),
      history=s.history,fields=dict(x_m=array(x[0,0]),y_m=array(y[0,:,0]),
       u_m_s=center[0,0],v_m_s=center[1,0],p_pa=array(s.pressure[0]),
       velocity_faces_m_s=faces,reference_velocity_faces_m_s=array(exact),
       reference_pressure_pa=array(exactp),pressure_pa_3d=array(s.pressure),
       initial_faces_m_s=array(initial),accepted_faces_m_s=np.stack(states),
       semidiscrete_reference_faces_m_s=array(semiexact),
       pressure_x_m=array(x[0,0]),pressure_curve_pa=array(s.pressure[0,n//2]),
       reference_pressure_curve_pa=array(exactp[0,n//2]),
       profile_y_m=array(y[0,:,0]),profile_u_m_s=faces[0,0,:,n//4],
       reference_u_profile_m_s=array(exact[0,0,:,n//4]),
       divergence_s_inv=array(s.divergence(s.velocity))))
