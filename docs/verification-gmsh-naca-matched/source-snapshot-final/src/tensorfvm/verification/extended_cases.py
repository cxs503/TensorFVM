"""Additional analytic cases using existing production FVM and external BGK kernels."""
from dataclasses import asdict
import math
import time
import numpy as np
import torch
from .cases import array
from .core import Metric, evaluate, relative_error
from .lbm import load_lbm
from ..periodic_mac import PeriodicMACConfig, PeriodicMACSolver


def exact_velocity(kind, xyz, t, nu=.1):
    x,y,z=xyz
    if kind=='abc':
        return torch.stack((torch.sin(z)+torch.cos(y),torch.sin(x)+torch.cos(z),torch.sin(y)+torch.cos(x)))*math.exp(-nu*t)
    if kind=='advected-shear':
        return torch.stack((torch.sin(y-.7*t)*math.exp(-nu*t),torch.full_like(x,.7),torch.zeros_like(x)))
    raise ValueError(kind)


def exact_pressure(kind, xyz, t):
    x,y,z=xyz
    if kind=='abc':
        v=exact_velocity(kind,xyz,t)
        p=-.5*torch.sum(v*v,0)
        return p-p.mean()
    return torch.zeros_like(x)


def analytic_metrics(kind, velocity, reference, pressure, reference_pressure):
    rows=[Metric('velocity_l2',relative_error(velocity,reference),.03,'all components at their actual staggered face positions'),
          Metric('velocity_linf',relative_error(velocity,reference,'linf'),.03,'peak-normalized maximum face velocity error')]
    if kind=='abc':
        rows += [Metric('pressure_l2',relative_error(pressure,reference_pressure),.03,'cell pressure at last midpoint; mean gauge'),
                 Metric('pressure_linf',relative_error(pressure,reference_pressure,'linf'),.03,'peak-normalized maximum pressure error')]
    else:
        rows += [Metric('shear_velocity_l2',relative_error(velocity[0],reference[0]),.03,'transported shear component; excludes constant mean flow'),
                 Metric('pressure_dynamic_scaled_linf',float(np.max(abs(pressure)))/(.5*(1+.7**2)),.03,'zero analytic pressure: absolute error normalized by initial dynamic pressure')]
    return rows


def mac_case(kind,n,dt=None,end_time=None,role='spatial'):
    dt=dt if dt is not None else (.005 if kind=='abc' else .01)
    end_time=end_time if end_time is not None else (.1 if kind=='abc' else .5)
    steps=round(end_time/dt)
    if steps<1 or abs(steps*dt-end_time)>1e-13:raise ValueError('positive integer step count required')
    start=time.perf_counter()
    c=PeriodicMACConfig(nx=n,ny=n,nz=n if kind=='abc' else 4,time_step_s=dt,viscosity_pa_s=.1)
    s=PeriodicMACSolver(c)
    for a in range(3):s.velocity[a]=exact_velocity(kind,s.face_coordinates(a),0)[a]
    states=[array(s.velocity).copy()]
    for _ in range(steps):s.step();states.append(array(s.velocity).copy())
    expected=torch.stack([exact_velocity(kind,s.face_coordinates(a),end_time)[a] for a in range(3)])
    ep=exact_pressure(kind,s.centers(),end_time-dt/2)
    metrics=analytic_metrics(kind,array(s.velocity),array(expected),array(s.pressure),array(ep))
    metrics += [Metric('scaled_divergence',max(h['max_divergence_s_inv'] for h in s.history),1e-10,'maximum over all accepted face states','1/s'),
      Metric('nonlinear_residual',max(h['nonlinear_residual_m_s'] for h in s.history),1.01e-12,'all-step true fixed-point residual','m/s'),
      Metric('energy_defect',max(abs(h['energy_balance_defect_J']) for h in s.history),1e-10,'all-step discrete kinetic/work defect','J'),
      Metric('momentum_change',max(max(abs(v) for v in h['momentum_change_Ns']) for h in s.history),1e-10,'all-step total momentum change','Ns'),
      Metric('convective_power',max(abs(h['convective_power_W']) for h in s.history),1e-10,'all-step convection work','W'),
      Metric('energy_increase',max(0.,max(h['kinetic_after_J']-h['kinetic_before_J'] for h in s.history)),1e-12,'no accepted step gains kinetic energy','J'),
      Metric('positive_viscous_power',max(0.,max(h['viscous_power_W'] for h in s.history)),1e-12,'viscous work is nonpositive','W'),
      Metric('pressure_failure',0. if all(h['projection']['pressure_converged'] for h in s.history) else 1.,.5,'all-step pressure solve success')]
    rows,passed=evaluate(metrics);x,y,z=s.centers();f=array(s.velocity)
    center=.5*(f+np.stack([np.roll(f[a],1,axis=2-a) for a in range(3)]))
    return dict(case=kind,role=role,resolution=f'{n}x{n}x{c.nz}',spacing_m=s.spacing[0],config=asdict(c),
      time_s=end_time,pressure_time_s=end_time-dt/2,iterations=steps,elapsed_s=time.perf_counter()-start,
      metrics=rows,passed=passed,history=s.history,
      reference=dict(scope='genuine three-dimensional periodic Beltrami flow' if kind=='abc' else '2D advected shear embedded in four periodic z planes',
                     mean_velocity_m_s=[0,.7,0] if kind!='abc' else [0,0,0],pressure_gauge='mean zero',
                     pressure_normalization='relative analytic norm' if kind=='abc' else 'absolute pressure / (0.5 rho (1+0.7^2))'),
      fields=dict(x_m=array(x[0,0]),y_m=array(y[0,:,0]),z_m=array(z[:,0,0]),
        u_m_s=center[0,0],v_m_s=center[1,0],w_m_s=center[2,0],p_pa=array(s.pressure[0]),
        velocity_faces_m_s=f,reference_velocity_faces_m_s=array(expected),reference_pressure_pa=array(ep),
        pressure_pa_3d=array(s.pressure),accepted_faces_m_s=np.stack(states),
        profile_y_m=array(y[0,:,0]),profile_u_m_s=f[0,0,:,n//4],reference_u_profile_m_s=array(expected[0,0,:,n//4]),
        pressure_x_m=array(x[0,0]),pressure_curve_pa=array(s.pressure[0,n//2]),reference_pressure_curve_pa=array(ep[0,n//2])))


def shear_lbm(repo,n=32,mach=.05):
    d,solver,prov=load_lbm(repo);start=time.perf_counter();dx=2*math.pi/n;end=.5
    steps=math.ceil(end*math.sqrt(3)*math.sqrt(1+.7**2)/(mach*dx));dt=end/steps;scale=dx/dt;tau=.5+.3*dt/dx**2
    coord=(torch.arange(n,dtype=torch.float64)+.5)*dx;y,x=torch.meshgrid(coord,coord,indexing='ij')
    f=d.equilibrium(torch.ones_like(x),torch.sin(y)/scale,torch.full_like(x,.7/scale));states=[array(f).copy()];history=[];mass=float(f.sum())
    for i in range(steps):
        f=solver.stream(solver.collide_bgk(f,tau));rho,u,v=d.macroscopic(f)
        if not torch.isfinite(f).all():raise ValueError('nonfinite BGK state')
        history.append(dict(step=i+1,time_s=(i+1)*dt,relative_mass_drift=abs(float(f.sum())-mass)/mass,
          kinetic_J=float(.5*dx**2*(rho*((u*scale)**2+(v*scale)**2)).sum()),maximum_mach=float(torch.hypot(u,v).max())*math.sqrt(3)))
        states.append(array(f).copy())
    rho,u,v=d.macroscopic(f);vel=torch.stack((u*scale,v*scale,torch.zeros_like(u)));ref=exact_velocity('advected-shear',(x,y,torch.zeros_like(x)),end)
    p=(rho-rho.mean())*scale**2/3
    rows,passed=evaluate(analytic_metrics('advected-shear',array(vel),array(ref),array(p),array(torch.zeros_like(p))))
    return dict(case='advected-shear-lbm',role='comparison',resolution=f'{n}x{n}',spacing_m=dx,time_s=end,pressure_time_s=end,
      iterations=steps,elapsed_s=time.perf_counter()-start,provenance=prov,config=dict(nx=n,ny=n,length_m=2*math.pi,time_step_s=dt,tau=tau,
        viscosity_pa_s=.1,density_kg_m3=1.,physical_speed_scale_m_s=scale,mach_target=mach,actual_initial_mach=math.sqrt(3)*math.sqrt(1+.7**2)/scale),
      reference=dict(scope='same continuum shear; node versus face layouts; BGK baseline only',precision='float64 populations; unchanged upstream float32 weights',
        pressure_normalization='absolute EOS pressure / initial dynamic pressure',initialization='uniform density, equilibrium with analytic velocity; no mass correction'),
      metrics=rows,passed=passed,history=history,fields=dict(x_m=array(coord),y_m=array(coord),u_m_s=array(vel[0]),v_m_s=array(vel[1]),w_m_s=array(vel[2]),p_pa=array(p),
        populations=array(f),accepted_populations=np.stack(states),reference_velocity_faces_m_s=array(ref),reference_pressure_pa=np.zeros((n,n)),
        profile_y_m=array(coord),profile_u_m_s=array(vel[0,:,n//4]),reference_u_profile_m_s=array(ref[0,:,n//4]),
        pressure_x_m=array(coord),pressure_curve_pa=array(p[n//2]),reference_pressure_curve_pa=np.zeros(n)))
