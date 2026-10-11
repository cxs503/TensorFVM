"""External TensorLBM adapter; production equilibrium/collision/stream are unmodified."""
import importlib
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .core import Metric, ERROR_LIMIT, evaluate, relative_error, sha
from .cases import array


def load_lbm(repo):
    repo=Path(repo).resolve()
    expected=repo/"src/tensorlbm"
    # Reject silently importing the legacy TensorFVM/tensorlbm directory.
    loaded=sys.modules.get("tensorlbm")
    if loaded is not None and Path(loaded.__file__).resolve().parent!=expected:
        raise ValueError("another tensorlbm checkout is already imported")
    sys.path.insert(0,str(repo/"src"))
    d=importlib.import_module("tensorlbm.d2q9")
    solver=importlib.import_module("tensorlbm.solver")
    for module in (d,solver):
        if not Path(module.__file__).resolve().is_relative_to(expected):
            raise ValueError("external LBM module provenance mismatch")
    files=["src/tensorlbm/__init__.py","src/tensorlbm/_version.py",
           "src/tensorlbm/d2q9.py","src/tensorlbm/solver.py","src/tensorlbm/boundaries.py"]
    return d,solver,dict(repository=str(repo),
      git_base=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
      source_sha256={p:sha(repo/p) for p in files})


def taylor_green_lbm(repo,n=32,mach_target=.05):
    d,solver,provenance=load_lbm(repo)
    started=time.perf_counter();torch.set_num_threads(1)
    dx=2*math.pi/n
    # Choose integer steps to hit exactly t=0.5; do not compare different clocks.
    steps=math.ceil(.5/(mach_target*dx/math.sqrt(3)))
    dt=.5/steps;scale=dx/dt;nu=.1;rho0=1.
    tau=.5+3*nu*dt/dx**2
    coord=(torch.arange(n,dtype=torch.float64)+.5)*dx
    y,x=torch.meshgrid(coord,coord,indexing="ij")
    ux=torch.sin(x)*torch.cos(y);uy=-torch.cos(x)*torch.sin(y)
    p0=.25*(torch.cos(2*x)+torch.cos(2*y))
    rho=rho0+3*p0/scale**2
    f=d.equilibrium(rho,ux/scale,uy/scale)
    initial=f.clone();history=[];states=[array(f).copy()]
    target_mass=float(f.sum())
    for k in range(1,steps+1):
        f=solver.stream(solver.collide_bgk(f,tau))
        rh,ul,vl=d.macroscopic(f)
        if not bool(torch.isfinite(f).all()):raise FloatingPointError("LBM nonfinite population")
        up,vp=ul*scale,vl*scale
        # A central diagnostic is shared with the cell-sampled analytic field.
        div=(torch.roll(up,-1,1)-torch.roll(up,1,1))/(2*dx)+(torch.roll(vp,-1,0)-torch.roll(vp,1,0))/(2*dx)
        history.append(dict(step=k,time_s=k*dt,
             relative_mass_drift=abs(float(f.sum())-target_mass)/target_mass,
             max_divergence_s_inv=float(div.abs().max()),
             kinetic_J=float(.5*dx**2*(rh*(up**2+vp**2)).sum()),
             maximum_density_variation=float((rh-rh.mean()).abs().max()/rh.mean()),
             maximum_mach=float(torch.hypot(ul,vl).max())*math.sqrt(3)))
        states.append(array(f).copy())
    rh,ul,vl=d.macroscopic(f);up,vp=ul*scale,vl*scale
    p=(rh-rh.mean())*scale**2/3
    exactu=ux*math.exp(-2*nu*.5);exactv=uy*math.exp(-2*nu*.5)
    exactp=p0*math.exp(-4*nu*.5);exactp-=exactp.mean()
    metrics=[
      Metric("velocity_l2",relative_error(array(torch.stack((up,vp))),array(torch.stack((exactu,exactv)))),ERROR_LIMIT,"continuous velocity sampled at LBM nodes"),
      Metric("velocity_linf",relative_error(array(torch.stack((up,vp))),array(torch.stack((exactu,exactv))),"linf"),ERROR_LIMIT,"node velocity reference-peak normalized maximum error"),
      Metric("pressure_l2",relative_error(array(p),array(exactp)),ERROR_LIMIT,"isothermal EOS pressure, mean zero at t=0.5 s"),
      Metric("pressure_linf",relative_error(array(p),array(exactp),"linf"),ERROR_LIMIT,"EOS pressure reference-peak maximum error"),
    ]
    rows,passed=evaluate(metrics)
    return dict(case="taylor-green-lbm",role="comparison",resolution=f"{n}x{n}",
      spacing_m=dx,config=dict(nx=n,ny=n,length_m=2*math.pi,density_kg_m3=rho0,
         viscosity_pa_s=.1,time_step_s=dt,tau=tau,mach_target=mach_target,
         actual_reference_mach=math.sqrt(3)/scale,physical_speed_scale_m_s=scale),
      elapsed_s=time.perf_counter()-started,iterations=steps,time_s=.5,pressure_time_s=.5,
      provenance=provenance,metrics=rows,passed=passed,
      reference=dict(scope="same periodic continuum problem; different face/node layouts and pressure clocks",
        initialization="analytic density/velocity, production equilibrium populations; no artificial mass correction",
        precision="float64 populations; production D2Q9 weights retain upstream float32 constants",
        comparison_limit="BGK D2Q9 baseline only, not all TensorLBM collision models or optimized GPU implementations",
        pressure_formula="p=(rho-mean(rho))*(dx/dt)^2/3"),
      performance=dict(state_storage_bytes=f.numel()*f.element_size(),
          state_scope="nine population arrays only; excludes temporary fields, cached stream indices and saved history",
          timing_scope="one CPU execution including diagnostics and raw sampling; no speed superiority inference"),
      history=history,fields=dict(x_m=array(coord),y_m=array(coord),u_m_s=array(up),v_m_s=array(vp),p_pa=array(p),
       density_kg_m3=array(rh),populations=array(f),initial_populations=array(initial),
       accepted_populations=np.stack(states),reference_u_m_s=array(exactu),reference_v_m_s=array(exactv),
       reference_pressure_pa=array(exactp),pressure_x_m=array(coord),pressure_curve_pa=array(p[n//2]),
       reference_pressure_curve_pa=array(exactp[n//2]),profile_y_m=array(coord),
       profile_u_m_s=array(up[:,n//4]),reference_u_profile_m_s=array(exactu[:,n//4]),
       divergence_s_inv=array(div)))
