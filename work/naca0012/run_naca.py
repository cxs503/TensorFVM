#!/usr/bin/env python3
"""NACA 0012 (TMR) through the 30P30N solver lineage with identical numerical settings to
scripts/run_three_element_steady.py: SA with nu_tilde_inf = 3 nu, relaxation .5/.3/.3,
skew-corrected Gauss pressure gradient, bounded linear upwind blended in over iterations
40-100, pseudo time step .02 for the first 100 iterations, krylov linear solvers.

usage: run_naca.py MESH OUTDIR [alpha=10] [Re=6e6] [max_it=1500] [save_every=10]
environment switches (experiments):
  SA_LINEAR_UPWIND=1   second-order (bounded linear upwind) convection of nu_tilde
  UNBOUNDED=1          linear upwind without the Barth-Jespersen bound
  NO_STRESS=1          drop the transpose/deviatoric stress correction source
  BLEND=x              fixed convection blend (0 = first-order upwind) after the ramp
  U_RELAX / P_RELAX / SA_RELAX   relaxation factors
Reference (TMR, CFL3D SA, Re 6e6, M 0.15): alpha 10: Cl 1.0909 Cd 0.01231 (7 codes 1.089-1.100
/ 0.01225-0.01245); alpha 0: Cd 0.00819; alpha 15: Cl 1.5461 Cd 0.02124.
Writes JSON lines (it, residuals, Cl, Cd, Cf stations), OUTDIR/wall.dat and a resumable
OUTDIR/checkpoint.npz every 5*save_every iterations.
"""
import os, sys, json, time, numpy as np, torch
from pathlib import Path
from dataclasses import replace, asdict
from tensorfvm.solver import SolverConfig
from tensorfvm.multi_element_skew_pressure import SkewPressureMultiElementSolver

torch.set_num_threads(1)
mesh = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 10.
Re = float(sys.argv[4]) if len(sys.argv) > 4 else 6e6
maxit = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
every = int(sys.argv[6]) if len(sys.argv) > 6 else 10
env = os.environ.get
config = SolverConfig(nx=1, ny=1, length=1000, height=1000, cylinder_radius=None, mesh_type='three-element',
                      mesh_file=str(mesh), airfoil_chord=1, inlet_velocity=1, density=1, reynolds=Re,
                      angle_of_attack=alpha, turbulence_model='spalart-allmaras', sa_freestream_ratio=3,
                      velocity_relaxation=float(env('U_RELAX', .5)), pressure_relaxation=float(env('P_RELAX', .3)),
                      turbulence_relaxation=float(env('SA_RELAX', .3)), pseudo_time_step=.02, tolerance=1e-5,
                      max_iterations=maxit, device='cpu')
config.single_element = True
s = SkewPressureMultiElementSolver(config, bounded=not int(env('UNBOUNDED', '0')))
s.linear_backend = 'krylov'; s.lu_refresh_interval = 10
s.sa_linear_upwind = bool(int(env('SA_LINEAR_UPWIND', '0')))
if int(env('NO_STRESS', '0')):
    s.stress_correction = lambda velocity: torch.zeros_like(s.S)
bounds = s.mesh.vertices.max(0).values - s.mesh.vertices.min(0).values
s.config = replace(s.config, length=float(bounds[0]), height=float(bounds[1])); config = s.config
if not (out / 'mesh.msh').exists():
    os.symlink(os.path.relpath(mesh, out), out / 'mesh.msh')
json.dump(dict(reynolds=Re, alpha=alpha, mesh=str(mesh), env={k: v for k, v in os.environ.items()
          if k in ('SA_LINEAR_UPWIND', 'UNBOUNDED', 'NO_STRESS', 'BLEND', 'U_RELAX', 'P_RELAX', 'SA_RELAX')}),
          open(out / 'config.json', 'w'))
print('cells', s.count, 'nu', s.kinematic_viscosity, 'alpha', alpha, 'bounded', s.bounded,
      'sa_linear_upwind', s.sa_linear_upwind, 'env', dict(os.environ) and {k: env(k) for k in
      ('SA_LINEAR_UPWIND', 'UNBOUNDED', 'NO_STRESS', 'BLEND') if env(k)}, flush=True)
resumed = 0
if (out / 'checkpoint.npz').exists():
    with np.load(out / 'checkpoint.npz') as ck:
        s.velocity.copy_(torch.from_numpy(ck['velocity'])); s.p.copy_(torch.from_numpy(ck['pressure']))
        s.nu_tilde.copy_(torch.from_numpy(ck['nu_tilde'])); s.mass_flux.copy_(torch.from_numpy(ck['mass_flux']))
        resumed = int(ck['iteration'])
    s.turbulent_kinematic_viscosity = s._sa_eddy_viscosity(s.nu_tilde); print('resumed from iteration', resumed, flush=True)

def wall_data():
    wall = s.mesh.masks['main']; own = s.o[wall]; fc = s.mesh.face_centers[wall]; cc = s.mesh.centers.reshape(-1, 2)[own]
    n = -s.mesh.face_normals[wall]; n = n / n.norm(dim=1, keepdim=True); y = ((cc - fc) * n).sum(1).abs()
    t = torch.stack([n[:, 1], -n[:, 0]], 1); ut = (s.velocity[own] * t).sum(1)
    cf = 2 * s.kinematic_viscosity * ut / y; cp = 2 * s.p[own]
    return fc.numpy(), n.numpy(), cp.numpy(), cf.numpy()

def save():
    fc, n, cp, cf = wall_data(); np.savetxt(out / 'wall.dat', np.c_[fc, n, cp, cf], header='x y nx ny Cp Cf(along t=(ny,-nx))')
    np.savez_compressed(out / 'checkpoint.npz', velocity=s.velocity.numpy(), pressure=s.p.numpy(), nu_tilde=s.nu_tilde.numpy(),
                        mass_flux=s.mass_flux.numpy(), iteration=resumed + len(history))

start = time.perf_counter(); history = []
for i in range(resumed, maxit):
    s.convection_blend = min(1., max(0., (i - 40) / 60))
    if env('BLEND') is not None and i >= 100: s.convection_blend = float(env('BLEND'))
    if i >= 100: s.config = replace(s.config, pseudo_time_step=None)
    m = s.step(); history.append(m)
    if i % every == 0 or s.converged or i == maxit - 1:
        c = s._aerodynamic_coefficients(); fc, n, cp, cf = wall_data(); up = n[:, 1] > 0
        def at(xx):
            k = np.flatnonzero(up); j = k[np.abs(fc[k, 0] - xx).argmin()]; return float(cf[j])
        print(json.dumps(dict(it=i, dU=m['velocity_change'], mom=m['momentum'], sa=m['turbulence'], blend=s.convection_blend,
                              Cl=c['lift'], Cd=c['drag'], cp_min=float(cp.min()), cf_up_0p05=at(.05), cf_up_0p1=at(.1),
                              cf_up_0p3=at(.3), cf_up_0p6=at(.6), cf_up_0p9=at(.9), reversed_upper=int(((cf < 0) & up).sum()),
                              elapsed=time.perf_counter() - start)), flush=True)
    if (i + 1) % (every * 5) == 0 or s.converged: save()
    if s.converged: break
save(); json.dump(dict(config=asdict(config), history=history, cells=s.count), open(out / 'history.json', 'w'))
