"""Steady multi-element flow with exact sparse solves, live metrics and restart."""
import argparse,json,time,shutil
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
import torch
from tensorfvm.solver import SolverConfig
from tensorfvm.multi_element_flow import MultiElementFlowSolver
from tensorfvm.benchmark_30p30n import export_result,write_gmsh_geo
from tensorfvm.verification.core import provenance,sha,write_json,artifact_manifest


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mesh-file',type=Path,required=True);p.add_argument('--reynolds',type=float,default=9000000);p.add_argument('--chord',type=float,default=1);p.add_argument('--alpha',type=float,default=0);p.add_argument('--max-iterations',type=int,default=2000);p.add_argument('--resume',action='store_true');p.add_argument('--anderson-depth',type=int,default=0);p.add_argument('--tolerance',type=float,default=1e-5);p.add_argument('--pressure-gradient',choices=['least-squares','gauss-linear','gauss-skew-corrected'],default='least-squares');p.add_argument('--allow-poor-mesh',action='store_true');p.add_argument('--sa-newton',action='store_true');p.add_argument('--convection-blend',type=float);p.add_argument('--implicit-convection',action='store_true');p.add_argument('--coupled',action='store_true');p.add_argument('--initialize-from',type=Path);p.add_argument('--sa-relaxation',type=float,default=.3);p.add_argument('--u-relaxation',type=float,default=.5);p.add_argument('--p-relaxation',type=float,default=.3);p.add_argument('--sa-inner-iterations',type=int,default=3);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
 if args.convection_blend is not None and not 0<=args.convection_blend<=1:p.error('Convection blend must be in [0,1]')
 if any(out.iterdir()) and not args.resume:p.error('Output must be empty unless --resume')
 config=SolverConfig(nx=1,ny=1,length=60,height=60,cylinder_radius=None,mesh_type='three-element',mesh_file=str(args.mesh_file.resolve()),airfoil_chord=args.chord,inlet_velocity=1,density=1,reynolds=args.reynolds,angle_of_attack=args.alpha,turbulence_model='spalart-allmaras',sa_freestream_ratio=3,velocity_relaxation=args.u_relaxation,pressure_relaxation=args.p_relaxation,turbulence_relaxation=args.sa_relaxation,pseudo_time_step=.02,tolerance=args.tolerance,max_iterations=args.max_iterations,device='cpu')
 from tensorfvm.hlpw30p30n import mesh_quality
 from tensorfvm.multi_element_coupled import CoupledMultiElementSolver
 base=CoupledMultiElementSolver if args.coupled else MultiElementFlowSolver
 if args.pressure_gradient=='least-squares':
  from tensorfvm.multi_element_least_squares import LeastSquaresMultiElementSolver,LeastSquaresCoupledMultiElementSolver
  base=LeastSquaresCoupledMultiElementSolver if args.coupled else LeastSquaresMultiElementSolver
 if args.pressure_gradient=='gauss-skew-corrected':
  from tensorfvm.multi_element_skew_pressure import SkewPressureMultiElementSolver,SkewPressureCoupledMultiElementSolver
  base=SkewPressureCoupledMultiElementSolver if args.coupled else SkewPressureMultiElementSolver
 if args.sa_newton:
  from tensorfvm.sa_newton import NewtonSaMixin
  class NewtonSolver(NewtonSaMixin,base):pass
  base=NewtonSolver
 solver=base(config);solver.implicit_convection=args.implicit_convection;quality=mesh_quality(solver.mesh,args.alpha)
 if not quality['accepted'] and not args.allow_poor_mesh:p.error('Mesh quality rejected: '+json.dumps(quality))
 bounds=solver.mesh.vertices.max(0).values-solver.mesh.vertices.min(0).values;config=replace(config,length=float(bounds[0]),height=float(bounds[1]));solver.config=config;solver.anderson_depth=args.anderson_depth;solver.sa_inner_iterations=1 if args.sa_newton else args.sa_inner_iterations;prov=provenance();prov['numerical_settings']=dict(anderson_depth=args.anderson_depth,sa_inner_iterations=solver.sa_inner_iterations,coupled=args.coupled,sa_newton=args.sa_newton,sa_vorticity_floor=solver.sa_vorticity_floor,pressure_gradient=args.pressure_gradient,momentum_diffusion='fully implicit nonorthogonal',momentum_linearization='implicit frozen bounded convection' if args.implicit_convection else 'deferred bounded convection',fixed_convection_blend=args.convection_blend,sa_relaxation='single implicit',viscous_stress='full symmetric deviatoric variable viscosity')
 sources=['scripts/run_three_element_steady.py','src/tensorfvm/multi_element_flow.py','src/tensorfvm/hlpw30p30n.py','src/tensorfvm/multi_element_coupled.py','src/tensorfvm/multi_element_least_squares.py','src/tensorfvm/sa_newton.py','src/tensorfvm/multi_element_skew_pressure.py','src/tensorfvm/external_anderson.py','src/tensorfvm/external_coupled.py','src/tensorfvm/coupled_coarse.py','src/tensorfvm/multi_element.py','src/tensorfvm/benchmark_30p30n.py','src/tensorfvm/external_cached.py','src/tensorfvm/external_linear.py','src/tensorfvm/external_high_order.py','src/tensorfvm/sparse_simple.py','src/tensorfvm/conservative_pressure.py']
 for name in sources:prov['source_sha256'][name]=sha(root/name)
 if args.initialize_from and args.resume:p.error('Use either initialize-from or resume')
 if args.initialize_from:
  previous=args.initialize_from.resolve();oldconfig=json.loads((previous/'config.json').read_text())
  for key in ['reynolds','airfoil_chord','angle_of_attack']:
   if oldconfig[key]!=asdict(config)[key]:raise ValueError('Initialization condition mismatch: '+key)
  if sha(previous/'mesh.msh')!=sha(args.mesh_file):raise ValueError('Initialization mesh mismatch')
  with np.load(previous/'checkpoint.npz') as checkpoint:
   for name,target in [('velocity',solver.velocity),('pressure',solver.p),('mass_flux',solver.mass_flux),('nu_tilde',solver.nu_tilde)]:
    value=checkpoint[name]
    if value.shape!=tuple(target.shape) or not np.isfinite(value).all():raise ValueError('Invalid initial checkpoint')
    target.copy_(torch.from_numpy(value))
  solver.turbulent_kinematic_viscosity=solver._sa_eddy_viscosity(solver.nu_tilde);solver.config=replace(config,pseudo_time_step=None);config=solver.config
  prov['initialization']=dict(directory=str(previous),checkpoint_sha256=sha(previous/'checkpoint.npz'),producer_sha256=sha(previous/'producer.json'))
 if args.resume:
  old=json.loads((out/'producer.json').read_text())
  if old.get('numerical_settings')!=prov['numerical_settings']:raise ValueError('Restart numerical settings mismatch')
  if old['source_sha256']!=prov['source_sha256']:raise ValueError('Restart requires identical numerical producer')
  oldconfig=json.loads((out/'config.json').read_text())
  for key in ['mesh_file','reynolds','airfoil_chord','angle_of_attack','tolerance','velocity_relaxation','pressure_relaxation','turbulence_relaxation']:
   if oldconfig[key]!=asdict(config)[key]:raise ValueError('Restart configuration mismatch: '+key)
  checkpoint=np.load(out/'checkpoint.npz');solver.velocity.copy_(torch.from_numpy(checkpoint['velocity']));solver.p.copy_(torch.from_numpy(checkpoint['pressure']));solver.mass_flux.copy_(torch.from_numpy(checkpoint['mass_flux']));solver.nu_tilde.copy_(torch.from_numpy(checkpoint['nu_tilde']));solver.turbulent_kinematic_viscosity=solver._sa_eddy_viscosity(solver.nu_tilde);solver.history=json.loads(str(checkpoint['history'].item())) if 'history' in checkpoint.files else json.loads((out/'history-live.json').read_text());checkpoint.close()
 else:
  write_json(out/'producer.json',prov);write_json(out/'config.json',asdict(config));shutil.copy2(args.mesh_file,out/'mesh.msh')
  mesh_directory=args.mesh_file.resolve().parent
  for name in ['mesh-metadata.json','manifest.json']:
   if (mesh_directory/name).exists():shutil.copy2(mesh_directory/name,out/('mesh-producer-'+name))
  for name in prov['source_sha256']:
   target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes())
 write_json(out/'mesh-quality.json',quality)
 initialized_stage=bool(args.initialize_from) or (args.resume and bool(old.get('initialization')))
 if initialized_stage:solver.config=replace(solver.config,pseudo_time_step=None)
 start=time.perf_counter();failure=None;stage_converged=False
 def save():
  np.savez_compressed(out/'checkpoint-next.npz',iteration=len(solver.history),history=json.dumps(solver.history),velocity=solver.velocity.numpy(),pressure=solver.p.numpy(),mass_flux=solver.mass_flux.numpy(),nu_tilde=solver.nu_tilde.numpy());(out/'checkpoint-next.npz').replace(out/'checkpoint.npz')
  write_json(out/'sa-newton-history.json',getattr(solver,'sa_newton_history',[]));write_json(out/'coupled-history.json',getattr(solver,'coupled_history',[]));write_json(out/'history-live.json',solver.history);write_json(out/'linear-history.json',solver.linear_history)
 try:
  while len(solver.history)<args.max_iterations and not solver.converged:
   i=len(solver.history);solver.convection_blend=1. if initialized_stage else min(1.,max(0.,(i-40)/60))
   if args.convection_blend is not None:solver.convection_blend=args.convection_blend
   if i>=100:solver.config=replace(solver.config,pseudo_time_step=None)
   metrics=solver.step()
   if not all(np.isfinite(v) for v in metrics.values()):raise RuntimeError('Nonfinite outer metrics')
   if i%10==0 or solver.converged:print(json.dumps(dict(metrics,elapsed_s=time.perf_counter()-start,coefficients=solver._aerodynamic_coefficients())),flush=True)
   if (i+1)%10==0 or solver.converged:save()
   if args.convection_blend is not None and args.convection_blend<1:
    stage_converged=max(metrics[k] for k in ('momentum','turbulence','continuity','mass_imbalance','rhie_chow_max_normal_velocity_defect'))<args.tolerance and metrics['rhie_chow_flux_defect']<1e-7
    if stage_converged:break
 except (RuntimeError,ValueError,KeyboardInterrupt) as e:failure=str(e) or 'Interrupted; only complete checkpoints retained';print('FAILED',failure,flush=True)
 if failure is None:
  save();solver.config=replace(solver.config,max_iterations=len(solver.history));wall_pressure=solver.reconstructed_pressure_faces(solver.p) if hasattr(solver,'reconstructed_pressure_faces') else None;export_result(solver.solve(),out,wall_pressure=wall_pressure)
 write_json(out/'steady-status.json',dict(converged=solver.converged and failure is None,stage_converged=stage_converged,fixed_convection_blend=args.convection_blend,failure=failure,iterations=len(solver.history),elapsed_s=time.perf_counter()-start,computed=solver.history[-1] if solver.history else None,physics_accepted=False,compressibility_model='incompressible',momentum_convection='bounded linear upwind',pressure_gradient=args.pressure_gradient,reference_chord=args.chord,mesh_sha256=sha(out/'mesh.msh')))
 write_json(out/'manifest.json',dict(source_snapshot_root='source-snapshot',source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(out)))
 return 0 if solver.converged and failure is None else 2

if __name__=='__main__':raise SystemExit(main())
