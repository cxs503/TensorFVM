#!/usr/bin/env python3
"""Compare native momentum iterations at a frozen saved SUBOFF SST viscosity.

This is one momentum subproblem, not a full steady RANS qualification.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithm',choices=['simplec','newton'],default='simplec');p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--relaxation',type=float,default=.5);a=p.parse_args()
    if a.iterations<1 or not 0<a.relaxation<=1:p.error('positive iteration limit and relaxation in (0,1] required')
    root=a.run;cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if not summary['initialized'] or not summary['source_integrity_verified']:raise ValueError('verified SST fields required')
    if digest(root/'fields.npz')!=summary['fields_sha256']:raise ValueError('saved fields changed')
    for name,sha in json.loads((root/'sources.json').read_text()).items():
        if digest(root/'source-snapshot'/name)!=sha:raise ValueError('producer sources changed')
    for name,sha in cfg['initial_bindings'].items():
        if digest(root/('initial-'+name))!=sha:raise ValueError('initial binding changed')
    if cfg['rho']!=1. or cfg['U']!=1. or cfg['L']!=1.:raise ValueError('unit density, velocity and length required')
    evidence=NativeRunEvidence(a.output,__file__);out=a.output;start=time.perf_counter()
    write_json(out/'settings.json',dict(scope=__doc__,input_bindings={name:digest(root/name) for name in
        ('fields.npz','settings.json','summary.json','initial-input.msh')},run=str(root),algorithm=a.algorithm,
        iterations=a.iterations,relaxation=a.relaxation,tolerance=1e-9,viscosity='frozen saved rho*(nu+nu_t)',
        scalar_equations_advanced=False,physical_suboff_benchmark_passed=False))
    m=NativeMesh3D.from_gmsh(root/'initial-input.msh',triangulate_faces=True)
    f=np.load(root/'fields.npz',allow_pickle=False);fixed=(m.neighbor<0)&(m.patch_labels!='outlet');wall=m.patch_labels=='wall'
    velocity=np.tile([1.,0.,0.],(len(m.owner),1));velocity[wall]=0.
    mu=cfg['nu']+f['nu_t'];s=NativeHybridNavierStokes3D(m,fixed,velocity,viscosity=mu,convection_order=2,backflow_velocity=[1,0,0])
    np.testing.assert_array_equal(f['owner'],m.owner);np.testing.assert_array_equal(f['neighbor'],m.neighbor)
    np.testing.assert_array_equal(f['face_velocity'][fixed],velocity[fixed])
    s.state=np.r_[f['velocity'].ravel(),f['face_velocity'].ravel()];s.p=f['modified_pressure'].copy()
    np.savez_compressed(out/'inputs.npz',velocity=f['velocity'],face_velocity=f['face_velocity'],modified_pressure=s.p,dynamic_viscosity=mu)
    def progress(row):
        write_json(out/'progress.json',row)
        if row['iteration']%10==0 or a.algorithm=='newton':print(row,flush=True)
    failure=None
    try:
        if a.algorithm=='newton':r=s.solve_newton(max_iterations=a.iterations,tolerance=1e-9,callback=progress,require_initial_correction=True)
        else:r=s.solve(algorithm='simplec',max_outer_iterations=a.iterations,max_iterations=1,relaxation=a.relaxation,tolerance=1e-9,callback=progress)
    except (ValueError,RuntimeError) as exc:failure=str(exc);r=s.result()
    full,source=s._current_system();space=s.viscous.velocity_space
    residual=space.restrict(full@s.state+source-s.divergence.T@s.p-s.load)
    scale=max(np.linalg.norm(space.restrict(s.load-source-full@s.prescribed)),1.)
    np.savez_compressed(out/'fields.npz',**{k:r[k] for k in ('velocity','face_velocity','pressure','flux','total_traction_integrated','momentum_flux_integrated')})
    write_json(out/'history.json',getattr(s,'newton_history',[]) if a.algorithm=='newton' else s.outer_history)
    write_json(out/'linear-history.json',s.coupled_linear_history if a.algorithm=='newton' else s.history)
    evidence.verify_sources()
    result=dict(failure=failure,source_integrity_verified=True,inputs_sha256=digest(out/'inputs.npz'),fields_sha256=digest(out/'fields.npz'),
        numerically_converged=bool(r['numerically_converged']),runtime_seconds=time.perf_counter()-start,
        current_momentum_residual=float(np.linalg.norm(residual)/scale),divergence_max=float(np.max(abs(s.divergence@s.state/m.cell_volumes))),
        **{k:r[k] for k in ('local_momentum_balance_max','global_momentum_balance_max','interface_traction_defect_max','energy_balance_error')},
        Cd_L2=float(-2*r['total_traction_integrated'][wall,0].sum()),full_rans_converged=False,physical_suboff_benchmark_passed=False)
    write_json(out/'summary.json',result);print(result,flush=True)
    raise SystemExit(0 if result['numerically_converged'] else 2)


if __name__=='__main__':main()
