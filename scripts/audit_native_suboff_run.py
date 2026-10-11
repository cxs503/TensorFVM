#!/usr/bin/env python3
"""Replay one frozen native SUBOFF run's actual current equations, never solve."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bindings(run):
    config=json.loads((run/'config.json').read_text());sources=config.get('source_sha256',{})
    if not sources:raise ValueError('producer config has no frozen source hash bindings')
    snapshot=(run/'source-snapshot').resolve()
    for name,expected in sources.items():
        path=(snapshot/name).resolve()
        if not path.is_relative_to(snapshot):raise ValueError('source binding escapes snapshot')
        if not path.is_file() or digest(path)!=expected:raise ValueError('frozen source hash mismatch: '+name)
    mesh=run/'input.msh'
    if not config.get('mesh_sha256') or digest(mesh)!=config['mesh_sha256']:raise ValueError('input mesh hash mismatch or missing producer mesh hash')
    initial=config.get('initial_state')
    if initial is not None:
        path=run/'initial-numerical-state.npz'
        if not isinstance(initial,dict) or digest(path)!=initial.get('sha256'):raise ValueError('initial numerical state hash mismatch')
        if 'producer_result_sha256' in initial:
            result_path=run/'initial-producer-result.json'
            if digest(result_path)!=initial['producer_result_sha256']:raise ValueError('initial producer result hash mismatch')
            if json.loads(result_path.read_text())['converged']!=initial.get('producer_converged'):raise ValueError('initial producer convergence metadata mismatch')
    merged=config.get('merge_coplanar_faces',False)
    if not isinstance(merged,bool):raise ValueError('merge_coplanar_faces must be boolean')
    return config


def relative_l2(actual,reference,volume,remove_mean=False):
    import numpy as np
    a=np.array(actual,dtype=float,copy=True);r=np.array(reference,dtype=float,copy=True);v=np.asarray(volume,float)
    if a.shape!=r.shape or a.shape[0]!=len(v) or np.any(v<=0) or not all(np.isfinite(x).all() for x in (a,r,v)):raise ValueError('invalid error arrays or positive volume weights')
    if remove_mean:
        if a.ndim!=1:raise ValueError('pressure mean removal requires scalar field')
        a-=np.sum(v*a)/v.sum();r-=np.sum(v*r)/v.sum()
    weight=v if a.ndim==1 else v[:,None]
    denominator=np.sum(weight*r*r)
    if denominator<=0:raise ValueError('zero reference norm')
    return float(np.sqrt(np.sum(weight*(a-r)**2)/denominator))


def worker(run,out):
    """Only this subprocess imports tensorfvm, exclusively from launch snapshot."""
    import numpy as np
    import inspect
    config=verify_bindings(run);snapshot=(run/'source-snapshot'/'src').resolve()
    sys.path.insert(0,str(snapshot))
    from tensorfvm.native_mesh3d import NativeMesh3D
    from tensorfvm.native_collocated3d import NativeSteadySolver3D,NativeSteadySettings
    from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux
    import importlib.util
    acceptance_path=out/'audit-acceptance-core.py'
    spec=importlib.util.spec_from_file_location('_audit_acceptance_core',acceptance_path)
    acceptance=importlib.util.module_from_spec(spec);sys.modules[spec.name]=acceptance;spec.loader.exec_module(acceptance)
    Metric,evaluate=acceptance.Metric,acceptance.evaluate
    settings_data=config.get('settings')
    if not isinstance(settings_data,dict):raise ValueError('explicit producer settings are required for exact replay')
    settings=NativeSteadySettings(**settings_data)
    mesh=NativeMesh3D.from_gmsh(run/'input.msh',triangulate_faces=True)
    merge_metadata=None
    if config.get('merge_coplanar_faces',False):
        from tensorfvm.native_face_geometry3d import merge_coplanar_faces
        mesh,merge_metadata=merge_coplanar_faces(mesh)
    mode=config.get('mode');reconstruction=config.get('reconstruction','linear')
    if mode=='manufactured':
        field=lambda p:abc_fields(p,nu=settings.viscosity)
        exact=field(mesh.cell_centers);extra={}
        cls=NativeSteadySolver3D
        if reconstruction=='quadratic':
            from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
            from tensorfvm.native_manufactured3d import exact_boundary_velocity_means
            cls=NativeQuadraticSteadySolver3D
            extra=dict(boundary_velocity_means=exact_boundary_velocity_means(mesh,field),boundary_velocity_gradients=field(mesh.face_centers)['velocity_gradient'])
            algorithm=config.get('quadratic_linear_solver','matrix-free')
            if 'quadratic_linear_solver' in inspect.signature(cls.__init__).parameters:extra['quadratic_linear_solver']=algorithm
            elif algorithm!='matrix-free':raise ValueError('frozen solver cannot reproduce requested assembled algorithm')
            viscous=config.get('viscous_reconstruction','quadratic')
            if 'viscous_reconstruction' in inspect.signature(cls.__init__).parameters:extra['viscous_reconstruction']=viscous
            elif viscous!='quadratic':raise ValueError('frozen solver cannot reproduce requested viscous reconstruction')
        elif reconstruction!='linear':raise ValueError('unsupported reconstruction')
        solver=cls(mesh,mesh.neighbor<0,field(mesh.face_centers)['velocity'],**extra,settings=settings,body_force=exact['acceleration'],boundary_flux=exact_boundary_flux(mesh,field),gauge_value=exact['pressure'][0])
    elif mode=='laminar' and reconstruction=='linear':
        boundary=mesh.neighbor<0;fixed=boundary & (mesh.patch_labels!='outlet');velocity=np.tile([1.,0.,0.],(len(mesh.owner),1));velocity[mesh.patch_labels=='wall']=0
        solver=NativeSteadySolver3D(mesh,fixed,velocity,pressure_dirichlet=mesh.patch_labels=='outlet',settings=settings);exact=None
    else:raise ValueError('unsupported physical/reconstruction mode')
    with np.load(run/'fields.npz',allow_pickle=False) as fields:
        u=fields['velocity'].copy();p=fields['pressure'].copy();q=fields['flux'].copy()
        if u.shape!=(len(mesh.cells),3) or p.shape!=(len(mesh.cells),) or q.shape!=(len(mesh.owner),):raise ValueError('actual field shapes mismatch reconstructed native topology')
        if not all(np.isfinite(x).all() for x in (u,p,q)):raise ValueError('actual numerical fields are nonfinite')
        references={k:fields[k].copy() for k in ('reference_velocity','reference_pressure') if k in fields}
    solver.u,solver.p,solver.flux=u,p,q
    if reconstruction=='quadratic' and hasattr(solver,'_assess_state'):
        *_,residuals=solver._assess_state(u,p,q);replay_api='frozen quadratic _assess_state(actual u,p,q)'
    else:
        block,rhs,_,correction,source=solver._system(q)
        consistent=solver.velocity_flux_matrix@u.ravel()+correction@p+source
        residual=block@np.r_[u.ravel(),p]-rhs
        residuals=dict(momentum_residual=float(np.linalg.norm(residual[:3*solver.N])/max(np.linalg.norm(solver.force_integrated)+np.linalg.norm(solver.diffusion_source)+np.linalg.norm(solver.p_force_source),1)),native_divergence_max=float(np.max(abs(solver.viscous.op.integrate(q)/mesh.cell_volumes))),current_reconstructed_divergence_max=float(np.max(abs(solver.viscous.op.integrate(consistent)/mesh.cell_volumes))),face_flux_consistency_max=float(np.max(abs(q-consistent)/np.linalg.norm(mesh.area_vectors,axis=1))))
        replay_api='frozen current _system(actual q), coupled operator on actual u,p; no linear solve'
    ledger=json.loads((run/'result.json').read_text());history=ledger.get('history',[])
    metrics=[Metric('recorded-nonlinear-failure',int(ledger.get('converged') is not True),.5,'Actual producer converged flag required'),Metric('missing-recorded-history',int(not history),.5,'Producer nonlinear change ledger required')]
    tol=settings.nonlinear_tolerance
    metrics.extend(Metric(key,float(value),tol,'Independent current-equation replay') for key,value in residuals.items())
    if history:
        for key in ('velocity_change','pressure_change','flux_change'):metrics.append(Metric('recorded-'+key,float(history[-1].get(key,1.)),tol,'Recorded last numerical update, not independently recomputed'))
    errors={}
    if exact is not None:
        errors=dict(velocity=relative_l2(u,exact['velocity'],mesh.cell_volumes),pressure=relative_l2(p,exact['pressure'],mesh.cell_volumes,True))
        errors.update({f'velocity_{axis}':relative_l2(u[:,i],exact['velocity'][:,i],mesh.cell_volumes) for i,axis in enumerate('xyz')})
        metrics.extend(Metric('relative-L2-'+key,value,.03,'Independent continuum field at actual volume centroid') for key,value in errors.items())
        for name,exact_name in (('reference_velocity','velocity'),('reference_pressure','pressure')):
            metrics.append(Metric(name+'-missing',int(name not in references),.5,'Archived reference required for independent replay consistency'))
            if name in references:
                if references[name].shape!=exact[exact_name].shape:raise ValueError('archived reference field shape mismatch')
                metrics.append(Metric(name+'-independent-reference-defect',float(np.max(abs(references[name]-exact[exact_name]))),1e-12,'Frozen analytic expression versus archived reference; not used to initialize solution'))
    initial=config.get('initial_state')
    if initial is not None:
        with np.load(run/'initial-numerical-state.npz',allow_pickle=False) as seed:
            finite=seed['velocity'].shape==u.shape and seed['pressure'].shape==p.shape and np.isfinite(seed['velocity']).all() and np.isfinite(seed['pressure']).all()
        metrics.append(Metric('invalid-initial-numerical-fields',int(not finite),.5,'Bound numerical initialization cell fields'))
    imported={}
    for name,module in list(sys.modules.items()):
        if (name=='tensorfvm' or name.startswith('tensorfvm.')) and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve()
            if not path.is_relative_to(snapshot):raise ValueError('nonfrozen tensorfvm module imported: '+name)
            relative=str(path.relative_to(run/'source-snapshot'));expected=config['source_sha256'].get(relative)
            if expected is None:raise ValueError('imported numerical module is not bound by producer: '+name)
            if digest(path)!=expected:raise ValueError('imported frozen source hash mismatch: '+name)
            imported[name]=dict(path=str(path),sha256=digest(path),bound_by_producer=expected is not None)
    records,passed=evaluate(metrics)
    return dict(passed=bool(passed and exact is not None),current_equation_and_recorded_change_gates_passed=all(r['passed'] for r in records if not r['name'].startswith(('relative-L2-','reference_'))),manufactured_pointwise_accuracy_passed=bool(passed and exact is not None),mesh_convergence_qualified=False,experimental_accuracy_qualified=False,suboff_physical_benchmark_passed=False,metrics=records,relative_L2_errors=errors,current_equation_residuals=residuals,recorded_final_history=history[-1] if history else None,replay_api=replay_api,mode=mode,reconstruction=reconstruction,quadratic_linear_solver=config.get('quadratic_linear_solver','matrix-free'),face_policy='merged_coplanar_polygon_faces' if merge_metadata else 'triangulated_native_faces',cells=len(mesh.cells),faces=len(mesh.owner),frozen_imports=imported,limitations=['Single-grid pointwise MMS verification is not physical SUBOFF drag/Cp validation.','Recorded field-change gates cannot be independently reconstructed from one final state.','No linear solve or time advancement is performed by this audit.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    if args.worker:
        result=worker(run,out);(out/'worker-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return 0
    if out.exists():parser.error('audit output must be new to preserve evidence')
    out.mkdir(parents=True)
    try:config=verify_bindings(run)
    except Exception as error:
        shutil.copy2(Path(__file__).resolve(),out/'executed-audit.py')
        result=dict(passed=False,provenance_verified=False,error_type=type(error).__name__,error=str(error),experimental_accuracy_qualified=False,suboff_physical_benchmark_passed=False)
        (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.iterdir() if p.is_file()},indent=2)+'\n')
        print(json.dumps(result,indent=2));return 2
    inputs=[run/name for name in ('config.json','input.msh','fields.npz','result.json')]
    if config.get('initial_state') is not None:inputs.append(run/'initial-numerical-state.npz')
    if (config.get('initial_state') or {}).get('producer_result_sha256'):inputs.append(run/'initial-producer-result.json')
    before={str(p):digest(p) for p in inputs};script=Path(__file__).resolve();shutil.copy2(script,out/'executed-audit.py')
    acceptance_source=script.parents[1]/'src/tensorfvm/verification/core.py'
    shutil.copy2(acceptance_source,out/'audit-acceptance-core.py')
    env=os.environ.copy();env['PYTHONPATH']=str(run/'source-snapshot'/'src');env['PYTHONDONTWRITEBYTECODE']='1';env.setdefault('OMP_NUM_THREADS','1');env.setdefault('OPENBLAS_NUM_THREADS','1');env.setdefault('MKL_NUM_THREADS','1')
    command=[sys.executable,str(script),'--worker','--run',str(run),'--output',str(out)]
    with (out/'replay.log').open('w') as stream:completed=subprocess.run(command,cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT)
    unchanged=all(digest(Path(p))==h for p,h in before.items())
    result=json.loads((out/'worker-result.json').read_text()) if completed.returncode==0 and (out/'worker-result.json').exists() else dict(passed=False,replay_failed=True,subprocess_returncode=completed.returncode,error='Frozen replay failed; inspect replay.log',experimental_accuracy_qualified=False)
    result.update(producer_input_sha256=before,producer_inputs_unchanged_during_replay=unchanged,audit_source_sha256=digest(out/'executed-audit.py'),acceptance_source_sha256=digest(out/'audit-acceptance-core.py'),subprocess_command=command)
    if not unchanged:result['passed']=False
    (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    manifest={str(p.relative_to(out)):digest(p) for p in out.iterdir() if p.is_file()};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','frozen_imports','producer_input_sha256','recorded_final_history')},indent=2));return 0 if result.get('passed',False) else 2

if __name__=='__main__':raise SystemExit(main())
