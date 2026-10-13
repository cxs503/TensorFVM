#!/usr/bin/env python3
"""Exercise native hybrid SST startup on an actual no-slip SUBOFF mesh.

A completed startup step is not a converged RANS or physical accuracy result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_hybrid_rans3d import NativeHybridRans3D
from tensorfvm.native_rans3d import true_triangle_wall_distance
from tensorfvm.native_sst3d import inflow_from_intensity,wall_omega,VARIANT
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--laminar-run',type=Path,required=True)
    p.add_argument('--restart-run',type=Path,help='source-verified accepted SST iterate on the same mesh and conditions')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--momentum-method',choices=['newton','simple','simplec','coupled'],default='newton')
    p.add_argument('--momentum-coupling',choices=['converged','sweeps'],default='converged')
    p.add_argument('--momentum-sweeps',type=int,default=1)
    p.add_argument('--momentum-relaxation',type=float,default=.5)
    p.add_argument('--face-initialization',choices=['diffusion_equilibrium','cell_average'],default='cell_average')
    p.add_argument('--scalar-correction',choices=['none','nonlinear_mmp'],default='nonlinear_mmp')
    p.add_argument('--scalar-work-balance-policy',choices=['absolute','relative'],default='absolute')
    a=p.parse_args()
    if a.steps<1 or not np.isfinite(a.dt) or a.dt<=0:p.error('positive step count and dt required')
    if a.momentum_coupling=='sweeps' and a.momentum_method not in ('simple','simplec'):
        p.error('sweeps require SIMPLE or SIMPLEC')
    if a.momentum_sweeps<1 or not np.isfinite(a.momentum_relaxation) or not 0<a.momentum_relaxation<=1:
        p.error('positive momentum sweep count and relaxation in (0,1] required')
    source=a.laminar_run;old=json.loads((source/'settings.json').read_text());old_summary=json.loads((source/'summary.json').read_text())
    if old.get('backend')!='native_hybrid_cell_and_shared_face_velocity' or old.get('wall_friction') is not None:
        raise ValueError('native no-slip laminar numerical source required')
    if not old_summary['numerically_converged'] or not old_summary['source_integrity_verified']:
        raise ValueError('converged source-verified laminar seed required')
    if digest(source/'fields.npz')!=old_summary['fields_sha256'] or digest(source/'input.msh')!=old['mesh_sha256']:
        raise ValueError('initial fields/mesh changed')
    for name,sha in json.loads((source/'sources.json').read_text()).items():
        if digest(source/'source-snapshot'/name)!=sha:raise ValueError('initial numerical source changed')
    restart_bindings={};previous_steps=0
    if a.restart_run is not None:
        root=a.restart_run;rs=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
        if not summary['initialized'] or not summary['source_integrity_verified']:
            raise ValueError('source-verified initialized SST restart required')
        if (rs['backend']!=old['backend'] or rs['Re_L']!=old['Re_L'] or rs['nu']!=old['nu']
                or rs['variant']!=VARIANT or rs['rho']!=1. or rs['U']!=1. or rs['L']!=1.
                or rs['intensity']!=.005 or rs['length_scale_L']!=.05
                or rs['scalar_correction']!=a.scalar_correction
                or digest(root/'initial-input.msh')!=old['mesh_sha256']):
            raise ValueError('restart mesh, model or physical settings mismatch')
        if digest(root/'fields.npz')!=summary['fields_sha256']:raise ValueError('restart fields changed')
        for name,sha in json.loads((root/'sources.json').read_text()).items():
            if digest(root/'source-snapshot'/name)!=sha:raise ValueError('restart numerical source changed')
        previous_steps=summary.get('total_completed_steps',summary['completed_steps'])
        restart_bindings={name:digest(root/name) for name in ('fields.npz','settings.json','summary.json','sources.json')}
    evidence=NativeRunEvidence(a.output,__file__);out=a.output;start=time.perf_counter();binding={}
    for name in ('fields.npz','settings.json','summary.json','input.msh'):
        shutil.copy2(source/name,out/('initial-'+name));binding[name]=digest(out/('initial-'+name))
    if a.restart_run is not None:
        for name in restart_bindings:shutil.copy2(a.restart_run/name,out/('restart-'+name))
        shutil.copytree(a.restart_run/'source-snapshot',out/'restart-source-snapshot')
    nu=old['nu'];k0,w0=inflow_from_intensity(1.,.005,.05)
    write_json(out/'settings.json',dict(backend='native_hybrid_cell_and_shared_face_velocity',kind='actual SUBOFF SST startup diagnostic',
        Re_L=old['Re_L'],nu=nu,rho=1.,U=1.,L=1.,variant=VARIANT,steps=a.steps,dt=a.dt,relaxation=.7,tolerance=1e-8,
        k_inflow=float(k0),omega_inflow=float(w0),intensity=.005,length_scale_L=.05,
        inlet_provenance='explicit modeling assumptions, not measured experimental inflow',
        wall_treatment='resolved; k=0 and omega=60*nu/(beta1*d1^2); no wall function',
        trip='no geometric trip or transition model',initial_laminar_run=str(source),initial_bindings=binding,
        face_initialization='restored_accepted_fields' if a.restart_run is not None else a.face_initialization,
        requested_face_initialization=a.face_initialization,scalar_correction=a.scalar_correction,scalar_work_balance_policy=a.scalar_work_balance_policy,
        momentum_method=a.momentum_method,momentum_coupling=a.momentum_coupling,
        momentum_sweeps=a.momentum_sweeps,momentum_relaxation=a.momentum_relaxation,
        restart_run=str(a.restart_run) if a.restart_run is not None else None,
        restart_bindings=restart_bindings,completed_steps_before_restart=previous_steps,
        physical_suboff_benchmark_passed=False))
    m=NativeMesh3D.from_gmsh(out/'initial-input.msh',triangulate_faces=True)
    b=m.neighbor<0;wall=m.patch_labels=='wall';fixed=b&(m.patch_labels!='outlet')
    if set(m.patch_labels[b])!={'wall','inlet','outlet','farfield'}:raise ValueError('invalid boundary patches')
    velocity=np.tile([1.,0.,0.],(len(b),1));velocity[wall]=0.
    flow=NativeHybridNavierStokes3D(m,fixed,velocity,viscosity=nu,convection_order=2,backflow_velocity=[1,0,0])
    with np.load(out/'initial-fields.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['owner'],m.owner);np.testing.assert_array_equal(f['neighbor'],m.neighbor)
        np.testing.assert_allclose(f['face_velocity'][fixed],velocity[fixed],atol=1e-14,rtol=0)
        flow.state=np.r_[f['velocity'].ravel(),f['face_velocity'].ravel()]
        flow.p=f['pressure'].copy()+2/3*k0
    y=true_triangle_wall_distance(m);kb=np.full(len(b),k0);wb=np.full(len(b),w0)
    kb[wall]=0.;wb[wall]=wall_omega(nu,y[m.owner[wall]])
    s=NativeHybridRans3D(flow,y,nu,k_dirichlet_mask=fixed,omega_dirichlet_mask=fixed,
        k_boundary=kb,omega_boundary=wb,k_inflow=k0,omega_inflow=w0,scalar_correction=a.scalar_correction,scalar_work_balance_policy=a.scalar_work_balance_policy)
    failure=None;completed=False;result=None
    try:
        if a.restart_run is None:s.initialize(k0,w0,face_initialization=a.face_initialization)
        else:
            with np.load(out/'restart-fields.npz',allow_pickle=False) as f:
                for key,value in (('owner',m.owner),('neighbor',m.neighbor),('vertices',m.vertices),
                    ('area_vectors',m.area_vectors),('k_boundary',kb),('omega_boundary',wb),('wall_distance',y)):
                    np.testing.assert_array_equal(f[key],value)
                s.restore_accepted_fields(f)
        write_json(out/'initialization.json',dict(**s.initialization,k_face_min=float(s.k_faces.min()),omega_face_min=float(s.omega_faces.min())))
        np.savez_compressed(out/'initial-turbulence.npz',k=s.k,omega=s.omega,k_faces=s.k_faces,omega_faces=s.omega_faces)
        print(dict(stage='SST_STARTUP',cells=flow.N,faces=flow.F,face_initialization=s.initialization['face_initialization']),flush=True)
        def progress(row):write_json(out/'progress.json',row);print(row,flush=True)
        result=s.solve(max_iterations=a.steps,dt=a.dt,relaxation=.7,tolerance=1e-8,
            momentum_method=a.momentum_method,momentum_coupling=a.momentum_coupling,
            momentum_sweeps=a.momentum_sweeps,momentum_relaxation=a.momentum_relaxation,callback=progress)
        completed=True
    except (ValueError,RuntimeError) as exc:
        failure=f'{type(exc).__name__}: {exc}';s.converged=False
        if hasattr(exc,'scalar_result'):
            raw={k:v for k,v in exc.scalar_result.items() if isinstance(v,np.ndarray)}
            np.savez_compressed(out/'rejected-scalar-candidate.npz',**raw)
            def plain(value):
                if isinstance(value,np.ndarray):return None
                if isinstance(value,dict):return {k:plain(v) for k,v in value.items() if not isinstance(v,np.ndarray)}
                if isinstance(value,(list,tuple)):return [plain(v) for v in value]
                if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
                if isinstance(value,np.integer):return int(value)
                if isinstance(value,np.bool_):return bool(value)
                return value
            write_json(out/'rejected-scalar-candidate.json',dict(scalar=getattr(exc,'scalar_name',None),failure=failure,
                diagnostics=plain(exc.scalar_result),
                fields_sha256=digest(out/'rejected-scalar-candidate.npz'),changes_accepted_turbulence=False))
        print(dict(failure=failure),flush=True)
    fields=dict(vertices=m.vertices,face_vertices=np.asarray(m.face_vertices),cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,
        owner=m.owner,neighbor=m.neighbor,area_vectors=m.area_vectors,face_centers=m.face_centers,
        patch_labels=m.patch_labels,wall_distance=y,k_boundary=kb,omega_boundary=wb,
        velocity=flow.state[:3*flow.N].reshape(-1,3),face_velocity=flow.state[3*flow.N:].reshape(-1,3),modified_pressure=flow.p)
    if s.initialized:
        for key in ('k','omega','k_faces','omega_faces'):fields[key]=getattr(s,key)
        fields['physical_pressure']=flow.p-2/3*s.k
        fields['nu_t']=s._closure()['nu_t']
    if result is not None:
        for key in ('flux','total_traction_integrated','momentum_flux_integrated'):
            fields[key]=result[key]
        for name in ('k','omega'):
            transport=result['current_rans_residual']['transport'][name]
            for key in ('free_residual','total_flux','local_shared_flux_balance'):
                fields[name+'_'+key]=transport[key]
    np.savez_compressed(out/'fields.npz',**fields)
    write_json(out/'history.json',s.history)
    write_json(out/'momentum-history.json',getattr(flow,'newton_history',[]) if a.momentum_method=='newton' else flow.history)
    write_json(out/'momentum-outer-history.json',flow.outer_history)
    evidence.verify_sources()
    summary=dict(initialized=s.initialized,startup_steps_completed=completed,completed_steps=len(s.history),failure=failure,
        completed_steps_before_restart=previous_steps,total_completed_steps=previous_steps+len(s.history),
        numerically_converged=False,solver_converged=bool(completed and s.converged),
        qualification='startup diagnostic only; complete independent RANS force/work/transport audit not performed',physical_suboff_benchmark_passed=False,
        cells=flow.N,faces=flow.F,Re_L=old['Re_L'],wall_function_verified=False,
        source_integrity_verified=True,fields_sha256=digest(out/'fields.npz'),runtime_seconds=time.perf_counter()-start)
    write_json(out/'summary.json',summary);print(summary,flush=True)
    raise SystemExit(0 if completed else 2)


if __name__=='__main__':main()
