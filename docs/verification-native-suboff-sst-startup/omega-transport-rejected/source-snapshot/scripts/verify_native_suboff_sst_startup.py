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
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--face-initialization',choices=['diffusion_equilibrium','cell_average'],default='cell_average')
    p.add_argument('--scalar-correction',choices=['none','nonlinear_mmp'],default='nonlinear_mmp')
    a=p.parse_args()
    if a.steps<1 or not np.isfinite(a.dt) or a.dt<=0:p.error('positive step count and dt required')
    source=a.laminar_run;old=json.loads((source/'settings.json').read_text());old_summary=json.loads((source/'summary.json').read_text())
    if old.get('backend')!='native_hybrid_cell_and_shared_face_velocity' or old.get('wall_friction') is not None:
        raise ValueError('native no-slip laminar numerical source required')
    if not old_summary['numerically_converged'] or not old_summary['source_integrity_verified']:
        raise ValueError('converged source-verified laminar seed required')
    if digest(source/'fields.npz')!=old_summary['fields_sha256'] or digest(source/'input.msh')!=old['mesh_sha256']:
        raise ValueError('initial fields/mesh changed')
    for name,sha in json.loads((source/'sources.json').read_text()).items():
        if digest(source/'source-snapshot'/name)!=sha:raise ValueError('initial numerical source changed')
    evidence=NativeRunEvidence(a.output,__file__);out=a.output;start=time.perf_counter();binding={}
    for name in ('fields.npz','settings.json','summary.json','input.msh'):
        shutil.copy2(source/name,out/('initial-'+name));binding[name]=digest(out/('initial-'+name))
    nu=old['nu'];k0,w0=inflow_from_intensity(1.,.005,.05)
    write_json(out/'settings.json',dict(backend='native_hybrid_cell_and_shared_face_velocity',kind='actual SUBOFF SST startup diagnostic',
        Re_L=old['Re_L'],nu=nu,rho=1.,U=1.,L=1.,variant=VARIANT,steps=a.steps,dt=a.dt,relaxation=.7,tolerance=1e-8,
        k_inflow=float(k0),omega_inflow=float(w0),intensity=.005,length_scale_L=.05,
        inlet_provenance='explicit modeling assumptions, not measured experimental inflow',
        wall_treatment='resolved; k=0 and omega=60*nu/(beta1*d1^2); no wall function',
        trip='no geometric trip or transition model',initial_laminar_run=str(source),initial_bindings=binding,
        face_initialization=a.face_initialization,scalar_correction=a.scalar_correction,
        momentum_method='native coupled Newton',physical_suboff_benchmark_passed=False))
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
        k_boundary=kb,omega_boundary=wb,k_inflow=k0,omega_inflow=w0,scalar_correction=a.scalar_correction)
    failure=None;completed=False
    try:
        s.initialize(k0,w0,face_initialization=a.face_initialization)
        write_json(out/'initialization.json',dict(**s.initialization,k_face_min=float(s.k_faces.min()),omega_face_min=float(s.omega_faces.min())))
        np.savez_compressed(out/'initial-turbulence.npz',k=s.k,omega=s.omega,k_faces=s.k_faces,omega_faces=s.omega_faces)
        print(dict(stage='SST_STARTUP',cells=flow.N,faces=flow.F,face_initialization=a.face_initialization),flush=True)
        def progress(row):write_json(out/'progress.json',row);print(row,flush=True)
        result=s.solve(max_iterations=a.steps,dt=a.dt,relaxation=.7,tolerance=1e-8,momentum_method='newton',callback=progress)
        completed=True
    except (ValueError,RuntimeError) as exc:
        failure=f'{type(exc).__name__}: {exc}';s.converged=False
        if hasattr(exc,'scalar_result'):
            raw={k:v for k,v in exc.scalar_result.items() if isinstance(v,np.ndarray)}
            np.savez_compressed(out/'rejected-scalar-candidate.npz',**raw)
            write_json(out/'rejected-scalar-candidate.json',dict(scalar=getattr(exc,'scalar_name',None),failure=failure,
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
    np.savez_compressed(out/'fields.npz',**fields)
    write_json(out/'history.json',s.history);write_json(out/'momentum-history.json',flow.newton_history)
    evidence.verify_sources()
    summary=dict(initialized=s.initialized,startup_steps_completed=completed,completed_steps=len(s.history),failure=failure,
        numerically_converged=bool(completed and s.converged),physical_suboff_benchmark_passed=False,
        cells=flow.N,faces=flow.F,Re_L=old['Re_L'],wall_function_verified=False,
        source_integrity_verified=True,fields_sha256=digest(out/'fields.npz'),runtime_seconds=time.perf_counter()-start)
    write_json(out/'summary.json',summary);print(summary,flush=True)
    raise SystemExit(0 if completed else 2)


if __name__=='__main__':main()
