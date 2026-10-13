#!/usr/bin/env python3
"""Replay a rejected native SUBOFF k or omega solve with bound original inputs."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_sst3d import sst2003_closure,BETA_STAR
from tensorfvm.native_scalar_mmp3d import ScalarMMPConvergenceError
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items() if not isinstance(v,np.ndarray)}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--work-balance-policy',choices=['absolute','relative'],default='absolute');a=p.parse_args()
    root=a.run;summary=json.loads((root/'summary.json').read_text());settings=json.loads((root/'settings.json').read_text())
    rejected=json.loads((root/'rejected-scalar-candidate.json').read_text())
    name=rejected['scalar']
    assert name in ('k','omega') and summary['source_integrity_verified']
    assert digest(root/'fields.npz')==summary['fields_sha256']
    assert digest(root/'rejected-scalar-candidate.npz')==rejected['fields_sha256']
    for relative,sha in json.loads((root/'sources.json').read_text()).items():assert digest(root/'source-snapshot'/relative)==sha
    for relative,sha in settings['initial_bindings'].items():assert digest(root/('initial-'+relative))==sha
    evidence=NativeRunEvidence(a.output,__file__);out=a.output;start=time.perf_counter()
    write_json(out/'settings.json',dict(original_run=str(root),original_bindings={name:digest(root/name) for name in
        ('settings.json','summary.json','fields.npz','initial-input.msh','rejected-scalar-candidate.npz')},
        work_balance_policy=a.work_balance_policy,physical_suboff_benchmark_passed=False,scalar=name,scope='one original scalar subsolve; coupled state not advanced'))
    m=NativeMesh3D.from_gmsh(root/'initial-input.msh',triangulate_faces=True)
    f=np.load(root/'fields.npz');candidate=np.load(root/'rejected-scalar-candidate.npz')
    mask=(m.neighbor<0)&(m.patch_labels!='outlet');base=NativeHybridScalar3D(m,dirichlet_mask=mask)
    face=f['face_velocity'];gradient=np.stack([base.D@(face*m.area_vectors[:,j,None])/m.cell_volumes[:,None] for j in range(3)],axis=2)
    # Match the transport's explicit input conversion, including accepted
    # longdouble fields after previous relaxed SST steps.
    old_k=np.asarray(f['k'],float);old_w=np.asarray(f['omega'],float)
    closure=sst2003_closure(old_k,old_w,f['wall_distance'],settings['nu'],gradient,
        base.gradient(np.asarray(f['k_faces'],float)),base.gradient(np.asarray(f['omega_faces'],float)),rho=1.)
    coefficient=settings['nu']+closure['sigma_'+name]*closure['nu_t']
    production=(closure['production_limited'] if name=='k' else closure['omega_production']+closure['cross_diffusion_source'])
    old=old_k if name=='k' else old_w
    destruction=BETA_STAR*old_w if name=='k' else closure['beta']*old_w
    source=np.maximum(production,0.);sink=destruction+np.divide(-np.minimum(production,0.),old,out=np.zeros_like(old),where=old>0)
    q=np.sum(face*m.area_vectors,axis=1);dt=settings['dt'];wb=f[name+'_boundary'];inflow=settings[name+'_inflow']
    np.testing.assert_array_equal(coefficient,candidate['diffusivity'])
    np.testing.assert_array_equal(m.cell_volumes*source,candidate['production'])
    np.testing.assert_allclose(m.cell_volumes*sink*candidate['cell_values'],candidate['destruction'],rtol=1e-15,atol=0)
    np.testing.assert_allclose(m.cell_volumes/dt*(candidate['cell_values']-old),candidate['storage'],rtol=1e-15,atol=0)
    write_json(out/'input-audit.json',dict(diffusivity_and_production_bitwise_equal=True,destruction_and_storage_relative_difference_below=1e-15,
        cell_count=len(m.cells),face_count=len(m.owner),same_physical_equation_inputs=True))
    np.savez_compressed(out/'inputs.npz',coefficient=coefficient,source=source,sink=sink,old=old,volume_flux=q,boundary_values=wb,
        cell_volumes=m.cell_volumes,owner=m.owner,neighbor=m.neighbor,dirichlet=mask)
    scalar=NativeHybridScalar3D(m,coefficient,dirichlet_mask=mask,correction='nonlinear_mmp',work_balance_policy=a.work_balance_policy)
    failure=None
    print('SOLVE_ORIGINAL_'+name.upper(),flush=True)
    try:r=scalar.solve(q,source=source,sink=sink,boundary_values=wb,inflow_values=inflow,old_values=old,dt=dt)
    except ScalarMMPConvergenceError as exc:r=exc.scalar_result;failure=str(exc)
    np.savez_compressed(out/'candidate.npz',**{key:value for key,value in r.items() if isinstance(value,np.ndarray)})
    write_json(out/'diagnostics.json',plain(r));evidence.verify_sources()
    summary=dict(scalar_numerically_converged=failure is None,failure=failure,source_integrity_verified=True,
        candidate_sha256=digest(out/'candidate.npz'),inputs_sha256=digest(out/'inputs.npz'),runtime_seconds=time.perf_counter()-start,
        cell_balance_max=r['cell_balance_max'],interface_flux_defect_max=r['interface_flux_defect_max'],
        global_balance=r['global_balance'],work_balance=r['work_balance'],work_balance_scale=r['work_balance_scale'],
        relative_work_balance=r['relative_work_balance'],absolute_work_gate_passed=r['absolute_work_gate_passed'],work_balance_policy=a.work_balance_policy,
        physical_suboff_benchmark_passed=False,full_rans_step_completed=False)
    write_json(out/'summary.json',summary);print(summary,flush=True)
    raise SystemExit(0 if failure is None else 2)


if __name__=='__main__':main()
