#!/usr/bin/env python3
"""Diagnostic precision sensitivity of the SAME SST closure at saved SUBOFF fields.

No solve is performed. The production closure is unchanged: a separate function
object executes its exact code with an extended-precision input validator.
The current production discretization is not qualified by these diagnostics.
"""
import argparse,copy,hashlib,json,types
from pathlib import Path
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_sst3d import sst2003_closure
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();root=a.run
    cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if not summary['initialized'] or not summary['source_integrity_verified']:
        raise ValueError('verified initialized producer required')
    if digest(root/'fields.npz')!=summary['fields_sha256']:raise ValueError('fields changed')
    for name,sha in json.loads((root/'sources.json').read_text()).items():
        if digest(root/'source-snapshot'/name)!=sha:raise ValueError('producer source changed')
    for name,sha in cfg['initial_bindings'].items():
        if digest(root/('initial-'+name))!=sha:raise ValueError('initial binding changed')
    evidence=NativeRunEvidence(a.output,__file__);out=a.output
    write_json(out/'settings.json',dict(scope=__doc__,run=str(root),bindings={name:digest(root/name)
        for name in ('settings.json','summary.json','fields.npz','sources.json','initial-input.msh')},
        precision_bits=int(np.finfo(np.longdouble).nmant+1),physical_suboff_benchmark_passed=False))
    mesh=NativeMesh3D.from_gmsh(root/'initial-input.msh',triangulate_faces=True)
    f=np.load(root/'fields.npz',allow_pickle=False)
    np.testing.assert_array_equal(mesh.owner,f['owner']);np.testing.assert_array_equal(mesh.neighbor,f['neighbor'])
    fixed=(mesh.neighbor<0)&(mesh.patch_labels!='outlet')
    base=NativeHybridScalar3D(mesh,dirichlet_mask=fixed)
    face=f['face_velocity'];q=np.sum(face*mesh.area_vectors,axis=1)
    gradient=np.stack([base.D@(face*mesh.area_vectors[:,j,None])/mesh.cell_volumes[:,None] for j in range(3)],axis=2)
    def extended_gradient(value):
        return np.column_stack([base.D@(value.astype(np.longdouble)*mesh.area_vectors[:,j])
            for j in range(3)])/mesh.cell_volumes[:,None]
    def extended_finite(name,value):
        value=np.asarray(value,dtype=np.longdouble)
        if not np.isfinite(value).all():raise ValueError(name+' must be finite')
        return value
    # Isolated globals: never monkeypatch the production module or function.
    namespace=dict(sst2003_closure.__globals__);namespace['_finite']=extended_finite
    extended=types.FunctionType(sst2003_closure.__code__,namespace,'diagnostic_extended_sst',sst2003_closure.__defaults__)
    extended.__kwdefaults__=dict(sst2003_closure.__kwdefaults__)
    arguments=(f['k'],f['omega'],f['wall_distance'],cfg['nu'],gradient)
    ordinary=sst2003_closure(*arguments,base.gradient(f['k_faces']),base.gradient(f['omega_faces']))
    precise=extended(*arguments,extended_gradient(f['k_faces']),extended_gradient(f['omega_faces']))
    report={};raw={}
    for name in ('k','omega'):
        state=np.r_[f[name],f[name+'_faces']]
        results={}
        for mode,closure in [('saved_state_float64_closure',ordinary),('saved_state_extended_closure',precise)]:
            op=NativeHybridScalar3D(mesh,cfg['nu']+closure['sigma_'+name]*closure['nu_t'],dirichlet_mask=fixed,
                correction=cfg['scalar_correction'],work_balance_policy=cfg['scalar_work_balance_policy'])
            linear=copy.copy(op);linear.mmp=None
            if op.mmp is not None:linear.diffusion,_=op.mmp.operator(state)
            result=linear._solution_result(state,q,closure[name+'_source'],np.zeros(base.N),np.zeros(base.N),0.,cfg[name+'_inflow'],0.)
            results[mode]={key:result[key] for key in ('cell_balance_max','interface_flux_defect_max','global_balance',
                'work_balance','relative_work_balance')}
            results[mode]['diagnostic_ledger_passed']=op.ledger_acceptable(result)
            raw[name+'_'+mode+'_source']=closure[name+'_source']
            raw[name+'_'+mode+'_cell_balance']=result['cell_balance']
            raw[name+'_'+mode+'_viscosity']=op.diffusion.mu
        results['max_integrated_source_change']=float(np.max(abs(mesh.cell_volumes*(precise[name+'_source']-ordinary[name+'_source']))))
        results['max_assembled_viscosity_change']=float(np.max(abs(raw[name+'_saved_state_extended_closure_viscosity']-
            raw[name+'_saved_state_float64_closure_viscosity'])))
        report[name]=results
    np.savez_compressed(out/'precision-ledgers.npz',**raw);evidence.verify_sources()
    write_json(out/'audit.json',dict(source_integrity_verified=True,producer_sources_verified=True,
        raw_sha256=digest(out/'precision-ledgers.npz'),scalar=report,production_solver_changed=False,
        numerical_qualification=False,physical_suboff_benchmark_passed=False))
    print(json.loads((out/'audit.json').read_text()),flush=True)


if __name__=='__main__':main()
