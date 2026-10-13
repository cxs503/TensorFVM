#!/usr/bin/env python3
"""Resume an archived zero-source scalar control on its original equations."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_scalar_mmp3d import ScalarMMPConvergenceError
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;evidence=NativeRunEvidence(out,__file__)
    prior=json.loads((a.run/'summary.json').read_text())
    if not prior['source_integrity_verified']:raise ValueError('unverified initial source')
    for name,digest in json.loads((a.run/'sources.json').read_text()).items():
        if sha(a.run/'source-snapshot'/name)!=digest:raise ValueError('initial source hash mismatch')
    row=next(v for v in json.loads((a.run/'cases.json').read_text()) if v['case']==a.case)
    if row['correction']!='nonlinear_mmp' or not a.case.startswith('suboff_layer_affine-r'):
        raise ValueError('expected recorded affine SUBOFF scalar control')
    path=a.run/(a.case+'-fields.npz')
    if sha(path)!=row['fields_sha256']:raise ValueError('initial field hash mismatch')
    shutil.copy2(path,out/'initial-fields.npz')
    shutil.copy2(a.run/'sources.json',out/'initial-sources.json')
    shutil.copytree(a.run/'source-snapshot',out/'initial-source-snapshot')
    write_json(out/'initial-case.json',row)
    with np.load(path,allow_pickle=False) as f:
        for key in ('storage','destruction','production','advective_flux'):
            if np.any(f[key]!=0):raise ValueError('only steady zero-source zero-advection restart supported')
        geom={k:f[k].copy() for k in ('vertices','cell_vertices','cell_offsets','face_vertices','owner','neighbor',
            'cell_centers','cell_volumes','face_centers','area_vectors','boundary_values')}
        cells=[geom['cell_vertices'][i:j] for i,j in zip(geom['cell_offsets'][:-1],geom['cell_offsets'][1:])]
        kinds=[{4:'tetra',6:'wedge',8:'hexahedron'}[len(c)] for c in cells]
        m=NativeMesh3D.from_cells(geom['vertices'],cells,kinds,triangulate_faces=True)
        for key in ('owner','neighbor','face_vertices'):np.testing.assert_array_equal(getattr(m,key),geom[key])
        for key in ('cell_centers','cell_volumes','face_centers','area_vectors'):np.testing.assert_allclose(getattr(m,key),geom[key],rtol=0,atol=1e-14)
        state=np.r_[f['cell_values'],f['face_values']]
    op=NativeHybridScalar3D(m,correction='nonlinear_mmp');tick=time.perf_counter();failure=None
    print(dict(stage='SOLVE_ORIGINAL_EQUATIONS',cells=op.N,faces=op.F,previous_converged=row['converged']),flush=True)
    try:r=op.solve(boundary_values=geom['boundary_values'],nonlinear_initial_state=state)
    except ScalarMMPConvergenceError as error:r=error.scalar_result;failure=r['failure']
    name=a.case+'-restart';path=out/(name+'-fields.npz')
    np.savez_compressed(path,**geom,**{k:v for k,v in r.items() if k!='nonlinear_history'})
    write_json(out/(name+'-history.json'),r['nonlinear_history'])
    row.update(case=name,converged=failure is None,failure=failure,runtime_seconds=time.perf_counter()-tick,
        iterations=len(r['nonlinear_history']),fields_sha256=sha(path),final_smoothing=r['final_smoothing'],
        cell_min=r['cell_min'],face_min=r['face_min'],cell_max=float(r['cell_values'].max()),face_max=float(r['face_values'].max()),
        affine_max_absolute_error=float(np.max(np.abs(r['cell_values']-1-m.cell_centers@np.array([.1,-.2,.3])))))
    for key in ('current_nonlinear_residual','cell_balance_max','interface_flux_defect_max','global_balance','work_balance',
                'solution_precision_bits','work_balance_direct_float64','work_balance_precision_bits'):row[key]=r[key]
    write_json(out/'cases.json',[row]);evidence.verify_sources()
    write_json(out/'summary.json',dict(source_integrity_verified=True,all_requested_controls_converged=failure is None,
        initial_fields_sha256=sha(out/'initial-fields.npz'),numerical_restart=True,physical_suboff_benchmark_passed=False,
        suboff_scope='scalar affine control only; no exterior flow or Cp/Cd'))
    print(row,flush=True);raise SystemExit(0 if failure is None else 2)


if __name__=='__main__':main()
