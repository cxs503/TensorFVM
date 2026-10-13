#!/usr/bin/env python3
"""Reproduce and preserve unaccepted nonlinear diffusion candidates."""
import argparse
import time
from pathlib import Path
import numpy as np
from tensorfvm.native_scalar_mmp3d import ScalarMMPConvergenceError
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.suboff_hybrid_mesh import structured_layers
from tensorfvm.verification.native_hybrid_rans import coupled_shear_solver
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;evidence=NativeRunEvidence(out,__file__);rows=[]
    for name in ('suboff_layer_affine','low_re_n8'):
        if name=='suboff_layer_affine':
            points,cells,types,_,_,_=structured_layers(axial=12,azimuthal=8,layers=2,first_height=.001,growth=1.2)
            m=NativeMesh3D.from_cells(points,cells,types,triangulate_faces=True)
            bd=1+m.face_centers@np.array([.1,-.2,.3])
            op=NativeHybridScalar3D(m,correction='nonlinear_mmp',mmp_solver='picard')
            run=lambda:op.solve(boundary_values=bd)
        else:
            m,s=coupled_shear_solver(8,'low_re',triangulate_faces=True,scalar_correction='nonlinear_mmp',mmp_solver='picard')
            run=lambda:s.solve(max_iterations=120,dt=1.,tolerance=1e-8)
        tick=time.perf_counter()
        try:run()
        except ScalarMMPConvergenceError as error:
            r=error.scalar_result
            np.savez_compressed(out/(name+'-candidate.npz'),vertices=m.vertices,
                cell_vertices=np.concatenate(m.cells),cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
                face_vertices=np.asarray(m.face_vertices),owner=m.owner,neighbor=m.neighbor,
                cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,face_centers=m.face_centers,area_vectors=m.area_vectors,
                **{k:v for k,v in r.items() if k!='nonlinear_history'})
            write_json(out/(name+'-history.json'),r['nonlinear_history'])
            row=dict(case=name,cells=len(m.cells),faces=len(m.owner),scalar=error.scalar_name,rejected=True,
                reason=str(error),runtime_seconds=time.perf_counter()-tick,
                cell_min=r['cell_min'],face_min=r['face_min'],current_nonlinear_residual=r['current_nonlinear_residual'],
                change=r['nonlinear_history'][-1]['change'],iterations=len(r['nonlinear_history']),
                local_balance=r['cell_balance_max'],interface_defect=r['interface_flux_defect_max'],
                global_balance=r['global_balance'],work_balance=r['work_balance'],physical_benchmark_passed=False)
            rows.append(row);write_json(out/'cases.json',rows);print(row,flush=True)
        else:raise RuntimeError('expected convergence failure no longer reproduces; update evidence honestly')
    evidence.verify_sources()
    write_json(out/'summary.json',dict(source_integrity_verified=True,unaccepted_candidates_saved=True,
        all_requested_physics_passed=False,physical_suboff_benchmark_passed=False))


if __name__=='__main__':main()
