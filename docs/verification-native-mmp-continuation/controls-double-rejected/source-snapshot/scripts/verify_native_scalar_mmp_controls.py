#!/usr/bin/env python3
"""Positive harmonic and actual SUBOFF-layer scalar controls, not submarine flow."""
import argparse
import hashlib
import numpy as np
from pathlib import Path
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.suboff_hybrid_mesh import structured_layers
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suboff-refinements',type=int,nargs='+',default=[1])
    args=p.parse_args();out=args.output
    if any(v<1 for v in args.suboff_refinements):p.error('positive refinement multipliers required')
    evidence=NativeRunEvidence(out,__file__);rows=[]
    for kind,refinement in [('harmonic',1)]+[(f'suboff_layer_affine-r{r}',r) for r in args.suboff_refinements]:
        if kind=='harmonic':
            m=pressure_mesh(2);bd=np.zeros(len(m.owner));bd[[38,52,60,90]]=1.;exact=None
        else:
            points,cells,types,_,_,_=structured_layers(axial=12*refinement,azimuthal=8*refinement,layers=2*refinement,
                first_height=.0022/np.sum((1.2**(1/refinement))**np.arange(2*refinement)),growth=1.2**(1/refinement))
            m=NativeMesh3D.from_cells(points,cells,types,triangulate_faces=True)
            g=np.array([.1,-.2,.3]);bd=1+m.face_centers@g;exact=1+m.cell_centers@g
        for correction in ('none','nonlinear_mmp'):
            op=NativeHybridScalar3D(m,correction=correction)
            r=op.solve(boundary_values=bd)
            row=dict(case=kind+'-'+correction,cells=len(m.cells),faces=len(m.owner),
                cell_min=r['cell_min'],face_min=r['face_min'],cell_max=float(r['cell_values'].max()),face_max=float(r['face_values'].max()),
                correction=correction,current_nonlinear_residual=r.get('current_nonlinear_residual'),
                iterations=len(r.get('nonlinear_history',[])),cell_balance_max=r['cell_balance_max'],
                interface_flux_defect_max=r['interface_flux_defect_max'],global_balance=r['global_balance'],work_balance=r['work_balance'])
            if exact is not None:row['affine_max_absolute_error']=float(np.max(np.abs(r['cell_values']-exact)))
            file=out/(row['case']+'-fields.npz')
            np.savez_compressed(file,vertices=m.vertices,cell_vertices=np.concatenate(m.cells),cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
                face_vertices=np.asarray(m.face_vertices),owner=m.owner,neighbor=m.neighbor,cell_centers=m.cell_centers,
                cell_volumes=m.cell_volumes,face_centers=m.face_centers,area_vectors=m.area_vectors,boundary_values=bd,
                **{k:v for k,v in r.items() if k!='nonlinear_history'})
            if 'nonlinear_history' in r:write_json(out/(row['case']+'-history.json'),r['nonlinear_history'])
            row.update(refinement=refinement,final_smoothing=r.get('final_smoothing'),solver=r.get('nonlinear_solver','linear'))
            row['fields_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();rows.append(row)
            write_json(out/'cases.json',rows);print(row,flush=True)
    evidence.verify_sources()
    write_json(out/'summary.json',dict(source_integrity_verified=True,
        corrected_harmonic_positive=rows[1]['cell_min']>=0 and rows[1]['face_min']>=0,
        physical_suboff_benchmark_passed=False,suboff_scope='scalar affine consistency on refined thin-layer meshes; no flow, no farfield, no Cp/Cd'))


if __name__=='__main__':main()
