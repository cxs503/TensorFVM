#!/usr/bin/env python3
"""Verify boundary pressure postprocessing on actual refined SUBOFF layers."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from tensorfvm.native_boundary_pressure3d import NativeBoundaryPressureTrace3D
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.suboff_hybrid_mesh import structured_layers
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    e=NativeRunEvidence(a.output,__file__);rows=[]
    write_json(a.output/'settings.json',dict(scope='pressure trace of independently sampled smooth functions; no flow or experimental Cp',
        levels=[1,2,3],axial='12*r',azimuthal='8*r',layers='2*r',total_layer_height=.0044,growth='1.2**(1/r)',
        trace='distance-weighted internal connected-cell linear least squares; max 3 rings',physical_suboff_benchmark_passed=False))
    for r in (1,2,3):
        growth=1.2**(1/r);height=.0044/np.sum(growth**np.arange(2*r))
        points,cells,kinds,wall,outer,_=structured_layers(axial=12*r,azimuthal=8*r,layers=2*r,first_height=height,growth=growth)
        m=NativeMesh3D.from_cells(points,cells,kinds,{**{tuple(f):'wall' for f in wall},**{tuple(f):'outer' for f in outer}},triangulate_faces=True)
        mask=m.patch_labels=='wall';op=NativeBoundaryPressureTrace3D(m,mask);g=np.array([.7,-1.3,2.1])
        affine=op.reconstruct(.23+m.cell_centers@g);affine_error=float(np.max(abs(affine['pressure']-(.23+m.face_centers[mask]@g))))
        def smooth(x):return 1+.2*np.sin(2*np.pi*x[:,0])+.3*x[:,1]**2-.1*x[:,2]
        values=smooth(m.cell_centers);exact=smooth(m.face_centers[mask]);trace=op.reconstruct(values);owner=values[m.owner[mask]]
        area=np.linalg.norm(m.area_vectors[mask],axis=1)
        norm=lambda error:float(np.sqrt(np.sum(area*error**2)/np.sum(area*exact**2)))
        row=dict(refinement=r,cells=len(m.cells),wall_faces=int(mask.sum()),affine_max_error=affine_error,
            trace_relative_l2=norm(trace['pressure']-exact),owner_relative_l2=norm(owner-exact),
            trace_max_error=float(np.max(abs(trace['pressure']-exact))),maximum_rings=int(trace['stencil_rings'].max()),
            minimum_singular_ratio=float(trace['singular_ratio'].min()),maximum_pressure_noise_amplification=float(trace['pressure_noise_amplification'].max()),physical_suboff_benchmark_passed=False)
        file=a.output/f'r{r}-fields.npz'
        np.savez_compressed(file,vertices=m.vertices,face_vertices=np.asarray(m.face_vertices),cell_centers=m.cell_centers,
            face_centers=m.face_centers,owner=m.owner,neighbor=m.neighbor,area_vectors=m.area_vectors,
            cell_pressure=values,reference_pressure=exact,owner_pressure=owner,**trace)
        row['fields_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();rows.append(row);write_json(a.output/'cases.json',rows);print(row,flush=True)
    e.verify_sources()
    passed=all(row['affine_max_error']<1e-10 for row in rows) and all(b['trace_relative_l2']<a['trace_relative_l2'] for a,b in zip(rows,rows[1:]))
    write_json(a.output/'summary.json',dict(source_integrity_verified=True,affine_and_refinement_passed=passed,physical_suboff_benchmark_passed=False,wall_Cp_certified=False))
    raise SystemExit(0 if passed else 2)


if __name__=='__main__':main()
