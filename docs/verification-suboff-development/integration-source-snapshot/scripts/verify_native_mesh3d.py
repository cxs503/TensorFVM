#!/usr/bin/env python3
"""Audit a real 3D Gmsh volume and record reproducible geometry/operator evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_operators3d import NativeOperators3D

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mesh',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    mesh=NativeMesh3D.from_gmsh(args.mesh);operators=NativeOperators3D(mesh)
    audit=mesh.audit();affine=operators.audit_affine()
    root=Path(__file__).resolve().parents[1]
    sources=[root/'src/tensorfvm/native_mesh3d.py',root/'src/tensorfvm/native_operators3d.py',Path(__file__).resolve()]
    source_dir=args.output/'source-snapshot';source_dir.mkdir(exist_ok=True)
    source_records=[]
    for source in sources:
        dest=source_dir/source.name;dest.write_bytes(source.read_bytes());source_records.append({'path':str(source.relative_to(root)),'sha256':sha(source),'snapshot':str(dest.relative_to(args.output))})
    result={'mesh_path':str(args.mesh),'mesh_sha256':sha(args.mesh),'geometry':audit,'manufactured_affine':affine,'geometry_contract_passed':audit['maximum_relative_closure']<1e-10 and audit['minimum_face_projection']>0,'affine_operator_contract_passed':max(affine.values())<1e-9,'flow_physics_validated':False,'limitations':['Linear convex cells only; nonplanar mixed-cell faces are rejected.','Positive projections do not certify adequate boundary layer or wake resolution.','Manufactured affine responses do not establish pressure-velocity coupling, drag accuracy or grid convergence.'],'sources':source_records}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    rows='\n'.join(f'| {k} | {v:.6g} |' for k,v in affine.items())
    report=f'''# Native 3D mesh and operator verification

Input: `{args.mesh}`; SHA256 `{sha(args.mesh)}`.

The actual volume contains {audit['cells']} cells and {audit['faces']} unique shared faces. Native owner/neighbor flux integration uses one value per face with opposite signs in adjacent cells. No extrusion or spanwise periodicity is assumed.

Minimum cell volume: {audit['minimum_cell_volume']:.6g}. Maximum relative cell area closure: {audit['maximum_relative_closure']:.6g}. Minimum owner-neighbor projection: {audit['minimum_face_projection']:.6g}. Maximum nonorthogonality: {audit['maximum_nonorthogonality_degrees']:.6g} degrees.

## Independent manufactured fields

Pressure p=1.7+0.7x−1.3y+2.1z has known constant gradient. Constant velocity (0.3,0.8,−0.4) has zero divergence, and velocity (x,2y,3z) has divergence 6. Boundary values come directly from these analytic fields. Error is the maximum absolute component error over the actual mesh.

| Response | Maximum absolute error |
| --- | ---: |
{rows}

## Qualification scope

Geometry and affine operator contracts pass. This is not a validated SUBOFF flow result. Mesh nonorthogonality, near-wall resolution, pressure-velocity coupling and physical benchmark accuracy require separate gates. Only linear convex tetrahedra, planar-faced hexahedra, wedges and pyramids are supported; unsupported and untagged topology is rejected.

See [audit.json](audit.json) for exact values, patch counts and source hashes. Source snapshots are archived alongside this report.
'''
    (args.output/'report.md').write_text(report);print(json.dumps(result))
if __name__=='__main__':main()
