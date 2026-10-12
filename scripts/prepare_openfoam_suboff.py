#!/usr/bin/env python3
"""Write a Foundation 12 SUBOFF case retaining the native diagnostic mesh."""
import argparse,hashlib,json
from pathlib import Path
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.openfoam_case import write_rans_case
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mesh',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--reynolds',type=float,default=12e6)
p.add_argument('--iterations',type=int,default=300)
p.add_argument('--convection',choices=['upwind','linearUpwind'],default='linearUpwind')
a=p.parse_args()
m=NativeMesh3D.from_gmsh(a.mesh,triangulate_faces=True)
write_rans_case(m,a.output,reynolds=a.reynolds,iterations=a.iterations,convection=a.convection)
(a.output/'provenance.json').write_text(json.dumps(dict(kind='OpenFOAM Foundation 12 AFF-1 same-mesh diagnostic',
    mesh=str(a.mesh),mesh_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),cells=len(m.cells),faces=len(m.owner),
    reynolds=a.reynolds,iterations=a.iterations,convection=a.convection,
    geometry='AFF-1 bare hull, zero incidence, fully turbulent, no explicit trip',
    turbulence_inlet=dict(intensity=.005,length_scale_L=.05,source='project assumption'),
    literature_exact_reproduction=False,physical_benchmark_passed=False),indent=2)+'\n')
print(a.output)
