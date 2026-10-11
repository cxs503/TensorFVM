#!/usr/bin/env python3
"""Prepare an actual unstructured SUBOFF body-fitted mesh and geometry report."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorfvm.suboff_geometry import radius,SOURCE_SHA256,LENGTH_M,RADIUS_OVER_LENGTH
from tensorfvm.suboff_mesh import generate_suboff_mesh
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--wall-size',type=float,default=.035);p.add_argument('--far-size',type=float,default=.35);a=p.parse_args()
if a.output.exists() and any(a.output.iterdir()):
    raise SystemExit('Output directory must be empty to preserve source-bound evidence')
a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[1]
source_dir=a.output/'source-snapshot';source_dir.mkdir()
for rel in ('scripts/prepare_suboff_mesh.py','src/tensorfvm/suboff_geometry.py','src/tensorfvm/suboff_mesh.py','src/tensorfvm/native_mesh3d.py'):
    target=source_dir/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/rel).read_bytes())
s=generate_suboff_mesh(a.output/'suboff.msh',wall_size=a.wall_size,far_size=a.far_size)
s.update({'source_geometry_sha256':SOURCE_SHA256,'length_m':LENGTH_M,'diameter_L':2*RADIUS_OVER_LENGTH,'domain_L':[[-1,3],[-1,1],[-1,1]]})
import gmsh
from types import SimpleNamespace
gmsh.initialize();gmsh.option.setNumber('General.Terminal',0);gmsh.open(str(a.output/'suboff.msh'))
node_tags,coords,_=gmsh.model.mesh.getNodes();points=np.asarray(coords).reshape(-1,3)
index={int(t):i for i,t in enumerate(node_tags)}
def connectivity(raw,width):
    return np.asarray([index[int(t)] for t in raw],dtype=int).reshape(-1,width)
wall_parts=[]
for surf in gmsh.model.getEntitiesForPhysicalGroup(2,1):
    _,_,nodes=gmsh.model.mesh.getElements(2,int(surf))
    wall_parts.extend(connectivity(raw,3) for raw in nodes)
wall=np.concatenate(wall_parts)
_,_,raw=gmsh.model.mesh.getElements(3)
tetra=np.concatenate([connectivity(n,4) for n in raw])
gmsh.finalize()
m=SimpleNamespace(points=points,cells_dict={'tetra':tetra})
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig=plt.figure(figsize=(13,7));ax=fig.add_subplot(211,projection='3d')
ax.add_collection3d(Poly3DCollection(m.points[wall],facecolor='lightblue',edgecolor='black',linewidth=.15));ax.set_xlim(0,1);ax.set_ylim(-.07,.07);ax.set_zlim(-.07,.07);ax.set_box_aspect((5,1,1));ax.set_xlabel('x/L');ax.set_ylabel('y/L');ax.set_zlabel('z/L');ax.set_title('Actual Gmsh body surface triangles')
ax=fig.add_subplot(212)
x=np.linspace(0,1,2001);ax.plot(x,radius(x),'k-',label='TensorLBM mother polynomial');ax.plot(x,-radius(x),'k-')
segments=[]
for cell in m.cells_dict['tetra']:
    v=m.points[cell]
    if v[:,2].min()<=0<=v[:,2].max() and v[:,0].max()>-.15 and v[:,0].min()<1.3 and np.min(np.linalg.norm(v[:,1:],axis=1))<.15:
        for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            segments.append(v[[i,j],:2])
from matplotlib.collections import LineCollection
ax.add_collection(LineCollection(segments,colors='grey',linewidths=.15))
ax.set_xlim(-.15,1.3);ax.set_ylim(-.2,.2);ax.set_aspect('equal');ax.set_xlabel('x/L');ax.set_ylabel('y/L');ax.set_title('Edges of tetrahedra intersecting z=0; projection of those edges');ax.legend(loc='upper right')
fig.tight_layout();fig.savefig(a.output/'mesh.png',dpi=180);fig.savefig(a.output/'mesh.svg');fig.savefig(a.output/'report.pdf');plt.close(fig)
try:
    from tensorfvm.native_mesh3d import NativeMesh3D
    native=NativeMesh3D.from_gmsh(a.output/'suboff.msh')
    s['native_mesh_audit']=native.audit()
except ImportError:
    s['native_mesh_audit']='Generic importer not available at launch'
(a.output/'summary.json').write_text(json.dumps(s,indent=2)+'\n')

(a.output/'report.md').write_text(f'''# SUBOFF AFF-1 geometry and mesh preparation

The mother hull is ported from TensorLBM's `suboff_cad.py` (SHA256 `{SOURCE_SHA256}`), preserving its four polynomial segments and original feet dimensions. The source attributes these to Groves, Huang and Chang (1989), DTRC/SHD-1298-01; this stage does not independently certify the original offset tables. Hull length {LENGTH_M:.8f} m, diameter {2*RADIUS_OVER_LENGTH:.8f} L. Axis x, bow x=0, stern x=1.

The OCC segmented spline meridian is revolved, then subtracted from a [-1,3]×[-1,1]×[-1,1] L box. Gmsh {s['gmsh_version']} generated {s['cells']} actual tetrahedra and {s['nodes']} nodes, with wall/inlet/outlet/farfield physical patches. Maximum mesh-node radial discrepancy from the mother polynomial: {s['max_wall_radial_error_L']:.6g} L. This is an interpolation diagnostic, not a global Hausdorff certification. The outer boundary is deliberately a geometry development domain, not a domain independence result.

![Actual surface and volume-grid evidence](mesh.png)

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/prepare_suboff_mesh.py --output NEW_DIRECTORY --wall-size {a.wall_size} --far-size {a.far_size}
```

## Acceptance scope

Body-fitted tetrahedral geometry preparation only. No boundary layer prism cells, no flow solution, no y+ estimate, no drag/Cp validation, no mesh convergence, and no <3% claim. AFF-8 appendages are not included. Turbulent experimental-Reynolds computation must first add wall resolution and verified 3D pressure/velocity coupling. `suboff.msh` carries native topology and physical patch IDs for the generic 3D importer.
''')
(a.output/'manifest.json').write_text(json.dumps({str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file() and f.name!='manifest.json'},indent=2)+'\n')
print(json.dumps(s,indent=2))
