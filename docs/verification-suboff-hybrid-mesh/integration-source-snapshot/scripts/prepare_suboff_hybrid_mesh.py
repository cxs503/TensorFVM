#!/usr/bin/env python3
"""Generate source-bound hybrid mesh evidence, including real facet sections."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tensorfvm.suboff_hybrid_mesh import generate_suboff_hybrid_mesh
from tensorfvm.native_operators3d import NativeOperators3D
from tensorfvm.suboff_geometry import radius,LENGTH_M,SOURCE_SHA256


def section(mesh,axis,value):
    """Intersect native triangular facets with a plane, without projecting edges."""
    groups={'tetra':[],'hexahedron':[],'wedge':[]}
    other=[a for a in range(3) if a!=axis]
    for fi,face in enumerate(mesh.face_vertices):
        pts=mesh.vertices[list(face)];d=pts[:,axis]-value
        d[np.abs(d)<1e-13]=0
        if d.min()>0 or d.max()<0:continue
        intersections=[]
        for a,b in [(0,1),(1,2),(2,0)]:
            if d[a]==0:intersections.append(pts[a])
            if d[a]*d[b]<0:intersections.append(pts[a]+d[a]/(d[a]-d[b])*(pts[b]-pts[a]))
        if not intersections:continue
        unique=np.unique(np.round(np.asarray(intersections),13),axis=0)
        kind=mesh.cell_types[mesh.owner[fi]]
        if mesh.neighbor[fi]>=0 and mesh.cell_types[mesh.neighbor[fi]]!='tetra':kind=mesh.cell_types[mesh.neighbor[fi]]
        if len(unique)==2:groups[kind].append(unique[:,other])
        elif len(unique)==3:
            groups[kind].extend(unique[[a,b]][:,other] for a,b in [(0,1),(1,2),(2,0)])
    return groups


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--axial',type=int,default=32);parser.add_argument('--azimuthal',type=int,default=24)
    parser.add_argument('--layers',type=int,default=4);parser.add_argument('--first-height',type=float,default=.0004)
    parser.add_argument('--growth',type=float,default=1.25);parser.add_argument('--far-size',type=float,default=.35)
    a=parser.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):raise SystemExit('Output must be empty; evidence is never overwritten')
    out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[1]
    for rel in ['scripts/prepare_suboff_hybrid_mesh.py','src/tensorfvm/suboff_hybrid_mesh.py','src/tensorfvm/suboff_geometry.py','src/tensorfvm/native_mesh3d.py','src/tensorfvm/native_operators3d.py','tests/test_suboff_hybrid_mesh.py']:
        dst=out/'source-snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((root/rel).read_bytes())
    config=vars(a).copy();config['output']=str(out);(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    start=time.perf_counter()
    m,s=generate_suboff_hybrid_mesh(out/'suboff-hybrid.msh',axial=a.axial,azimuthal=a.azimuthal,layers=a.layers,first_height=a.first_height,growth=a.growth,far_size=a.far_size)
    s['generation_and_import_seconds']=time.perf_counter()-start
    op=NativeOperators3D(m);s['affine_audit']=op.audit_affine()
    internal=m.neighbor>=0;left=np.asarray(m.cell_types)[m.owner];right=np.asarray(m.cell_types)[m.neighbor]
    interface=internal & ((left=='tetra')!=(right=='tetra'))
    arbitrary=np.zeros(len(m.owner));arbitrary[interface]=np.sin(np.flatnonzero(interface)*.37)
    integrated=op.integrate(arbitrary);s['interface_arbitrary_flux_cancellation']=float(abs(integrated.sum()))
    wall=m.patch_labels=='wall';wall_ids=np.unique(np.concatenate([m.face_vertices[i] for i in np.flatnonzero(wall)]))
    discrepancy=np.abs(np.linalg.norm(m.vertices[wall_ids,1:],axis=1)-radius(m.vertices[wall_ids,0]))
    # The source profile has a small roundoff/decimal-coefficient tip residual;
    # both tips are explicitly closed at the axis.
    s['wall_node_radial_error_L']=float(discrepancy.max())
    s['physical_length_m']=LENGTH_M;s['source_geometry_sha256']=SOURCE_SHA256
    wall_volume=-np.einsum('ij,ij->i',m.face_centers[wall],m.area_vectors[wall]).sum()/3
    s['faceted_body_volume_L3']=float(wall_volume)
    s['domain_volume_balance_error_L3']=float(abs(m.cell_volumes.sum()+wall_volume-16))
    s['geometry_passed']=bool(s['verified_shared_interface_faces']==s['interface_triangles'] and s['mesh_audit']['maximum_relative_closure']<1e-12 and max(s['affine_audit'].values())<1e-8 and s['domain_volume_balance_error_L3']<1e-10 and s['interface_arbitrary_flux_cancellation']<1e-10)
    s['numerical_flow_qualified']=False;s['physical_qualified']=False
    np.savez_compressed(out/'native-geometry.npz',vertices=m.vertices,owner=m.owner,neighbor=m.neighbor,area_vectors=m.area_vectors,face_centers=m.face_centers,cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,patch_labels=m.patch_labels,face_vertices=np.asarray(m.face_vertices),cell_types=np.asarray(m.cell_types),interface_face_indices=np.flatnonzero(interface))
    fig=plt.figure(figsize=(14,11));ax=fig.add_subplot(311,projection='3d')
    # Reconstruct original surface quads for plotting from the native facet pairs
    # by reading the source geometry builder's wall connectivity.
    from tensorfvm.suboff_hybrid_mesh import structured_layers
    p,_,_,walls,_,_=structured_layers(axial=a.axial,azimuthal=a.azimuthal,layers=a.layers,first_height=a.first_height,growth=a.growth)
    ax.add_collection3d(Poly3DCollection([p[list(f)] for f in walls],facecolor='#9dd7b6',edgecolor='#245c41',linewidth=.3))
    ax.set_xlim(0,1);ax.set_ylim(-.065,.065);ax.set_zlim(-.065,.065);ax.set_box_aspect((6,1,1));ax.set_title('Actual structured surface quads; triangular cap closures');ax.set_xlabel('x/L');ax.set_ylabel('y/L');ax.set_zlabel('z/L')
    groups=section(m,2,0)
    colors={'tetra':'#999999','hexahedron':'#137a41','wedge':'#206bb3'}
    for index,(limits,title) in enumerate([((-.035,1.035,-.09,.09),'True native facet intersections with z=0'),((.93,1.015,-.022,.022),'Tail detail: structured layers and exterior tetrahedra')],2):
        ax=fig.add_subplot(3,1,index)
        for kind,lines in groups.items():ax.add_collection(LineCollection(lines,colors=colors[kind],linewidths=.25 if kind=='tetra' else .5,label=kind))
        ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:]);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x/L');ax.set_ylabel('y/L');ax.legend(loc='upper right')
    fig.tight_layout();fig.savefig(out/'mesh.png',dpi=180);fig.savefig(out/'mesh.pdf');plt.close(fig)
    s['first_height_m']=a.first_height*LENGTH_M
    (out/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    (out/'report.md').write_text(f'''# SUBOFF structured near-wall / unstructured exterior mesh

## Geometry and use

Bare AFF-1 mother profile from TensorLBM (source SHA256 `{SOURCE_SHA256}`). The original experiment offset table remains independently unverified. Length {LENGTH_M:.8f} m. Surface coordinates use cosine-spaced axial stations with the nearest interior samples replaced by the three exact polynomial junctions (avoiding nearly coincident points), uniform periodic azimuth, and finite-difference meridian normals. Both tips close on the axis using wedges, rather than repeated degenerate hexahedra. Normal layer distances are measured from the same original surface, without accumulating marching-normal drift.

Actual mesh: **{s['structured_hexes']} near-wall hexes**, **{s['cap_wedges']} cap wedges**, **{s['exterior_tetrahedra']} exterior tetrahedra**. {a.layers} layers, first height {a.first_height:.6g} L = {s['first_height_m']:.6g} m, growth {a.growth}, total thickness {s['total_thickness_L']:.6g} L. These are geometry-development parameters, not a Reynolds/y+ design.

Gmsh {s['gmsh_version']} meshes the exterior of the imposed layer skin inside a [-1,3] × [-1,1] × [-1,1] L box. This is genuine unstructured tetrahedral generation. Every outer structured quad is divided on its smallest-vertex-ID diagonal. The exact triangles are imposed on Gmsh and shared in the finite-volume importer: **{s['verified_shared_interface_faces']} verified internal triangles**. No overset interpolation or pyramid transition is used. The native volume mesh retains hex/wedge cell types; only FV integration faces are triangulated.

```python
from tensorfvm.native_mesh3d import NativeMesh3D
mesh = NativeMesh3D.from_gmsh('suboff-hybrid.msh', triangulate_faces=True)
```

The opt-in mode treats warped quads as two planar facets, not as bilinear curved faces. File node tags are sorted before triangulation to preserve diagonal choices across round-trips. The default planar-face importer remains strict.

## Mesh evidence

![Actual structured wall and native facet sections](mesh.png)

Green: hex facets; blue: cap-wedge facets; gray: exterior tetrahedral facets. The lower panels intersect actual native facets with z=0; they do not project off-plane edges. Displayed diagonals reflect the actual FV face representation. No pressure or velocity solution is shown.

## Verification

- Minimum volume: {s['mesh_audit']['minimum_cell_volume']:.6g} L³.
- Maximum relative cell closure: {s['mesh_audit']['maximum_relative_closure']:.6g}.
- Minimum S·d: {s['mesh_audit']['minimum_face_projection']:.6g}.
- Maximum nonorthogonality: {s['mesh_audit']['maximum_nonorthogonality_degrees']:.6g} degrees.
- Maximum of affine reconstruction / gradient / divergence audit errors: {max(s['affine_audit'].values()):.6g}.
- Arbitrary internal interface flux global cancellation: {s['interface_arbitrary_flux_cancellation']:.6g}.
- Fluid + faceted body volume vs box-volume error: {s['domain_volume_balance_error_L3']:.6g} L³.
- Sampled wall-node radial error (including explicitly closed tip): {s['wall_node_radial_error_L']:.6g} L.
- Geometry gates: **{s['geometry_passed']}**; numerical/physical flow qualified: **False**.

Full per-metric values are in `summary.json`; `.msh` and `native-geometry.npz` retain native geometry. Source snapshots and SHA256 manifest bind this execution. The PNG/PDF are mesh figures, not a flow report.

## Limits and next gates

This is a thin-layer mesh feasibility result. Surface facet chord error, normal-offset discretization, high aspect ratio, layer/exterior size jump and tail quality need refinement studies before viscous use. Maximum nonorthogonality alone does not certify suitability for turbulence. Fixed outer-domain dimensions are not a domain-independence study. No attachments, y+, wall model, viscous momentum solution, GPU qualification, force/Cp convergence or <3% claim. Layer thickness >0.02 L is rejected pending offset collision controls; candidate meshes fail immediately on invalid geometric projections.

TensorLBM `dg_curv.py` motivated layer height/growth and local-normal geometry, but its DG geometric factors and exterior coupling are not copied into this FVM geometry. SUBOFF DG-band tests do not certify this mesh.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/prepare_suboff_hybrid_mesh.py --output NEW_EMPTY_DIRECTORY --axial {a.axial} --azimuthal {a.azimuthal} --layers {a.layers} --first-height {a.first_height} --growth {a.growth} --far-size {a.far_size}
```
''')
    (out/'manifest.json').write_text(json.dumps({str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'},indent=2)+'\n')
    print(json.dumps(s,indent=2))
    if not s['geometry_passed']:raise SystemExit('Mesh verification gates failed')

if __name__=='__main__':main()
