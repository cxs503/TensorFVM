#!/usr/bin/env python3
"""Generate source-bound adaptive hybrid mesh and actual facet error evidence."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tensorfvm.suboff_hybrid_mesh import generate_suboff_hybrid_mesh,wall_facet_geometry_audit,structured_layers
from tensorfvm.native_operators3d import NativeOperators3D
from tensorfvm.suboff_geometry import radius,RADIUS_OVER_LENGTH
from prepare_suboff_hybrid_mesh import section


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chord-tolerance',type=float,default=.0002);p.add_argument('--minimum-spacing',type=float,default=1e-5)
    p.add_argument('--azimuthal',type=int,default=48);p.add_argument('--layers',type=int,default=3)
    p.add_argument('--first-height',type=float,default=.0001);p.add_argument('--growth',type=float,default=1.25)
    p.add_argument('--far-size',type=float,default=.35);a=p.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):raise SystemExit('Output must be empty')
    out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[1]
    files=['scripts/prepare_suboff_adaptive_mesh.py','scripts/prepare_suboff_hybrid_mesh.py','src/tensorfvm/suboff_hybrid_mesh.py','src/tensorfvm/suboff_geometry.py','src/tensorfvm/native_mesh3d.py','src/tensorfvm/native_operators3d.py','tests/test_suboff_adaptive_mesh.py']
    for rel in files:
        dst=out/'source-snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((root/rel).read_bytes())
    cfg=vars(a).copy();cfg['output']=str(out);(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    settings={k:v for k,v in vars(a).items() if k not in ('output','far_size')}
    try:
        start=time.perf_counter();m,s=generate_suboff_hybrid_mesh(out/'suboff-hybrid.msh',far_size=a.far_size,**settings)
        s['generation_seconds']=time.perf_counter()-start
        s['wall_facet_geometry_audit']=wall_facet_geometry_audit(m)
        s['affine_audit']=NativeOperators3D(m).audit_affine()
        s['azimuthal_midbody_sagitta_L']=float(RADIUS_OVER_LENGTH*(1-np.cos(np.pi/a.azimuthal)))
        np.savez_compressed(out/'native-geometry.npz',vertices=m.vertices,owner=m.owner,neighbor=m.neighbor,face_vertices=np.asarray(m.face_vertices),area_vectors=m.area_vectors,face_centers=m.face_centers,cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,patch_labels=m.patch_labels,cell_types=np.asarray(m.cell_types))
        fig,axes=plt.subplots(3,1,figsize=(13,9));points,_,_,walls,_,info=structured_layers(**settings)
        wall_ids=np.unique(np.concatenate(walls));stations=np.unique(points[wall_ids,0]);x=np.linspace(0,1,4001)
        axes[0].plot(x,radius(x),color='black',label='Continuous mother profile')
        axes[0].plot(stations,radius(stations),'o-',ms=3,label='Adaptive actual stations');axes[0].set_ylabel('r/L');axes[0].legend()
        groups=section(m,2,0);colors={'tetra':'#999999','hexahedron':'#137a41','wedge':'#206bb3'}
        for ax,limits,title in zip(axes[1:],[(0,1,-.08,.08),(.94,1.01,-.025,.025)],['Actual native face intersections z=0','Tail actual face intersections z=0']):
            for kind,lines in groups.items():ax.add_collection(LineCollection(lines,colors=colors[kind],linewidths=.3,label=kind))
            ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:]);ax.set_aspect('equal');ax.set_title(title);ax.set_ylabel('y/L');ax.set_xlabel('x/L');ax.legend(loc='upper right')
        fig.tight_layout();fig.savefig(out/'mesh.png',dpi=180);fig.savefig(out/'mesh.pdf');plt.close(fig)
        (out/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
        d=s['wall_facet_geometry_audit'];audit=s['mesh_audit']
        (out/'report.md').write_text(f'''# Adaptive SUBOFF conforming hybrid mesh

## Method and purpose

Optional adaptive meridian sampling splits each of four original polynomial segments at the maximum Euclidean curve-to-chord distance. It checks 8,193 cosine-spaced reference points per segment and preserves exact junctions and closed tips. Target chord distance {a.chord_tolerance:g} L, minimum axial separation {a.minimum_spacing:g} L, at most 512 stations. Spacing or station-limit conflicts fail explicitly. Original default cosine sampling is unchanged. This controls meridian curvature; azimuthal chord error remains independently controlled by {a.azimuthal} uniform periodic stations. Dense sampled tolerances are not a rigorous continuous Hausdorff certificate.

## Actual geometry and native topology

Actual axial stations: {s['axial_stations']}; minimum axial separation {s['adaptive_sampling']['actual_minimum_axial_spacing_L']:.6g} L; dense reference maximum meridian chord error {s['adaptive_sampling']['maximum_dense_meridian_chord_error_L']:.6g} L.

{a.layers} layers, first height {a.first_height:g} L, growth {a.growth}; {s['structured_hexes']} hexahedra, {s['cap_wedges']} cap wedges, {s['exterior_tetrahedra']} Gmsh exterior tetrahedra. All {s['verified_shared_interface_faces']} interface triangles are shared native internal faces, preserving conformity. Minimum volume {audit['minimum_cell_volume']:.6g} L³; minimum S·d {audit['minimum_face_projection']:.6g}; relative closure {audit['maximum_relative_closure']:.6g}; maximum nonorthogonality {audit['maximum_nonorthogonality_degrees']:.6g}°. Maximum affine audit error {max(s['affine_audit'].values()):.6g}.

![Adaptive meridian and real native mesh sections](mesh.png)

## Actual facet interior error

This audit evaluates **actual triangular FV wall faces**, not just original nodes: three vertices, three edge midpoints, centroid, and three interior quarter points per triangle. Axisymmetry reduces closest distance to the continuous hull to a 2D minimization in axial coordinate and radius. Maximum sampled interior distance **{d['maximum_sampled_interior_distance_L']:.6g} L**, versus maximum sampled vertex distance **{d['maximum_sampled_vertex_distance_L']:.6g} L**. P95 of all sampled distances {d['p95_sampled_surface_distance_L']:.6g} L. Analytic midbody angular sagitta {s['azimuthal_midbody_sagitta_L']:.6g} L. The difference between exact nodes and finite interior chord error is explicitly exposed. These are sampled diagnostics, not a continuous worst-case guarantee. Geometric closeness does not establish flow accuracy.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/prepare_suboff_adaptive_mesh.py --output NEW_EMPTY_DIRECTORY --chord-tolerance {a.chord_tolerance} --minimum-spacing {a.minimum_spacing} --azimuthal {a.azimuthal} --layers {a.layers} --first-height {a.first_height} --growth {a.growth} --far-size {a.far_size}
```

Source snapshot, native geometry NPZ, Gmsh file, config, summary, PNG/PDF and SHA256 manifest preserve this actual execution. The thin layers are a development mesh, not a full resolved boundary layer. Normal offsets still use the existing finite-difference mother normals; high aspect ratio, nonorthogonality, exterior size jump, domain independence and wall resolution remain separate gates. No turbulence, y+, force/Cp convergence or flow error <3% is claimed. Native triangulation approximates warped quads by planar facets.
''')
    except Exception as e:
        (out/'failure.json').write_text(json.dumps({'error':type(e).__name__,'message':str(e),'physics_qualified':False},indent=2)+'\n');raise
    finally:
        (out/'manifest.json').write_text(json.dumps({str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'},indent=2)+'\n')
    print(json.dumps(s,indent=2))

if __name__=='__main__':main()
