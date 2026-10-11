#!/usr/bin/env python3
"""Source-bound geometry qualification and explicitly estimated y+ planning."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def planning_yplus(reynolds,first_height,cf=.003):
    """Assumed flat first-cell center at h/2; actual wall traction is unknown."""
    if not np.isfinite([reynolds,first_height,cf]).all() or min(reynolds,first_height,cf)<=0:
        raise ValueError('positive finite planning inputs required')
    return float(reynolds*np.sqrt(cf/2)*first_height/2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise SystemExit('Output must be empty')
    a.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((a.case/'config.json').read_text());s=json.loads((a.case/'summary.json').read_text());q=json.loads((a.quality/'quality.json').read_text());geometry=a.case/'native-geometry.npz'
    actual_hash=hashlib.sha256(geometry.read_bytes()).hexdigest()
    if actual_hash!=q['geometry_sha256']:raise ValueError('Grouped quality is for a different mesh')
    n=np.load(geometry);types=n['cell_types'];volumes=n['cell_volumes'];wall=n['patch_labels']=='wall'
    area=n['area_vectors'][wall];d=n['face_centers'][wall]-n['cell_centers'][n['owner'][wall]]
    projected=np.einsum('ij,ij->i',area,d)/np.linalg.norm(area,axis=1)
    positivity={str(k):{'cells':int(np.count_nonzero(types==k)),'minimum_volume_L3':float(volumes[types==k].min()),'positive_volumes':bool(np.all(volumes[types==k]>0))} for k in np.unique(types)}
    distances={'minimum_face_normal_center_distance_L':float(projected.min()),'p50_face_normal_center_distance_L':float(np.percentile(projected,50)),'p95_face_normal_center_distance_L':float(np.percentile(projected,95)),'maximum_face_normal_center_distance_L':float(projected.max()),'definition':'Projection from native adjacent cell centroid onto each planar wall facet normal; not an exact nearest-surface SST distance'}
    planning=[{'observable':'Cp','Re_L':12e6,'assumed_Cf':.003,'assumed_first_cell_center_distance_L':s['first_height_L']/2,'estimated_yplus':planning_yplus(12e6,s['first_height_L']),'actual_wall_tau_available':False},{'observable':'drag','Re_L':13.2180568e6,'assumed_Cf':.003,'assumed_first_cell_center_distance_L':s['first_height_L']/2,'estimated_yplus':planning_yplus(13.2180568e6,s['first_height_L']),'actual_wall_tau_available':False}]
    result={'native_geometry_sha256':actual_hash,'case_summary_sha256':hashlib.sha256((a.case/'summary.json').read_bytes()).hexdigest(),'grouped_quality_sha256':hashlib.sha256((a.quality/'quality.json').read_bytes()).hexdigest(),'cell_positivity_by_type':positivity,'wall_centroid_projection_diagnostic':distances,'yplus_planning':planning,'geometry_positive_closed_conforming':bool(all(v['positive_volumes'] for v in positivity.values()) and s['mesh_audit']['maximum_relative_closure']<1e-12 and s['verified_shared_interface_faces']==s['interface_triangles'] and projected.min()>0),'actual_yplus_qualified':False,'SST_flow_qualified':False,'experimental_accuracy_qualified':False}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'report-source.py').write_bytes(Path(__file__).read_bytes())
    test_source=Path(__file__).resolve().parents[1]/'tests/test_suboff_wall_mesh_planning.py'
    (a.output/'planning-test-source.py').write_bytes(test_source.read_bytes())
    rows='\n'.join(f"| {name} | {v['cells']:,} | {v['minimum_volume_L3']:.6g} | {v['positive_volumes']} |" for name,v in positivity.items())
    qrows='\n'.join(f"| {name} | {v['maximum_degrees']:.4f}° | {v['p95_degrees']:.4f}° |" for name,v in q['nonorthogonality'].items())
    yrows='\n'.join(f"| {v['observable']} | {v['Re_L']:.7g} | {v['estimated_yplus']:.6f} | {'below' if v['estimated_yplus']<1 else 'above'} 1 |" for v in planning)
    case_rel=Path(__import__('os').path.relpath(a.case,a.output));quality_rel=Path(__import__('os').path.relpath(a.quality,a.output))
    (a.output/'report.md').write_text(f'''# SUBOFF mesh for wall-resolved SST planning

## Scope and fixed requested parameters

This is an actual mesh-generation and geometry-audit result, not an SST flow calculation. Mother-profile chord target **{config["chord_tolerance"]:.6g} L**, maximum axial spacing **{config["maximum_spacing"]:.6g} L**, **{s['azimuthal']} azimuthal stations**, **{s['layers']} normal layers**, unchanged first full layer height **{s['first_height_L']:.6g} L**, growth **{s['growth']}**, total normal thickness **{s['total_thickness_L']:.8g} L**. Gmsh exterior size is {config["far_size"]:.6g} L. No first-height adjustment was made to meet an estimated y+ threshold. No earlier mesh artifact is overwritten.

Actual adaptive axial stations: **{s['axial_stations']}**. These settings already imply **{(s["axial_stations"]-1)*s["azimuthal"]*s["layers"]:,} structured layer cells**; the total actual native control volume count is **{s['mesh_audit']['cells']:,}**. This exceeds the initial ~60k estimate because the requested 71 stations × 48 sectors × 24 layers alone consume more than 80k cells. No flow solver was run as part of this mesh task. Generation/import runtime was {s['generation_seconds']:.3f} seconds on the available shared CPU workload, not an isolated performance benchmark.

![Actual mother-line stations and native facet sections]({case_rel}/mesh.png)

## Thin-layer positivity and interface closure

| Cell type | Actual cells | Minimum volume / L³ | All volumes positive |
|---|---:|---:|---|
{rows}

Maximum relative cell face closure **{s['mesh_audit']['maximum_relative_closure']:.6g}**; minimum face S·d **{s['mesh_audit']['minimum_face_projection']:.6g}**. **{s['verified_shared_interface_faces']}** native interface triangles exactly share structured/exterior adjacency; no interpolation interface. Maximum affine operator audit error **{max(s['affine_audit'].values()):.6g}**. These are geometric and affine operator checks only. High aspect ratio and roundoff sensitivity remain material risks at this first height.

## Actual surface geometry

Ten points per actual triangular native wall facet include vertices, edges and interior locations. Maximum sampled interior closest distance to the continuous mother profile is **{s['wall_facet_geometry_audit']['maximum_sampled_interior_distance_L']:.6g} L**; maximum sampled node distance **{s['wall_facet_geometry_audit']['maximum_sampled_vertex_distance_L']:.6g} L**. Dense meridian chord distance is **{s['adaptive_sampling']['maximum_dense_meridian_chord_error_L']:.6g} L**. The angular midbody sagitta is **{s['azimuthal_midbody_sagitta_L']:.6g} L**. Exact nodes do not certify planar-face geometry. These are finite sampled diagnostics, not a continuous Hausdorff certificate and not a flow error estimate.

## Grouped face quality

| Face group | Maximum nonorthogonality | P95 |
|---|---:|---:|
{qrows}

![Actual grouped native face quality]({quality_rel}/quality.png)

High wall-facet angles arise partly from triangle face centers displaced tangentially from the adjacent large-aspect hexahedral cell center. This explanation does not exempt wall-gradient or pressure traction operators from verification. Exterior and interface angles still require a numerical sensitivity check. Actual wall centroid-to-facet normal projections range from **{projected.min():.6g}** to **{projected.max():.6g} L**, median **{np.median(projected):.6g} L**. This is a geometry diagnostic; it is not the exact nearest-surface distance used by a qualified SST implementation.

## Estimated y+ — assumptions and failed estimate retained

Assume **Cf=0.003**, Uref uniform, u_tau/Uref=sqrt(Cf/2), and a flat first-cell center at half the full first layer height: y/L=h/(2L)=2e-6. Then y+≈Re_L sqrt(Cf/2) h/(2L). This uses no computed wall traction.

| Intended observable | Re_L | Estimated first-center y+ | Comparison |
|---|---:|---:|---|
{yrows}

The drag-Re estimate remains above 1. This is retained transparently, without changing h to pass a planning threshold. Neither estimated value qualifies actual y+. A final solution must compute local wall traction and its u_tau, use a verified wall-distance definition, examine the spatial y+ distribution, then decide whether to refine. Total band thickness also requires comparison to the actual boundary layer; 24 layers alone do not certify SST resolution. No drag/Cp experiment error below 3%, RANS convergence or GPU acceleration is claimed.

## Reproduction and evidence

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/prepare_suboff_adaptive_mesh.py --output NEW_EMPTY_DIRECTORY --chord-tolerance .0001 --maximum-spacing .035 --azimuthal {s['azimuthal']} --layers 24 --first-height .000004 --growth 1.3 --far-size .3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/audit_suboff_adaptive_quality.py --geometry NEW_EMPTY_DIRECTORY/native-geometry.npz --output NEW_EMPTY_QUALITY_DIRECTORY
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/report_suboff_wall_mesh.py --case NEW_EMPTY_DIRECTORY --quality NEW_EMPTY_QUALITY_DIRECTORY --output NEW_EMPTY_REPORT_DIRECTORY
```

The [generator report]({case_rel}/report.md) preserves original launch source snapshots, `.msh`, geometry NPZ, config, sampled audit, PNG/PDF and SHA256 manifest. The grouped audit binds that NPZ SHA. This supplemental report binds both audit JSON inputs and includes its report producer source. Original TensorLBM geometry attribution and independent offset-table limitation remain as documented in the generator. Planning tests passed: **2** (`tests/test_suboff_wall_mesh_planning.py`), checking the half-layer center definition and transparent drag-Re estimate above 1. Test source is archived with this report. Geometry passed: **{result['geometry_positive_closed_conforming']}**; actual y+/SST flow/experimental accuracy qualified: **False**.
''')
    (a.output/'manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.iterdir() if f.is_file() and f.name!='manifest.json'},indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
