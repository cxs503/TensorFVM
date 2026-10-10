"""Independent NumPy mesh reconstruction from saved physical cell polygons."""
import argparse
import json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import write_json,sha,artifact_manifest


def audit(row, fields):
    cells=fields['cell_polygons_m'];known={};owners=[];neighbors=[];endpoints=[]
    nxt=np.roll(cells,-1,axis=1)
    cross=cells[:,:,0]*nxt[:,:,1]-nxt[:,:,0]*cells[:,:,1]
    area=cross.sum(1)/2
    center=((cells+nxt)*cross[:,:,None]).sum(1)/(6*area[:,None])
    np.testing.assert_allclose(area,fields['cell_volumes_m2'],rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(center,fields['cell_centers_m'].reshape(-1,2),rtol=1e-12,atol=1e-13)
    if not np.all(area>0):raise ValueError('Invalid signed cell areas')
    for ci,polygon in enumerate(cells):
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            if np.array_equal(a,b):continue
            aa,bb=tuple(a),tuple(b);key=tuple(sorted((aa,bb)))
            if key in known:
                fi=known[key]
                if neighbors[fi]!=-1 or endpoints[fi]!=(bb,aa):raise ValueError('Invalid edge incidence')
                neighbors[fi]=ci
            else:
                known[key]=len(owners);owners.append(ci);neighbors.append(-1);endpoints.append((aa,bb))
    owners=np.array(owners);neighbors=np.array(neighbors);endpoints=np.array(endpoints)
    np.testing.assert_array_equal(owners,fields['face_owner'])
    np.testing.assert_array_equal(neighbors,fields['face_neighbor'])
    internal=neighbors>=0
    np.testing.assert_array_equal(internal,fields['internal_face_mask'])
    edge=endpoints[:,1]-endpoints[:,0];S=np.stack([edge[:,1],-edge[:,0]],1);fc=endpoints.mean(1)
    np.testing.assert_allclose(S,fields['face_area_vectors_m'],rtol=1e-12,atol=1e-13)
    d=fc-center[owners];d[internal]=center[neighbors[internal]]-center[owners[internal]]
    projected=(S*d).sum(1)
    angles=np.degrees(np.arccos(np.clip(projected/np.linalg.norm(S,axis=1)/np.linalg.norm(d,axis=1),-1,1)))
    mask=fields['surface_mask']
    if np.any(internal[mask]):raise ValueError('Body mask contains internal faces')
    np.testing.assert_allclose(fc[mask],fields['surface_centers_m'],rtol=1e-12,atol=1e-13)
    wall=endpoints[mask].reshape(-1,2);config=row['config']
    if row['case']=='cylinder':
        deviation=float(np.max(abs(np.linalg.norm(wall-[.2,.2],axis=1)-.05)))
        tolerance=1e-10
    else:
        x=(1-np.cos(np.linspace(0,np.pi,4097)))/2
        y=.6*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
        local=np.r_[np.c_[x[::-1],y[::-1]],np.c_[x[1:],-y[1:]]]
        angle=-np.radians(row.get('mesh_generation',{}).get('geometric_incidence_deg',0))
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        profile=(local-[.25,0])@rotation.T+[config['airfoil_x']+.25,config['airfoil_y']]
        a,b=profile[:-1],profile[1:];segment=b-a
        deviation=0.
        for start in range(0,len(wall),64):
            delta=wall[start:start+64,None,:]-a
            fraction=np.clip(np.sum(delta*segment,axis=-1)/np.sum(segment*segment,axis=-1),0,1)
            distance=np.min(np.linalg.norm(delta-fraction[:,:,None]*segment,axis=-1),axis=1)
            deviation=max(deviation,float(distance.max()))
        tolerance=5e-6
    # Every vertex in the closed body edge loop must have degree two.
    valence={}
    for point in wall:
        key=tuple(point);valence[key]=valence.get(key,0)+1
    closed=all(value==2 for value in valence.values())
    body_signed_area=np.sum(endpoints[mask,0,0]*endpoints[mask,1,1]-endpoints[mask,1,0]*endpoints[mask,0,1])/2
    domain_area=config['length']*config['height']+body_signed_area
    area_error=abs(area.sum()-domain_area)
    max_angle=float(angles.max())
    np.testing.assert_allclose(max_angle,row['mesh_quality']['nonorthogonality_max_deg'],rtol=1e-10,atol=1e-10)
    passed=bool(closed and deviation<tolerance and np.all(projected>0) and max_angle<70 and area_error<1e-10)
    if not passed:raise ValueError('Body conformity/mesh quality audit failed')
    return dict(passed=passed,cells=len(cells),maximum_nonorthogonality_deg=max_angle,body_vertex_max_deviation_m=deviation,body_geometry_tolerance_m=tolerance,closed_body_loop=closed,fluid_domain_area_error_m2=float(area_error),physical_accuracy_qualified=row['passed'],scope='Independent polygon areas/centroids, complete oriented face incidence, normals, projected distances, nonorthogonality, circle or dense analytic NACA0012 profile and closed body loop; no mesh generator or solver imported')


def process(output):
    summary=json.loads((output/'summary.json').read_text());records=[]
    for row in summary['runs']:
        with np.load(output/row['directory']/'fields.npz') as fields:
            record=audit(row,fields)
        write_json(output/row['directory']/'mesh-audit.json',record);records.append(dict(resolution=row['resolution'],**record))
    write_json(output/'mesh-audit.json',dict(runs=records,passed=all(r['passed'] for r in records)))
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['source_sha256']['scripts/audit_external_gmsh_mesh.py']=sha(Path(__file__))
    if manifest.get('source_snapshot_root'):
        import shutil
        target=output/manifest['source_snapshot_root']/'scripts/audit_external_gmsh_mesh.py';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),target)
    manifest['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',manifest)
    print(output,'independent body-conformity audit passed:',len(records),'meshes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('outputs',type=Path,nargs='+');args=p.parse_args()
    for output in args.outputs:process(output)
