"""Compare actual wall geometry and solved coefficients on matched NACA domains."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def inspect(directory,row):
    with np.load(directory/row['directory']/'fields.npz') as f:
        mask=f['surface_mask'];S=f['face_area_vectors_m'][mask]
        normals=S/np.linalg.norm(S,axis=1)[:,None]
        h=np.sum(f['surface_displacement_m']*normals,axis=1)
        centers=f['surface_centers_m']
        angle=-np.radians(row['mesh_generation']['geometric_incidence_deg'])
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        local=(centers-[row['config']['airfoil_x']+.25,row['config']['airfoil_y']])@rotation+[.25,0]
        tail=local[:,0]>.95
        lengths=np.linalg.norm(S,axis=1)
    return dict(study=directory.name,resolution=row['resolution'],cells=row['mesh_quality']['cells'],
                wall_faces=int(mask.sum()),wall_distance_median_m=float(np.median(h)),
                trailing_wall_distance_max_m=float(h[tail].max()),
                trailing_wall_edge_max_m=float(lengths[tail].max()),
                drag=row['computed']['drag'],lift=row['computed']['lift'],
                drag_error=next(q['value'] for q in row['metrics'] if q['name']=='drag_relative_error'),
                lift_error=next(q['value'] for q in row['metrics'] if q['name']=='lift_relative_error'),
                iterations=row['iterations'],converged=row['converged'],passed=row['passed'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--baseline',type=Path,default=Path('docs/verification-gmsh-naca-matched'))
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    rows=[];inputs={};baseline=None
    for directory in (args.baseline,args.output):
        summary=json.loads((directory/'summary.json').read_text())
        inputs[str((directory/'summary.json').resolve().relative_to(root))]=sha(directory/'summary.json') if directory!=args.output else None
        for row in summary['runs']:
            if not row.get('reference_domain_matched'):raise ValueError('Unmatched reference domain')
            signature={key:row['config'][key] for key in ('density','reynolds','inlet_velocity','length','height','airfoil_x','airfoil_y','angle_of_attack','velocity_relaxation','pressure_relaxation','pseudo_time_step')}
            signature.update(reference=row['reference'],discretization=row['discretization'],incidence=row['physical_angle_of_attack_deg'])
            if baseline is None:baseline=signature
            elif signature!=baseline:raise ValueError('Different physical problem or discretization')
            rows.append(inspect(directory,row))
            if directory!=args.output:
                for name in ('fields.npz','result.json'):
                    path=directory/row['directory']/name
                    inputs[str(path.resolve().relative_to(root))]=sha(path)
    inputs={key:value for key,value in inputs.items() if value is not None}
    write_json(args.output/'mesh-comparison.json',dict(runs=rows,matched_configuration=baseline,scope='Actual CPU solutions with common equations and physical domain. Different meshes and concurrent process timings prevent a controlled speed claim. No matched TensorLBM calculation.',input_sha256=inputs))
    with (args.output/'mesh-comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=json.loads((args.output/'manifest.json').read_text())
    manifest.setdefault('input_sha256',{}).update(inputs)
    manifest['source_sha256']['scripts/compare_hybrid_wall_mesh.py']=sha(Path(__file__))
    if manifest.get('source_snapshot_root'):
        import shutil
        target=args.output/manifest['source_snapshot_root']/'scripts/compare_hybrid_wall_mesh.py';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),target)
    manifest['artifacts_sha256']=artifact_manifest(args.output);write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
