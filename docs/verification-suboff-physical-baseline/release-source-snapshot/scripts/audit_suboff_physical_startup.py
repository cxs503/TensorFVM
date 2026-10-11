#!/usr/bin/env python3
"""Locate actual physical-run extrema and replay stored native continuity."""
import argparse
from pathlib import Path
import json
import shutil
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_operators3d import NativeOperators3D
from run_native_suboff_benchmark import digest, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):p.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True);records=[]
    for run in a.run:
        config=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
        if digest(run/'input.msh')!=config['mesh_sha256']:raise ValueError('mesh binding mismatch')
        m=NativeMesh3D.from_gmsh(run/'input.msh',triangulate_faces=True);f=np.load(run/'fields.npz')
        op=NativeOperators3D(m);div=op.integrate(f['flux'])/m.cell_volumes
        speed=np.linalg.norm(f['velocity'],axis=1);records_by_field={}
        for name,values in [('speed',speed),('absolute_pressure',abs(f['pressure'])),('absolute_stored_divergence',abs(div))]:
            entries=[]
            for ci in np.argsort(values)[-5:][::-1]:
                boundary=(m.neighbor<0)&(m.owner==ci)
                entries.append(dict(cell=int(ci),value=float(values[ci]),center_L=m.cell_centers[ci].tolist(),
                    cell_type=m.cell_types[ci],wall_distance_L=float(f['wall_distance'][ci]),
                    boundary_patches=np.unique(m.patch_labels[boundary]).tolist()))
            records_by_field[name]=entries
        records.append(dict(run=str(run),numerically_converged=result['numerically_converged'],
            completed_outer_iterations=len(result['history']),extrema=records_by_field,
            stored_divergence_max_replayed=float(abs(div).max()),
            last_logged_stored_divergence=result['momentum_history'][-1]['native_divergence_max'],
            global_boundary_volume_flux=float(f['flux'][m.neighbor<0].sum()),
            mesh_sha256=digest(run/'input.msh'),fields_sha256=digest(run/'fields.npz'),result_sha256=digest(run/'result.json')))
    write(out/'audit.json',dict(cases=records,physical_validation_passed=False))
    shutil.copy2(__file__,out/'audit-source.py')
    write(out/'manifest.json',{path.name:digest(path) for path in out.iterdir() if path.is_file()})
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
