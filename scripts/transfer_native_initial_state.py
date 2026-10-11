#!/usr/bin/env python3
"""Freeze a numerical initialization transferred between native meshes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_field_transfer3d import transfer_cell_values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output
    if out.exists():p.error('Output must be new')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    source=a.source_run
    config=json.loads((source/'config.json').read_text())
    audit=json.loads((source/'audit/audit.json').read_text())
    if sha(source/'input.msh')!=config['mesh_sha256']:raise ValueError('Source mesh binding mismatch')
    if not json.loads((source/'result.json').read_text())['converged']:raise ValueError('Source run must be converged')
    # The audit schema is checked below against its bound artifacts as well as
    # its pass flag; a stale success on different fields must not authorize use.
    if not audit.get('passed'):raise ValueError('Source run must pass independent audit')
    for name in ('config.json','input.msh','fields.npz','result.json'):
        matches=[v for k,v in audit['producer_input_sha256'].items() if Path(k).name==name]
        if len(matches)!=1 or matches[0]!=sha(source/name):raise ValueError('Independent audit binding mismatch: '+name)
    old=NativeMesh3D.from_gmsh(source/'input.msh',triangulate_faces=True)
    new=NativeMesh3D.from_gmsh(a.mesh,triangulate_faces=True)
    with np.load(source/'fields.npz') as f:
        samples=np.column_stack((f['velocity'],f['pressure']))
    values,stats=transfer_cell_values(old.cell_centers,samples,new.cell_centers)
    out.mkdir(parents=True);shutil.copy2(a.mesh,out/'input.msh')
    np.savez_compressed(out/'fields.npz',velocity=values[:,:3],pressure=values[:,3])
    bindings={str(path.resolve()):sha(path) for path in [source/'config.json',source/'result.json',source/'input.msh',source/'fields.npz',source/'audit/audit.json']}
    root=Path(__file__).resolve().parents[1]
    for relative in ['scripts/transfer_native_initial_state.py','src/tensorfvm/native_field_transfer3d.py','src/tensorfvm/native_mesh3d.py']:
        target=out/'source-snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/relative,target)
    (out/'config.json').write_text(json.dumps(dict(kind='transferred numerical initial guess',mesh_sha256=sha(out/'input.msh'),source_inputs_sha256=bindings,transfer=stats,analytic_reference_used=False),indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(dict(converged=False,flow_solved=False,source_producer_converged=True,transfer=stats),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()},indent=2)+'\n')
    print(json.dumps(stats))


if __name__=='__main__':main()
